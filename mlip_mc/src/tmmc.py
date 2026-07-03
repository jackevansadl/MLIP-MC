import os
import json
import sys
import struct
import traceback
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import numpy as np

from ase import Atoms
from ase.io import read, write
from ase import units as ase_units
from ase.units import bar
from .utilities import _random_rotation, random_position, vdw_overlap
from .gcmc import e_interaction_of_adsorption, EXP_THRESHOLD, VDW_OFFSET, HIGH_ENERGY, TRANSLATION_STEP, ROTATION_CIRCLEFRAC
from .tmmc_analysis import lnPi_from_collection, normalize_lnPi


# Cumulative move probabilities. 'md' and 'volume' default to zero so the
# defaults reproduce rigid-framework GCMC-style sampling; enable them for
# flexible frameworks (hybrid MC/MD) and cell fluctuations.
TMMC_MOVE_PROBABILITIES = {
    'md': 0.0,
    'volume': 0.0,
    'insertion': 0.25,
    'deletion': 0.5,
    'translation': 0.75,
    'rotation': 1.0,
}


class MLP_TMMC:
    """
    Transition-Matrix Monte Carlo simulation for adsorption isotherms.

    Samples the grand-canonical insertion/deletion dynamics of an
    adsorbate in a (possibly flexible) framework while accumulating the
    collection matrix C(N -> N') of unbiased transition probabilities
    over the macrostate N = number of adsorbed molecules. The macrostate
    probability distribution ln Pi(N) is periodically reconstructed from
    C and used as a flat-histogram bias, so the full range
    [N_min, N_max] is sampled uniformly regardless of free-energy
    barriers.

    A single simulation at one reference fugacity yields ln Pi(N), which
    is reweighted to any pressure to obtain the complete isotherm,
    including metastable adsorption/desorption branches (hysteresis) when
    ln Pi is bimodal. See mlip_mc.src.tmmc_analysis.

    Framework flexibility is sampled with hybrid MC/MD moves: short MD
    trajectories of the full system (framework + adsorbates) with
    Maxwell-Boltzmann-resampled velocities, in NVT (default) or NPT.
    MC volume moves are also available for cell fluctuations with
    backends where NPT MD is impractical.

    References
    ----------
    Errington, J. Chem. Phys. 118, 9915 (2003).
    Goeminne & Van Speybroeck, J. Am. Chem. Soc. 147, 15 (2025).

    Parameters
    ----------
    model : calculator
        ASE calculator (MLIP or classical backend). Must support forces
        for MD moves and stress for NPT MD.
    atoms_frame : Atoms
        Framework structure. May be empty (bulk fluid TMMC).
    atoms_ads : Atoms
        Single adsorbate molecule template.
    T : float
        Temperature in Kelvin.
    P : float
        Reference pressure in ASE units (eV/A^3). Used for file naming
        and as the default external pressure for NPT/volume moves.
    fugacity : float
        Reference fugacity in ASE units. ln Pi is sampled at this
        fugacity and reweighted afterwards.
    device : str
        Device for calculations ('cuda' or 'cpu').
    vdw_radii : array_like
        Van der Waals radii indexed by atomic number.
    N_min, N_max : int
        Macrostate window; attempts to leave it are rejected (and
        recorded in the collection matrix diagonal).
    bias_update_interval : int, optional
        Steps between ln Pi recomputations from C (default: 10000).
    move_probabilities : dict, optional
        Cumulative move probabilities with keys 'md', 'volume',
        'insertion', 'deletion', 'translation', 'rotation' (values
        increasing, ending at 1.0). Default: TMMC_MOVE_PROBABILITIES.
    translation_step : float, optional
        Translation step in Angstrom (default: 0.5).
    rotation_circlefrac : float, optional
        Fraction of full rotation circle (default: 0.1).
    md_ensemble : str, optional
        'nvt' (Nose-Hoover chain, fixed cell; default) or 'npt'
        (ASE NPT dynamics; requires stress support and an
        upper-triangular cell).
    md_timestep : float, optional
        MD timestep in fs (default: 1.0).
    md_steps : int, optional
        MD steps per hybrid move (default: 1000).
    md_damp : float, optional
        Thermostat damping time in fs (default: 100).
    npt_pfactor : float, optional
        ASE NPT barostat pfactor (default: (1000 fs)^2 * 0.06 eV/A^3,
        appropriate for a ~10 GPa bulk modulus).
    external_pressure : float, optional
        External pressure for NPT MD and volume moves in ASE units
        (default: P).
    max_delta_lnV : float, optional
        Maximum ln-volume displacement per MC volume move (default: 0.01).
    debug : bool, optional
        Enable debug printing (default: False).
    output_dir : str, optional
        Output directory for results (default: 'results').
    n_steps : int, optional
        Target total number of MC steps (used for restart bookkeeping).
    checkpoint_interval : int, optional
        Steps between periodic restart saves (default: 1000). The
        collection matrix accumulates on attempted moves, so restart
        state is saved on this interval (and at every bias update)
        rather than on accepted moves.
    write_trajectory : bool, optional
        Write XYZ trajectory files (default: False).
    trajectory_interval : int, optional
        Steps between trajectory writes (default: 100).
    """

    def __init__(
        self,
        model,
        atoms_frame,
        atoms_ads,
        T,
        P,
        fugacity,
        device,
        vdw_radii,
        N_min=0,
        N_max=100,
        bias_update_interval=10000,
        move_probabilities=None,
        translation_step=TRANSLATION_STEP,
        rotation_circlefrac=ROTATION_CIRCLEFRAC,
        md_ensemble='nvt',
        md_timestep=1.0,
        md_steps=1000,
        md_damp=100,
        npt_pfactor=None,
        external_pressure=None,
        max_delta_lnV=0.01,
        debug=False,
        output_dir='results',
        n_steps=None,
        checkpoint_interval=1000,
        write_trajectory=False,
        trajectory_interval=100,
    ):
        self.model = model
        self.atoms_frame = atoms_frame
        self.n_frame = len(self.atoms_frame)
        self.atoms_ads = atoms_ads
        self.n_ads = len(self.atoms_ads)
        self.cell = np.array(self.atoms_frame.get_cell())
        self.V = np.linalg.det(self.cell)
        self.T = T
        self.P = P
        self.fugacity = fugacity
        self.device = device
        self.boltzmann = ase_units.kB
        self.beta = 1 / (self.boltzmann * T)
        self.Z_ads = 0

        if N_min < 0 or N_max <= N_min:
            raise ValueError(f"Invalid macrostate window [{N_min}, {N_max}]")
        self.N_min = int(N_min)
        self.N_max = int(N_max)
        self.M = self.N_max - self.N_min + 1
        self.bias_update_interval = bias_update_interval

        # Collection matrix (transitions N -> N-1 / N / N+1), visit
        # histogram, and the current bias estimate of ln Pi(N)
        self.C_down = np.zeros(self.M)
        self.C_stay = np.zeros(self.M)
        self.C_up = np.zeros(self.M)
        self.H = np.zeros(self.M, dtype=np.int64)
        self.lnPi = np.zeros(self.M)

        if move_probabilities is not None:
            self.move_probabilities = move_probabilities
        else:
            self.move_probabilities = TMMC_MOVE_PROBABILITIES.copy()

        self.translation_step = translation_step
        self.rotation_circlefrac = rotation_circlefrac

        if md_ensemble not in ('nvt', 'npt'):
            raise ValueError(f"md_ensemble must be 'nvt' or 'npt', got {md_ensemble!r}")
        self.md_ensemble = md_ensemble
        self.md_timestep = md_timestep
        self.md_steps = md_steps
        self.md_damp = md_damp
        if npt_pfactor is None:
            # (barostat time)^2 * bulk modulus; 0.06 eV/A^3 ~ 10 GPa
            npt_pfactor = (1000.0 * ase_units.fs) ** 2 * 0.06
        self.npt_pfactor = npt_pfactor
        self.external_pressure = P if external_pressure is None else external_pressure
        self.max_delta_lnV = max_delta_lnV

        self.insertion_rejected_due_to_vdw = 0
        self.insertion_rejected_due_to_acceptance = 0
        self.deletion_rejected_due_to_acceptance = 0
        self.window_rejections = 0

        self.vdw = vdw_radii - VDW_OFFSET
        self.debug = debug
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)

        self.n_steps = n_steps
        self.checkpoint_interval = checkpoint_interval
        self.write_trajectory = write_trajectory
        self.trajectory_interval = trajectory_interval

        # Restart tracking (set when loading restart info)
        self._restart_steps_completed = 0

        self.moves = {
            'md': {'attempted': 0, 'accepted': 0},
            'volume': {'attempted': 0, 'accepted': 0},
            'insertion': {'attempted': 0, 'accepted': 0},
            'deletion': {'attempted': 0, 'accepted': 0},
            'translation': {'attempted': 0, 'accepted': 0},
            'rotation': {'attempted': 0, 'accepted': 0},
        }

        self.log_file_path = Path(self.output_dir) / f"log_tmmc_{self.P/bar:.4f}bar.bin"

    def _debug_print(self, message: str) -> None:
        """Print debug message if debug mode is enabled."""
        if self.debug:
            print(message)

    # ------------------------------------------------------------------
    # Macrostate bookkeeping
    # ------------------------------------------------------------------

    def _state_index(self, N: int) -> int:
        """Index of macrostate N in the collection-matrix arrays."""
        return N - self.N_min

    def _update_C(self, i: int, direction: int, a: float) -> None:
        """
        Record an attempted N-changing move in the collection matrix.

        Parameters
        ----------
        i : int
            Index of the current macrostate.
        direction : int
            +1 for insertion, -1 for deletion, 0 for an attempt that
            cannot change the macrostate (window boundary).
        a : float
            Unbiased Metropolis acceptance probability of the attempt.
        """
        if direction > 0:
            self.C_up[i] += a
        elif direction < 0:
            self.C_down[i] += a
        self.C_stay[i] += 1.0 - a

    def _recompute_lnPi(self) -> None:
        """Refresh the bias estimate of ln Pi(N) from the collection matrix."""
        self.lnPi = lnPi_from_collection(self.C_down, self.C_stay, self.C_up)

    # ------------------------------------------------------------------
    # Unbiased acceptance probabilities (log space)
    # ------------------------------------------------------------------

    def _insertion_ln_ratio(self, e_trial: float, e: float, N: int, V: float) -> float:
        """
        Log of the raw (unclamped) grand-canonical insertion acceptance
        ratio for the N -> N+1 attempt.

        Parameters
        ----------
        e_trial, e : float
            Interaction energies after/before the insertion.
        N : int
            Number of adsorbates before the insertion.
        V : float
            Instantaneous cell volume.
        """
        exp_value = self.beta * (e - e_trial)
        if exp_value > EXP_THRESHOLD:
            exp_value = EXP_THRESHOLD
        elif exp_value < -EXP_THRESHOLD:
            return -np.inf
        return np.log(V * self.beta * self.fugacity / (N + 1)) + exp_value

    def _deletion_ln_ratio(self, e_trial: float, e: float, N: int, V: float) -> float:
        """
        Log of the raw grand-canonical deletion acceptance ratio for the
        N -> N-1 attempt.

        Parameters
        ----------
        e_trial, e : float
            Interaction energies after/before the deletion.
        N : int
            Number of adsorbates before the deletion (>= 1).
        V : float
            Instantaneous cell volume.
        """
        exp_value = -self.beta * (e_trial - e)
        if exp_value > EXP_THRESHOLD:
            exp_value = EXP_THRESHOLD
        elif exp_value < -EXP_THRESHOLD:
            return -np.inf
        return np.log(N / (V * self.beta * self.fugacity)) + exp_value

    def _accept_biased(self, ln_ratio: float, i_old: int, i_new: int) -> bool:
        """
        Metropolis decision on the biased acceptance probability
        min(1, ratio * exp(eta_new - eta_old)) with eta = -ln Pi.
        """
        ln_acc = ln_ratio + self.lnPi[i_old] - self.lnPi[i_new]
        if ln_acc >= 0:
            return True
        return bool(np.log(max(np.random.rand(), 1e-300)) < ln_acc)

    # ------------------------------------------------------------------
    # Moves
    # ------------------------------------------------------------------

    def _attempt_insertion(self, atoms, e, interaction_E, framework_E, adsorbate_E):
        """Attempt a molecule insertion. Returns (atoms, e, interaction_E, success)."""
        self.moves['insertion']['attempted'] += 1
        i = self._state_index(self.Z_ads)

        if self.Z_ads >= self.N_max:
            self._update_C(i, 0, 0.0)
            self.window_rejections += 1
            return atoms, e, interaction_E, False

        atoms_trial = atoms + self.atoms_ads.copy()
        pos = atoms_trial.get_positions()
        pos[-self.n_ads:] = random_position(pos[-self.n_ads:], atoms_trial.get_cell())
        atoms_trial.set_positions(pos)

        if vdw_overlap(atoms_trial, self.vdw, self.n_frame, self.n_ads, self.Z_ads):
            self.insertion_rejected_due_to_vdw += 1
            self._update_C(i, +1, 0.0)
            return atoms, e, interaction_E, False

        atoms_trial.calc = self.model
        e_trial = atoms_trial.get_potential_energy()

        initial_int_E = e_interaction_of_adsorption(e, framework_E, adsorbate_E, self.Z_ads)
        final_int_E = e_interaction_of_adsorption(e_trial, framework_E, adsorbate_E, self.Z_ads + 1)

        ln_ratio = self._insertion_ln_ratio(final_int_E, initial_int_E, self.Z_ads, self.V)
        a = np.exp(min(0.0, ln_ratio))
        self._update_C(i, +1, a)

        if self._accept_biased(ln_ratio, i, i + 1):
            self.Z_ads += 1
            self.moves['insertion']['accepted'] += 1
            self._debug_print(f'✅ accepted insertion, N: {self.Z_ads}, a: {a:.4g}')
            return atoms_trial, e_trial, final_int_E, True

        self.insertion_rejected_due_to_acceptance += 1
        self._debug_print(f'❌ rejected insertion, N: {self.Z_ads}, a: {a:.4g}')
        return atoms, e, interaction_E, False

    def _attempt_deletion(self, atoms, e, interaction_E, framework_E, adsorbate_E):
        """Attempt a molecule deletion. Returns (atoms, e, interaction_E, success)."""
        self.moves['deletion']['attempted'] += 1
        i = self._state_index(self.Z_ads)

        if self.Z_ads <= self.N_min:
            self._update_C(i, 0, 0.0)
            self.window_rejections += 1
            return atoms, e, interaction_E, False

        i_ads = np.random.randint(self.Z_ads)
        atoms_trial = atoms.copy()
        del atoms_trial[self.n_frame + self.n_ads*i_ads : self.n_frame + self.n_ads*(i_ads+1)]
        atoms_trial.calc = self.model
        e_trial = atoms_trial.get_potential_energy()

        initial_int_E = e_interaction_of_adsorption(e, framework_E, adsorbate_E, self.Z_ads)
        final_int_E = e_interaction_of_adsorption(e_trial, framework_E, adsorbate_E, self.Z_ads - 1)

        ln_ratio = self._deletion_ln_ratio(final_int_E, initial_int_E, self.Z_ads, self.V)
        a = np.exp(min(0.0, ln_ratio))
        self._update_C(i, -1, a)

        if self._accept_biased(ln_ratio, i, i - 1):
            self.Z_ads -= 1
            self.moves['deletion']['accepted'] += 1
            self._debug_print(f'✅ accepted deletion, N: {self.Z_ads}, a: {a:.4g}')
            return atoms_trial, e_trial, final_int_E, True

        self.deletion_rejected_due_to_acceptance += 1
        self._debug_print(f'❌ rejected deletion, N: {self.Z_ads}, a: {a:.4g}')
        return atoms, e, interaction_E, False

    def _attempt_translation(self, atoms, e, framework_E, adsorbate_E):
        """Attempt a molecule translation. Returns (atoms, e, interaction_E or None, success)."""
        self.moves['translation']['attempted'] += 1
        if self.Z_ads == 0:
            return atoms, e, None, False
        i_ads = np.random.randint(self.Z_ads)
        atoms_trial = atoms.copy()
        pos = atoms_trial.get_positions()
        pos[self.n_frame + self.n_ads*i_ads : self.n_frame + self.n_ads*(i_ads+1)] += (
            self.translation_step * (np.random.rand(3) - 0.5)
        )
        atoms_trial.set_positions(pos)
        if vdw_overlap(atoms_trial, self.vdw, self.n_frame, self.n_ads, i_ads):
            e_trial = HIGH_ENERGY
        else:
            atoms_trial.calc = self.model
            e_trial = atoms_trial.get_potential_energy()
        acc = min(1, np.exp(-self.beta*(e_trial-e)))
        if acc > np.random.rand():
            interaction_E = e_trial - framework_E - self.Z_ads * adsorbate_E
            self.moves['translation']['accepted'] += 1
            return atoms_trial, e_trial, interaction_E, True
        return atoms, e, None, False

    def _attempt_rotation(self, atoms, e, framework_E, adsorbate_E):
        """Attempt a molecule rotation. Returns (atoms, e, interaction_E or None, success)."""
        self.moves['rotation']['attempted'] += 1
        if self.Z_ads == 0:
            return atoms, e, None, False
        i_ads = np.random.randint(self.Z_ads)
        atoms_trial = atoms.copy()
        pos = atoms_trial.get_positions()
        start_idx = self.n_frame + self.n_ads * i_ads
        end_idx = self.n_frame + self.n_ads * (i_ads + 1)
        pos[start_idx:end_idx] = _random_rotation(
            pos[start_idx:end_idx],
            circlefrac=self.rotation_circlefrac
        )
        atoms_trial.set_positions(pos)
        if vdw_overlap(atoms_trial, self.vdw, self.n_frame, self.n_ads, i_ads):
            e_trial = HIGH_ENERGY
        else:
            atoms_trial.calc = self.model
            e_trial = atoms_trial.get_potential_energy()
        acc = min(1, np.exp(-self.beta*(e_trial-e)))
        if acc > np.random.rand():
            interaction_E = e_trial - framework_E - self.Z_ads * adsorbate_E
            self.moves['rotation']['accepted'] += 1
            return atoms_trial, e_trial, interaction_E, True
        return atoms, e, None, False

    def _move_md(self, atoms, e):
        """
        Hybrid MC/MD move: thermalize the full system (framework +
        adsorbates) with a short MD trajectory using freshly resampled
        Maxwell-Boltzmann velocities. Always accepted, following the
        Gibbs-ensemble MD thermalization convention (Heijmans et al.
        2021). In NPT mode the cell fluctuates and self.cell / self.V
        are refreshed afterwards.

        Returns (atoms, e, success).
        """
        from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

        self.moves['md']['attempted'] += 1
        if len(atoms) == 0:
            return atoms, e, False

        atoms_md = atoms.copy()
        atoms_md.info["charge"] = 0
        atoms_md.info["spin"] = 1
        atoms_md.calc = self.model

        MaxwellBoltzmannDistribution(atoms_md, temperature_K=self.T)

        if self.md_ensemble == 'npt':
            from ase.md.npt import NPT
            cell = np.array(atoms_md.get_cell())
            if not np.allclose(cell, np.triu(cell)):
                raise ValueError(
                    "NPT MD requires an upper-triangular cell. "
                    "Re-orient the framework cell (e.g. with "
                    "ase.build.niggli_reduce or a manual rotation) or use "
                    "md_ensemble='nvt' with MC volume moves."
                )
            dyn = NPT(
                atoms_md,
                timestep=self.md_timestep * ase_units.fs,
                temperature_K=self.T,
                externalstress=self.external_pressure,
                ttime=self.md_damp * ase_units.fs,
                pfactor=self.npt_pfactor,
            )
        else:
            from ase.md.nose_hoover_chain import NoseHooverChainNVT
            dyn = NoseHooverChainNVT(
                atoms_md,
                timestep=self.md_timestep * ase_units.fs,
                temperature_K=self.T,
                tdamp=self.md_damp * ase_units.fs,
            )

        dyn.run(self.md_steps)

        e_new = atoms_md.get_potential_energy()
        if self.md_ensemble == 'npt':
            self.cell = np.array(atoms_md.get_cell())
            self.V = np.linalg.det(self.cell)

        self.moves['md']['accepted'] += 1
        self._debug_print(f'Accepted MD move ({self.md_ensemble}), V: {self.V:.1f} A^3')
        return atoms_md, e_new, True

    def _scale_system(self, atoms, scale):
        """
        Return a copy of ``atoms`` with the cell scaled isotropically by
        ``scale``: framework atoms move affinely, adsorbate molecules are
        shifted by their center of mass (internal geometry preserved).
        """
        pos = atoms.get_positions()
        new_pos = pos.copy()
        new_pos[:self.n_frame] = pos[:self.n_frame] * scale
        for m in range(self.Z_ads):
            s = self.n_frame + self.n_ads * m
            t = s + self.n_ads
            com = pos[s:t].mean(axis=0)
            new_pos[s:t] = pos[s:t] + (scale - 1.0) * com
        atoms_new = atoms.copy()
        atoms_new.set_cell(atoms.get_cell() * scale, scale_atoms=False)
        atoms_new.set_positions(new_pos)
        return atoms_new

    def _move_volume(self, atoms, e):
        """
        MC volume move at constant external pressure: a random
        displacement in ln V, framework scaled affinely and molecules by
        center of mass. Acceptance includes the -beta*P_ext*dV work term
        and the (n_entities + 1) ln(V'/V) Jacobian from sampling in ln V,
        where entities = framework atoms + molecules.

        Returns (atoms, e, success).
        """
        self.moves['volume']['attempted'] += 1
        V_old = self.V
        delta = (np.random.rand() - 0.5) * 2.0 * self.max_delta_lnV
        V_new = V_old * np.exp(delta)
        scale = (V_new / V_old) ** (1.0 / 3.0)

        atoms_trial = self._scale_system(atoms, scale)
        atoms_trial.calc = self.model
        e_trial = atoms_trial.get_potential_energy()

        n_entities = self.n_frame + self.Z_ads
        ln_acc = (
            -self.beta * ((e_trial - e) + self.external_pressure * (V_new - V_old))
            + (n_entities + 1) * delta
        )
        if ln_acc >= 0 or np.log(max(np.random.rand(), 1e-300)) < ln_acc:
            self.cell = np.array(atoms_trial.get_cell())
            self.V = V_new
            self.moves['volume']['accepted'] += 1
            self._debug_print(f'✅ accepted volume move, V: {self.V:.1f} A^3')
            return atoms_trial, e_trial, True
        return atoms, e, False

    # ------------------------------------------------------------------
    # Output / restart
    # ------------------------------------------------------------------

    def _log_step_binary(self, step: int, uptake: int, interaction_energy: float, total_energy: float, atoms: Atoms) -> None:
        """Append a step record to the binary log file (same layout as GCMC)."""
        n_atoms = len(atoms)
        with open(self.log_file_path, "ab") as f:
            f.write(struct.pack("iiddi", step, int(uptake), float(interaction_energy), float(total_energy), n_atoms))

    def _write_trajectory_xyz(self, atoms: Atoms) -> None:
        """Append trajectory snapshot to XYZ file."""
        atoms_for_writing = Atoms(numbers=atoms.numbers, positions=atoms.positions,
                                  cell=atoms.cell, pbc=True)
        traj_file = Path(self.output_dir) / f'traj_tmmc_{self.P/bar:.4f}bar.xyz'
        write(str(traj_file), atoms_for_writing, append=True)

    def _dump_lnPi(self, step: int) -> None:
        """Write the current ln Pi estimate and collection matrix to JSON."""
        data = {
            'step': step,
            'N_min': self.N_min,
            'N_max': self.N_max,
            'N_grid': list(range(self.N_min, self.N_max + 1)),
            'fugacity': float(self.fugacity),
            'temperature': float(self.T),
            'lnPi': self.lnPi.tolist(),
            'H': self.H.tolist(),
            'C_down': self.C_down.tolist(),
            'C_stay': self.C_stay.tolist(),
            'C_up': self.C_up.tolist(),
        }
        filename = Path(self.output_dir) / f"lnPi_{self.P/bar:.4f}bar.json"
        with open(filename, 'w') as f:
            json.dump(data, f, indent=4)

    def _get_restart_paths(self) -> Tuple[str, str]:
        """Get restart file paths."""
        restart_dir = os.path.join(self.output_dir, 'restart')
        restart_xyz = os.path.join(restart_dir, f'restart_tmmc_{self.P/bar:.4f}bar.xyz')
        restart_json = os.path.join(restart_dir, f'restart_tmmc_{self.P/bar:.4f}bar.json')
        return restart_xyz, restart_json

    def _save_restart(self, atoms: Atoms, absolute_step: int) -> None:
        """Save restart info (overwrites previous restart info)."""
        restart_dir = Path(self.output_dir) / 'restart'
        restart_dir.mkdir(parents=True, exist_ok=True)

        atoms_clean = Atoms(
            numbers=atoms.numbers,
            positions=atoms.positions,
            cell=atoms.cell,
            pbc=atoms.pbc
        )
        atoms_clean.calc = None

        restart_xyz, restart_json = self._get_restart_paths()
        write(restart_xyz, atoms_clean)

        restart_data = {
            'Z_ads': self.Z_ads,
            'n_steps': self.n_steps,
            'n_steps_completed': absolute_step,
            'moves': self.moves,
            'rejections': {
                'insertion_vdw': self.insertion_rejected_due_to_vdw,
                'insertion_acc': self.insertion_rejected_due_to_acceptance,
                'deletion_acc': self.deletion_rejected_due_to_acceptance,
                'window': self.window_rejections,
            },
            'tmmc': {
                'N_min': self.N_min,
                'N_max': self.N_max,
                'fugacity': float(self.fugacity),
                'C_down': self.C_down.tolist(),
                'C_stay': self.C_stay.tolist(),
                'C_up': self.C_up.tolist(),
                'H': self.H.tolist(),
                'lnPi': self.lnPi.tolist(),
            }
        }
        with open(restart_json, 'w') as f:
            json.dump(restart_data, f, indent=4)

    def _load_restart_info(self) -> Optional[Atoms]:
        """Load the state from restart directory (automatic restart)."""
        if not self.output_dir:
            return None

        try:
            restart_dir = Path(self.output_dir) / 'restart'
            if not restart_dir.exists():
                print("--- No restart directory found. Starting new simulation. ---")
                return None

            restart_xyz, restart_json = self._get_restart_paths()

            if os.path.exists(restart_xyz) and os.path.exists(restart_json):
                atoms = read(restart_xyz)
                with open(restart_json, 'r') as f:
                    restart_data = json.load(f)

                tmmc = restart_data.get('tmmc')
                if tmmc is None:
                    print("--- Restart file has no TMMC block. Starting new simulation. ---")
                    return None
                if tmmc['N_min'] != self.N_min or tmmc['N_max'] != self.N_max:
                    print(
                        f"--- Restart window [{tmmc['N_min']}, {tmmc['N_max']}] does not "
                        f"match requested [{self.N_min}, {self.N_max}]. Starting new simulation. ---"
                    )
                    return None

                self.Z_ads = restart_data.get('Z_ads', 0)
                self.moves = restart_data.get('moves', self.moves)
                rejections = restart_data.get('rejections', {})
                self.insertion_rejected_due_to_vdw = rejections.get('insertion_vdw', 0)
                self.insertion_rejected_due_to_acceptance = rejections.get('insertion_acc', 0)
                self.deletion_rejected_due_to_acceptance = rejections.get('deletion_acc', 0)
                self.window_rejections = rejections.get('window', 0)

                self.C_down = np.array(tmmc['C_down'], dtype=float)
                self.C_stay = np.array(tmmc['C_stay'], dtype=float)
                self.C_up = np.array(tmmc['C_up'], dtype=float)
                self.H = np.array(tmmc['H'], dtype=np.int64)
                self.lnPi = np.array(tmmc['lnPi'], dtype=float)

                self._restart_steps_completed = restart_data.get('n_steps_completed', 0) or 0

                # Cell may have changed via NPT/volume moves
                self.cell = np.array(atoms.get_cell())
                self.V = np.linalg.det(self.cell)

                print(f"--- Restarting TMMC from restart info, Z_ads = {self.Z_ads}, "
                      f"{self._restart_steps_completed} steps completed ---")
                return atoms

            print("--- No valid restart files found in restart directory. Starting new simulation. ---")
            return None
        except Exception as e:
            print(f"--- Error loading restart files: {e}. Starting new simulation. ---", file=sys.stderr)
            traceback.print_exc()
            return None

    def _print_statistics(self):
        """Prints the final MC move statistics."""
        print("\n  ┌" + "─" * 66 + "┐")
        print("  │" + " TMMC Move Statistics".center(66) + "│")
        print("  ├" + "─" * 66 + "┤")

        for move_type, stats in self.moves.items():
            attempted = stats['attempted']
            accepted = stats['accepted']
            if attempted > 0:
                acceptance_rate = (accepted / attempted) * 100
                move_name = move_type.capitalize()
                line = f"  │ {move_name:<13} Attempted: {attempted:<7} Accepted: {accepted:<7} Rate: {acceptance_rate:>6.2f}% │"
                print(line)
            else:
                line = f"  │ {move_type.capitalize():<13} Not attempted{'':<37} │"
                print(line)

        print("  ├" + "─" * 66 + "┤")
        visited = int(np.sum(self.H > 0))
        stats_lines = [
            ("VDW Overlap Rejections", self.insertion_rejected_due_to_vdw),
            ("Insertion Rejections", self.insertion_rejected_due_to_acceptance),
            ("Deletion Rejections", self.deletion_rejected_due_to_acceptance),
            ("Window Boundary Rejections", self.window_rejections),
            ("Macrostates Visited", f"{visited}/{self.M}"),
        ]
        for label, value in stats_lines:
            print(f"  │ {label:<30} {str(value):>33} │")

        print("  └" + "─" * 66 + "┘\n")

    def _save_results_json(self) -> None:
        """Save the final ln Pi, histogram, and collection matrix to JSON."""
        results_data = {
            'temperature': float(self.T),
            'reference_pressure_bar': float(self.P / bar),
            'reference_fugacity': float(self.fugacity),
            'N_min': self.N_min,
            'N_max': self.N_max,
            'N_grid': list(range(self.N_min, self.N_max + 1)),
            'lnPi': self.lnPi.tolist(),
            'H': self.H.tolist(),
            'C_down': self.C_down.tolist(),
            'C_stay': self.C_stay.tolist(),
            'C_up': self.C_up.tolist(),
        }
        filename = Path(self.output_dir) / f"results_tmmc_{self.P/bar:.4f}bar.json"
        with open(filename, 'w') as f:
            json.dump(results_data, f, indent=4)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def _initialize_loading(self, atoms, target_N, max_tries_per_molecule=1000):
        """Insert molecules at random non-overlapping positions until the
        loading reaches ``target_N`` (used to enter the macrostate window)."""
        while self.Z_ads < target_N:
            placed = False
            for _ in range(max_tries_per_molecule):
                atoms_trial = atoms + self.atoms_ads.copy()
                pos = atoms_trial.get_positions()
                pos[-self.n_ads:] = random_position(pos[-self.n_ads:], atoms_trial.get_cell())
                atoms_trial.set_positions(pos)
                if not vdw_overlap(atoms_trial, self.vdw, self.n_frame, self.n_ads, self.Z_ads):
                    atoms = atoms_trial
                    self.Z_ads += 1
                    placed = True
                    break
            if not placed:
                raise RuntimeError(
                    f"Could not place molecule {self.Z_ads + 1}/{target_N} "
                    f"without VDW overlap; N_min may be too high for this cell."
                )
        return atoms

    def run(self, N: int) -> None:
        """
        Run the TMMC simulation for N Monte Carlo steps.

        If restarting, continues from the saved collection matrix and
        runs only the remaining steps up to N total.

        Parameters
        ----------
        N : int
            Total number of Monte Carlo steps (target across restarts).
        """
        atoms = self._load_restart_info()
        if atoms is None:
            atoms = self.atoms_frame.copy()
            self.Z_ads = 0
            self._restart_steps_completed = 0
            if self.N_min > 0:
                atoms = self._initialize_loading(atoms, self.N_min)

        if len(atoms) > 0:
            atoms.calc = self.model
            e = atoms.get_potential_energy()
        else:
            e = 0.0

        # Reference energies for the interaction-energy bookkeeping.
        # With MD moves the framework deforms, but only energy differences
        # enter the acceptance ratios, so constant references are exact.
        atoms_ads_cell = self.atoms_ads.copy()
        atoms_ads_cell.calc = self.model
        atoms_ads_cell.set_cell(self.cell, scale_atoms=False)
        atoms_ads_cell.set_pbc(True)
        positions = atoms_ads_cell.get_positions()
        positions = random_position(positions, self.cell)
        atoms_ads_cell.set_positions(positions)
        adsorbate_E = atoms_ads_cell.get_potential_energy()

        if self.n_frame > 0:
            framework = self.atoms_frame.copy()
            framework.calc = self.model
            framework_E = framework.get_potential_energy()
        else:
            framework_E = 0.0

        interaction_E = e_interaction_of_adsorption(e, framework_E, adsorbate_E, self.Z_ads)
        print(f'framework E: {framework_E}, adsorbate_E: {adsorbate_E}, initial interaction E: {interaction_E}')

        steps_to_run = N - self._restart_steps_completed
        if steps_to_run <= 0:
            print(f"--- Already completed {self._restart_steps_completed} steps. "
                  f"Target is {N} steps. Nothing to do. ---")
            return
        if self._restart_steps_completed > 0:
            print(f"--- Continuing TMMC: {steps_to_run} steps remaining "
                  f"(already completed {self._restart_steps_completed}/{N}) ---")

        p = self.move_probabilities
        for iteration in range(steps_to_run):
            success = False
            switch = np.random.rand()

            if switch < p['md']:
                atoms, e, success = self._move_md(atoms, e)
                if success:
                    interaction_E = e_interaction_of_adsorption(e, framework_E, adsorbate_E, self.Z_ads)
            elif switch < p['volume']:
                atoms, e, success = self._move_volume(atoms, e)
                if success:
                    interaction_E = e_interaction_of_adsorption(e, framework_E, adsorbate_E, self.Z_ads)
            elif switch < p['insertion']:
                atoms, e, interaction_E, success = self._attempt_insertion(
                    atoms, e, interaction_E, framework_E, adsorbate_E)
            elif switch < p['deletion']:
                atoms, e, interaction_E, success = self._attempt_deletion(
                    atoms, e, interaction_E, framework_E, adsorbate_E)
            elif switch < p['translation']:
                atoms, e, new_int_E, success = self._attempt_translation(
                    atoms, e, framework_E, adsorbate_E)
                if success:
                    interaction_E = new_int_E
            else:
                atoms, e, new_int_E, success = self._attempt_rotation(
                    atoms, e, framework_E, adsorbate_E)
                if success:
                    interaction_E = new_int_E

            self.H[self._state_index(self.Z_ads)] += 1
            current_step = iteration + 1 + self._restart_steps_completed

            if success:
                self._log_step_binary(current_step, self.Z_ads, interaction_E, e, atoms)

            if current_step % self.bias_update_interval == 0:
                self._recompute_lnPi()
                self._dump_lnPi(current_step)
                self._save_restart(atoms, current_step)
            elif current_step % self.checkpoint_interval == 0:
                # C accumulates on attempts, not only accepted moves, so
                # persist periodically even without accepted moves
                self._save_restart(atoms, current_step)

            if current_step % self.trajectory_interval == 0 and self.write_trajectory:
                self._write_trajectory_xyz(atoms)

        print("--- Simulation finished. Performing final save. ---")
        total_steps_completed = self._restart_steps_completed + steps_to_run
        self._recompute_lnPi()
        self._dump_lnPi(total_steps_completed)
        self._save_restart(atoms, total_steps_completed)
        self._save_results_json()
        self._print_statistics()
