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
from .utilities import _random_rotation, random_position, vdw_overlap


# Constants
EXP_THRESHOLD = 100.0
VDW_OFFSET = 0.35
HIGH_ENERGY = 10**10
TRANSLATION_STEP = 0.5
ROTATION_CIRCLEFRAC = 0.1

GIBBS_MOVE_PROBABILITIES = {
    'translation': 0.40,
    'rotation': 0.70,
    'volume': 0.80,
    'swap': 1.00,
}


class MLP_Gibbs:
    """
    Gibbs Ensemble Monte Carlo simulation for vapor-liquid equilibria.

    Simulates two cubic boxes of pure fluid that exchange volume and
    particles at constant total N, V, and T. Machine-learned interatomic
    potentials provide energy evaluations.

    Parameters
    ----------
    model : calculator
        ASE calculator (MLIP backend)
    atoms_mol : Atoms
        Single molecule template
    T : float
        Temperature in Kelvin
    N1_init : int
        Initial number of molecules in box 1
    N2_init : int
        Initial number of molecules in box 2
    L1_init : float
        Initial side length of box 1 in Angstrom
    L2_init : float
        Initial side length of box 2 in Angstrom
    device : str
        Device for calculations ('cuda' or 'cpu')
    vdw_radii : array_like
        Van der Waals radii indexed by atomic number
    move_probabilities : dict, optional
        Cumulative move probabilities. Keys: 'translation', 'rotation',
        'volume', 'swap'. Values must be increasing and end at 1.0.
    max_delta_V : float, optional
        Maximum volume change per move in A^3 (default: 50.0)
    translation_step : float, optional
        Translation step size in Angstrom (default: 0.5)
    rotation_circlefrac : float, optional
        Fraction of full rotation circle (default: 0.1)
    debug : bool, optional
        Enable debug printing (default: False)
    output_dir : str, optional
        Output directory for results (default: 'results')
    n_equilibration_steps : int, optional
        Target equilibration steps
    n_production_steps : int, optional
        Target production steps
    checkpoint_interval : int, optional
        Steps between checkpoints (default: 10000)
    write_trajectory : bool, optional
        Write XYZ trajectory files (default: False)
    trajectory_interval : int, optional
        Steps between trajectory writes (default: 100)
    overwrite_checkpoints : bool, optional
        Overwrite checkpoint files (default: False)
    """

    def __init__(
        self,
        model,
        atoms_mol,
        T,
        N1_init,
        N2_init,
        L1_init,
        L2_init,
        device,
        vdw_radii,
        move_probabilities=None,
        max_delta_V=50.0,
        translation_step=0.5,
        rotation_circlefrac=0.1,
        debug=False,
        output_dir='results',
        n_equilibration_steps=None,
        n_production_steps=None,
        checkpoint_interval=10000,
        write_trajectory=False,
        trajectory_interval=10,
        overwrite_checkpoints=False,
    ):
        self.model = model
        self.atoms_mol = atoms_mol
        self.n_mol = len(atoms_mol)
        self.T = T
        self.device = device
        self.boltzmann = ase_units.kB
        self.beta = 1.0 / (self.boltzmann * T)
        self.debug = debug
        self.output_dir = output_dir
        self.max_delta_V = max_delta_V
        self.translation_step = translation_step
        self.rotation_circlefrac = rotation_circlefrac
        self.n_equilibration_steps = n_equilibration_steps
        self.n_production_steps = n_production_steps
        self.checkpoint_interval = checkpoint_interval
        self.write_trajectory = write_trajectory
        self.trajectory_interval = trajectory_interval
        self.overwrite_checkpoints = overwrite_checkpoints

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)

        # Move probabilities (cumulative)
        if move_probabilities is not None:
            self.move_probabilities = move_probabilities
        else:
            self.move_probabilities = GIBBS_MOVE_PROBABILITIES.copy()

        # VDW radii with offset
        self.vdw = vdw_radii - VDW_OFFSET

        # Initialize molecule counts
        self.N1 = N1_init
        self.N2 = N2_init

        # Initialize boxes
        self.atoms_box1 = self._initialize_box(L1_init, N1_init)
        self.atoms_box2 = self._initialize_box(L2_init, N2_init)

        # Volumes
        self.V1 = L1_init ** 3
        self.V2 = L2_init ** 3
        self.V_total = self.V1 + self.V2

        # Energies (computed at start of run)
        self.E1 = None
        self.E2 = None

        # Restart tracking
        self._restart_n_equil_completed = 0
        self._restart_n_prod_completed = 0
        self._restart_iteration = -1

        # Move statistics
        self.moves = {
            'translation': {'attempted': 0, 'accepted': 0},
            'rotation': {'attempted': 0, 'accepted': 0},
            'volume': {'attempted': 0, 'accepted': 0},
            'swap': {'attempted': 0, 'accepted': 0},
        }

        # Rejection tracking
        self.swap_rejected_vdw = 0
        self.swap_rejected_empty_source = 0
        self.volume_rejected_negative = 0
        self.volume_rejected_overlap = 0

        # Binary log file path
        self.log_file_path = Path(self.output_dir) / f"log_gibbs_{self.T:.1f}K.bin"

    def _initialize_box(self, L, N_molecules):
        """
        Create a cubic box and place molecules randomly inside.

        Parameters
        ----------
        L : float
            Cubic box side length in Angstrom
        N_molecules : int
            Number of molecules to place

        Returns
        -------
        Atoms
            ASE Atoms object with cubic cell containing N_molecules
        """
        cell = np.diag([L, L, L])
        atoms = Atoms(cell=cell, pbc=True)

        for i in range(N_molecules):
            mol = self.atoms_mol.copy()
            mol.set_cell(cell, scale_atoms=False)
            mol.set_pbc(True)

            placed = False
            for attempt in range(1000):
                pos = mol.get_positions().copy()
                pos = random_position(pos, cell)
                mol.set_positions(pos)

                # Build trial structure with all placed molecules + this one
                trial = atoms + mol
                if i == 0 or not vdw_overlap(trial, self.vdw, 0, self.n_mol, i):
                    atoms = trial
                    placed = True
                    break

            if not placed:
                raise RuntimeError(
                    f"Failed to place molecule {i+1}/{N_molecules} in box of "
                    f"side length {L} A after 1000 attempts. "
                    f"Try a larger box or fewer molecules."
                )

        return atoms

    def _compute_energy(self, atoms):
        """
        Compute potential energy of a box. Returns 0.0 for empty boxes.

        Parameters
        ----------
        atoms : Atoms
            Box configuration

        Returns
        -------
        float
            Potential energy in eV
        """
        if len(atoms) == 0:
            return 0.0
        atoms.info["charge"] = 0  # total charge
        atoms.info["spin"] = 1  #  spin multiplicity
        atoms.calc = self.model
        return atoms.get_potential_energy()

    def _any_overlap_in_box(self, atoms, N_molecules):
        """
        Check if any molecule in the box overlaps with any other.

        Parameters
        ----------
        atoms : Atoms
            Box configuration
        N_molecules : int
            Number of molecules in the box

        Returns
        -------
        bool
            True if any overlap found
        """
        for i in range(N_molecules):
            if vdw_overlap(atoms, self.vdw, 0, self.n_mol, i):
                return True
        return False

    def _debug_print(self, message):
        """Print debug message if debug mode is enabled."""
        if self.debug:
            print(message)

    def _move_translation(self):
        """
        Perform a translation move on a random molecule in a random box.

        Returns
        -------
        bool
            True if accepted
        """
        N_total = self.N1 + self.N2
        if N_total == 0:
            return False

        # Select box weighted by molecule count
        if np.random.rand() < self.N1 / N_total:
            box_id = 1
            atoms = self.atoms_box1
            N = self.N1
            E_current = self.E1
        else:
            box_id = 2
            atoms = self.atoms_box2
            N = self.N2
            E_current = self.E2

        if N == 0:
            return False

        i_mol = np.random.randint(N)
        atoms_trial = atoms.copy()
        pos = atoms_trial.get_positions()
        start = self.n_mol * i_mol
        end = self.n_mol * (i_mol + 1)
        pos[start:end] += self.translation_step * (np.random.rand(3) - 0.5)
        atoms_trial.set_positions(pos)

        if vdw_overlap(atoms_trial, self.vdw, 0, self.n_mol, i_mol):
            e_trial = HIGH_ENERGY
        else:
            e_trial = self._compute_energy(atoms_trial)

        acc = min(1.0, np.exp(-self.beta * (e_trial - E_current)))
        if np.random.rand() < acc:
            if box_id == 1:
                self.atoms_box1 = atoms_trial
                self.E1 = e_trial
            else:
                self.atoms_box2 = atoms_trial
                self.E2 = e_trial
            self.moves['translation']['accepted'] += 1
            self._debug_print(f'Accepted translation in box {box_id}')
            return True

        return False

    def _move_rotation(self):
        """
        Perform a rotation move on a random molecule in a random box.

        Returns
        -------
        bool
            True if accepted
        """
        N_total = self.N1 + self.N2
        if N_total == 0:
            return False

        if np.random.rand() < self.N1 / N_total:
            box_id = 1
            atoms = self.atoms_box1
            N = self.N1
            E_current = self.E1
        else:
            box_id = 2
            atoms = self.atoms_box2
            N = self.N2
            E_current = self.E2

        if N == 0:
            return False

        i_mol = np.random.randint(N)
        atoms_trial = atoms.copy()
        pos = atoms_trial.get_positions()
        start = self.n_mol * i_mol
        end = self.n_mol * (i_mol + 1)
        pos[start:end] = _random_rotation(
            pos[start:end], circlefrac=self.rotation_circlefrac
        )
        atoms_trial.set_positions(pos)

        if vdw_overlap(atoms_trial, self.vdw, 0, self.n_mol, i_mol):
            e_trial = HIGH_ENERGY
        else:
            e_trial = self._compute_energy(atoms_trial)

        acc = min(1.0, np.exp(-self.beta * (e_trial - E_current)))
        if np.random.rand() < acc:
            if box_id == 1:
                self.atoms_box1 = atoms_trial
                self.E1 = e_trial
            else:
                self.atoms_box2 = atoms_trial
                self.E2 = e_trial
            self.moves['rotation']['accepted'] += 1
            self._debug_print(f'Accepted rotation in box {box_id}')
            return True

        return False

    def _move_volume(self):
        """
        Perform a coupled volume change move preserving total volume.

        Returns
        -------
        bool
            True if accepted
        """
        dV = self.max_delta_V * (2.0 * np.random.rand() - 1.0)
        V1_new = self.V1 + dV
        V2_new = self.V2 - dV

        # Reject if either volume is non-positive
        if V1_new <= 0.0 or V2_new <= 0.0:
            self.volume_rejected_negative += 1
            return False

        L1_new = V1_new ** (1.0 / 3.0)
        L2_new = V2_new ** (1.0 / 3.0)

        # Scale box 1
        atoms_trial1 = self.atoms_box1.copy()
        if len(atoms_trial1) > 0:
            atoms_trial1.set_cell([L1_new, L1_new, L1_new], scale_atoms=False)

        # Scale box 2
        atoms_trial2 = self.atoms_box2.copy()
        if len(atoms_trial2) > 0:
            atoms_trial2.set_cell([L2_new, L2_new, L2_new], scale_atoms=False)

        # Check overlaps in both boxes
        if self.N1 > 1 and self._any_overlap_in_box(atoms_trial1, self.N1):
            self.volume_rejected_overlap += 1
            return False
        if self.N2 > 1 and self._any_overlap_in_box(atoms_trial2, self.N2):
            self.volume_rejected_overlap += 1
            return False

        # Compute trial energies
        E1_trial = self._compute_energy(atoms_trial1)
        E2_trial = self._compute_energy(atoms_trial2)

        # Acceptance criterion
        dE = (E1_trial + E2_trial) - (self.E1 + self.E2)
        arg = -self.beta * dE
        if self.N1 > 0:
            arg += self.N1 * np.log(V1_new / self.V1)
        if self.N2 > 0:
            arg += self.N2 * np.log(V2_new / self.V2)

        if arg > EXP_THRESHOLD:
            accepted = True
        elif arg < -EXP_THRESHOLD:
            accepted = False
        else:
            acc = min(1.0, np.exp(arg))
            accepted = np.random.rand() < acc

        if accepted:
            self.atoms_box1 = atoms_trial1
            self.atoms_box2 = atoms_trial2
            self.E1 = E1_trial
            self.E2 = E2_trial
            self.V1 = V1_new
            self.V2 = V2_new
            self.moves['volume']['accepted'] += 1
            self._debug_print(
                f'Accepted volume change: V1={V1_new:.1f}, V2={V2_new:.1f}'
            )
            return True

        return False

    def _move_swap(self):
        """
        Transfer a molecule from one box to the other.

        Returns
        -------
        bool
            True if accepted
        """
        # Pick source box randomly (50/50)
        if np.random.rand() < 0.5:
            source_id = 1
            N_source, N_target = self.N1, self.N2
            atoms_source, atoms_target = self.atoms_box1, self.atoms_box2
            E_source, E_target = self.E1, self.E2
            V_source, V_target = self.V1, self.V2
        else:
            source_id = 2
            N_source, N_target = self.N2, self.N1
            atoms_source, atoms_target = self.atoms_box2, self.atoms_box1
            E_source, E_target = self.E2, self.E1
            V_source, V_target = self.V2, self.V1

        if N_source == 0:
            self.swap_rejected_empty_source += 1
            return False

        # Select random molecule from source
        i_mol = np.random.randint(N_source)
        start = self.n_mol * i_mol
        end = self.n_mol * (i_mol + 1)

        # Extract the molecule being transferred
        mol_positions = atoms_source.get_positions()[start:end].copy()
        mol_numbers = atoms_source.get_atomic_numbers()[start:end].copy()
        transferred_mol = Atoms(
            numbers=mol_numbers,
            positions=mol_positions,
        )

        # Remove molecule from source
        atoms_source_trial = atoms_source.copy()
        del atoms_source_trial[start:end]

        # Insert molecule into target at random position
        transferred_mol_copy = transferred_mol.copy()
        target_cell = atoms_target.get_cell()
        new_pos = random_position(transferred_mol_copy.get_positions(), target_cell)
        transferred_mol_copy.set_positions(new_pos)

        atoms_target_trial = atoms_target + transferred_mol_copy
        atoms_target_trial.set_cell(target_cell, scale_atoms=False)
        atoms_target_trial.set_pbc(True)

        # VDW overlap check in target (new molecule is at index N_target)
        if vdw_overlap(atoms_target_trial, self.vdw, 0, self.n_mol, N_target):
            self.swap_rejected_vdw += 1
            return False

        # Compute trial energies
        E_source_trial = self._compute_energy(atoms_source_trial)
        E_target_trial = self._compute_energy(atoms_target_trial)

        # Acceptance criterion
        dE = (E_source_trial + E_target_trial) - (E_source + E_target)
        arg = -self.beta * dE + np.log(
            N_source * V_target / ((N_target + 1) * V_source)
        )

        if arg > EXP_THRESHOLD:
            accepted = True
        elif arg < -EXP_THRESHOLD:
            accepted = False
        else:
            acc = min(1.0, np.exp(arg))
            accepted = np.random.rand() < acc

        if accepted:
            if source_id == 1:
                self.atoms_box1 = atoms_source_trial
                self.atoms_box2 = atoms_target_trial
                self.E1 = E_source_trial
                self.E2 = E_target_trial
                self.N1 -= 1
                self.N2 += 1
            else:
                self.atoms_box2 = atoms_source_trial
                self.atoms_box1 = atoms_target_trial
                self.E2 = E_source_trial
                self.E1 = E_target_trial
                self.N2 -= 1
                self.N1 += 1
            self.moves['swap']['accepted'] += 1
            self._debug_print(
                f'Accepted swap from box {source_id}: N1={self.N1}, N2={self.N2}'
            )
            return True

        return False

    def _log_step_binary(self, step):
        """Append step record to binary log file."""
        rho1 = self.N1 / self.V1 if self.V1 > 0 else 0.0
        rho2 = self.N2 / self.V2 if self.V2 > 0 else 0.0
        with open(self.log_file_path, "ab") as f:
            f.write(struct.pack(
                "iiidddddd",
                step, self.N1, self.N2,
                self.V1, self.V2,
                self.E1, self.E2,
                rho1, rho2,
            ))

    def _write_trajectory_xyz(self):
        """Append trajectory snapshots for both boxes."""
        for box_id, atoms in [(1, self.atoms_box1), (2, self.atoms_box2)]:
            if len(atoms) == 0:
                continue
            atoms_clean = Atoms(
                numbers=atoms.numbers,
                positions=atoms.positions,
                cell=atoms.cell,
                pbc=True,
            )
            traj_file = Path(self.output_dir) / f'traj_gibbs_{self.T:.1f}K_box{box_id}.xyz'
            write(str(traj_file), atoms_clean, append=True)

    def _get_restart_paths(self):
        """Get restart file paths."""
        restart_dir = os.path.join(self.output_dir, 'restart')
        xyz1 = os.path.join(restart_dir, f'restart_gibbs_{self.T:.1f}K_box1.xyz')
        xyz2 = os.path.join(restart_dir, f'restart_gibbs_{self.T:.1f}K_box2.xyz')
        json_path = os.path.join(restart_dir, f'restart_gibbs_{self.T:.1f}K.json')
        return xyz1, xyz2, json_path

    def _save_history_checkpoint(self, step):
        """Save a history checkpoint."""
        parent = Path(self.output_dir) / f'checkpoints_gibbs_{self.T:.1f}K'
        parent.mkdir(parents=True, exist_ok=True)

        if self.overwrite_checkpoints:
            checkpoint_dir = parent / 'checkpoint'
        else:
            checkpoint_dir = parent / f'checkpoint_{step}'
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        for box_id, atoms in [(1, self.atoms_box1), (2, self.atoms_box2)]:
            if len(atoms) > 0:
                atoms_clean = Atoms(
                    numbers=atoms.numbers,
                    positions=atoms.positions,
                    cell=atoms.cell,
                    pbc=True,
                )
                write(str(checkpoint_dir / f'box{box_id}.xyz'), atoms_clean)

        results_data = {
            'n_iter': step,
            'N1': self.N1,
            'N2': self.N2,
            'V1': self.V1,
            'V2': self.V2,
            'E1': float(self.E1) if self.E1 is not None else 0.0,
            'E2': float(self.E2) if self.E2 is not None else 0.0,
        }
        with open(checkpoint_dir / 'results.json', 'w') as f:
            json.dump(results_data, f, indent=4)

    def _save_restart(self, absolute_step):
        """Save restart state."""
        restart_dir = Path(self.output_dir) / 'restart'
        restart_dir.mkdir(parents=True, exist_ok=True)

        xyz1, xyz2, json_path = self._get_restart_paths()

        for path, atoms in [(xyz1, self.atoms_box1), (xyz2, self.atoms_box2)]:
            if len(atoms) > 0:
                atoms_clean = Atoms(
                    numbers=atoms.numbers,
                    positions=atoms.positions,
                    cell=atoms.cell,
                    pbc=True,
                )
                write(path, atoms_clean)
            else:
                # Write empty marker for empty boxes
                with open(path, 'w') as f:
                    f.write('')

        # Calculate equilibration/production progress
        if self.n_equilibration_steps is not None:
            if absolute_step <= self.n_equilibration_steps:
                n_equil_completed = absolute_step
                n_prod_completed = 0
            else:
                n_equil_completed = self.n_equilibration_steps
                n_prod_completed = absolute_step - self.n_equilibration_steps
        else:
            n_equil_completed = None
            n_prod_completed = None

        restart_data = {
            'N1': self.N1,
            'N2': self.N2,
            'V1': self.V1,
            'V2': self.V2,
            'E1': float(self.E1) if self.E1 is not None else 0.0,
            'E2': float(self.E2) if self.E2 is not None else 0.0,
            'n_equil': self.n_equilibration_steps,
            'n_equil_completed': n_equil_completed,
            'n_prod': self.n_production_steps,
            'n_prod_completed': n_prod_completed,
            'moves': self.moves,
            'rejections': {
                'swap_vdw': self.swap_rejected_vdw,
                'swap_empty_source': self.swap_rejected_empty_source,
                'volume_negative': self.volume_rejected_negative,
                'volume_overlap': self.volume_rejected_overlap,
            },
        }
        with open(json_path, 'w') as f:
            json.dump(restart_data, f, indent=4)

    def _load_restart_info(self):
        """Load state from restart directory."""
        if not self.output_dir:
            return False

        try:
            restart_dir = Path(self.output_dir) / 'restart'
            if not restart_dir.exists():
                print("--- No restart directory found. Starting new simulation. ---")
                return False

            xyz1, xyz2, json_path = self._get_restart_paths()

            if not os.path.exists(json_path):
                print("--- No valid restart files found. Starting new simulation. ---")
                return False

            with open(json_path, 'r') as f:
                restart_data = json.load(f)

            self.N1 = restart_data.get('N1', 0)
            self.N2 = restart_data.get('N2', 0)
            self.V1 = restart_data.get('V1', self.V1)
            self.V2 = restart_data.get('V2', self.V2)
            self.E1 = restart_data.get('E1', 0.0)
            self.E2 = restart_data.get('E2', 0.0)
            self.moves = restart_data.get('moves', self.moves)

            rejections = restart_data.get('rejections', {})
            self.swap_rejected_vdw = rejections.get('swap_vdw', 0)
            self.swap_rejected_empty_source = rejections.get('swap_empty_source', 0)
            self.volume_rejected_negative = rejections.get('volume_negative', 0)
            self.volume_rejected_overlap = rejections.get('volume_overlap', 0)

            n_equil_completed = restart_data.get('n_equil_completed', 0) or 0
            n_prod_completed = restart_data.get('n_prod_completed', 0) or 0
            self._restart_n_equil_completed = n_equil_completed
            self._restart_n_prod_completed = n_prod_completed

            total_steps_completed = n_equil_completed + n_prod_completed
            self._restart_iteration = total_steps_completed - 1 if total_steps_completed > 0 else 0

            # Load box structures
            if os.path.exists(xyz1) and os.path.getsize(xyz1) > 0:
                self.atoms_box1 = read(xyz1)
            else:
                L1 = self.V1 ** (1.0 / 3.0)
                self.atoms_box1 = Atoms(cell=[L1, L1, L1], pbc=True)

            if os.path.exists(xyz2) and os.path.getsize(xyz2) > 0:
                self.atoms_box2 = read(xyz2)
            else:
                L2 = self.V2 ** (1.0 / 3.0)
                self.atoms_box2 = Atoms(cell=[L2, L2, L2], pbc=True)

            print(f"--- Restarting Gibbs simulation, N1={self.N1}, N2={self.N2} ---")
            print(f"    Equilibration: {n_equil_completed}/{self.n_equilibration_steps} steps completed")
            print(f"    Production: {n_prod_completed}/{self.n_production_steps} steps completed")

            return True

        except Exception as e:
            print(f"--- Error loading restart files: {e}. Starting new simulation. ---", file=sys.stderr)
            traceback.print_exc()
            return False

    def _print_statistics(self):
        """Print final MC move statistics."""
        print("\n  +" + "-" * 66 + "+")
        print("  |" + " Gibbs Ensemble MC Move Statistics".center(66) + "|")
        print("  +" + "-" * 66 + "+")

        for move_type, stats in self.moves.items():
            attempted = stats['attempted']
            accepted = stats['accepted']
            if attempted > 0:
                rate = (accepted / attempted) * 100
                name = move_type.capitalize()
                print(f"  | {name:<13} Attempted: {attempted:<7} Accepted: {accepted:<7} Rate: {rate:>6.2f}% |")
            else:
                print(f"  | {move_type.capitalize():<13} Not attempted{'':<37} |")

        print("  +" + "-" * 66 + "+")
        stats_lines = [
            ("Swap VDW Rejections", self.swap_rejected_vdw),
            ("Swap Empty Source", self.swap_rejected_empty_source),
            ("Volume Negative Rejections", self.volume_rejected_negative),
            ("Volume Overlap Rejections", self.volume_rejected_overlap),
        ]
        for label, value in stats_lines:
            print(f"  | {label:<30} {value:>33} |")

        print("  +" + "-" * 66 + "+")

        # Final state
        rho1 = self.N1 / self.V1 if self.V1 > 0 else 0.0
        rho2 = self.N2 / self.V2 if self.V2 > 0 else 0.0
        print(f"\n  Final State:")
        print(f"    Box 1: N={self.N1}, V={self.V1:.1f} A^3, rho={rho1:.6f} mol/A^3")
        print(f"    Box 2: N={self.N2}, V={self.V2:.1f} A^3, rho={rho2:.6f} mol/A^3")

    def _save_results_json(self, N1_list, N2_list, V1_list, V2_list, E1_list, E2_list):
        """Save results to JSON file."""
        results_data = {
            'N1': N1_list,
            'N2': N2_list,
            'V1': V1_list,
            'V2': V2_list,
            'E1': E1_list,
            'E2': E2_list,
        }
        filename = Path(self.output_dir) / f"results_gibbs_{self.T:.1f}K.json"
        with open(filename, 'w') as f:
            json.dump(results_data, f, indent=4)

    def run(self, N):
        """
        Run the Gibbs Ensemble MC simulation for N steps.

        Parameters
        ----------
        N : int
            Total number of MC steps (equilibration + production).
            If restarting, only the remaining steps will be run.
        """
        # Result lists
        N1_list = []
        N2_list = []
        V1_list = []
        V2_list = []
        E1_list = []
        E2_list = []

        # Load previous results if restarting
        results_file = Path(self.output_dir) / f"results_gibbs_{self.T:.1f}K.json"
        if results_file.exists():
            try:
                with open(results_file, 'r') as f:
                    prev = json.load(f)
                N1_list = prev.get('N1', [])
                N2_list = prev.get('N2', [])
                V1_list = prev.get('V1', [])
                V2_list = prev.get('V2', [])
                E1_list = prev.get('E1', [])
                E2_list = prev.get('E2', [])
            except Exception as e:
                print(f"Warning: could not load previous results: {e}")

        restarted = self._load_restart_info()

        # Compute initial energies
        self.E1 = self._compute_energy(self.atoms_box1)
        self.E2 = self._compute_energy(self.atoms_box2)

        print(f'Initial E1: {self.E1}, E2: {self.E2}')
        print(f'Initial N1: {self.N1}, N2: {self.N2}')
        print(f'Initial V1: {self.V1:.1f}, V2: {self.V2:.1f}')

        # Calculate steps
        if restarted:
            total_completed = self._restart_n_equil_completed + self._restart_n_prod_completed
            if self.n_equilibration_steps is not None and self.n_production_steps is not None:
                total_target = self.n_equilibration_steps + self.n_production_steps
            else:
                total_target = N
            steps_to_run = total_target - total_completed
            if steps_to_run <= 0:
                print(f"--- Already completed {total_completed} steps. Nothing to do. ---")
                return
            print(f"--- Continuing: {steps_to_run} steps remaining ({total_completed}/{total_target}) ---")
            iteration_offset = self._restart_iteration + 1 if self._restart_iteration >= 0 else 0
        else:
            steps_to_run = N
            iteration_offset = 0

        probs = self.move_probabilities

        for iteration in range(steps_to_run):
            success = False
            switch = np.random.rand()

            if switch < probs['translation']:
                self.moves['translation']['attempted'] += 1
                success = self._move_translation()

            elif switch < probs['rotation']:
                self.moves['rotation']['attempted'] += 1
                success = self._move_rotation()

            elif switch < probs['volume']:
                self.moves['volume']['attempted'] += 1
                success = self._move_volume()

            else:
                self.moves['swap']['attempted'] += 1
                success = self._move_swap()

            current_step = iteration + 1 + iteration_offset

            if success:
                self._log_step_binary(current_step)
                self._save_restart(current_step)
                N1_list.append(self.N1)
                N2_list.append(self.N2)
                V1_list.append(self.V1)
                V2_list.append(self.V2)
                E1_list.append(float(self.E1))
                E2_list.append(float(self.E2))

            if current_step % self.checkpoint_interval == 0:
                self._save_history_checkpoint(current_step)

            if current_step % self.trajectory_interval == 0 and self.write_trajectory:
                self._write_trajectory_xyz()

        print("--- Simulation finished. Performing final save. ---")
        total_steps_completed = iteration_offset + steps_to_run
        self._save_restart(total_steps_completed)

        if total_steps_completed % self.checkpoint_interval != 0:
            self._save_history_checkpoint(total_steps_completed)

        self._save_results_json(N1_list, N2_list, V1_list, V2_list, E1_list, E2_list)
        self._print_statistics()
