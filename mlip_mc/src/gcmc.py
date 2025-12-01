import os
import json
import sys
import traceback
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import numpy as np

from ase import Atoms
from ase.io import read, write
from ase import units as ase_units
from ase.units import bar
from .utilities import _random_rotation, random_position, vdw_overlap


# Constants
EXP_THRESHOLD = 100.0  # Threshold for exp_value in acceptance calculations
VDW_OFFSET = 0.35  # Offset for van der Waals radii
HIGH_ENERGY = 10**10  # High energy value for rejected moves
MOVE_PROBABILITIES = {
    'insertion': 0.25,
    'deletion': 0.5,
    'translation': 0.75,
    'rotation': 1.0
}
TRANSLATION_STEP = 0.5  # Step size for translation moves
ROTATION_CIRCLEFRAC = 0.1  # Circle fraction for rotation moves


def e_interaction_of_adsorption(
    e_system: float,
    framework_E: float,
    ads_energy: float,
    n_adsorbed_species: int
) -> float:
    """
    Calculate the interaction energy of adsorption.
    
    Parameters
    ----------
    e_system : float
        Total energy of the system (framework + adsorbates)
    framework_E : float
        Energy of the empty framework
    ads_energy : float
        Energy of a single isolated adsorbate
    n_adsorbed_species : int
        Number of adsorbed species
        
    Returns
    -------
    float
        Interaction energy
    """
    return e_system - framework_E - n_adsorbed_species * ads_energy

class MLP_GCMC:
    """
    Grand Canonical Monte Carlo simulation class for adsorption studies.
    
    This class performs GCMC simulations using machine-learned interatomic
    potentials to study gas adsorption in porous materials.
    """
    
    def __init__(self, model, atoms_frame, atoms_ads, T, P, fugacity, device, vdw_radii, debug=False, output_dir='results', restart_prefix=None, n_equilibration_steps=None, n_production_steps=None):
        # Note: restart_prefix is kept for backward compatibility but not used
        # Checkpoints and results are now saved automatically every step
        # n_equilibration_steps and n_production_steps: used to track equilibration vs production phase in checkpoints
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
        self.insertion_rejected_due_to_vdw = 0
        self.insertion_rejected_due_to_acceptance = 0
        self.insertion_accepted_due_to_acceptance_100 = 0
        self.insertion_rejected_due_to_acceptance_100 = 0

        self.deletion_rejected_due_to_acceptance = 0
        self.deletion_accepted_due_to_acceptance_100 = 0
        self.deletion_rejected_due_to_acceptance_100 = 0

        self.vdw = vdw_radii - VDW_OFFSET
        self.debug = debug
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)
        
        self.restart_prefix = restart_prefix  # Kept for backward compatibility (not used)
        self.n_equilibration_steps = n_equilibration_steps  # Total equilibration steps (target)
        self.n_production_steps = n_production_steps  # Total production steps (target)
        
        # Restart tracking (set when loading checkpoint)
        self._restart_n_equil_completed = 0
        self._restart_n_prod_completed = 0
        self._restart_iteration = -1

        self.moves = {
            'insertion': {'attempted': 0, 'accepted': 0},
            'deletion': {'attempted': 0, 'accepted': 0},
            'translation': {'attempted': 0, 'accepted': 0},
            'rotation': {'attempted': 0, 'accepted': 0}
        }

    def _debug_print(self, message: str, accepted: bool) -> None:
        """Print debug message if debug mode is enabled."""
        if self.debug:
            print(message)

    def _insertion_acceptance(self, e_trial: float, e: float) -> bool:
        """
        Calculate acceptance probability for insertion move.
        
        Parameters
        ----------
        e_trial : float
            Interaction energy after insertion
        e : float
            Interaction energy before insertion
            
        Returns
        -------
        bool
            True if move is accepted, False otherwise
        """
        exp_value = self.beta * (e - e_trial)
        delta_E = e_trial - e
        
        if exp_value > EXP_THRESHOLD:
            self._debug_print(
                f'✅ accepted insertion, delta E: {delta_E}, e: {e}, e_trial: {e_trial}, '
                f'exp_value: {exp_value} > {EXP_THRESHOLD}',
                True
            )
            self.moves['insertion']['accepted'] += 1
            self.insertion_accepted_due_to_acceptance_100 += 1
            return True
        
        if exp_value < -EXP_THRESHOLD:
            self._debug_print(
                f'❌ rejected insertion, delta E: {delta_E}, e: {e}, e_trial: {e_trial}, '
                f'exp_value: {exp_value} < -{EXP_THRESHOLD}',
                False
            )
            self.insertion_rejected_due_to_acceptance_100 += 1
            return False
        
        # Note: Z_ads is incremented before calling this function, so it's always >= 1
        acc = min(1, self.V * self.beta * self.fugacity / self.Z_ads * np.exp(exp_value))
        test_int = np.random.rand()
        accepted = test_int < acc
        
        if accepted:
            self.moves['insertion']['accepted'] += 1
            self._debug_print(
                f'✅ accepted insertion, delta E: {delta_E}, e: {e}, e_trial: {e_trial}, '
                f'acc: {acc}, exp_value: {exp_value}',
                True
            )
        else:
            self.insertion_rejected_due_to_acceptance += 1
            self._debug_print(
                f'❌ rejected insertion, delta E: {delta_E}, e: {e}, e_trial: {e_trial}, '
                f'acc: {acc}, exp_value: {exp_value}',
                False
            )
        
        return accepted

    def _deletion_acceptance(self, e_trial: float, e: float) -> bool:
        """
        Calculate acceptance probability for deletion move.
        
        Parameters
        ----------
        e_trial : float
            Interaction energy after deletion
        e : float
            Interaction energy before deletion
            
        Returns
        -------
        bool
            True if move is accepted, False otherwise
        """
        exp_value = -self.beta * (e_trial - e)
        delta_E = e_trial - e
        
        if exp_value > EXP_THRESHOLD:
            self._debug_print(
                f'✅ accepted deletion, delta E: {delta_E}, e: {e}, e_trial: {e_trial}, '
                f'exp_value: {exp_value} > {EXP_THRESHOLD}',
                True
            )
            self.moves['deletion']['accepted'] += 1
            self.deletion_accepted_due_to_acceptance_100 += 1
            return True
        
        acc = min(1, (self.Z_ads + 1) / self.V / self.beta / self.fugacity * np.exp(exp_value))
        test_int = np.random.rand()
        accepted = test_int < acc
        
        if accepted:
            self.moves['deletion']['accepted'] += 1
            self._debug_print(
                f'✅ accepted deletion, delta E: {delta_E}, e: {e}, e_trial: {e_trial}, '
                f'acc: {acc}, exp_value: {exp_value}',
                True
            )
        else:
            self.deletion_rejected_due_to_acceptance += 1
            self._debug_print(
                f'❌ rejected deletion, delta E: {delta_E}, e: {e}, e_trial: {e_trial}, '
                f'acc: {acc}, exp_value: {exp_value}',
                False
            )
        
        return accepted

    def _get_checkpoint_paths(self) -> Tuple[str, str]:
        """Get checkpoint file paths."""
        checkpoint_dir = os.path.join(self.output_dir, 'checkpoint')
        restart_xyz = os.path.join(checkpoint_dir, f'restart_{self.P/bar:.5f}bar.xyz')
        restart_json = os.path.join(checkpoint_dir, f'restart_{self.P/bar:.5f}bar.json')
        return restart_xyz, restart_json

    def _save_checkpoint(self, atoms: Atoms, steps_in_this_run: int) -> None:
        """Saves checkpoint to checkpoint directory (overwrites previous checkpoint).
        
        Parameters
        ----------
        atoms : Atoms
            Current atomic structure
        steps_in_this_run : int
            Number of steps completed in this run (1-indexed, so 1 means first step completed)
        """
        # Create checkpoint directory
        checkpoint_dir = Path(self.output_dir) / 'checkpoint'
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Clean atoms object (remove calculator)
        atoms_clean = Atoms(
            numbers=atoms.numbers,
            positions=atoms.positions,
            cell=atoms.cell,
            pbc=atoms.pbc
        )
        atoms_clean.calc = None
        
        # Save checkpoint files (overwrite previous)
        restart_xyz, restart_json = self._get_checkpoint_paths()
        write(restart_xyz, atoms_clean)
        
        # Calculate equilibration and production steps completed
        # steps_in_this_run is 1-indexed: 1 = first step completed, 2 = second step completed, etc.
        if self.n_equilibration_steps is not None:
            
            # If we restarted, we need to account for previously completed steps
            if self._restart_iteration >= 0:
                # We restarted - calculate total steps completed including previous run
                prev_equil_completed = self._restart_n_equil_completed
                prev_prod_completed = self._restart_n_prod_completed
                
                # Calculate how many equilibration steps we've done in this run
                equil_remaining = self.n_equilibration_steps - prev_equil_completed
                
                if equil_remaining > 0:
                    # Still have equilibration steps to complete
                    equil_done_in_this_run = min(steps_in_this_run, equil_remaining)
                    prod_done_in_this_run = max(0, steps_in_this_run - equil_remaining)
                else:
                    # Equilibration already complete, all new steps are production
                    equil_done_in_this_run = 0
                    prod_done_in_this_run = steps_in_this_run
                
                # Total completed = previous + new in this run
                n_equil_completed = prev_equil_completed + equil_done_in_this_run
                n_prod_completed = prev_prod_completed + prod_done_in_this_run
            else:
                # Fresh start - no previous steps
                total_steps_completed = steps_in_this_run
                if total_steps_completed <= self.n_equilibration_steps:
                    # Still in equilibration phase
                    n_equil_completed = total_steps_completed
                    n_prod_completed = 0
                else:
                    # Past equilibration, now in production phase
                    n_equil_completed = self.n_equilibration_steps
                    n_prod_completed = total_steps_completed - self.n_equilibration_steps
        else:
            n_equil_completed = None
            n_prod_completed = None
        
        restart_data = {
            'Z_ads': self.Z_ads,
            'n_equil': self.n_equilibration_steps,
            'n_equil_completed': n_equil_completed,
            'n_prod': self.n_production_steps,
            'n_prod_completed': n_prod_completed,
            'moves': self.moves,
            'rejections': {
                'insertion_vdw': self.insertion_rejected_due_to_vdw,
                'insertion_acc': self.insertion_rejected_due_to_acceptance,
                'insertion_acc_100': self.insertion_accepted_due_to_acceptance_100,
                'insertion_rej_100': self.insertion_rejected_due_to_acceptance_100,
                'deletion_acc': self.deletion_rejected_due_to_acceptance,
                'deletion_acc_100': self.deletion_accepted_due_to_acceptance_100,
                'deletion_rej_100': self.deletion_rejected_due_to_acceptance_100
            }
        }
        with open(restart_json, 'w') as f:
            json.dump(restart_data, f, indent=4)

    def _load_restart_info(self) -> Optional[Atoms]:
        """Loads the state from checkpoint directory (automatic restart)."""
        if not self.output_dir:
            return None
        
        try:
            checkpoint_dir = Path(self.output_dir) / 'checkpoint'
            
            if not checkpoint_dir.exists():
                print("--- No checkpoint directory found. Starting new simulation. ---")
                return None
            
            restart_xyz, restart_json = self._get_checkpoint_paths()
            
            if os.path.exists(restart_xyz) and os.path.exists(restart_json):
                atoms = read(restart_xyz)
                with open(restart_json, 'r') as f:
                    restart_data = json.load(f)
                
                self.Z_ads = restart_data.get('Z_ads', 0)
                self.moves = restart_data.get('moves', self.moves)
                
                rejections = restart_data.get('rejections', {})
                self.insertion_rejected_due_to_vdw = rejections.get('insertion_vdw', 0)
                self.insertion_rejected_due_to_acceptance = rejections.get('insertion_acc', 0)
                self.insertion_accepted_due_to_acceptance_100 = rejections.get('insertion_acc_100', 0)
                self.insertion_rejected_due_to_acceptance_100 = rejections.get('insertion_rej_100', 0)
                self.deletion_rejected_due_to_acceptance = rejections.get('deletion_acc', 0)
                self.deletion_accepted_due_to_acceptance_100 = rejections.get('deletion_acc_100', 0)
                self.deletion_rejected_due_to_acceptance_100 = rejections.get('deletion_rej_100', 0)

                # Get completed steps from checkpoint
                n_equil_completed = restart_data.get('n_equil_completed')
                n_prod_completed = restart_data.get('n_prod_completed')
                
                # Handle None values (when equilibration/production steps weren't set)
                if n_equil_completed is None:
                    n_equil_completed = 0
                if n_prod_completed is None:
                    n_prod_completed = 0
                
                # Use the passed n_equil and n_prod values (not from checkpoint)
                # This allows changing the target steps on restart
                # Store the starting point for continuation
                self._restart_n_equil_completed = n_equil_completed
                self._restart_n_prod_completed = n_prod_completed
                
                # Calculate total steps completed (for determining starting iteration)
                total_steps_completed = n_equil_completed + n_prod_completed
                # Set _restart_iteration to indicate we found a checkpoint
                # Use total_steps_completed - 1 (0-indexed), or 0 if no steps completed yet
                self._restart_iteration = total_steps_completed - 1 if total_steps_completed > 0 else 0
                
                if self.n_equilibration_steps is not None and self.n_production_steps is not None:
                    print(f"--- Restarting simulation from checkpoint, Z_ads = {self.Z_ads} ---")
                    print(f"    Equilibration: {n_equil_completed}/{self.n_equilibration_steps} steps completed")
                    print(f"    Production: {n_prod_completed}/{self.n_production_steps} steps completed")
                    if n_equil_completed < self.n_equilibration_steps:
                        remaining_equil = self.n_equilibration_steps - n_equil_completed
                        print(f"    Continuing with {remaining_equil} remaining equilibration steps")
                    elif n_prod_completed < self.n_production_steps:
                        remaining_prod = self.n_production_steps - n_prod_completed
                        print(f"    Continuing with {remaining_prod} remaining production steps")
                else:
                    print(f"--- Restarting simulation from checkpoint, Z_ads = {self.Z_ads} ---")
                return atoms
            
            print("--- No valid restart files found in checkpoint directory. Starting new simulation. ---")
            return None
        except Exception as e:
            print(f"--- Error loading restart files: {e}. Starting new simulation. ---", file=sys.stderr)
            traceback.print_exc()
            return None


    def _print_statistics(self):
        """Prints the final MC move statistics."""
        print("\n  ┌" + "─" * 66 + "┐")
        print("  │" + " MC Move Statistics".center(66) + "│")
        print("  ├" + "─" * 66 + "┤")
        
        # Main move statistics
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
        
        # Additional statistics
        print("  ├" + "─" * 66 + "┤")
        stats_lines = [
            ("VDW Overlap Rejections", self.insertion_rejected_due_to_vdw),
            ("Insertion Rejections", self.insertion_rejected_due_to_acceptance),
            ("Deletion Rejections", self.deletion_rejected_due_to_acceptance),
        ]
        for label, value in stats_lines:
            print(f"  │ {label:<30} {value:>33} │")
        
        print("  └" + "─" * 66 + "┘\n")

    def _save_results_json(
        self,
        uptake: List[int],
        interaction_energy: List[float],
        total_energy: List[float]
    ) -> None:
        """Saves uptake and energy data to a JSON file."""
        results_data = {
            'uptake': uptake,
            'interaction_energy': interaction_energy,
            'total_energy': total_energy
        }
        filename = Path(self.output_dir) / f"results_{self.P/bar:.5f}bar.json"
        with open(filename, 'w') as f:
            json.dump(results_data, f, indent=4)

    def run(self, N: int) -> None:
        """
        Run the GCMC simulation for N Monte Carlo steps.
        
        If restarting, this will continue from where the checkpoint left off,
        using the passed n_equilibration_steps and n_production_steps as targets.
        It will automatically calculate how many steps remain to complete the targets.
        
        Parameters
        ----------
        N : int
            Total number of Monte Carlo steps (n_equilibration_steps + n_production_steps)
            If restarting, only the remaining steps will be run.
        """
        
        # Automatically check for checkpoint and restart if available
        atoms = self._load_restart_info()
        # If _load_restart_info returns None, it means no restart files were found
        if atoms is None:
            # No restart found, start fresh
            atoms = self.atoms_frame.copy()
            self.Z_ads = 0
            self._restart_iteration = -1
            self._restart_n_equil_completed = 0
            self._restart_n_prod_completed = 0

        atoms.calc = self.model
        e = atoms.get_potential_energy()
        interaction_E = 0
        # for interaction energy, need to know enegry of initial framework and adsorbates in the cell
        atoms_ads_cell = self.atoms_ads.copy()
        atoms_ads_cell.calc = self.model
        # put adsorbate into the framework cell without scaling its internal coordinates
        atoms_ads_cell.set_cell(self.cell, scale_atoms=False)
        atoms_ads_cell.set_pbc(True)
        # place the adsorbate at a random position inside the cell
        positions = atoms_ads_cell.get_positions()
        positions = random_position(positions, self.cell)
        atoms_ads_cell.set_positions(positions)
        print(f'pbc: {atoms_ads_cell.pbc}, cell: {atoms_ads_cell.cell}')
        
        adsorbate_E = atoms_ads_cell.get_potential_energy()

        framework = self.atoms_frame.copy()
        framework.calc = self.model
        framework_E = atoms.get_potential_energy()

        print(f'framework E: {framework_E}, adsorbate_E: {adsorbate_E}')

        uptake = []
        interaction_energy = []
        total_energy = []

        # Load previous results if they exist (when restarting)
        # Check if we have a checkpoint by checking if restart attributes are set
        has_checkpoint = (hasattr(self, '_restart_n_equil_completed') and 
                         hasattr(self, '_restart_n_prod_completed'))
        results_filename = Path(self.output_dir) / f"results_{self.P/bar:.5f}bar.json"
        if has_checkpoint and results_filename.exists():
            try:
                with open(results_filename, 'r') as f:
                    data = json.load(f)
                    uptake = data.get('uptake', [])
                    interaction_energy = data.get('interaction_energy', [])
                    total_energy = data.get('total_energy', [])
            except Exception as err:
                print(f"Warning: Could not load previous results: {err}", file=sys.stderr)

        # Determine how many steps to run
        # Check if we have a checkpoint (by checking if checkpoint was loaded)
        checkpoint_loaded = (self._restart_iteration >= 0 or 
                            (hasattr(self, '_restart_n_equil_completed') and 
                             hasattr(self, '_restart_n_prod_completed')))
        
        if checkpoint_loaded:
            # We restarted - calculate remaining steps needed
            total_completed = self._restart_n_equil_completed + self._restart_n_prod_completed
            if (self.n_equilibration_steps is not None and 
                self.n_production_steps is not None):
                total_target = self.n_equilibration_steps + self.n_production_steps
            else:
                total_target = N
            steps_to_run = total_target - total_completed
            if steps_to_run <= 0:
                print(f"--- Already completed {total_completed} steps. Target is {total_target} steps. Nothing to do. ---")
                return
            print(f"--- Continuing simulation: {steps_to_run} steps remaining (already completed {total_completed}/{total_target}) ---")
            # Adjust iteration offset so that iteration 0 in this run corresponds to the continuation point
            iteration_offset = self._restart_iteration + 1 if self._restart_iteration >= 0 else 0
        else:
            # Fresh start
            steps_to_run = N
            iteration_offset = 0
        
        for iteration in range(steps_to_run):
            switch = np.random.rand()
            if switch < MOVE_PROBABILITIES['insertion']:
                self.moves['insertion']['attempted'] += 1
                self.Z_ads += 1
                atoms_trial = atoms + self.atoms_ads.copy()
                pos = atoms_trial.get_positions()
                pos[-self.n_ads:] = random_position(pos[-self.n_ads:], atoms_trial.get_cell())
                atoms_trial.set_positions(pos)

                if vdw_overlap(atoms_trial, self.vdw, self.n_frame, self.n_ads, self.Z_ads-1):
                    e_trial = HIGH_ENERGY
                    self.insertion_rejected_due_to_vdw += 1
                else:
                    atoms_trial.calc = self.model
                    e_trial = atoms_trial.get_potential_energy()

                    adsorbate = atoms_trial[-self.n_ads:]
                    adsorbate.calc = self.model
                    ads_energy = adsorbate.get_potential_energy()

                initial_int_E = e_interaction_of_adsorption(e, framework_E, adsorbate_E, self.Z_ads - 1)
                final_int_E = e_interaction_of_adsorption(e_trial, framework_E, adsorbate_E, self.Z_ads)

                if self._insertion_acceptance(final_int_E, initial_int_E):
                    atoms = atoms_trial.copy()
                    e = e_trial
                    interaction_E = final_int_E
                    # self.moves['insertion']['accepted'] += 1
                else:
                    self.Z_ads -= 1

            # Deletion
            elif switch < MOVE_PROBABILITIES['deletion']:
                self.moves['deletion']['attempted'] += 1
                if self.Z_ads > 0:
                    i_ads = np.random.randint(self.Z_ads)
                    atoms_trial = atoms.copy()
                    self.Z_ads -= 1
                    del atoms_trial[self.n_frame + self.n_ads*i_ads : self.n_frame + self.n_ads*(i_ads+1)]
                    atoms_trial.calc = self.model
                    e_trial = atoms_trial.get_potential_energy()

                    initial_int_E = e_interaction_of_adsorption(e, framework_E, adsorbate_E, self.Z_ads + 1)
                    final_int_E = e_interaction_of_adsorption(e_trial, framework_E, adsorbate_E, self.Z_ads)

                    if self._deletion_acceptance(final_int_E, initial_int_E):
                        atoms = atoms_trial.copy()
                        e = e_trial
                        interaction_E = final_int_E
                        # self.moves['deletion']['accepted'] += 1
                    else:
                        self.Z_ads += 1

            # Translation
            elif switch < MOVE_PROBABILITIES['translation']:
                self.moves['translation']['attempted'] += 1
                if self.Z_ads > 0:
                    i_ads = np.random.randint(self.Z_ads)
                    atoms_trial = atoms.copy()
                    pos = atoms_trial.get_positions()
                    pos[self.n_frame + self.n_ads*i_ads : self.n_frame + self.n_ads*(i_ads+1)] += (
                        TRANSLATION_STEP * (np.random.rand(3) - 0.5)
                    )
                    atoms_trial.set_positions(pos)
                    if vdw_overlap(atoms_trial, self.vdw, self.n_frame, self.n_ads, i_ads):
                        e_trial = HIGH_ENERGY 
                    else:
                        atoms_trial.calc = self.model
                        e_trial = atoms_trial.get_potential_energy() 
                    acc = min(1, np.exp(-self.beta*(e_trial-e)))
                    if acc > np.random.rand():
                        atoms = atoms_trial.copy()
                        interaction_E = e_trial - framework_E - self.Z_ads * adsorbate_E
                        e = e_trial
                        self.moves['translation']['accepted'] += 1

            # Rotation
            elif switch >= MOVE_PROBABILITIES['translation']:
                self.moves['rotation']['attempted'] += 1
                if self.Z_ads > 0:
                    i_ads = np.random.randint(self.Z_ads)
                    atoms_trial = atoms.copy()
                    pos = atoms_trial.get_positions()
                    start_idx = self.n_frame + self.n_ads * i_ads
                    end_idx = self.n_frame + self.n_ads * (i_ads + 1)
                    pos[start_idx:end_idx] = _random_rotation(
                        pos[start_idx:end_idx],
                        circlefrac=ROTATION_CIRCLEFRAC
                    )
                    atoms_trial.set_positions(pos)
                    if vdw_overlap(atoms_trial, self.vdw, self.n_frame, self.n_ads, i_ads):
                        e_trial = HIGH_ENERGY 
                    else:
                        atoms_trial.calc = self.model
                        e_trial = atoms_trial.get_potential_energy() 
                    acc = min(1, np.exp(-self.beta*(e_trial-e)))
                    if acc > np.random.rand():
                        atoms = atoms_trial.copy()
                        interaction_E = e_trial - framework_E - self.Z_ads * adsorbate_E
                        e = e_trial
                        self.moves['rotation']['accepted'] += 1

            uptake.append(self.Z_ads)
            interaction_energy.append(interaction_E)
            total_energy.append(e)

            # Always save results and checkpoint every step (automatic saving)
            self._save_results_json(uptake, interaction_energy, total_energy)
            # Pass steps_in_this_run (1-indexed) to _save_checkpoint
            self._save_checkpoint(atoms, iteration + 1)

            # Write trajectory
            numbers = atoms.numbers
            positions = atoms.positions
            cell = atoms.cell

            # Rebuild a clean Atoms object without any outdated arrays
            atoms_for_writing = Atoms(numbers=numbers, positions=positions,
                        cell=cell, pbc=True)

            traj_file = Path(self.output_dir) / f'traj_{self.P/bar:.5f}bar.xyz'
            write(str(traj_file), atoms_for_writing, append=True)

        print("--- Simulation finished. Performing final save. ---")
        self._save_results_json(uptake, interaction_energy, total_energy)
        
        # Save final checkpoint (already saved in the loop, but save again to ensure it's up to date)
        if steps_to_run > 0:
            # Pass steps_in_this_run (1-indexed) - this is the number of steps completed in this run
            self._save_checkpoint(atoms, steps_to_run)
        self._print_statistics()