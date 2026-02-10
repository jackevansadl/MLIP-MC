import os
import json
import struct
import numpy as np
from pathlib import Path
from typing import Optional, Tuple

from ase import Atoms
from ase.io import read, write
from ase import units as ase_units
from .utilities import random_position, vdw_overlap


class MLP_Widom:
    """
    Widom insertion method for calculating Henry's constants and adsorption energies.
    
    This class performs Widom insertion trials to estimate the excess chemical
    potential and related thermodynamic properties.
    """
    def __init__(self, model, atoms_frame, atoms_ads, T, device, vdw_radii, output_dir='results'):
        self.model = model
        self.atoms_frame = atoms_frame
        self.n_frame = len(self.atoms_frame)
        self.atoms_ads = atoms_ads
        self.n_ads = len(self.atoms_ads)
        self.cell = np.array(self.atoms_frame.get_cell())
        self.V = np.linalg.det(self.cell)  # Volume in A^3
        self.T = T 
        self.device = device
        self.boltzmann = ase_units.kB
        self.beta = 1 / (self.boltzmann * T)
        
        self.vdw = vdw_radii - 0.35
        self.output_dir = output_dir
        
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        # Binary log file for storing adsorption energies
        self.log_file_path = Path(self.output_dir) / "log_widom.bin"
            
        self.stats = {
            'attempts': 0,
            'valid_insertions': 0,  # No VDW overlap
            'vdw_overlaps': 0
        }
        
        # Restart tracking
        self._restart_trials_completed = 0
        self._n_trials_target = None  # Target number of trials (set in run())

    def _print_statistics(self, e_adsorptions):
        """Prints the Widom statistics and calculated properties."""
        print("\n--- Widom Statistics ---")
        print(f"Total Attempts: {self.stats['attempts']}")
        print(f"Valid Insertions (No Overlap): {self.stats['valid_insertions']}")
        print(f"VDW Overlaps: {self.stats['vdw_overlaps']}")
        
        if len(e_adsorptions) > 0:
            # Boltzmann factor: exp(-beta * delta_E)
            boltzmann_factors = np.exp(-self.beta * np.array(e_adsorptions))
            avg_bf = np.mean(boltzmann_factors)
            
            # Henry's constant (simplified proportional representation)
            # K_H = beta * < exp(-beta * U) >
            henry_constant = self.beta * avg_bf
            
            # Weighted average energy (using Boltzmann factors as weights)
            # <U> = < U * exp(-beta * U) > / < exp(-beta * U) >
            weighted_E = (np.sum(np.array(e_adsorptions) * boltzmann_factors) /
                          np.sum(boltzmann_factors))
            
            print(f"Average Boltzmann Factor: {avg_bf:.5e}")
            print(f"Calculated Henry's Constant (approx): {henry_constant:.5e}")
            print(f"Widom insertion heat of adsorption: {weighted_E:.5f} eV")
            print(f"Arithmetic Avg Adsorption Energy: {np.mean(e_adsorptions):.5f} eV")
        else:
            print("No valid insertions found.")
        print("--------------------------\n")

    def _get_restart_paths(self) -> Tuple[str, str]:
        """Get restart file paths."""
        restart_dir = os.path.join(self.output_dir, 'restart')
        restart_xyz = os.path.join(restart_dir, 'restart_widom.xyz')
        restart_json = os.path.join(restart_dir, 'restart_widom.json')
        return restart_xyz, restart_json

    def _log_energy_binary(self, trial: int, adsorption_energy: float, total_energy: float, atoms_trial: Atoms) -> None:
        """Append an adsorption energy and trajectory record to the binary log file.
        Only called for valid insertions (no VDW overlap).
        
        Parameters
        ----------
        trial : int
            Trial number
        adsorption_energy : float
            Adsorption energy (interaction energy) in eV
        total_energy : float
            Total energy of the system in eV
        atoms_trial : Atoms
            Atomic structure (framework + adsorbate) to save
        """
        n_atoms = len(atoms_trial)
        numbers = atoms_trial.numbers
        positions = atoms_trial.get_positions().flatten()
        cell = atoms_trial.get_cell().flatten()
        
        with open(self.log_file_path, "ab") as f:
            # Write header: trial, adsorption_energy, total_energy, n_atoms
            f.write(struct.pack("iddi", trial, float(adsorption_energy), float(total_energy), n_atoms))
            # Write atomic numbers
            f.write(struct.pack(f"{n_atoms}i", *numbers))
            # Write positions (n_atoms * 3 doubles)
            f.write(struct.pack(f"{n_atoms * 3}d", *positions))
            # Write cell (3x3 = 9 doubles)
            f.write(struct.pack("9d", *cell))

    def _read_energies_from_log(self) -> list:
        """Read all adsorption energies from the binary log file.
        Note: This only reads energies, skipping trajectory data for efficiency.
        Use _read_full_log() if you need trajectory data.
        
        Returns
        -------
        list
            List of adsorption energies (interaction energies)
        """
        if not os.path.exists(self.log_file_path):
            return []
        
        energies = []
        try:
            with open(self.log_file_path, "rb") as f:
                while True:
                    # Read header: trial, adsorption_energy, total_energy, n_atoms
                    header_bytes = f.read(struct.calcsize("iddi"))
                    if len(header_bytes) < struct.calcsize("iddi"):
                        break
                    trial, adsorption_energy, total_energy, n_atoms = struct.unpack("iddi", header_bytes)
                    energies.append(adsorption_energy)
                    
                    # Skip trajectory data: numbers, positions, cell
                    # numbers: n_atoms * int (4 bytes each)
                    # positions: n_atoms * 3 * double (8 bytes each)
                    # cell: 9 * double (8 bytes each)
                    skip_bytes = n_atoms * 4 + n_atoms * 3 * 8 + 9 * 8
                    f.seek(skip_bytes, 1)  # Seek forward from current position
        except Exception as e:
            print(f"Warning: Error reading binary log: {e}")
            return []
        
        return energies

    def _save_restart(self, trials_completed: int) -> None:
        """Save restart info to restart directory (overwrites previous restart info).
        This is called every step for crash recovery.
        Note: Adsorption energies are stored in binary log, not in restart JSON.
        
        Parameters
        ----------
        trials_completed : int
            Number of trials completed
        """
        # Create restart directory
        restart_dir = Path(self.output_dir) / 'restart'
        restart_dir.mkdir(parents=True, exist_ok=True)
        
        # Save a snapshot of the framework (for reference, not critical for restart)
        atoms_clean = Atoms(
            numbers=self.atoms_frame.numbers,
            positions=self.atoms_frame.positions,
            cell=self.atoms_frame.cell,
            pbc=self.atoms_frame.pbc
        )
        atoms_clean.calc = None
        
        # Save restart files (overwrite previous)
        restart_xyz, restart_json = self._get_restart_paths()
        write(restart_xyz, atoms_clean)
        
        restart_data = {
            'temperature': self.T,
            'trials_completed': trials_completed,
            'n_trials': self._n_trials_target,
            'stats': self.stats.copy(),
            'n_frame': self.n_frame,
            'n_ads': self.n_ads
        }
        
        with open(restart_json, 'w') as f:
            json.dump(restart_data, f, indent=4)

    def _load_restart_info(self) -> Optional[list]:
        """Load the state from restart directory (automatic restart).
        Adsorption energies are loaded from the binary log file.
        
        Returns
        -------
        list or None
            List of previous adsorption energies if restart found, None otherwise
        """
        try:
            restart_dir = Path(self.output_dir) / 'restart'
            
            if not restart_dir.exists():
                print("--- No restart directory found. Starting new Widom simulation. ---")
                return None
            
            restart_xyz, restart_json = self._get_restart_paths()
            
            if os.path.exists(restart_json):
                with open(restart_json, 'r') as f:
                    restart_data = json.load(f)
                
                # Restore state
                self._restart_trials_completed = restart_data.get('trials_completed', 0)
                self.stats = restart_data.get('stats', self.stats)
                
                # Restore target number of trials if available (for display purposes)
                # Note: The new N passed to run() will override this
                saved_n_trials = restart_data.get('n_trials')
                
                # Load energies from binary log
                e_adsorptions = self._read_energies_from_log()
                
                print(f"--- Restarting Widom simulation from restart info ---")
                print(f"    Trials completed: {self._restart_trials_completed}")
                if saved_n_trials is not None:
                    print(f"    Previous target: {saved_n_trials} trials")
                print(f"    Valid insertions: {self.stats['valid_insertions']}")
                print(f"    Energies loaded from log: {len(e_adsorptions)}")
                print(f"    Continuing with remaining trials...")
                
                return e_adsorptions
            
            print("--- No valid restart files found in restart directory. Starting new simulation. ---")
            return None
        except Exception as e:
            print(f"--- Error loading restart files: {e}. Starting new simulation. ---")
            import traceback
            traceback.print_exc()
            return None

    def _save_results_json(self, e_adsorptions):
        """Saves Widom results to a JSON file."""
        
        # Calculate derived properties if possible
        results_data = {
            'temperature': self.T,
            'attempts': self.stats['attempts'],
            'valid_insertions': self.stats['valid_insertions'],
            'raw_adsorption_energies': e_adsorptions
        }
        
        if len(e_adsorptions) > 0:
            boltzmann_factors = np.exp(-self.beta * np.array(e_adsorptions))
            avg_bf = np.mean(boltzmann_factors)
            weighted_E = np.sum(np.array(e_adsorptions) * boltzmann_factors) / np.sum(boltzmann_factors)
            
            results_data['average_boltzmann_factor'] = avg_bf
            results_data['widom_adsorption_energy'] = weighted_E
            results_data['arithmetic_adsorption_energy'] = np.mean(e_adsorptions)

        filename = os.path.join(self.output_dir, "widom_results.json")
        with open(filename, 'w') as f:
            json.dump(results_data, f, indent=4)

    def run(self, N):
        """
        Runs N Widom insertion trials.
        
        If restart files exist, this will continue from where the previous run left off,
        merging the previous results with new trials.
        
        Parameters
        ----------
        N : int
            Total number of trials to run. If restarting, only the remaining trials
            will be executed.
        """
        # Store target number of trials for restart file
        self._n_trials_target = N
        
        # Try to load restart info
        e_adsorptions = self._load_restart_info()
        
        if e_adsorptions is None:
            # No restart found, start fresh
            e_adsorptions = []
            self._restart_trials_completed = 0
            self.stats = {
                'attempts': 0,
                'valid_insertions': 0,
                'vdw_overlaps': 0
            }
            # Clear binary log for new run (only if starting fresh)
            if os.path.exists(self.log_file_path):
                os.remove(self.log_file_path)
        # If restarting, binary log already exists and will be appended to
        
        # Calculate remaining trials to run
        remaining_trials = N - self._restart_trials_completed
        
        if remaining_trials <= 0:
            print(f"--- All {N} trials already completed. No additional trials needed. ---")
            # Still save final results
            self._save_results_json(e_adsorptions)
            self._print_statistics(e_adsorptions)
            return
        
        # 1. Calculate Baseline Energies (only if starting fresh or if not cached)
        # Framework Energy
        atoms_f = self.atoms_frame.copy()
        atoms_f.calc = self.model
        framework_E = atoms_f.get_potential_energy()
        
        # Isolated Adsorbate Energy (in box)
        atoms_ads_cell = self.atoms_ads.copy()
        atoms_ads_cell.calc = self.model
        atoms_ads_cell.set_cell(self.cell, scale_atoms=False)
        atoms_ads_cell.set_pbc(True)
        # Center to avoid edge effects during isolated calculation
        atoms_ads_cell.center() 
        adsorbate_E = atoms_ads_cell.get_potential_energy()

        if self._restart_trials_completed == 0:
            print(f"Baseline Energies -- Framework: {framework_E:.4f} eV, Adsorbate: {adsorbate_E:.4f} eV")
            print(f"--- Starting Widom simulation for ({N} attempts) ---")
        else:
            print(f"--- Continuing Widom simulation: {remaining_trials} remaining trials (total target: {N}) ---")

        # Track last valid insertion structure
        last_valid_atoms = None
        
        # Run the remaining trials
        for iteration in range(remaining_trials):
            self.stats['attempts'] += 1
            
            # Create trial system
            atoms_trial = self.atoms_frame + self.atoms_ads.copy()
            
            # Randomize Position & Rotation
            pos = atoms_trial.get_positions()
            # Select only the adsorbate atoms (last n_ads atoms)
            pos[-self.n_ads:] = random_position(pos[-self.n_ads:], atoms_trial.get_cell())
            atoms_trial.set_positions(pos)

            # Check VDW Overlap
            # Note: vdw_overlap expects (atoms, vdw, n_frame, n_ads, index_of_adsorbate)
            # Since Widom only ever has 1 adsorbate at index 0 (relative to adsorbates), we pass 0
            if vdw_overlap(atoms_trial, self.vdw, self.n_frame, self.n_ads, 0):
                self.stats['vdw_overlaps'] += 1
                # Overlap = Infinite energy = 0 probability.
                # Skip expensive ML calculation for overlapping configurations
            else:
                self.stats['valid_insertions'] += 1
                
                # ML Calculation
                atoms_trial.calc = self.model
                e_trial = atoms_trial.get_potential_energy()

                # Calculate Interaction Energy (Delta E)
                # E_int = E_total - E_framework - E_adsorbate
                interaction_E = e_trial # - framework_E - adsorbate_E, Note: the finetuned model used here is trained to predict interaction energies directly, so we can use e_trial as the interaction energy
                e_adsorptions.append(interaction_E)
                
                # Log to binary file (only for valid insertions)
                current_trial = self._restart_trials_completed + iteration + 1
                self._log_energy_binary(current_trial, interaction_E, e_trial, atoms_trial)
                
                # Track last valid insertion
                last_valid_atoms = atoms_trial.copy()
            
            # Update total trials completed
            current_trials = self._restart_trials_completed + iteration + 1
            
            # Save restart info every step (for crash recovery)
            # Note: energies are stored in binary log, not in restart JSON
            self._save_restart(current_trials)

        # Final save
        total_trials_completed = self._restart_trials_completed + remaining_trials
        self._save_restart(total_trials_completed)
        
        # Save last trajectory if we have one
        if last_valid_atoms is not None:
            traj_file = os.path.join(self.output_dir, 'widom_trajectory.xyz')
            atoms_clean = Atoms(
                numbers=last_valid_atoms.numbers,
                positions=last_valid_atoms.positions,
                cell=last_valid_atoms.cell,
                pbc=last_valid_atoms.pbc
            )
            atoms_clean.calc = None
            write(traj_file, atoms_clean)
        
        # Save Results
        self._save_results_json(e_adsorptions)
        self._print_statistics(e_adsorptions)