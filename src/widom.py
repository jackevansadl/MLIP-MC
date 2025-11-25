import os
import numpy as np
import json

from ase import Atoms
from ase.io import read, write
from ase import units as ase_units
from utilities import _random_rotation, random_position, vdw_overlap

class MLP_Widom():
    def __init__(self, model, atoms_frame, atoms_ads, T, device, vdw_radii):
        self.model = model
        self.atoms_frame = atoms_frame
        self.n_frame = len(self.atoms_frame)
        self.atoms_ads = atoms_ads
        self.n_ads = len(self.atoms_ads)
        self.cell = np.array(self.atoms_frame.get_cell())
        self.V = np.linalg.det(self.cell) # Volume in A^3
        self.T = T 
        self.device = device
        self.boltzmann = ase_units.kB
        self.beta = 1 / (self.boltzmann * T)
        
        self.vdw = vdw_radii - 0.35
        
        if not os.path.exists('results'):
            os.mkdir('results')
            
        self.stats = {
            'attempts': 0,
            'valid_insertions': 0, # No VDW overlap
            'vdw_overlaps': 0
        }

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
            weighted_E = np.sum(np.array(e_adsorptions) * boltzmann_factors) / np.sum(boltzmann_factors)
            
            print(f"Average Boltzmann Factor: {avg_bf:.5e}")
            print(f"Calculated Henry's Constant (approx): {henry_constant:.5e}")
            print(f"Widom insertion heat of adsorption: {weighted_E:.5f} eV")
            print(f"Arithmetic Avg Adsorption Energy: {np.mean(e_adsorptions):.5f} eV")
        else:
            print("No valid insertions found.")
        print("--------------------------\n")

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

        filename = f"results/widom_results.json"
        with open(filename, 'w') as f:
            json.dump(results_data, f, indent=4)

    def run(self, N):
        """
        Runs N Widom insertion trials.
        """
        # 1. Calculate Baseline Energies
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

        print(f"Baseline Energies -- Framework: {framework_E:.4f} eV, Adsorbate: {adsorbate_E:.4f} eV")

        e_adsorptions = []
        
        # Clean output file
        traj_filename = f'results/widom_traj.xyz'
        if os.path.exists(traj_filename):
            os.remove(traj_filename)

        print(f"--- Starting Widom simulation for ({N} attempts) ---")

        for iteration in range(N):
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
                # Overlap = Infinite energy = 0 probability. We usually don't store the energy 
                # for pure statistics unless doing specific void fraction calc, but we skip expensive ML calc.
                # print(f"Iter {iteration}: VDW Overlap")
                continue
            else:
                self.stats['valid_insertions'] += 1
                
                # ML Calculation
                atoms_trial.calc = self.model
                e_trial = atoms_trial.get_potential_energy()

                # Calculate Interaction Energy (Delta E)
                # E_int = E_total - E_framework - E_adsorbate
                interaction_E = e_trial - framework_E - adsorbate_E
                
                e_adsorptions.append(interaction_E)
                
                # for debugging
                #print(f"Iter {iteration}: Valid. E_int: {interaction_E:.4f} eV")

                # Save Trajectory of valid insertions
                # Rebuild atoms to ensure clean write
                atoms_for_writing = Atoms(
                    numbers=atoms_trial.numbers, 
                    positions=atoms_trial.positions,
                    cell=atoms_trial.cell, 
                    pbc=True
                )
                write(traj_filename, atoms_for_writing, append=True)

        # Save Results
        self._save_results_json(e_adsorptions)
        self._print_statistics(e_adsorptions)