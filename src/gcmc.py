import os
import numpy as np

from ase import Atoms
from ase.io import read, write
from ase import units as ase_units
from ase.units import bar
from utilities import _random_rotation, random_position, vdw_overlap
import json

def e_intergy_of_adsorption(e_system, framework_E, ads_energy, n_adsorbed_species):
    int_e = e_system - framework_E - n_adsorbed_species * ads_energy
    return int_e

class MLP_GCMC():
    def __init__(self, model, atoms_frame, atoms_ads, T, P, fugacity, device, vdw_radii, debug=False):
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

        self.vdw = vdw_radii - 0.35
        self.debug = debug
        if not os.path.exists('results'):
            os.mkdir('results')
        self.moves = {
            'insertion': {'attempted': 0, 'accepted': 0},
            'deletion': {'attempted': 0, 'accepted': 0},
            'translation': {'attempted': 0, 'accepted': 0},
            'rotation': {'attempted': 0, 'accepted': 0}
        }
    def _insertion_acceptance(self, e_trial, e):
        exp_value = self.beta * (e - e_trial)
        if exp_value > 100:
            if self.debug:
                print(f'✅ accepted insertion, delta E: {e_trial - e}, e: {e}, e_trial: {e_trial}, exp_value: {exp_value} > 100')
            self.moves['insertion']['accepted'] += 1
            self.insertion_accepted_due_to_acceptance_100
            return True
        elif exp_value < -100:
            if self.debug:
                print(f'❌ rejected insertion, delta E: {e_trial - e}, e: {e}, e_trial: {e_trial}, exp_value: {exp_value} < -100')
            self.insertion_rejected_due_to_acceptance_100 += 1
            return False
        else:
            acc = min(1, self.V*self.beta*self.fugacity/self.Z_ads * np.exp(exp_value))
            test_int = np.random.rand()
            if acc > test_int:
                self.moves['insertion']['accepted'] += 1
                if self.debug:
                    print(f'✅ accepted insertion, delta E: {e_trial - e}, e: {e}, e_trial: {e_trial}, acc: {acc}, exp_value: {exp_value}')
            else:
                if self.debug:
                    print(f'❌ rejected insertion, delta E: {e_trial - e}, e: {e}, e_trial: {e_trial}, acc: {acc}, exp_value: {exp_value}')
                self.insertion_rejected_due_to_acceptance+=1
            return test_int < acc

    def _deletion_acceptance(self, e_trial, e):
        exp_value = -self.beta * (e_trial - e)
        if exp_value > 100:
            if self.debug:
                print(f'✅ accepted deletion, delta E: {e_trial - e}, e: {e}, e_trial: {e_trial}, exp_value: {exp_value} > 100')
            self.moves['deletion']['accepted'] += 1
            self.deletion_accepted_due_to_acceptance_100 += 1
            return True
        else:
            acc = min(1, (self.Z_ads+1)/self.V/self.beta/self.fugacity * np.exp(exp_value))
            test_int = np.random.rand()
            if acc > test_int:
                self.moves['deletion']['accepted'] += 1
                if self.debug:
                    print(f'✅ accepted deletion, delta E: {e_trial - e}, e: {e}, e_trial: {e_trial}, acc: {acc}, exp_value: {exp_value}')
            else:
                if self.debug:
                    print(f'❌ rejected deletion, delta E: {e_trial - e}, e: {e}, e_trial: {e_trial}, acc: {acc}, exp_value: {exp_value}')
                self.deletion_rejected_due_to_acceptance += 1
            return test_int < acc

    def _print_statistics(self):
        """Prints the final MC move statistics."""
        print("\n--- MC Move Statistics ---")
        for move_type, stats in self.moves.items():
            attempted = stats['attempted']
            accepted = stats['accepted']
            if attempted > 0:
                acceptance_rate = (accepted / attempted) * 100
                print(f"{move_type.capitalize():<12}: Attempted={attempted:<8} Accepted={accepted:<8} Rate={acceptance_rate:.2f}%")
            else:
                print(f"{move_type.capitalize():<12}: Not attempted.")
        print(f"Insertion rejections due to VDW overlap: {self.insertion_rejected_due_to_vdw}")
        print(f"Insertion rejections due to acceptance criteria: {self.insertion_rejected_due_to_acceptance}")
        print(f"Delection rejections due to acceptance criteria: {self.deletion_rejected_due_to_acceptance}")
        print(f"Insertion accepted due to exp_value > 100: {self.insertion_accepted_due_to_acceptance_100}")
        print(f"Insertion rejected due to exp_value < -100: {self.insertion_rejected_due_to_acceptance_100}")
        print(f"Deletion accepted due to exp_value > 100: {self.deletion_accepted_due_to_acceptance_100}")
        print(f"Deletion rejected due to exp_value < -100: {self.deletion_rejected_due_to_acceptance_100}")
        print("--------------------------\n")

    def _save_results_json(self, uptake, adsorption_energy):
        """Saves uptake and energy data to a JSON file."""
        results_data = {
            'uptake': uptake,
            'adsorption_energy': adsorption_energy
        }
        filename = f"results/results_{self.P/bar:.5f}bar.json"
        with open(filename, 'w') as f:
            json.dump(results_data, f, indent=4)

    def run(self, N):

        atoms = self.atoms_frame.copy()
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
        adsorption_energy = []
        for iteration in range(N):
            switch = np.random.rand()
            if switch < 0.25:
                self.moves['insertion']['attempted'] += 1
                self.Z_ads += 1
                atoms_trial = atoms + self.atoms_ads.copy()
                pos = atoms_trial.get_positions()
                pos[-self.n_ads:] = random_position(pos[-self.n_ads:], atoms_trial.get_cell())
                atoms_trial.set_positions(pos)

                if vdw_overlap(atoms_trial, self.vdw, self.n_frame, self.n_ads, self.Z_ads-1):
                    e_trial = 10**10
                    self.insertion_rejected_due_to_vdw += 1
                else:
                    atoms_trial.calc = self.model
                    e_trial = atoms_trial.get_potential_energy()

                    adsorbate = atoms_trial[-self.n_ads:]
                    adsorbate.calc = self.model
                    ads_energy = adsorbate.get_potential_energy()

                initial_int_E = e_intergy_of_adsorption(e, framework_E, adsorbate_E, self.Z_ads - 1)
                final_int_E = e_intergy_of_adsorption(e_trial, framework_E, adsorbate_E, self.Z_ads)

                if self._insertion_acceptance(final_int_E, initial_int_E):
                    atoms = atoms_trial.copy()
                    e = e_trial
                    interaction_E = final_int_E
                    # self.moves['insertion']['accepted'] += 1
                else:
                    self.Z_ads -= 1

            # Deletion
            elif switch < 0.5:
                self.moves['deletion']['attempted'] += 1
                if self.Z_ads != 0:
                    i_ads = np.random.randint(self.Z_ads)
                    atoms_trial = atoms.copy()
                    self.Z_ads -= 1
                    del atoms_trial[self.n_frame + self.n_ads*i_ads : self.n_frame + self.n_ads*(i_ads+1)]
                    atoms_trial.calc = self.model
                    e_trial = atoms_trial.get_potential_energy()

                    initial_int_E = e_intergy_of_adsorption(e, framework_E, adsorbate_E, self.Z_ads + 1)
                    final_int_E = e_intergy_of_adsorption(e_trial, framework_E, adsorbate_E, self.Z_ads)

                    if self._deletion_acceptance(final_int_E, initial_int_E):
                        atoms = atoms_trial.copy()
                        e = e_trial
                        interaction_E = final_int_E
                        # self.moves['deletion']['accepted'] += 1
                    else:
                        self.Z_ads += 1

            # Translation
            elif switch < 0.75:
                self.moves['translation']['attempted'] += 1
                if self.Z_ads != 0:
                    i_ads = np.random.randint(self.Z_ads)
                    atoms_trial = atoms.copy()
                    pos = atoms_trial.get_positions()
                    pos[self.n_frame + self.n_ads*i_ads : self.n_frame + self.n_ads*(i_ads+1)] += 0.5 * (np.random.rand(3) - 0.5)
                    atoms_trial.set_positions(pos)
                    if vdw_overlap(atoms_trial, self.vdw, self.n_frame, self.n_ads, i_ads):
                        e_trial = 10**10 
                    else:
                        atoms_trial.calc = self.model
                        e_trial = atoms_trial.get_potential_energy() 
                    acc = min(1, np.exp(-self.beta*(e_trial-e)))
                    if acc > np.random.rand():
                        atoms = atoms_trial.copy()
                        e = e_trial
                        self.moves['translation']['accepted'] += 1

            # Rotation
            elif switch >= 0.75:
                self.moves['rotation']['attempted'] += 1
                if self.Z_ads != 0:
                    i_ads = np.random.randint(self.Z_ads)
                    atoms_trial = atoms.copy()
                    pos = atoms_trial.get_positions()
                    pos[self.n_frame + self.n_ads*i_ads : self.n_frame + self.n_ads*(i_ads+1)] = _random_rotation(pos[self.n_frame + self.n_ads*i_ads : self.n_frame + self.n_ads*(i_ads+1)], circlefrac = 0.1)
                    atoms_trial.set_positions(pos)
                    if vdw_overlap(atoms_trial, self.vdw, self.n_frame, self.n_ads, i_ads):
                        e_trial = 10**10 
                    else:
                        atoms_trial.calc = self.model
                        e_trial = atoms_trial.get_potential_energy() 
                    acc = min(1, np.exp(-self.beta*(e_trial-e)))
                    if acc > np.random.rand():
                        atoms = atoms_trial.copy()
                        e = e_trial
                        self.moves['rotation']['accepted'] += 1

            uptake.append(self.Z_ads)
            adsorption_energy.append(interaction_E)

            # if iteration % 10000 == 0:
            #     write('results/snapshot_%.5fbar_iteration_%d.xyz'%(self.P/bar, iteration), atoms)

            # write traj

            from ase import Atoms

            # Suppose 'mof' is your ASE object whose atom count may have changed
            numbers = atoms.numbers
            positions = atoms.positions
            cell = atoms.cell

            # ✅ rebuild a clean Atoms object without any outdated arrays
            atoms_for_writing = Atoms(numbers=numbers, positions=positions,
                        cell=cell, pbc=True)

            write(f'results/traj_{self.P/bar:.5f}bar.xyz',
                atoms_for_writing, append=True)

        # np.save('results/uptake_%.5fbar.npy'%(self.P/bar), np.array(uptake))
        # np.save('results/adsorption_energy_%.5fbar.npy'%(self.P/bar), np.array(adsorption_energy))
        self._save_results_json(uptake, adsorption_energy)


        self._print_statistics()