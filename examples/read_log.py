from mlip_mc import read_binary_log

# Read the log file
data = read_binary_log('results/log_0.10000bar.bin')

# Access the data
for record in data:
    print(f"Step {record['step']}: Uptake={record['uptake']}, "
          f"E_int={record['interaction_energy']:.4f} eV, "
          f"E_total={record['total_energy']:.4f} eV")
    
    # Access the atomic structure if needed
    atoms = record['atoms']
    print(f"  Structure: {len(atoms)} atoms, formula: {atoms.get_chemical_formula()}")

# Example: Save all trajectories to a single XYZ file
from ase.io import write
from pathlib import Path

output_dir = Path('results/gcmc_trajectories')
output_dir.mkdir(exist_ok=True, parents=True)

traj_xyz = output_dir / 'gcmc_trajectory.xyz'
all_atoms = [record['atoms'] for record in data]
write(str(traj_xyz), all_atoms)

print(f"\nAll trajectories saved to: {traj_xyz} ({len(data)} structures)")