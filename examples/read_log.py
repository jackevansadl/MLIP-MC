from mlip_mc import read_binary_log

# Read the log file
data = read_binary_log('results/log_0.10000bar.bin')

# Access the data
for record in data:
    print(f"Iteration {record['iteration']}: Uptake={record['uptake']}, "
          f"E_int={record['interaction_energy']:.4f} eV",
          f"E_total={record['total_energy']:.4f} eV")