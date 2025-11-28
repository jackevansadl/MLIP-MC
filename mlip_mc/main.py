#!/usr/bin/env python3
"""
MLIP-MC: Monte Carlo Simulations with Machine-Learned Interatomic Potentials

Main entry point for running GCMC isotherm simulations.
"""

import os
import sys
import json
from pathlib import Path
import numpy as np
import multiprocessing as mp
from multiprocessing import Process, Queue
import argparse

# Package imports - no need to modify path

from ase import Atoms
from ase.io import read
from ase.build import molecule
from ase.units import bar
from ase.data import vdw_radii

# NOTE: We delay import of torch, fairchem, and other heavy libraries 
# until inside the process to ensure CUDA environment variables take effect first.


DEFAULT_HF_REPO = "fengxuyoung/MLIP-MC"
DEFAULT_HF_FILENAME = "model.pt"
DEFAULT_LOCAL_MODEL = "models/model.pt"
# Cache root directory: use MLIP_MC_CACHE env var if set, otherwise ~/.cache/mlip-mc
_cache_dir = os.environ.get("MLIP_MC_CACHE")
if _cache_dir:
    MODEL_CACHE_ROOT = Path(_cache_dir).expanduser()
else:
    MODEL_CACHE_ROOT = Path.home() / ".cache" / "mlip-mc"


def download_model_from_huggingface(model_path, repo_id=DEFAULT_HF_REPO, filename=None, token=None):
    """
    Download model from Hugging Face Hub if it doesn't exist locally.
    
    Parameters
    ----------
    model_path : str
        Local path where the model should be saved
    repo_id : str
        Hugging Face repository ID (e.g., "author/project")
    filename : str, optional
        Filename in the Hugging Face repository (default: "model.pt")
    token : str, optional
        Hugging Face token. If None, will try to use cached token or prompt for login.
    
    Returns
    -------
    str
        Path to the downloaded model file
    """
    # If model already exists locally, return the path
    if os.path.exists(model_path):
        return model_path
    
    # Set default filename if not provided
    if filename is None:
        filename = DEFAULT_HF_FILENAME
    
    # Always use cache directory for downloads to ensure consistent caching
    # as documented: ~/.cache/mlip-mc/<repo>/<filename>
    cache_path = MODEL_CACHE_ROOT / repo_id / filename
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    target_path = str(cache_path)
    
    try:
        from huggingface_hub import hf_hub_download, login
        
        # Print statements removed - handled by caller
        
        # Try to download (will use cached token if available)
        try:
            # Download to Hugging Face cache first
            downloaded_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                token=token,
                local_dir=None  # Download to HF cache first
            )
            
            # Copy to our target location (cache directory or user-specified path)
            if downloaded_path != target_path:
                import shutil
                # Ensure target directory exists
                target_dir = os.path.dirname(target_path)
                if target_dir and not os.path.exists(target_dir):
                    os.makedirs(target_dir, exist_ok=True)
                # Copy to the target location
                shutil.copy2(downloaded_path, target_path)
                return target_path
            else:
                return downloaded_path
            
        except Exception as e:
            # If download fails due to authentication, try to login
            if "authentication" in str(e).lower() or "token" in str(e).lower():
                print("\n  Authentication required for Hugging Face Hub")
                print("  Options:")
                print("    1. Run: huggingface_hub.login()")
                print("    2. Set token: export HF_TOKEN=your_token")
                print()
                if token:
                    login(token=token)
                    # Retry download
                    downloaded_path = hf_hub_download(
                        repo_id=repo_id,
                        filename=filename,
                        token=token,
                        local_dir=None
                    )
                    # Copy to target location if different
                    if downloaded_path != target_path:
                        import shutil
                        target_dir = os.path.dirname(target_path)
                        if target_dir and not os.path.exists(target_dir):
                            os.makedirs(target_dir, exist_ok=True)
                        shutil.copy2(downloaded_path, target_path)
                        return target_path
                    return downloaded_path
            raise
    
    except ImportError:
        raise ImportError(
            "huggingface_hub is required to download models from Hugging Face.\n"
            "Install it with: pip install huggingface_hub"
        )


HF_PREFIX = "hf://"


def _print_banner():
    """Print ASCII art banner for MLIP-MC."""
    banner_lines = [
        "███    ███ ██      ██ ██████        ███    ███  ██████ ",
        "████  ████ ██      ██ ██   ██       ████  ████ ██      ",
        "██ ████ ██ ██      ██ ██████  █████ ██ ████ ██ ██      ",
        "██  ██  ██ ██      ██ ██            ██  ██  ██ ██      ",
        "██      ██ ███████ ██ ██            ██      ██  ██████ ",
        "",
        "Monte Carlo Simulations with Machine-Learned Interatomic Potentials"
    ]
    
    # Find the maximum width (strip trailing spaces for accurate measurement)
    max_width = max(len(line.rstrip()) for line in banner_lines if line.strip())
    
    # Create border
    border_top = "╔" + "═" * (max_width + 2) + "╗"
    border_bottom = "╚" + "═" * (max_width + 2) + "╝"
    
    # Print banner with borders
    print(border_top)
    for line in banner_lines:
        if line.strip():
            # Left-align the line within the border
            line_clean = line.rstrip()
            padding = max_width - len(line_clean)
            print(f"║ {line_clean}{' ' * padding} ║")
        else:
            print(f"║{' ' * (max_width + 2)}║")
    print(border_bottom)


def _print_section_header(title, width=70):
    """Print a formatted section header."""
    print()
    print("╔" + "═" * (width - 2) + "╗")
    print("║" + title.center(width - 2) + "║")
    print("╚" + "═" * (width - 2) + "╝")
    print()


def _print_subsection(title, width=70):
    """Print a formatted subsection header."""
    print()
    print("─" * width)
    print(f"  {title}")
    print("─" * width)


def _print_info(label, value, indent=2):
    """Print a formatted info line."""
    print(f"{' ' * indent}{label:<25} {value}")


def _print_warning(message):
    """Print a formatted warning message."""
    print(f"\n{'!' * 70}")
    print(f"  WARNING: {message}")
    print(f"{'!' * 70}\n")


def _print_success(message):
    """Print a formatted success message."""
    print(f"  [OK] {message}")


def _print_table(headers, rows, width=70):
    """Print a formatted table."""
    # Calculate column widths
    num_cols = len(headers)
    col_widths = [len(str(h)) for h in headers]
    for row in rows:
        for i, val in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(val)))
    
    # Add padding
    col_widths = [w + 2 for w in col_widths]
    total_width = sum(col_widths) + num_cols + 1
    
    # Print header
    header_line = "│ " + " │ ".join(str(h).ljust(col_widths[i] - 2) for i, h in enumerate(headers)) + " │"
    print("┌" + "─" * (total_width - 2) + "┐")
    print(header_line)
    print("├" + "─" * (total_width - 2) + "┤")
    
    # Print rows
    for row in rows:
        row_line = "│ " + " │ ".join(str(val).ljust(col_widths[i] - 2) for i, val in enumerate(row)) + " │"
        print(row_line)
    
    print("└" + "─" * (total_width - 2) + "┘")


def _resolve_model_spec(model_spec: str):
    """
    Normalize the model argument into a local path plus (optional) Hugging Face filename.

    Returns
    -------
    tuple[str, str, str]
        (local_path, repo_id, hf_filename)
    """
    if not model_spec:
        model_spec = DEFAULT_LOCAL_MODEL

    model_spec = model_spec.strip()

    if not model_spec.startswith(HF_PREFIX):
        repo_candidate = model_spec.split(":", 1)[0]
        repo_candidate = repo_candidate.strip()
        if (
            "/" in repo_candidate
            and not os.path.splitext(repo_candidate)[1]
        ):
            model_spec = f"{HF_PREFIX}{model_spec}"

    if model_spec.startswith(HF_PREFIX):
        spec = model_spec[len(HF_PREFIX):].strip()
        if not spec:
            spec = DEFAULT_HF_REPO

        if ":" in spec:
            repo_part, file_part = spec.split(":", 1)
            repo_id = repo_part.strip() or DEFAULT_HF_REPO
            hf_filename = file_part.strip() or DEFAULT_HF_FILENAME
        else:
            repo_id = spec
            hf_filename = DEFAULT_HF_FILENAME

        hf_filename = hf_filename.lstrip("/")
        repo_id = repo_id.strip().strip("/")
        local_path = MODEL_CACHE_ROOT / repo_id / hf_filename
        return str(local_path), repo_id, hf_filename

    basename = os.path.basename(model_spec)
    hf_filename = basename if basename else DEFAULT_HF_FILENAME
    return model_spec, DEFAULT_HF_REPO, hf_filename


def run_single_pressure(P_bar, T, model_path, atoms_frame, atoms_ads, n_equilibration_steps, 
                        n_production_steps, n_total_steps, gpu_id, result_queue, 
                        adsorbate_name=None, output_dir='results', save_interval=1000, restart=False):
    """
    Run GCMC simulation for a single pressure point on a specific GPU.
    
    Parameters
    ----------
    P_bar : float
        Pressure in bar
    T : float
        Temperature in Kelvin
    model_path : str
        Path to the model file
    atoms_frame : Atoms
        Framework structure
    atoms_ads : Atoms
        Adsorbate molecule
    n_equilibration_steps : int
        Number of equilibration steps
    n_production_steps : int
        Number of production steps
    n_total_steps : int
        Total number of steps
    gpu_id : int or str
        GPU device ID to use (int for GPU, 'cpu' for CPU)
    result_queue : Queue
        Queue to return results
    output_dir : str, optional
        Directory to save results (default: 'results')
    save_interval : int, optional
        Interval for saving restart files (default: 1000)
    restart : bool, optional
        Whether to enable restart functionality (default: False)
    """
    try:
        # 1. Set Environment Variables for GPU Isolation
        if isinstance(gpu_id, int):
            os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        
        # 2. Import libraries AFTER setting environment variables
        import torch
        from fairchem.core import FAIRChemCalculator
        from fairchem.core.units.mlip_unit import load_predict_unit
        from mlip_mc.src.gcmc import MLP_GCMC
        from mlip_mc.src.utilities import PREOS
        
        # 3. Determine device (now restricted to just the one visible GPU)
        if isinstance(gpu_id, int) and torch.cuda.is_available():
            # Even though we requested gpu_id, inside this process it will appear as cuda:0
            # because we masked the others with CUDA_VISIBLE_DEVICES
            # IMPORTANT: fairchem strictly requires 'cuda' or 'cpu', not 'cuda:0'
            device = 'cuda' 
        else:
            device = 'cpu'
        
        # Load model on this GPU
        predictor = load_predict_unit(model_path, device=device)
        model = FAIRChemCalculator(predictor, task_name="odac")
        
        P = P_bar * bar
        device_str = f"GPU {gpu_id}" if isinstance(gpu_id, int) else "CPU"
        print(f"  [{device_str}] Starting simulation at P = {P_bar:.2f} bar")
        
        # Calculate Fugacity using Peng-Robinson EOS
        if adsorbate_name:
            try:
                eos = PREOS.from_name(adsorbate_name)
                fugacity = eos.calculate_fugacity(T, P)
                print(f"  [{device_str}] Calculated Fugacity: {fugacity/bar:.4f} bar")
            except Exception as e:
                print(f"  [{device_str}] Warning: Error calculating fugacity: {e}")
                fugacity = P  # Fallback for ideal gas
        else:
            fugacity = P  # Use ideal gas approximation

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Clean trajectory file if it exists AND we are not restarting
        traj_file = os.path.join(output_dir, f'traj_{P/bar:.5f}bar.xyz')
        if os.path.exists(traj_file) and not restart:
            os.remove(traj_file)

        # Determine restart prefix
        restart_prefix = None
        if restart:
            restart_prefix = os.path.join(output_dir, f"restart_{P/bar:.5f}bar")

        # Run GCMC Simulation
        gcmc = MLP_GCMC(
            model=model,
            atoms_frame=atoms_frame,
            atoms_ads=atoms_ads,
            T=T,
            P=P,
            fugacity=fugacity,
            device=device,
            vdw_radii=vdw_radii,
            debug=False,
            output_dir=output_dir,
            restart_prefix=restart_prefix,
            save_interval=save_interval
        )
        
        print(f"  [{device_str}] Running {n_total_steps} steps...")
        gcmc.run(N=n_total_steps)
        
        # Load and process results
        results_file = os.path.join(output_dir, f"results_{P/bar:.5f}bar.json")
        if os.path.exists(results_file):
            with open(results_file, 'r') as f:
                results = json.load(f)
            
            uptake_data = results['uptake'][n_equilibration_steps:]
            energy_data = results['interaction_energy'][n_equilibration_steps:]
            
            avg_uptake = np.mean(uptake_data)
            std_uptake = np.std(uptake_data)
            avg_energy = np.mean([e for e in energy_data if e != 0]) if any(e != 0 for e in energy_data) else 0.0
            
            device_str = f"GPU {gpu_id}" if isinstance(gpu_id, int) else "CPU"
            print(f"  [{device_str}] Completed P = {P_bar:.2f} bar: Uptake = {avg_uptake:.3f} ± {std_uptake:.3f}")
            
            # Return results via queue
            result_queue.put({
                'pressure': P_bar,
                'uptake': avg_uptake,
                'uptake_std': std_uptake,
                'energy': avg_energy
            })
        else:
            print(f"[GPU {gpu_id}] Warning: Results file not found: {results_file}")
            result_queue.put({
                'pressure': P_bar,
                'uptake': None,
                'uptake_std': None,
                'energy': None
            })
    except Exception as e:
        device_str = f"GPU {gpu_id}" if isinstance(gpu_id, int) else "CPU"
        print(f"  [{device_str}] ERROR in simulation at P = {P_bar:.2f} bar: {e}")
        import traceback
        traceback.print_exc()
        result_queue.put({
            'pressure': P_bar,
            'uptake': None,
            'uptake_std': None,
            'energy': None,
            'error': str(e)
        })


def run_gcmc(
    adsorbent_path,
    adsorbate_path=None,
    adsorbate_molecule=None,
    temperature=298.0,
    pressure_points=None,
    n_equilibration_steps=10000,
    n_production_steps=20000,
    model_path="models/model.pt",
    output_dir='results',
    plot_isotherm=True,
    adsorbate_label=None,
    hf_token=None,
    save_interval=1000,
    restart=False
):
    """
    Run GCMC isotherm simulation for gas adsorption in a porous material.
    
    Parameters
    ----------
    adsorbent_path : str
        Path to adsorbent (framework) structure file (.xyz, .cif, or other ASE-readable format)
    adsorbate_path : str, optional
        Path to adsorbate molecule structure file (.xyz, .cif, etc.)
        If None, adsorbate_molecule must be provided
    adsorbate_molecule : str, optional
        Name of molecule to build using ASE (e.g., 'CO2', 'CH4', 'H2O')
        Used only if adsorbate_path is None
    temperature : float, optional
        Temperature in Kelvin (default: 298.0)
    pressure_points : float or list of float
        Pressure point(s) in bar. If single number, runs one simulation.
        If list, distributes across available GPUs. Required parameter.
    n_equilibration_steps : int, optional
        Number of Monte Carlo steps for equilibration (default: 10000)
    n_production_steps : int, optional
        Number of Monte Carlo steps for production/data collection (default: 20000)
    model_path : str, optional
        Path to MLIP model file (default: "models/model.pt").
        Can be a local path or a Hugging Face repository name (e.g., "fengxuyoung/MLIP-MC").
        Missing files are automatically downloaded from Hugging Face and cached.
    output_dir : str, optional
        Directory to save results (default: 'results')
    plot_isotherm : bool, optional
        Whether to generate and save isotherm plot (default: True)
    adsorbate_label : str, optional
        Label for adsorbate in plot title (default: inferred from file/molecule name)
    hf_token : str, optional
        Hugging Face authentication token. If None, uses cached token or prompts for login
    save_interval : int, optional
        Interval for saving restart files (default: 1000)
    restart : bool, optional
        Whether to enable restart functionality (default: False)
    
    Returns
    -------
    dict
        Dictionary containing isotherm data:
        - 'pressures': list of pressures (bar)
        - 'uptakes': list of average uptakes (molecules/unit cell)
        - 'uptake_stds': list of standard deviations
        - 'adsorption_energies': list of average adsorption energies (eV)
        - 'temperature': temperature (K)
        - 'unit_cell_volume_A3': unit cell volume (A^3)
        - 'unit_cell_volume_cm3': unit cell volume (cm^3)
    """
    _print_banner()
    _print_section_header("GCMC Isotherm Simulation")
    
    # Set multiprocessing start method to 'spawn' for CUDA compatibility
    # This must be done before creating any processes
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        # Context might already be set, try without force
        try:
            mp.set_start_method('spawn')
        except RuntimeError:
            pass  # Already set, continue
    
    # Helper import for device counting in main process
    import torch
    import matplotlib.pyplot as plt

    # Check available GPUs
    _print_subsection("Hardware Configuration")
    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
        _print_info("GPUs Available", f"{n_gpus}")
        for i in range(n_gpus):
            _print_info(f"  GPU {i}", torch.cuda.get_device_name(i))
    else:
        n_gpus = 0
        _print_warning("No GPUs available, using CPU")
    
    model_path, hf_repo_id, hf_model_filename = _resolve_model_spec(model_path)

    # 1. Load model (download from Hugging Face if not found locally)
    _print_subsection("Model Configuration")
    
    # Check cache directory first (for consistent caching across working directories)
    cache_path = MODEL_CACHE_ROOT / hf_repo_id / hf_model_filename
    if os.path.exists(cache_path):
        model_path = str(cache_path)
        _print_info("Model Status", f"Found in cache at {model_path}")
    elif not os.path.exists(model_path):
        _print_info("Model Status", f"Not found at {model_path}")
        _print_info("Action", "Downloading from Hugging Face Hub...")
        # Try to download from Hugging Face
        try:
            model_path = download_model_from_huggingface(
                model_path,
                repo_id=hf_repo_id,
                filename=hf_model_filename,
                token=hf_token
            )
            _print_success(f"Model downloaded successfully to: {model_path}")
        except Exception as e:
            raise FileNotFoundError(
                f"Model file not found at {model_path}\n"
                f"Failed to download from Hugging Face: {e}\n"
                "Please ensure the model file exists or check your Hugging Face authentication."
            )
    else:
        _print_info("Model Status", f"Found at {model_path}")
    
    # 2. Load adsorbent (framework) structure
    _print_subsection("Structure Loading")
    if not os.path.exists(adsorbent_path):
        raise FileNotFoundError(f"Adsorbent file not found: {adsorbent_path}")
    
    _print_info("Adsorbent", f"Loading from {adsorbent_path}...")
    atoms_frame = read(adsorbent_path)
    
    # Ensure clean Atoms object (no calculator attached) for multiprocessing
    atoms_frame = Atoms(
        numbers=atoms_frame.numbers,
        positions=atoms_frame.positions,
        cell=atoms_frame.cell,
        pbc=atoms_frame.pbc
    )
    
    # 3. Load or create adsorbate
    if adsorbate_path is not None:
        if not os.path.exists(adsorbate_path):
            raise FileNotFoundError(f"Adsorbate file not found: {adsorbate_path}")
        _print_info("Adsorbate", f"Loading from {adsorbate_path}...")
        atoms_ads = read(adsorbate_path)
        atoms_ads = Atoms(numbers=atoms_ads.numbers, positions=atoms_ads.positions)
        if adsorbate_label is None:
            adsorbate_label = os.path.basename(adsorbate_path).replace('.xyz', '').replace('.cif', '')
        # Extract chemical formula from the loaded structure to match with fugacity table
        # But if adsorbate_molecule is also provided, prioritize that for fugacity matching
        if adsorbate_molecule is not None:
            # Both file and molecule provided: use molecule name for fugacity matching
            adsorbate_name = adsorbate_molecule
            _print_info("Adsorbate Name", adsorbate_name)
        else:
            # Only file provided: extract formula from file
            adsorbate_name = atoms_ads.get_chemical_formula()
            # Normalize: remove spaces to match CSV format (e.g., "C O2" -> "CO2")
            adsorbate_name = adsorbate_name.replace(' ', '')
            _print_info("Adsorbate Name", f"{adsorbate_name} (extracted from file)")
    elif adsorbate_molecule is not None:
        _print_info("Adsorbate", f"Creating {adsorbate_molecule} molecule...")
        atoms_ads = molecule(adsorbate_molecule)
        atoms_ads = Atoms(numbers=atoms_ads.numbers, positions=atoms_ads.positions)
        if adsorbate_label is None:
            adsorbate_label = adsorbate_molecule
        # Use the molecule name directly to match with fugacity table
        adsorbate_name = adsorbate_molecule
        _print_info("Adsorbate Name", adsorbate_name)
    else:
        raise ValueError("Either adsorbate_path or adsorbate_molecule must be provided")
    
    # 4. Process pressure points
    if pressure_points is None:
        raise ValueError("pressure_points must be provided (float or list of floats)")
    
    if isinstance(pressure_points, (int, float)):
        pressure_points = [float(pressure_points)]
    else:
        pressure_points = [float(p) for p in pressure_points]
    
    _print_subsection("Simulation Parameters")
    _print_info("Temperature", f"{temperature} K")
    _print_info("Pressure Points", f"{pressure_points} bar")
    _print_info("Equilibration Steps", f"{n_equilibration_steps}")
    _print_info("Production Steps", f"{n_production_steps}")
    
    # Calculate unit cell volume
    cell_volume = np.linalg.det(atoms_frame.get_cell())  # A^3
    cell_volume_cm3 = cell_volume * 1e-24  # Convert to cm^3
    
    # 5. Distribute pressure points across GPUs
    _print_subsection("Simulation Execution")
    if n_gpus == 0:
        _print_warning("Running sequentially on CPU (no GPUs available)")
        devices = ['cpu']
    else:
        devices = list(range(n_gpus))
        active_gpus = min(len(pressure_points), n_gpus)
        _print_info("Distribution", f"{len(pressure_points)} pressure point(s) across {active_gpus} of {n_gpus} GPU(s)")
    
    # Create processes for parallel execution
    processes = []
    result_queue = Queue()
    n_total_steps = n_equilibration_steps + n_production_steps
    
    # Distribute pressure points across GPUs (round-robin)
    for i, P_bar in enumerate(pressure_points):
        gpu_id = devices[i % len(devices)] if isinstance(devices[0], int) else 'cpu'
        
        p = Process(target=run_single_pressure, args=(
            P_bar, temperature, model_path, atoms_frame, atoms_ads,
            n_equilibration_steps, n_production_steps, n_total_steps,
            gpu_id, result_queue, adsorbate_name, output_dir, save_interval, restart
        ))
        processes.append(p)
        p.start()
        device_str = f"GPU {gpu_id}" if isinstance(gpu_id, int) else str(gpu_id).upper()
        _print_info(f"Started", f"P = {P_bar:.2f} bar on {device_str}")
    
    # Wait for all processes to complete
    print(f"\n  Waiting for {len(processes)} simulation(s) to complete...")
    for p in processes:
        p.join()
    
    # Collect results
    print("\n  Collecting results...")
    results = []
    while not result_queue.empty():
        results.append(result_queue.get())
    
    # Sort results by pressure
    results.sort(key=lambda x: x['pressure'])
    
    # Extract data
    pressures = []
    uptakes = []
    uptake_stds = []
    adsorption_energies = []
    
    for r in results:
        if r['uptake'] is not None:
            pressures.append(r['pressure'])
            uptakes.append(r['uptake'])
            uptake_stds.append(r['uptake_std'])
            adsorption_energies.append(r['energy'])
        else:
            _print_warning(f"No data for P = {r['pressure']:.2f} bar")
            if 'error' in r:
                print(f"    Error: {r['error']}")
    
    # 6. Plot Isotherm (if requested and we have data)
    if plot_isotherm and len(pressures) > 0:
        _print_subsection("Post-Processing")
        print("  Generating isotherm plot...")
        
        plt.figure(figsize=(10, 6))
        plt.errorbar(pressures, uptakes, yerr=uptake_stds, 
                     marker='o', linestyle='-', linewidth=2, markersize=8,
                     capsize=5, capthick=2, label='GCMC Simulation')
        plt.xlabel('Pressure (bar)', fontsize=12)
        plt.ylabel('Uptake (molecules/unit cell)', fontsize=12)
        
        title = f'{adsorbate_label} Adsorption Isotherm'
        if hasattr(atoms_frame, 'get_chemical_symbols'):
            title += f' in {atoms_frame.get_chemical_symbols()[0]} framework'
        title += f' at {temperature} K'
        plt.title(title, fontsize=14, fontweight='bold')
        
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=11)
        plt.tight_layout()
        
        # Save plot
        os.makedirs(output_dir, exist_ok=True)
        plot_filename = os.path.join(output_dir, 'isotherm.png')
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        _print_success(f"Isotherm plot saved to {plot_filename}")
    
    # 7. Save data to JSON
    isotherm_data = {
        'temperature': temperature,
        'pressures': pressures,
        'uptakes': uptakes,
        'uptake_stds': uptake_stds,
        'adsorption_energies': adsorption_energies,
        'unit_cell_volume_A3': cell_volume,
        'unit_cell_volume_cm3': cell_volume_cm3,
        'adsorbent_path': adsorbent_path,
        'adsorbate_path': adsorbate_path,
        'adsorbate_molecule': adsorbate_molecule,
        'n_equilibration_steps': n_equilibration_steps,
        'n_production_steps': n_production_steps
    }
    
    os.makedirs(output_dir, exist_ok=True)
    isotherm_file = os.path.join(output_dir, 'isotherm_data.json')
    with open(isotherm_file, 'w') as f:
        json.dump(isotherm_data, f, indent=4)
    _print_success(f"Isotherm data saved to {isotherm_file}")
    
    # 8. Print summary table
    if len(pressures) > 0:
        _print_section_header("Isotherm Summary")
        headers = ["Pressure (bar)", "Uptake (mol/uc)", "Std Dev", "Energy (eV)"]
        rows = [[f"{p:.2f}", f"{u:.3f}", f"{std:.3f}", f"{e:.4f}"] 
                for p, u, std, e in zip(pressures, uptakes, uptake_stds, adsorption_energies)]
        _print_table(headers, rows)
    
    print()
    _print_section_header("Simulation Complete")
    print(f"  Results saved to: {output_dir}/")
    print()
    
    # Show plot if in interactive environment
    if plot_isotherm and len(pressures) > 0:
        try:
            plt.show()
        except:
            print("  (Plot display not available in non-interactive environment)")
    
    return isotherm_data


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='MLIP-MC: Run GCMC isotherm simulation for gas adsorption',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python main.py --adsorbent tests/zif8.xyz --adsorbate-molecule CO2 --temperature 298.0 --pressures 0.1,0.5,1.0,2.0,5.0,10.0,20.0

  # Custom parameters
  python main.py \\
      --adsorbent framework.xyz \\
      --adsorbate-path co2.xyz \\
      --temperature 300.0 \\
      --pressures 0.1,0.5,1.0,2.0,5.0,10.0,20.0 \\
      --n-equil 10000 \\
      --n-prod 20000 \\
      --model models/model.pt \\
      --output-dir my_results

  # Single pressure point
  python main.py \\
      --adsorbent tests/zif8.xyz \\
      --adsorbate-molecule CO2 \\
      --temperature 298.0 \\
      --pressures 1.0 \\
      --n-equil 5000 \\
      --n-prod 15000

  # Skip plotting
  python main.py \\
      --adsorbent tests/zif8.xyz \\
      --adsorbate-molecule CO2 \\
      --temperature 298.0 \\
      --pressures 1.0 \\
      --no-plot
        """
    )
    
    parser.add_argument('--adsorbent', type=str, required=True,
                        help='Path to adsorbent structure file (.xyz, .cif, etc.)')
    parser.add_argument('--adsorbate-path', type=str, default=None,
                        help='Path to adsorbate structure file (.xyz, .cif, etc.)')
    parser.add_argument('--adsorbate-molecule', type=str, default=None,
                        help='Name of adsorbate molecule (e.g., CO2, CH4) if not using file')
    parser.add_argument('--temperature', type=float, required=True,
                        help='Temperature in Kelvin (required)')
    parser.add_argument('--pressures', type=str, required=True,
                        help='Comma-separated list of pressures in bar, or single number (required)')
    parser.add_argument('--n-equil', type=int, default=10000,
                        help='Number of equilibration steps (default: 10000)')
    parser.add_argument('--n-prod', type=int, default=20000,
                        help='Number of production steps (default: 20000)')
    parser.add_argument('--model', type=str, default='models/model.pt',
                        help='Path to MLIP model file. Can be a local path or Hugging Face repo (e.g., fengxuyoung/MLIP-MC). Missing files auto-download and are cached.')
    parser.add_argument('--hf-token', type=str, default=None,
                        help='Hugging Face authentication token (optional, uses cached token if available)')
    parser.add_argument('--output-dir', type=str, default='results',
                        help='Output directory for results (default: results)')
    parser.add_argument('--no-plot', action='store_true',
                        help='Skip generating isotherm plot')
    parser.add_argument('--save-interval', type=int, default=1000,
                        help='Interval for saving restart files (default: 1000)')
    parser.add_argument('--restart', action='store_true',
                        help='Enable restart functionality. Will look for restart files in output directory and resume if found.')
    
    return parser.parse_args()


def main():
    """Main entry point for the CLI."""
    # Use 'spawn' to ensure fresh processes for CUDA isolation
    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        pass  # Context might already be set
    
    args = parse_arguments()
    
    # Validate arguments
    if args.adsorbate_path is None and args.adsorbate_molecule is None:
        print("ERROR: Either --adsorbate-path or --adsorbate-molecule must be provided")
        sys.exit(1)
    
    # Parse pressure points
    try:
        pressure_points = [float(p.strip()) for p in args.pressures.split(',')]
    except ValueError:
        # Try as single number
        try:
            pressure_points = float(args.pressures)
        except ValueError:
            print(f"ERROR: Invalid pressure format: {args.pressures}")
            print("Please provide comma-separated numbers or a single number")
            sys.exit(1)
    
    # Run simulation
    try:
        run_gcmc(
            adsorbent_path=args.adsorbent,
            adsorbate_path=args.adsorbate_path,
            adsorbate_molecule=args.adsorbate_molecule,
            temperature=args.temperature,
            pressure_points=pressure_points,
            n_equilibration_steps=args.n_equil,
            n_production_steps=args.n_prod,
            model_path=args.model,
            output_dir=args.output_dir,
            plot_isotherm=not args.no_plot,
            hf_token=args.hf_token,
            save_interval=args.save_interval,
            restart=args.restart
        )
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


# CLI entry point is in mlip_mc.cli

