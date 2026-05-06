#!/usr/bin/env python3
"""
MLIP-MC: Monte Carlo Simulations with Machine-Learned Interatomic Potentials

Main entry point for running GCMC isotherm simulations.
"""

import os
import sys
import json
import shutil
import traceback
from pathlib import Path
from typing import Union, List, Tuple, Dict, Any, Optional
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

# NOTE: We delay import of torch, fairchem/mace-torch, and other heavy libraries 
# until inside the process to ensure CUDA environment variables take effect first.


DEFAULT_HF_FILENAME = "model.pt"
# Cache root directory: use MLIP_MC_CACHE env var if set, otherwise ~/.cache/mlip-mc
_cache_dir = os.environ.get("MLIP_MC_CACHE")
if _cache_dir:
    MODEL_CACHE_ROOT = Path(_cache_dir).expanduser()
else:
    MODEL_CACHE_ROOT = Path.home() / ".cache" / "mlip-mc"


def _read_binary_log(log_path: str) -> List[Dict[str, Any]]:
    """Read binary log file and return list of records."""
    from mlip_mc.src.utilities import read_binary_log
    return read_binary_log(log_path)


def _copy_model_to_target(downloaded_path: str, target_path: str) -> str:
    """Copy downloaded model to target location."""
    target_dir = Path(target_path).parent
    target_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(downloaded_path, target_path)
    return target_path


def download_model_from_huggingface(
    model_path: str,
    repo_id: str,
    filename: str,
    token: Optional[str] = None
) -> str:
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
    if os.path.exists(model_path):
        return model_path
    
    # Always use cache directory for downloads to ensure consistent caching
    # as documented: ~/.cache/mlip-mc/<repo>/<filename>
    cache_path = MODEL_CACHE_ROOT / repo_id / filename
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    target_path = str(cache_path)
    
    try:
        from huggingface_hub import hf_hub_download, login
        
        try:
            downloaded_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                token=token,
                local_dir=None
            )
            
            if downloaded_path != target_path:
                return _copy_model_to_target(downloaded_path, target_path)
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
                    downloaded_path = hf_hub_download(
                        repo_id=repo_id,
                        filename=filename,
                        token=token,
                        local_dir=None
                    )
                    if downloaded_path != target_path:
                        return _copy_model_to_target(downloaded_path, target_path)
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


def _resolve_model_spec(model_spec: str) -> Tuple[str, Optional[str], Optional[str], bool]:
    """
    Normalize the model argument into a local path plus (optional) Hugging Face info.

    Users must provide either a local filesystem path or an explicit ``hf://``
    URI.  There is no default Hugging Face repository.

    Returns
    -------
    tuple[str, str | None, str | None, bool]
        (local_path, repo_id, hf_filename, is_hf)
        - local_path : normalized local filesystem path
        - repo_id    : Hugging Face repo id if applicable, None for local paths
        - hf_filename: filename within the repo, None for local paths
        - is_hf      : True if the user explicitly requested an HF model via the
                       ``hf://`` scheme, False otherwise
    """
    if not model_spec:
        raise ValueError(
            "model_path is required. Provide a local path to your model file "
            "or an explicit Hugging Face URI (e.g., hf://your-org/your-repo)."
        )

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
            raise ValueError(
                "Empty Hugging Face URI. Provide a repo id, "
                "e.g., hf://your-org/your-repo or hf://your-org/your-repo:model.pt"
            )

        if ":" in spec:
            repo_part, file_part = spec.split(":", 1)
            repo_id = repo_part.strip()
            hf_filename = file_part.strip() or DEFAULT_HF_FILENAME
            if not repo_id:
                raise ValueError(
                    "Empty repo id in Hugging Face URI. "
                    "Provide a repo id, e.g., hf://your-org/your-repo:model.pt"
                )
        else:
            repo_id = spec
            hf_filename = DEFAULT_HF_FILENAME

        hf_filename = hf_filename.lstrip("/")
        repo_id = repo_id.strip().strip("/")
        local_path = MODEL_CACHE_ROOT / repo_id / hf_filename
        return str(local_path), repo_id, hf_filename, True

    # Local path — no Hugging Face info.
    return model_spec, None, None, False


def _format_device_str(gpu_id: Union[int, str]) -> str:
    """Format device string for display."""
    return f"GPU {gpu_id}" if isinstance(gpu_id, int) else "CPU"


def _detect_backend() -> str:
    """
    Detect which MLIP backend is available.
    
    Returns
    -------
    str
        'fairchem', 'mace-torch', or 'orb-models', raises ImportError if none is available
    """
    try:
        import fairchem.core
        return 'fairchem'
    except ImportError:
        pass
    
    try:
        from mace.calculators import mace_mp
        return 'mace-torch'
    except ImportError:
        pass
    
    try:
        from orb_models.forcefield import pretrained
        from orb_models.forcefield.calculator import ORBCalculator
        return 'orb-models'
    except ImportError:
        pass
    
    raise ImportError(
        "No MLIP backend found. Please install one of:\n"
        "  pip install mlip-mc[fairchem]\n"
        "  pip install mlip-mc[mace-torch]\n"
        "  pip install mlip-mc[orb-models]"
    )


def _load_model(model_path: str, device: str, backend: Optional[str] = None, orb_model_variant: str = 'omat') -> Any:
    """
    Load an MLIP model using the appropriate backend.

    Parameters
    ----------
    model_path : str
        Path to the model file (local path or HuggingFace repo).
        For ORB models, this should be the path to the weights checkpoint file.
    device : str
        Device to use ('cuda' or 'cpu')
    backend : str, optional
        Backend to use ('fairchem', 'mace-torch', or 'orb-models'). If None, auto-detects.
    orb_model_variant : str, optional
        ORB model variant to use: 'omat' for orb_v3_conservative_inf_omat (default)
        or 'omol' for orb_v3_conservative_omol (requires charge=0, spin=1).

    Returns
    -------
    Any
        ASE calculator instance (FAIRChemCalculator, mace_mp, or ORBCalculator)
    """
    if backend is None:
        backend = _detect_backend()

    if backend == 'fairchem':
        from fairchem.core import FAIRChemCalculator
        from fairchem.core.units.mlip_unit import load_predict_unit

        predictor = load_predict_unit(model_path, device=device)
        model = FAIRChemCalculator(predictor, task_name="odac")
        return model

    elif backend == 'mace-torch':
        from mace.calculators import mace_mp

        # MACE uses 'cuda' or 'cpu' for device, same as fairchem
        # Convert device string if needed (should already be 'cuda' or 'cpu')
        mace_device = device if device in ('cuda', 'cpu') else 'cpu'

        model = mace_mp(
            model=model_path,
            default_dtype="float64",
            device=mace_device
        )
        return model

    elif backend == 'orb-models':
        from orb_models.forcefield import pretrained
        from orb_models.forcefield.calculator import ORBCalculator

        # Select the pretrained loader based on the variant flag.
        if orb_model_variant == 'omol':
            _loader = pretrained.orb_v3_conservative_omol
        else:
            _loader = pretrained.orb_v3_conservative_inf_omat

        # ORB models normally download pretrained weights automatically. We make
        # ``weights_path`` optional so that:
        #   - By default, the pretrained loader handles downloading and caching.
        #   - If the user provides ``--model some/orb/weights.pt`` and that file
        #     exists, we pass it through as ``weights_path`` to override the
        #     default checkpoint.
        if model_path is not None and os.path.exists(model_path):
            orbff = _loader(
                device=device,
                precision="float32-high",
                weights_path=model_path,
            )
        else:
            orbff = _loader(
                device=device,
                precision="float32-high",
            )

        calc = ORBCalculator(orbff, device=device)
        return calc

    elif backend == 'lammps-classical':
        import json
        from .src.lammps_backend import build_lammps_calculator

        config_path = Path(model_path)
        with open(config_path, "r") as fh:
            config = json.load(fh)

        lmp_input_path = Path(config["lammps_input"])
        if not lmp_input_path.is_absolute():
            lmp_input_path = config_path.parent / lmp_input_path

        atom_types = {str(k): int(v) for k, v in config["atom_types"].items()}
        charges = config.get("charges")

        calc = build_lammps_calculator(
            lmpcmds=lmp_input_path,
            atom_types=atom_types,
            log_file=config.get("log_file"),
            keep_alive=bool(config.get("keep_alive", True)),
            lammps_header=config.get("lammps_header"),
            lammps_name=config.get("lammps_name"),
            create_box_extra=config.get("create_box_extra"),
            intra_bonds=config.get("intra_bonds"),
        )
        if charges is not None:
            calc._mlipmc_charges = {str(k): float(v) for k, v in charges.items()}
        return calc

    else:
        raise ValueError(f"Unknown backend: {backend}")


def run_single_pressure(
    P_bar: float,
    T: float,
    model_path: str,
    atoms_frame: Atoms,
    atoms_ads: Atoms,
    n_equilibration_steps: int,
    n_production_steps: int,
    n_total_steps: int,
    gpu_id: Union[int, str],
    result_queue: Queue,
    adsorbate_name: Optional[str] = None,
    output_dir: str = 'results',
    checkpoint_interval: int = 10000,
    write_trajectory: bool = False,
    trajectory_interval: int = 100,
    overwrite_checkpoints: bool = False,
    orb_model_variant: str = 'omat',
) -> None:
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
    checkpoint_interval : int, optional
        Interval for saving checkpoints (default: 10000)
    write_trajectory : flag, optional
        If specified will output an ase atoms trajectory
    trajectory_interval : int, optional
        Interval for saving trajectory (default: 100)
    overwrite_checkpoints : flag, optional
        If specified will overwrite previous checkpoints when writing new checkpoint
    """
    try:
        # 1. Set Environment Variables for GPU Isolation
        if isinstance(gpu_id, int):
            os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        
        # 2. Import libraries AFTER setting environment variables
        import torch
        from mlip_mc.src.gcmc import MLP_GCMC
        from mlip_mc.src.utilities import PREOS
        
        # 3. Determine device (now restricted to just the one visible GPU)
        if isinstance(gpu_id, int) and torch.cuda.is_available():
            # Even though we requested gpu_id, inside this process it will appear as cuda:0
            # because we masked the others with CUDA_VISIBLE_DEVICES
            # IMPORTANT: both backends require 'cuda' or 'cpu', not 'cuda:0'
            device = 'cuda' 
        else:
            device = 'cpu'
        
        # 4. Detect backend and load model
        backend = _detect_backend()
        device_str = _format_device_str(gpu_id)
        print(f"  [{device_str}] Using backend: {backend}")
        model = _load_model(model_path, device=device, backend=backend, orb_model_variant=orb_model_variant)

        P = P_bar * bar
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
            n_equilibration_steps=n_equilibration_steps,  # Total equilibration steps (target)
            n_production_steps=n_production_steps,  # Total production steps (target)
            checkpoint_interval=checkpoint_interval,
            write_trajectory=write_trajectory,
            trajectory_interval=trajectory_interval,
            overwrite_checkpoints=overwrite_checkpoints
        )
        
        print(f"  [{device_str}] Running {n_total_steps} steps...")
        gcmc.run(N=n_total_steps)
        
        log_file_bin = os.path.join(output_dir, f"log_{P/bar:.2f}bar.bin")
        results_file_npz = os.path.join(output_dir, f"results_{P/bar:.2f}bar.npz")
        results_file_json = os.path.join(output_dir, f"results_{P/bar:.2f}bar.json")
        
        uptake_data = []
        energy_data = []
        
        if os.path.exists(log_file_bin):
            log_data = _read_binary_log(log_file_bin)
            uptake_data = [d['uptake'] for d in log_data if d['step'] > n_equilibration_steps]
            energy_data = [d['interaction_energy'] for d in log_data if d['step'] > n_equilibration_steps]
            
            if not uptake_data:
                uptake_data = [d['uptake'] for d in log_data]
                energy_data = [d['interaction_energy'] for d in log_data]
            
            if uptake_data:
                avg_uptake = np.mean(uptake_data)
                std_uptake = np.std(uptake_data)
                avg_energy = np.mean([e for e in energy_data if e != 0]) if any(e != 0 for e in energy_data) else 0.0
                
                device_str = _format_device_str(gpu_id)
                print(f"  [{device_str}] Completed P = {P_bar:.2f} bar: Uptake = {avg_uptake:.3f} ± {std_uptake:.3f}")
                
                # Return results via queue
                result_queue.put({
                    'pressure': P_bar,
                    'uptake': avg_uptake,
                    'uptake_std': std_uptake,
                    'energy': avg_energy
                })
            else:
                 device_str = _format_device_str(gpu_id)
                 print(f"  [{device_str}] Warning: No valid data found in log for P = {P_bar:.2f} bar")
                 result_queue.put({
                    'pressure': P_bar,
                    'uptake': None,
                    'uptake_std': None,
                    'energy': None
                 })
                 
        elif os.path.exists(results_file_npz):
            data = np.load(results_file_npz)
            uptake_data = data['uptake'][n_equilibration_steps:]
            energy_data = data['interaction_energy'][n_equilibration_steps:]
            data.close()
            
            avg_uptake = np.mean(uptake_data)
            std_uptake = np.std(uptake_data)
            avg_energy = np.mean([e for e in energy_data if e != 0]) if any(e != 0 for e in energy_data) else 0.0
            
            device_str = _format_device_str(gpu_id)
            print(f"  [{device_str}] Completed P = {P_bar:.2f} bar: Uptake = {avg_uptake:.3f} ± {std_uptake:.3f}")
            
            # Return results via queue
            result_queue.put({
                'pressure': P_bar,
                'uptake': avg_uptake,
                'uptake_std': std_uptake,
                'energy': avg_energy
            })
        elif os.path.exists(results_file_json):
            with open(results_file_json, 'r') as f:
                results = json.load(f)
            
            uptake_data = results['uptake'][n_equilibration_steps:]
            energy_data = results['interaction_energy'][n_equilibration_steps:]
            
            avg_uptake = np.mean(uptake_data)
            std_uptake = np.std(uptake_data)
            avg_energy = np.mean([e for e in energy_data if e != 0]) if any(e != 0 for e in energy_data) else 0.0
            
            device_str = _format_device_str(gpu_id)
            print(f"  [{device_str}] Completed P = {P_bar:.2f} bar: Uptake = {avg_uptake:.3f} ± {std_uptake:.3f}")
            
            # Return results via queue
            result_queue.put({
                'pressure': P_bar,
                'uptake': avg_uptake,
                'uptake_std': std_uptake,
                'energy': avg_energy
            })
        else:
            device_str = _format_device_str(gpu_id)
            print(f"  [{device_str}] Warning: Results file not found for P = {P_bar:.2f} bar")
            result_queue.put({
                'pressure': P_bar,
                'uptake': None,
                'uptake_std': None,
                'energy': None
            })
    except Exception as e:
        device_str = _format_device_str(gpu_id)
        print(f"  [{device_str}] ERROR in simulation at P = {P_bar:.2f} bar: {e}", file=sys.stderr)
        traceback.print_exc()
        result_queue.put({
            'pressure': P_bar,
            'uptake': None,
            'uptake_std': None,
            'energy': None,
            'error': str(e)
        })


def _normalize_pressure_points(pressure_points: Union[float, List[float]]) -> List[float]:
    """Normalize pressure points to a list of floats."""
    if isinstance(pressure_points, (int, float)):
        return [float(pressure_points)]
    return [float(p) for p in pressure_points]


def run_gcmc(
    adsorbent_path: str,
    model_path: str,
    adsorbate_path: Optional[str] = None,
    adsorbate_molecule: Optional[str] = None,
    temperature: float = 298.0,
    pressure_points: Optional[Union[float, List[float]]] = None,
    n_equilibration_steps: int = 10000,
    n_production_steps: int = 20000,
    output_dir: str = 'results',
    hf_token: Optional[str] = None,
    checkpoint_interval: int = 10000,
    write_trajectory: bool = False,
    trajectory_interval: int = 100,
    overwrite_checkpoints: bool = False,
    orb_model_variant: str = 'omat',
) -> Dict[str, Any]:
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
    model_path : str
        Path to the MLIP model file (required).
        Can be a local path or a Hugging Face URI (e.g., "hf://your-org/your-repo").
        When using the ``hf://`` scheme, files are downloaded and cached automatically.
    output_dir : str, optional
        Directory to save results (default: 'results')
    hf_token : str, optional
        Hugging Face authentication token. If None, uses cached token or prompts for login
    checkpoint_interval : int, optional
        Interval for saving checkpoints (default: 1000)
    
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
    
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        try:
            mp.set_start_method('spawn')
        except RuntimeError:
            pass
    
    import torch

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

    # Detect backend once in the parent process so we can make backend-aware
    # decisions about how to interpret model paths (e.g., which defaults or
    # Hugging Face models are applicable).
    backend = _detect_backend()

    model_path, hf_repo_id, hf_model_filename, is_hf = _resolve_model_spec(model_path)

    _print_subsection("Model Configuration")

    if is_hf:
        # User explicitly requested a Hugging Face model via hf:// scheme.
        cache_path = MODEL_CACHE_ROOT / hf_repo_id / hf_model_filename
        if os.path.exists(cache_path):
            model_path = str(cache_path)
            _print_info("Model Status", f"Found in cache at {model_path}")
        elif not os.path.exists(model_path):
            _print_info("Model Status", f"Not found at {model_path}")
            _print_info("Action", "Downloading from Hugging Face Hub...")
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
    else:
        # Local path provided.
        if not os.path.exists(model_path):
            if backend == 'orb-models':
                _print_info(
                    "Model Status",
                    "Local model file not found; using orb-models pretrained checkpoint "
                    "(weights will be downloaded/cached automatically).",
                )
                model_path = None
            else:
                raise FileNotFoundError(
                    f"Model file not found at {model_path}\n"
                    "Please provide a valid local model path or use the "
                    "'hf://' scheme to download from Hugging Face."
                )
        else:
            _print_info("Model Status", f"Using local model at {model_path}")
    
    _print_subsection("Structure Loading")
    if not os.path.exists(adsorbent_path):
        raise FileNotFoundError(f"Adsorbent file not found: {adsorbent_path}")

    _print_info("Adsorbent", f"Loading from {adsorbent_path}...")
    atoms_frame = read(adsorbent_path)
    atoms_frame = Atoms(
        numbers=atoms_frame.numbers,
        positions=atoms_frame.positions,
        cell=atoms_frame.cell,
        pbc=atoms_frame.pbc
    )

    atoms_frame.info["charge"] = 0  # total charge
    atoms_frame.info["spin"] = 1  #  spin multiplicity
    
    if adsorbate_path is not None:
        if not os.path.exists(adsorbate_path):
            raise FileNotFoundError(f"Adsorbate file not found: {adsorbate_path}")
        _print_info("Adsorbate", f"Loading from {adsorbate_path}...")
        atoms_ads = read(adsorbate_path)
        atoms_ads = Atoms(numbers=atoms_ads.numbers, positions=atoms_ads.positions)
        if adsorbate_molecule is not None:
            adsorbate_name = adsorbate_molecule
            _print_info("Adsorbate Name", adsorbate_name)
        else:
            adsorbate_name = atoms_ads.get_chemical_formula().replace(' ', '')
            _print_info("Adsorbate Name", f"{adsorbate_name} (extracted from file)")
    elif adsorbate_molecule is not None:
        _print_info("Adsorbate", f"Creating {adsorbate_molecule} molecule...")
        atoms_ads = molecule(adsorbate_molecule)
        atoms_ads = Atoms(numbers=atoms_ads.numbers, positions=atoms_ads.positions)
        adsorbate_name = adsorbate_molecule
        _print_info("Adsorbate Name", adsorbate_name)
    else:
        raise ValueError("Either adsorbate_path or adsorbate_molecule must be provided")
    
    atoms_ads.info["charge"] = 0  # total charge
    atoms_ads.info["spin"] = 1  #  spin multiplicity
    
    if pressure_points is None:
        raise ValueError("pressure_points must be provided (float or list of floats)")
    
    pressure_points = _normalize_pressure_points(pressure_points)
    
    _print_subsection("Simulation Parameters")
    _print_info("Temperature", f"{temperature} K")
    _print_info("Pressure Points", f"{pressure_points} bar")
    _print_info("Equilibration Steps", f"{n_equilibration_steps}")
    _print_info("Production Steps", f"{n_production_steps}")
    
    cell_volume = np.linalg.det(atoms_frame.get_cell())
    cell_volume_cm3 = cell_volume * 1e-24
    
    _print_subsection("Simulation Execution")
    if n_gpus == 0:
        _print_warning("Running sequentially on CPU (no GPUs available)")
        devices = ['cpu']
    else:
        devices = list(range(n_gpus))
        active_gpus = min(len(pressure_points), n_gpus)
        _print_info("Distribution", f"{len(pressure_points)} pressure point(s) across {active_gpus} of {n_gpus} GPU(s)")
    
    processes = []
    result_queue = Queue()
    n_total_steps = n_equilibration_steps + n_production_steps
    max_concurrent = n_gpus if n_gpus > 0 else 1
    
    import time
    active_processes = []
    
    for i, P_bar in enumerate(pressure_points):
        while len(active_processes) >= max_concurrent:
            for p in active_processes[:]:
                if not p.is_alive():
                    p.join()
                    active_processes.remove(p)
            if len(active_processes) >= max_concurrent:
                time.sleep(0.5)

        gpu_id = devices[i % len(devices)] if isinstance(devices[0], int) else 'cpu'
        
        p = Process(target=run_single_pressure, args=(
            P_bar, temperature, model_path, atoms_frame, atoms_ads,
            n_equilibration_steps, n_production_steps, n_total_steps,
            gpu_id, result_queue, adsorbate_name, output_dir, checkpoint_interval,
            write_trajectory, trajectory_interval, overwrite_checkpoints,
            orb_model_variant,
        ))
        p.start()
        active_processes.append(p)
        processes.append(p)
        
        device_str = _format_device_str(gpu_id) if isinstance(gpu_id, int) else str(gpu_id).upper()
        _print_info("Started", f"P = {P_bar:.2f} bar on {device_str}")
    
    print(f"\n  Waiting for {len(processes)} simulation(s) to complete...")
    for p in processes:
        if p.is_alive():
            p.join()
    
    print("\n  Collecting results...")
    results = []
    while not result_queue.empty():
        results.append(result_queue.get())
    
    results.sort(key=lambda x: x['pressure'])
    
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
                print(f"    Error: {r['error']}", file=sys.stderr)
    
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
    
    return isotherm_data


def run_widom(
    adsorbent_path: str,
    model_path: str,
    adsorbate_path: Optional[str] = None,
    adsorbate_molecule: Optional[str] = None,
    temperature: float = 298.0,
    n_trials: int = 10000,
    output_dir: str = 'results',
    hf_token: Optional[str] = None,
    gpu_id: Union[int, str] = 0,
    orb_model_variant: str = 'omat',
) -> Dict[str, Any]:
    """
    Run Widom insertion calculation for estimating Henry's constants and adsorption energies.
    
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
    n_trials : int, optional
        Number of Widom insertion trials (default: 10000)
    model_path : str
        Path to the MLIP model file (required).
        Can be a local path or a Hugging Face URI (e.g., "hf://your-org/your-repo").
        When using the ``hf://`` scheme, files are downloaded and cached automatically.
    output_dir : str, optional
        Directory to save results (default: 'results')
    hf_token : str, optional
        Hugging Face authentication token. If None, uses cached token or prompts for login
    gpu_id : int or str, optional
        GPU device ID to use (int for GPU, 'cpu' for CPU, default: 0)
    
    Returns
    -------
    dict
        Dictionary containing Widom results:
        - 'temperature': temperature (K)
        - 'attempts': total number of insertion attempts
        - 'valid_insertions': number of valid insertions (no VDW overlap)
        - 'vdw_overlaps': number of rejected insertions due to VDW overlap
        - 'widom_adsorption_energy': weighted average adsorption energy (eV)
        - 'arithmetic_adsorption_energy': arithmetic average adsorption energy (eV)
        - 'average_boltzmann_factor': average Boltzmann factor
        - 'raw_adsorption_energies': list of all interaction energies (eV)
    """
    _print_banner()
    _print_section_header("Widom Insertion Calculation")
    
    import torch
    from mlip_mc.src.widom import MLP_Widom
    
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
    
    # Detect backend
    backend = _detect_backend()
    
    model_path, hf_repo_id, hf_model_filename, is_hf = _resolve_model_spec(model_path)
    
    _print_subsection("Model Configuration")
    
    if is_hf:
        # User explicitly requested a Hugging Face model via hf:// scheme.
        cache_path = MODEL_CACHE_ROOT / hf_repo_id / hf_model_filename
        if os.path.exists(cache_path):
            model_path = str(cache_path)
            _print_info("Model Status", f"Found in cache at {model_path}")
        elif not os.path.exists(model_path):
            _print_info("Model Status", f"Not found at {model_path}")
            _print_info("Action", "Downloading from Hugging Face Hub...")
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
    else:
        # Local path provided.
        if not os.path.exists(model_path):
            if backend == 'orb-models':
                _print_info(
                    "Model Status",
                    "Local model file not found; using orb-models pretrained checkpoint "
                    "(weights will be downloaded/cached automatically).",
                )
                model_path = None
            else:
                raise FileNotFoundError(
                    f"Model file not found at {model_path}\n"
                    "Please provide a valid local model path or use the "
                    "'hf://' scheme to download from Hugging Face."
                )
        else:
            _print_info("Model Status", f"Using local model at {model_path}")
    
    _print_subsection("Structure Loading")
    if not os.path.exists(adsorbent_path):
        raise FileNotFoundError(f"Adsorbent file not found: {adsorbent_path}")
    
    _print_info("Adsorbent", f"Loading from {adsorbent_path}...")
    atoms_frame = read(adsorbent_path)
    atoms_frame = Atoms(
        numbers=atoms_frame.numbers,
        positions=atoms_frame.positions,
        cell=atoms_frame.cell,
        pbc=atoms_frame.pbc
    )
    atoms_frame.info["charge"] = 0  # total charge
    atoms_frame.info["spin"] = 1  #  spin multiplicity
    
    if adsorbate_path is not None:
        if not os.path.exists(adsorbate_path):
            raise FileNotFoundError(f"Adsorbate file not found: {adsorbate_path}")
        _print_info("Adsorbate", f"Loading from {adsorbate_path}...")
        atoms_ads = read(adsorbate_path)
        atoms_ads = Atoms(numbers=atoms_ads.numbers, positions=atoms_ads.positions)
        adsorbate_name = atoms_ads.get_chemical_formula().replace(' ', '')
        _print_info("Adsorbate Name", adsorbate_name)
    elif adsorbate_molecule is not None:
        _print_info("Adsorbate", f"Creating {adsorbate_molecule} molecule...")
        atoms_ads = molecule(adsorbate_molecule)
        atoms_ads = Atoms(numbers=atoms_ads.numbers, positions=atoms_ads.positions)
        adsorbate_name = adsorbate_molecule
        _print_info("Adsorbate Name", adsorbate_name)
    else:
        raise ValueError("Either adsorbate_path or adsorbate_molecule must be provided")
    
    atoms_ads.info["charge"] = 0  # total charge
    atoms_ads.info["spin"] = 1  #  spin multiplicity
    _print_subsection("Simulation Parameters")
    _print_info("Temperature", f"{temperature} K")
    _print_info("Number of Trials", f"{n_trials}")
    
    cell_volume = np.linalg.det(atoms_frame.get_cell())
    cell_volume_cm3 = cell_volume * 1e-24
    _print_info("Unit Cell Volume", f"{cell_volume:.2f} Å³ ({cell_volume_cm3:.2e} cm³)")
    
    _print_subsection("Simulation Execution")
    
    # Set device
    if isinstance(gpu_id, int) and gpu_id >= 0 and torch.cuda.is_available():
        if gpu_id >= torch.cuda.device_count():
            _print_warning(f"GPU {gpu_id} not available, using GPU 0")
            gpu_id = 0
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        device = 'cuda'
        device_str = f"GPU {gpu_id}"
    else:
        device = 'cpu'
        device_str = "CPU"
    
    _print_info("Device", device_str)
    _print_info("Backend", backend)
    
    # Load model
    model = _load_model(model_path, device=device, backend=backend, orb_model_variant=orb_model_variant)

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Create Widom instance
    widom = MLP_Widom(
        model=model,
        atoms_frame=atoms_frame,
        atoms_ads=atoms_ads,
        T=temperature,
        device=device,
        vdw_radii=vdw_radii,
        output_dir=output_dir
    )
    
    # Run simulation
    print(f"\n  [{device_str}] Running {n_trials} Widom insertion trials...")
    widom.run(N=n_trials)
    
    # Read results
    results_file = os.path.join(output_dir, 'widom_results.json')
    
    widom_data = {}
    if os.path.exists(results_file):
        with open(results_file, 'r') as f:
            widom_data = json.load(f)
    else:
        # Fallback: construct from widom stats
        widom_data = {
            'temperature': temperature,
            'attempts': widom.stats['attempts'],
            'valid_insertions': widom.stats['valid_insertions'],
            'vdw_overlaps': widom.stats['vdw_overlaps']
        }
    
    _print_section_header("Results Summary")
    _print_info("Temperature", f"{widom_data.get('temperature', temperature)} K")
    _print_info("Total Attempts", f"{widom_data.get('attempts', 0)}")
    _print_info("Valid Insertions", f"{widom_data.get('valid_insertions', 0)}")
    _print_info("VDW Overlaps", f"{widom_data.get('vdw_overlaps', 0)}")
    
    if 'widom_adsorption_energy' in widom_data:
        _print_info("Widom Adsorption Energy", f"{widom_data['widom_adsorption_energy']:.5f} eV")
        _print_info("Arithmetic Avg Energy", f"{widom_data['arithmetic_adsorption_energy']:.5f} eV")
        _print_info("Avg Boltzmann Factor", f"{widom_data['average_boltzmann_factor']:.5e}")
    
    print()
    _print_section_header("Simulation Complete")
    print(f"  Results saved to: {output_dir}/")
    print()
    
    return widom_data


def parse_arguments() -> argparse.Namespace:
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
      --output-dir my_results \\
      --save-interval 1000
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
    parser.add_argument('--model', type=str, required=True,
                        help='Path to MLIP model file (required). Can be a local path or Hugging Face URI (e.g., hf://your-org/your-repo).')
    parser.add_argument('--hf-token', type=str, default=None,
                        help='Hugging Face authentication token (optional, uses cached token if available)')
    parser.add_argument('--output-dir', type=str, default='results',
                        help='Output directory for results (default: results)')
    parser.add_argument('--checkpoint-interval', type=int, default=10000,
                        help='Interval for saving checkpoints (default: 10000)')
    parser.add_argument('--write-trajectory', action='store_true',
                        help='Write trajectory files')
    parser.add_argument('--trajectory-interval', type=int, default=100,
                        help='Interval for writing structures (default: 100)')
    parser.add_argument('--overwrite-checkpoints', action='store_true',
                        help='Write trajectory files')
    parser.add_argument('--orb-variant', type=str, default='omat', choices=['omat', 'omol'],
                        help='ORB model variant: omat (default, orb_v3_conservative_inf_omat) or omol (orb_v3_conservative_omol, charge=0 spin=1)')
    return parser.parse_args()


def _parse_pressures(pressure_str: str) -> Union[float, List[float]]:
    """
    Parse pressure string into float or list of floats.
    
    Parameters
    ----------
    pressure_str : str
        Comma-separated list of pressures or a single number
        
    Returns
    -------
    Union[float, List[float]]
        Single pressure value or list of pressure values
        
    Raises
    ------
    ValueError
        If the pressure string cannot be parsed
    """
    parts = [p.strip() for p in pressure_str.split(',')]
    
    if len(parts) == 1:
        return float(parts[0])
    
    return [float(p) for p in parts]


def main() -> None:
    """Main entry point for the CLI."""
    # Use 'spawn' to ensure fresh processes for CUDA isolation
    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        pass  # Context might already be set
    
    args = parse_arguments()
    
    # Validate arguments
    if not args.adsorbate_path and not args.adsorbate_molecule:
        print("ERROR: Either --adsorbate-path or --adsorbate-molecule must be provided",
              file=sys.stderr)
        sys.exit(1)
    
    # Parse pressure points
    try:
        pressure_points = _parse_pressures(args.pressures)
    except ValueError as e:
        print(f"ERROR: Invalid pressure format: {args.pressures}", file=sys.stderr)
        print("Please provide comma-separated numbers or a single number", file=sys.stderr)
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
            hf_token=args.hf_token,
            checkpoint_interval=args.checkpoint_interval,
            write_trajectory=args.write_trajectory,
            trajectory_interval=args.trajectory_interval,
            overwrite_checkpoints=args.overwrite_checkpoints,
            orb_model_variant=args.orb_variant,
        )
    except Exception as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)


# CLI entry point is in mlip_mc.cli

