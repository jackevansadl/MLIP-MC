#!/usr/bin/env python3
"""
Example script that runs a short CO2 GCMC calculation in ZIF-8 with MLIP-MC.

The script reuses the helper exposed by the package (`run_gcmc`)
and keeps the configuration minimal so it can serve as a quick sanity check
after installation.  Modify the parameters directly in the function call below.
"""

from __future__ import annotations

from pathlib import Path

from mlip_mc import run_gcmc


def main() -> None:
    example_dir = Path(__file__).resolve().parent
    framework_file = example_dir / "zif8.xyz"
    output_dir = example_dir / "results"
    output_dir.mkdir(exist_ok=True)

    results = run_gcmc(
        adsorbent_path=str(framework_file),
        adsorbate_molecule="CO2",
        temperature=298.0,
        pressure_points=[0.1, 1.0, 5.0],
        n_equilibration_steps=2000,
        n_production_steps=5000,
        model_path="fengxuyoung/MLIP-MC",
        output_dir=str(output_dir),
        hf_token=None,
    )

    print("\nIsotherm summary:")
    for pressure, uptake, uptake_std in zip(
        results["pressures"], results["uptakes"], results["uptake_stds"]
    ):
        uptake_str = f"{uptake:.3f}" if uptake is not None else "n/a"
        std_str = f"{uptake_std:.3f}" if uptake_std is not None else "n/a"
        print(f"  {pressure:>6.2f} bar -> uptake {uptake_str} ± {std_str} molecules/unit cell")

    print(f"\nDetailed artifacts have been written to: {output_dir}")


if __name__ == "__main__":
    main()
