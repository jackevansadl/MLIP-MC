#!/usr/bin/env python3
"""Plot convergence of a Gibbs-ensemble run from its binary log.

Each record in ``log_gibbs_<T>K.bin`` is packed as ``iiidddddd``:
    step (i32), N1 (i32), N2 (i32),
    V1 (f64), V2 (f64), E1 (f64), E2 (f64), rho1 (f64), rho2 (f64)

where Vi is in Å^3, Ei in eV, and rhoi in molecules/Å^3.

Usage:
    python scripts/plot_gibbs_convergence.py results_gibbs_co2_trappe/log_gibbs_240.0K.bin
    python scripts/plot_gibbs_convergence.py <log.bin> --out conv.png --mw 44.009
"""

import argparse
import struct
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# Avogadro's number; converts (molecules / Å^3) × (g/mol) -> g/cm^3.
N_A = 6.02214076e23
ANGSTROM3_PER_CM3 = 1.0e24
NUM_DENSITY_TO_GCM3 = ANGSTROM3_PER_CM3 / N_A  # rho_n * Mw / N_A * (1e8)^3 = rho_n * Mw * NUM_DENSITY_TO_GCM3 / 1


def read_log(path: Path):
    """Return a structured numpy array with one row per record.

    The writer uses ``struct.pack("iiidddddd", ...)`` with **native
    alignment**, which inserts 4 bytes of padding after the three int32s
    so the float64s start on an 8-byte boundary. That makes each record
    64 bytes, not the 60 you'd get from a packed layout. We mirror that
    with ``np.dtype(..., align=True)``.
    """
    dtype = np.dtype(
        [
            ("step", "i4"),
            ("N1", "i4"),
            ("N2", "i4"),
            ("V1", "f8"),
            ("V2", "f8"),
            ("E1", "f8"),
            ("E2", "f8"),
            ("rho1", "f8"),
            ("rho2", "f8"),
        ],
        align=True,
    )
    expected_record_size = struct.calcsize("iiidddddd")
    if dtype.itemsize != expected_record_size:
        raise SystemExit(
            f"internal: numpy dtype itemsize {dtype.itemsize} != "
            f"struct.calcsize('iiidddddd') = {expected_record_size}; "
            "alignment mismatch between writer and reader."
        )

    raw = path.read_bytes()
    if len(raw) % dtype.itemsize != 0:
        print(
            f"warning: {path} length {len(raw)} is not a multiple of "
            f"{dtype.itemsize}; trailing bytes will be ignored",
            file=sys.stderr,
        )
        raw = raw[: len(raw) - (len(raw) % dtype.itemsize)]
    if not raw:
        raise SystemExit(f"{path} contains no complete records")
    return np.frombuffer(raw, dtype=dtype)


def plot(data, out_path: Path, mw_gmol: float, ref_liq=None, ref_vap=None,
         downsample=1):
    """Render a 2x2 convergence plot."""
    if downsample > 1:
        data = data[::downsample]
    step = data["step"]

    # Mass density (g/cm^3) per box from number density.
    # rho_n [molecules/Å^3] * Mw [g/mol] / N_A [1/mol] * (1e8 Å/cm)^3 [cm^3/Å^3]
    factor = mw_gmol * ANGSTROM3_PER_CM3 / N_A
    rho1_mass = data["rho1"] * factor
    rho2_mass = data["rho2"] * factor

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)

    ax = axes[0, 0]
    ax.plot(step, data["N1"], label="N$_1$ (liquid?)", lw=0.9)
    ax.plot(step, data["N2"], label="N$_2$ (vapor?)", lw=0.9)
    ax.set_ylabel("Molecules")
    ax.set_title("Particle counts")
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.plot(step, data["V1"], label="V$_1$", lw=0.9)
    ax.plot(step, data["V2"], label="V$_2$", lw=0.9)
    ax.set_ylabel("Volume / Å$^3$")
    ax.set_title("Box volumes")
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    ax.plot(step, rho1_mass, label="ρ$_1$", lw=0.9)
    ax.plot(step, rho2_mass, label="ρ$_2$", lw=0.9)
    if ref_liq is not None:
        ax.axhline(ref_liq, ls="--", color="C0", alpha=0.6,
                   label=f"ρ$_{{liq}}^{{ref}}$ = {ref_liq:.3f}")
    if ref_vap is not None:
        ax.axhline(ref_vap, ls="--", color="C1", alpha=0.6,
                   label=f"ρ$_{{vap}}^{{ref}}$ = {ref_vap:.3f}")
    ax.set_ylabel("Mass density / g cm$^{-3}$")
    ax.set_xlabel("MC step")
    ax.set_title("Coexistence densities")
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    ax.plot(step, data["E1"], label="E$_1$", lw=0.9)
    ax.plot(step, data["E2"], label="E$_2$", lw=0.9)
    ax.plot(step, data["E1"] + data["E2"], label="E$_1$+E$_2$", lw=0.9, color="k")
    ax.set_ylabel("Energy / eV")
    ax.set_xlabel("MC step")
    ax.set_title("Box energies")
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)

    fig.suptitle(
        f"Gibbs convergence — {len(data):,} logged accepted steps "
        f"(Mw = {mw_gmol:g} g/mol)"
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"wrote {out_path}")

    # Print a tail summary so the user gets numbers in stdout too.
    tail = data[-max(50, len(data) // 20):]
    print()
    print("Tail statistics (last {} records):".format(len(tail)))
    print(f"  N1   mean ± std: {tail['N1'].mean():.1f} ± {tail['N1'].std():.1f}")
    print(f"  N2   mean ± std: {tail['N2'].mean():.1f} ± {tail['N2'].std():.1f}")
    print(f"  ρ1   mean ± std: "
          f"{(tail['rho1'] * factor).mean():.4f} ± "
          f"{(tail['rho1'] * factor).std():.4f} g/cm^3")
    print(f"  ρ2   mean ± std: "
          f"{(tail['rho2'] * factor).mean():.4f} ± "
          f"{(tail['rho2'] * factor).std():.4f} g/cm^3")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("logfile", type=Path, help="Path to log_gibbs_<T>K.bin")
    p.add_argument("--out", type=Path, default=None,
                   help="Output figure path (default: alongside the log file)")
    p.add_argument("--mw", type=float, default=44.009,
                   help="Adsorbate molar mass in g/mol (default 44.009 for CO2)")
    p.add_argument("--ref-liq", type=float, default=None,
                   help="Reference liquid density in g/cm^3 (e.g. 0.926 for CO2 at 240 K)")
    p.add_argument("--ref-vap", type=float, default=None,
                   help="Reference vapor density in g/cm^3 (e.g. 0.074 for CO2 at 240 K)")
    p.add_argument("--downsample", type=int, default=1,
                   help="Take every Nth point (useful for very long logs)")
    args = p.parse_args()

    if not args.logfile.exists():
        raise SystemExit(f"no such file: {args.logfile}")

    data = read_log(args.logfile)
    out = args.out or args.logfile.with_suffix(".png")
    plot(data, out, args.mw,
         ref_liq=args.ref_liq, ref_vap=args.ref_vap,
         downsample=args.downsample)


if __name__ == "__main__":
    main()
