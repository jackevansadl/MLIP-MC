"""Classical-potential backend built on ASE's LAMMPSlib.

Exposes a single factory, ``build_lammps_calculator``, that returns an ASE
calculator backed by an in-process LAMMPS instance. The caller supplies raw
LAMMPS commands verbatim, so any pair_style / kspace_style / set charge
sequence supported by LAMMPS is available without a translation layer.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Union


def _read_lmp_file(path: Union[str, Path]) -> List[str]:
    lines: List[str] = []
    with open(path, "r") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            lines.append(line)
    return lines


def build_lammps_calculator(
    lmpcmds: Union[List[str], str, Path],
    atom_types: Dict[str, int],
    log_file: Optional[str] = None,
    keep_alive: bool = True,
    lammps_header: Optional[List[str]] = None,
    lammps_name: Optional[str] = None,
):
    """Return a configured ``LAMMPSlib`` ASE calculator.

    Parameters
    ----------
    lmpcmds
        Either a list of raw LAMMPS command strings or a path to a ``.lmp``
        text file whose non-empty, non-comment lines will be used.
    atom_types
        Mapping from ASE element symbol (e.g. ``"C"``) to LAMMPS integer
        type index (1-based). LAMMPSlib requires this because the ``Atoms``
        object has no native concept of LAMMPS types.
    log_file
        Path for the LAMMPS log. ``None`` suppresses it (``/dev/null``
        equivalent inside LAMMPSlib).
    keep_alive
        Keep the LAMMPS instance resident across energy evaluations. Should
        be ``True`` for Monte Carlo throughput.
    lammps_header
        Optional override for the LAMMPS preamble (units, atom_style, etc.).
        Defaults to ``["units real", "atom_style full",
        "atom_modify map array sort 0 0"]`` which matches the TraPPE-style
        inputs used by the reference CO2 Gibbs example.
    lammps_name
        Suffix for the LAMMPS shared library to load (e.g. ``"serial"`` to
        load ``liblammps_serial.{so,dylib}``). Useful on systems where only
        a suffixed build of LAMMPS is installed (Homebrew etc.).
    """
    from ase.calculators.lammpslib import LAMMPSlib

    if isinstance(lmpcmds, (str, Path)):
        cmds = _read_lmp_file(lmpcmds)
    else:
        cmds = list(lmpcmds)

    if lammps_header is None:
        lammps_header = [
            "units real",
            "atom_style full",
            "atom_modify map array sort 0 0",
        ]

    kwargs = dict(
        lmpcmds=cmds,
        atom_types=atom_types,
        keep_alive=keep_alive,
        lammps_header=lammps_header,
    )
    if log_file is not None:
        kwargs["log_file"] = log_file
    if lammps_name is not None:
        kwargs["lammps_name"] = lammps_name

    return LAMMPSlib(**kwargs)
