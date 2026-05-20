"""Classical-potential backend built on ASE's LAMMPSlib.

Exposes a single factory, ``build_lammps_calculator``, that returns an ASE
calculator backed by an in-process LAMMPS instance. The caller supplies raw
LAMMPS commands verbatim, so any pair_style / kspace_style / set charge
sequence supported by LAMMPS is available without a translation layer.

The factory returns an ``LAMMPSlib`` (or a thin subclass) configured for the
chosen options. When ``create_box_extra`` is supplied, a subclass is used
that injects extra arguments into the auto-generated ``create_box`` line so
bonded force fields (``bond_style ... ; create_bonds ...``) and 1-2 / 1-3
exclusions (``special_bonds ...``) can be set up via subsequent ``lmpcmds``.
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


def _make_bonded_lammpslib_class():
    """Create a thin LAMMPSlib subclass that supports extra create_box args.

    Constructed lazily (and only once per process) so importing this module
    doesn't require ASE to be installed.
    """
    from ase.calculators import lammpslib as _ll

    class BondedLAMMPSlib(_ll.LAMMPSlib):
        """LAMMPSlib that injects extra args into the auto ``create_box`` line.

        Stock LAMMPSlib hard-codes ``create_box {N} cell`` with no slot for
        ``bond/types`` / ``extra/bond/per/atom`` / ``extra/special/per/atom``.
        Without those, you can't declare bonds after atom creation, which means
        no ``special_bonds`` exclusions and no realistic rigid-molecule force
        field. This subclass intercepts the LAMMPS python constructor for the
        duration of ``start_lammps`` and wraps the resulting object so any
        ``create_box ... cell`` command is rewritten to
        ``create_box ... cell <extra>``. Every other ``self.lmp.*`` call is
        passed through unchanged.

        ``create_box`` is issued from ``initialise_lammps`` (which runs
        *after* ``start_lammps``), so the proxy must remain installed past
        the end of ``start_lammps``. Per-command overhead is one string
        startswith/endswith check; non-command access goes through
        ``__getattr__`` which forwards directly to the bare LAMMPS handle.
        """

        def __init__(
            self,
            *args,
            create_box_extra: str = "",
            intra_bonds: Optional[dict] = None,
            **kwargs,
        ):
            super().__init__(*args, **kwargs)
            self._create_box_extra = create_box_extra
            # intra_bonds = {"atoms_per_molecule": int,
            #                "bond_pairs": [[i, j], ...] (0-indexed within mol),
            #                "bond_type": int (default 1)}
            self._intra_bonds = intra_bonds
            self._bonded_atom_count = 0  # how many atoms have been bonded so far

        def start_lammps(self):
            extra = self._create_box_extra
            if not extra:
                return super().start_lammps()

            # Upstream LAMMPSlib imports the lammps constructor lazily inside
            # start_lammps via ``from lammps import lammps``, so we patch the
            # symbol on the ``lammps`` module itself for the duration of the
            # call. The freshly imported binding inside the function picks up
            # the patched name.
            import lammps as _lammps_mod
            real_ctor = _lammps_mod.lammps

            class _CreateBoxProxy:
                __slots__ = ("_lmp",)

                def __init__(self, lmp):
                    self._lmp = lmp

                def command(self, cmd):
                    s = cmd.strip()
                    if s.startswith("create_box ") and s.endswith(" cell"):
                        cmd = f"{s} {extra}"
                    return self._lmp.command(cmd)

                def __getattr__(self, name):
                    return getattr(self._lmp, name)

            def _patched_ctor(*a, **kw):
                return _CreateBoxProxy(real_ctor(*a, **kw))

            _lammps_mod.lammps = _patched_ctor
            try:
                super().start_lammps()
            finally:
                _lammps_mod.lammps = real_ctor

        # ----- intra-molecular bond plumbing -----

        def _emit_intra_bonds(self, first_atom_id: int, n_atoms: int) -> None:
            """Issue ``create_bonds single/bond`` for every molecule whose atoms
            lie in the inclusive ID range ``[first_atom_id, first_atom_id +
            n_atoms - 1]``. Atom IDs are assumed to be sequential and grouped
            by molecule (which holds for both the initial build and the random
            insertions LAMMPSlib does on grow-rebuilds)."""
            cfg = self._intra_bonds
            if not cfg or n_atoms <= 0:
                return
            atoms_per_mol = int(cfg["atoms_per_molecule"])
            bond_pairs = cfg["bond_pairs"]
            bond_type = int(cfg.get("bond_type", 1))
            if n_atoms % atoms_per_mol != 0:
                raise RuntimeError(
                    f"BondedLAMMPSlib: atom count delta ({n_atoms}) is not a "
                    f"multiple of atoms_per_molecule ({atoms_per_mol})."
                )
            n_mol = n_atoms // atoms_per_mol
            for m in range(n_mol):
                base = first_atom_id + m * atoms_per_mol
                for (i, j) in bond_pairs:
                    self.lmp.command(
                        f"create_bonds single/bond {bond_type} "
                        f"{base + int(i)} {base + int(j)}"
                    )

        def initialise_lammps(self, atoms):
            super().initialise_lammps(atoms)
            if self._intra_bonds and self._bonded_atom_count == 0:
                n = len(atoms)
                self._emit_intra_bonds(1, n)
                self._bonded_atom_count = n

        def rebuild(self, atoms):
            n_before = self._bonded_atom_count
            super().rebuild(atoms)
            n_after = len(atoms)
            if not self._intra_bonds:
                self._bonded_atom_count = n_after
                return
            if n_after > n_before:
                self._emit_intra_bonds(n_before + 1, n_after - n_before)
            # When n_after < n_before, LAMMPSlib already deleted the trailing
            # atoms (and their bonds) via delete_atoms; nothing to redo.
            self._bonded_atom_count = n_after

    return BondedLAMMPSlib


def build_lammps_calculator(
    lmpcmds: Union[List[str], str, Path],
    atom_types: Dict[str, int],
    log_file: Optional[str] = None,
    keep_alive: bool = True,
    lammps_header: Optional[List[str]] = None,
    lammps_name: Optional[str] = None,
    create_box_extra: Optional[str] = None,
    intra_bonds: Optional[dict] = None,
    lammps_threads: Optional[int] = None,
    extra_cmd_args: Optional[List[str]] = None,
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
    create_box_extra
        Extra arguments appended to the auto ``create_box N_types cell ...``
        line, used to enable bonded force fields with intra exclusions, e.g.
        ``"bond/types 1 extra/bond/per/atom 2 extra/special/per/atom 4"``.
        When provided, a thin LAMMPSlib subclass is used to inject these
        args; the runtime cost is zero after initialisation.
    intra_bonds
        Optional dict describing the bond topology of a single rigid
        molecule template. Keys:

        * ``atoms_per_molecule`` (int): atoms per molecule in input order.
        * ``bond_pairs`` (list of [i, j] pairs, 0-indexed within molecule):
          which atoms are bonded.
        * ``bond_type`` (int, default 1): LAMMPS bond type to assign.

        When set, the calculator declares ``create_bonds single/bond ...``
        for every molecule after each (re)build, so MC moves that grow or
        shrink the system keep the intra-molecular topology consistent.
        Pair with ``special_bonds`` in the user lmpcmds to exclude 1-2 /
        1-3 / 1-4 LJ + Coulomb terms.
    lammps_threads
        Number of OpenMP threads for LAMMPS. When set (>= 1), the LAMMPS
        process is launched with ``-sf omp -pk omp <N>`` so all pair /
        bond / kspace styles use their ``/omp`` variants. The image must
        have been built with ``-D PKG_OPENMP=yes`` (see Dockerfile.lammps).
        Also set the host environment variable ``OMP_NUM_THREADS`` (and
        give the container enough cores via ``--cpus``) for full effect.
    extra_cmd_args
        Additional LAMMPS command-line arguments to append after the
        threading flags. Useful for ``--screen``/``-pk gpu N`` etc.
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

    cmd_args: List[str] = []
    if lammps_threads is not None and int(lammps_threads) >= 1:
        n_thr = int(lammps_threads)
        cmd_args.extend(["-sf", "omp", "-pk", "omp", str(n_thr)])
    if extra_cmd_args:
        cmd_args.extend(list(extra_cmd_args))
    if cmd_args:
        kwargs["extra_cmd_args"] = tuple(cmd_args)

    if create_box_extra or intra_bonds:
        cls = _make_bonded_lammpslib_class()
        if create_box_extra:
            kwargs["create_box_extra"] = create_box_extra
        if intra_bonds:
            kwargs["intra_bonds"] = intra_bonds
        return cls(**kwargs)

    return LAMMPSlib(**kwargs)
