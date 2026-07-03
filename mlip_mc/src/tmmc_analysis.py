"""
Analysis utilities for Transition-Matrix Monte Carlo (TMMC) simulations.

Pure numerical functions operating on the collection matrix and the
macrostate probability distribution ln Pi(N) produced by MLP_TMMC:

- reconstruction of ln Pi(N) from the collection matrix
- histogram reweighting of ln Pi(N) to arbitrary fugacity
- adsorption isotherms over a pressure grid
- hysteresis analysis of bimodal distributions (metastable adsorption /
  desorption branches and the equilibrium transition pressure)

None of these functions require an ASE calculator.
"""
import numpy as np
from typing import Any, Dict, List, Optional

from ase.units import bar

from .utilities import PREOS


def _logsumexp(x: np.ndarray) -> float:
    """Numerically stable log(sum(exp(x)))."""
    x = np.asarray(x, dtype=float)
    m = np.max(x)
    if not np.isfinite(m):
        return m
    return float(m + np.log(np.sum(np.exp(x - m))))


def normalize_lnPi(lnPi: np.ndarray) -> np.ndarray:
    """Normalize ln Pi so that sum(exp(lnPi)) = 1."""
    lnPi = np.asarray(lnPi, dtype=float)
    return lnPi - _logsumexp(lnPi)


def lnPi_from_collection(
    C_down: np.ndarray,
    C_stay: np.ndarray,
    C_up: np.ndarray
) -> np.ndarray:
    """
    Reconstruct ln Pi(N) from the TMMC collection matrix.

    Uses the detailed-balance relation (Errington, J. Chem. Phys. 2003):

        ln Pi(N+1) = ln Pi(N) + ln P(N -> N+1) - ln P(N+1 -> N)

    where P(N -> N') = C(N -> N') / sum_k C(N -> k). Macrostates with
    no recorded transitions carry the previous value forward (flat
    extension), which leaves unvisited regions unbiased.

    Parameters
    ----------
    C_down, C_stay, C_up : array_like
        Collection-matrix entries for transitions N -> N-1, N -> N and
        N -> N+1, indexed by macrostate.

    Returns
    -------
    np.ndarray
        Normalized ln Pi over the same macrostate grid.
    """
    C_down = np.asarray(C_down, dtype=float)
    C_stay = np.asarray(C_stay, dtype=float)
    C_up = np.asarray(C_up, dtype=float)
    total = C_down + C_stay + C_up

    M = len(C_stay)
    lnPi = np.zeros(M)
    for i in range(M - 1):
        if C_up[i] > 0 and C_down[i + 1] > 0 and total[i] > 0 and total[i + 1] > 0:
            delta = (np.log(C_up[i]) - np.log(total[i])) - (
                np.log(C_down[i + 1]) - np.log(total[i + 1]))
        else:
            # No information about this transition yet: flat extension
            delta = 0.0
        lnPi[i + 1] = lnPi[i] + delta
    return normalize_lnPi(lnPi)


def reweight_lnPi(
    lnPi_ref: np.ndarray,
    N_grid: np.ndarray,
    f: float,
    f_ref: float
) -> np.ndarray:
    """
    Reweight ln Pi from the reference fugacity to fugacity ``f``.

        ln Pi_f(N) = ln Pi_fref(N) + N * ln(f / f_ref)

    Parameters
    ----------
    lnPi_ref : array_like
        ln Pi at the reference fugacity.
    N_grid : array_like
        Macrostate values N corresponding to each entry of lnPi_ref.
    f : float
        Target fugacity (same units as f_ref).
    f_ref : float
        Reference fugacity the distribution was sampled at.

    Returns
    -------
    np.ndarray
        Normalized ln Pi at fugacity ``f``.
    """
    lnPi_ref = np.asarray(lnPi_ref, dtype=float)
    N_grid = np.asarray(N_grid, dtype=float)
    return normalize_lnPi(lnPi_ref + N_grid * np.log(f / f_ref))


def mean_N(lnPi: np.ndarray, N_grid: np.ndarray) -> float:
    """Average macrostate <N> under the (normalized) distribution ln Pi."""
    lnPi = normalize_lnPi(lnPi)
    return float(np.sum(np.asarray(N_grid, dtype=float) * np.exp(lnPi)))


def find_barrier(lnPi: np.ndarray, min_depth: float = 1.0) -> Optional[int]:
    """
    Locate the free-energy barrier separating two basins in ln Pi.

    Finds the interior point with the largest depth below the smaller of
    the maxima on either side. Returns None if the distribution is
    unimodal (no interior point at least ``min_depth`` (in kT units,
    i.e. ln-probability units) below both flanking maxima).

    Parameters
    ----------
    lnPi : array_like
        Macrostate log-probability distribution.
    min_depth : float, optional
        Minimum barrier depth (default: 1.0 kT).

    Returns
    -------
    int or None
        Index of the barrier macrostate, or None if unimodal.
    """
    lnPi = np.asarray(lnPi, dtype=float)
    M = len(lnPi)
    if M < 3:
        return None

    left_max = np.maximum.accumulate(lnPi)
    right_max = np.maximum.accumulate(lnPi[::-1])[::-1]

    best_i = None
    best_depth = min_depth
    for i in range(1, M - 1):
        depth = min(left_max[i - 1], right_max[i + 1]) - lnPi[i]
        if depth > best_depth:
            best_depth = depth
            best_i = i
    return best_i


def basin_averages(
    lnPi: np.ndarray,
    N_grid: np.ndarray,
    i_barrier: int
) -> Dict[str, float]:
    """
    Basin-restricted averages of a bimodal ln Pi split at ``i_barrier``.

    The low-N basin covers indices [0, i_barrier], the high-N basin
    (i_barrier, end]. These correspond to the metastable states sampled
    on the adsorption (low) and desorption (high) branches.

    Returns
    -------
    dict
        'N_low', 'N_high' : basin-restricted <N>
        'lnW_low', 'lnW_high' : log basin weights (log partition sums)
    """
    lnPi = normalize_lnPi(lnPi)
    N_grid = np.asarray(N_grid, dtype=float)
    low = slice(0, i_barrier + 1)
    high = slice(i_barrier + 1, len(lnPi))
    lnW_low = _logsumexp(lnPi[low])
    lnW_high = _logsumexp(lnPi[high])
    N_low = float(np.sum(N_grid[low] * np.exp(lnPi[low] - lnW_low)))
    N_high = float(np.sum(N_grid[high] * np.exp(lnPi[high] - lnW_high)))
    return {
        'N_low': N_low,
        'N_high': N_high,
        'lnW_low': lnW_low,
        'lnW_high': lnW_high,
    }


def fugacity_from_pressure(T: float, P: float, molecule: Optional[str] = None) -> float:
    """
    Convert pressure to fugacity via the Peng-Robinson EOS.

    Falls back to the ideal-gas result (f = P) if the molecule is not
    provided or not present in the EOS data file, mirroring the GCMC
    entry-point behavior.

    Parameters
    ----------
    T : float
        Temperature in Kelvin.
    P : float
        Pressure in ASE units (eV/A^3).
    molecule : str, optional
        Compound name in critical_acentric.csv (e.g. 'CO2', 'H2O').
    """
    if molecule:
        try:
            eos = PREOS.from_name(molecule)
            return float(eos.calculate_fugacity(T, P))
        except Exception:
            return float(P)
    return float(P)


def compute_isotherm(
    lnPi_ref: np.ndarray,
    N_grid: np.ndarray,
    T: float,
    pressures_bar: List[float],
    f_ref: float,
    molecule: Optional[str] = None,
    min_barrier_depth: float = 1.0,
) -> Dict[str, Any]:
    """
    Compute the full adsorption isotherm (with hysteresis branches) by
    reweighting ln Pi over a pressure grid.

    At each pressure the distribution is reweighted and, if bimodal, the
    basin-restricted averages give the two metastable branches:
    the low-N basin is the state reached on adsorption (increasing P),
    the high-N basin the state reached on desorption (decreasing P).
    The equilibrium transition pressure is where both basins carry equal
    weight.

    Parameters
    ----------
    lnPi_ref : array_like
        ln Pi sampled at the reference fugacity ``f_ref``.
    N_grid : array_like
        Macrostate values N.
    T : float
        Temperature in Kelvin.
    pressures_bar : list of float
        Pressures (in bar) to evaluate the isotherm at.
    f_ref : float
        Reference fugacity in ASE units (eV/A^3).
    molecule : str, optional
        Compound name for the Peng-Robinson fugacity conversion.
    min_barrier_depth : float, optional
        Minimum barrier depth (kT) to count a distribution as bimodal.

    Returns
    -------
    dict
        'pressure_bar', 'fugacity', 'mean_N' : equilibrium isotherm
        'N_low', 'N_high' : metastable branch loadings (None where the
        distribution is unimodal at that pressure)
        'stable_branch' : 'low' or 'high' (which basin dominates)
        'transition_pressure_bar' : equilibrium step pressure, or None
    """
    result: Dict[str, Any] = {
        'pressure_bar': [float(p) for p in pressures_bar],
        'fugacity': [],
        'mean_N': [],
        'N_low': [],
        'N_high': [],
        'stable_branch': [],
    }
    lnW_diffs = []
    for P_bar in pressures_bar:
        f = fugacity_from_pressure(T, float(P_bar) * bar, molecule)
        lnPi = reweight_lnPi(lnPi_ref, N_grid, f, f_ref)
        result['fugacity'].append(f)
        result['mean_N'].append(mean_N(lnPi, N_grid))

        i_barrier = find_barrier(lnPi, min_depth=min_barrier_depth)
        if i_barrier is None:
            result['N_low'].append(None)
            result['N_high'].append(None)
            result['stable_branch'].append(None)
            lnW_diffs.append(None)
        else:
            basins = basin_averages(lnPi, N_grid, i_barrier)
            result['N_low'].append(basins['N_low'])
            result['N_high'].append(basins['N_high'])
            diff = basins['lnW_high'] - basins['lnW_low']
            result['stable_branch'].append('high' if diff > 0 else 'low')
            lnW_diffs.append(diff)

    result['transition_pressure_bar'] = _transition_pressure(
        lnPi_ref, N_grid, T, result['pressure_bar'], lnW_diffs, f_ref,
        molecule, min_barrier_depth)
    return result


def _transition_pressure(
    lnPi_ref: np.ndarray,
    N_grid: np.ndarray,
    T: float,
    pressures_bar: List[float],
    lnW_diffs: List[Optional[float]],
    f_ref: float,
    molecule: Optional[str],
    min_barrier_depth: float,
) -> Optional[float]:
    """Bisect for the pressure where both basins have equal weight."""
    # Find a bracketing pair of bimodal pressures with a sign change
    bracket = None
    for i in range(len(pressures_bar) - 1):
        d1, d2 = lnW_diffs[i], lnW_diffs[i + 1]
        if d1 is not None and d2 is not None and d1 * d2 < 0:
            bracket = (pressures_bar[i], pressures_bar[i + 1])
            break
    if bracket is None:
        return None

    def weight_diff(P_bar: float) -> Optional[float]:
        f = fugacity_from_pressure(T, P_bar * bar, molecule)
        lnPi = reweight_lnPi(lnPi_ref, N_grid, f, f_ref)
        i_barrier = find_barrier(lnPi, min_depth=min_barrier_depth)
        if i_barrier is None:
            return None
        basins = basin_averages(lnPi, N_grid, i_barrier)
        return basins['lnW_high'] - basins['lnW_low']

    P_lo, P_hi = bracket
    d_lo = weight_diff(P_lo)
    for _ in range(60):
        P_mid = 0.5 * (P_lo + P_hi)
        d_mid = weight_diff(P_mid)
        if d_mid is None:
            break
        if d_lo is not None and d_lo * d_mid <= 0:
            P_hi = P_mid
        else:
            P_lo, d_lo = P_mid, d_mid
        if abs(P_hi - P_lo) < 1e-10 * max(1.0, abs(P_hi)):
            break
    return float(0.5 * (P_lo + P_hi))


def uptake_mol_per_kg(n_molecules: float, framework_mass_amu: float) -> float:
    """
    Convert a loading in molecules per unit cell to mol/kg of framework.

    Parameters
    ----------
    n_molecules : float
        Number of adsorbed molecules per simulation cell.
    framework_mass_amu : float
        Total framework mass in amu (g/mol) per simulation cell.
    """
    return float(n_molecules) * 1000.0 / float(framework_mass_amu)
