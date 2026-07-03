"""
Unit tests for TMMC analysis utilities.
"""
import numpy as np
import pytest
from ase.units import bar, kB

from mlip_mc.src.tmmc_analysis import (
    normalize_lnPi,
    lnPi_from_collection,
    reweight_lnPi,
    mean_N,
    find_barrier,
    basin_averages,
    fugacity_from_pressure,
    compute_isotherm,
    uptake_mol_per_kg,
)


def _poisson_lnPi(mean, N_grid):
    """ln Pi of an ideal-gas (Poisson) distribution with the given mean."""
    from scipy.special import gammaln
    N = np.asarray(N_grid, dtype=float)
    lnPi = N * np.log(mean) - gammaln(N + 1)
    return normalize_lnPi(lnPi)


class TestNormalizeLnPi:

    def test_sums_to_one(self):
        lnPi = normalize_lnPi(np.array([0.0, 1.0, 2.0, -3.0]))
        assert np.exp(lnPi).sum() == pytest.approx(1.0)

    def test_invariant_to_shift(self):
        x = np.array([0.5, -1.0, 2.0])
        np.testing.assert_allclose(normalize_lnPi(x), normalize_lnPi(x + 100.0))


class TestLnPiFromCollection:

    def test_detailed_balance_reconstruction(self):
        # Build a synthetic collection matrix consistent with a known
        # distribution: Pi(i) * P(i -> i+1) = Pi(i+1) * P(i+1 -> i)
        lnPi_true = normalize_lnPi(np.array([0.0, 1.0, 1.5, 0.5]))
        M = len(lnPi_true)
        p_up = np.full(M, 0.2)
        p_up[-1] = 0.0
        p_down = np.zeros(M)
        for i in range(M - 1):
            p_down[i + 1] = p_up[i] * np.exp(lnPi_true[i] - lnPi_true[i + 1])
        # Each state's row scaled by a different visit count, which must
        # not affect the reconstruction.
        attempts = np.array([1000.0, 2000.0, 500.0, 3000.0])
        C_up = attempts * p_up
        C_down = attempts * p_down
        C_stay = attempts * (1 - p_up - p_down)

        lnPi = lnPi_from_collection(C_down, C_stay, C_up)
        np.testing.assert_allclose(lnPi, lnPi_true, atol=1e-10)

    def test_unvisited_states_flat(self):
        # No transitions recorded at all: distribution should be flat
        M = 5
        lnPi = lnPi_from_collection(np.zeros(M), np.zeros(M), np.zeros(M))
        np.testing.assert_allclose(lnPi, np.full(M, -np.log(M)))

    def test_partial_data_carries_forward(self):
        # Only the 0->1 transition has data; the rest stays flat
        C_down = np.array([0.0, 10.0, 0.0, 0.0])
        C_stay = np.array([90.0, 90.0, 0.0, 0.0])
        C_up = np.array([10.0, 0.0, 0.0, 0.0])
        lnPi = lnPi_from_collection(C_down, C_stay, C_up)
        # P(0->1) = 0.1, P(1->0) = 0.1 -> Pi(1) = Pi(0); all flat
        np.testing.assert_allclose(lnPi, np.full(4, -np.log(4)), atol=1e-12)


class TestReweighting:

    def test_reweight_identity(self):
        lnPi = normalize_lnPi(np.array([0.0, 2.0, 1.0]))
        N_grid = np.arange(3)
        np.testing.assert_allclose(
            reweight_lnPi(lnPi, N_grid, 1.0, 1.0), lnPi)

    def test_reweight_linearity(self):
        lnPi = normalize_lnPi(np.random.RandomState(1).rand(10))
        N_grid = np.arange(10)
        # Reweighting f_ref -> f1 -> f2 equals direct f_ref -> f2
        step1 = reweight_lnPi(lnPi, N_grid, 2.0, 1.0)
        step2 = reweight_lnPi(step1, N_grid, 6.0, 2.0)
        direct = reweight_lnPi(lnPi, N_grid, 6.0, 1.0)
        np.testing.assert_allclose(step2, direct, atol=1e-12)

    def test_poisson_reweight_mean(self):
        # An ideal-gas Pi(N) at fugacity f has mean beta*f*V. Doubling f
        # must double the mean.
        N_grid = np.arange(200)
        lnPi = _poisson_lnPi(10.0, N_grid)
        assert mean_N(lnPi, N_grid) == pytest.approx(10.0, rel=1e-6)
        lnPi2 = reweight_lnPi(lnPi, N_grid, 2.0, 1.0)
        assert mean_N(lnPi2, N_grid) == pytest.approx(20.0, rel=1e-6)


class TestHysteresis:

    def _double_well(self, N_grid, n1=10.0, n2=60.0, w1=25.0, w2=25.0, shift=0.0):
        """Two Gaussian basins in ln-space with adjustable relative weight."""
        N = np.asarray(N_grid, dtype=float)
        pi = np.exp(-(N - n1) ** 2 / w1) + np.exp(shift) * np.exp(-(N - n2) ** 2 / w2)
        return normalize_lnPi(np.log(pi))

    def test_find_barrier_bimodal(self):
        N_grid = np.arange(80)
        lnPi = self._double_well(N_grid)
        i_barrier = find_barrier(lnPi)
        assert i_barrier is not None
        assert 10 < i_barrier < 60

    def test_find_barrier_unimodal(self):
        N_grid = np.arange(80)
        lnPi = normalize_lnPi(-(N_grid - 30.0) ** 2 / 50.0)
        assert find_barrier(lnPi) is None

    def test_basin_averages(self):
        N_grid = np.arange(80)
        lnPi = self._double_well(N_grid)
        i_barrier = find_barrier(lnPi)
        basins = basin_averages(lnPi, N_grid, i_barrier)
        assert basins['N_low'] == pytest.approx(10.0, abs=0.5)
        assert basins['N_high'] == pytest.approx(60.0, abs=0.5)

    def test_stable_branch_follows_weight(self):
        N_grid = np.arange(80)
        lnPi_low = self._double_well(N_grid, shift=-5.0)   # low basin favored
        lnPi_high = self._double_well(N_grid, shift=+5.0)  # high basin favored
        i_b = find_barrier(lnPi_low)
        b = basin_averages(lnPi_low, N_grid, i_b)
        assert b['lnW_low'] > b['lnW_high']
        i_b = find_barrier(lnPi_high)
        b = basin_averages(lnPi_high, N_grid, i_b)
        assert b['lnW_high'] > b['lnW_low']


class TestIsotherm:

    def test_ideal_gas_isotherm(self):
        # Poisson lnPi at reference fugacity: reweighted mean must scale
        # linearly with pressure (ideal gas, no molecule -> f = P).
        T = 300.0
        f_ref = 1.0 * bar
        mean_ref = 20.0
        N_grid = np.arange(400)
        lnPi_ref = _poisson_lnPi(mean_ref, N_grid)

        pressures = [0.5, 1.0, 2.0]
        result = compute_isotherm(lnPi_ref, N_grid, T, pressures, f_ref)
        np.testing.assert_allclose(
            result['mean_N'], [10.0, 20.0, 40.0], rtol=1e-5)
        # Unimodal: no hysteresis branches, no transition pressure
        assert result['N_low'] == [None, None, None]
        assert result['transition_pressure_bar'] is None

    def test_transition_pressure_bimodal(self):
        # Construct lnPi_ref such that reweighting shifts weight from the
        # low to the high basin as pressure increases; the transition
        # pressure is where both basins have equal weight.
        T = 300.0
        f_ref = 1.0 * bar
        N_grid = np.arange(120)
        N = N_grid.astype(float)
        # Equal-weight double well at f_ref -> transition at exactly 1 bar
        lnPi_ref = normalize_lnPi(np.log(
            np.exp(-(N - 10.0) ** 2 / 20.0) + np.exp(-(N - 80.0) ** 2 / 20.0)))

        pressures = list(np.linspace(0.5, 2.0, 16))
        result = compute_isotherm(lnPi_ref, N_grid, T, pressures, f_ref)
        # At low P the low basin dominates; at high P the high basin
        assert result['stable_branch'][0] == 'low'
        assert result['stable_branch'][-1] == 'high'
        assert result['transition_pressure_bar'] == pytest.approx(1.0, rel=1e-3)


class TestFugacity:

    def test_ideal_fallback(self):
        assert fugacity_from_pressure(300.0, 2.5) == 2.5
        assert fugacity_from_pressure(300.0, 2.5, molecule=None) == 2.5

    def test_preos_co2_below_ideal(self):
        # At 10 bar / 300 K, CO2 fugacity is slightly below pressure
        P = 10.0 * bar
        f = fugacity_from_pressure(300.0, P, molecule='CO2')
        assert 0.9 * P < f < P

    def test_unknown_molecule_fallback(self):
        assert fugacity_from_pressure(300.0, 1.0, molecule='NOTAMOLECULE') == 1.0


class TestUptakeConversion:

    def test_mol_per_kg(self):
        # 10 molecules in a 5000 amu (g/mol) framework = 2 mol/kg
        assert uptake_mol_per_kg(10, 5000.0) == pytest.approx(2.0)
