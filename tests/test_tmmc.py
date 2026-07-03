"""
Unit tests for the Transition-Matrix Monte Carlo (TMMC) module.
"""
import json
import os
import pytest
import numpy as np
from unittest.mock import Mock

from mlip_mc.src.tmmc import MLP_TMMC, TMMC_MOVE_PROBABILITIES
from mlip_mc.src.tmmc_analysis import normalize_lnPi, reweight_lnPi, mean_N
from ase import Atoms
from ase.build import molecule
from ase.calculators.calculator import Calculator, all_changes
from ase.data import vdw_radii
from ase.units import bar, kB


class ZeroCalculator(Calculator):
    """ASE calculator returning zero energy, forces, and stress."""
    implemented_properties = ['energy', 'forces', 'stress']

    def calculate(self, atoms=None, properties=['energy'],
                  system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.results['energy'] = 0.0
        self.results['forces'] = np.zeros((len(atoms), 3))
        self.results['stress'] = np.zeros(6)


@pytest.fixture
def mock_model():
    """Mock MLIP model returning zero energy."""
    model = Mock()
    model.get_potential_energy = Mock(return_value=0.0)
    return model


@pytest.fixture
def zero_vdw():
    """VDW radii that never produce overlaps."""
    radii = vdw_radii.copy()
    radii[:] = 0.0
    return radii


@pytest.fixture
def atoms_frame():
    """Simple framework structure."""
    return Atoms('H2', positions=[[0, 0, 0], [1, 0, 0]], cell=[10, 10, 10], pbc=True)


@pytest.fixture
def empty_frame():
    """Empty box for bulk-fluid TMMC."""
    return Atoms(cell=[10, 10, 10], pbc=True)


@pytest.fixture
def atoms_ads():
    """Single-atom adsorbate."""
    return Atoms('He', positions=[[0, 0, 0]])


def make_tmmc(model, frame, ads, radii, out_dir, **kwargs):
    defaults = dict(
        T=300.0,
        P=1.0 * bar,
        fugacity=1.0 * bar,
        device='cpu',
        N_min=0,
        N_max=10,
    )
    defaults.update(kwargs)
    return MLP_TMMC(
        model=model,
        atoms_frame=frame,
        atoms_ads=ads,
        vdw_radii=radii,
        output_dir=str(out_dir),
        **defaults,
    )


class TestInitialization:

    def test_window_and_arrays(self, mock_model, atoms_frame, atoms_ads, zero_vdw, tmp_path):
        tmmc = make_tmmc(mock_model, atoms_frame, atoms_ads, zero_vdw, tmp_path,
                         N_min=2, N_max=12)
        assert tmmc.M == 11
        assert tmmc.C_up.shape == (11,)
        assert tmmc.C_down.shape == (11,)
        assert tmmc.C_stay.shape == (11,)
        assert tmmc.H.shape == (11,)
        assert np.all(tmmc.lnPi == 0.0)
        for move in ('md', 'volume', 'insertion', 'deletion', 'translation', 'rotation'):
            assert move in tmmc.moves

    def test_invalid_window(self, mock_model, atoms_frame, atoms_ads, zero_vdw, tmp_path):
        with pytest.raises(ValueError):
            make_tmmc(mock_model, atoms_frame, atoms_ads, zero_vdw, tmp_path,
                      N_min=5, N_max=5)
        with pytest.raises(ValueError):
            make_tmmc(mock_model, atoms_frame, atoms_ads, zero_vdw, tmp_path,
                      N_min=-1, N_max=5)

    def test_invalid_md_ensemble(self, mock_model, atoms_frame, atoms_ads, zero_vdw, tmp_path):
        with pytest.raises(ValueError):
            make_tmmc(mock_model, atoms_frame, atoms_ads, zero_vdw, tmp_path,
                      md_ensemble='nve')


class TestCollectionMatrix:

    def test_update_C_insertion(self, mock_model, atoms_frame, atoms_ads, zero_vdw, tmp_path):
        tmmc = make_tmmc(mock_model, atoms_frame, atoms_ads, zero_vdw, tmp_path)
        tmmc._update_C(3, +1, 0.4)
        assert tmmc.C_up[3] == pytest.approx(0.4)
        assert tmmc.C_stay[3] == pytest.approx(0.6)
        assert tmmc.C_down[3] == 0.0

    def test_update_C_deletion(self, mock_model, atoms_frame, atoms_ads, zero_vdw, tmp_path):
        tmmc = make_tmmc(mock_model, atoms_frame, atoms_ads, zero_vdw, tmp_path)
        tmmc._update_C(5, -1, 0.9)
        assert tmmc.C_down[5] == pytest.approx(0.9)
        assert tmmc.C_stay[5] == pytest.approx(0.1)

    def test_update_C_boundary(self, mock_model, atoms_frame, atoms_ads, zero_vdw, tmp_path):
        tmmc = make_tmmc(mock_model, atoms_frame, atoms_ads, zero_vdw, tmp_path)
        tmmc._update_C(0, 0, 0.0)
        assert tmmc.C_stay[0] == pytest.approx(1.0)


class TestAcceptanceProbabilities:

    def test_insertion_ln_ratio(self, mock_model, atoms_frame, atoms_ads, zero_vdw, tmp_path):
        tmmc = make_tmmc(mock_model, atoms_frame, atoms_ads, zero_vdw, tmp_path)
        # Zero energy change: ratio = V*beta*f/(N+1)
        N = 4
        expected = np.log(tmmc.V * tmmc.beta * tmmc.fugacity / (N + 1))
        assert tmmc._insertion_ln_ratio(0.0, 0.0, N, tmmc.V) == pytest.approx(expected)
        # Strongly unfavorable energy: -inf
        assert tmmc._insertion_ln_ratio(1000.0, 0.0, N, tmmc.V) == -np.inf

    def test_deletion_ln_ratio(self, mock_model, atoms_frame, atoms_ads, zero_vdw, tmp_path):
        tmmc = make_tmmc(mock_model, atoms_frame, atoms_ads, zero_vdw, tmp_path)
        N = 4
        expected = np.log(N / (tmmc.V * tmmc.beta * tmmc.fugacity))
        assert tmmc._deletion_ln_ratio(0.0, 0.0, N, tmmc.V) == pytest.approx(expected)

    def test_insertion_deletion_detailed_balance(self, mock_model, atoms_frame, atoms_ads, zero_vdw, tmp_path):
        # ln ratio of insertion N->N+1 must be the negative of the ln
        # ratio of deletion N+1->N at equal energies
        tmmc = make_tmmc(mock_model, atoms_frame, atoms_ads, zero_vdw, tmp_path)
        ins = tmmc._insertion_ln_ratio(0.0, 0.0, 4, tmmc.V)
        dele = tmmc._deletion_ln_ratio(0.0, 0.0, 5, tmmc.V)
        assert ins == pytest.approx(-dele)

    def test_accept_biased_deterministic(self, mock_model, atoms_frame, atoms_ads, zero_vdw, tmp_path):
        tmmc = make_tmmc(mock_model, atoms_frame, atoms_ads, zero_vdw, tmp_path)
        # Bias strongly favors target state -> always accept
        tmmc.lnPi = np.zeros(tmmc.M)
        tmmc.lnPi[1] = -100.0  # target has tiny estimated Pi -> bias pushes in
        assert tmmc._accept_biased(0.0, 0, 1) is True
        # ln_ratio -inf -> never accept
        assert tmmc._accept_biased(-np.inf, 0, 1) is False


class TestWindowBoundaries:

    def test_insertion_at_N_max(self, mock_model, atoms_frame, atoms_ads, zero_vdw, tmp_path):
        tmmc = make_tmmc(mock_model, atoms_frame, atoms_ads, zero_vdw, tmp_path,
                         N_min=0, N_max=3)
        tmmc.Z_ads = 3
        atoms = atoms_frame.copy()
        _, _, _, success = tmmc._attempt_insertion(atoms, 0.0, 0.0, 0.0, 0.0)
        assert success is False
        assert tmmc.C_stay[tmmc._state_index(3)] == pytest.approx(1.0)
        assert tmmc.window_rejections == 1
        # No energy evaluation for boundary rejections
        mock_model.get_potential_energy.assert_not_called()

    def test_deletion_at_N_min(self, mock_model, atoms_frame, atoms_ads, zero_vdw, tmp_path):
        tmmc = make_tmmc(mock_model, atoms_frame, atoms_ads, zero_vdw, tmp_path,
                         N_min=0, N_max=3)
        tmmc.Z_ads = 0
        atoms = atoms_frame.copy()
        _, _, _, success = tmmc._attempt_deletion(atoms, 0.0, 0.0, 0.0, 0.0)
        assert success is False
        assert tmmc.C_stay[0] == pytest.approx(1.0)
        assert tmmc.window_rejections == 1


class TestScaleSystem:

    def test_molecule_geometry_preserved(self, mock_model, zero_vdw, tmp_path):
        frame = Atoms('H2', positions=[[1, 1, 1], [2, 1, 1]], cell=[10, 10, 10], pbc=True)
        ads = molecule('H2')
        tmmc = make_tmmc(mock_model, frame, ads, zero_vdw, tmp_path)
        tmmc.Z_ads = 1
        atoms = frame + ads
        scale = 1.1
        scaled = tmmc._scale_system(atoms, scale)
        # Framework scaled affinely
        np.testing.assert_allclose(
            scaled.get_positions()[:2], atoms.get_positions()[:2] * scale)
        # Molecule bond length preserved
        d_before = np.linalg.norm(
            atoms.get_positions()[2] - atoms.get_positions()[3])
        d_after = np.linalg.norm(
            scaled.get_positions()[2] - scaled.get_positions()[3])
        assert d_after == pytest.approx(d_before)
        # Cell scaled
        np.testing.assert_allclose(
            np.array(scaled.get_cell()), np.array(atoms.get_cell()) * scale)


class TestMDMove:

    def test_nvt_md_move(self, zero_vdw, tmp_path):
        frame = Atoms('H2', positions=[[0, 0, 0], [1, 0, 0]], cell=[10, 10, 10], pbc=True)
        ads = Atoms('He', positions=[[0, 0, 0]])
        tmmc = make_tmmc(ZeroCalculator(), frame, ads, zero_vdw, tmp_path,
                         md_steps=5, md_timestep=0.5)
        atoms = frame.copy()
        atoms_new, e_new, success = tmmc._move_md(atoms, 0.0)
        assert success is True
        assert len(atoms_new) == len(atoms)
        assert e_new == pytest.approx(0.0)
        assert tmmc.moves['md']['accepted'] == 1

    def test_volume_move_updates_V(self, zero_vdw, tmp_path):
        np.random.seed(3)
        frame = Atoms('H2', positions=[[0, 0, 0], [1, 0, 0]], cell=[10, 10, 10], pbc=True)
        ads = Atoms('He', positions=[[0, 0, 0]])
        tmmc = make_tmmc(ZeroCalculator(), frame, ads, zero_vdw, tmp_path,
                         external_pressure=0.0, max_delta_lnV=0.05)
        V_init = tmmc.V
        atoms = frame.copy()
        # With zero energy and zero external pressure, acceptance is
        # driven by the Jacobian only; run several moves and require at
        # least one acceptance with a consistent volume update.
        accepted = False
        for _ in range(20):
            atoms, e, success = tmmc._move_volume(atoms, 0.0)
            if success:
                accepted = True
                assert tmmc.V == pytest.approx(np.linalg.det(np.array(atoms.get_cell())))
        assert accepted
        assert tmmc.V != pytest.approx(V_init)


class TestRestart:

    def test_restart_round_trip(self, mock_model, atoms_frame, atoms_ads, zero_vdw, tmp_path):
        tmmc = make_tmmc(mock_model, atoms_frame, atoms_ads, zero_vdw, tmp_path,
                         N_min=0, N_max=5)
        tmmc.Z_ads = 2
        tmmc.C_up[:] = np.arange(6, dtype=float)
        tmmc.C_down[:] = np.arange(6, dtype=float) * 0.5
        tmmc.C_stay[:] = 10.0
        tmmc.H[:] = np.arange(6)
        tmmc._recompute_lnPi()
        lnPi_saved = tmmc.lnPi.copy()
        atoms = atoms_frame + atoms_ads + atoms_ads
        tmmc._save_restart(atoms, 1234)

        tmmc2 = make_tmmc(mock_model, atoms_frame, atoms_ads, zero_vdw, tmp_path,
                          N_min=0, N_max=5)
        atoms_loaded = tmmc2._load_restart_info()
        assert atoms_loaded is not None
        assert len(atoms_loaded) == len(atoms)
        assert tmmc2.Z_ads == 2
        assert tmmc2._restart_steps_completed == 1234
        np.testing.assert_allclose(tmmc2.C_up, tmmc.C_up)
        np.testing.assert_allclose(tmmc2.C_down, tmmc.C_down)
        np.testing.assert_allclose(tmmc2.C_stay, tmmc.C_stay)
        np.testing.assert_allclose(tmmc2.lnPi, lnPi_saved)
        np.testing.assert_array_equal(tmmc2.H, tmmc.H)

    def test_restart_window_mismatch_rejected(self, mock_model, atoms_frame, atoms_ads, zero_vdw, tmp_path):
        tmmc = make_tmmc(mock_model, atoms_frame, atoms_ads, zero_vdw, tmp_path,
                         N_min=0, N_max=5)
        tmmc._save_restart(atoms_frame.copy(), 100)

        tmmc2 = make_tmmc(mock_model, atoms_frame, atoms_ads, zero_vdw, tmp_path,
                          N_min=0, N_max=8)
        assert tmmc2._load_restart_info() is None
        assert tmmc2._restart_steps_completed == 0


class TestIdealGasValidation:
    """
    Validation gate: for a non-interacting (ideal gas) system, the
    grand-canonical distribution is Poisson with mean beta*f*V. TMMC
    must reproduce it, and reweighting must reproduce the linear
    ideal-gas isotherm <N>(f) = beta*f*V.
    """

    def test_lnPi_matches_poisson(self, mock_model, empty_frame, atoms_ads, zero_vdw, tmp_path):
        np.random.seed(42)
        from scipy.special import gammaln

        T = 300.0
        V = 1000.0
        beta = 1.0 / (kB * T)
        mean_target = 4.0
        f_ref = mean_target / (beta * V)
        N_max = 15

        tmmc = make_tmmc(
            mock_model, empty_frame, atoms_ads, zero_vdw, tmp_path,
            T=T, P=f_ref, fugacity=f_ref,
            N_min=0, N_max=N_max,
            bias_update_interval=2000,
            checkpoint_interval=10**9,
            # Insertion/deletion moves only: fastest C accumulation
            move_probabilities={
                'md': 0.0, 'volume': 0.0,
                'insertion': 0.5, 'deletion': 1.0,
                'translation': 1.0, 'rotation': 1.0,
            },
        )
        tmmc.run(N=20000)

        N_grid = np.arange(N_max + 1)
        lnPi_exact = normalize_lnPi(
            N_grid * np.log(mean_target) - gammaln(N_grid + 1.0))

        # The sampled distribution should match the Poisson closely where
        # it carries weight
        significant = lnPi_exact > -12
        np.testing.assert_allclose(
            tmmc.lnPi[significant], lnPi_exact[significant], atol=0.5)

        # Reweighted isotherm: <N>(2*f_ref) = 2 * mean_target
        lnPi_2f = reweight_lnPi(tmmc.lnPi, N_grid, 2 * f_ref, f_ref)
        assert mean_N(lnPi_2f, N_grid) == pytest.approx(2 * mean_target, rel=0.15)
        assert mean_N(tmmc.lnPi, N_grid) == pytest.approx(mean_target, rel=0.15)

    def test_flat_histogram_coverage(self, mock_model, empty_frame, atoms_ads, zero_vdw, tmp_path):
        # The bias must drive the walker across the entire window, even
        # far above the natural mean loading (<N> = 1)
        np.random.seed(7)
        T = 300.0
        V = 1000.0
        beta = 1.0 / (kB * T)
        f_ref = 1.0 / (beta * V)
        N_max = 12

        tmmc = make_tmmc(
            mock_model, empty_frame, atoms_ads, zero_vdw, tmp_path,
            T=T, P=f_ref, fugacity=f_ref,
            N_min=0, N_max=N_max,
            bias_update_interval=1000,
            checkpoint_interval=10**9,
            move_probabilities={
                'md': 0.0, 'volume': 0.0,
                'insertion': 0.5, 'deletion': 1.0,
                'translation': 1.0, 'rotation': 1.0,
            },
        )
        tmmc.run(N=15000)
        # Without bias, states N >= 8 have Poisson(1) probability < 1e-5;
        # the flat-histogram bias must still visit all of them
        assert np.all(tmmc.H > 0)

    def test_run_with_md_moves(self, empty_frame, atoms_ads, zero_vdw, tmp_path):
        # Full run() loop with hybrid MC/MD moves enabled (flexible
        # system, zero-energy calculator with forces)
        np.random.seed(5)
        T = 300.0
        beta = 1.0 / (kB * T)
        f_ref = 2.0 / (beta * 1000.0)
        tmmc = make_tmmc(
            ZeroCalculator(), empty_frame, atoms_ads, zero_vdw, tmp_path,
            T=T, P=f_ref, fugacity=f_ref,
            N_min=0, N_max=5,
            bias_update_interval=200,
            checkpoint_interval=10**9,
            md_steps=3, md_timestep=0.5,
            move_probabilities={
                'md': 0.2, 'volume': 0.4,
                'insertion': 0.6, 'deletion': 0.8,
                'translation': 0.9, 'rotation': 1.0,
            },
        )
        tmmc.run(N=500)
        assert tmmc.moves['md']['attempted'] > 0
        assert tmmc.moves['volume']['attempted'] > 0
        # MD moves on the empty box (Z_ads = 0, no atoms) are skipped,
        # accepted ones only when atoms exist
        assert tmmc.moves['md']['accepted'] <= tmmc.moves['md']['attempted']
        assert np.all(tmmc.H[:2] > 0)

    def test_results_files_written(self, mock_model, empty_frame, atoms_ads, zero_vdw, tmp_path):
        np.random.seed(1)
        T = 300.0
        beta = 1.0 / (kB * T)
        f_ref = 2.0 / (beta * 1000.0)
        tmmc = make_tmmc(
            mock_model, empty_frame, atoms_ads, zero_vdw, tmp_path,
            T=T, P=f_ref, fugacity=f_ref,
            N_min=0, N_max=6, bias_update_interval=500,
            checkpoint_interval=10**9,
        )
        tmmc.run(N=1000)

        results_file = tmp_path / f"results_tmmc_{f_ref/bar:.4f}bar.json"
        lnPi_file = tmp_path / f"lnPi_{f_ref/bar:.4f}bar.json"
        assert results_file.exists()
        assert lnPi_file.exists()
        with open(results_file) as f:
            data = json.load(f)
        assert data['N_min'] == 0
        assert data['N_max'] == 6
        assert len(data['lnPi']) == 7
        assert len(data['N_grid']) == 7

        # Restart from the results should report nothing to do
        tmmc2 = make_tmmc(
            mock_model, empty_frame, atoms_ads, zero_vdw, tmp_path,
            T=T, P=f_ref, fugacity=f_ref,
            N_min=0, N_max=6,
        )
        tmmc2.run(N=1000)  # already completed
        assert tmmc2._restart_steps_completed == 1000
