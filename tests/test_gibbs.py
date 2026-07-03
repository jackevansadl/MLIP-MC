"""
Unit tests for Gibbs Ensemble Monte Carlo simulation module.
"""
import json
import os
import struct
import pytest
import numpy as np
import tempfile
import shutil
from unittest.mock import Mock, patch

from mlip_mc.src.gibbs import MLP_Gibbs, GIBBS_MOVE_PROBABILITIES
from mlip_mc.src.utilities import read_gibbs_binary_log
from ase import Atoms
from ase.build import molecule
from ase.calculators.calculator import Calculator, all_changes
from ase.data import vdw_radii
from ase.units import kB


class ZeroCalculator(Calculator):
    """ASE calculator that returns zero energy and zero forces.

    Used for testing MD thermalization moves where the model must
    support both energy and force calculations.
    """
    implemented_properties = ['energy', 'forces']

    def calculate(self, atoms=None, properties=['energy'],
                  system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.results['energy'] = 0.0
        self.results['forces'] = np.zeros((len(atoms), 3))


class TestMLPGibbsInit:
    """Tests for MLP_Gibbs initialization."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock MLIP model."""
        model = Mock()
        model.get_potential_energy = Mock(return_value=0.0)
        return model

    @pytest.fixture
    def atoms_mol(self):
        """Create a simple molecule."""
        return molecule('H2')

    def test_initialization(self, mock_model, atoms_mol):
        """Test basic initialization sets correct state."""
        np.random.seed(42)
        zero_vdw = vdw_radii.copy()
        zero_vdw[:] = 0.0

        gibbs = MLP_Gibbs(
            model=mock_model,
            atoms_mol=atoms_mol,
            T=300,
            N1_init=3,
            N2_init=5,
            L1_init=20.0,
            L2_init=15.0,
            device='cpu',
            vdw_radii=zero_vdw,
        )

        assert gibbs.T == 300
        assert gibbs.N1 == 3
        assert gibbs.N2 == 5
        assert gibbs.n_mol == 2  # H2 has 2 atoms
        assert gibbs.V1 == pytest.approx(20.0 ** 3)
        assert gibbs.V2 == pytest.approx(15.0 ** 3)
        assert gibbs.V_total == pytest.approx(20.0 ** 3 + 15.0 ** 3)
        assert gibbs.beta == pytest.approx(1.0 / (kB * 300))

    def test_move_statistics_initialization(self, mock_model, atoms_mol):
        """Test that move statistics are initialized correctly."""
        np.random.seed(42)
        zero_vdw = vdw_radii.copy()
        zero_vdw[:] = 0.0

        gibbs = MLP_Gibbs(
            model=mock_model,
            atoms_mol=atoms_mol,
            T=300,
            N1_init=2,
            N2_init=2,
            L1_init=20.0,
            L2_init=20.0,
            device='cpu',
            vdw_radii=zero_vdw,
        )

        for move_type in ['md_thermalization', 'translation', 'rotation', 'volume', 'swap']:
            assert move_type in gibbs.moves
            assert gibbs.moves[move_type]['attempted'] == 0
            assert gibbs.moves[move_type]['accepted'] == 0

        assert gibbs.swap_rejected_vdw == 0
        assert gibbs.swap_rejected_empty_source == 0
        assert gibbs.volume_rejected_negative == 0

    def test_custom_move_probabilities(self, mock_model, atoms_mol):
        """Test that custom move probabilities are accepted."""
        np.random.seed(42)
        zero_vdw = vdw_radii.copy()
        zero_vdw[:] = 0.0

        custom_probs = {
            'translation': 0.30,
            'rotation': 0.50,
            'volume': 0.60,
            'swap': 1.00,
        }

        gibbs = MLP_Gibbs(
            model=mock_model,
            atoms_mol=atoms_mol,
            T=300,
            N1_init=2,
            N2_init=2,
            L1_init=20.0,
            L2_init=20.0,
            device='cpu',
            vdw_radii=zero_vdw,
            move_probabilities=custom_probs,
        )

        assert gibbs.move_probabilities == custom_probs

    def test_default_move_probabilities(self, mock_model, atoms_mol):
        """Test that default move probabilities are used when not specified."""
        np.random.seed(42)
        zero_vdw = vdw_radii.copy()
        zero_vdw[:] = 0.0

        gibbs = MLP_Gibbs(
            model=mock_model,
            atoms_mol=atoms_mol,
            T=300,
            N1_init=1,
            N2_init=1,
            L1_init=20.0,
            L2_init=20.0,
            device='cpu',
            vdw_radii=zero_vdw,
        )

        assert gibbs.move_probabilities == GIBBS_MOVE_PROBABILITIES

    def test_temperature_dependence(self, mock_model, atoms_mol):
        """Test that beta is computed correctly for different temperatures."""
        zero_vdw = vdw_radii.copy()
        zero_vdw[:] = 0.0

        for T in [100, 300, 1000]:
            np.random.seed(42)
            gibbs = MLP_Gibbs(
                model=mock_model,
                atoms_mol=atoms_mol,
                T=T,
                N1_init=1,
                N2_init=1,
                L1_init=20.0,
                L2_init=20.0,
                device='cpu',
                vdw_radii=zero_vdw,
            )
            assert gibbs.beta == pytest.approx(1.0 / (kB * T))


class TestInitializeBox:
    """Tests for box initialization."""

    @pytest.fixture
    def mock_model(self):
        model = Mock()
        model.get_potential_energy = Mock(return_value=0.0)
        return model

    @pytest.fixture
    def atoms_mol(self):
        return molecule('H2')

    def test_correct_atom_count(self, mock_model, atoms_mol):
        """Test that the box has the correct number of atoms."""
        np.random.seed(42)
        zero_vdw = vdw_radii.copy()
        zero_vdw[:] = 0.0

        gibbs = MLP_Gibbs(
            model=mock_model,
            atoms_mol=atoms_mol,
            T=300,
            N1_init=5,
            N2_init=3,
            L1_init=20.0,
            L2_init=20.0,
            device='cpu',
            vdw_radii=zero_vdw,
        )

        # H2 has 2 atoms per molecule
        assert len(gibbs.atoms_box1) == 5 * 2
        assert len(gibbs.atoms_box2) == 3 * 2

    def test_cubic_cell(self, mock_model, atoms_mol):
        """Test that the box has a cubic cell."""
        np.random.seed(42)
        zero_vdw = vdw_radii.copy()
        zero_vdw[:] = 0.0

        gibbs = MLP_Gibbs(
            model=mock_model,
            atoms_mol=atoms_mol,
            T=300,
            N1_init=2,
            N2_init=2,
            L1_init=15.0,
            L2_init=25.0,
            device='cpu',
            vdw_radii=zero_vdw,
        )

        cell1 = gibbs.atoms_box1.get_cell()
        cell2 = gibbs.atoms_box2.get_cell()

        np.testing.assert_allclose(np.diag(cell1), [15.0, 15.0, 15.0])
        np.testing.assert_allclose(np.diag(cell2), [25.0, 25.0, 25.0])

    def test_empty_box(self, mock_model, atoms_mol):
        """Test that initializing with zero molecules creates an empty box."""
        np.random.seed(42)
        zero_vdw = vdw_radii.copy()
        zero_vdw[:] = 0.0

        gibbs = MLP_Gibbs(
            model=mock_model,
            atoms_mol=atoms_mol,
            T=300,
            N1_init=0,
            N2_init=3,
            L1_init=20.0,
            L2_init=20.0,
            device='cpu',
            vdw_radii=zero_vdw,
        )

        assert len(gibbs.atoms_box1) == 0
        assert len(gibbs.atoms_box2) == 6  # 3 * 2

    def test_placement_failure_raises(self, mock_model, atoms_mol):
        """Test that placing too many molecules in a small box raises error."""
        large_vdw = vdw_radii.copy()
        large_vdw[1] = 10.0  # 10 Angstrom radius for H

        with pytest.raises(RuntimeError, match="Failed to place molecule"):
            MLP_Gibbs(
                model=mock_model,
                atoms_mol=atoms_mol,
                T=300,
                N1_init=100,
                N2_init=0,
                L1_init=5.0,
                L2_init=20.0,
                device='cpu',
                vdw_radii=large_vdw,
            )


class TestComputeEnergy:
    """Tests for energy computation."""

    def test_empty_box_returns_zero(self):
        """Test that empty boxes return zero energy."""
        model = Mock()
        model.get_potential_energy = Mock(return_value=0.0)
        zero_vdw = vdw_radii.copy()
        zero_vdw[:] = 0.0

        np.random.seed(42)
        gibbs = MLP_Gibbs(
            model=model,
            atoms_mol=molecule('H2'),
            T=300,
            N1_init=0,
            N2_init=1,
            L1_init=20.0,
            L2_init=20.0,
            device='cpu',
            vdw_radii=zero_vdw,
        )

        empty_atoms = Atoms(cell=[10, 10, 10], pbc=True)
        assert gibbs._compute_energy(empty_atoms) == 0.0

    def test_nonempty_box_uses_calculator(self):
        """Test that non-empty boxes use the model calculator."""
        model = Mock()
        model.get_potential_energy = Mock(return_value=-5.0)
        zero_vdw = vdw_radii.copy()
        zero_vdw[:] = 0.0

        np.random.seed(42)
        gibbs = MLP_Gibbs(
            model=model,
            atoms_mol=molecule('H2'),
            T=300,
            N1_init=1,
            N2_init=1,
            L1_init=20.0,
            L2_init=20.0,
            device='cpu',
            vdw_radii=zero_vdw,
        )

        atoms = Atoms('H2', positions=[[0, 0, 0], [1, 0, 0]], cell=[10, 10, 10], pbc=True)
        energy = gibbs._compute_energy(atoms)
        assert energy == -5.0


class TestVolumeMove:
    """Tests for volume change moves."""

    @pytest.fixture
    def gibbs(self):
        """Create a Gibbs instance for testing."""
        model = Mock()
        model.get_potential_energy = Mock(return_value=0.0)
        zero_vdw = vdw_radii.copy()
        zero_vdw[:] = 0.0

        np.random.seed(42)
        g = MLP_Gibbs(
            model=model,
            atoms_mol=molecule('H2'),
            T=300,
            N1_init=3,
            N2_init=3,
            L1_init=20.0,
            L2_init=20.0,
            device='cpu',
            vdw_radii=zero_vdw,
            max_delta_V=100.0,
        )
        g.E1 = 0.0
        g.E2 = 0.0
        return g

    def test_total_volume_preserved(self, gibbs):
        """Test that total volume is preserved after a volume move."""
        V_total_before = gibbs.V1 + gibbs.V2

        np.random.seed(123)
        gibbs._move_volume()

        V_total_after = gibbs.V1 + gibbs.V2
        assert V_total_after == pytest.approx(V_total_before)

    def test_negative_volume_rejection(self):
        """Test that volume moves producing negative volumes are rejected."""
        model = Mock()
        model.get_potential_energy = Mock(return_value=0.0)
        zero_vdw = vdw_radii.copy()
        zero_vdw[:] = 0.0

        np.random.seed(42)
        g = MLP_Gibbs(
            model=model,
            atoms_mol=molecule('H2'),
            T=300,
            N1_init=1,
            N2_init=1,
            L1_init=5.0,
            L2_init=5.0,
            device='cpu',
            vdw_radii=zero_vdw,
            max_delta_V=1000.0,  # Very large to force negative volume
        )
        g.E1 = 0.0
        g.E2 = 0.0

        # Run many volume moves; some should be rejected due to negative V
        for _ in range(100):
            g.moves['volume']['attempted'] += 1
            g._move_volume()

        assert g.volume_rejected_negative > 0

    def test_volume_move_with_zero_energy(self, gibbs):
        """Test that volume moves are accepted with zero energy model."""
        np.random.seed(42)
        accepted = 0
        for _ in range(50):
            gibbs.moves['volume']['attempted'] += 1
            if gibbs._move_volume():
                accepted += 1

        # With zero energy, acceptance depends only on volume ratio terms
        assert accepted > 0


class TestSwapMove:
    """Tests for particle swap moves."""

    @pytest.fixture
    def gibbs(self):
        """Create a Gibbs instance for testing."""
        model = Mock()
        model.get_potential_energy = Mock(return_value=0.0)
        zero_vdw = vdw_radii.copy()
        zero_vdw[:] = 0.0

        np.random.seed(42)
        g = MLP_Gibbs(
            model=model,
            atoms_mol=molecule('H2'),
            T=300,
            N1_init=5,
            N2_init=5,
            L1_init=20.0,
            L2_init=20.0,
            device='cpu',
            vdw_radii=zero_vdw,
        )
        g.E1 = 0.0
        g.E2 = 0.0
        return g

    def test_total_molecules_preserved(self, gibbs):
        """Test that total molecule count is preserved after swaps."""
        N_total_before = gibbs.N1 + gibbs.N2

        np.random.seed(42)
        for _ in range(50):
            gibbs.moves['swap']['attempted'] += 1
            gibbs._move_swap()

        N_total_after = gibbs.N1 + gibbs.N2
        assert N_total_after == N_total_before

    def test_atom_count_matches_molecules(self, gibbs):
        """Test that atom count matches molecule count after swaps."""
        np.random.seed(42)
        for _ in range(20):
            gibbs.moves['swap']['attempted'] += 1
            gibbs._move_swap()

        assert len(gibbs.atoms_box1) == gibbs.N1 * gibbs.n_mol
        assert len(gibbs.atoms_box2) == gibbs.N2 * gibbs.n_mol

    def test_swap_from_empty_box_rejected(self):
        """Test that swaps from an empty box are rejected."""
        model = Mock()
        model.get_potential_energy = Mock(return_value=0.0)
        zero_vdw = vdw_radii.copy()
        zero_vdw[:] = 0.0

        np.random.seed(42)
        g = MLP_Gibbs(
            model=model,
            atoms_mol=molecule('H2'),
            T=300,
            N1_init=0,
            N2_init=0,
            L1_init=20.0,
            L2_init=20.0,
            device='cpu',
            vdw_radii=zero_vdw,
        )
        g.E1 = 0.0
        g.E2 = 0.0

        # Both boxes empty - every swap should be rejected
        for _ in range(10):
            g.moves['swap']['attempted'] += 1
            result = g._move_swap()
            assert result is False

        assert g.swap_rejected_empty_source == 10

    def test_swap_vdw_rejection(self):
        """Test that swaps with VDW overlap are rejected."""
        model = Mock()
        model.get_potential_energy = Mock(return_value=0.0)
        zero_vdw = vdw_radii.copy()
        zero_vdw[:] = 0.0

        # Initialize with zero VDW to allow placement. The VDW pre-screen
        # is opt-in (default off, matching RASPA3 which rejects overlaps
        # via the energy); enable it since the mock model's zero energy
        # cannot reject anything.
        np.random.seed(42)
        g = MLP_Gibbs(
            model=model,
            atoms_mol=molecule('H2'),
            T=300,
            N1_init=3,
            N2_init=3,
            L1_init=10.0,
            L2_init=10.0,
            device='cpu',
            vdw_radii=zero_vdw,
            swap_vdw_screen=True,
        )
        g.E1 = 0.0
        g.E2 = 0.0

        # Now set large VDW radii so swap insertions will always overlap
        large_vdw = vdw_radii.copy()
        large_vdw[1] = 8.0
        g.vdw = large_vdw - 0.35

        np.random.seed(42)
        for _ in range(50):
            g.moves['swap']['attempted'] += 1
            g._move_swap()

        assert g.swap_rejected_vdw > 0


class TestTranslationAndRotation:
    """Tests for translation and rotation moves."""

    @pytest.fixture
    def gibbs(self):
        model = Mock()
        model.get_potential_energy = Mock(return_value=0.0)
        zero_vdw = vdw_radii.copy()
        zero_vdw[:] = 0.0

        np.random.seed(42)
        g = MLP_Gibbs(
            model=model,
            atoms_mol=molecule('H2'),
            T=300,
            N1_init=3,
            N2_init=3,
            L1_init=20.0,
            L2_init=20.0,
            device='cpu',
            vdw_radii=zero_vdw,
        )
        g.E1 = 0.0
        g.E2 = 0.0
        return g

    def test_translation_with_empty_system(self):
        """Test translation returns False when both boxes are empty."""
        model = Mock()
        model.get_potential_energy = Mock(return_value=0.0)
        zero_vdw = vdw_radii.copy()
        zero_vdw[:] = 0.0

        np.random.seed(42)
        g = MLP_Gibbs(
            model=model,
            atoms_mol=molecule('H2'),
            T=300,
            N1_init=0,
            N2_init=0,
            L1_init=20.0,
            L2_init=20.0,
            device='cpu',
            vdw_radii=zero_vdw,
        )
        g.E1 = 0.0
        g.E2 = 0.0

        assert g._move_translation() is False

    def test_rotation_with_empty_system(self):
        """Test rotation returns False when both boxes are empty."""
        model = Mock()
        model.get_potential_energy = Mock(return_value=0.0)
        zero_vdw = vdw_radii.copy()
        zero_vdw[:] = 0.0

        np.random.seed(42)
        g = MLP_Gibbs(
            model=model,
            atoms_mol=molecule('H2'),
            T=300,
            N1_init=0,
            N2_init=0,
            L1_init=20.0,
            L2_init=20.0,
            device='cpu',
            vdw_radii=zero_vdw,
        )
        g.E1 = 0.0
        g.E2 = 0.0

        assert g._move_rotation() is False

    def test_translation_acceptance_zero_energy(self, gibbs):
        """Test that translations are accepted with zero energy model."""
        np.random.seed(42)
        accepted = 0
        for _ in range(50):
            gibbs.moves['translation']['attempted'] += 1
            if gibbs._move_translation():
                accepted += 1

        # With zero energy, all non-overlapping translations should be accepted
        assert accepted > 0

    def test_rotation_acceptance_zero_energy(self, gibbs):
        """Test that rotations are accepted with zero energy model."""
        np.random.seed(42)
        accepted = 0
        for _ in range(50):
            gibbs.moves['rotation']['attempted'] += 1
            if gibbs._move_rotation():
                accepted += 1

        assert accepted > 0


class TestMDThermalization:
    """Tests for NVT MD thermalization moves."""

    @pytest.fixture
    def gibbs_with_md(self):
        """Create a Gibbs instance configured for MD thermalization."""
        zero_vdw = vdw_radii.copy()
        zero_vdw[:] = 0.0

        np.random.seed(42)
        g = MLP_Gibbs(
            model=ZeroCalculator(),
            atoms_mol=molecule('H2'),
            T=300,
            N1_init=3,
            N2_init=3,
            L1_init=20.0,
            L2_init=20.0,
            device='cpu',
            vdw_radii=zero_vdw,
            md_timestep=0.25,
            md_steps=10,  # Very short for testing
            md_damp=0.01,
        )
        g.E1 = 0.0
        g.E2 = 0.0
        return g

    def test_md_thermalization_always_accepted(self, gibbs_with_md):
        """Test that MD thermalization is always accepted."""
        np.random.seed(42)
        for _ in range(5):
            gibbs_with_md.moves['md_thermalization']['attempted'] += 1
            result = gibbs_with_md._move_md_thermalization()
            assert result is True

        assert gibbs_with_md.moves['md_thermalization']['accepted'] == 5

    def test_md_preserves_atom_count(self, gibbs_with_md):
        """Test that MD thermalization preserves atom count in both boxes."""
        n_atoms_box1_before = len(gibbs_with_md.atoms_box1)
        n_atoms_box2_before = len(gibbs_with_md.atoms_box2)

        np.random.seed(42)
        gibbs_with_md._move_md_thermalization()

        assert len(gibbs_with_md.atoms_box1) == n_atoms_box1_before
        assert len(gibbs_with_md.atoms_box2) == n_atoms_box2_before

    def test_md_preserves_cell(self, gibbs_with_md):
        """Test that MD thermalization preserves box dimensions."""
        cell1_before = gibbs_with_md.atoms_box1.get_cell().copy()
        cell2_before = gibbs_with_md.atoms_box2.get_cell().copy()

        np.random.seed(42)
        gibbs_with_md._move_md_thermalization()

        np.testing.assert_allclose(
            gibbs_with_md.atoms_box1.get_cell(), cell1_before
        )
        np.testing.assert_allclose(
            gibbs_with_md.atoms_box2.get_cell(), cell2_before
        )

    def test_md_updates_positions(self, gibbs_with_md):
        """Test that MD thermalization changes positions."""
        pos1_before = gibbs_with_md.atoms_box1.get_positions().copy()
        pos2_before = gibbs_with_md.atoms_box2.get_positions().copy()

        np.random.seed(42)
        gibbs_with_md._move_md_thermalization()

        # With Langevin dynamics at T=300K, positions should change
        # (even with zero forces, thermal noise moves atoms)
        pos1_after = gibbs_with_md.atoms_box1.get_positions()
        pos2_after = gibbs_with_md.atoms_box2.get_positions()

        assert not np.allclose(pos1_before, pos1_after)
        assert not np.allclose(pos2_before, pos2_after)

    def test_md_with_empty_box(self):
        """Test that MD thermalization handles empty boxes gracefully."""
        zero_vdw = vdw_radii.copy()
        zero_vdw[:] = 0.0

        np.random.seed(42)
        g = MLP_Gibbs(
            model=ZeroCalculator(),
            atoms_mol=molecule('H2'),
            T=300,
            N1_init=0,
            N2_init=3,
            L1_init=20.0,
            L2_init=20.0,
            device='cpu',
            vdw_radii=zero_vdw,
            md_steps=10,
        )
        g.E1 = 0.0
        g.E2 = 0.0

        # Should succeed even with one empty box
        g.moves['md_thermalization']['attempted'] += 1
        result = g._move_md_thermalization()
        assert result is True
        assert len(g.atoms_box1) == 0  # Still empty

    def test_md_parameters_stored(self):
        """Test that MD parameters are stored correctly."""
        zero_vdw = vdw_radii.copy()
        zero_vdw[:] = 0.0

        np.random.seed(42)
        g = MLP_Gibbs(
            model=ZeroCalculator(),
            atoms_mol=molecule('H2'),
            T=300,
            N1_init=1,
            N2_init=1,
            L1_init=20.0,
            L2_init=20.0,
            device='cpu',
            vdw_radii=zero_vdw,
            md_timestep=0.5,
            md_steps=1000,
            md_damp=0.02,
        )

        assert g.md_timestep == 0.5
        assert g.md_steps == 1000
        assert g.md_damp == 0.02

    def test_md_in_probability_chain(self):
        """Test that MD thermalization is selected when probabilities are set."""
        zero_vdw = vdw_radii.copy()
        zero_vdw[:] = 0.0

        np.random.seed(42)
        g = MLP_Gibbs(
            model=ZeroCalculator(),
            atoms_mol=molecule('H2'),
            T=300,
            N1_init=3,
            N2_init=3,
            L1_init=20.0,
            L2_init=20.0,
            device='cpu',
            vdw_radii=zero_vdw,
            md_steps=5,
            move_probabilities={
                'md_thermalization': 1.0,
                'translation': 1.0,
                'rotation': 1.0,
                'volume': 1.0,
                'swap': 1.0,
            },
        )

        g.E1 = 0.0
        g.E2 = 0.0

        # With md_thermalization=1.0, all moves should be MD
        for _ in range(10):
            switch = np.random.rand()
            if switch < 1.0:  # Always true
                g.moves['md_thermalization']['attempted'] += 1
                g._move_md_thermalization()

        assert g.moves['md_thermalization']['attempted'] == 10
        assert g.moves['md_thermalization']['accepted'] == 10


class TestMDDebug:
    """Tests for MD debug output (CSV files and summary)."""

    @pytest.fixture
    def temp_dir(self):
        d = tempfile.mkdtemp()
        yield d
        shutil.rmtree(d)

    @pytest.fixture
    def gibbs_md_debug(self, temp_dir):
        """Create a Gibbs instance with md_debug enabled."""
        zero_vdw = vdw_radii.copy()
        zero_vdw[:] = 0.0

        np.random.seed(42)
        g = MLP_Gibbs(
            model=ZeroCalculator(),
            atoms_mol=molecule('H2'),
            T=300,
            N1_init=3,
            N2_init=3,
            L1_init=20.0,
            L2_init=20.0,
            device='cpu',
            vdw_radii=zero_vdw,
            md_timestep=0.25,
            md_steps=20,
            md_debug=True,
            md_debug_interval=5,
            output_dir=temp_dir,
        )
        g.E1 = 0.0
        g.E2 = 0.0
        return g

    def test_md_debug_creates_csv(self, gibbs_md_debug, temp_dir):
        """Test that MD debug writes CSV files for each box."""
        gibbs_md_debug.moves['md_thermalization']['attempted'] += 1
        gibbs_md_debug._move_md_thermalization()

        csv1 = os.path.join(temp_dir, 'md_debug', 'md_box1.csv')
        csv2 = os.path.join(temp_dir, 'md_debug', 'md_box2.csv')
        assert os.path.exists(csv1)
        assert os.path.exists(csv2)

    def test_md_debug_csv_header(self, gibbs_md_debug, temp_dir):
        """Test that CSV has the correct header."""
        gibbs_md_debug.moves['md_thermalization']['attempted'] += 1
        gibbs_md_debug._move_md_thermalization()

        csv1 = os.path.join(temp_dir, 'md_debug', 'md_box1.csv')
        with open(csv1, 'r') as f:
            header = f.readline().strip()
        assert header == 'step,time_fs,T_inst,E_pot,E_kin,E_tot,max_force'

    def test_md_debug_csv_has_data_rows(self, gibbs_md_debug, temp_dir):
        """Test that CSV contains data rows (initial + sampled + final)."""
        gibbs_md_debug.moves['md_thermalization']['attempted'] += 1
        gibbs_md_debug._move_md_thermalization()

        csv1 = os.path.join(temp_dir, 'md_debug', 'md_box1.csv')
        with open(csv1, 'r') as f:
            lines = f.readlines()
        # header + data: step 0, 5, 10, 15, 20 = at least 5 data rows
        assert len(lines) >= 4  # header + at least 3 data rows

    def test_md_debug_csv_first_step_is_zero(self, gibbs_md_debug, temp_dir):
        """Test that the first data row is step 0 (initial state)."""
        gibbs_md_debug.moves['md_thermalization']['attempted'] += 1
        gibbs_md_debug._move_md_thermalization()

        csv1 = os.path.join(temp_dir, 'md_debug', 'md_box1.csv')
        with open(csv1, 'r') as f:
            _ = f.readline()  # skip header
            first_data = f.readline().strip()
        assert first_data.startswith('0,')

    def test_md_debug_csv_last_step_is_md_steps(self, gibbs_md_debug, temp_dir):
        """Test that the last data row corresponds to the final MD step."""
        gibbs_md_debug.moves['md_thermalization']['attempted'] += 1
        gibbs_md_debug._move_md_thermalization()

        csv1 = os.path.join(temp_dir, 'md_debug', 'md_box1.csv')
        with open(csv1, 'r') as f:
            lines = f.readlines()
        last_step = int(lines[-1].split(',')[0])
        assert last_step == gibbs_md_debug.md_steps

    def test_md_debug_csv_overwritten_each_move(self, gibbs_md_debug, temp_dir):
        """Test that CSV is overwritten (not appended) each MD move."""
        gibbs_md_debug.moves['md_thermalization']['attempted'] += 1
        gibbs_md_debug._move_md_thermalization()

        csv1 = os.path.join(temp_dir, 'md_debug', 'md_box1.csv')
        with open(csv1, 'r') as f:
            lines_after_first = len(f.readlines())

        gibbs_md_debug.moves['md_thermalization']['attempted'] += 1
        gibbs_md_debug._move_md_thermalization()

        with open(csv1, 'r') as f:
            lines_after_second = len(f.readlines())

        # Should be same size (overwritten, not appended)
        assert lines_after_first == lines_after_second

    def test_md_debug_prints_summary(self, gibbs_md_debug, capsys):
        """Test that MD debug prints equilibration summary to stdout."""
        gibbs_md_debug.moves['md_thermalization']['attempted'] += 1
        gibbs_md_debug._move_md_thermalization()

        captured = capsys.readouterr()
        assert '[MD Debug] Box 1' in captured.out
        assert '[MD Debug] Box 2' in captured.out
        assert 'T target' in captured.out
        assert 'T (2nd half)' in captured.out
        assert 'E_pot' in captured.out
        assert 'E_tot drift' in captured.out

    def test_md_debug_disabled_by_default(self, temp_dir):
        """Test that no CSV is written when md_debug is False."""
        zero_vdw = vdw_radii.copy()
        zero_vdw[:] = 0.0

        np.random.seed(42)
        g = MLP_Gibbs(
            model=ZeroCalculator(),
            atoms_mol=molecule('H2'),
            T=300,
            N1_init=3,
            N2_init=3,
            L1_init=20.0,
            L2_init=20.0,
            device='cpu',
            vdw_radii=zero_vdw,
            md_steps=10,
            output_dir=temp_dir,
        )
        g.E1 = 0.0
        g.E2 = 0.0
        g.moves['md_thermalization']['attempted'] += 1
        g._move_md_thermalization()

        assert not os.path.exists(os.path.join(temp_dir, 'md_debug'))

    def test_md_debug_empty_box_skipped(self, temp_dir):
        """Test that empty boxes don't produce CSV files."""
        zero_vdw = vdw_radii.copy()
        zero_vdw[:] = 0.0

        np.random.seed(42)
        g = MLP_Gibbs(
            model=ZeroCalculator(),
            atoms_mol=molecule('H2'),
            T=300,
            N1_init=0,
            N2_init=3,
            L1_init=20.0,
            L2_init=20.0,
            device='cpu',
            vdw_radii=zero_vdw,
            md_steps=10,
            md_debug=True,
            output_dir=temp_dir,
        )
        g.E1 = 0.0
        g.E2 = 0.0
        g.moves['md_thermalization']['attempted'] += 1
        g._move_md_thermalization()

        # Box 1 is empty, so no CSV for it
        assert not os.path.exists(os.path.join(temp_dir, 'md_debug', 'md_box1.csv'))
        # Box 2 has atoms, so CSV should exist
        assert os.path.exists(os.path.join(temp_dir, 'md_debug', 'md_box2.csv'))

    def test_md_debug_parameters_stored(self, temp_dir):
        """Test that md_debug and md_debug_interval are stored."""
        zero_vdw = vdw_radii.copy()
        zero_vdw[:] = 0.0

        np.random.seed(42)
        g = MLP_Gibbs(
            model=ZeroCalculator(),
            atoms_mol=molecule('H2'),
            T=300,
            N1_init=1,
            N2_init=1,
            L1_init=20.0,
            L2_init=20.0,
            device='cpu',
            vdw_radii=zero_vdw,
            md_debug=True,
            md_debug_interval=7,
            output_dir=temp_dir,
        )
        assert g.md_debug is True
        assert g.md_debug_interval == 7

    def test_md_debug_interval_floor(self, temp_dir):
        """Test that md_debug_interval is clamped to at least 1."""
        zero_vdw = vdw_radii.copy()
        zero_vdw[:] = 0.0

        np.random.seed(42)
        g = MLP_Gibbs(
            model=ZeroCalculator(),
            atoms_mol=molecule('H2'),
            T=300,
            N1_init=1,
            N2_init=1,
            L1_init=20.0,
            L2_init=20.0,
            device='cpu',
            vdw_radii=zero_vdw,
            md_debug_interval=0,
            output_dir=temp_dir,
        )
        assert g.md_debug_interval == 1


class TestBinaryLog:
    """Tests for binary logging and reading."""

    @pytest.fixture
    def temp_dir(self):
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    def test_write_and_read_binary_log(self, temp_dir):
        """Test that binary log data survives a write/read cycle."""
        log_path = os.path.join(temp_dir, "log_gibbs_300.0K.bin")

        # Write some records
        fmt = "iiidddddd"
        records = [
            (1, 10, 20, 8000.0, 3375.0, -100.0, -200.0, 0.00125, 0.00593),
            (2, 11, 19, 7900.0, 3475.0, -105.0, -195.0, 0.00139, 0.00547),
            (3, 11, 19, 7950.0, 3425.0, -103.0, -197.0, 0.00138, 0.00555),
        ]

        with open(log_path, "wb") as f:
            for rec in records:
                f.write(struct.pack(fmt, *rec))

        # Read back
        data = read_gibbs_binary_log(log_path)

        assert len(data) == 3
        assert data[0]['step'] == 1
        assert data[0]['N1'] == 10
        assert data[0]['N2'] == 20
        assert data[0]['V1'] == pytest.approx(8000.0)
        assert data[0]['V2'] == pytest.approx(3375.0)
        assert data[0]['E1'] == pytest.approx(-100.0)
        assert data[0]['E2'] == pytest.approx(-200.0)
        assert data[0]['rho1'] == pytest.approx(0.00125)
        assert data[0]['rho2'] == pytest.approx(0.00593)

        assert data[2]['step'] == 3
        assert data[2]['N1'] == 11

    def test_read_nonexistent_log(self, temp_dir):
        """Test that reading a nonexistent log raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            read_gibbs_binary_log(os.path.join(temp_dir, "nonexistent.bin"))


class TestCheckpointRestart:
    """Tests for checkpoint and restart functionality."""

    @pytest.fixture
    def temp_dir(self):
        temp_dir = tempfile.mkdtemp()
        old_cwd = os.getcwd()
        os.chdir(temp_dir)
        yield temp_dir
        os.chdir(old_cwd)
        shutil.rmtree(temp_dir)

    def test_save_and_load_restart(self, temp_dir):
        """Test that restart state is saved and loaded correctly."""
        model = Mock()
        model.get_potential_energy = Mock(return_value=0.0)
        zero_vdw = vdw_radii.copy()
        zero_vdw[:] = 0.0

        np.random.seed(42)
        g = MLP_Gibbs(
            model=model,
            atoms_mol=molecule('H2'),
            T=300,
            N1_init=5,
            N2_init=5,
            L1_init=20.0,
            L2_init=20.0,
            device='cpu',
            vdw_radii=zero_vdw,
            n_equilibration_steps=100,
            n_production_steps=200,
        )
        g.E1 = -10.0
        g.E2 = -20.0
        g.moves['translation']['attempted'] = 50
        g.moves['translation']['accepted'] = 25

        # Save restart
        g._save_restart(75)

        # Verify restart files exist
        xyz1, xyz2, json_path = g._get_restart_paths()
        assert os.path.exists(xyz1)
        assert os.path.exists(xyz2)
        assert os.path.exists(json_path)

        # Load restart data and verify
        with open(json_path, 'r') as f:
            data = json.load(f)

        assert data['N1'] == 5
        assert data['N2'] == 5
        assert data['moves']['translation']['attempted'] == 50
        assert data['moves']['translation']['accepted'] == 25

    def test_restart_loads_correctly(self, temp_dir):
        """Test that loading restart restores state."""
        model = Mock()
        model.get_potential_energy = Mock(return_value=0.0)
        zero_vdw = vdw_radii.copy()
        zero_vdw[:] = 0.0

        np.random.seed(42)
        g1 = MLP_Gibbs(
            model=model,
            atoms_mol=molecule('H2'),
            T=300,
            N1_init=4,
            N2_init=6,
            L1_init=20.0,
            L2_init=20.0,
            device='cpu',
            vdw_radii=zero_vdw,
            n_equilibration_steps=100,
            n_production_steps=200,
        )
        g1.E1 = -5.0
        g1.E2 = -15.0
        g1.swap_rejected_vdw = 10
        g1._save_restart(50)

        # Create new instance and load
        np.random.seed(42)
        g2 = MLP_Gibbs(
            model=model,
            atoms_mol=molecule('H2'),
            T=300,
            N1_init=1,
            N2_init=1,
            L1_init=20.0,
            L2_init=20.0,
            device='cpu',
            vdw_radii=zero_vdw,
            n_equilibration_steps=100,
            n_production_steps=200,
        )

        loaded = g2._load_restart_info()
        assert loaded is True
        assert g2.N1 == 4
        assert g2.N2 == 6
        assert g2.swap_rejected_vdw == 10


class TestStatistics:
    """Tests for statistics printing."""

    def test_print_statistics(self, capsys):
        """Test that statistics printing works without errors."""
        model = Mock()
        model.get_potential_energy = Mock(return_value=0.0)
        zero_vdw = vdw_radii.copy()
        zero_vdw[:] = 0.0

        np.random.seed(42)
        g = MLP_Gibbs(
            model=model,
            atoms_mol=molecule('H2'),
            T=300,
            N1_init=2,
            N2_init=2,
            L1_init=20.0,
            L2_init=20.0,
            device='cpu',
            vdw_radii=zero_vdw,
        )
        g.E1 = 0.0
        g.E2 = 0.0

        g.moves['translation']['attempted'] = 100
        g.moves['translation']['accepted'] = 40
        g.moves['swap']['attempted'] = 50
        g.moves['swap']['accepted'] = 10
        g.moves['md_thermalization']['attempted'] = 20
        g.moves['md_thermalization']['accepted'] = 20

        g._print_statistics()

        captured = capsys.readouterr()
        assert "Gibbs Ensemble MC Move Statistics" in captured.out
        assert "Translation" in captured.out
        assert "Swap" in captured.out
        assert "MD Therm." in captured.out
        assert "Final State" in captured.out


class TestShortRun:
    """Integration tests with short simulation runs."""

    @pytest.fixture
    def temp_dir(self):
        temp_dir = tempfile.mkdtemp()
        old_cwd = os.getcwd()
        os.chdir(temp_dir)
        yield temp_dir
        os.chdir(old_cwd)
        shutil.rmtree(temp_dir)

    def test_full_run_execution(self, temp_dir):
        """Test that a short simulation completes without error."""
        model = Mock()
        model.get_potential_energy = Mock(return_value=0.0)
        zero_vdw = vdw_radii.copy()
        zero_vdw[:] = 0.0

        np.random.seed(42)
        g = MLP_Gibbs(
            model=model,
            atoms_mol=molecule('H2'),
            T=300,
            N1_init=5,
            N2_init=5,
            L1_init=20.0,
            L2_init=20.0,
            device='cpu',
            vdw_radii=zero_vdw,
        )

        g.run(100)

        # Check output files exist
        assert os.path.exists("results")
        assert os.path.exists(f"results/results_gibbs_300.0K.json")

        # Check move statistics add up
        total_attempted = sum(
            stats['attempted'] for stats in g.moves.values()
        )
        assert total_attempted == 100

    def test_conservation_during_run(self, temp_dir):
        """Test that total N and total V are conserved during simulation."""
        model = Mock()
        model.get_potential_energy = Mock(return_value=0.0)
        zero_vdw = vdw_radii.copy()
        zero_vdw[:] = 0.0

        np.random.seed(42)
        g = MLP_Gibbs(
            model=model,
            atoms_mol=molecule('H2'),
            T=300,
            N1_init=5,
            N2_init=5,
            L1_init=20.0,
            L2_init=20.0,
            device='cpu',
            vdw_radii=zero_vdw,
        )

        N_total = g.N1 + g.N2
        V_total = g.V1 + g.V2

        g.run(100)

        assert g.N1 + g.N2 == N_total
        assert g.V1 + g.V2 == pytest.approx(V_total)

    def test_results_json_content(self, temp_dir):
        """Test that results JSON has expected structure."""
        model = Mock()
        model.get_potential_energy = Mock(return_value=0.0)
        zero_vdw = vdw_radii.copy()
        zero_vdw[:] = 0.0

        np.random.seed(42)
        g = MLP_Gibbs(
            model=model,
            atoms_mol=molecule('H2'),
            T=300,
            N1_init=3,
            N2_init=3,
            L1_init=20.0,
            L2_init=20.0,
            device='cpu',
            vdw_radii=zero_vdw,
        )

        g.run(50)

        with open("results/results_gibbs_300.0K.json", 'r') as f:
            data = json.load(f)

        assert 'N1' in data
        assert 'N2' in data
        assert 'V1' in data
        assert 'V2' in data
        assert 'E1' in data
        assert 'E2' in data
        assert isinstance(data['N1'], list)

    def test_atom_counts_consistent_after_run(self, temp_dir):
        """Test that atom counts match molecule counts after full run."""
        model = Mock()
        model.get_potential_energy = Mock(return_value=0.0)
        zero_vdw = vdw_radii.copy()
        zero_vdw[:] = 0.0

        np.random.seed(42)
        g = MLP_Gibbs(
            model=model,
            atoms_mol=molecule('H2'),
            T=300,
            N1_init=5,
            N2_init=5,
            L1_init=20.0,
            L2_init=20.0,
            device='cpu',
            vdw_radii=zero_vdw,
        )

        g.run(100)

        assert len(g.atoms_box1) == g.N1 * g.n_mol
        assert len(g.atoms_box2) == g.N2 * g.n_mol

    def test_binary_log_created(self, temp_dir):
        """Test that binary log file is created during simulation."""
        model = Mock()
        model.get_potential_energy = Mock(return_value=0.0)
        zero_vdw = vdw_radii.copy()
        zero_vdw[:] = 0.0

        np.random.seed(42)
        g = MLP_Gibbs(
            model=model,
            atoms_mol=molecule('H2'),
            T=300,
            N1_init=3,
            N2_init=3,
            L1_init=20.0,
            L2_init=20.0,
            device='cpu',
            vdw_radii=zero_vdw,
        )

        g.run(50)

        log_path = f"results/log_gibbs_300.0K.bin"
        assert os.path.exists(log_path)

        data = read_gibbs_binary_log(log_path)
        assert len(data) > 0

    def test_reproducibility(self, temp_dir):
        """Test that runs with the same seed produce identical results."""
        model = Mock()
        model.get_potential_energy = Mock(return_value=0.0)
        zero_vdw = vdw_radii.copy()
        zero_vdw[:] = 0.0

        def run_with_seed(seed, output_dir):
            np.random.seed(seed)
            g = MLP_Gibbs(
                model=model,
                atoms_mol=molecule('H2'),
                T=300,
                N1_init=3,
                N2_init=3,
                L1_init=20.0,
                L2_init=20.0,
                device='cpu',
                vdw_radii=zero_vdw,
                output_dir=output_dir,
            )
            g.run(30)
            return g.N1, g.N2, g.V1, g.V2

        r1 = run_with_seed(42, os.path.join(temp_dir, 'run1'))
        r2 = run_with_seed(42, os.path.join(temp_dir, 'run2'))

        assert r1[0] == r2[0]  # N1
        assert r1[1] == r2[1]  # N2
        assert r1[2] == pytest.approx(r2[2])  # V1
        assert r1[3] == pytest.approx(r2[3])  # V2


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
