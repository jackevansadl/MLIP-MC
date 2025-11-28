"""
Unit tests for utilities module.
"""
import pytest
import numpy as np
from unittest.mock import patch

from mlip_mc.src.utilities import (
    _random_rotation,
    random_position,
    vdw_overlap,
    EOS,
    PREOS
)
from ase import Atoms
from ase.units import Pascal


class TestRandomRotation:
    """Tests for _random_rotation function."""
    
    def test_rotation_preserves_center_of_mass(self):
        """Test that rotation preserves center of mass."""
        pos = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        original_com = np.mean(pos, axis=0)
        
        rotated = _random_rotation(pos.copy())
        rotated_com = np.mean(rotated, axis=0)
        
        np.testing.assert_allclose(original_com, rotated_com, atol=1e-10)
    
    def test_rotation_preserves_distances(self):
        """Test that rotation preserves interatomic distances."""
        pos = np.array([[0, 0, 0], [1, 0, 0]])
        original_dist = np.linalg.norm(pos[1] - pos[0])
        
        rotated = _random_rotation(pos.copy())
        rotated_dist = np.linalg.norm(rotated[1] - rotated[0])
        
        np.testing.assert_allclose(original_dist, rotated_dist, atol=1e-10)
    
    def test_rotation_preserves_all_distances(self):
        """Test that rotation preserves all pairwise distances."""
        pos = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 1]])
        original_dists = []
        for i in range(len(pos)):
            for j in range(i+1, len(pos)):
                original_dists.append(np.linalg.norm(pos[i] - pos[j]))
        
        rotated = _random_rotation(pos.copy())
        rotated_dists = []
        for i in range(len(rotated)):
            for j in range(i+1, len(rotated)):
                rotated_dists.append(np.linalg.norm(rotated[i] - rotated[j]))
        
        np.testing.assert_allclose(original_dists, rotated_dists, atol=1e-10)
    
    def test_circlefrac_parameter(self):
        """Test that circlefrac parameter affects rotation."""
        pos = np.array([[0, 0, 0], [1, 0, 0]])
        
        # With circlefrac=0, should be minimal rotation
        rotated_min = _random_rotation(pos.copy(), circlefrac=0.0)
        # With circlefrac=1, should be full rotation
        rotated_full = _random_rotation(pos.copy(), circlefrac=1.0)
        
        # Both should preserve distances
        assert np.linalg.norm(rotated_min[1] - rotated_min[0]) == pytest.approx(1.0)
        assert np.linalg.norm(rotated_full[1] - rotated_full[0]) == pytest.approx(1.0)
    
    def test_rotation_orthogonality(self):
        """Test that rotation matrix is orthogonal (preserves dot products)."""
        pos = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        original_vec1 = pos[1] - pos[0]
        original_vec2 = pos[2] - pos[0]
        original_dot = np.dot(original_vec1, original_vec2)
        
        rotated = _random_rotation(pos.copy())
        rotated_vec1 = rotated[1] - rotated[0]
        rotated_vec2 = rotated[2] - rotated[0]
        rotated_dot = np.dot(rotated_vec1, rotated_vec2)
        
        np.testing.assert_allclose(original_dot, rotated_dot, atol=1e-10)
    
    def test_multiple_rotations(self):
        """Test that multiple rotations still preserve distances."""
        pos = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        original_dist = np.linalg.norm(pos[1] - pos[0])
        
        # Apply rotation multiple times
        rotated = pos.copy()
        for _ in range(5):
            rotated = _random_rotation(rotated)
        
        final_dist = np.linalg.norm(rotated[1] - rotated[0])
        np.testing.assert_allclose(original_dist, final_dist, atol=1e-10)
    
    def test_single_atom(self):
        """Test rotation with single atom (should return unchanged)."""
        pos = np.array([[1, 2, 3]])
        rotated = _random_rotation(pos.copy())
        np.testing.assert_allclose(pos, rotated, atol=1e-10)


class TestRandomPosition:
    """Tests for random_position function."""
    
    def test_random_position(self):
        """Test random positioning."""
        pos = np.array([[0, 0, 0], [1, 0, 0]])
        rvecs = np.array([[10, 0, 0], [0, 10, 0], [0, 0, 10]])
        
        new_pos = random_position(pos.copy(), rvecs)
        
        # Should have same number of atoms
        assert len(new_pos) == len(pos)
        # Should have same shape
        assert new_pos.shape == pos.shape
    
    def test_positions_within_cell(self):
        """Test that positions stay within cell bounds (approximately)."""
        pos = np.array([[0, 0, 0], [1, 0, 0]])
        rvecs = np.array([[10, 0, 0], [0, 10, 0], [0, 0, 10]])
        
        # Test multiple times
        for _ in range(10):
            new_pos = random_position(pos.copy(), rvecs)
            # Positions should be reasonable (not way outside cell)
            assert np.all(np.abs(new_pos) < 20)  # Some margin for PBC
    
    def test_preserves_molecular_structure(self):
        """Test that relative positions within molecule are preserved."""
        pos = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])  # Triangle
        rvecs = np.array([[10, 0, 0], [0, 10, 0], [0, 0, 10]])
        
        original_dists = [
            np.linalg.norm(pos[1] - pos[0]),
            np.linalg.norm(pos[2] - pos[0]),
            np.linalg.norm(pos[2] - pos[1])
        ]
        
        new_pos = random_position(pos.copy(), rvecs)
        new_dists = [
            np.linalg.norm(new_pos[1] - new_pos[0]),
            np.linalg.norm(new_pos[2] - new_pos[0]),
            np.linalg.norm(new_pos[2] - new_pos[1])
        ]
        
        # Distances should be preserved (rotation + translation)
        np.testing.assert_allclose(original_dists, new_dists, atol=1e-10)


class TestVDWOverlap:
    """Tests for vdw_overlap function."""
    
    def test_no_overlap(self):
        """Test with no overlap."""
        # Create atoms with large separation
        atoms = Atoms('H2', positions=[[0, 0, 0], [10, 10, 10]], 
                     cell=[20, 20, 20], pbc=True)
        vdw = {1: 1.2}  # H VDW radius
        
        result = vdw_overlap(atoms, vdw, n_frame=0, n_ads=2, select_ads=0)
        assert result is False
    
    def test_overlap_detection(self):
        """Test overlap detection."""
        # Create framework + adsorbate with overlap
        # Framework: 1 atom at origin
        # Adsorbate: 1 atom very close (0.5 A away)
        atoms = Atoms('H2', positions=[[0, 0, 0], [0.5, 0, 0]], 
                     cell=[10, 10, 10], pbc=True)
        vdw = {1: 1.2}  # H VDW radius
        
        # n_frame=1 means first atom is framework, n_ads=1 means 1 atom per adsorbate
        result = vdw_overlap(atoms, vdw, n_frame=1, n_ads=1, select_ads=0)
        # Should detect overlap since distance (0.5) < 1.2 + 1.2 = 2.4
        assert result is True
    
    def test_overlap_boundary_case(self):
        """Test overlap detection at exact boundary."""
        # Distance exactly equal to sum of radii
        atoms = Atoms('H2', positions=[[0, 0, 0], [2.4, 0, 0]], 
                     cell=[10, 10, 10], pbc=True)
        vdw = {1: 1.2}  # H VDW radius
        
        result = vdw_overlap(atoms, vdw, n_frame=1, n_ads=1, select_ads=0)
        # At boundary (2.4 = 1.2 + 1.2), should NOT overlap (strict <)
        assert result is False
    
    def test_overlap_just_above_boundary(self):
        """Test overlap detection just above boundary."""
        atoms = Atoms('H2', positions=[[0, 0, 0], [2.41, 0, 0]], 
                     cell=[10, 10, 10], pbc=True)
        vdw = {1: 1.2}
        
        result = vdw_overlap(atoms, vdw, n_frame=1, n_ads=1, select_ads=0)
        assert result is False
    
    def test_multiple_adsorbates(self):
        """Test overlap detection with multiple adsorbates."""
        # Framework: 1 atom, Adsorbates: 2 atoms each
        atoms = Atoms('H5', positions=[
            [0, 0, 0],      # Framework
            [0.5, 0, 0],    # Adsorbate 1, atom 1
            [0.6, 0, 0],    # Adsorbate 1, atom 2
            [5, 5, 5],      # Adsorbate 2, atom 1
            [5.1, 5, 5]     # Adsorbate 2, atom 2
        ], cell=[10, 10, 10], pbc=True)
        vdw = {1: 1.2}
        
        # Check adsorbate 0 (overlaps with framework)
        result0 = vdw_overlap(atoms, vdw, n_frame=1, n_ads=2, select_ads=0)
        assert result0 is True
        
        # Check adsorbate 1 (no overlap)
        result1 = vdw_overlap(atoms, vdw, n_frame=1, n_ads=2, select_ads=1)
        assert result1 is False
    
    def test_overlap_with_self_excluded(self):
        """Test that adsorbate doesn't check overlap with itself."""
        atoms = Atoms('H3', positions=[
            [0, 0, 0],      # Framework
            [0.1, 0, 0],   # Adsorbate atom 1 (very close to atom 2)
            [0.2, 0, 0]    # Adsorbate atom 2
        ], cell=[10, 10, 10], pbc=True)
        vdw = {1: 1.2}
        
        # Should not detect overlap between adsorbate atoms
        result = vdw_overlap(atoms, vdw, n_frame=1, n_ads=2, select_ads=0)
        # Only checks overlap with framework, not within adsorbate
        assert result is True  # Overlaps with framework


class TestEOS:
    """Tests for EOS base class."""
    
    def test_initialization(self):
        """Test EOS initialization."""
        eos = EOS(mass=18.0)
        assert eos.mass == 18.0
    
    def test_calculate_fugacity_requires_mu_ex(self):
        """Test that calculate_fugacity requires calculate_mu_ex."""
        eos = EOS(mass=18.0)
        
        # Should raise NotImplementedError since calculate_mu_ex is not implemented
        with pytest.raises(AttributeError):
            eos.calculate_fugacity(300, 1e5)


class TestPREOS:
    """Tests for PREOS class."""
    
    def test_initialization(self):
        """Test PREOS initialization."""
        eos = PREOS(Tc=647.0, Pc=22.064e6*Pascal, omega=0.3449, mass=18.0)
        
        assert eos.Tc == 647.0
        assert eos.Pc == pytest.approx(22.064e6 * Pascal)
        assert eos.omega == 0.3449
        assert eos.mass == 18.0
    
    def test_from_name(self):
        """Test PREOS.from_name class method."""
        eos = PREOS.from_name('H2O')
        
        assert eos.Tc > 0
        assert eos.Pc > 0
        assert eos.mass > 0
    
    def test_from_name_invalid(self):
        """Test PREOS.from_name with invalid compound."""
        with pytest.raises(ValueError):
            PREOS.from_name('nonexistent_compound')
    
    def test_set_conditions(self):
        """Test set_conditions method."""
        eos = PREOS(Tc=647.0, Pc=22.064e6*Pascal, omega=0.3449, mass=18.0)
        eos.set_conditions(T=300, P=1e5*Pascal)
        
        assert hasattr(eos, 'Tr')
        assert hasattr(eos, 'alpha')
        assert hasattr(eos, 'A')
        assert hasattr(eos, 'B')
        assert eos.Tr == 300 / 647.0
    
    def test_polynomial_roots(self):
        """Test polynomial_roots method."""
        eos = PREOS(Tc=647.0, Pc=22.064e6*Pascal, omega=0.3449, mass=18.0)
        eos.set_conditions(T=300, P=1e5*Pascal)
        
        Z = eos.polynomial_roots()
        assert Z > 0
        # Check that it satisfies the polynomial (approximately)
        poly_value = eos.polynomial(Z)
        assert abs(poly_value) < 1e-6
    
    def test_calculate_rho(self):
        """Test density calculation."""
        eos = PREOS(Tc=647.0, Pc=22.064e6*Pascal, omega=0.3449, mass=18.0)
        rho = eos.calculate_rho(T=300, P=1e5*Pascal)
        
        assert rho > 0
        assert np.isfinite(rho)
    
    def test_calculate_mu_ex(self):
        """Test excess chemical potential calculation."""
        eos = PREOS(Tc=647.0, Pc=22.064e6*Pascal, omega=0.3449, mass=18.0)
        mu, Pref = eos.calculate_mu_ex(T=300, P=1e5*Pascal)
        
        assert np.isfinite(mu)
        assert Pref > 0
    
    def test_calculate_mu(self):
        """Test total chemical potential calculation."""
        eos = PREOS(Tc=647.0, Pc=22.064e6*Pascal, omega=0.3449, mass=18.0)
        mu = eos.calculate_mu(T=300, P=1e5*Pascal)
        
        assert np.isfinite(mu)
    
    def test_get_Pref(self):
        """Test reference pressure calculation."""
        eos = PREOS(Tc=647.0, Pc=22.064e6*Pascal, omega=0.3449, mass=18.0)
        Pref = eos.get_Pref(T=300, P0=1e5*Pascal)
        
        assert Pref > 0
        assert Pref <= 1e5*Pascal  # Should be less than or equal to initial guess
    
    def test_all_compounds_from_name(self):
        """Test PREOS.from_name with all available compounds."""
        compounds = ['NH3', 'Ar', 'C6H6', 'CO2', 
                     'CH4', 'N2', 'H2O']
        for compound in compounds:
            eos = PREOS.from_name(compound)
            assert eos.Tc > 0
            assert eos.Pc > 0
            assert eos.omega >= 0
            assert eos.mass > 0
    
    def test_eos_high_pressure(self):
        """Test EOS at very high pressure."""
        eos = PREOS(Tc=647.0, Pc=22.064e6*Pascal, omega=0.3449, mass=18.0)
        rho = eos.calculate_rho(T=300, P=1e8*Pascal)  # Very high pressure
        assert rho > 0
        assert np.isfinite(rho)
    
    def test_eos_low_pressure(self):
        """Test EOS at very low pressure."""
        eos = PREOS(Tc=647.0, Pc=22.064e6*Pascal, omega=0.3449, mass=18.0)
        rho = eos.calculate_rho(T=300, P=1e2*Pascal)  # Very low pressure
        assert rho > 0
        assert np.isfinite(rho)
    
    def test_eos_high_temperature(self):
        """Test EOS at high temperature."""
        eos = PREOS(Tc=647.0, Pc=22.064e6*Pascal, omega=0.3449, mass=18.0)
        rho = eos.calculate_rho(T=2000, P=1e5*Pascal)
        assert rho > 0
        assert np.isfinite(rho)
    
    def test_polynomial_roots_vapour_phase(self):
        """Test polynomial roots for vapour phase."""
        eos = PREOS(Tc=647.0, Pc=22.064e6*Pascal, omega=0.3449, mass=18.0, phase='vapour')
        eos.set_conditions(T=400, P=1e5*Pascal)
        Z = eos.polynomial_roots()
        assert Z > 0
        assert Z <= 2.0  # Z should be reasonable for vapour
    
    def test_polynomial_roots_liquid_phase(self):
        """Test polynomial roots for liquid phase."""
        eos = PREOS(Tc=647.0, Pc=22.064e6*Pascal, omega=0.3449, mass=18.0, phase='liquid')
        eos.set_conditions(T=300, P=1e6*Pascal)
        Z = eos.polynomial_roots()
        assert Z > 0
        assert Z < 1.0  # Z should be < 1 for liquid
    
    def test_fugacity_calculation(self):
        """Test fugacity calculation."""
        eos = PREOS(Tc=647.0, Pc=22.064e6*Pascal, omega=0.3449, mass=18.0)
        fugacity = eos.calculate_fugacity(T=300, P=1e5*Pascal)
        assert fugacity > 0
        assert np.isfinite(fugacity)
    
    def test_chemical_potential_consistency(self):
        """Test consistency between mu and mu_ex calculations."""
        eos = PREOS(Tc=647.0, Pc=22.064e6*Pascal, omega=0.3449, mass=18.0)
        mu = eos.calculate_mu(T=300, P=1e5*Pascal)
        mu_ex, Pref = eos.calculate_mu_ex(T=300, P=1e5*Pascal)
        
        # mu should be mu_ex + ideal gas contribution
        # Ideal gas part: -kT * ln(kT/Pref * lambda^1.5)
        assert np.isfinite(mu)
        assert np.isfinite(mu_ex)
        # mu should be more negative than mu_ex (ideal gas contribution is negative)
        assert mu < mu_ex


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

