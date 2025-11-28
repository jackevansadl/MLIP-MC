"""
Unit tests for Widom insertion method module.
"""
import json
import os
import pytest
import numpy as np
import tempfile
import shutil
from unittest.mock import Mock, patch

from mlip_mc.src.widom import MLP_Widom
from ase import Atoms
from ase.build import molecule
from ase.data import vdw_radii
from ase.units import kB


class TestMLPWidom:
    """Tests for MLP_Widom class."""
    
    @pytest.fixture
    def mock_model(self):
        """Create a mock MLIP model."""
        model = Mock()
        model.get_potential_energy = Mock(return_value=0.0)
        return model
    
    @pytest.fixture
    def atoms_frame(self):
        """Create a simple framework structure."""
        return Atoms('H2', positions=[[0, 0, 0], [1, 0, 0]], cell=[10, 10, 10], pbc=True)
    
    @pytest.fixture
    def atoms_ads(self):
        """Create a simple adsorbate."""
        return molecule('H2')
    
    def test_initialization(self, mock_model, atoms_frame, atoms_ads):
        """Test Widom initialization."""
        with patch('os.path.exists', return_value=True):
            widom = MLP_Widom(
                model=mock_model,
                atoms_frame=atoms_frame,
                atoms_ads=atoms_ads,
                T=300,
                device='cpu',
                vdw_radii=vdw_radii
            )
        
        assert widom.T == 300
        assert widom.n_frame == len(atoms_frame)
        assert widom.n_ads == len(atoms_ads)
        assert widom.stats['attempts'] == 0
        assert widom.stats['valid_insertions'] == 0
        assert widom.stats['vdw_overlaps'] == 0
    
    def test_statistics_initialization(self, mock_model, atoms_frame, atoms_ads):
        """Test that statistics are initialized correctly."""
        with patch('os.path.exists', return_value=True):
            widom = MLP_Widom(
                model=mock_model,
                atoms_frame=atoms_frame,
                atoms_ads=atoms_ads,
                T=300,
                device='cpu',
                vdw_radii=vdw_radii
            )
        
        assert isinstance(widom.stats, dict)
        assert 'attempts' in widom.stats
        assert 'valid_insertions' in widom.stats
        assert 'vdw_overlaps' in widom.stats
    
    def test_save_results_json(self, mock_model, atoms_frame, atoms_ads):
        """Test JSON results saving."""
        with patch('os.path.exists', return_value=True), \
             patch('os.mkdir'), \
             patch('builtins.open', create=True) as mock_open:
            
            widom = MLP_Widom(
                model=mock_model,
                atoms_frame=atoms_frame,
                atoms_ads=atoms_ads,
                T=300,
                device='cpu',
                vdw_radii=vdw_radii
            )
            
            e_adsorptions = [0.1, 0.2, 0.3]
            widom._save_results_json(e_adsorptions)
            
            # Check that file was opened for writing
            assert mock_open.called
    
    def test_print_statistics_empty(self, mock_model, atoms_frame, atoms_ads, capsys):
        """Test statistics printing with no data."""
        with patch('os.path.exists', return_value=True):
            widom = MLP_Widom(
                model=mock_model,
                atoms_frame=atoms_frame,
                atoms_ads=atoms_ads,
                T=300,
                device='cpu',
                vdw_radii=vdw_radii
            )
            
            widom._print_statistics([])
            captured = capsys.readouterr()
            assert "No valid insertions found" in captured.out
    
    def test_print_statistics_with_data(self, mock_model, atoms_frame, atoms_ads, capsys):
        """Test statistics printing with data."""
        with patch('os.path.exists', return_value=True):
            widom = MLP_Widom(
                model=mock_model,
                atoms_frame=atoms_frame,
                atoms_ads=atoms_ads,
                T=300,
                device='cpu',
                vdw_radii=vdw_radii
            )
            
            e_adsorptions = [-0.1, -0.2, -0.3]  # Negative (favorable)
            widom._print_statistics(e_adsorptions)
            captured = capsys.readouterr()
            assert "Widom Statistics" in captured.out
            assert "Average Boltzmann Factor" in captured.out
    
    def test_boltzmann_factor_calculation(self, mock_model, atoms_frame, atoms_ads):
        """Test Boltzmann factor calculation in statistics."""
        with patch('os.path.exists', return_value=True):
            widom = MLP_Widom(
                model=mock_model,
                atoms_frame=atoms_frame,
                atoms_ads=atoms_ads,
                T=300,
                device='cpu',
                vdw_radii=vdw_radii
            )
            
            e_adsorptions = [-0.1, -0.2, -0.3]
            boltzmann_factors = np.exp(-widom.beta * np.array(e_adsorptions))
            
            # All should be > 1 (favorable energies)
            assert np.all(boltzmann_factors > 1.0)
            # Should be in descending order (more negative = larger BF)
            assert boltzmann_factors[2] > boltzmann_factors[1] > boltzmann_factors[0]
    
    def test_weighted_energy_calculation(self, mock_model, atoms_frame, atoms_ads):
        """Test weighted energy calculation."""
        with patch('os.path.exists', return_value=True):
            widom = MLP_Widom(
                model=mock_model,
                atoms_frame=atoms_frame,
                atoms_ads=atoms_ads,
                T=300,
                device='cpu',
                vdw_radii=vdw_radii
            )
            
            e_adsorptions = [-0.1, -0.2, -0.3]
            boltzmann_factors = np.exp(-widom.beta * np.array(e_adsorptions))
            weighted_E = (np.sum(np.array(e_adsorptions) * boltzmann_factors) /
                         np.sum(boltzmann_factors))
            
            # Weighted energy should be more negative than arithmetic mean
            # (more weight on favorable energies)
            arithmetic_mean = np.mean(e_adsorptions)
            assert weighted_E < arithmetic_mean
    
    def test_statistics_accumulation(self, mock_model, atoms_frame, atoms_ads):
        """Test that statistics accumulate correctly during simulation."""
        with patch('os.path.exists', return_value=True):
            widom = MLP_Widom(
                model=mock_model,
                atoms_frame=atoms_frame,
                atoms_ads=atoms_ads,
                T=300,
                device='cpu',
                vdw_radii=vdw_radii
            )
            
            # Simulate some statistics
            widom.stats['attempts'] = 100
            widom.stats['valid_insertions'] = 80
            widom.stats['vdw_overlaps'] = 20
            
            assert widom.stats['attempts'] == 100
            assert widom.stats['valid_insertions'] == 80
            assert widom.stats['vdw_overlaps'] == 20
            assert widom.stats['attempts'] == (widom.stats['valid_insertions'] +
                                               widom.stats['vdw_overlaps'])
    
    def test_temperature_dependence(self, mock_model, atoms_frame, atoms_ads):
        """Test that beta is calculated correctly for different temperatures."""
        for T in [100, 300, 1000]:
            with patch('os.path.exists', return_value=True):
                widom = MLP_Widom(
                    model=mock_model,
                    atoms_frame=atoms_frame,
                    atoms_ads=atoms_ads,
                    T=T,
                    device='cpu',
                    vdw_radii=vdw_radii
                )
                expected_beta = 1.0 / (8.617e-5 * T)  # eV^-1
                assert widom.beta == pytest.approx(expected_beta, rel=1e-3)
    
    def test_volume_calculation(self, mock_model, atoms_frame, atoms_ads):
        """Test that volume is calculated correctly."""
        with patch('os.path.exists', return_value=True):
            widom = MLP_Widom(
                model=mock_model,
                atoms_frame=atoms_frame,
                atoms_ads=atoms_ads,
                T=300,
                device='cpu',
                vdw_radii=vdw_radii
            )
            expected_volume = np.linalg.det(atoms_frame.get_cell())
            assert widom.V == pytest.approx(expected_volume)
    
    def test_empty_adsorptions_handling(self, mock_model, atoms_frame, atoms_ads):
        """Test handling of empty adsorption energy list."""
        with patch('os.path.exists', return_value=True), \
             patch('os.mkdir'):
            widom = MLP_Widom(
                model=mock_model,
                atoms_frame=atoms_frame,
                atoms_ads=atoms_ads,
                T=300,
                device='cpu',
                vdw_radii=vdw_radii
            )
            
            # Should handle empty list gracefully
            with patch('builtins.open', create=True):
                widom._save_results_json([])
            widom._print_statistics([])


class TestWidomRigorous:
    """Rigorous tests for Widom simulation including integration and physics checks."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock MLIP model."""
        model = Mock()
        model.get_potential_energy = Mock(return_value=0.0)
        return model
    
    @pytest.fixture
    def atoms_frame(self):
        """Create a simple framework structure."""
        # Simple cubic cell
        return Atoms('H8', positions=np.array([[0,0,0], [10,0,0], [0,10,0], [0,0,10], 
                                               [10,10,0], [10,0,10], [0,10,10], [10,10,10]]), 
                     cell=[10, 10, 10], pbc=True)
    
    @pytest.fixture
    def atoms_ads(self):
        """Create a simple adsorbate."""
        return molecule('H2')

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test results and switch to it."""
        temp_dir = tempfile.mkdtemp()
        old_cwd = os.getcwd()
        os.chdir(temp_dir)
        yield temp_dir
        os.chdir(old_cwd)
        shutil.rmtree(temp_dir)

    def test_full_run_execution(self, mock_model, atoms_frame, atoms_ads, temp_dir):
        """
        Integration test: Run a short simulation and ensure it completes without error
        and generates output files.
        """
        widom = MLP_Widom(
            model=mock_model,
            atoms_frame=atoms_frame,
            atoms_ads=atoms_ads,
            T=300,
            device='cpu',
            vdw_radii=vdw_radii
        )
        
        # Run for a small number of steps
        steps = 50
        widom.run(steps)
        
        # Check if results directory and files exist
        assert os.path.exists("results")
        assert os.path.exists("results/widom_results.json")
        assert os.path.exists("results/widom_traj.xyz")
        
        # Check stats
        assert widom.stats['attempts'] == steps
        assert widom.stats['valid_insertions'] + widom.stats['vdw_overlaps'] == steps

    def test_reproducibility(self, mock_model, atoms_frame, atoms_ads, temp_dir):
        """Test that runs with the same seed produce identical results."""
        
        def run_with_seed(seed):
            np.random.seed(seed)
            widom = MLP_Widom(
                model=mock_model,
                atoms_frame=atoms_frame,
                atoms_ads=atoms_ads,
                T=300,
                device='cpu',
                vdw_radii=vdw_radii
            )
            widom.run(20)
            
            # Read results
            with open("results/widom_results.json", 'r') as f:
                data = json.load(f)
            return data['raw_adsorption_energies']
            
        e1 = run_with_seed(42)
        e2 = run_with_seed(42)
        
        assert e1 == e2
        
    def test_vdw_overlap_rejection(self, mock_model, atoms_ads, temp_dir):
        """
        Test that insertions are rejected due to VDW overlap.
        We'll use a tiny cell with atoms to ensure overlap.
        """
        # Create a tiny cell where overlap is guaranteed
        tiny_frame = Atoms('H1', positions=[[0.5, 0.5, 0.5]], cell=[1, 1, 1], pbc=True)
        
        # Set VDW radii very large for Hydrogen
        large_vdw = vdw_radii.copy()
        large_vdw[1] = 10.0 # 10 Angstrom radius for H
        
        widom = MLP_Widom(
            model=mock_model,
            atoms_frame=tiny_frame,
            atoms_ads=atoms_ads,
            T=300,
            device='cpu',
            vdw_radii=large_vdw
        )
        
        widom.run(50)
        
        # Should have high number of overlaps
        assert widom.stats['vdw_overlaps'] > 0
        # Likely all overlaps given the setup
        assert widom.stats['valid_insertions'] == 0

    def test_zero_potential_limit(self, mock_model, atoms_frame, atoms_ads, temp_dir):
        """
        Test that for a system with zero potential energy (ideal gas limit),
        the average Boltzmann factor should be 1.0.
        """
        # Mock model returns 0 energy always -> Non-interacting gas
        mock_model.get_potential_energy.return_value = 0.0
        
        # Ensure no VDW overlaps by making VDW radii zero
        zero_vdw = vdw_radii.copy()
        zero_vdw[:] = 0.0
        
        np.random.seed(42)
        widom = MLP_Widom(
            model=mock_model,
            atoms_frame=atoms_frame,
            atoms_ads=atoms_ads,
            T=300,
            device='cpu',
            vdw_radii=zero_vdw
        )
        
        widom.run(50)
        
        # Load results
        with open("results/widom_results.json", 'r') as f:
            data = json.load(f)
            
        # In zero potential, all energies should be 0 (relative to baseline)
        # Baseline: E_frame (0) + E_ads (0) = 0. Trial E = 0. Interaction E = 0.
        raw_energies = data['raw_adsorption_energies']
        assert np.allclose(raw_energies, 0.0)
        
        # Average Boltzmann factor should be exp(-beta * 0) = 1
        avg_bf = data['average_boltzmann_factor']
        assert avg_bf == pytest.approx(1.0)

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
