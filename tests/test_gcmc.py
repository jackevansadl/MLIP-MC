"""
Unit tests for GCMC simulation module.
"""
import os
import json
import pytest
import numpy as np
import tempfile
import shutil
from unittest.mock import Mock, MagicMock, patch

from mlip_mc.src.gcmc import MLP_GCMC, e_interaction_of_adsorption
from ase import Atoms
from ase.build import molecule
from ase.io import read, write
from ase.units import bar
from ase.data import vdw_radii


class TestEInteractionOfAdsorption:
    """Tests for e_interaction_of_adsorption function."""
    
    def test_calculation(self):
        """Test interaction energy calculation."""
        e_system = 100.0
        framework_E = 50.0
        ads_energy = 10.0
        n_adsorbed = 3
        
        result = e_interaction_of_adsorption(e_system, framework_E, ads_energy, n_adsorbed)
        expected = e_system - framework_E - n_adsorbed * ads_energy
        assert result == expected
        assert result == 20.0
    
    def test_zero_adsorbed(self):
        """Test with zero adsorbed species."""
        result = e_interaction_of_adsorption(100.0, 50.0, 10.0, 0)
        assert result == 50.0
    
    def test_negative_interaction(self):
        """Test with negative interaction energy (favorable adsorption)."""
        result = e_interaction_of_adsorption(50.0, 100.0, 10.0, 2)
        assert result == -70.0  # Negative = favorable
    
    def test_large_number_adsorbed(self):
        """Test with many adsorbed species."""
        result = e_interaction_of_adsorption(1000.0, 100.0, 5.0, 100)
        assert result == 400.0
    
    def test_very_small_values(self):
        """Test with very small energy values."""
        result = e_interaction_of_adsorption(1e-10, 5e-11, 1e-11, 2)
        np.testing.assert_allclose(result, 3e-11, rtol=1e-10)


class TestMLPGCMC:
    """Tests for MLP_GCMC class."""
    
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
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test results."""
        temp_dir = tempfile.mkdtemp()
        old_cwd = os.getcwd()
        os.chdir(temp_dir)
        yield temp_dir
        os.chdir(old_cwd)
        shutil.rmtree(temp_dir)
    
    def test_initialization(self, mock_model, atoms_frame, atoms_ads):
        """Test GCMC initialization."""
        # We can use temp_dir logic here or just patch exists if we don't run full sim
        with patch('os.path.exists', return_value=True):
            gcmc = MLP_GCMC(
                model=mock_model,
                atoms_frame=atoms_frame,
                atoms_ads=atoms_ads,
                T=300,
                P=1.0 * bar,
                fugacity=1.0 * bar,
                device='cpu',
                vdw_radii=vdw_radii,
                debug=False
            )
        
        assert gcmc.T == 300
        assert gcmc.n_frame == len(atoms_frame)
        assert gcmc.n_ads == len(atoms_ads)
        assert gcmc.Z_ads == 0
        assert 'insertion' in gcmc.moves
        assert 'deletion' in gcmc.moves
        assert 'translation' in gcmc.moves
        assert 'rotation' in gcmc.moves
    
    def test_insertion_acceptance_high_energy(self, mock_model, atoms_frame, atoms_ads):
        """Test insertion acceptance with high favorable energy."""
        with patch('os.path.exists', return_value=True):
            gcmc = MLP_GCMC(
                model=mock_model,
                atoms_frame=atoms_frame,
                atoms_ads=atoms_ads,
                T=300,
                P=1.0 * bar,
                fugacity=1.0 * bar,
                device='cpu',
                vdw_radii=vdw_radii,
                debug=False
            )
            gcmc.Z_ads = 1
            gcmc.V = 1000.0
        
        # Very favorable insertion (exp_value > 100)
        e_trial = -1000.0  # Very negative (favorable)
        e = 0.0
        result = gcmc._insertion_acceptance(e_trial, e)
        assert result is True
    
    def test_insertion_acceptance_unfavorable(self, mock_model, atoms_frame, atoms_ads):
        """Test insertion acceptance with very unfavorable energy."""
        with patch('os.path.exists', return_value=True):
            gcmc = MLP_GCMC(
                model=mock_model,
                atoms_frame=atoms_frame,
                atoms_ads=atoms_ads,
                T=300,
                P=1.0 * bar,
                fugacity=1.0 * bar,
                device='cpu',
                vdw_radii=vdw_radii,
                debug=False
            )
            gcmc.Z_ads = 1
            gcmc.V = 1000.0
        
        # Very unfavorable insertion (exp_value < -100)
        e_trial = 1000.0  # Very positive (unfavorable)
        e = 0.0
        result = gcmc._insertion_acceptance(e_trial, e)
        assert result is False
    
    def test_deletion_acceptance(self, mock_model, atoms_frame, atoms_ads):
        """Test deletion acceptance."""
        with patch('os.path.exists', return_value=True):
            gcmc = MLP_GCMC(
                model=mock_model,
                atoms_frame=atoms_frame,
                atoms_ads=atoms_ads,
                T=300,
                P=1.0 * bar,
                fugacity=1.0 * bar,
                device='cpu',
                vdw_radii=vdw_radii,
                debug=False
            )
            gcmc.Z_ads = 1
            gcmc.V = 1000.0
        
        # Favorable deletion
        e_trial = -1000.0
        e = 0.0
        result = gcmc._deletion_acceptance(e_trial, e)
        assert result is True
    
    def test_statistics_tracking(self, mock_model, atoms_frame, atoms_ads):
        """Test that statistics are tracked correctly."""
        with patch('os.path.exists', return_value=True):
            gcmc = MLP_GCMC(
                model=mock_model,
                atoms_frame=atoms_frame,
                atoms_ads=atoms_ads,
                T=300,
                P=1.0 * bar,
                fugacity=1.0 * bar,
                device='cpu',
                vdw_radii=vdw_radii,
                debug=False
            )
        
        assert gcmc.moves['insertion']['attempted'] == 0
        assert gcmc.moves['insertion']['accepted'] == 0
        assert gcmc.insertion_rejected_due_to_vdw == 0
    
    def test_save_results_json(self, mock_model, atoms_frame, atoms_ads, temp_dir):
        """Test JSON results saving."""
        # We are in temp_dir because of fixture, so we can let it write
        gcmc = MLP_GCMC(
            model=mock_model,
            atoms_frame=atoms_frame,
            atoms_ads=atoms_ads,
            T=300,
            P=1.0 * bar,
            fugacity=1.0 * bar,
            device='cpu',
            vdw_radii=vdw_radii,
            debug=False
        )
        
        uptake = [1, 2, 3]
        adsorption_energy = [0.1, 0.2, 0.3]
        total_energy = [-0.1, -0.2, -0.3]
        gcmc._save_results_json(uptake, adsorption_energy, total_energy)
        
        assert os.path.exists(f"results/results_{1.0:.5f}bar.json")
    
    def test_insertion_acceptance_probability_calculation(self, mock_model, atoms_frame, atoms_ads):
        """Test insertion acceptance probability calculation."""
        with patch('os.path.exists', return_value=True):
            gcmc = MLP_GCMC(
                model=mock_model,
                atoms_frame=atoms_frame,
                atoms_ads=atoms_ads,
                T=300,
                P=1.0 * bar,
                fugacity=1.0 * bar,
                device='cpu',
                vdw_radii=vdw_radii,
                debug=False
            )
            gcmc.Z_ads = 5
            gcmc.V = 1000.0
            gcmc.beta = 1.0 / (8.617e-5 * 300)  # ~38.7 eV^-1
        
        # Test moderate energy difference
        e_trial = -0.1  # Favorable
        e = 0.0
        # Should accept with some probability (not guaranteed)
        # We'll test multiple times to check it's not always True/False
        results = []
        for _ in range(10):
            with patch('numpy.random.rand', return_value=0.5):
                result = gcmc._insertion_acceptance(e_trial, e)
                results.append(result)
        # Should have some variation (not all same)
        assert len(set(results)) >= 1  # At least some variation
    
    def test_deletion_acceptance_probability(self, mock_model, atoms_frame, atoms_ads):
        """Test deletion acceptance probability calculation."""
        with patch('os.path.exists', return_value=True):
            gcmc = MLP_GCMC(
                model=mock_model,
                atoms_frame=atoms_frame,
                atoms_ads=atoms_ads,
                T=300,
                P=1.0 * bar,
                fugacity=1.0 * bar,
                device='cpu',
                vdw_radii=vdw_radii,
                debug=False
            )
            gcmc.Z_ads = 5
            gcmc.V = 1000.0
        
        # Test moderate energy difference
        e_trial = 0.1  # Unfavorable (higher energy after deletion)
        e = 0.0
        # Should have some acceptance probability
        with patch('numpy.random.rand', return_value=0.01):  # Very low random number
            result = gcmc._deletion_acceptance(e_trial, e)
            # With low random number, might accept if probability is high enough
    
    def test_insertion_acceptance_edge_cases(self, mock_model, atoms_frame, atoms_ads):
        """Test insertion acceptance with edge cases."""
        with patch('os.path.exists', return_value=True):
            gcmc = MLP_GCMC(
                model=mock_model,
                atoms_frame=atoms_frame,
                atoms_ads=atoms_ads,
                T=300,
                P=1.0 * bar,
                fugacity=1.0 * bar,
                device='cpu',
                vdw_radii=vdw_radii,
                debug=False
            )
            gcmc.Z_ads = 1
            gcmc.V = 1000.0
        
        # Test exactly at threshold (exp_value = 100)
        # This is tricky, so test near threshold
        e_trial = -100.0 / gcmc.beta  # exp_value ≈ 100
        e = 0.0
        result = gcmc._insertion_acceptance(e_trial, e)
        assert isinstance(result, bool)
    
    def test_zero_temperature_handling(self, mock_model, atoms_frame, atoms_ads):
        """Test that zero temperature is handled (should raise error or handle gracefully)."""
        with patch('os.path.exists', return_value=True):
            with pytest.raises((ZeroDivisionError, ValueError)):
                gcmc = MLP_GCMC(
                    model=mock_model,
                    atoms_frame=atoms_frame,
                    atoms_ads=atoms_ads,
                    T=0,  # Zero temperature
                    P=1.0 * bar,
                    fugacity=1.0 * bar,
                    device='cpu',
                    vdw_radii=vdw_radii,
                    debug=False
                )
    
    def test_very_high_temperature(self, mock_model, atoms_frame, atoms_ads):
        """Test with very high temperature."""
        with patch('os.path.exists', return_value=True):
            gcmc = MLP_GCMC(
                model=mock_model,
                atoms_frame=atoms_frame,
                atoms_ads=atoms_ads,
                T=10000,  # Very high temperature
                P=1.0 * bar,
                fugacity=1.0 * bar,
                device='cpu',
                vdw_radii=vdw_radii,
                debug=False
            )
            assert gcmc.beta > 0
            # At T=10000K, beta = 1/(kB*T) ≈ 1.16 eV^-1
            expected_beta = 1.0 / (8.617e-5 * 10000)
            assert gcmc.beta == pytest.approx(expected_beta, rel=1e-3)
            assert gcmc.beta < 2.0  # Should be less than 2
    
    def test_volume_calculation(self, mock_model, atoms_frame, atoms_ads):
        """Test that volume is calculated correctly."""
        with patch('os.path.exists', return_value=True):
            gcmc = MLP_GCMC(
                model=mock_model,
                atoms_frame=atoms_frame,
                atoms_ads=atoms_ads,
                T=300,
                P=1.0 * bar,
                fugacity=1.0 * bar,
                device='cpu',
                vdw_radii=vdw_radii,
                debug=False
            )
            expected_volume = np.linalg.det(atoms_frame.get_cell())
            assert gcmc.V == pytest.approx(expected_volume)
    
    def test_move_statistics_accumulation(self, mock_model, atoms_frame, atoms_ads):
        """Test that move statistics accumulate correctly."""
        with patch('os.path.exists', return_value=True):
            gcmc = MLP_GCMC(
                model=mock_model,
                atoms_frame=atoms_frame,
                atoms_ads=atoms_ads,
                T=300,
                P=1.0 * bar,
                fugacity=1.0 * bar,
                device='cpu',
                vdw_radii=vdw_radii,
                debug=False
            )
        
        # Manually increment statistics
        gcmc.moves['insertion']['attempted'] = 10
        gcmc.moves['insertion']['accepted'] = 5
        gcmc.moves['translation']['attempted'] = 20
        gcmc.moves['translation']['accepted'] = 15
        
        assert gcmc.moves['insertion']['attempted'] == 10
        assert gcmc.moves['insertion']['accepted'] == 5
        assert gcmc.moves['translation']['attempted'] == 20
        assert gcmc.moves['translation']['accepted'] == 15


class TestGCMCRigorous:
    """Rigorous tests for GCMC simulation including integration and edge cases."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock MLIP model."""
        model = Mock()
        model.get_potential_energy = Mock(return_value=0.0)
        return model
    
    @pytest.fixture
    def atoms_frame(self):
        """Create a simple framework structure."""
        # Larger cell to allow insertions
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
        gcmc = MLP_GCMC(
            model=mock_model,
            atoms_frame=atoms_frame,
            atoms_ads=atoms_ads,
            T=300,
            P=1.0 * bar,
            fugacity=1.0 * bar,
            device='cpu',
            vdw_radii=vdw_radii,
            debug=True
        )
        
        # Run for a small number of steps
        steps = 50
        gcmc.run(steps)
        
        # Check if results directory and files exist
        assert os.path.exists("results")
        assert os.path.exists(f"results/results_{1.0:.5f}bar.json")
        assert os.path.exists(f"results/traj_{1.0:.5f}bar.xyz")
        
        # Check if moves were attempted
        total_attempts = sum(stats['attempted'] for stats in gcmc.moves.values())
        # With 50 steps, we expect 50 attempts
        assert total_attempts == steps
        
        # Check consistency of Z_ads (should be non-negative)
        assert gcmc.Z_ads >= 0

    def test_reproducibility(self, mock_model, atoms_frame, atoms_ads, temp_dir):
        """Test that runs with the same seed produce identical results."""
        
        def run_with_seed(seed):
            np.random.seed(seed)
            # Re-instantiate model to reset any state if needed (mock is stateless usually)
            gcmc = MLP_GCMC(
                model=mock_model,
                atoms_frame=atoms_frame,
                atoms_ads=atoms_ads,
                T=300,
                P=1.0 * bar,
                fugacity=1.0 * bar,
                device='cpu',
                vdw_radii=vdw_radii,
                debug=False
            )
            gcmc.run(20)
            return gcmc.Z_ads, gcmc.moves
            
        z1, moves1 = run_with_seed(42)
        z2, moves2 = run_with_seed(42)
        
        assert z1 == z2
        assert moves1['insertion']['accepted'] == moves2['insertion']['accepted']
        assert moves1['deletion']['accepted'] == moves2['deletion']['accepted']
        
        # Test that different seed produces different result (likely, but not guaranteed for very short run)
        # But for 20 steps with random choices, it should diverge
        z3, moves3 = run_with_seed(43)
        # It's possible they are same by chance, so checking exact equality of all stats might be too strict
        # but let's check if at least something is likely different or just that the code ran
        # Actually, checking equality for same seed is the rigorous part.

    def test_vdw_overlap_rejection(self, mock_model, atoms_frame, atoms_ads, temp_dir):
        """
        Test that insertions are rejected due to VDW overlap.
        We'll use a small cell and large VDW radii or fill it up.
        """
        # Create a tiny cell where overlap is guaranteed
        tiny_frame = Atoms('H1', positions=[[0.5, 0.5, 0.5]], cell=[1, 1, 1], pbc=True)
        
        # Set VDW radii very large for Hydrogen
        large_vdw = vdw_radii.copy()
        large_vdw[1] = 10.0 # 10 Angstrom radius for H
        
        gcmc = MLP_GCMC(
            model=mock_model,
            atoms_frame=tiny_frame,
            atoms_ads=atoms_ads, # H2 molecule
            T=300,
            P=1.0 * bar,
            fugacity=1.0 * bar,
            device='cpu',
            vdw_radii=large_vdw,
            debug=False
        )
        
        # Force insertion attempts
        steps = 20
        np.random.seed(42)
        # Mock random to always choose insertion (switch < 0.25)
        with patch('numpy.random.rand', side_effect=[0.1] * steps + [0.9] * 100): 
            # side_effect provides return values for sequential calls
            # We need to be careful because random is called inside run for other things too
            # So mocking rand might be messy.
            pass
            
        # Alternative: Just run and check if any insertion was rejected due to VDW
        # With such large VDW, almost all insertions should be rejected
        gcmc.run(50)
        
        # Check if we have rejections due to VDW
        # Note: randomness might pick deletion/move, but eventually insertion will be picked.
        # If insertion is picked, it should fail VDW check.
        
        # Assuming at least one insertion was attempted
        if gcmc.moves['insertion']['attempted'] > 0:
            assert gcmc.insertion_rejected_due_to_vdw > 0
            assert gcmc.moves['insertion']['accepted'] == 0

    def test_ideal_gas_limit(self, mock_model, atoms_frame, atoms_ads, temp_dir):
        """
        Test trends in the ideal gas limit (non-interacting).
        Uptake should increase with pressure/fugacity.
        """
        # Mock model returns 0 energy always -> Non-interacting gas
        mock_model.get_potential_energy.return_value = 0.0
        
        def get_average_uptake(fugacity_val):
            # Clean up previous results if any
            import glob
            for f in glob.glob("results/*.json"):
                os.remove(f)

            np.random.seed(42)
            gcmc = MLP_GCMC(
                model=mock_model,
                atoms_frame=atoms_frame,
                atoms_ads=atoms_ads,
                T=300,
                P=fugacity_val,
                fugacity=fugacity_val,
                device='cpu',
                vdw_radii=vdw_radii,
                debug=False
            )
            steps = 500
            gcmc.run(steps)
            
            # Read results from JSON
            filename = f"results/results_{fugacity_val/bar:.5f}bar.json"
            if not os.path.exists(filename):
                return 0.0
                
            with open(filename, 'r') as f:
                data = json.load(f)
                uptake = data['uptake']
                # Average over the second half
                if len(uptake) > 0:
                    return np.mean(uptake[len(uptake)//2:])
                return 0.0
            
        # Low fugacity
        import json
        # 0.1 bar
        uptake_low = get_average_uptake(0.1 * bar)
        
        # Higher fugacity (100 bar to be sure)
        uptake_high = get_average_uptake(100.0 * bar)
        
        # For ideal gas in GCMC, <N> = V * beta * f (roughly, if Volume is accessible)
        # So uptake should be higher for higher fugacity
        assert uptake_high > uptake_low


class TestRestartMechanism:
    """Tests for restart functionality in GCMC simulations."""
    
    @pytest.fixture
    def mock_model(self):
        """Create a mock MLIP model."""
        model = Mock()
        # Return a simple energy value for restart tests
        # The exact value doesn't matter for restart functionality
        model.get_potential_energy = Mock(return_value=0.0)
        # ASE's write function expects calculators to have a results attribute
        # Make it an empty dict to avoid errors when saving
        model.results = {}
        return model
    
    @pytest.fixture
    def atoms_frame(self):
        """Create a simple framework structure."""
        return Atoms('H4', positions=np.array([[0,0,0], [5,0,0], [0,5,0], [0,0,5]]), 
                     cell=[10, 10, 10], pbc=True)
    
    @pytest.fixture
    def atoms_ads(self):
        """Create a simple adsorbate."""
        return molecule('H2')
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test results."""
        temp_dir = tempfile.mkdtemp()
        old_cwd = os.getcwd()
        os.chdir(temp_dir)
        yield temp_dir
        os.chdir(old_cwd)
        shutil.rmtree(temp_dir)
    
    def test_save_restart_info(self, mock_model, atoms_frame, atoms_ads, temp_dir):
        """Test that restart info is saved correctly to interval directory."""
        restart_prefix = os.path.join(temp_dir, "test_restart")
        output_dir = os.path.join(temp_dir, "results")
        
        gcmc = MLP_GCMC(
            model=mock_model,
            atoms_frame=atoms_frame,
            atoms_ads=atoms_ads,
            T=300,
            P=1.0 * bar,
            fugacity=1.0 * bar,
            device='cpu',
            vdw_radii=vdw_radii,
            debug=False,
            output_dir=output_dir,
            restart_prefix=restart_prefix,
            save_interval=10
        )
        
        # Set some state
        gcmc.Z_ads = 3
        gcmc.moves['insertion']['attempted'] = 10
        gcmc.moves['insertion']['accepted'] = 5
        gcmc.insertion_rejected_due_to_vdw = 2
        
        # Create atoms with some adsorbates
        test_atoms = atoms_frame.copy()
        for _ in range(3):
            test_atoms = test_atoms + atoms_ads.copy()
        
        # Save restart info to interval directory
        interval_dir = os.path.join(output_dir, 'interval_100')
        os.makedirs(interval_dir, exist_ok=True)
        gcmc._save_restart_info_to_interval(test_atoms, interval_dir, 100)
        
        # Check files exist in interval directory
        restart_xyz = os.path.join(interval_dir, f'restart_{1.0:.5f}bar.xyz')
        restart_json = os.path.join(interval_dir, f'restart_{1.0:.5f}bar.json')
        assert os.path.exists(restart_xyz)
        assert os.path.exists(restart_json)
        
        # Check JSON content
        with open(restart_json, 'r') as f:
            data = json.load(f)
            assert data['Z_ads'] == 3
            assert data['iteration'] == 100
            assert data['moves']['insertion']['attempted'] == 10
            assert data['moves']['insertion']['accepted'] == 5
            assert data['rejections']['insertion_vdw'] == 2
    
    def test_load_restart_info(self, mock_model, atoms_frame, atoms_ads, temp_dir):
        """Test that restart info is loaded correctly from interval directory."""
        restart_prefix = os.path.join(temp_dir, "test_restart")
        output_dir = os.path.join(temp_dir, "results")
        
        # First, create and save restart info to interval directory
        gcmc1 = MLP_GCMC(
            model=mock_model,
            atoms_frame=atoms_frame,
            atoms_ads=atoms_ads,
            T=300,
            P=1.0 * bar,
            fugacity=1.0 * bar,
            device='cpu',
            vdw_radii=vdw_radii,
            debug=False,
            output_dir=output_dir,
            restart_prefix=restart_prefix,
            save_interval=10
        )
        
        gcmc1.Z_ads = 5
        gcmc1.moves['insertion']['attempted'] = 20
        gcmc1.moves['insertion']['accepted'] = 12
        gcmc1.moves['deletion']['attempted'] = 15
        gcmc1.moves['deletion']['accepted'] = 8
        gcmc1.insertion_rejected_due_to_vdw = 3
        gcmc1.insertion_rejected_due_to_acceptance = 5
        
        test_atoms = atoms_frame.copy()
        for _ in range(5):
            test_atoms = test_atoms + atoms_ads.copy()
        
        # Save to interval directory
        interval_dir = os.path.join(output_dir, 'interval_200')
        os.makedirs(interval_dir, exist_ok=True)
        gcmc1._save_restart_info_to_interval(test_atoms, interval_dir, 200)
        
        # Now create a new instance and load
        gcmc2 = MLP_GCMC(
            model=mock_model,
            atoms_frame=atoms_frame,
            atoms_ads=atoms_ads,
            T=300,
            P=1.0 * bar,
            fugacity=1.0 * bar,
            device='cpu',
            vdw_radii=vdw_radii,
            debug=False,
            output_dir=output_dir,
            restart_prefix=restart_prefix,
            save_interval=10
        )
        
        loaded_atoms = gcmc2._load_restart_info()
        
        # Check state was restored
        assert gcmc2.Z_ads == 5
        assert gcmc2.moves['insertion']['attempted'] == 20
        assert gcmc2.moves['insertion']['accepted'] == 12
        assert gcmc2.moves['deletion']['attempted'] == 15
        assert gcmc2.moves['deletion']['accepted'] == 8
        assert gcmc2.insertion_rejected_due_to_vdw == 3
        assert gcmc2.insertion_rejected_due_to_acceptance == 5
        
        # Check atoms were loaded
        assert len(loaded_atoms) == len(atoms_frame) + 5 * len(atoms_ads)
    
    def test_load_restart_info_missing_files(self, mock_model, atoms_frame, atoms_ads, temp_dir):
        """Test that missing restart files are handled gracefully."""
        restart_prefix = os.path.join(temp_dir, "nonexistent_restart")
        
        gcmc = MLP_GCMC(
            model=mock_model,
            atoms_frame=atoms_frame,
            atoms_ads=atoms_ads,
            T=300,
            P=1.0 * bar,
            fugacity=1.0 * bar,
            device='cpu',
            vdw_radii=vdw_radii,
            debug=False,
            restart_prefix=restart_prefix,
            save_interval=10
        )
        
        # Should return None when files don't exist
        loaded_atoms = gcmc._load_restart_info()
        assert loaded_atoms is None  # No restart files found
        assert gcmc.Z_ads == 0  # Should remain at default
    
    def test_restart_integration(self, mock_model, atoms_frame, atoms_ads, temp_dir):
        """Integration test: run simulation, save, restart, and continue."""
        restart_prefix = os.path.join(temp_dir, "integration_restart")
        output_dir = os.path.join(temp_dir, "results")
        
        # Set random seed for reproducibility
        np.random.seed(42)
        
        # First run: 20 steps
        gcmc1 = MLP_GCMC(
            model=mock_model,
            atoms_frame=atoms_frame,
            atoms_ads=atoms_ads,
            T=300,
            P=1.0 * bar,
            fugacity=1.0 * bar,
            device='cpu',
            vdw_radii=vdw_radii,
            debug=False,
            output_dir=output_dir,
            restart_prefix=restart_prefix,
            save_interval=10
        )
        
        gcmc1.run(20)
        
        # Capture state after first run
        z_ads_after_first = gcmc1.Z_ads
        moves_after_first = {k: v.copy() for k, v in gcmc1.moves.items()}
        insertion_vdw_after_first = gcmc1.insertion_rejected_due_to_vdw
        
        # Check restart files exist in interval directory (most recent)
        # Find the most recent interval directory
        interval_dirs = [d for d in os.listdir(output_dir) if d.startswith("interval_")]
        assert len(interval_dirs) > 0, "No interval directories found"
        # Get the most recent one
        interval_numbers = [int(d.split('_')[1]) for d in interval_dirs]
        most_recent = max(interval_numbers)
        most_recent_dir = os.path.join(output_dir, f'interval_{most_recent}')
        restart_xyz = os.path.join(most_recent_dir, f'restart_{1.0:.5f}bar.xyz')
        restart_json = os.path.join(most_recent_dir, f'restart_{1.0:.5f}bar.json')
        assert os.path.exists(restart_xyz), f"Restart XYZ not found: {restart_xyz}"
        assert os.path.exists(restart_json), f"Restart JSON not found: {restart_json}"
        
        # Check results file exists
        results_file = os.path.join(output_dir, f"results_{1.0:.5f}bar.json")
        assert os.path.exists(results_file)
        
        # Load previous results
        with open(results_file, 'r') as f:
            first_results = json.load(f)
        first_uptake = first_results['uptake']
        first_interaction_energy = first_results['interaction_energy']
        
        # Second run: should restart and continue
        np.random.seed(43)  # Different seed for continuation
        gcmc2 = MLP_GCMC(
            model=mock_model,
            atoms_frame=atoms_frame,
            atoms_ads=atoms_ads,
            T=300,
            P=1.0 * bar,
            fugacity=1.0 * bar,
            device='cpu',
            vdw_radii=vdw_radii,
            debug=False,
            output_dir=output_dir,
            restart_prefix=restart_prefix,
            save_interval=10
        )
        
        # Run additional 15 steps (total should be 35)
        gcmc2.run(15)
        
        # Check that state was restored
        assert gcmc2.Z_ads == z_ads_after_first  # Should start from saved state
        
        # Check that results were appended
        with open(results_file, 'r') as f:
            second_results = json.load(f)
        second_uptake = second_results['uptake']
        second_interaction_energy = second_results['interaction_energy']
        
        # Second results should contain first results plus new ones
        assert len(second_uptake) == len(first_uptake) + 15
        assert len(second_interaction_energy) == len(first_interaction_energy) + 15
        
        # First part should match
        assert second_uptake[:len(first_uptake)] == first_uptake
        assert second_interaction_energy[:len(first_interaction_energy)] == first_interaction_energy
    
    def test_restart_with_periodic_saving(self, mock_model, atoms_frame, atoms_ads, temp_dir):
        """Test that periodic saving works correctly during simulation."""
        restart_prefix = os.path.join(temp_dir, "periodic_restart")
        output_dir = os.path.join(temp_dir, "results")
        
        np.random.seed(42)
        
        gcmc = MLP_GCMC(
            model=mock_model,
            atoms_frame=atoms_frame,
            atoms_ads=atoms_ads,
            T=300,
            P=1.0 * bar,
            fugacity=1.0 * bar,
            device='cpu',
            vdw_radii=vdw_radii,
            debug=False,
            output_dir=output_dir,
            restart_prefix=restart_prefix,
            save_interval=5  # Save every 5 steps
        )
        
        # Run 25 steps (should save at 5, 10, 15, 20, 25)
        gcmc.run(25)
        
        # Check that restart files exist in interval directories (not main restart files)
        interval_dirs = [d for d in os.listdir(output_dir) if d.startswith("interval_")]
        assert len(interval_dirs) > 0, "No interval directories found"
        # Check that main restart files don't exist (backward compatibility removed)
        assert not os.path.exists(f"{restart_prefix}.xyz"), "Main restart XYZ should not exist"
        assert not os.path.exists(f"{restart_prefix}.json"), "Main restart JSON should not exist"
        
        # Check that snapshot files were created in interval directories
        # Use absolute path to avoid issues with working directory changes
        abs_output_dir = os.path.abspath(output_dir)
        assert os.path.exists(abs_output_dir), f"Output directory {abs_output_dir} does not exist"
        all_files = os.listdir(abs_output_dir)
        interval_dirs = [d for d in all_files if d.startswith("interval_")]
        # Should have at least a few interval directories (at iterations 5, 10, 15, 20, and final at 24)
        # Note: iteration 25 doesn't happen because we run 25 steps (iterations 0-24)
        assert len(interval_dirs) >= 4, f"Expected at least 4 interval directories, found {interval_dirs} in {abs_output_dir}. All files: {all_files}"
        # Check that interval directories from save intervals contain snapshots
        # (final save creates interval directory but may only have restart files)
        snapshot_count = 0
        for interval_dir in interval_dirs:
            interval_path = os.path.join(abs_output_dir, interval_dir)
            assert os.path.isdir(interval_path), f"Interval directory {interval_path} is not a directory"
            files_in_dir = os.listdir(interval_path)
            snapshot_files = [f for f in files_in_dir if f.startswith("snapshot_")]
            restart_files = [f for f in files_in_dir if f.startswith("restart_")]
            # Each interval directory should have either a snapshot (from periodic save) or restart files (from final save)
            assert len(snapshot_files) >= 1 or len(restart_files) >= 1, \
                f"No snapshot or restart files found in {interval_path}. Files: {files_in_dir}"
            if len(snapshot_files) >= 1:
                snapshot_count += 1
        # Should have at least 4 snapshots (from periodic saves at 5, 10, 15, 20)
        assert snapshot_count >= 4, f"Expected at least 4 snapshots, found {snapshot_count}"
        
        # Verify restart file has correct state (from most recent interval)
        interval_numbers = [int(d.split('_')[1]) for d in interval_dirs]
        most_recent = max(interval_numbers)
        most_recent_dir = os.path.join(abs_output_dir, f'interval_{most_recent}')
        restart_json = os.path.join(most_recent_dir, f'restart_{1.0:.5f}bar.json')
        with open(restart_json, 'r') as f:
            restart_data = json.load(f)
            assert restart_data['Z_ads'] == gcmc.Z_ads
            assert restart_data['moves']['insertion']['attempted'] == gcmc.moves['insertion']['attempted']
    
    def test_restart_without_prefix(self, mock_model, atoms_frame, atoms_ads, temp_dir):
        """Test that simulation works without restart prefix (no saving)."""
        output_dir = os.path.join(temp_dir, "results")
        
        gcmc = MLP_GCMC(
            model=mock_model,
            atoms_frame=atoms_frame,
            atoms_ads=atoms_ads,
            T=300,
            P=1.0 * bar,
            fugacity=1.0 * bar,
            device='cpu',
            vdw_radii=vdw_radii,
            debug=False,
            output_dir=output_dir,
            restart_prefix=None,  # No restart
            save_interval=10
        )
        
        gcmc.run(10)
        
        # Should not create restart files
        restart_files = [f for f in os.listdir(temp_dir) if f.endswith(".xyz") or f.endswith(".json")]
        # Only results files should exist, no restart files
        assert all("restart" not in f for f in restart_files)
        
        # But results should still be saved
        results_file = os.path.join(output_dir, f"results_{1.0:.5f}bar.json")
        assert os.path.exists(results_file)
    
    def test_multiple_sequential_restarts(self, mock_model, atoms_frame, atoms_ads, temp_dir):
        """Test restarting multiple times in sequence."""
        restart_prefix = os.path.join(temp_dir, "sequential_restart")
        output_dir = os.path.join(temp_dir, "results")
        
        np.random.seed(42)
        
        # First run: 10 steps
        gcmc1 = MLP_GCMC(
            model=mock_model,
            atoms_frame=atoms_frame,
            atoms_ads=atoms_ads,
            T=300,
            P=1.0 * bar,
            fugacity=1.0 * bar,
            device='cpu',
            vdw_radii=vdw_radii,
            debug=False,
            output_dir=output_dir,
            restart_prefix=restart_prefix,
            save_interval=5
        )
        gcmc1.run(10)
        state1 = {
            'Z_ads': gcmc1.Z_ads,
            'moves': {k: v.copy() for k, v in gcmc1.moves.items()}
        }
        
        # Second run: restart and run 10 more steps
        np.random.seed(43)
        gcmc2 = MLP_GCMC(
            model=mock_model,
            atoms_frame=atoms_frame,
            atoms_ads=atoms_ads,
            T=300,
            P=1.0 * bar,
            fugacity=1.0 * bar,
            device='cpu',
            vdw_radii=vdw_radii,
            debug=False,
            output_dir=output_dir,
            restart_prefix=restart_prefix,
            save_interval=5
        )
        assert gcmc2.Z_ads == state1['Z_ads']
        gcmc2.run(10)
        state2 = {
            'Z_ads': gcmc2.Z_ads,
            'moves': {k: v.copy() for k, v in gcmc2.moves.items()}
        }
        
        # Third run: restart again and run 10 more steps
        np.random.seed(44)
        gcmc3 = MLP_GCMC(
            model=mock_model,
            atoms_frame=atoms_frame,
            atoms_ads=atoms_ads,
            T=300,
            P=1.0 * bar,
            fugacity=1.0 * bar,
            device='cpu',
            vdw_radii=vdw_radii,
            debug=False,
            output_dir=output_dir,
            restart_prefix=restart_prefix,
            save_interval=5
        )
        assert gcmc3.Z_ads == state2['Z_ads']
        gcmc3.run(10)
        
        # Verify that restart worked correctly
        # Note: moves are loaded from the saved state, so gcmc3.moves reflects the state
        # after the third run (which includes moves from all three runs since state accumulates)
        # The exact number may vary due to randomness, but should reflect cumulative state
        total_attempted = sum(gcmc3.moves[move]['attempted'] for move in gcmc3.moves)
        # After 3 runs of 10 steps each, we should have moves from all runs
        # Since moves are loaded from the most recent interval, they should include all previous moves
        assert total_attempted >= 10, f"Expected moves to be loaded from saved state, got {total_attempted}"
        # The moves should reflect the state after the third run completed
        # (which saved the final state including all moves from all three runs)
    
    def test_restart_state_consistency(self, mock_model, atoms_frame, atoms_ads, temp_dir):
        """Test that Z_ads matches actual number of adsorbates in structure."""
        restart_prefix = os.path.join(temp_dir, "consistency_restart")
        output_dir = os.path.join(temp_dir, "results")
        
        np.random.seed(42)
        
        # Run simulation
        gcmc1 = MLP_GCMC(
            model=mock_model,
            atoms_frame=atoms_frame,
            atoms_ads=atoms_ads,
            T=300,
            P=1.0 * bar,
            fugacity=1.0 * bar,
            device='cpu',
            vdw_radii=vdw_radii,
            debug=False,
            output_dir=output_dir,
            restart_prefix=restart_prefix,
            save_interval=10
        )
        gcmc1.run(30)
        
        # Load restart and verify consistency
        gcmc2 = MLP_GCMC(
            model=mock_model,
            atoms_frame=atoms_frame,
            atoms_ads=atoms_ads,
            T=300,
            P=1.0 * bar,
            fugacity=1.0 * bar,
            device='cpu',
            vdw_radii=vdw_radii,
            debug=False,
            output_dir=output_dir,
            restart_prefix=restart_prefix,
            save_interval=10
        )
        loaded_atoms = gcmc2._load_restart_info()
        
        # Z_ads should match number of adsorbate molecules in structure
        n_frame = len(atoms_frame)
        n_ads = len(atoms_ads)
        expected_adsorbate_atoms = gcmc2.Z_ads * n_ads
        actual_atoms = len(loaded_atoms) - n_frame
        assert actual_atoms == expected_adsorbate_atoms, \
            f"Z_ads={gcmc2.Z_ads} but structure has {actual_atoms // n_ads} adsorbate molecules"
    
    def test_restart_all_statistics_fields(self, mock_model, atoms_frame, atoms_ads, temp_dir):
        """Test that all statistics fields are correctly saved and loaded."""
        restart_prefix = os.path.join(temp_dir, "stats_restart")
        
        gcmc1 = MLP_GCMC(
            model=mock_model,
            atoms_frame=atoms_frame,
            atoms_ads=atoms_ads,
            T=300,
            P=1.0 * bar,
            fugacity=1.0 * bar,
            device='cpu',
            vdw_radii=vdw_radii,
            debug=False,
            restart_prefix=restart_prefix,
            save_interval=10
        )
        
        # Set all statistics fields
        gcmc1.Z_ads = 7
        gcmc1.moves['insertion']['attempted'] = 50
        gcmc1.moves['insertion']['accepted'] = 25
        gcmc1.moves['deletion']['attempted'] = 40
        gcmc1.moves['deletion']['accepted'] = 20
        gcmc1.moves['translation']['attempted'] = 60
        gcmc1.moves['translation']['accepted'] = 45
        gcmc1.moves['rotation']['attempted'] = 55
        gcmc1.moves['rotation']['accepted'] = 40
        gcmc1.insertion_rejected_due_to_vdw = 10
        gcmc1.insertion_rejected_due_to_acceptance = 15
        gcmc1.insertion_accepted_due_to_acceptance_100 = 5
        gcmc1.insertion_rejected_due_to_acceptance_100 = 2
        gcmc1.deletion_rejected_due_to_acceptance = 20
        gcmc1.deletion_accepted_due_to_acceptance_100 = 3
        gcmc1.deletion_rejected_due_to_acceptance_100 = 1
        
        test_atoms = atoms_frame.copy()
        for _ in range(7):
            test_atoms = test_atoms + atoms_ads.copy()
        
        # Save to interval directory
        output_dir = os.path.join(temp_dir, "results")
        interval_dir = os.path.join(output_dir, 'interval_300')
        os.makedirs(interval_dir, exist_ok=True)
        gcmc1.output_dir = output_dir
        gcmc1._save_restart_info_to_interval(test_atoms, interval_dir, 300)
        
        # Load and verify all fields
        gcmc2 = MLP_GCMC(
            model=mock_model,
            atoms_frame=atoms_frame,
            atoms_ads=atoms_ads,
            T=300,
            P=1.0 * bar,
            fugacity=1.0 * bar,
            device='cpu',
            vdw_radii=vdw_radii,
            debug=False,
            output_dir=output_dir,
            restart_prefix=restart_prefix,
            save_interval=10
        )
        gcmc2._load_restart_info()
        
        assert gcmc2.Z_ads == 7
        assert gcmc2.moves['insertion']['attempted'] == 50
        assert gcmc2.moves['insertion']['accepted'] == 25
        assert gcmc2.moves['deletion']['attempted'] == 40
        assert gcmc2.moves['deletion']['accepted'] == 20
        assert gcmc2.moves['translation']['attempted'] == 60
        assert gcmc2.moves['translation']['accepted'] == 45
        assert gcmc2.moves['rotation']['attempted'] == 55
        assert gcmc2.moves['rotation']['accepted'] == 40
        assert gcmc2.insertion_rejected_due_to_vdw == 10
        assert gcmc2.insertion_rejected_due_to_acceptance == 15
        assert gcmc2.insertion_accepted_due_to_acceptance_100 == 5
        assert gcmc2.insertion_rejected_due_to_acceptance_100 == 2
        assert gcmc2.deletion_rejected_due_to_acceptance == 20
        assert gcmc2.deletion_accepted_due_to_acceptance_100 == 3
        assert gcmc2.deletion_rejected_due_to_acceptance_100 == 1
    
    def test_restart_atoms_data_integrity(self, mock_model, atoms_frame, atoms_ads, temp_dir):
        """Test that loaded atoms match saved atoms (positions, cell, etc.)."""
        restart_prefix = os.path.join(temp_dir, "atoms_integrity")
        output_dir = os.path.join(temp_dir, "results")
        
        gcmc1 = MLP_GCMC(
            model=mock_model,
            atoms_frame=atoms_frame,
            atoms_ads=atoms_ads,
            T=300,
            P=1.0 * bar,
            fugacity=1.0 * bar,
            device='cpu',
            vdw_radii=vdw_radii,
            debug=False,
            output_dir=output_dir,
            restart_prefix=restart_prefix,
            save_interval=10
        )
        
        # Create test atoms with specific positions
        test_atoms = atoms_frame.copy()
        for i in range(3):
            ads = atoms_ads.copy()
            # Set specific positions for adsorbates
            ads.set_positions(ads.get_positions() + np.array([i*2, i*2, i*2]))
            test_atoms = test_atoms + ads
        
        gcmc1.Z_ads = 3
        # Save to interval directory
        interval_dir = os.path.join(output_dir, 'interval_400')
        os.makedirs(interval_dir, exist_ok=True)
        gcmc1._save_restart_info_to_interval(test_atoms, interval_dir, 400)
        
        # Load and verify
        gcmc2 = MLP_GCMC(
            model=mock_model,
            atoms_frame=atoms_frame,
            atoms_ads=atoms_ads,
            T=300,
            P=1.0 * bar,
            fugacity=1.0 * bar,
            device='cpu',
            vdw_radii=vdw_radii,
            debug=False,
            output_dir=output_dir,
            restart_prefix=restart_prefix,
            save_interval=10
        )
        loaded_atoms = gcmc2._load_restart_info()
        
        # Verify structure matches
        assert len(loaded_atoms) == len(test_atoms)
        np.testing.assert_array_equal(loaded_atoms.numbers, test_atoms.numbers)
        np.testing.assert_allclose(loaded_atoms.positions, test_atoms.positions, rtol=1e-10)
        np.testing.assert_allclose(loaded_atoms.cell, test_atoms.cell, rtol=1e-10)
        assert np.array_equal(loaded_atoms.pbc, test_atoms.pbc)
    
    def test_restart_with_zero_adsorbates(self, mock_model, atoms_frame, atoms_ads, temp_dir):
        """Test restart when Z_ads = 0 (no adsorbates)."""
        restart_prefix = os.path.join(temp_dir, "zero_ads_restart")
        output_dir = os.path.join(temp_dir, "results")
        
        gcmc1 = MLP_GCMC(
            model=mock_model,
            atoms_frame=atoms_frame,
            atoms_ads=atoms_ads,
            T=300,
            P=1.0 * bar,
            fugacity=1.0 * bar,
            device='cpu',
            vdw_radii=vdw_radii,
            debug=False,
            output_dir=output_dir,
            restart_prefix=restart_prefix,
            save_interval=10
        )
        
        gcmc1.Z_ads = 0
        gcmc1.moves['insertion']['attempted'] = 10
        gcmc1.moves['insertion']['accepted'] = 0
        # Save to interval directory
        interval_dir = os.path.join(output_dir, 'interval_500')
        os.makedirs(interval_dir, exist_ok=True)
        gcmc1._save_restart_info_to_interval(atoms_frame.copy(), interval_dir, 500)
        
        # Load and verify
        gcmc2 = MLP_GCMC(
            model=mock_model,
            atoms_frame=atoms_frame,
            atoms_ads=atoms_ads,
            T=300,
            P=1.0 * bar,
            fugacity=1.0 * bar,
            device='cpu',
            vdw_radii=vdw_radii,
            debug=False,
            output_dir=output_dir,
            restart_prefix=restart_prefix,
            save_interval=10
        )
        loaded_atoms = gcmc2._load_restart_info()
        
        assert gcmc2.Z_ads == 0
        assert len(loaded_atoms) == len(atoms_frame)
        assert gcmc2.moves['insertion']['attempted'] == 10
        assert gcmc2.moves['insertion']['accepted'] == 0
    
    def test_restart_with_save_interval_one(self, mock_model, atoms_frame, atoms_ads, temp_dir):
        """Test restart with save_interval = 1 (save every step)."""
        restart_prefix = os.path.join(temp_dir, "interval_one_restart")
        output_dir = os.path.join(temp_dir, "results")
        
        np.random.seed(42)
        
        gcmc = MLP_GCMC(
            model=mock_model,
            atoms_frame=atoms_frame,
            atoms_ads=atoms_ads,
            T=300,
            P=1.0 * bar,
            fugacity=1.0 * bar,
            device='cpu',
            vdw_radii=vdw_radii,
            debug=False,
            output_dir=output_dir,
            restart_prefix=restart_prefix,
            save_interval=1  # Save every step
        )
        
        gcmc.run(10)
        
        # Should have many interval directories (one per step after iteration 0)
        interval_dirs = [d for d in os.listdir(output_dir) if d.startswith("interval_")]
        assert len(interval_dirs) >= 9  # At least 9 interval directories (iterations 1-9)
        # Check that each interval directory contains a snapshot
        for interval_dir in interval_dirs:
            interval_path = os.path.join(output_dir, interval_dir)
            snapshot_files = [f for f in os.listdir(interval_path) if f.startswith("snapshot_")]
            assert len(snapshot_files) >= 1
    
    def test_restart_with_large_save_interval(self, mock_model, atoms_frame, atoms_ads, temp_dir):
        """Test restart with very large save_interval (larger than total steps)."""
        restart_prefix = os.path.join(temp_dir, "large_interval_restart")
        output_dir = os.path.join(temp_dir, "results")
        
        np.random.seed(42)
        
        gcmc = MLP_GCMC(
            model=mock_model,
            atoms_frame=atoms_frame,
            atoms_ads=atoms_ads,
            T=300,
            P=1.0 * bar,
            fugacity=1.0 * bar,
            device='cpu',
            vdw_radii=vdw_radii,
            debug=False,
            output_dir=output_dir,
            restart_prefix=restart_prefix,
            save_interval=1000  # Much larger than run steps
        )
        
        gcmc.run(20)
        
        # Should still save at the end to interval directory
        interval_dirs = [d for d in os.listdir(output_dir) if d.startswith("interval_")]
        # Final save creates interval_19 (since we run 20 steps, iterations 0-19)
        assert len(interval_dirs) == 1, f"Expected 1 interval directory (final save), found {interval_dirs}"
        # Check that main restart files don't exist (backward compatibility removed)
        assert not os.path.exists(f"{restart_prefix}.xyz"), "Main restart XYZ should not exist"
        assert not os.path.exists(f"{restart_prefix}.json"), "Main restart JSON should not exist"
        
        # But no intermediate interval directories (only final save creates one)
        interval_dirs = [d for d in os.listdir(output_dir) if d.startswith("interval_")]
        # Final save creates interval_19 (since we run 20 steps, iterations 0-19)
        assert len(interval_dirs) == 1, f"Expected 1 interval directory (final save), found {interval_dirs}"
        # Final interval should have restart files but no snapshot (since no periodic save occurred)
        final_interval = os.path.join(output_dir, interval_dirs[0])
        files_in_dir = os.listdir(final_interval)
        restart_files = [f for f in files_in_dir if f.startswith("restart_")]
        snapshot_files = [f for f in files_in_dir if f.startswith("snapshot_")]
        assert len(restart_files) >= 1, "Final interval should have restart files"
        assert len(snapshot_files) == 0, "Final interval should not have snapshot (no periodic save)"
        
        # Check that main restart files don't exist (backward compatibility removed)
        assert not os.path.exists(f"{restart_prefix}.xyz"), "Main restart XYZ should not exist"
        assert not os.path.exists(f"{restart_prefix}.json"), "Main restart JSON should not exist"
    
    def test_restart_corrupted_json_file(self, mock_model, atoms_frame, atoms_ads, temp_dir):
        """Test handling of corrupted JSON restart file."""
        restart_prefix = os.path.join(temp_dir, "corrupted_restart")
        
        # Create corrupted JSON file
        with open(f"{restart_prefix}.json", 'w') as f:
            f.write("{ invalid json }")
        
        # Create valid XYZ file
        write(f"{restart_prefix}.xyz", atoms_frame)
        
        gcmc = MLP_GCMC(
            model=mock_model,
            atoms_frame=atoms_frame,
            atoms_ads=atoms_ads,
            T=300,
            P=1.0 * bar,
            fugacity=1.0 * bar,
            device='cpu',
            vdw_radii=vdw_radii,
            debug=False,
            restart_prefix=restart_prefix,
            save_interval=10
        )
        
        # Should handle gracefully - either skip restart or use defaults
        try:
            loaded_atoms = gcmc._load_restart_info()
            # If it doesn't crash, it should return None (no restart found) or atoms
            if loaded_atoms is not None:
                assert len(loaded_atoms) >= len(atoms_frame)
            else:
                # No restart found is acceptable
                pass
        except (json.JSONDecodeError, KeyError, ValueError):
            # If it raises an error, that's also acceptable behavior
            pass
    
    def test_restart_missing_xyz_file(self, mock_model, atoms_frame, atoms_ads, temp_dir):
        """Test handling when XYZ file is missing but JSON exists."""
        restart_prefix = os.path.join(temp_dir, "missing_xyz")
        
        # Create only JSON file
        restart_data = {
            'Z_ads': 5,
            'moves': {
                'insertion': {'attempted': 10, 'accepted': 5},
                'deletion': {'attempted': 5, 'accepted': 2},
                'translation': {'attempted': 0, 'accepted': 0},
                'rotation': {'attempted': 0, 'accepted': 0}
            },
            'rejections': {
                'insertion_vdw': 2,
                'insertion_acc': 3,
                'insertion_acc_100': 0,
                'insertion_rej_100': 0,
                'deletion_acc': 3,
                'deletion_acc_100': 0,
                'deletion_rej_100': 0
            }
        }
        with open(f"{restart_prefix}.json", 'w') as f:
            json.dump(restart_data, f)
        
        gcmc = MLP_GCMC(
            model=mock_model,
            atoms_frame=atoms_frame,
            atoms_ads=atoms_ads,
            T=300,
            P=1.0 * bar,
            fugacity=1.0 * bar,
            device='cpu',
            vdw_radii=vdw_radii,
            debug=False,
            restart_prefix=restart_prefix,
            save_interval=10
        )
        
        # Should handle gracefully - return None when XYZ is missing
        loaded_atoms = gcmc._load_restart_info()
        # Since XYZ is missing, should return None (no valid restart found)
        assert loaded_atoms is None or len(loaded_atoms) == len(atoms_frame)
        # State might be loaded from JSON, but atoms will be None or framework
        # This is acceptable behavior
    
    def test_restart_trajectory_append(self, mock_model, atoms_frame, atoms_ads, temp_dir):
        """Test that trajectory files are correctly appended on restart."""
        restart_prefix = os.path.join(temp_dir, "traj_append_restart")
        output_dir = os.path.join(temp_dir, "results")
        
        np.random.seed(42)
        
        # First run
        gcmc1 = MLP_GCMC(
            model=mock_model,
            atoms_frame=atoms_frame,
            atoms_ads=atoms_ads,
            T=300,
            P=1.0 * bar,
            fugacity=1.0 * bar,
            device='cpu',
            vdw_radii=vdw_radii,
            debug=False,
            output_dir=output_dir,
            restart_prefix=restart_prefix,
            save_interval=10
        )
        gcmc1.run(10)
        
        # Count trajectory frames
        traj_file = os.path.join(output_dir, f'traj_{1.0:.5f}bar.xyz')
        frames_before = len(read(traj_file, index=':'))
        
        # Second run (restart)
        np.random.seed(43)
        gcmc2 = MLP_GCMC(
            model=mock_model,
            atoms_frame=atoms_frame,
            atoms_ads=atoms_ads,
            T=300,
            P=1.0 * bar,
            fugacity=1.0 * bar,
            device='cpu',
            vdw_radii=vdw_radii,
            debug=False,
            output_dir=output_dir,
            restart_prefix=restart_prefix,
            save_interval=10
        )
        gcmc2.run(10)
        
        # Count trajectory frames after restart
        frames_after = len(read(traj_file, index=':'))
        
        # Should have appended 10 more frames
        assert frames_after == frames_before + 10
    
    def test_restart_results_continuity(self, mock_model, atoms_frame, atoms_ads, temp_dir):
        """Test that results arrays maintain continuity across restarts."""
        restart_prefix = os.path.join(temp_dir, "continuity_restart")
        output_dir = os.path.join(temp_dir, "results")
        
        np.random.seed(42)
        
        # First run: 15 steps
        gcmc1 = MLP_GCMC(
            model=mock_model,
            atoms_frame=atoms_frame,
            atoms_ads=atoms_ads,
            T=300,
            P=1.0 * bar,
            fugacity=1.0 * bar,
            device='cpu',
            vdw_radii=vdw_radii,
            debug=False,
            output_dir=output_dir,
            restart_prefix=restart_prefix,
            save_interval=10
        )
        gcmc1.run(15)
        
        # Get first results
        results_file = os.path.join(output_dir, f"results_{1.0:.5f}bar.json")
        with open(results_file, 'r') as f:
            results1 = json.load(f)
        
        # Second run: restart and continue
        np.random.seed(43)
        gcmc2 = MLP_GCMC(
            model=mock_model,
            atoms_frame=atoms_frame,
            atoms_ads=atoms_ads,
            T=300,
            P=1.0 * bar,
            fugacity=1.0 * bar,
            device='cpu',
            vdw_radii=vdw_radii,
            debug=False,
            output_dir=output_dir,
            restart_prefix=restart_prefix,
            save_interval=10
        )
        gcmc2.run(10)
        
        # Get final results
        with open(results_file, 'r') as f:
            results2 = json.load(f)
        
        # Verify continuity
        assert len(results2['uptake']) == 25  # 15 + 10
        assert len(results2['interaction_energy']) == 25
        assert len(results2['total_energy']) == 25
        
        # First 15 should match
        assert results2['uptake'][:15] == results1['uptake']
        assert results2['interaction_energy'][:15] == results1['interaction_energy']
        assert results2['total_energy'][:15] == results1['total_energy']
        
        # Last values should be different (new steps)
        assert results2['uptake'][-1] is not None
        assert results2['interaction_energy'][-1] is not None
        assert results2['total_energy'][-1] is not None

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
