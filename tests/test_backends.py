"""
Unit tests for backend detection and model loading in main.py.

The heavy MLIP packages are not test dependencies, so these tests fake
the relevant modules in sys.modules and verify the dispatch logic.
"""
import sys
import types

import pytest

from mlip_mc.main import _detect_backend, _load_model


class FakeMetatomicCalculator:
    """Stands in for metatomic.torch.ase_calculator.MetatomicCalculator."""

    def __init__(self, model, extensions_directory=None, device=None):
        self.model = model
        self.extensions_directory = extensions_directory
        self.device = device


@pytest.fixture
def fake_metatomic(monkeypatch):
    """Install a fake metatomic.torch.ase_calculator into sys.modules."""
    metatomic = types.ModuleType('metatomic')
    metatomic_torch = types.ModuleType('metatomic.torch')
    ase_calculator = types.ModuleType('metatomic.torch.ase_calculator')
    ase_calculator.MetatomicCalculator = FakeMetatomicCalculator
    metatomic.torch = metatomic_torch
    metatomic_torch.ase_calculator = ase_calculator
    monkeypatch.setitem(sys.modules, 'metatomic', metatomic)
    monkeypatch.setitem(sys.modules, 'metatomic.torch', metatomic_torch)
    monkeypatch.setitem(sys.modules, 'metatomic.torch.ase_calculator', ase_calculator)
    return ase_calculator


@pytest.fixture
def no_other_backends(monkeypatch):
    """Force ImportError for the other MLIP backends."""
    for name in ('fairchem', 'fairchem.core', 'mace', 'orb_models'):
        monkeypatch.setitem(sys.modules, name, None)


class TestDetectBackend:

    def test_detects_metatomic(self, fake_metatomic, no_other_backends):
        assert _detect_backend() == 'metatomic'

    def test_no_backend_raises(self, monkeypatch, no_other_backends):
        monkeypatch.setitem(sys.modules, 'metatomic', None)
        monkeypatch.setitem(sys.modules, 'metatomic.torch', None)
        with pytest.raises(ImportError, match='metatomic'):
            _detect_backend()


class TestLoadModelMetatomic:

    def test_loads_with_extensions_dir(self, fake_metatomic, tmp_path):
        model_file = tmp_path / 'model.pt'
        model_file.touch()
        extensions = tmp_path / 'extensions'
        extensions.mkdir()

        calc = _load_model(str(model_file), device='cpu', backend='metatomic')
        assert isinstance(calc, FakeMetatomicCalculator)
        assert calc.model == str(model_file)
        assert calc.extensions_directory == str(extensions)
        assert calc.device == 'cpu'

    def test_loads_without_extensions_dir(self, fake_metatomic, tmp_path):
        model_file = tmp_path / 'model.pt'
        model_file.touch()

        calc = _load_model(str(model_file), device='cuda', backend='metatomic')
        assert isinstance(calc, FakeMetatomicCalculator)
        assert calc.extensions_directory is None
        assert calc.device == 'cuda'

    def test_missing_model_file_raises(self, fake_metatomic, tmp_path):
        with pytest.raises(FileNotFoundError, match='mtt export'):
            _load_model(str(tmp_path / 'nope.pt'), device='cpu', backend='metatomic')

    def test_unknown_backend_raises(self):
        with pytest.raises(ValueError, match='Unknown backend'):
            _load_model('model.pt', device='cpu', backend='not-a-backend')

    def test_explicit_backend_bypasses_detection(self, fake_metatomic, tmp_path, monkeypatch):
        # With mace "installed" (auto-detection would pick it first), an
        # explicit backend must still select metatomic
        monkeypatch.setitem(sys.modules, 'mace', types.ModuleType('mace'))
        model_file = tmp_path / 'model.pt'
        model_file.touch()
        calc = _load_model(str(model_file), device='cpu', backend='metatomic')
        assert isinstance(calc, FakeMetatomicCalculator)


class TestBackendCLI:

    def _parse(self, monkeypatch, extra):
        from mlip_mc.cli import parse_arguments
        argv = ['mlip_mc', '--adsorbent', 'f.cif', '--adsorbate-molecule', 'CO2',
                '--temperature', '298', '--model', 'm.pt'] + extra
        monkeypatch.setattr(sys, 'argv', argv)
        return parse_arguments()

    def test_backend_flag_default_none(self, monkeypatch):
        args = self._parse(monkeypatch, [])
        assert args.backend is None

    def test_backend_flag_accepts_metatomic(self, monkeypatch):
        args = self._parse(monkeypatch, ['--backend', 'metatomic'])
        assert args.backend == 'metatomic'

    def test_backend_flag_rejects_unknown(self, monkeypatch, capsys):
        with pytest.raises(SystemExit):
            self._parse(monkeypatch, ['--backend', 'quantum-espresso'])
