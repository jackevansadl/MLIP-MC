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


class FakeSystem:
    """Duck-typed metatomic System with a torch-like device attribute."""

    def __init__(self, device):
        self.device = types.SimpleNamespace(type=device) if isinstance(device, str) else device
        self._device_str = device if isinstance(device, str) else device.type

    def to(self, device):
        return FakeSystem(device)


class TestPatchMetatomicForRocm:

    def _make_calc(self):
        seen = {}

        class FakeNL:
            def compute(self, systems):
                seen['devices'] = [s._device_str for s in systems]
                return systems

        calc = types.SimpleNamespace(_nl_calculators=FakeNL())
        return calc, seen

    def test_noop_without_hip(self, monkeypatch):
        import torch
        from mlip_mc.main import patch_metatomic_for_rocm
        monkeypatch.setattr(torch.version, 'hip', None, raising=False)
        calc, _ = self._make_calc()
        assert patch_metatomic_for_rocm(calc) is calc
        # The patch installs an instance-level override; without HIP the
        # class method must remain untouched
        assert 'compute' not in vars(calc._nl_calculators)

    def test_hip_computes_nl_on_cpu_and_moves_back(self, monkeypatch):
        import torch
        from mlip_mc.main import patch_metatomic_for_rocm
        monkeypatch.setattr(torch.version, 'hip', '6.3.42', raising=False)
        calc, seen = self._make_calc()
        patch_metatomic_for_rocm(calc)

        out = calc._nl_calculators.compute([FakeSystem('cuda'), FakeSystem('cuda')])
        # vesin must only ever see CPU systems...
        assert seen['devices'] == ['cpu', 'cpu']
        # ...and the results must come back on the original device
        assert [s._device_str for s in out] == ['cuda', 'cuda']

    def test_hip_missing_internals_warns_but_returns(self, monkeypatch, capsys):
        import torch
        from mlip_mc.main import patch_metatomic_for_rocm
        monkeypatch.setattr(torch.version, 'hip', '6.3.42', raising=False)
        calc = types.SimpleNamespace()  # no _nl_calculators
        assert patch_metatomic_for_rocm(calc) is calc
        assert 'could not patch' in capsys.readouterr().err


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
