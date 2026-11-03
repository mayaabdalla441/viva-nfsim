"""Integration tests for NFSim composites."""

import pytest
from process_bigraph import Composite, allocate_core
from process_bigraph.emitter import RAMEmitter
from pbg_nfsim.processes import NFSimProcess, MonomerProduction
from pbg_nfsim.composites import make_complexation_document, make_production_document


@pytest.fixture
def core():
    c = allocate_core()
    c.register_link('nfsim', NFSimProcess)
    c.register_link('monomer-production', MonomerProduction)
    c.register_link('ram-emitter', RAMEmitter)
    return c


def test_complexation_assembly(core):
    doc = make_complexation_document(n_steps=10, interval=50.0)
    sim = Composite({'state': doc}, core=core)
    assert sim is not None


def test_production_assembly(core):
    doc = make_production_document(
        n_steps=10,
        complexation_interval=50.0,
        production_interval=1.0)
    sim = Composite({'state': doc}, core=core)
    assert sim is not None


def test_document_factory_params(core):
    doc = make_complexation_document(n_steps=50, interval=25.0)
    assert doc['complexation']['config']['n_steps'] == 50
    assert doc['complexation']['interval'] == 25.0


def test_production_document_rates(core):
    doc = make_production_document(production_rate_scale=2.0)
    rates = doc['production']['config']['production_rates']
    assert len(rates) > 0
    # All rates should be positive (scaled by 2.0)
    for name, rate in rates.items():
        assert rate > 0


_BLOCK_VIVA_SUPERPOWERS = """
import sys
sys.modules['viva_superpowers'] = None
from pbg_nfsim.composites import build_composite
sim = build_composite('flagella-assembly-demo', overrides={'n_steps': 5})
assert sim is not None
print('OK')
"""


def test_default_composite_builds_without_viva_superpowers():
    """Regression: with viva_superpowers absent (or blocked), the default
    composite path must build. The check runs in a subprocess so the
    blocked module cannot leak into other tests' import state."""
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, '-c', _BLOCK_VIVA_SUPERPOWERS],
        capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stderr
    assert 'OK' in result.stdout


def test_register_nfsim_reraises_unrelated_import_errors(monkeypatch):
    """Regression: register_nfsim swallowed every ImportError, so a broken
    visualization dependency made FlagellaAssemblyPlots vanish silently.
    Only a missing viva_superpowers may be skipped; anything else must raise."""
    import builtins

    from pbg_nfsim.composites import register_nfsim

    real_import = builtins.__import__

    def failing_import(name, *args, **kwargs):
        if name == 'pbg_nfsim.visualizations':
            raise ModuleNotFoundError("No module named 'numpy'", name='numpy')
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', failing_import)
    with pytest.raises(ModuleNotFoundError, match='numpy'):
        register_nfsim()


def test_register_nfsim_reraises_missing_viva_submodule(monkeypatch):
    """Regression: only a missing viva_superpowers package is skippable. A
    missing submodule of an installed viva_superpowers must surface."""
    import builtins

    from pbg_nfsim.composites import register_nfsim

    real_import = builtins.__import__

    def failing_import(name, *args, **kwargs):
        if name == 'pbg_nfsim.visualizations':
            raise ModuleNotFoundError(
                "No module named 'viva_superpowers.visualization'",
                name='viva_superpowers.visualization')
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', failing_import)
    with pytest.raises(ModuleNotFoundError, match='viva_superpowers.visualization'):
        register_nfsim()
