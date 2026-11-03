"""Packaging: the non-Python model and composite files must ship in the wheel."""

import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REQUIRED_PATTERNS = ['models/*.bngl', 'composites/*.composite.yaml']


def _package_data():
    pyproject = tomllib.loads((ROOT / 'pyproject.toml').read_text())
    return pyproject['tool']['setuptools']['package-data']['pbg_nfsim']


def test_package_data_covers_bngl_and_composite_specs():
    """Regression: packages.find ships only .py files, so the built wheel
    omitted flagella_complexation.bngl and the *.composite.yaml specs."""
    declared = _package_data()
    for pattern in REQUIRED_PATTERNS:
        assert pattern in declared, f'{pattern} missing from package-data'


def test_every_shipped_data_file_is_declared():
    """Every non-Python file under pbg_nfsim must match a declared pattern."""
    package_dir = ROOT / 'pbg_nfsim'
    declared = _package_data()
    shipped = [
        p for p in package_dir.rglob('*')
        if p.is_file() and p.suffix not in ('.py', '.pyc') and '__pycache__' not in p.parts
    ]
    assert shipped, 'expected bngl and composite files in pbg_nfsim'
    for path in shipped:
        rel = path.relative_to(package_dir).as_posix()
        assert any(path.match(str(package_dir / p)) for p in declared), (
            f'{rel} is not covered by package-data {declared}')
