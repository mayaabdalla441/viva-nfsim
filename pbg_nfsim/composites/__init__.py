"""NFSim composite documents + composite-spec discovery.

Two flavors of composite construction live in this package:

1. **Hand-coded factories** — `make_complexation_document()` and
   `make_production_document()` build a PBG state-dict programmatically
   for callers that want full control over the BNGL model + wiring.
   Used by `demo/demo_report.py` for the multi-configuration assembly
   experiments.

2. **Declarative `*.composite.yaml`** — sibling files in this directory
   follow the viva-superpowers composite-spec convention.
   `build_composite()` loads one by name and instantiates
   `process_bigraph.Composite` with parameter substitution. The
   dashboard's composite explorer discovers these automatically once
   the package is installed in a workspace.

Both flavors are equivalent — pick the one that fits your use case.
"""
from __future__ import annotations
import importlib.util
import re
from pathlib import Path
from typing import Any

import yaml
from process_bigraph import allocate_core
from process_bigraph.emitter import RAMEmitter

from pbg_nfsim.processes import NFSimProcess, MonomerProduction
from pbg_nfsim.models.generate_flagella_bngl import (
    get_model_path,
    default_production_rates,
)


# ---------------------------------------------------------------------------
# Hand-coded composite factories (legacy / programmatic API)
# ---------------------------------------------------------------------------


def make_complexation_document(
    model_file=None,
    n_steps=100,
    interval=50.0,
):
    """Create a composite document for standalone NFSim complexation.

    Returns a document dict ready for use with Composite().

    Args:
        model_file: Path to BNGL model file (defaults to flagella model)
        n_steps: Number of NFSim steps per interval
        interval: Time interval between process updates (seconds)

    Returns:
        dict: Composite document with NFSim process, stores, and emitter
    """
    if model_file is None:
        model_file = get_model_path()

    return {
        'complexation': {
            '_type': 'process',
            'address': 'local:nfsim',
            'config': {
                'model_file': model_file,
                'n_steps': n_steps,
            },
            'inputs': {
                'observables': ['species'],
            },
            'outputs': {
                'observables': ['species'],
            },
            'interval': interval,
        },
        'species': {},
        'emitter': {
            '_type': 'step',
            'address': 'local:ram-emitter',
            'config': {
                'emit': {
                    'species': 'map[float]',
                    'time': 'float',
                },
            },
            'inputs': {
                'species': ['species'],
                'time': ['global_time'],
            },
        },
    }


def make_production_document(
    model_file=None,
    n_steps=100,
    complexation_interval=50.0,
    production_interval=1.0,
    production_rate_scale=1.0,
):
    """Create a composite document for production + complexation.

    Composes MonomerProduction and NFSimProcess wired to shared species store.

    Args:
        model_file: Path to BNGL model file (defaults to flagella model)
        n_steps: Number of NFSim steps per complexation interval
        complexation_interval: Time interval for NFSim process (seconds)
        production_interval: Time interval for monomer production (seconds)
        production_rate_scale: Multiplier for production rates

    Returns:
        dict: Composite document with production, complexation, stores, emitter
    """
    if model_file is None:
        model_file = get_model_path()

    rates = default_production_rates()
    scaled_rates = {
        name: rate * production_rate_scale
        for name, rate in rates.items()
    }

    return {
        'production': {
            '_type': 'process',
            'address': 'local:monomer-production',
            'config': {
                'production_rates': scaled_rates,
            },
            'outputs': {
                'monomers': ['species'],
            },
            'interval': production_interval,
        },
        'complexation': {
            '_type': 'process',
            'address': 'local:nfsim',
            'config': {
                'model_file': model_file,
                'n_steps': n_steps,
            },
            'inputs': {
                'observables': ['species'],
            },
            'outputs': {
                'observables': ['species'],
            },
            'interval': complexation_interval,
        },
        'species': {},
        'emitter': {
            '_type': 'step',
            'address': 'local:ram-emitter',
            'config': {
                'emit': {
                    'species': 'map[float]',
                    'time': 'float',
                },
            },
            'inputs': {
                'species': ['species'],
                'time': ['global_time'],
            },
        },
    }


def register_nfsim(core=None):
    """Return a core with NFSimProcess, MonomerProduction, the RAM emitter,
    and, when viva-superpowers is installed (`dashboard` extra), the
    flagella-assembly Visualization Step registered."""
    if core is None:
        core = allocate_core()
    core.register_link('NFSimProcess', NFSimProcess)
    core.register_link('nfsim', NFSimProcess)
    core.register_link('MonomerProduction', MonomerProduction)
    core.register_link('monomer-production', MonomerProduction)
    core.register_link('RAMEmitter', RAMEmitter)
    core.register_link('ram-emitter', RAMEmitter)
    # Register Visualization Steps so composites can wire them by name.
    # viva-superpowers is the optional `dashboard` extra. Skip only when it
    # is missing; any other missing module is a real error and must surface.
    try:
        from pbg_nfsim.visualizations import FlagellaAssemblyPlots
    except ModuleNotFoundError as exc:
        # find_spec returns None when the package is absent or blocked in
        # sys.modules; an installed package with a missing submodule must raise.
        viva_absent = importlib.util.find_spec('viva_superpowers') is None
        if not (viva_absent and (exc.name or '').startswith('viva_superpowers')):
            raise
        # viva-superpowers not installed: viz composites are unavailable,
        # the rest of the package works.
    else:
        core.register_link('FlagellaAssemblyPlots', FlagellaAssemblyPlots)
    return core


# ---------------------------------------------------------------------------
# Declarative composite-spec loader (*.composite.yaml)
# ---------------------------------------------------------------------------

_COMPOSITES_DIR = Path(__file__).parent

_FULL_PLACEHOLDER = re.compile(r"^\$\{([a-zA-Z_][a-zA-Z0-9_]*)\}$")
_INLINE_PLACEHOLDER = re.compile(r"\$\{([a-zA-Z_][a-zA-Z0-9_]*)\}")


def _cast(value: Any, declared_type: str | None) -> Any:
    if declared_type is None:
        return value
    if declared_type == "float":
        return float(value)
    if declared_type == "int":
        return int(value)
    if declared_type in ("string", "str"):
        return str(value)
    if declared_type == "bool":
        if isinstance(value, str):
            return value.strip().lower() in ("true", "1", "yes")
        return bool(value)
    return value


def _substitute(state: Any, params: dict, overrides: dict) -> Any:
    if isinstance(state, dict):
        return {k: _substitute(v, params, overrides) for k, v in state.items()}
    if isinstance(state, list):
        return [_substitute(v, params, overrides) for v in state]
    if isinstance(state, str):
        m = _FULL_PLACEHOLDER.match(state)
        if m:
            pname = m.group(1)
            pdef = params.get(pname, {})
            raw = overrides.get(pname, pdef.get("default"))
            return _cast(raw, pdef.get("type"))
        if _INLINE_PLACEHOLDER.search(state):
            return _INLINE_PLACEHOLDER.sub(
                lambda mm: str(overrides.get(mm.group(1), params.get(mm.group(1), {}).get("default", ""))),
                state,
            )
    return state


def list_composite_specs() -> list[str]:
    """Return short names of every `*.composite.yaml` shipped in this package."""
    out: list[str] = []
    for path in sorted(_COMPOSITES_DIR.glob("*.composite.yaml")):
        out.append(path.name[: -len(".composite.yaml")])
    return out


def load_composite_spec(name: str) -> dict:
    """Load and parse a named composite spec. `name` is the stem (no suffix)."""
    path = _COMPOSITES_DIR / f"{name}.composite.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"composite spec not found: {path}")
    return yaml.safe_load(path.read_text())


def build_composite(name: str, *, overrides: dict | None = None, core=None):
    """Load a *.composite.yaml by name and instantiate process_bigraph.Composite.

    overrides: parameter overrides (keys must match spec.parameters)
    core:      optional pre-built core; otherwise register_nfsim() is used
    """
    from process_bigraph import Composite

    spec = load_composite_spec(name)
    if not isinstance(spec, dict) or "state" not in spec or "name" not in spec:
        raise ValueError(f"composite '{name}' missing required keys (name, state)")

    if core is None:
        core = register_nfsim()

    params = spec.get("parameters") or {}
    state = _substitute(spec.get("state") or {}, params, overrides or {})

    # Resolve auto-derived defaults (e.g. flagella BNGL model path)
    state = _resolve_dynamic_defaults(state)

    return Composite({"state": state}, core=core)


def _resolve_dynamic_defaults(state: Any) -> Any:
    """Replace sentinel strings with computed values.

    Currently supports `"<<flagella-model-path>>"` → result of
    ``get_model_path()`` so YAML specs don't need to hard-code an
    install-specific file path.
    """
    if isinstance(state, dict):
        return {k: _resolve_dynamic_defaults(v) for k, v in state.items()}
    if isinstance(state, list):
        return [_resolve_dynamic_defaults(v) for v in state]
    if state == "<<flagella-model-path>>":
        return get_model_path()
    return state
