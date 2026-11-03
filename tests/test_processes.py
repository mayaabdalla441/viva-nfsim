"""Unit tests for NFSimProcess and MonomerProduction."""

import os
import re
import tempfile

import pytest
from process_bigraph import allocate_core
from pbg_nfsim.processes import NFSimProcess, MonomerProduction, _parse_bngl_text
from pbg_nfsim.models.generate_flagella_bngl import get_model_path, default_production_rates


@pytest.fixture
def core():
    c = allocate_core()
    c.register_link('nfsim', NFSimProcess)
    c.register_link('monomer-production', MonomerProduction)
    return c


@pytest.fixture
def model_path():
    return get_model_path()


def test_nfsim_instantiation(core, model_path):
    proc = NFSimProcess(
        config={'model_file': model_path, 'n_steps': 10},
        core=core)
    assert proc.config['model_file'] == model_path
    assert proc.config['n_steps'] == 10
    assert len(proc.observable_names) > 0


def test_nfsim_initial_state(core, model_path):
    proc = NFSimProcess(
        config={'model_file': model_path, 'n_steps': 10},
        core=core)
    state = proc.initial_state()
    assert 'observables' in state
    # All observables should start at 0
    for name, val in state['observables'].items():
        assert val == 0.0, f'{name} should be 0.0, got {val}'


def test_nfsim_inputs_outputs(core, model_path):
    proc = NFSimProcess(
        config={'model_file': model_path, 'n_steps': 10},
        core=core)
    inputs = proc.inputs()
    outputs = proc.outputs()
    assert 'observables' in inputs
    assert 'observables' in outputs
    # Should have the same observable names
    assert set(inputs['observables'].keys()) == set(outputs['observables'].keys())
    assert len(inputs['observables']) == len(proc.observable_names)


def test_nfsim_seedability(core, model_path):
    proc = NFSimProcess(
        config={'model_file': model_path, 'n_steps': 10},
        core=core)
    # Should have seedable observables (free monomers)
    assert len(proc.seedable_obs) > 0
    # Growing intermediates should NOT be seedable
    for name in proc.observable_names:
        if name.startswith('Growing_'):
            assert name not in proc.seedable_obs


def test_nfsim_update_zero_state(core, model_path):
    """With zero state, update should return zero deltas (no simulation)."""
    proc = NFSimProcess(
        config={'model_file': model_path, 'n_steps': 10},
        core=core)
    state = proc.initial_state()
    result = proc.update(state, interval=10.0)
    assert 'observables' in result
    # All deltas should be zero since there's nothing to simulate
    for name, val in result['observables'].items():
        assert val == 0.0


def test_nfsim_config_defaults(core, model_path):
    proc = NFSimProcess(
        config={'model_file': model_path},
        core=core)
    assert proc.config['n_steps'] == 100


def test_monomer_production_instantiation(core):
    rates = {'Free_fliG': 1.3, 'Free_flgE': 6.0}
    proc = MonomerProduction(
        config={'production_rates': rates},
        core=core)
    assert proc.rates == rates


def test_monomer_production_outputs(core):
    rates = {'Free_fliG': 1.3, 'Free_flgE': 6.0}
    proc = MonomerProduction(
        config={'production_rates': rates},
        core=core)
    outputs = proc.outputs()
    assert 'monomers' in outputs
    assert 'Free_fliG' in outputs['monomers']
    assert 'Free_flgE' in outputs['monomers']


def test_monomer_production_update(core):
    rates = {'Free_fliG': 1.3, 'Free_flgE': 6.0}
    proc = MonomerProduction(
        config={'production_rates': rates},
        core=core)
    result = proc.update({}, interval=10.0)
    assert abs(result['monomers']['Free_fliG'] - 13.0) < 1e-6
    assert abs(result['monomers']['Free_flgE'] - 60.0) < 1e-6


def test_default_production_rates():
    rates = default_production_rates()
    assert len(rates) > 0
    # All rates should be positive
    for name, rate in rates.items():
        assert rate > 0, f'{name} rate should be positive'
        assert name.startswith('Free_'), f'{name} should start with Free_'


# ---------------------------------------------------------------------------
# Scaffold-state persistence (counter-state species carried across chunks)
# ---------------------------------------------------------------------------
#
# Regression coverage for the bug where NFSimProcess discarded all Growing_X
# counter-state species between chunked update() calls: each call built a
# fresh temp BNGL model seeded only from the "simple" (plain-count)
# observables, so any multi-step assembly chain in progress -- anything
# needing more binding events than fit in a single interval -- could never
# accumulate across calls and therefore could never complete, no matter how
# many total intervals were run. Confirmed on a real downstream model (a
# 120-subunit assembly chain): 1,904 completions in one continuous run vs. 0
# completions across 432 combined chunked intervals before this fix.
#
# _SCAFFOLD_CHAIN_BNGL below is a small, self-contained analog: A() monomers
# nucleate into Growing(c~2), then must acquire 6 more sequential binds
# (c~3..c~8) before completing into Done(). Rates are tuned so the expected
# total assembly time (~4-5s) is comfortably longer than a single chunk's
# interval (1.0s) but comfortably shorter than the total simulated window
# (20 chunks x 1.0s = 20s) -- so completion within the test's time budget is
# only possible if the Growing() scaffold state actually survives being
# carried from one update() call to the next.

_SCAFFOLD_CHAIN_BNGL = """\
begin model

begin parameters
    k_nuc 0.0005
    k_bind 0.01
    k_complete 1.0
    A0 200
end parameters

begin molecule types
    A()
    Growing(c~2~3~4~5~6~7~8)
    Done()
end molecule types

begin seed species
    A() A0
end seed species

begin observables
    Molecules FreeA A()
    Molecules Done Done()
end observables

begin reaction rules
    A() + A() -> Growing(c~2) k_nuc
    Growing(c~2) + A() -> Growing(c~3) k_bind
    Growing(c~3) + A() -> Growing(c~4) k_bind
    Growing(c~4) + A() -> Growing(c~5) k_bind
    Growing(c~5) + A() -> Growing(c~6) k_bind
    Growing(c~6) + A() -> Growing(c~7) k_bind
    Growing(c~7) + A() -> Growing(c~8) k_bind
    Growing(c~8) -> Done() k_complete
end reaction rules

end model
"""


@pytest.fixture
def scaffold_chain_model_path(tmp_path):
    path = tmp_path / "scaffold_chain.bngl"
    path.write_text(_SCAFFOLD_CHAIN_BNGL)
    return str(path)


def test_parse_bngl_text_identifies_counter_molecule_types():
    """_parse_bngl_text must distinguish counter-state (Growing_X) molecule
    types, which need exact-state re-seeding, from simple ones, whose plain
    counts are already handled via the observables delta port."""
    (observable_names, obs_to_pattern, seed_pattern_to_param,
     simple_molecule_types, counter_molecule_types) = _parse_bngl_text(_SCAFFOLD_CHAIN_BNGL)

    assert 'A' in simple_molecule_types
    assert 'Done' in simple_molecule_types
    assert 'Growing' in counter_molecule_types
    assert 'Growing' not in simple_molecule_types
    assert 'A' not in counter_molecule_types


def test_nfsim_scaffold_species_ports_present(core, scaffold_chain_model_path):
    """The scaffold_species port must exist on inputs/outputs/initial_state,
    alongside the pre-existing observables port."""
    proc = NFSimProcess(
        config={'model_file': scaffold_chain_model_path, 'n_steps': 20},
        core=core)
    assert 'scaffold_species' in proc.inputs()
    assert 'scaffold_species' in proc.outputs()
    assert proc.initial_state()['scaffold_species'] == {}
    assert 'Growing' in proc.counter_molecule_types


def test_scaffold_species_persist_across_chunks(core, scaffold_chain_model_path):
    """End-to-end regression test: a multi-step assembly chain that cannot
    plausibly complete within a single short chunk must still complete when
    run across many chunks, PROVIDED scaffold_species is carried forward
    from each update() call into the next -- proving partial-assembly state
    (not just simple/complete species counts) survives chunk boundaries."""
    proc = NFSimProcess(
        config={'model_file': scaffold_chain_model_path, 'n_steps': 20},
        core=core)

    state = proc.initial_state()
    # NFSimProcess.initial_state() always starts observables at 0.0 -- it
    # does NOT read the model's own seed-species defaults (A0=200 in the
    # BNGL above). The 'param'-kind seeding path substitutes whatever is in
    # state['observables'] for the named parameter on every call, so the
    # caller is expected to seed the starting pool explicitly (this mirrors
    # how the real composite feeds monomers in via a separate
    # MonomerProduction process before NFSimProcess ever sees them).
    state['observables']['FreeA'] = 200.0
    interval = 1.0
    n_chunks = 20  # 20s total -- ~4-5x the expected ~4-5s assembly time

    saw_nonempty_scaffold = False
    for _ in range(n_chunks):
        result = proc.update(state, interval)
        for name, delta in result['observables'].items():
            state['observables'][name] = state['observables'].get(name, 0.0) + delta
        state['scaffold_species'] = result['scaffold_species']
        if state['scaffold_species']:
            saw_nonempty_scaffold = True
        if state['observables'].get('Done', 0.0) > 0:
            break

    assert saw_nonempty_scaffold, (
        'expected at least one chunk to report in-progress Growing() '
        'scaffold state via scaffold_species')
    assert state['observables'].get('Done', 0.0) > 0, (
        'assembly chain never completed across 20s of chunked simulation '
        '(1.0s/chunk) despite an ~4-5s expected completion time -- this is '
        'exactly the failure mode the scaffold-persistence fix addresses'
    )


def test_scaffold_species_reset_between_runs_does_not_carry_stale_state(core, scaffold_chain_model_path):
    """A fresh NFSimProcess/state pair must start with empty scaffold_species
    -- guards against accidentally sharing mutable state across instances."""
    proc_a = NFSimProcess(
        config={'model_file': scaffold_chain_model_path, 'n_steps': 20},
        core=core)
    state_a = proc_a.initial_state()
    proc_a.update(state_a, 5.0)

    proc_b = NFSimProcess(
        config={'model_file': scaffold_chain_model_path, 'n_steps': 20},
        core=core)
    assert proc_b.initial_state()['scaffold_species'] == {}


def _seeded_chunk(core, model_file, seed, interval=5.0):
    """Run one fresh NFSimProcess chunk from the same starting state."""
    config = {'model_file': model_file, 'n_steps': 20}
    if seed is not None:
        config['seed'] = seed
    proc = NFSimProcess(config=config, core=core)
    state = proc.initial_state()
    state['observables']['FreeA'] = 200.0
    return proc.update(state, interval)


def test_seed_makes_runs_reproducible(core, scaffold_chain_model_path):
    """Regression: the seed was dropped from the NFsim simulate action, so
    identical configurations produced different trajectories. The same seed
    must now give identical output, and a different seed must differ."""
    first = _seeded_chunk(core, scaffold_chain_model_path, seed=42)
    second = _seeded_chunk(core, scaffold_chain_model_path, seed=42)
    other = _seeded_chunk(core, scaffold_chain_model_path, seed=7)

    assert first == second
    assert first != other


def test_missing_observable_raises(core, model_path, monkeypatch):
    """Regression: an observable absent from NFsim's output used to default
    to 0.0, silently wiping that species. It must raise instead. NFsim's
    output is replaced with a single-column table, the only way to make a
    named observable go missing without depending on NFsim internals."""
    import numpy as np
    import pbg_nfsim.processes as processes

    proc = NFSimProcess(
        config={'model_file': model_path, 'n_steps': 10},
        core=core)
    gdat = np.zeros(2, dtype=[('time', 'f8'), (proc.observable_names[0], 'f8')])

    class _FakeResult(dict):
        pass

    result = _FakeResult({'model': gdat})
    monkeypatch.setattr(processes.bionetgen, 'run', lambda *a, **k: result)

    state = proc.initial_state()
    state['observables'][proc.observable_names[1]] = 5.0
    with pytest.raises(RuntimeError, match=proc.observable_names[1]):
        proc.update(state, 1.0)
