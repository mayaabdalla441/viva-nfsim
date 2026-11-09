"""
NFSim Processes
===============

Process-bigraph wrappers for BioNetGen/NFSim rule-based simulations.

NFSimProcess: runs a BNGL model with NFSim, returning delta changes
in observable values. Suitable for composition with other processes.

MonomerProduction: produces flagellar protein monomers at constant
rates, mimicking gene expression.
"""
import os
import re
import tempfile

import bionetgen

from process_bigraph.composite import Process


def _parse_bngl_text(bngl_text):
    """Parse a BNGL model text to extract observables, seed species, and molecule types.

    Returns:
        observable_names: list of observable names
        obs_to_pattern: dict mapping observable name -> BNGL pattern
        seed_pattern_to_param: dict mapping seed species pattern -> parameter name
        simple_molecule_types: set of molecule type names with no internal states
        counter_molecule_types: set of molecule type names WITH internal counter
            states (e.g. "Growing_hook(flgE~0~1~2...)") -- added 2026-08-12 so
            NFSimProcess can identify which species need exact-state
            re-seeding (see SCAFFOLD STATE PERSISTENCE FIX below), as opposed
            to simple_molecule_types, whose plain counts are already handled
            via observables.
    """
    observable_names = []
    obs_to_pattern = {}
    seed_pattern_to_param = {}
    simple_molecule_types = set()
    counter_molecule_types = set()

    # Parse observables block
    in_observables = False
    for line in bngl_text.splitlines():
        stripped = line.strip()
        if stripped == 'begin observables':
            in_observables = True
            continue
        if stripped == 'end observables':
            break
        if in_observables and stripped and not stripped.startswith('#'):
            parts = stripped.split()
            if len(parts) >= 3:
                name = parts[1]
                pattern = parts[2]
                observable_names.append(name)
                obs_to_pattern[name] = pattern

    # Parse seed species block
    in_seeds = False
    for line in bngl_text.splitlines():
        stripped = line.strip()
        if stripped == 'begin seed species':
            in_seeds = True
            continue
        if stripped == 'end seed species':
            break
        if in_seeds and stripped and not stripped.startswith('#'):
            parts = stripped.split()
            if len(parts) >= 2:
                pattern = parts[0]
                param = parts[1]
                seed_pattern_to_param[pattern] = param

    # Parse molecule types block
    in_mol_types = False
    for line in bngl_text.splitlines():
        stripped = line.strip()
        if stripped == 'begin molecule types':
            in_mol_types = True
            continue
        if stripped == 'end molecule types':
            break
        if in_mol_types and stripped and not stripped.startswith('#'):
            mol_type = stripped
            # Simple molecule type: Name() with no internal states (no ~)
            if '~' not in mol_type and mol_type.endswith('()'):
                name = mol_type[:-2]
                simple_molecule_types.add(name)
            elif '~' in mol_type:
                # Counter/scaffold type: Name(comp~0~1~2~...) -- name is
                # everything before the first '('.
                name = mol_type.split('(', 1)[0].strip()
                counter_molecule_types.add(name)

    return (observable_names, obs_to_pattern, seed_pattern_to_param,
            simple_molecule_types, counter_molecule_types)


class NFSimProcess(Process):
    """A generic process that wraps BioNetGen/NFSim network-free simulations.

    Loads a BNGL model, runs NFSim for each time step, and returns
    delta changes in observable values. Seed species counts are set
    from the current state before each run, allowing composition with
    other processes.

    Reproducibility: ``seed`` (default None = unseeded) is a BASE seed. Each
    update() call passes NFsim ``seed=>(base + call index)``, so the same
    base seed reproduces the same chunked trajectory, while successive
    chunks use different random streams (not one stream restarted every
    interval).

    Observables whose molecule types have no internal states (simple
    molecules) are "seedable" -- their counts carry over between steps.

    SCAFFOLD STATE PERSISTENCE FIX (2026-08-12): growing intermediates with
    counter states (e.g. a partially-assembled complex tracked as
    Growing_X(subunit~N)) previously had NO representation that survived
    between invocations at all -- each call built a fresh temp BNGL model,
    seeded only from the plain-count `observables` state, and any
    in-progress scaffold was silently reset to zero every time this Step
    fired. For a multi-hundred-subunit assembly chain running under a short
    per-interval budget, that made it structurally impossible for such a
    reaction to ever complete, no matter how many total intervals were run
    -- confirmed directly (v2ecoli flagella-cascade investigation,
    2026-08-12): an isolated, unchunked run of one such reaction completed
    1,904 times over one continuous simulation, while the SAME reaction
    embedded in this Process's normal chunked operation completed 0 times
    across 432 combined chunked intervals (~1.6M total reaction events).

    Fixed by reading NFsim's own full end-of-run species list (the
    `<model>.species` file BioNetGen/NFsim already writes, listing every
    distinct species -- including every individual scaffold occupancy state
    -- with its exact count) instead of relying only on the aggregate
    `observables` block, and re-seeding ALL of it (not just the simple/
    plain-count species) as exact BNGL seed species on the next call. This
    is exposed as a new `scaffold_species` overwrite port (see inputs()/
    outputs()) alongside the existing `observables` delta port -- growing
    intermediates now persist exactly across steps, the same way ordinary
    monomer counts already did.
    """

    config_schema = {
        'model_file': 'string',
        'n_steps': {
            '_type': 'integer',
            '_default': 100,
        },
        # Base NFsim RNG seed (None = unseeded). Each update() call passes
        # NFsim its own seed derived from this base and the call index (see
        # update()), so a run is reproducible for a given base seed while
        # successive chunks still use different random streams. (Merged
        # 2026-11-10: upstream #7's maybe[integer]/None default kept; the
        # per-call derivation is from 0278c8b.)
        'seed': {
            '_type': 'maybe[integer]',
            '_default': None,
        },
    }

    def __init__(self, config=None, core=None):
        super().__init__(config, core)
        self.model_file = self.config['model_file']
        self._call_count = 0

        # Read and store the BNGL template
        with open(self.model_file) as f:
            self.bngl_template = f.read()

        # Parse model structure
        (self.observable_names,
         self.obs_to_pattern,
         self.seed_pattern_to_param,
         self.simple_molecule_types,
         self.counter_molecule_types) = _parse_bngl_text(self.bngl_template)

        # Build seedability mapping for each observable
        # seedable_obs maps observable_name -> ('param', param_name) or ('add', pattern)
        self.seedable_obs = {}
        for name in self.observable_names:
            pattern = self.obs_to_pattern.get(name, '')
            if pattern in self.seed_pattern_to_param:
                # Has an existing seed species with a parameter
                self.seedable_obs[name] = ('param', self.seed_pattern_to_param[pattern])
            elif pattern.endswith('()'):
                # Check if molecule type is simple (no internal states)
                mol_name = pattern[:-2]
                if mol_name in self.simple_molecule_types:
                    self.seedable_obs[name] = ('add', pattern)

        # Model's own basename, used to locate the .species file BNG writes
        # alongside the temp .bngl file each run (added 2026-08-12).
        self._model_basename = 'model'

    def initial_state(self):
        return {
            'observables': {name: 0.0 for name in self.observable_names},
            'scaffold_species': {},
            'gross_positive_deltas': {name: 0.0 for name in self.observable_names},
        }

    def inputs(self):
        return {
            'observables': {
                name: 'float' for name in self.observable_names
            },
            'scaffold_species': 'map[float]',
        }

    def outputs(self):
        return {
            'observables': {
                name: 'float' for name in self.observable_names
            },
            'scaffold_species': 'overwrite[map[float]]',
            # Added 2026-09-01: 'observables' is a NET delta over the whole
            # chunk (final - initial), correct for real bulk-count updates
            # but structurally blind to a species produced AND consumed
            # within the same chunk -- confirmed real for fast intermediate
            # stages (C-ring, export apparatus, motor complex), whose
            # cumulative "ever formed" trackers read flat 0 despite real
            # completions happening. NFsim already returns a full
            # ~50-point time series per chunk (gdat), previously discarded
            # down to just the last row. This port sums only the POSITIVE
            # jumps between consecutive sub-steps -- catches a
            # rise-then-fall within the chunk, down to sub-step (~1200s/
            # n_steps) resolution instead of whole-chunk resolution. Not
            # used for real bulk-count mass balance (that stays net-delta,
            # correctly conservative) -- only for callers that want a true
            # "how much was ever produced" count, like the cumulative
            # trackers in flagella_nfsim_complexation.py.
            'gross_positive_deltas': {
                name: 'float' for name in self.observable_names
            },
        }

    def _parse_species_file(self, path):
        """Parse a BNG-written .species file (added 2026-08-12).

        Format (one per line, tab/space-separated, '#'-prefixed comments):
            Growing_hook(flgE~45)  1
            flagellar_hook()  1918
        Returns dict: exact BNGL pattern -> count (int).
        """
        species = {}
        if not os.path.exists(path):
            return species
        with open(path) as f:
            for line in f:
                stripped = line.strip()
                if not stripped or stripped.startswith('#'):
                    continue
                parts = stripped.rsplit(None, 1)
                if len(parts) != 2:
                    continue
                pattern, count_str = parts
                try:
                    count = int(float(count_str))
                except ValueError:
                    continue
                if count > 0:
                    species[pattern] = count
        return species

    def update(self, state, interval):
        # Read current observable values
        current = {
            name: max(0, int(state['observables'].get(name, 0)))
            for name in self.observable_names
        }
        # Exact scaffold-state population carried forward from the previous
        # call (added 2026-08-12 -- see class docstring, SCAFFOLD STATE
        # PERSISTENCE FIX). Keys are literal BNGL patterns as written by BNG
        # itself in the .species file, e.g. "Growing_hook(flgE~45)".
        incoming_scaffold = dict(state.get('scaffold_species') or {})

        # Skip if nothing to simulate
        total_seedable = sum(
            current[name] for name in self.observable_names
            if name in self.seedable_obs
        )
        total_scaffold = sum(incoming_scaffold.values())
        if total_seedable == 0 and total_scaffold == 0:
            return {
                'observables': {name: 0.0 for name in self.observable_names},
                'scaffold_species': {},
                'gross_positive_deltas': {name: 0.0 for name in self.observable_names},
            }

        # Build BNGL text with current state
        bngl_text = self.bngl_template

        # Update existing seed species parameters
        for name in self.observable_names:
            if name not in self.seedable_obs:
                continue
            kind, ref = self.seedable_obs[name]
            count = current[name]
            if kind == 'param':
                # Replace parameter value (match within a single line only)
                pattern = rf'([ \t]+{re.escape(ref)}[ \t]+)\S+'
                bngl_text = re.sub(pattern, rf'\g<1>{count}', bngl_text)

        # Add new seed species for seedable observables without existing params
        extra_seeds = []
        for name in self.observable_names:
            if name not in self.seedable_obs:
                continue
            kind, ref = self.seedable_obs[name]
            count = current[name]
            if kind == 'add' and count > 0:
                extra_seeds.append(f'    {ref}  {count}')

        # Re-seed the EXACT incoming scaffold population (added 2026-08-12)
        # -- one seed species line per distinct occupancy state, so
        # partially-assembled complexes resume from exactly where the
        # previous interval left them instead of resetting to zero.
        for pattern, count in incoming_scaffold.items():
            if count > 0:
                extra_seeds.append(f'    {pattern}  {count}')

        if extra_seeds:
            bngl_text = bngl_text.replace(
                'end seed species',
                '\n'.join(extra_seeds) + '\nend seed species')

        # Append simulate action. With a base seed, each call gets
        # seed=>(base + call index): reproducible per (base seed, call),
        # but different call-to-call. Upstream #7 passed the SAME seed on
        # every call, which restarts the RNG each chunk and correlates the
        # chunks of a chunked run (its old docstring said so). Upstream
        # version, kept for reference:
        #   seed = self.config.get('seed')
        #   seed_arg = '' if seed is None else f',seed=>{int(seed)}'
        # .get, not []: process-bigraph 1.9.1 does not fill a maybe[...] default
        # into self.config when the key is omitted.
        seed = self.config.get('seed')
        if seed is None:
            seed_arg = ''
        else:
            call_seed = (int(seed) + self._call_count) & 0x7FFFFFFF
            seed_arg = f',seed=>{call_seed}'
        self._call_count += 1
        bngl_text += (
            f'\nsimulate({{method=>"nf",'
            f't_end=>{interval},'
            f'n_steps=>{self.config["n_steps"]}{seed_arg}}});\n'
        )

        # Run NFSim
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_bngl = os.path.join(tmpdir, f'{self._model_basename}.bngl')
            with open(tmp_bngl, 'w') as f:
                f.write(bngl_text)

            result = bionetgen.run(tmp_bngl, out=tmpdir)
            gdat = result[self._model_basename]

            # Read the full final species population (added 2026-08-12) --
            # NFsim's own <model>.species file, listing every distinct
            # species (including every scaffold occupancy state) with its
            # exact count, not just the aggregate named observables.
            species_path = os.path.join(tmpdir, f'{self._model_basename}.species')
            final_species = self._parse_species_file(species_path)

        # Compute observable deltas (unchanged from before this fix) --
        # net change over the whole chunk, used for real bulk-count mass
        # balance. Correct for that purpose; NOT a "how much was ever
        # produced" count (see gross_positive_deltas below).
        deltas = {}
        # Gross positive-jump sum across the chunk's own sub-steps (added
        # 2026-09-01, see outputs()) -- separate from deltas above.
        gross_positive = {}
        for name in self.observable_names:
            if name not in gdat.dtype.names:
                raise RuntimeError(
                    f'NFsim output has no column for observable {name!r}; '
                    'refusing to treat it as zero')
            initial = current.get(name, 0)
            # Column presence is already enforced by the raise above
            # (upstream #7), so no 0.0 fallback here (0278c8b had one).
            series = gdat[name]
            final = float(series[-1])
            gross_positive[name] = float(sum(
                max(0.0, float(series[k + 1]) - float(series[k]))
                for k in range(len(series) - 1)
            ))
            deltas[name] = final - initial

        # New scaffold_species snapshot: every final-species entry whose
        # molecule name is a counter/scaffold type (has internal states),
        # keyed by its exact BNGL pattern. Simple/complete species are
        # already covered by the observables/deltas mechanism above, so
        # excluded here to avoid double-representing the same count in two
        # different ports.
        new_scaffold = {
            pattern: float(count)
            for pattern, count in final_species.items()
            if pattern.split('(', 1)[0] in self.counter_molecule_types
        }

        return {
            'observables': deltas,
            'scaffold_species': new_scaffold,
            'gross_positive_deltas': gross_positive,
        }


class MonomerProduction(Process):
    """Produces flagellar protein monomers at constant rates.

    Config:
        production_rates: dict mapping monomer name -> rate (molecules/second).
            Defaults to rates that produce ~1 flagellum worth per 100 seconds.
    """

    config_schema = {
        'production_rates': {
            '_type': 'map[float]',
            '_default': {},
        },
    }

    def __init__(self, config=None, core=None):
        super().__init__(config, core)
        self.rates = self.config['production_rates']

    def inputs(self):
        return {}

    def outputs(self):
        return {
            'monomers': {
                name: 'float' for name in self.rates
            },
        }

    def update(self, state, interval):
        return {
            'monomers': {
                name: rate * interval
                for name, rate in self.rates.items()
            },
        }
