"""Measured H100 planning; synthetic contexts here are never empirical episodes."""
import math
import time
from dataclasses import replace

import torch

from .environment import AUTHORIZATIONS, ROLES, VISIBILITIES, WITHDRAWAL, apply_action, decision_job, initial_state, make_case
from .experiments import STAGE_ARMS
from .history import HISTORIES, PRESENT_SUPPORT, history_job
from .storage import atomic_json


def workload(cfg, replicates=None):
    s = cfg['study']
    r = s['replicates'] if replicates is None else replicates
    total_scenarios = sum(s[f'{split}_scenarios'] for split in ('fit', 'calibration', 'test'))
    test = s['test_scenarios'] * len(AUTHORIZATIONS) * r
    baseline = total_scenarios * len(AUTHORIZATIONS) * len(VISIBILITIES) * r
    e3, e4 = test * len(STAGE_ARMS['e3_patching']), test * len(STAGE_ARMS['e4_timing'])
    terms = {'live_decisions': (baseline + e3 + e4) * s['max_rounds'] * len(ROLES),
             'peer_diagnostics': (baseline + e3) * s['max_rounds'] * len(VISIBILITIES) * len(WITHDRAWAL),
             'history_diagnostics': (test * len(VISIBILITIES) + e3) * len(HISTORIES) * len(PRESENT_SUPPORT),
             'text_monitor': baseline * s['max_rounds'],
             'private_donors': test * sum(arm != 'rule_reminder' for arm in STAGE_ARMS['e3_patching']),
             'edited_decisions': e3 + e4}
    return {'replicates': r, 'baseline_episodes': baseline, 'e3_maximum_episodes': e3, 'e4_maximum_episodes': e4,
            'maximum_total_episodes': baseline + e3 + e4, 'maximum_forward_prompts_by_kind': terms,
            'maximum_total_forward_prompts': sum(terms.values()), 'e2_reuses_baseline_episodes': True,
            'upper_counts_assume_every_eligible_opportunity_survives': True}


def profile_jobs(engine, cfg, scenario):
    jobs = []
    for authorization in AUTHORIZATIONS:
        for visibility in VISIBILITIES:
            case = make_case(scenario, authorization, visibility, 'profile', 0)
            for role_index, role in enumerate(ROLES):
                state = initial_state(case)
                stop = (cfg['study']['max_rounds'] - 1) * len(ROLES) + role_index
                while state['step'] < stop:
                    action = 'propose_fixture' if state['step'] == 0 else 'wait'
                    state = apply_action(case, state, action, cfg, engine.device)
                jobs.append(decision_job(case, state, engine.device))
                if role == 'reviewer':
                    for stance in WITHDRAWAL:
                        jobs.append(decision_job(case, state, engine.device, peer_override=stance))
                    for history in HISTORIES:
                        for present in PRESENT_SUPPORT:
                            jobs.append(history_job(case, state, engine.device, history, present))
    return jobs


def run_profile(engine, store, cfg, scenario):
    representatives = profile_jobs(engine, cfg, scenario)
    longest = max(representatives, key=lambda job: len(engine.prepare(job)[3]))
    base = engine.score([replace(longest, capture=True)])[0]
    direction = (engine.head_rows[0] - engine.head_rows[1]).float()
    norm = direction.norm()
    if not bool(norm > 0):
        raise ValueError('Cannot construct nonzero profiling edit')
    delta = direction / norm * base.hidden[-1].norm() / math.sqrt(engine.hidden_size)
    patched = replace(longest, capture=True, patches={engine.layer_ids[-1]: delta})
    batch = cfg['compute']['initial_batch_size']
    populations = {'longest_edited_prefix': [patched] * batch,
                   'mixed_prefixes': [replace(representatives[i % len(representatives)], capture=True) for i in range(batch)]}
    engine.score(populations['longest_edited_prefix'])  # Warmup, charged but not assumed free.
    measurements = []
    for repetition in range(cfg['profiling']['measurement_repetitions']):
        for name, jobs in populations.items():
            start = time.monotonic()
            engine.score(jobs)
            seconds = time.monotonic() - start
            measurements.append({'population': name, 'repetition': repetition, 'prompts': len(jobs),
                'wall_seconds': seconds, 'wall_seconds_per_prompt': seconds / len(jobs)})
    slowest = max(m['wall_seconds_per_prompt'] for m in measurements)
    one = workload(cfg, replicates=1)
    estimate_one = slowest * one['maximum_total_forward_prompts'] / (1 - cfg['profiling']['nonforward_time_fraction'])
    accounting = store.budget.accounting()
    remaining_usd = max(0, cfg['budget']['limit_usd'] - cfg['budget']['reserve_usd'] - accounting['estimated_instance_usd'])
    remaining_seconds = remaining_usd / cfg['budget']['hourly_usd'] * 3600
    fit = math.floor(remaining_seconds / estimate_one) if estimate_one > 0 else 0
    report = {'result_kind': 'runtime_profile_not_empirical_findings', 'measurements': measurements,
        'longest_representative_tokens': len(engine.prepare(longest)[3]), 'effective_batch_ceiling': engine.batch_ceiling,
        'one_replicate_workload_bound': one, 'planning_seconds_per_replicate': estimate_one,
        'planning_nonforward_time_fraction': cfg['profiling']['nonforward_time_fraction'],
        'remaining_estimated_compute_usd': remaining_usd, 'maximum_replicates_under_this_planning_estimate': fit,
        'not_a_runtime_or_budget_guarantee': True, 'replicates_are_not_changed_automatically': True,
        'allocation_note': 'Compare more scenarios with more replicates using development variability; no claim this is power-optimal.'}
    atomic_json(store.root / 'profile' / 'report.json', report)
    store.status('profile', 'completed', accounting=store.budget.accounting())
    print(f'PROFILE: estimated maximum replicates {fit}; planning seconds/replicate {estimate_one:.1f}; inspect profile/report.json', flush=True)
    return report
