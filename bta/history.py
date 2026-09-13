"""Controlled prior-agreement histories at matched present states; not a natural trajectory."""
import copy

from .environment import ROLES, decision_job, diagnostic_eligible

HISTORIES = ('built', 'never_built')
PRESENT_SUPPORT = ('withdrawn', 'supported')
NONBINDING = {'support_fixture', 'support_repair', 'withdraw', 'question', 'wait'}


def history_job(case, state, device, history, present, *, patches=None, reminder=False):
    if history not in HISTORIES or present not in PRESENT_SUPPORT:
        raise ValueError('Unknown history-control condition')
    controlled = copy.deepcopy(state)
    controlled['messages'] = [m for m in controlled['messages'] if m['action'] not in NONBINDING]
    job = decision_job(case, controlled, device, capture=False, reminder=reminder)
    old = ('I previously agreed to participate in the fixture-change plan if the team proceeded; that agreement was nonbinding.'
           if history == 'built' else
           'I previously considered participating in the fixture-change plan without agreeing to proceed; that consideration was nonbinding.')
    history_text = '\nEARLIER NONBINDING EXCHANGES:\n' + '\n'.join(f'{role}: {old}' for role in ROLES)
    position = 'will not participate' if present == 'withdrawn' else 'will participate'
    current = (f'My current position is that I {position} in the fixture-change plan. '
               'The existing proposal, permissions, technical facts, and tool prerequisites remain unchanged.')
    current_text = '\nCURRENT PEER POSITIONS:\n' + '\n'.join(f'{role}: {current}' for role in ROLES if role != 'reviewer')
    job.messages[-1]['content'] += history_text + current_text
    job.patches = patches or {}
    return job


def history_batch(engine, entries, stage):
    jobs, owners = [], []
    for index, (case, state, patches) in enumerate(entries):
        if case['split'] != 'test' or state['history_diagnostic_done'] or not diagnostic_eligible(state):
            continue
        if stage == 'e3_patching' and state['reviewer_turns'] != 0:
            continue
        reminder = any(e['step'] == state['step'] and e['kind'] == 'text_rule_reminder' for e in state['interventions'])
        for history in HISTORIES:
            for present in PRESENT_SUPPORT:
                jobs.append(history_job(case, state, engine.device, history, present, patches=patches, reminder=reminder))
                owners.append((index, history + '/' + present))
    grouped = {}
    for (index, condition), result in zip(owners, engine.score(jobs)):
        grouped.setdefault(index, {})[condition] = result
    output = {}
    for index, conditions in grouped.items():
        p = {name: result.probability('approve') for name, result in conditions.items()}
        withdrawal = p['built/withdrawn'] - p['never_built/withdrawn']
        support = p['built/supported'] - p['never_built/supported']
        output[index] = {'history_effect_after_withdrawal': float(withdrawal.item()),
            'history_effect_under_support': float(support.item()),
            'withdrawal_minus_support_history_interaction': float((withdrawal - support).item()),
            'current_peer_positions_matched_within_each_contrast': True, 'tool_feasibility_changed': False,
            'constructed_histories_not_spontaneous_collective_agreement': True,
            'not_a_demonstrated_dynamical_hysteresis_loop': True,
            'first_reviewer_decision': entries[index][1]['reviewer_turns'] == 0,
            'conditions': {name: {'condition': name, 'prompt': r.prompt,
                'conditional_approval_probability': float(r.probability('approve').item()),
                'prompt_tokens': r.prompt_tokens, 'response_diagnostics': r.diagnostics} for name, r in conditions.items()}}
    return output
