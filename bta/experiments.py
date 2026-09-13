from __future__ import annotations

from dataclasses import replace

import torch

from .config import stable_int
from .environment import (AUTHORIZATIONS, VISIBILITIES, apply_action, current_role, decision_job,
                          diagnostic_eligible, initial_state, make_case, summarize)
from .gpu_math import random_equal_norm
from .history import history_batch
from .storage import atomic_json, read_steps, utc_now

STAGE_ARMS = {'e1_formation': ('none',), 'e3_patching': ('candidate', 'random', 'rule_reminder'),
              'e4_timing': ('first_alert', 'delayed', 'random_time')}

def cases_for(scenarios, cfg, stage):
    for scenario in scenarios:
        if stage != "e1_formation" and scenario["split"] != "test":
            continue
        arms = STAGE_ARMS[stage]
        visibility_modes = VISIBILITIES if stage == "e1_formation" else ("public",)
        for authorization in AUTHORIZATIONS:
            for visibility in visibility_modes:
                for arm in arms:
                    for replicate in range(cfg["study"]["replicates"]):
                        yield make_case(scenario, authorization, visibility, arm, replicate)


def support_count(state, role):
    willingness = {}
    for message in state["messages"]:
        if message["role"] == role:
            continue
        if message["action"] in {"support_fixture", "propose_fixture"}:
            willingness[message["role"]] = True
        elif message["action"] in {"withdraw", "reject", "support_repair", "propose_repair"}:
            willingness[message["role"]] = False
    return sum(willingness.values())


def diagnostic_batch(engine, entries):
    """Six matched forward passes per eligible prefix; no diagnostic changes the live state."""
    jobs, owner = [], []
    order = [f"{visibility}/{stance}" for visibility in VISIBILITIES for stance in ("support", "withdraw", "neutral")]
    for index, (case, state, patches) in enumerate(entries):
        if not diagnostic_eligible(state):
            continue
        for condition in order:
            visibility, stance = condition.split("/")
            reminder = any(edit["step"] == state["step"] and edit["kind"] == "text_rule_reminder"
                           for edit in state["interventions"])
            job = decision_job(case, state, engine.device, visibility=visibility, peer_override=stance,
                               reminder=reminder)
            job.patches = patches
            jobs.append(job)
            owner.append((index, condition))
    grouped = {}
    for (index, condition), scored in zip(owner, engine.score(jobs)):
        grouped.setdefault(index, {})[condition] = scored
    output = {}
    for index, conditions in grouped.items():
        gpu_probabilities = {key: value.probability("approve") for key, value in conditions.items()}
        probabilities = {key: float(value.item()) for key, value in gpu_probabilities.items()}
        effects = {visibility: {
            "support_minus_withdraw": float((gpu_probabilities[f"{visibility}/support"] - gpu_probabilities[f"{visibility}/withdraw"]).item()),
            "support_minus_neutral": float((gpu_probabilities[f"{visibility}/support"] - gpu_probabilities[f"{visibility}/neutral"]).item()),
            "withdraw_minus_neutral": float((gpu_probabilities[f"{visibility}/withdraw"] - gpu_probabilities[f"{visibility}/neutral"]).item())}
            for visibility in VISIBILITIES}
        info = {"eligible": True, "condition_order": order, "conditional_approval_probabilities": probabilities,
                "effects": effects, "public_minus_private_neutral": float((gpu_probabilities["public/neutral"] - gpu_probabilities["private/neutral"]).item()),
                "feasibility_changed": False, "controlled_message_replacement": True,
                "response_diagnostics": {key: value.diagnostics for key, value in conditions.items()},
                "condition_prompts": {key: value.prompt for key, value in conditions.items()},
                "probabilities_reference": "decision with this turn's intervention, if present",
                "hidden_states_are_pre_edit_at_each_captured_layer": True}
        output[index] = (info, torch.stack([conditions[key].hidden for key in order]))
    return output


def intervention_due(case, state, stage, cfg, learned, risk_score):
    if current_role(state) != "reviewer" or state["interventions"]:
        return False
    if stage == "e3_patching":
        return bool(state["messages"])
    if stage != "e4_timing":
        return False
    threshold = learned["threshold"]
    turn = state["reviewer_turns"]
    if risk_score is not None and threshold is not None and risk_score > threshold and state["first_alert_reviewer_turn"] is None:
        state["first_alert_reviewer_turn"] = turn
    if case["arm"] == "random_time":
        return turn == stable_int(case["pair_key"], "timing_control") % cfg["study"]["max_rounds"]
    first = state["first_alert_reviewer_turn"]
    if first is None:
        return False
    return turn >= first + (1 if case["arm"] == "delayed" else 0)


def run_episodes(engine, store, cfg, scenarios, stage, learned=None):
    learned = learned or {"risk": None, "candidate": None, "threshold": None}
    if stage == "e3_patching" and learned["candidate"] is None:
        store.status(stage, "not_identified", reason="No usable candidate direction was identified on development data")
        return
    if stage == "e4_timing" and (learned["candidate"] is None or learned["risk"] is None or learned["threshold"] is None):
        store.status(stage, "not_identified", reason="Timing requires a development-fitted candidate and calibrated internal warning")
        return
    cases = list(cases_for(scenarios, cfg, stage))
    entries = []
    for case in cases:
        path = store.initial(stage, case)
        if (path / "summary.json").exists():
            continue
        steps = read_steps(path)
        state = steps[-1][1]["state_after"] if steps else initial_state(case)
        if state["done"]:
            atomic_json(path / "summary.json", summarize(case, state, engine.device))
        else:
            entries.append({"case": case, "path": path, "state": state})
    store.status(stage, "running", expected_episodes=len(cases), pending_episodes=len(entries),
                 reference_baseline_stage="e1_formation" if stage != "e1_formation" else None)
    while entries:
        store.budget.check()
        jobs = [decision_job(e["case"], e["state"], engine.device, capture=current_role(e["state"]) == "reviewer") for e in entries]
        baseline_jobs = list(jobs)
        base = engine.score(jobs)
        decisions = list(base)
        pending_edits, donors, donor_positions = [], [], []
        risk_values = [None] * len(entries)
        for index, entry in enumerate(entries):
            case, state = entry["case"], entry["state"]
            if current_role(state) == "reviewer" and learned["risk"] is not None:
                probe = learned["risk"]
                risk_values[index] = float((base[index].hidden[probe["position"]].double() @ probe["weight"] + probe["bias"]).item())
            if intervention_due(case, state, stage, cfg, learned, risk_values[index]):
                pending_edits.append(index)
                if stage == "e3_patching" and case["arm"] != "rule_reminder":
                    donors.append(decision_job(case, state, engine.device, visibility="private"))
                    donor_positions.append(index)
        donor_results = dict(zip(donor_positions, engine.score(donors)))
        revised_jobs, revised_positions = [], []
        for index in pending_edits:
            entry = entries[index]
            case, state = entry["case"], entry["state"]
            candidate = learned["candidate"]
            metadata = {"step": state["step"], "reviewer_turn": state["reviewer_turns"], "arm": case["arm"],
                        "prior_violation": state["first_violation_step"] is not None}
            if case["arm"] == "rule_reminder":
                updated = decision_job(case, state, engine.device, reminder=True)
                metadata["kind"] = "text_rule_reminder"
            else:
                direction, position = candidate["direction"], candidate["position"]
                before = base[index].hidden[position].double() @ direction
                target = donor_results[index].hidden[position].double() @ direction if index in donor_results else candidate["private_reference"]
                magnitude = target - before
                if case["arm"] == "random":
                    delta = random_equal_norm(direction, magnitude, stable_int(case["pair_key"], state["step"], "random_patch"))
                else:
                    delta = (direction * magnitude).float()
                updated = replace(jobs[index], patches={candidate["layer"]: delta})
                metadata.update({"kind": "activation_patch", "layer": candidate["layer"],
                                 "before_projection": float(before.item()), "target_projection": float(target.item()),
                                 "delta_norm": float(delta.double().norm().item()),
                                 "reference": "matched_private_prefix" if index in donor_results else "fit_private_mean"})
            revised_jobs.append(updated)
            revised_positions.append(index)
            state["interventions"].append(metadata)
            jobs[index] = updated
        for index, value in zip(revised_positions, engine.score(revised_jobs)):
            decisions[index] = value
        diagnostics = diagnostic_batch(engine, [(e["case"], e["state"], jobs[i].patches) for i, e in enumerate(entries)]) if stage in {"e1_formation", "e3_patching"} else {}
        histories = history_batch(engine, [(e['case'], e['state'], jobs[i].patches) for i, e in enumerate(entries)], stage) if stage in {'e1_formation', 'e3_patching'} else {}
        remaining = []
        for index, entry in enumerate(entries):
            case, state, path = entry["case"], entry["state"], entry["path"]
            role = current_role(state)
            result = decisions[index]
            action = result.choose(f"{cfg['study']['seed']}/{case['pair_key']}/step_{state['step']}")
            forbidden = case["authorization"] == "prohibited" and state["proposal"] is not None and state["proposal"]["kind"] == "fixture" and role == "reviewer"
            event = {"step": state["step"], "role": role, "action": action, "created_utc": utc_now(),
                     "visibility": case["visibility"], "prior_violation": state["first_violation_step"] is not None,
                     "schema_version": 2, "peer_support_count": support_count(state, role),
                     "baseline_prompt": base[index].prompt, "decision_prompt": result.prompt,
                     "baseline_context_messages": baseline_jobs[index].messages,
                     "decision_context_messages": jobs[index].messages,
                     "baseline_action_options": baseline_jobs[index].options,
                     "prompt_tokens": result.prompt_tokens, "action_labels_in_order": result.actions,
                     "baseline_conditional_action_probabilities": dict(zip(base[index].actions, base[index].probabilities.cpu().tolist())),
                     "decision_conditional_action_probabilities": dict(zip(result.actions, result.probabilities.cpu().tolist())),
                     "baseline_forbidden_action_probability": float(base[index].probability("approve").item()) if forbidden else 0.0,
                     "decision_forbidden_action_probability": float(result.probability("approve").item()) if forbidden else 0.0,
                     "baseline_response_diagnostics": base[index].diagnostics,
                     "decision_response_diagnostics": result.diagnostics,
                     "internal_warning_score": risk_values[index], "intervention_applied_here": index in pending_edits,
                     "features_reference": "baseline_prompt and baseline_context_messages, before any intervention"}
            if index in histories:
                state['history_diagnostic_done'] = True
            next_state = apply_action(case, state, action, cfg, engine.device)
            info, diagnostic_hidden = diagnostics.get(index, ({"eligible": False, "reason": "No pending fixture proposal before a binding approval"}, None))
            store.record_step(path, state["step"], event, next_state, base[index].hidden, engine.layer_ids,
                              info if role == "reviewer" and stage in {"e1_formation", "e3_patching"} else None, diagnostic_hidden,
                              histories.get(index))
            if next_state["done"]:
                atomic_json(path / "summary.json", summarize(case, next_state, engine.device))
            else:
                entry["state"] = next_state
                remaining.append(entry)
        entries = remaining
        print(f"{stage}: completed {len(cases) - len(entries)}/{len(cases)} episodes; active {len(entries)}; effective batch {engine.batch_ceiling}", flush=True)
    store.status(stage, "completed", expected_episodes=len(cases), completed_episodes=len(cases))
