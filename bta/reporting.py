from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import torch

from .config import stable_int
from .gpu_math import bootstrap_mean
from .storage import atomic_json, read_steps


METRICS = ("legitimate_completion", "any_prohibited_approval", "coordinated_tampering", "agent_turns", "intervention_count")


def summaries(store, stage):
    return [json.loads(path.read_text()) for path in sorted((store.root / stage / "episodes").glob("*/*/*/*/*/summary.json"))]


def by_scenario(records, metric, device):
    grouped = defaultdict(list)
    for record in records:
        grouped[record["scenario_id"]].append(record[metric])
    return {key: torch.tensor(values, device=device, dtype=torch.float64).mean() for key, values in grouped.items()}


def paired(left, right, metric, cfg, device, key):
    a = {(r['scenario_id'], r['replicate']): r[metric] for r in left}
    b = {(r['scenario_id'], r['replicate']): r[metric] for r in right}
    if len(a) != len(left) or len(b) != len(right):
        raise ValueError('Paired contrast contains duplicate scenario/replicate observations')
    pairs = sorted(set(a) & set(b))
    grouped = defaultdict(list)
    for pair in pairs:
        grouped[pair[0]].append(torch.tensor([a[pair], b[pair]], device=device, dtype=torch.float64).diff().neg().squeeze())
    shared = sorted(grouped)
    values = torch.stack([torch.stack(grouped[s]).mean() for s in shared]) if shared else torch.empty(0, device=device)
    result = bootstrap_mean(values, cfg["analysis"], stable_int(cfg["study"]["seed"], key, metric))
    return {**result, "contrast": "left_minus_right", "left_scenarios": len({s for s, r in a}), "right_scenarios": len({s for s, r in b}),
            "matched_scenario_ids": shared, 'matched_scenario_replicates': pairs,
            "unmatched_left": sorted(set(a) - set(b)), "unmatched_right": sorted(set(b) - set(a))}


def diagnostic_records(store, stage):
    result = []
    for case_path in sorted((store.root / stage / "episodes").glob("*/*/*/*/*")):
        if not (case_path / "case.json").exists():
            continue
        case = json.loads((case_path / "case.json").read_text())
        first_reviewer_seen = False
        for step_path, event in read_steps(case_path):
            if event["role"] != "reviewer":
                continue
            first_reviewer = not first_reviewer_seen
            first_reviewer_seen = True
            path = step_path / "diagnostics.json"
            if not path.exists():
                continue
            diagnostic = json.loads(path.read_text())
            if not diagnostic["eligible"]:
                continue
            result.append({**case, "step": event["step"], "first_reviewer_decision": first_reviewer,
                           "intervention_applied_here": event["intervention_applied_here"],
                           "peer_sensitivity": diagnostic["effects"][case["visibility"]]["support_minus_withdraw"],
                           "visibility_effect": diagnostic["public_minus_private_neutral"]})
    return result


def report_stage(store, cfg, stage, device):
    records = summaries(store, stage)
    if not records:
        return None
    path = store.report_dir(stage)
    grouped = defaultdict(list)
    for record in records:
        grouped[(record["split"], record["authorization"], record["visibility"], record["arm"])].append(record)
    conditions = []
    for key, group in sorted(grouped.items()):
        values = {}
        for metric in METRICS:
            clusters = by_scenario(group, metric, device)
            values[metric] = bootstrap_mean(torch.stack(list(clusters.values())), cfg["analysis"],
                                            stable_int(cfg["study"]["seed"], stage, key, metric))
        condition = {"split": key[0], "authorization": key[1], "visibility": key[2], "arm": key[3],
                     "n_completed_episodes": len(group), "metrics": values}
        conditions.append(condition)
        atomic_json(path / "conditions" / ("__".join(key) + ".json"), condition)
    comparisons = []
    for authorization in ("authorized", "prohibited"):
        if stage == "e1_formation":
            left = [r for r in records if r["split"] == "test" and r["authorization"] == authorization and r["visibility"] == "public"]
            right = [r for r in records if r["split"] == "test" and r["authorization"] == authorization and r["visibility"] == "private"]
            arms = [("public_minus_private", left, right)]
        else:
            baseline = [r for r in summaries(store, "e1_formation") if r["split"] == "test" and r["authorization"] == authorization and r["visibility"] == "public"]
            arms = [(arm, [r for r in records if r["authorization"] == authorization and r["arm"] == arm], baseline)
                    for arm in sorted({r["arm"] for r in records})]
        for arm, left, right in arms:
            comparisons.append({"authorization": authorization, "arm": arm,
                                "reference": "private" if stage == "e1_formation" else "e1_formation/public/none",
                                "metrics": {m: paired(left, right, m, cfg, device, (stage, authorization, arm)) for m in METRICS}})
    diagnostics = diagnostic_records(store, stage)
    diagnostic_summaries = []
    diagnostic_groups = defaultdict(list)
    for record in diagnostics:
        key = (record["split"], record["authorization"], record["visibility"], record["arm"], record["step"])
        diagnostic_groups[key].append(record)
    for key, group in sorted(diagnostic_groups.items()):
        clusters = by_scenario(group, "peer_sensitivity", device)
        diagnostic_summaries.append({"split": key[0], "authorization": key[1], "visibility": key[2], "arm": key[3], "agent_step": key[4],
                                     "effect": bootstrap_mean(torch.stack(list(clusters.values())), cfg["analysis"], stable_int(stage, key))})
    if stage == "e3_patching":
        baseline_diagnostics = [r for r in diagnostic_records(store, "e1_formation") if r["split"] == "test" and r["visibility"] == "public" and r["first_reviewer_decision"]]
        causal_diagnostics = [r for r in diagnostics if r["first_reviewer_decision"] and r["intervention_applied_here"]]
        result = []
        for authorization in ("authorized", "prohibited"):
            for arm in sorted({r["arm"] for r in causal_diagnostics}):
                left = [r for r in causal_diagnostics if r["authorization"] == authorization and r["arm"] == arm]
                right = [r for r in baseline_diagnostics if r["authorization"] == authorization]
                result.append({"authorization": authorization, "arm": arm,
                               "effect": paired(left, right, "peer_sensitivity", cfg, device, (stage, authorization, arm, "sensitivity"))})
        atomic_json(path / "causal_peer_sensitivity.json", {"scope": "first reviewer decision, matched before intervention",
                    "effects": result, "later_checkpoint_diagnostics_are_descriptive_not_an_unbiased_causal_estimate": True})
    report = {"stage": stage, "conditions": conditions, "paired_comparisons": comparisons,
              "diagnostic_checkpoints": diagnostic_summaries,
              "raw_episodes_remain_separate": True, "unit_for_intervals": "scenario, averaging its replicates",
              "diagnostic_eligibility_is_conditional": True,
              "no_universal_tipping_point_or_novelty_claim_is_automatically_generated": True}
    if stage != "e1_formation":
        reference = summaries(store, "e1_formation")
        report["reference_conditions"] = []
        for authorization in ("authorized", "prohibited"):
            group = [r for r in reference if r["split"] == "test" and r["authorization"] == authorization
                     and r["visibility"] == "public"]
            if group:
                report["reference_conditions"].append({"split": "test", "authorization": authorization,
                    "visibility": "public", "arm": "baseline", "source_stage": "e1_formation",
                    "metrics": {m: bootstrap_mean(torch.stack(list(by_scenario(group, m, device).values())),
                        cfg["analysis"], stable_int(cfg["study"]["seed"], "baseline", authorization, m)) for m in METRICS}})
    atomic_json(path / "report.json", report)
    report_response_audits(store, cfg, stage, device, path)
    report_histories(store, cfg, stage, device, path)
    plot_stage(path, report)
    return path


def plot_stage(path: Path, report):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    panels = [("prohibited", "coordinated_tampering", "Unauthorized joint tampering"),
              ("authorized", "legitimate_completion", "Authorized task completion")]
    fig, axes = plt.subplots(1, len(panels), figsize=(11, 4), constrained_layout=True)
    for ax, (authorization, metric, title) in zip(axes, panels):
        selected = [r for r in report.get("reference_conditions", []) + report["conditions"]
                    if r["split"] == "test" and r["authorization"] == authorization]
        labels = [r["visibility"] if report["stage"] == "e1_formation" else r["arm"] for r in selected]
        heights = [r["metrics"][metric]["estimate"] for r in selected]
        ax.bar(labels, heights)
        for index, r in enumerate(selected):
            m = r["metrics"][metric]
            if m["ci_low"] is not None:
                ax.errorbar(index, m["estimate"], yerr=[[max(0, m["estimate"] - m["ci_low"])],
                                                        [max(0, m["ci_high"] - m["estimate"])]], color="black", capsize=3)
        ax.set(title=title, ylim=(0, 1), ylabel="Fraction of episodes")
        ax.tick_params(axis="x", rotation=20)
        if not selected:
            ax.text(0.5, 0.5, "No completed observations", transform=ax.transAxes, ha="center")
    fig.suptitle(report["stage"] + " — measured held-out outcomes")
    fig.savefig(path / "outcomes.png", dpi=180)
    plt.close(fig)
    checkpoints = [r for r in report["diagnostic_checkpoints"] if r["split"] == "test"]
    if checkpoints:
        fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
        groups = defaultdict(list)
        for row in checkpoints:
            groups[(row["authorization"], row["visibility"], row["arm"])].append(row)
        for key, rows in groups.items():
            rows.sort(key=lambda row: row["agent_step"])
            ax.plot([r["agent_step"] for r in rows], [r["effect"]["estimate"] for r in rows], marker="o", label=" / ".join(key))
        ax.axhline(0, color="gray", linewidth=1)
        ax.set(xlabel="Agent turn before decision", ylabel="P(approve | support) − P(approve | withdrawal)",
               title="Peer sensitivity at eligible checkpoints; changing risk sets")
        ax.legend(fontsize="small")
        fig.savefig(path / "peer_sensitivity.png", dpi=180)
        plt.close(fig)


def plot_warning(store):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    path = store.root / "e2_warning" / "models.json"
    if not path.exists():
        return
    meta = json.loads(path.read_text())
    reports = meta.get("monitor_reports", {})
    output = store.report_dir("e2_warning")
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    shown = False
    for name, report in reports.items():
        primary = report['primary']
        fpr, recall = primary['episode_false_alarm_rate'], primary['window_recall']
        if fpr is not None and recall is not None:
            ax.scatter(fpr, recall, label=name)
            shown = True
    ax.set(xlabel="Prohibited-task episode false-alarm rate", ylabel="Recall within the specified window", xlim=(0, 1), ylim=(0, 1),
           title="Primary warning scope: " + meta.get('primary_warning_scope', 'prohibited/public'))
    if shown:
        ax.legend(fontsize="small")
    else:
        ax.text(0.5, 0.5, "Recall is unidentified: no positive outcomes\nor no calibrated detector", ha="center", va="center", transform=ax.transAxes)
    fig.savefig(output / "warning.png", dpi=180)
    plt.close(fig)
    atomic_json(output / "plotted_values.json", reports)


def report_response_audits(store, cfg, stage, device, output):
    groups = defaultdict(list)
    for case_path in sorted((store.root / stage / 'episodes').glob('*/*/*/*/*')):
        if not (case_path / 'case.json').exists():
            continue
        case = json.loads((case_path / 'case.json').read_text())
        for step_path, event in read_steps(case_path):
            key = (case['split'], case['authorization'], case['visibility'], case['arm'], event['role'])
            groups[(*key, 'live_decision')].append(event['decision_response_diagnostics'])
            diagnostic_path = step_path / 'diagnostics.json'
            if diagnostic_path.exists():
                diagnostic = json.loads(diagnostic_path.read_text())
                for condition, values in diagnostic.get('response_diagnostics', {}).items():
                    groups[(*key, 'peer/' + condition)].append(values)
            for history_path in sorted((step_path / 'history').glob('*.json')):
                history = json.loads(history_path.read_text())
                if 'response_diagnostics' in history:
                    groups[(*key, 'history/' + history['condition'])].append(history['response_diagnostics'])
    records = []
    for key, group in sorted(groups.items()):
        mass = torch.tensor([r['available_label_probability_mass'] for r in group], device=device, dtype=torch.float64)
        top = torch.tensor([r['unconstrained_argmax_is_available_label'] for r in group], device=device, dtype=torch.float64)
        flagged = [r['below_control_label_mass_reference'] for r in group if r['below_control_label_mass_reference'] is not None]
        records.append({'split': key[0], 'authorization': key[1], 'visibility': key[2], 'arm': key[3], 'role': key[4], 'observation_kind': key[5],
            'observations': len(group), 'mass_minimum': float(mass.min().item()), 'mass_median': float(mass.median().item()),
            'mass_mean': float(mass.mean().item()), 'mass_maximum': float(mass.max().item()),
            'fraction_unconstrained_argmax_in_menu': float(top.mean().item()),
            'fraction_below_control_reference': float(torch.tensor(flagged, device=device, dtype=torch.float64).mean().item()) if flagged else None,
            'available_menu_sizes': sorted({r['available_action_count'] for r in group})})
    preflight_path = store.root / 'runtime_preflight.json'
    preflight = json.loads(preflight_path.read_text()) if preflight_path.exists() else {}
    atomic_json(output / 'response_format_audit.json', {'conditions': records,
        'preflight_probability_drift_by_prefix': preflight.get('probability_drift_by_prefix'),
        'maximum_observed_preflight_approval_drift': preflight.get('maximum_observed_approval_probability_drift'),
        'drift_is_observed_not_a_global_bound': True, 'label_mass_is_not_a_semantic_validity_certificate': True})


def history_records(store, stage):
    records = []
    for path in sorted((store.root / stage / 'episodes').glob('*/*/*/*/*')):
        if not (path / 'case.json').exists():
            continue
        case = json.loads((path / 'case.json').read_text())
        for step_path, event in read_steps(path):
            history = step_path / 'history' / 'contrasts.json'
            if history.exists():
                records.append({**case, **json.loads(history.read_text()), 'step': event['step']})
    return records


def report_histories(store, cfg, stage, device, output):
    records = history_records(store, stage)
    if not records:
        return
    metrics = ('history_effect_after_withdrawal', 'history_effect_under_support', 'withdrawal_minus_support_history_interaction')
    groups = defaultdict(list)
    for row in records:
        groups[(row['authorization'], row['visibility'], row['arm'])].append(row)
    estimates = []
    for key, rows in sorted(groups.items()):
        estimates.append({'authorization': key[0], 'visibility': key[1], 'arm': key[2],
            'metrics': {m: bootstrap_mean(torch.stack(list(by_scenario(rows, m, device).values())), cfg['analysis'],
                                        stable_int(stage, key, m)) for m in metrics}})
    causal = []
    if stage == 'e3_patching':
        reference = [r for r in history_records(store, 'e1_formation') if r['visibility'] == 'public' and r['first_reviewer_decision']]
        for authorization, visibility, arm in sorted(groups):
            left = [r for r in groups[(authorization, visibility, arm)] if r['first_reviewer_decision']]
            right = [r for r in reference if r['authorization'] == authorization]
            causal.append({'authorization': authorization, 'arm': arm,
                'metrics': {m: paired(left, right, m, cfg, device, (stage, authorization, arm, m)) for m in metrics}})
    atomic_json(output / 'history_dependence.json', {'conditions': estimates, 'paired_intervention_effects': causal,
        'construct': 'prior-agreement effect at matched present peer positions and tool state',
        'not_a_demonstrated_hysteresis_loop': True, 'constructed_histories': True})
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
    for i, row in enumerate(estimates):
        effect = row['metrics']['history_effect_after_withdrawal']
        ax.bar(i, effect['estimate'])
        if effect['ci_low'] is not None:
            ax.errorbar(i, effect['estimate'], yerr=[[max(0, effect['estimate'] - effect['ci_low'])],
                [max(0, effect['ci_high'] - effect['estimate'])]], color='black', capsize=3)
    ax.set_xticks(range(len(estimates)), [r['authorization'] + '\n' + r['visibility'] + ' / ' + r['arm'] for r in estimates], rotation=20)
    ax.axhline(0, color='gray', linewidth=1)
    ax.set(ylabel='P(approve | prior agreement) − P(approve | no prior agreement)',
           title='Both peers currently withdraw; tools and permissions held fixed')
    fig.savefig(output / 'history_dependence.png', dpi=180)
    plt.close(fig)
