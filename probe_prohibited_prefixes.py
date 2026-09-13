#!/usr/bin/env python
"""
Measure peer-contingent approval pressure at CONSTRUCTED prohibited prefixes.

Why this exists
---------------
In the completed run, diagnostic eligibility requires an open fixture proposal
before any binding approval. Under prohibited authorisation the proposer almost
never opened one, so the prohibited peer contrasts are unidentified (n=0) and the
only prohibited peer numbers in the run come from the preflight drift check,
which is a single constructed prefix scored three times. That is not a sample.

This script builds the missing prefix explicitly, for every test scenario, in
BOTH authorisation conditions, so the prohibited and authorised contrasts are
matched at an identical state. It does not touch the existing run directory and
does not modify anything in bta/, so the completed run remains resumable and
its source hash unchanged.

Usage
-----
    # 1. signature + construction check, no model load, takes seconds
    .venv/bin/python probe_prohibited_prefixes.py --run-dir results/<RUN_ID> --check

    # 2. real measurement
    .venv/bin/python probe_prohibited_prefixes.py --run-dir results/<RUN_ID>

Writes <out-dir>/prohibited_prefix_probe.json (raw per-scenario probabilities)
and prints the report-ready summary to stdout.
"""

from __future__ import annotations

import argparse
import datetime as dt
import inspect
import json
import math
import random
import uuid
from pathlib import Path

VISIBILITIES = ("private", "public")
STANCES = ("support", "withdraw", "neutral")
AUTHORIZATIONS = ("prohibited", "authorized")


# --------------------------------------------------------------------------
# statistics: plain percentile bootstrap over scenario-level paired values,
# deliberately not importing bta.gpu_math so this stays robust to v0.2 changes
# --------------------------------------------------------------------------
def bootstrap(values, confidence=0.95, draws=2500, seed=0):
    values = [float(v) for v in values]
    n = len(values)
    out = {"n_scenarios": n, "estimate": (sum(values) / n) if n else None,
           "ci_low": None, "ci_high": None}
    if n < 2:
        out["status"] = "insufficient_scenarios_for_interval"
        return out
    rng = random.Random(seed)
    means = []
    for _ in range(draws):
        means.append(sum(values[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    tail = (1 - confidence) / 2
    lo = means[max(0, min(draws - 1, int(math.floor(tail * draws))))]
    hi = means[max(0, min(draws - 1, int(math.ceil((1 - tail) * draws)) - 1))]
    out.update(ci_low=lo, ci_high=hi, bootstrap_resamples=draws,
               confidence_level=confidence, status="estimated")
    return out


def odds(p):
    p = min(max(p, 1e-12), 1 - 1e-12)
    return p / (1 - p)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True,
                        help="Completed run directory (read only; never written to)")
    parser.add_argument("--out-dir", default=None,
                        help="Where to write results. Default: <run-dir>_prohibited_probe")
    parser.add_argument("--check", action="store_true",
                        help="Verify signatures and build prefixes without loading the model")
    parser.add_argument("--limit", type=int, default=0,
                        help="Only use the first N test scenarios (smoke test)")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    out_dir = Path(args.out_dir) if args.out_dir else run_dir.parent / (run_dir.name + "_prohibited_probe")

    from bta.environment import (apply_action, current_role, decision_job,
                                 diagnostic_eligible, initial_state, make_case)

    # ---- signature guard: fail loudly now, not after a 54 GB load -----------
    expected = {
        "make_case": ("scenario", "authorization", "visibility", "arm", "replicate"),
        "decision_job": ("case", "state"),
    }
    for name, fn in (("make_case", make_case), ("decision_job", decision_job)):
        params = list(inspect.signature(fn).parameters)
        missing = [p for p in expected[name] if p not in params]
        if missing:
            raise SystemExit(f"API drift: {name} is missing {missing}; params are {params}. "
                             f"Adjust this script before running.")
    dj_params = inspect.signature(decision_job).parameters
    supports_kwargs = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in dj_params.values())
    for needed in ("visibility", "peer_override", "reminder"):
        if needed not in dj_params and not supports_kwargs:
            raise SystemExit(f"API drift: decision_job cannot take {needed!r}.")

    cfg = json.loads((run_dir / "resolved_config.json").read_text())
    scenarios = [s for s in json.loads((run_dir / "scenario_registry.json").read_text())
                 if s["split"] == "test"]
    if args.limit:
        scenarios = scenarios[:args.limit]
    revision = json.loads((run_dir / "model_provenance.json").read_text())["resolved_revision"]
    print(f"{len(scenarios)} test scenarios; checkpoint revision {revision}")

    import torch
    device = torch.device("cpu" if args.check else cfg["model"]["device"])

    # ---- build the prefix that never arose naturally -----------------------
    prefixes = {}
    for scenario in scenarios:
        for authorization in AUTHORIZATIONS:
            case = make_case(scenario, authorization, "public", "prohibited_prefix_probe", 0)
            state = apply_action(case, initial_state(case), "propose_fixture", cfg, device)
            assert current_role(state) == "reviewer", "constructed prefix is not at a reviewer turn"
            assert diagnostic_eligible(state), "constructed prefix is not diagnostic-eligible"
            assert state["approved_version"] is None and state["first_violation_step"] is None
            prefixes[(scenario["scenario_id"], authorization)] = (case, state)
    print(f"built {len(prefixes)} eligible prefixes "
          f"(open fixture proposal, no approval, reviewer to act)")

    jobs, owner = [], []
    for (scenario_id, authorization), (case, state) in prefixes.items():
        for visibility in VISIBILITIES:
            for stance in STANCES:
                for reminder in (False, True):
                    jobs.append(decision_job(case, state, device, capture=False,
                                             visibility=visibility, peer_override=stance,
                                             reminder=reminder))
                    owner.append((scenario_id, authorization, visibility, stance, reminder))
    print(f"{len(jobs)} forward passes queued "
          f"({len(scenarios)} scenarios x 2 authorisations x 2 visibilities x 3 stances x 2 reminder)")

    menus = {len(j.options) for j in jobs}
    print(f"action menu sizes present: {sorted(menus)} (must be a single value for a matched contrast)")
    if len(menus) != 1:
        raise SystemExit("Action menus differ across branches; the contrast would not be matched.")

    if args.check:
        print("\nCHECK PASSED. Re-run without --check to measure.")
        return 0

    # ---- score --------------------------------------------------------------
    from bta.runtime import Engine
    engine = Engine(cfg, None, pinned_revision=revision)
    try:
        scored = engine.score(jobs)
    finally:
        engine.close()

    records = {}
    for (scenario_id, authorization, visibility, stance, reminder), result in zip(owner, scored):
        key = f"{authorization}/{visibility}/{'reminder' if reminder else 'plain'}/{stance}"
        records.setdefault(scenario_id, {})[key] = float(result.probability("approve").item())

    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "probe_id": uuid.uuid4().hex,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source_run": str(run_dir),
        "model_revision": revision,
        "prefix_construction": "propose_fixture applied to the initial state; reviewer turn, "
                               "open fixture proposal, no approval, no prior violation",
        "prefixes_are_constructed_not_naturally_occurring": True,
        "action_menu_size": menus.pop(),
        "scenarios": records,
        "telemetry": engine.telemetry(),
    }
    path = out_dir / "prohibited_prefix_probe.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"\nraw probabilities written to {path}\n")

    # ---- report-ready summary ----------------------------------------------
    def col(authorization, visibility, variant, stance):
        return [records[s["scenario_id"]][f"{authorization}/{visibility}/{variant}/{stance}"]
                for s in scenarios]

    print("=" * 78)
    print("PEER SENSITIVITY AT MATCHED CONSTRUCTED PREFIXES")
    print("C_t = P(approve | support) - P(approve | withdraw), paired by scenario")
    print("=" * 78)
    summary = {}
    for authorization in AUTHORIZATIONS:
        for visibility in VISIBILITIES:
            for variant in ("plain", "reminder"):
                sup, wit, neu = (col(authorization, visibility, variant, s) for s in STANCES)
                c = bootstrap([a - b for a, b in zip(sup, wit)], seed=1)
                ratio = [odds(a) / odds(b) for a, b in zip(sup, wit)]
                tag = f"{authorization}/{visibility}/{variant}"
                summary[tag] = {
                    "C_t": c,
                    "mean_P_support": sum(sup) / len(sup),
                    "mean_P_withdraw": sum(wit) / len(wit),
                    "mean_P_neutral": sum(neu) / len(neu),
                    "median_support_vs_withdraw_odds_ratio": sorted(ratio)[len(ratio) // 2],
                }
                print(f"{tag:34s} C_t={c['estimate']:+.4f} "
                      f"[{c['ci_low']:+.4f},{c['ci_high']:+.4f}]  "
                      f"P_sup={sum(sup)/len(sup):.4f} P_wit={sum(wit)/len(wit):.4f}  "
                      f"medianOR={sorted(ratio)[len(ratio)//2]:.2f}")

    print()
    print("=" * 78)
    print("AUTHORITY RESTATEMENT EFFECT (reminder minus plain, paired by scenario)")
    print("=" * 78)
    for authorization in AUTHORIZATIONS:
        for visibility in VISIBILITIES:
            plain = [a - b for a, b in zip(col(authorization, visibility, "plain", "support"),
                                           col(authorization, visibility, "plain", "withdraw"))]
            rem = [a - b for a, b in zip(col(authorization, visibility, "reminder", "support"),
                                         col(authorization, visibility, "reminder", "withdraw"))]
            d = bootstrap([a - b for a, b in zip(rem, plain)], seed=2)
            tag = f"{authorization}/{visibility}"
            summary[f"{tag}/reminder_effect_on_C_t"] = d
            print(f"{tag:34s} dC_t={d['estimate']:+.4f} [{d['ci_low']:+.4f},{d['ci_high']:+.4f}]")

    print()
    print("=" * 78)
    print("AUTHORISATION CONTRAST AT THE SAME PREFIX (prohibited minus authorized)")
    print("=" * 78)
    for visibility in VISIBILITIES:
        for variant in ("plain", "reminder"):
            pro = [a - b for a, b in zip(col("prohibited", visibility, variant, "support"),
                                         col("prohibited", visibility, variant, "withdraw"))]
            aut = [a - b for a, b in zip(col("authorized", visibility, variant, "support"),
                                         col("authorized", visibility, variant, "withdraw"))]
            d = bootstrap([a - b for a, b in zip(pro, aut)], seed=3)
            tag = f"{visibility}/{variant}"
            summary[f"{tag}/prohibited_minus_authorized_C_t"] = d
            print(f"{tag:34s} dC_t={d['estimate']:+.4f} [{d['ci_low']:+.4f},{d['ci_high']:+.4f}]")

    (out_dir / "prohibited_prefix_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"\nsummary written to {out_dir / 'prohibited_prefix_summary.json'}")
    print("\nNOTE FOR THE WRITE-UP: these prefixes are constructed, not naturally occurring. "
          "State that explicitly; it is the same technique the run already uses for history/built/*.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
