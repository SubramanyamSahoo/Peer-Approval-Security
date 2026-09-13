"""Plumbing validation only. Forced test actions are never used by the research runner."""
import copy
import json

import pytest
import torch

from bta.environment import initial_state, make_case, make_scenarios
from bta.experiments import intervention_due, run_episodes
from bta.learning import evaluate_warning, learn, load_models
from bta.reporting import plot_warning, report_stage, summaries
from bta.runtime import Scored
from bta.storage import Store, read_steps


def test_full_pipeline_and_resume_real_tiny_model(engine, cfg, monkeypatch):
    # Deterministic actions make positive and negative paths reachable in a unit test;
    # all forward passes, diagnostic probabilities, fitting, and patches remain real.
    def test_policy(self, key):
        first = key.endswith("step_0")
        if first and "propose_fixture" in self.actions:
            return "propose_fixture"
        if "approve" in self.actions and "_0000_" in key:
            return "approve"
        if "submit" in self.actions:
            return "submit"
        if "apply" in self.actions:
            return "apply"
        return "wait"
    monkeypatch.setattr(Scored, "choose", test_policy)
    scenarios = make_scenarios(cfg, engine.device)
    store = Store(cfg)
    run_episodes(engine, store, cfg, scenarios, "e1_formation")
    baseline = summaries(store, "e1_formation")
    assert len(baseline) == len(scenarios) * 4
    assert any(r["coordinated_tampering"] for r in baseline)
    assert any(r["legitimate_completion"] for r in baseline)
    first_forwards = engine.forward_batches
    run_episodes(engine, store, cfg, scenarios, "e1_formation")
    assert engine.forward_batches == first_forwards
    report = report_stage(store, cfg, "e1_formation", engine.device)
    assert (report / "outcomes.png").is_file()
    learned = learn(engine, store, cfg)
    assert learned["risk"] is not None
    assert learned["candidate"] is not None
    reloaded = load_models(store.root / "e2_warning", engine.device)
    torch.testing.assert_close(learned["risk"]["weight"], reloaded["risk"]["weight"])
    plot_warning(store)
    for stage in ("e3_patching", "e4_timing"):
        run_episodes(engine, store, cfg, scenarios, stage, learned)
        records = summaries(store, stage)
        assert len(records) == cfg["study"]["test_scenarios"] * 2 * 3
        assert max(r["intervention_count"] for r in records) <= 1
        report = report_stage(store, cfg, stage, engine.device)
        assert len(json.loads((report / "report.json").read_text())["reference_conditions"]) == 2
    for case_path in (store.root / "e3_patching" / "episodes").glob("*/*/*/rule_reminder/*"):
        reviewer = next(event for _, event in read_steps(case_path) if event["role"] == "reviewer")
        assert "TASK REMINDER" in reviewer["decision_prompt"]
        assert "TASK REMINDER" not in reviewer['baseline_prompt']
    assert Store(cfg, str(store.root)).root == store.root


def test_undefined_detector_is_not_reported_as_zero_performance(cfg):
    scores = torch.tensor([0.3, 0.7])
    rows = [{"case_key": "test", "event": {"step": i}} for i in range(2)]
    episodes = [{"case_key": "test", "split": "test", "any_prohibited_approval": True,
                 "first_violation_step": 1, "scenario_id": "s", "authorization": "prohibited", "visibility": "public"}]
    summary, _ = evaluate_warning(scores, rows, episodes, cfg['analysis'], "cpu")
    assert summary["threshold"] is None
    assert summary['primary']['window_recall'] is None
    assert summary['primary']['episode_auroc'] is None


def test_timing_delay_is_one_reviewer_decision(cfg):
    scenario = make_scenarios(cfg, "cpu")[0]
    learned = {"threshold": 0.5}
    case = make_case(scenario, "prohibited", "public", "delayed", 0)
    state = initial_state(case)
    state["step"] = 1
    assert not intervention_due(case, state, "e4_timing", cfg, learned, 0.6)
    assert state["first_alert_reviewer_turn"] == 0
    state["step"], state["reviewer_turns"] = 4, 1
    assert intervention_due(case, state, "e4_timing", cfg, learned, 0.2)
    state["interventions"] = [{"step": 4}]
    assert not intervention_due(case, state, "e4_timing", cfg, learned, 0.9)


def test_unidentified_mechanism_skips_causal_claims(engine, cfg):
    store = Store(cfg)
    scenarios = make_scenarios(cfg, engine.device)
    for stage in ("e3_patching", "e4_timing"):
        run_episodes(engine, store, cfg, scenarios, stage)
        assert json.loads((store.root / stage / "status.json").read_text())["status"] == "not_identified"
        assert not summaries(store, stage)


def test_resume_after_partial_step_commit(engine, cfg, monkeypatch):
    store = Store(cfg)
    scenarios = make_scenarios(cfg, engine.device)[:1]
    original = store.record_step
    committed = []
    def interrupted(*args, **kwargs):
        if committed:
            raise KeyboardInterrupt()
        original(*args, **kwargs)
        committed.append(args[0] / "steps" / f"step_{args[1]:06d}" / "event.json")
    monkeypatch.setattr(store, "record_step", interrupted)
    with pytest.raises(KeyboardInterrupt):
        run_episodes(engine, store, cfg, scenarios, "e1_formation")
    before = committed[0].read_bytes()
    resumed = Store(cfg, str(store.root))
    run_episodes(engine, resumed, cfg, scenarios, "e1_formation")
    assert committed[0].read_bytes() == before
    assert len(summaries(resumed, "e1_formation")) == 4
