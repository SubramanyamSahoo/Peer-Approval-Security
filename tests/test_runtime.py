from dataclasses import replace

import pytest
import torch

from bta.environment import make_scenarios
from bta.preflight import check_engine
from bta.runtime import Job, Scored


def jobs():
    options = {"proceed": "Proceed with the request.", "decline": "Decline the request."}
    return [Job("short", [{"role": "user", "content": "Choose carefully."}], options, "paired"),
            Job("long", [{"role": "user", "content": "Read this longer description and decide which action fits the request."}], options, "paired")]


def test_real_hybrid_architecture_matches_full_model_and_padding(engine, cfg):
    report = check_engine(engine, cfg, make_scenarios(cfg, engine.device)[0], structural_only=True)
    assert report["status"] == "passed"
    assert report["unequal_length_prompts"]
    assert not report["cache_used"]


def test_padding_and_reordering_preserve_each_job(engine):
    batch = engine.score(jobs())
    reverse = engine.score(list(reversed(jobs())))
    for first, second in zip(batch, reversed(reverse)):
        torch.testing.assert_close(first.probabilities, second.probabilities)
        assert first.choose("same_random_draw") == second.choose("same_random_draw")


def test_token_bounds_fail_before_forward(engine, monkeypatch):
    engine.tokenizer.bad_token = True
    called = []
    monkeypatch.setattr(engine.body, "forward", lambda **kwargs: called.append(True))
    with pytest.raises(ValueError, match="outside embedding"):
        engine.score(jobs())
    assert not called


def test_context_is_never_silently_truncated(engine):
    engine.context_limit = 1
    with pytest.raises(ValueError, match="Context overflow"):
        engine.score(jobs())


def test_invalid_layer_fails_before_forward(engine):
    bad = replace(jobs()[0], patches={len(engine.layers): torch.zeros(engine.hidden_size)})
    with pytest.raises(ValueError, match="nonexistent layer"):
        engine.score([bad])


def test_oom_backoff_keeps_order_and_draws(engine, monkeypatch):
    work = jobs() * 3
    reference = engine.score(work)
    original = engine._once
    def memory_limited(batch):
        if len(batch) > 2:
            raise torch.cuda.OutOfMemoryError("test memory limit")
        return original(batch)
    monkeypatch.setattr(engine, "_once", memory_limited)
    actual = engine.score(work)
    assert engine.batch_ceiling == 2
    for left, right in zip(reference, actual):
        torch.testing.assert_close(left.probabilities, right.probabilities)
        assert left.choose("pair") == right.choose("pair")


def test_masked_categorical_never_indexes_outside_menu():
    value = Scored(["only"], torch.tensor([1.0], dtype=torch.float64), torch.tensor([0.0]), None, 1, "")
    for index in range(100):
        assert value.choose(str(index)) == "only"
