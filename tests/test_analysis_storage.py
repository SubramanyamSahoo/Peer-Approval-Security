import copy
import json

import pytest
import torch

from bta.environment import initial_state, make_case, make_scenarios
from bta.gpu_math import bootstrap_mean, calibrate_threshold, random_equal_norm, ridge_candidates
from bta.storage import Store, atomic_json, read_steps


def test_calibration_controls_false_alarms_with_ties():
    scores = torch.tensor([0.1, 0.1, 0.2, 0.3, 0.3], dtype=torch.float64)
    threshold = calibrate_threshold(scores, 0.2)
    assert int((scores > threshold).sum()) <= 1
    assert calibrate_threshold(torch.empty(0), 0.1) is None


def test_degenerate_statistics_do_not_invent_numbers(cfg):
    result = bootstrap_mean(torch.empty(0), cfg["analysis"], 0)
    assert result["estimate"] is None
    assert result["ci_low"] is None
    one = bootstrap_mean(torch.tensor([1.0]), cfg["analysis"], 0)
    assert one["estimate"] == 1.0 and one["ci_low"] is None


def test_random_control_matches_edit_norm():
    direction = torch.tensor([1.0, 0.0, 0.0])
    noise = random_equal_norm(direction, torch.tensor(-2.0), 0)
    torch.testing.assert_close(noise.norm(), torch.tensor(2.0))
    torch.testing.assert_close(noise @ direction, torch.tensor(0.0))


def test_ridge_selects_from_fit_spectrum_and_calibration():
    x = torch.tensor([[0., 1.], [1., 0.], [2., 1.], [3., 0.]], dtype=torch.float64)
    y = x[:, 0] * 2 - x[:, 1]
    model = ridge_candidates(x, y, ["a", "a", "b", "b"], x + 0.2, (x[:, 0] + 0.2) * 2 - (x[:, 1] + 0.2), ["c", "c", "d", "d"])
    assert model is not None and torch.isfinite(model["weight"]).all()
    assert ridge_candidates(x, torch.zeros(4), list("abcd"), x, y, list("efgh")) is None


def test_unique_runs_atomic_steps_and_explicit_resume(cfg):
    one, two = Store(cfg), Store(cfg)
    assert one.root != two.root
    case = make_case(make_scenarios(cfg, torch.device("cpu"))[0], "authorized", "public", "none", 0)
    path = one.initial("e1_formation", case)
    state = initial_state(case)
    one.record_step(path, 0, {"step": 0}, state, torch.zeros(1, 3), [0])
    assert read_steps(path)[0][1]["state_after"] == state
    with pytest.raises(FileExistsError):
        one.record_step(path, 0, {"step": 0}, state, None, [0])
    assert Store(cfg, str(one.root)).root == one.root
    changed = copy.deepcopy(cfg)
    changed["study"]["seed"] += 1
    with pytest.raises(ValueError, match="differs"):
        Store(changed, str(one.root))


def test_atomic_json_refuses_overwrite(tmp_path):
    path = tmp_path / "result.json"
    atomic_json(path, {"observed": True})
    with pytest.raises(FileExistsError):
        atomic_json(path, {"observed": False})
    assert json.loads(path.read_text()) == {"observed": True}
