from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path


def stable_int(*parts: object) -> int:
    """An unsigned 64-bit identifier; never Python's process-randomized hash()."""
    data = json.dumps(parts, sort_keys=True, separators=(",", ":")).encode()
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "big")


def content_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def load_config(path: str | Path, overrides: list[str] = ()) -> dict:
    cfg = json.loads(Path(path).read_text())
    for expression in overrides:
        key, separator, raw = expression.partition("=")
        if not separator:
            raise ValueError(f"Override must have dotted.key=value form: {expression}")
        target = cfg
        segments = key.split(".")
        for segment in segments[:-1]:
            if segment not in target or not isinstance(target[segment], dict):
                raise KeyError(key)
            target = target[segment]
        if segments[-1] not in target:
            raise KeyError(f"Unknown configuration key: {key}")
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            value = raw
        target[segments[-1]] = value
    validate_config(cfg)
    return copy.deepcopy(cfg)


def validate_config(cfg: dict) -> None:
    positive = [
        ("compute", "initial_batch_size"), ("compute", "max_context_tokens"),
        ("study", "fit_scenarios"), ("study", "calibration_scenarios"),
        ("study", "test_scenarios"), ("study", "replicates"),
        ("study", "max_rounds"), ("analysis", "warning_horizon_agent_turns")
    ]
    for section, key in positive:
        value = cfg[section][key]
        if type(value) is not int or value <= 0:
            raise ValueError(f"{section}.{key} must be a positive integer")
    if type(cfg["study"]["seed"]) is not int:
        raise ValueError("study.seed must be an integer")
    if not -(1 << 63) <= cfg["study"]["seed"] < (1 << 64):
        raise ValueError("study.seed must fit PyTorch's supported 64-bit seed range")
    for key in ("target_episode_false_alarm_rate", "confidence_level", "bootstrap_mc_standard_error", "permutation_mc_standard_error"):
        value = cfg["analysis"][key]
        if not isinstance(value, (int, float)) or not 0 < value < 1:
            raise ValueError(f"analysis.{key} must be strictly between zero and one")
    if cfg["analysis"]["primary_warning_visibility"] != 'public':
        raise ValueError('v0.2 timing trials use public receipts; primary warning calibration must use public receipts too')
    p = cfg["profiling"]
    if type(p["measurement_repetitions"]) is not int or p["measurement_repetitions"] <= 0:
        raise ValueError("profiling.measurement_repetitions must be a positive integer")
    if not 0 <= p["nonforward_time_fraction"] < 1:
        raise ValueError("profiling.nonforward_time_fraction must be in [0, 1)")
    b = cfg["budget"]
    if not (math.isfinite(b["hourly_usd"]) and b["hourly_usd"] > 0):
        raise ValueError("budget.hourly_usd must be the actual positive instance rate")
    if not 0 <= b["reserve_usd"] < b["limit_usd"]:
        raise ValueError("Budget reserve must be smaller than the spending limit")
    if cfg["compute"]["gpu_reserve_gib"] < 0:
        raise ValueError("GPU memory reserve cannot be negative")
    if cfg["model"]["dtype"] not in {"bfloat16", "float32"}:
        raise ValueError("Use bfloat16 for the H100 study, or float32 for validation")
    s = cfg["study"]
    if any(type(s[key]) is not int for key in ("operand_min", "operand_max")):
        raise ValueError("Task operand bounds must be integers")
    if max(abs(s["operand_min"]), abs(s["operand_max"])) > ((1 << 63) - 1) // 3:
        raise ValueError("Task operand bounds could overflow the three-term int64 validation arithmetic")
    if s["operand_max"] <= s["operand_min"]:
        raise ValueError("The task operand interval must contain at least two integers")
    names = [set(s["styles"][split]) for split in ("fit", "calibration", "test")]
    if any(not x for x in names) or any(names[i] & names[j] for i in range(len(names)) for j in range(i)):
        raise ValueError("Wording families must be nonempty and disjoint across splits")
