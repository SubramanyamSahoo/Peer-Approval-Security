from __future__ import annotations

import datetime as dt
import fcntl
import hashlib
import json
import os
import time
import uuid
from pathlib import Path

from .config import content_hash


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def clean_error(exc: BaseException) -> str:
    text = f"{type(exc).__name__}: {exc}"
    token = os.environ.get("HF_TOKEN")
    return text.replace(token, "<redacted>") if token else text


def atomic_json(path: Path, data: object, *, replace: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("x") as stream:
        json.dump(data, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    # Linux advisory locking + atomic rename avoids requiring filesystem hard links.
    # Locks are released by the OS after a crash; persistent empty lock files are benign.
    try:
        with path.with_name('.lock_' + path.name + '.lock').open('a+b') as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            if path.exists() and not replace:
                raise FileExistsError(f'Refusing to overwrite {path}')
            os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def source_hash() -> str:
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


class BudgetExceeded(RuntimeError):
    pass


class Budget:
    def __init__(self, cfg: dict, started: str):
        self.cfg = cfg
        self.started = dt.datetime.fromisoformat(cfg["billing_started_utc"] or started)
        if self.started.tzinfo is None:
            raise ValueError("Billing start must include a timezone, e.g. +00:00")
        if self.started > dt.datetime.now(dt.timezone.utc):
            raise ValueError("Billing start cannot be in the future")
        self.last_batch_seconds = 0.0

    def accounting(self) -> dict:
        seconds = (dt.datetime.now(dt.timezone.utc) - self.started).total_seconds()
        estimate = seconds / 3600 * self.cfg["hourly_usd"]
        return {"elapsed_billing_seconds": seconds, "estimated_instance_usd": estimate,
                "limit_usd": self.cfg["limit_usd"], "reserve_usd": self.cfg["reserve_usd"],
                "instance_termination_performed": False}

    def check(self) -> None:
        estimate = self.accounting()["estimated_instance_usd"]
        next_batch = self.last_batch_seconds / 3600 * self.cfg["hourly_usd"]
        if estimate + next_batch >= self.cfg["limit_usd"] - self.cfg["reserve_usd"]:
            raise BudgetExceeded("Compute reserve reached; results are checkpointed. Stop the Lambda instance to end billing.")


class Store:
    def __init__(self, cfg: dict, resume: str | None = None, *, kind='empirical_run'):
        if resume:
            self.root = Path(resume).resolve()
            self.manifest = json.loads((self.root / "manifest.json").read_text())
            if self.manifest['result_kind'] != kind:
                raise ValueError('Profiling and empirical runs cannot share a result directory')
            if self.manifest["config_hash"] != content_hash(cfg):
                raise ValueError("Resume configuration differs. Use the saved resolved_config.json, or start a new run.")
            if self.manifest["source_hash"] != source_hash():
                raise ValueError("Code changed: start a new run; do not mix implementations in one result directory.")
        else:
            stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
            self.root = Path(cfg["output_root"]).resolve() / f"{stamp}_{uuid.uuid4().hex}"
            self.root.mkdir(parents=True, exist_ok=False)
            self.manifest = {"run_id": self.root.name, "created_utc": utc_now(),
                             "config_hash": content_hash(cfg), "source_hash": source_hash(),
                             "result_kind": kind, "schema_version": 2, "config": cfg}
            atomic_json(self.root / "manifest.json", self.manifest)
            atomic_json(self.root / "resolved_config.json", cfg)
        self.budget = Budget(cfg["budget"], self.manifest["created_utc"])

    def status(self, stage: str, status: str, **details) -> None:
        atomic_json(self.root / stage / "status.json",
                    {"status": status, "updated_utc": utc_now(), **details}, replace=True)

    def case_path(self, stage: str, case: dict) -> Path:
        return self.root / stage / "episodes" / case["split"] / case["authorization"] / case["visibility"] / case["arm"] / case["case_id"]

    def initial(self, stage: str, case: dict) -> Path:
        path = self.case_path(stage, case)
        if not (path / "case.json").exists():
            atomic_json(path / "case.json", case)
        else:
            if json.loads((path / "case.json").read_text()) != case:
                raise ValueError(f"Case identity collision: {path}")
        return path

    def record_step(self, path: Path, step: int, event: dict, state: dict, features, layers: list[int],
                    diagnostics: dict | None = None, diagnostic_hidden=None, history=None) -> None:
        from safetensors.torch import save_file
        target = path / "steps" / f"step_{step:06d}"
        target.parent.mkdir(parents=True, exist_ok=True)
        staging = target.parent / f".pending_{uuid.uuid4().hex}"
        staging.mkdir(exist_ok=False)
        atomic_json(staging / "event.json", {**event, "state_after": state})
        if features is not None:
            save_file({"hidden": features.detach().to("cpu").contiguous()}, str(staging / "features.safetensors"),
                      metadata={"layer_indices": json.dumps(layers)})
        if diagnostics is not None:
            atomic_json(staging / "diagnostics.json", diagnostics)
        if diagnostic_hidden is not None:
            save_file({"hidden": diagnostic_hidden.detach().to("cpu").contiguous()},
                      str(staging / "diagnostics.safetensors"), metadata={"layer_indices": json.dumps(layers)})
        if history is not None:
            atomic_json(staging / 'history' / 'contrasts.json', {k: v for k, v in history.items() if k != 'conditions'})
            for name, condition in history['conditions'].items():
                atomic_json(staging / 'history' / (name.replace('/', '__') + '.json'), condition)
        if target.exists():
            raise FileExistsError(f"Refusing to overwrite a committed step: {target}")
        staging.rename(target)

    def report_dir(self, stage: str) -> Path:
        path = self.root / stage / "analysis" / uuid.uuid4().hex
        path.mkdir(parents=True, exist_ok=False)
        return path


def read_steps(case_path: Path) -> list[tuple[Path, dict]]:
    return [(p.parent, json.loads(p.read_text())) for p in sorted((case_path / "steps").glob("step_*/event.json"))]
