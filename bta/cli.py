from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path

from .config import load_config
from .storage import BudgetExceeded, Store, atomic_json, clean_error


def main(argv=None):
    parser = argparse.ArgumentParser(description="Before They Act: local cooperative-AI / LLM-security experiments")
    parser.add_argument("command", choices=("plan", "profile", "run", "preflight", "analyze"))
    parser.add_argument("--config", default="configs/h100_qwen38.json")
    parser.add_argument("--set", action="append", default=[], metavar="DOTTED.KEY=JSON_VALUE")
    parser.add_argument("--resume", help="Explicitly continue the same immutable configuration and implementation")
    parser.add_argument("--run-dir", help="Existing result directory for GPU analysis")
    args = parser.parse_args(argv)
    if args.command == 'profile' and args.resume:
        parser.error('Profiling uses a fresh directory; it cannot resume empirical results')
    if args.command == "analyze":
        if not args.run_dir:
            parser.error("analyze requires --run-dir")
        args.resume = args.run_dir
    if args.resume:
        cfg = load_config(Path(args.resume) / "resolved_config.json", args.set)
    else:
        cfg = load_config(args.config, args.set)
    if args.command == "plan":
        from .profiling import workload
        print(json.dumps({"model": cfg["model"]["id"], **workload(cfg),
            "initial_batch_size": cfg["compute"]["initial_batch_size"],
            "guarded_compute_hours": (cfg["budget"]["limit_usd"] - cfg["budget"]["reserve_usd"]) / cfg["budget"]["hourly_usd"],
            "runtime_is_measured_not_guaranteed": True, "no_results_created": True}, indent=2))
        return 0
    store = Store(cfg, args.resume, kind='runtime_profile' if args.command == 'profile' else 'empirical_run')
    print(f"RESULT_DIRECTORY={store.root}", flush=True)
    engine = None
    try:
        import torch
        from .environment import make_scenarios
        from .experiments import run_episodes
        from .learning import learn
        from .preflight import check_engine
        from .reporting import plot_warning, report_stage
        from .runtime import Engine
        if args.command == "analyze":
            device = torch.device(cfg["model"]["device"])
            if device.type != "cuda" or not torch.cuda.is_available():
                raise RuntimeError("Analysis is configured for CUDA; no silent CPU fallback")
            for stage in ("e1_formation", "e3_patching", "e4_timing"):
                report_stage(store, cfg, stage, device)
            plot_warning(store)
            return 0
        provenance_path = store.root / "model_provenance.json"
        revision = json.loads(provenance_path.read_text())["resolved_revision"] if provenance_path.exists() else None
        store.budget.check()
        engine = Engine(cfg, store.budget, pinned_revision=revision)
        if provenance_path.exists():
            old = json.loads(provenance_path.read_text())
            if old["versions"] != engine.provenance["versions"]:
                raise RuntimeError("Software versions changed. Start a new run instead of mixing numerical implementations.")
        else:
            atomic_json(provenance_path, engine.provenance)
        scenarios_path = store.root / "scenario_registry.json"
        if scenarios_path.exists():
            scenarios = json.loads(scenarios_path.read_text())
        else:
            scenarios = make_scenarios(cfg, engine.device)
            atomic_json(scenarios_path, scenarios)
        preflight_path = store.root / "runtime_preflight.json"
        if not preflight_path.exists():
            atomic_json(preflight_path, check_engine(engine, cfg, scenarios[0]))
        checked = json.loads(preflight_path.read_text())
        if checked['status'] != 'passed' or not checked.get('semantic_controls_enforced'):
            raise RuntimeError('Preflight did not pass its semantic controls. Inspect runtime_preflight.json; empirical collection is blocked.')
        engine.label_mass_reference = checked['control_label_mass_reference']
        if args.command == 'profile':
            from .profiling import run_profile
            run_profile(engine, store, cfg, scenarios[0])
            return 0
        if args.command == "preflight":
            print("Runtime preflight passed. This is not an empirical experiment result.", flush=True)
            return 0
        run_episodes(engine, store, cfg, scenarios, "e1_formation")
        report_stage(store, cfg, "e1_formation", engine.device)
        learned = learn(engine, store, cfg)
        plot_warning(store)
        run_episodes(engine, store, cfg, scenarios, "e3_patching", learned)
        report_stage(store, cfg, "e3_patching", engine.device)
        run_episodes(engine, store, cfg, scenarios, "e4_timing", learned)
        report_stage(store, cfg, "e4_timing", engine.device)
        store.status("run", "completed", accounting=store.budget.accounting(), telemetry=engine.telemetry(),
                     interpretation="Read stage statuses: unidentified mechanisms are not positive findings.")
        print(f"Finished. Inspect {store.root}. Terminate the Lambda instance when done to stop billing.", flush=True)
        return 0
    except BudgetExceeded as exc:
        store.status("run", "budget_stopped", error=clean_error(exc), accounting=store.budget.accounting())
        print(clean_error(exc), file=sys.stderr, flush=True)
        return 75
    except KeyboardInterrupt:
        store.status("run", "interrupted", accounting=store.budget.accounting())
        return 130
    except Exception as exc:
        store.status("run", "failed", error=clean_error(exc), accounting=store.budget.accounting())
        print(clean_error(exc), file=sys.stderr, flush=True)
        print("Committed episodes are retained. CUDA device-assert errors require a fresh process, not an in-process retry.", file=sys.stderr)
        return 1
    finally:
        if engine is not None:
            atomic_json(store.root / "runtime_sessions" / f"{uuid.uuid4().hex}.json",
                        {"telemetry": engine.telemetry(), "accounting": store.budget.accounting()})
            engine.close()


if __name__ == "__main__":
    raise SystemExit(main())
