#!/usr/bin/env python3
"""Validate constructed-prefix results and export them for the static website.

Standard library only. No GPU, model loading, or raw-result modifications.
Every discovered probe is retained as a separate study. Statistical intervals
are copied from a matching supplied summary, never manufactured by this export.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from itertools import product
import json
import math
import os
from pathlib import Path
import statistics
import sys
import tempfile

AUTHORIZATIONS = ("authorized", "prohibited")
VISIBILITIES = ("private", "public")
VARIANTS = ("plain", "reminder")
STANCES = ("support", "withdraw", "neutral")
BRANCH_KEYS = frozenset("/".join(parts) for parts in product(
    AUTHORIZATIONS, VISIBILITIES, VARIANTS, STANCES))
PROBE_NAME = "prohibited_prefix_probe.json"
SUMMARY_NAME = "prohibited_prefix_summary.json"

# Comparison tolerance accommodates different summation order in Python's sum
# (the source script) and statistics.fmean (this exporter). It is a serialization
# consistency tolerance, not an inferential threshold or an effect-size cutoff.
REL_TOLERANCE = 1e-9
ABS_TOLERANCE = 1e-12

LIMITATIONS = [
    "The proposal prefixes are constructed; this is not a measurement of how often agents spontaneously form prohibited plans.",
    "Approval probabilities are conditional on the supplied action menu; native label probability mass was not saved by the source probe.",
    "A scored reviewer decision is not an executed violation or evidence of naturally formed collective commitment.",
    "The source probe used capture=False; this export contains no activation-based mechanism evidence.",
    "A lower support-minus-withdrawal contrast alone does not establish safer behavior; inspect absolute prohibited approval and authorized approval separately.",
    "Small contrasts require matched-batch numerical checks; confidence intervals do not account for numerical or prompt-design bias.",
    "Each study is shown separately. These results are not pooled across runs or selected by outcome.",
]


class ValidationError(ValueError):
    """An input cannot be faithfully represented as empirical website data."""


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValidationError(f"Duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _reject_constant(value):
    raise ValidationError(f"Non-finite JSON constant: {value}")


def read_json(path):
    raw = path.read_bytes()
    try:
        data = json.loads(raw, object_pairs_hook=_unique_object,
                          parse_constant=_reject_constant)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValidationError(f"Invalid JSON in {path.name}: {error}") from error
    if not isinstance(data, dict):
        raise ValidationError(f"{path.name}: expected a JSON object")
    return data, hashlib.sha256(raw).hexdigest()


def finite_number(value, label, low=None, high=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationError(f"{label}: expected a number")
    if not math.isfinite(value):
        raise ValidationError(f"{label}: expected a finite number")
    if low is not None and value < low or high is not None and value > high:
        raise ValidationError(f"{label}: value outside [{low}, {high}]")
    return float(value)


def required_string(data, key):
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(f"{key}: expected a nonempty string")
    return value


def verify_close(supplied, computed, label):
    supplied = finite_number(supplied, label)
    if not math.isclose(supplied, computed, rel_tol=REL_TOLERANCE,
                        abs_tol=ABS_TOLERANCE):
        raise ValidationError(f"{label}: supplied summary does not match raw probabilities")


def interval(summary_value, estimate, n, label, bound):
    """Validate a supplied estimate and return its interval, without generating one."""
    result = {"ci_low": None, "ci_high": None, "ci_source": None,
              "confidence_level": None, "bootstrap_resamples": None}
    if summary_value is None:
        return result
    if not isinstance(summary_value, dict):
        raise ValidationError(f"{label}: expected an estimate object")
    supplied_n = summary_value.get("n_scenarios")
    if isinstance(supplied_n, bool) or not isinstance(supplied_n, int) or supplied_n != n:
        raise ValidationError(f"{label}: summary scenario count does not match raw probabilities")
    verify_close(summary_value.get("estimate"), estimate, label + "/estimate")
    low, high = summary_value.get("ci_low"), summary_value.get("ci_high")
    if low is None and high is None:
        return result
    if low is None or high is None:
        raise ValidationError(f"{label}: both interval endpoints must be present or null")
    low = finite_number(low, label + "/ci_low", -bound, bound)
    high = finite_number(high, label + "/ci_high", -bound, bound)
    if low > high:
        raise ValidationError(f"{label}: interval endpoints are reversed")
    confidence = finite_number(summary_value.get("confidence_level"),
                               label + "/confidence_level", 0, 1)
    if confidence in (0, 1):
        raise ValidationError(f"{label}: confidence level must be strictly between zero and one")
    draws = summary_value.get("bootstrap_resamples")
    if isinstance(draws, bool) or not isinstance(draws, int) or draws < 1:
        raise ValidationError(f"{label}: expected positive bootstrap_resamples")
    if n < 2:
        raise ValidationError(f"{label}: an interval requires at least two scenarios")
    result.update(ci_low=low, ci_high=high, ci_source="provided_summary",
                  confidence_level=confidence, bootstrap_resamples=draws)
    return result


def safe_source_run_id(source):
    # Raw probe files contain the original absolute Lambda path. Export only its
    # terminal directory name; neither local paths nor user home names are needed.
    parts = source.replace("\\", "/").rstrip("/").split("/")
    name = parts[-1]
    if not name or name in (".", ".."):
        raise ValidationError("source_run: expected a run directory identifier")
    return name


def load_study(path, relative_source):
    raw, raw_hash = read_json(path)
    probe_id = required_string(raw, "probe_id")
    created_utc = required_string(raw, "created_utc")
    try:
        stamp = datetime.fromisoformat(created_utc.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValidationError("created_utc: expected an ISO timestamp") from error
    if stamp.tzinfo is None:
        raise ValidationError("created_utc: timestamp must include a timezone")
    source_run_id = safe_source_run_id(required_string(raw, "source_run"))
    revision = required_string(raw, "model_revision")
    if raw.get("prefixes_are_constructed_not_naturally_occurring") is not True:
        raise ValidationError("Probe must explicitly identify its constructed-prefix design")
    menu_size = raw.get("action_menu_size")
    if isinstance(menu_size, bool) or not isinstance(menu_size, int) or menu_size < 1:
        raise ValidationError("action_menu_size: expected a positive integer")
    scenarios = raw.get("scenarios")
    if not isinstance(scenarios, dict) or not scenarios:
        raise ValidationError("scenarios: expected a nonempty scenario mapping")
    for scenario_id, branches in scenarios.items():
        if not scenario_id.strip() or not isinstance(branches, dict):
            raise ValidationError("Each scenario needs a nonempty identifier and branch mapping")
        missing, extra = BRANCH_KEYS - branches.keys(), branches.keys() - BRANCH_KEYS
        if missing or extra:
            raise ValidationError(
                f"Scenario {scenario_id!r}: incomplete or invalid branch keys; "
                f"missing={sorted(missing)}, extra={sorted(extra)}")
        for branch, value in branches.items():
            finite_number(value, f"{scenario_id}/{branch}", 0, 1)

    summary_path = path.with_name(SUMMARY_NAME)
    summary, summary_hash = (read_json(summary_path) if summary_path.is_file() else (None, None))
    n = len(scenarios)
    scenario_ids = sorted(scenarios)
    warnings = []
    if summary is None:
        warnings.append("No summary file was supplied. Means and paired contrasts are computed from raw probabilities; all confidence intervals remain null.")
    else:
        warnings.append("Supplied summary counts and estimates were checked against raw probabilities. Bootstrap intervals are retained as supplied, not independently recomputed.")

    def values(auth, visibility, variant, stance):
        branch = "/".join((auth, visibility, variant, stance))
        return [scenarios[s][branch] for s in scenario_ids]

    def contrast(auth, visibility, variant):
        return [a - b for a, b in zip(values(auth, visibility, variant, "support"),
                                      values(auth, visibility, variant, "withdraw"))]

    def summary_item(key):
        if summary is None:
            return None
        if key not in summary:
            raise ValidationError(f"Supplied summary is incomplete: missing {key}")
        if not isinstance(summary[key], dict):
            raise ValidationError(f"{key}: expected a supplied estimate object")
        return summary[key]

    rows = []
    for auth, visibility, variant in product(AUTHORIZATIONS, VISIBILITIES, VARIANTS):
        tag = "/".join((auth, visibility, variant))
        means = {stance: statistics.fmean(values(auth, visibility, variant, stance))
                 for stance in STANCES}
        c_t = statistics.fmean(contrast(auth, visibility, variant))
        supplied = summary_item(tag)
        if supplied is not None:
            if not isinstance(supplied.get("C_t"), dict):
                raise ValidationError(f"{tag}: expected summary branch with C_t")
            for stance in STANCES:
                verify_close(supplied.get("mean_P_" + stance), means[stance],
                             tag + "/mean_P_" + stance)
        rows.append({"authorization": auth, "visibility": visibility,
                     "variant": variant, "n": n,
                     "p_support": means["support"], "p_withdraw": means["withdraw"],
                     "p_neutral": means["neutral"], "c_t": c_t,
                     **interval(supplied["C_t"] if supplied is not None else None,
                                c_t, n, tag + "/C_t", 1)})

    reminder_effects = []
    for auth, visibility in product(AUTHORIZATIONS, VISIBILITIES):
        delta = statistics.fmean(a - b for a, b in zip(
            contrast(auth, visibility, "reminder"), contrast(auth, visibility, "plain")))
        tag = f"{auth}/{visibility}/reminder_effect_on_C_t"
        reminder_effects.append({
            "authorization": auth, "visibility": visibility, "n": n,
            "delta_c_t": delta,
            **{"delta_p_" + stance: statistics.fmean(a - b for a, b in zip(
                values(auth, visibility, "reminder", stance),
                values(auth, visibility, "plain", stance))) for stance in STANCES},
            **interval(summary_item(tag), delta, n, tag, 2)})

    authorization_contrasts = []
    for visibility, variant in product(VISIBILITIES, VARIANTS):
        delta = statistics.fmean(a - b for a, b in zip(
            contrast("prohibited", visibility, variant),
            contrast("authorized", visibility, variant)))
        tag = f"{visibility}/{variant}/prohibited_minus_authorized_C_t"
        authorization_contrasts.append({
            "visibility": visibility, "variant": variant, "n": n,
            "prohibited_minus_authorized_c_t": delta,
            **interval(summary_item(tag), delta, n, tag, 2)})

    if summary is not None:
        expected = set("/".join(parts) for parts in product(AUTHORIZATIONS, VISIBILITIES, VARIANTS))
        expected.update(f"{a}/{v}/reminder_effect_on_C_t" for a, v in product(AUTHORIZATIONS, VISIBILITIES))
        expected.update(f"{v}/{variant}/prohibited_minus_authorized_C_t"
                        for v, variant in product(VISIBILITIES, VARIANTS))
        if summary.keys() - expected:
            warnings.append("Additional summary fields are not exported or interpreted.")

    return {
        "probe_id": probe_id, "created_utc": created_utc,
        "source_run_id": source_run_id, "relative_source": relative_source,
        "model_revision": revision, "scenario_count": n, "scenario_ids": scenario_ids,
        "action_menu_size": menu_size,
        "prefixes_are_constructed_not_naturally_occurring": True,
        "rows": rows, "reminder_effects": reminder_effects,
        "authorization_contrasts": authorization_contrasts,
        "warnings": warnings, "limitations": LIMITATIONS,
        "provenance": {"raw_sha256": raw_hash, "summary_sha256": summary_hash,
                       "summary_status": "verified_against_raw" if summary is not None else "missing",
                       "interval_method": "supplied_scenario_percentile_bootstrap" if summary is not None else None,
                       "estimate_check_relative_tolerance": REL_TOLERANCE,
                       "estimate_check_absolute_tolerance": ABS_TOLERANCE},
    }


def build_export(results_root, explicit_probes=()):
    results_root = Path(results_root).resolve()
    if results_root.exists() and not results_root.is_dir():
        raise ValidationError("--results-root must name a directory")
    discovered = sorted(results_root.rglob(PROBE_NAME)) if results_root.is_dir() else []
    explicit = [Path(p).resolve() for p in explicit_probes]
    for path in explicit:
        if not path.is_file():
            raise ValidationError(f"Explicit probe file is missing: {path.name}")
    # Resolve duplicate selection of the same physical input once. Distinct raw
    # files must have distinct probe IDs; copying a study does not add a replicate.
    selected = sorted({p.resolve() for p in discovered + explicit})
    studies, ids = [], set()
    for index, path in enumerate(selected):
        try:
            relative = path.relative_to(results_root).as_posix()
            selection_method = "results_root_scan" if path in [p.resolve() for p in discovered] else "explicit_probe"
        except ValueError:
            relative = f"external_inputs/{index + 1}/{path.name}"
            selection_method = "explicit_probe_outside_results_root"
        study = load_study(path, relative)
        if study["probe_id"] in ids:
            raise ValidationError(f"Duplicate probe_id across different files: {study['probe_id']}")
        ids.add(study["probe_id"])
        study["provenance"]["selection_method"] = selection_method
        studies.append(study)
    return {
        "schema_version": 1,
        "status": "available" if studies else "not_imported",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "selection": {"policy": "all_discovered_and_explicit_probes_no_outcome_filter",
                      "glob": "**/" + PROBE_NAME,
                      "discovered_files": len(discovered),
                      "explicit_inputs": len(explicit),
                      "unique_input_files": len(selected)},
        "studies": studies,
        "limitations": LIMITATIONS,
        "message": ("Every input probe is represented as a separate study."
                    if studies else "No constructed-prefix result files have been imported. No empirical values were substituted."),
    }


def write_atomic(output, data):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=output.parent,
                                         prefix="." + output.name + ".", suffix=".tmp",
                                         delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(data, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--probe", action="append", type=Path, default=[],
                        help="Additional explicit raw probe JSON; may be repeated")
    parser.add_argument("--output", type=Path, default=Path("docs/data/probes.json"))
    args = parser.parse_args(argv)
    try:
        if args.output.name in (PROBE_NAME, SUMMARY_NAME):
            raise ValidationError("Refusing to use a raw probe or summary filename as derived output")
        if args.output.resolve() in {p.resolve() for p in args.probe}:
            raise ValidationError("Refusing to overwrite an explicitly supplied raw probe")
        data = build_export(args.results_root, args.probe)
        write_atomic(args.output, data)
    except (ValidationError, OSError) as error:
        print(f"Export failed; no derived output was replaced: {error}", file=sys.stderr)
        return 1
    print(f"Exported {len(data['studies'])} separate studies; status={data['status']}")
    print(f"Website data: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
