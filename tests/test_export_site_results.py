"""Synthetic inputs in this file are test fixtures, never website results."""

from contextlib import redirect_stderr, redirect_stdout
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import statistics
import tempfile
import unittest

MODULE = Path(__file__).resolve().parents[1] / "tools" / "export_site_results.py"
SPEC = importlib.util.spec_from_file_location("export_site_results", MODULE)
exporter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(exporter)


def synthetic_probe(probe_id="TEST_ONLY_FIRST"):
    branches = {
        key: {"support": 0.7, "withdraw": 0.3, "neutral": 0.5}[key.split("/")[-1]]
        - (0.1 if "/reminder/" in key and key.endswith("support") else 0)
        for key in exporter.BRANCH_KEYS
    }
    return {
        "probe_id": probe_id,
        "created_utc": "2026-09-13T10:00:00+00:00",
        "source_run": "/home/private-username/results/TEST_ONLY_RUN",
        "model_revision": "TEST_ONLY_REVISION",
        "prefixes_are_constructed_not_naturally_occurring": True,
        "action_menu_size": 3,
        "scenarios": {"TEST_ONLY_S1": branches.copy(), "TEST_ONLY_S2": branches.copy()},
    }


def supplied_test_summary(probe):
    """A constant-data fixture has a degenerate interval under every resample."""
    scenarios = list(probe["scenarios"].values())

    def vals(a, v, variant, stance):
        return [row[f"{a}/{v}/{variant}/{stance}"] for row in scenarios]

    def c(a, v, variant):
        return [x - y for x, y in zip(vals(a, v, variant, "support"),
                                      vals(a, v, variant, "withdraw"))]

    def estimate(values):
        mean = statistics.fmean(values)
        return {"n_scenarios": len(values), "estimate": mean,
                "ci_low": mean, "ci_high": mean,
                "confidence_level": 0.95, "bootstrap_resamples": 2500}

    summary = {}
    for a in exporter.AUTHORIZATIONS:
        for v in exporter.VISIBILITIES:
            for variant in exporter.VARIANTS:
                summary[f"{a}/{v}/{variant}"] = {
                    "C_t": estimate(c(a, v, variant)),
                    **{"mean_P_" + stance: statistics.fmean(vals(a, v, variant, stance))
                       for stance in exporter.STANCES},
                }
            summary[f"{a}/{v}/reminder_effect_on_C_t"] = estimate(
                [x - y for x, y in zip(c(a, v, "reminder"), c(a, v, "plain"))])
    for v in exporter.VISIBILITIES:
        for variant in exporter.VARIANTS:
            summary[f"{v}/{variant}/prohibited_minus_authorized_C_t"] = estimate(
                [x - y for x, y in zip(c("prohibited", v, variant), c("authorized", v, variant))])
    return summary


class ExportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def write_probe(self, probe=None, folder="run"):
        path = self.root / folder / exporter.PROBE_NAME
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(probe or synthetic_probe()))
        return path

    def test_missing_results_is_explicit_not_imported(self):
        data = exporter.build_export(self.root / "missing")
        self.assertEqual(data["status"], "not_imported")
        self.assertEqual(data["studies"], [])

    def test_missing_summary_retains_raw_means_without_intervals(self):
        path = self.write_probe()
        data = exporter.build_export(self.root)
        study = data["studies"][0]
        self.assertEqual(study["scenario_count"], 2)
        self.assertEqual(len(study["rows"]), 8)
        self.assertEqual(len(study["reminder_effects"]), 4)
        self.assertEqual(study["source_run_id"], "TEST_ONLY_RUN")
        self.assertNotIn("private-username", json.dumps(data))
        self.assertEqual(study["relative_source"], f"run/{exporter.PROBE_NAME}")
        self.assertEqual(study["provenance"]["raw_sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
        for row in study["rows"]:
            self.assertIsNone(row["ci_low"])
            self.assertAlmostEqual(row["c_t"], 0.4 if row["variant"] == "plain" else 0.3)
        for row in study["reminder_effects"]:
            self.assertAlmostEqual(row["delta_c_t"], -0.1)
            self.assertAlmostEqual(row["delta_p_support"], -0.1)
            self.assertAlmostEqual(row["delta_p_withdraw"], 0)

    def test_valid_summary_retains_checked_intervals(self):
        probe = synthetic_probe()
        path = self.write_probe(probe)
        path.with_name(exporter.SUMMARY_NAME).write_text(json.dumps(supplied_test_summary(probe)))
        study = exporter.build_export(self.root)["studies"][0]
        self.assertEqual(study["provenance"]["summary_status"], "verified_against_raw")
        for row in study["rows"]:
            self.assertAlmostEqual(row["ci_low"], row["c_t"])
            self.assertEqual(row["ci_source"], "provided_summary")

    def test_invalid_branch_missing_extra_or_probability(self):
        changes = ("missing", "extra", "out_of_range", "not_numeric", "bool")
        for change in changes:
            with self.subTest(change=change):
                probe = synthetic_probe()
                row = probe["scenarios"]["TEST_ONLY_S1"]
                key = next(iter(row))
                if change == "missing":
                    del row[key]
                elif change == "extra":
                    row["unauthorized/key"] = 0.5
                else:
                    row[key] = {"out_of_range": 1.1, "not_numeric": "0.5", "bool": True}[change]
                self.write_probe(probe)
                with self.assertRaises(exporter.ValidationError):
                    exporter.build_export(self.root)

    def test_duplicate_json_keys_and_nonfinite_fail(self):
        path = self.write_probe()
        for text in ('{"duplicate":1,"duplicate":2}', '{"value":NaN}'):
            path.write_text(text)
            with self.assertRaises(exporter.ValidationError):
                exporter.build_export(self.root)

    def test_multiple_probes_preserve_independent_studies(self):
        self.write_probe(folder="first")
        self.write_probe(synthetic_probe("TEST_ONLY_SECOND"), folder="second")
        data = exporter.build_export(self.root)
        self.assertEqual(len(data["studies"]), 2)
        self.assertEqual([s["scenario_count"] for s in data["studies"]], [2, 2])

    def test_duplicate_probe_ids_fail_instead_of_double_counting(self):
        self.write_probe(folder="first")
        self.write_probe(folder="copied")
        with self.assertRaisesRegex(exporter.ValidationError, "Duplicate probe_id"):
            exporter.build_export(self.root)

    def test_mismatched_summary_does_not_replace_existing_output(self):
        probe = synthetic_probe()
        path = self.write_probe(probe)
        summary = supplied_test_summary(probe)
        summary["authorized/private/plain"]["C_t"]["n_scenarios"] = 10
        path.with_name(exporter.SUMMARY_NAME).write_text(json.dumps(summary))
        output = self.root / "site.json"
        output.write_text("ORIGINAL_DERIVED_OUTPUT")
        with redirect_stderr(io.StringIO()):
            code = exporter.main(["--results-root", str(self.root), "--output", str(output)])
        self.assertEqual(code, 1)
        self.assertEqual(output.read_text(), "ORIGINAL_DERIVED_OUTPUT")

    def test_summary_estimate_mismatch_is_rejected(self):
        probe = synthetic_probe()
        path = self.write_probe(probe)
        summary = supplied_test_summary(probe)
        summary["authorized/private/plain"]["mean_P_support"] = 0.8
        path.with_name(exporter.SUMMARY_NAME).write_text(json.dumps(summary))
        with self.assertRaisesRegex(exporter.ValidationError, "does not match"):
            exporter.build_export(self.root)

    def test_explicit_input_is_private_and_not_overwritten(self):
        path = self.write_probe()
        data = exporter.build_export(self.root / "separate_results", [path])
        study = data["studies"][0]
        self.assertEqual(study["relative_source"], "external_inputs/1/" + exporter.PROBE_NAME)
        with redirect_stderr(io.StringIO()):
            code = exporter.main(["--probe", str(path), "--output", str(path)])
        self.assertEqual(code, 1)
        self.assertEqual(json.loads(path.read_text())["probe_id"], "TEST_ONLY_FIRST")

    def test_same_path_selected_twice_is_only_one_input(self):
        path = self.write_probe()
        data = exporter.build_export(self.root, [path, path])
        self.assertEqual(len(data["studies"]), 1)


if __name__ == "__main__":
    unittest.main()
