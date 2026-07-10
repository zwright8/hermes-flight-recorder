import hashlib
import json
import os
import shutil
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import flightrecorder.heldout_manifest as heldout_manifest_module
from flightrecorder.cli import main
from flightrecorder.heldout_manifest import (
    HeldoutManifestError,
    build_heldout_manifest,
    write_heldout_manifest,
)
from flightrecorder.schema_registry import check_schema_file
from flightrecorder.validation import validate_artifacts


ROOT = Path(__file__).resolve().parents[1]


def run_cli(args):
    with redirect_stdout(StringIO()):
        return main(args)


class HeldoutManifestTests(unittest.TestCase):
    def test_heldout_manifest_rejects_output_aliases_to_suite_summary(self):
        for alias_kind in ("exact", "hardlink"):
            with (
                self.subTest(alias_kind=alias_kind),
                tempfile.TemporaryDirectory() as tmp,
            ):
                root = Path(tmp)
                suite = _suite_summary(
                    root / "suite_summary.json",
                    ["email_reply_completion"],
                )
                scenario = root / "scenarios" / "email_reply_completion.json"
                suite_before = suite.read_bytes()
                scenario_before = scenario.read_bytes()
                if alias_kind == "exact":
                    out = suite
                else:
                    out = root / "hardlinked_manifest.json"
                    try:
                        os.link(suite, out)
                    except OSError as exc:
                        self.skipTest(f"hardlink unavailable: {exc}")
                out_before = out.read_bytes()

                exit_code = None
                try:
                    run_cli(
                        [
                            "heldout-manifest",
                            "--suite-summary",
                            f"source={suite}",
                            "--out",
                            str(out),
                        ]
                    )
                except SystemExit as exc:
                    exit_code = exc.code

                self.assertEqual(suite.read_bytes(), suite_before)
                self.assertEqual(scenario.read_bytes(), scenario_before)
                self.assertEqual(out.read_bytes(), out_before)
                self.assertEqual(exit_code, 2)

    def test_heldout_manifest_rejects_output_aliases_to_scenario_source(self):
        for alias_kind in ("exact", "hardlink"):
            with (
                self.subTest(alias_kind=alias_kind),
                tempfile.TemporaryDirectory() as tmp,
            ):
                root = Path(tmp)
                suite = _suite_summary(
                    root / "suite_summary.json",
                    ["email_reply_completion"],
                )
                scenario = root / "scenarios" / "email_reply_completion.json"
                suite_before = suite.read_bytes()
                scenario_before = scenario.read_bytes()
                if alias_kind == "exact":
                    out = scenario
                else:
                    out = root / "hardlinked_manifest.json"
                    try:
                        os.link(scenario, out)
                    except OSError as exc:
                        self.skipTest(f"hardlink unavailable: {exc}")
                out_before = out.read_bytes()

                exit_code = None
                try:
                    run_cli(
                        [
                            "heldout-manifest",
                            "--suite-summary",
                            f"source={suite}",
                            "--out",
                            str(out),
                        ]
                    )
                except SystemExit as exc:
                    exit_code = exc.code

                self.assertEqual(suite.read_bytes(), suite_before)
                self.assertEqual(scenario.read_bytes(), scenario_before)
                self.assertEqual(out.read_bytes(), out_before)
                self.assertEqual(exit_code, 2)

    def test_heldout_manifest_protects_scenario_alias_when_fingerprint_is_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            suite = _suite_summary(
                root / "suite_summary.json",
                ["email_reply_completion"],
            )
            scenario = root / "scenarios" / "email_reply_completion.json"
            suite_payload = _read_json(suite)
            suite_payload["runs"][0].pop("scenario_sha256")
            suite.write_text(json.dumps(suite_payload), encoding="utf-8")
            suite_before = suite.read_bytes()
            scenario_before = scenario.read_bytes()

            with self.assertRaises(SystemExit) as raised:
                run_cli(
                    [
                        "heldout-manifest",
                        "--suite-summary",
                        f"source={suite}",
                        "--out",
                        str(scenario),
                    ]
                )

            self.assertEqual(raised.exception.code, 2)
            self.assertEqual(suite.read_bytes(), suite_before)
            self.assertEqual(scenario.read_bytes(), scenario_before)

    def test_heldout_manifest_alias_scan_does_not_depend_on_suite_semantics(self):
        cases = [
            "invalid_scenario_id",
            "later_run_after_malformed_path",
            "absolute_path",
            "traversal_path",
            "error_row",
        ]
        if os.name != "nt":
            cases.append("uri_like_local_path")
            cases.append("redacted_like_local_path")
        for case in cases:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                suite_dir = root / "suite"
                suite = _suite_summary(
                    suite_dir / "suite_summary.json",
                    ["first", "second"],
                )
                payload = _read_json(suite)
                if case == "invalid_scenario_id":
                    protected = suite_dir / payload["runs"][0]["scenario_path"]
                    payload["runs"][0]["scenario_id"] = None
                elif case == "later_run_after_malformed_path":
                    protected = suite_dir / payload["runs"][1]["scenario_path"]
                    payload["runs"][0]["scenario_path"] = ""
                elif case == "absolute_path":
                    protected = root / "absolute-scenario.json"
                    protected.write_bytes(b'{"owner":"protected"}\n')
                    payload["runs"][0]["scenario_path"] = str(protected)
                elif case == "traversal_path":
                    protected = root / "protected-scenario.json"
                    protected.write_bytes(b'{"owner":"protected"}\n')
                    payload["runs"][0]["scenario_path"] = "../protected-scenario.json"
                elif case == "error_row":
                    protected = suite_dir / "scenarios" / "error-source.json"
                    protected.write_bytes(b'{"owner":"protected"}\n')
                    payload["errors"] = [
                        {
                            "scenario_path": "scenarios/error-source.json",
                            "error": "synthetic harness error",
                        }
                    ]
                    payload["error_count"] = 1
                else:
                    raw_path = (
                        "scenario://uri-like-local-source.json"
                        if case == "uri_like_local_path"
                        else "<redacted:local-source.json>"
                    )
                    protected = suite_dir / Path(raw_path)
                    protected.parent.mkdir(parents=True, exist_ok=True)
                    protected.write_bytes(b'{"owner":"protected"}\n')
                    payload["runs"][0]["scenario_path"] = raw_path
                suite.write_text(json.dumps(payload), encoding="utf-8")
                suite_before = suite.read_bytes()
                protected_before = protected.read_bytes()

                with self.assertRaises(SystemExit) as raised:
                    run_cli(
                        [
                            "heldout-manifest",
                            "--suite-summary",
                            f"source={suite}",
                            "--out",
                            str(protected),
                        ]
                    )

                self.assertEqual(raised.exception.code, 2)
                self.assertEqual(suite.read_bytes(), suite_before)
                self.assertEqual(protected.read_bytes(), protected_before)

    def test_heldout_manifest_reports_embedded_nul_scenario_path_cleanly(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            suite = _suite_summary(root / "suite.json", ["email_reply_completion"])
            scenario = root / "scenarios" / "email_reply_completion.json"
            payload = _read_json(suite)
            payload["runs"][0]["scenario_path"] = "bad\x00path"
            suite.write_text(json.dumps(payload), encoding="utf-8")
            suite_before = suite.read_bytes()
            scenario_before = scenario.read_bytes()
            out = root / "heldout_manifest.json"

            with self.assertRaises(SystemExit) as raised:
                run_cli(
                    [
                        "heldout-manifest",
                        "--suite-summary",
                        str(suite),
                        "--out",
                        str(out),
                    ]
                )

            self.assertEqual(raised.exception.code, 2)
            self.assertEqual(suite.read_bytes(), suite_before)
            self.assertEqual(scenario.read_bytes(), scenario_before)
            self.assertFalse(out.exists())

    def test_direct_heldout_writer_rejects_source_aliases_without_builder_hint(self):
        for source_kind in ("suite", "scenario"):
            with (
                self.subTest(source_kind=source_kind),
                tempfile.TemporaryDirectory() as tmp,
            ):
                root = Path(tmp)
                suite = _suite_summary(
                    root / "suite_summary.json",
                    ["email_reply_completion"],
                )
                scenario = root / "scenarios" / "email_reply_completion.json"
                manifest = build_heldout_manifest(
                    suite_summary_specs=[f"source={suite}"]
                )
                protected = suite if source_kind == "suite" else scenario
                suite_before = suite.read_bytes()
                scenario_before = scenario.read_bytes()

                with self.assertRaisesRegex(HeldoutManifestError, "must not alias"):
                    write_heldout_manifest(manifest, protected)

                self.assertEqual(suite.read_bytes(), suite_before)
                self.assertEqual(scenario.read_bytes(), scenario_before)

    def test_direct_heldout_writer_rejects_suite_drift_after_build(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            suite = _suite_summary(
                root / "suite_summary.json",
                ["email_reply_completion"],
            )
            original = root / "scenarios" / "email_reply_completion.json"
            replacement = root / "scenarios" / "replacement.json"
            replacement.write_bytes(b'{"owner":"replacement"}\n')
            manifest = build_heldout_manifest(
                suite_summary_specs=[f"source={suite}"]
            )
            payload = _read_json(suite)
            payload["runs"][0]["scenario_path"] = "scenarios/replacement.json"
            payload["runs"][0]["scenario_sha256"] = hashlib.sha256(
                replacement.read_bytes()
            ).hexdigest()
            suite.write_text(json.dumps(payload), encoding="utf-8")
            suite_before = suite.read_bytes()
            original_before = original.read_bytes()

            with self.assertRaisesRegex(HeldoutManifestError, "changed after manifest build"):
                write_heldout_manifest(manifest, original)

            self.assertEqual(suite.read_bytes(), suite_before)
            self.assertEqual(original.read_bytes(), original_before)

    def test_direct_heldout_writer_binds_relative_sources_to_build_cwd(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            build_dir = root / "build"
            other_dir = root / "other"
            build_suite = _suite_summary(
                build_dir / "suite.json",
                ["email_reply_completion"],
            )
            original = build_dir / "scenarios" / "email_reply_completion.json"
            _suite_summary(other_dir / "suite.json", ["different_scenario"])
            original_before = original.read_bytes()
            previous_cwd = Path.cwd()
            try:
                os.chdir(build_dir)
                manifest = build_heldout_manifest(
                    suite_summary_specs=[build_suite.name]
                )
                os.chdir(other_dir)
                with self.assertRaisesRegex(HeldoutManifestError, "must not alias"):
                    write_heldout_manifest(manifest, original)
            finally:
                os.chdir(previous_cwd)

            self.assertEqual(original.read_bytes(), original_before)

    def test_direct_heldout_writer_rewrites_bound_source_after_cwd_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            build_dir = root / "build"
            other_dir = root / "other"
            build_suite = _suite_summary(build_dir / "suite.json", ["source_scenario"])
            _suite_summary(other_dir / "suite.json", ["other_scenario"])
            other_scenario = other_dir / "scenarios" / "other_scenario.json"
            other_before = other_scenario.read_bytes()
            out = other_dir / "heldout_manifest.json"
            previous_cwd = Path.cwd()
            try:
                os.chdir(build_dir)
                manifest = build_heldout_manifest(
                    suite_summary_specs=[build_suite.name]
                )
                os.chdir(other_dir)
                write_heldout_manifest(manifest, out)
            finally:
                os.chdir(previous_cwd)

            written = _read_json(out)
            expected_source = os.path.relpath(build_suite.resolve(), other_dir.resolve())
            self.assertEqual(written["sources"][0]["path"], expected_source)
            self.assertEqual(other_scenario.read_bytes(), other_before)

    def test_direct_heldout_writer_rejects_ambiguous_cwd_scenario_alias(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            build_dir = root / "build"
            other_dir = root / "other"
            build_suite = _suite_summary(build_dir / "suite.json", ["source_scenario"])
            _suite_summary(other_dir / "suite.json", ["other_scenario"])
            other_scenario = other_dir / "scenarios" / "other_scenario.json"
            other_before = other_scenario.read_bytes()
            previous_cwd = Path.cwd()
            try:
                os.chdir(build_dir)
                manifest = build_heldout_manifest(
                    suite_summary_specs=[build_suite.name]
                )
                os.chdir(other_dir)
                with self.assertRaisesRegex(HeldoutManifestError, "must not alias"):
                    write_heldout_manifest(manifest, other_scenario)
            finally:
                os.chdir(previous_cwd)

            self.assertEqual(other_scenario.read_bytes(), other_before)

    def test_direct_heldout_writer_survives_source_parent_symlink_retarget(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_a = root / "source-a"
            source_b = root / "source-b"
            suite_a = _suite_summary(source_a / "suite.json", ["shared_scenario"])
            source_b.mkdir()
            try:
                os.link(suite_a, source_b / "suite.json")
            except OSError as exc:
                self.skipTest(f"hardlinks unavailable: {exc}")
            scenario_b = source_b / "scenarios" / "shared_scenario.json"
            scenario_b.parent.mkdir()
            scenario_b.write_bytes(b'{"owner":"source-b"}\n')
            current = root / "current"
            try:
                current.symlink_to(source_a, target_is_directory=True)
            except OSError as exc:
                self.skipTest(f"symlinks unavailable: {exc}")
            scenario_a = source_a / "scenarios" / "shared_scenario.json"
            scenario_a_before = scenario_a.read_bytes()
            previous_cwd = Path.cwd()
            try:
                os.chdir(root)
                manifest = build_heldout_manifest(
                    suite_summary_specs=["current/suite.json"]
                )
                current.unlink()
                current.symlink_to(source_b, target_is_directory=True)
                with self.assertRaisesRegex(HeldoutManifestError, "must not alias"):
                    write_heldout_manifest(manifest, scenario_a)
            finally:
                os.chdir(previous_cwd)

            self.assertEqual(scenario_a.read_bytes(), scenario_a_before)

    def test_direct_heldout_writer_rejects_source_directory_replacement(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            current = root / "current"
            moved = root / "moved"
            suite = _suite_summary(current / "suite.json", ["shared_scenario"])
            original = current / "scenarios" / "shared_scenario.json"
            original_before = original.read_bytes()
            previous_cwd = Path.cwd()
            try:
                os.chdir(root)
                manifest = build_heldout_manifest(
                    suite_summary_specs=["current/suite.json"]
                )
                current.rename(moved)
                current.mkdir()
                try:
                    os.link(moved / suite.name, current / suite.name)
                except OSError as exc:
                    self.skipTest(f"hardlinks unavailable: {exc}")
                replacement = current / "scenarios" / "shared_scenario.json"
                replacement.parent.mkdir()
                replacement.write_bytes(b'{"owner":"replacement"}\n')
                with self.assertRaisesRegex(
                    HeldoutManifestError,
                    "scenario source changed after manifest build",
                ):
                    write_heldout_manifest(
                        manifest,
                        moved / "scenarios" / "shared_scenario.json",
                    )
            finally:
                os.chdir(previous_cwd)

            self.assertEqual(
                (moved / "scenarios" / "shared_scenario.json").read_bytes(),
                original_before,
            )

    def test_direct_heldout_writer_rejects_plain_unbound_manifest_dict(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            suite = _suite_summary(root / "suite.json", ["email_reply_completion"])
            manifest = build_heldout_manifest(suite_summary_specs=[suite])

            with self.assertRaisesRegex(HeldoutManifestError, "returned directly"):
                write_heldout_manifest(dict(manifest), root / "heldout_manifest.json")

    def test_direct_heldout_writer_rejects_mutated_source_projection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            suite = _suite_summary(root / "suite.json", ["email_reply_completion"])
            manifest = build_heldout_manifest(suite_summary_specs=[suite])
            protected = root / "unrelated-source.json"
            protected.write_bytes(b'{"owner":"protected"}\n')
            protected_before = protected.read_bytes()
            manifest["sources"][0]["path"] = str(protected)

            with self.assertRaisesRegex(HeldoutManifestError, "sources changed"):
                write_heldout_manifest(manifest, protected)

            self.assertEqual(protected.read_bytes(), protected_before)

    def test_direct_heldout_writer_captures_default_cas_baseline_at_entry(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            suite = _suite_summary(root / "suite.json", ["email_reply_completion"])
            manifest = build_heldout_manifest(suite_summary_specs=[suite])
            out = root / "heldout_manifest.json"
            out.write_bytes(b'{"owner":"initial"}\n')
            competing_bytes = b'{"owner":"competing"}\n'
            actual_reject_aliases = heldout_manifest_module._reject_output_aliases
            mutated = False

            def mutate_output_during_source_checks(*args, **kwargs):
                nonlocal mutated
                result = actual_reject_aliases(*args, **kwargs)
                if not mutated:
                    out.write_bytes(competing_bytes)
                    mutated = True
                return result

            with patch(
                "flightrecorder.heldout_manifest._reject_output_aliases",
                side_effect=mutate_output_during_source_checks,
            ):
                with self.assertRaisesRegex(ValueError, "changed concurrently"):
                    write_heldout_manifest(manifest, out)

            self.assertEqual(out.read_bytes(), competing_bytes)

    def test_heldout_manifest_rejects_leaf_output_symlink(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            suite = _suite_summary(
                root / "suite_summary.json",
                ["email_reply_completion"],
            )
            scenario = root / "scenarios" / "email_reply_completion.json"
            target = root / "protected.json"
            target.write_bytes(b'{"owner":"protected"}\n')
            out = root / "heldout_manifest.json"
            try:
                out.symlink_to(target)
            except OSError as exc:
                self.skipTest(f"symlink unavailable: {exc}")
            suite_before = suite.read_bytes()
            scenario_before = scenario.read_bytes()
            target_before = target.read_bytes()

            exit_code = None
            try:
                run_cli(
                    [
                        "heldout-manifest",
                        "--suite-summary",
                        f"source={suite}",
                        "--out",
                        str(out),
                    ]
                )
            except SystemExit as exc:
                exit_code = exc.code

            self.assertEqual(suite.read_bytes(), suite_before)
            self.assertEqual(scenario.read_bytes(), scenario_before)
            self.assertEqual(target.read_bytes(), target_before)
            self.assertTrue(out.is_symlink())
            self.assertEqual(exit_code, 2)

    def test_heldout_manifest_rejects_symlinked_output_parent(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            suite = _suite_summary(
                root / "suite_summary.json",
                ["email_reply_completion"],
            )
            scenario = root / "scenarios" / "email_reply_completion.json"
            target_parent = root / "protected"
            target_parent.mkdir()
            target = target_parent / "heldout_manifest.json"
            target.write_bytes(b'{"owner":"protected"}\n')
            linked_parent = root / "linked-output"
            try:
                linked_parent.symlink_to(target_parent, target_is_directory=True)
            except OSError as exc:
                self.skipTest(f"symlink unavailable: {exc}")
            out = linked_parent / target.name
            suite_before = suite.read_bytes()
            scenario_before = scenario.read_bytes()
            target_before = target.read_bytes()

            exit_code = None
            try:
                run_cli(
                    [
                        "heldout-manifest",
                        "--suite-summary",
                        f"source={suite}",
                        "--out",
                        str(out),
                    ]
                )
            except SystemExit as exc:
                exit_code = exc.code

            self.assertEqual(suite.read_bytes(), suite_before)
            self.assertEqual(scenario.read_bytes(), scenario_before)
            self.assertEqual(target.read_bytes(), target_before)
            self.assertTrue(linked_parent.is_symlink())
            self.assertEqual(exit_code, 2)

    def test_heldout_manifest_atomic_publish_rejects_post_digest_conflict(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            suite = _suite_summary(
                root / "suite_summary.json",
                ["email_reply_completion"],
            )
            scenario = root / "scenarios" / "email_reply_completion.json"
            out = root / "heldout_manifest.json"
            out.write_bytes(b'{"owner":"initial"}\n')
            suite_before = suite.read_bytes()
            scenario_before = scenario.read_bytes()
            competing_bytes = b'{"owner":"competing"}\n'

            from flightrecorder.cli import write_heldout_manifest as actual_write

            def compete_then_write(*args, **kwargs):
                out.write_bytes(competing_bytes)
                return actual_write(*args, **kwargs)

            exit_code = None
            with patch(
                "flightrecorder.cli.write_heldout_manifest",
                side_effect=compete_then_write,
            ):
                try:
                    run_cli(
                        [
                            "heldout-manifest",
                            "--suite-summary",
                            f"source={suite}",
                            "--out",
                            str(out),
                        ]
                    )
                except SystemExit as exc:
                    exit_code = exc.code

            self.assertEqual(suite.read_bytes(), suite_before)
            self.assertEqual(scenario.read_bytes(), scenario_before)
            self.assertEqual(out.read_bytes(), competing_bytes)
            self.assertEqual(exit_code, 2)

    def test_heldout_manifest_blocks_schema_invalid_suite_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            suite = _suite_summary(root / "baseline_suite.json", ["email_reply_completion"])
            payload = _read_json(suite)
            payload.pop("scenarios_dir")
            suite.write_text(json.dumps(payload), encoding="utf-8")
            out = root / "heldout_manifest.json"

            code = run_cli(["heldout-manifest", "--suite-summary", f"baseline={suite}", "--out", str(out)])

            self.assertEqual(code, 1)
            manifest = _read_json(out)
            self.assertFalse(manifest["ready"])
            self.assertIn("invalid_suite_summary_schema", manifest["blocking_reasons"])

    def test_committed_agentic_training_heldout_manifest_replays_suite_summaries(self):
        eval_root = ROOT / "examples" / "agentic_training" / "heldout_eval"
        suite_manifest_path = eval_root / "heldout_suite_manifest.json"
        manifest_path = eval_root / "heldout_manifest.json"
        manifest = _read_json(manifest_path)

        self.assertEqual(
            manifest["scenario_ids"],
            ["prompt_injection_bad", "prompt_injection_good", "subagent_claim_bad"],
        )
        self.assertTrue(manifest["ready"])
        self.assertEqual(manifest["status"], "identical")
        self.assertTrue(manifest["cross_arm_claims_allowed"])
        self.assertEqual({source["label"] for source in manifest["sources"]}, {"baseline", "candidate"})
        self.assertEqual(run_cli(["schemas", "--check", str(suite_manifest_path)]), 0)
        validation = validate_artifacts(
            eval_suite_manifest_paths=[suite_manifest_path],
            heldout_manifest_paths=[manifest_path],
            strict=True,
        )
        self.assertTrue(validation["passed"], validation)

    def test_heldout_manifest_allows_single_source_external_adapter_seed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            suite = _suite_summary(root / "baseline_suite.json", ["email_reply_completion"])
            out = root / "heldout_manifest.json"

            code = run_cli(["heldout-manifest", "--suite-summary", f"baseline={suite}", "--out", str(out)])
            validate_code = run_cli(["validate", "--heldout-manifest", str(out), "--strict"])
            schema_result = check_schema_file(out)

            self.assertEqual(code, 0)
            self.assertEqual(validate_code, 0)
            self.assertTrue(schema_result["passed"], schema_result["errors"])
            manifest = _read_json(out)
            self.assertTrue(manifest["ready"])
            self.assertEqual(manifest["status"], "single_source")
            self.assertFalse(manifest["cross_arm_claims_allowed"])
            self.assertEqual(manifest["scenario_ids"], ["email_reply_completion"])
            self.assertEqual(manifest["sources"][0]["path"], "baseline_suite.json")

    def test_heldout_manifest_proves_identical_cross_arm_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline = _suite_summary(root / "baseline_suite.json", ["email_reply_completion", "prompt_injection"])
            candidate = _suite_summary(root / "candidate_suite.json", ["prompt_injection", "email_reply_completion"])
            out = root / "heldout_manifest.json"

            code = run_cli(
                [
                    "heldout-manifest",
                    "--suite-summary",
                    f"baseline={baseline}",
                    "--suite-summary",
                    f"candidate={candidate}",
                    "--out",
                    str(out),
                ]
            )
            validate_code = run_cli(["validate", "--heldout-manifest", str(out), "--strict"])
            schema_result = check_schema_file(out)

            self.assertEqual(code, 0)
            self.assertEqual(validate_code, 0)
            self.assertTrue(schema_result["passed"], schema_result["errors"])
            manifest = _read_json(out)
            self.assertTrue(manifest["ready"])
            self.assertEqual(manifest["status"], "identical")
            self.assertTrue(manifest["identical"])
            self.assertTrue(manifest["cross_arm_claims_allowed"])
            self.assertEqual(manifest["scenario_count"], 2)

    def test_heldout_manifest_blocks_mismatched_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline = _suite_summary(root / "baseline_suite.json", ["email_reply_completion", "prompt_injection"])
            candidate = _suite_summary(root / "candidate_suite.json", ["email_reply_completion"])
            out = root / "heldout_manifest.json"

            code = run_cli(
                [
                    "heldout-manifest",
                    "--suite-summary",
                    f"baseline={baseline}",
                    "--suite-summary",
                    f"candidate={candidate}",
                    "--out",
                    str(out),
                ]
            )
            validate_code = run_cli(["validate", "--heldout-manifest", str(out), "--strict"])
            schema_result = check_schema_file(out)

            self.assertEqual(code, 1)
            self.assertEqual(validate_code, 0)
            self.assertTrue(schema_result["passed"], schema_result["errors"])
            manifest = _read_json(out)
            self.assertFalse(manifest["ready"])
            self.assertEqual(manifest["status"], "mismatched")
            self.assertIn("heldout_scenario_set_mismatch", manifest["blocking_reasons"])
            self.assertEqual(manifest["mismatches"][0]["missing_from_source"], ["prompt_injection"])

    def test_heldout_manifest_blocks_same_scenario_id_with_different_fingerprints(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline = _suite_summary(
                root / "baseline" / "suite_summary.json",
                ["email_reply_completion"],
                scenario_overrides=[{"prompt": "Complete the baseline held-out task."}],
            )
            candidate = _suite_summary(
                root / "candidate" / "suite_summary.json",
                ["email_reply_completion"],
                scenario_overrides=[{"prompt": "Complete the candidate held-out task."}],
            )
            baseline_sha = _read_json(baseline)["runs"][0]["scenario_sha256"]
            candidate_sha = _read_json(candidate)["runs"][0]["scenario_sha256"]
            out = root / "heldout_manifest.json"

            code = run_cli(
                [
                    "heldout-manifest",
                    "--suite-summary",
                    f"baseline={baseline}",
                    "--suite-summary",
                    f"candidate={candidate}",
                    "--out",
                    str(out),
                ]
            )
            validate_code = run_cli(["validate", "--heldout-manifest", str(out), "--strict"])

            self.assertEqual(code, 1)
            self.assertEqual(validate_code, 0)
            manifest = _read_json(out)
            self.assertFalse(manifest["ready"])
            self.assertEqual(manifest["status"], "mismatched")
            self.assertFalse(manifest["identical"])
            self.assertFalse(manifest["cross_arm_claims_allowed"])
            self.assertIn("heldout_scenario_fingerprint_mismatch", manifest["blocking_reasons"])
            mismatch_evidence = json.dumps(manifest["mismatches"], sort_keys=True)
            self.assertIn("email_reply_completion", mismatch_evidence)
            self.assertIn(baseline_sha, mismatch_evidence)
            self.assertIn(candidate_sha, mismatch_evidence)

    def test_heldout_manifest_blocks_recorded_fingerprint_that_does_not_match_scenario(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline = _suite_summary(
                root / "baseline" / "suite_summary.json",
                ["email_reply_completion"],
                scenario_overrides=[{"prompt": "Complete the baseline held-out task."}],
            )
            candidate = _suite_summary(
                root / "candidate" / "suite_summary.json",
                ["email_reply_completion"],
                scenario_overrides=[{"prompt": "Complete the candidate held-out task."}],
            )
            baseline_sha = _read_json(baseline)["runs"][0]["scenario_sha256"]
            candidate_payload = _read_json(candidate)
            candidate_payload["runs"][0]["scenario_sha256"] = baseline_sha
            candidate.write_text(
                json.dumps(candidate_payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            out = root / "heldout_manifest.json"

            code = run_cli(
                [
                    "heldout-manifest",
                    "--suite-summary",
                    f"baseline={baseline}",
                    "--suite-summary",
                    f"candidate={candidate}",
                    "--out",
                    str(out),
                ]
            )
            validate_code = run_cli(["validate", "--heldout-manifest", str(out), "--strict"])

            self.assertEqual(code, 1)
            self.assertEqual(validate_code, 0)
            manifest = _read_json(out)
            self.assertEqual(manifest["status"], "blocked")
            self.assertFalse(manifest["cross_arm_claims_allowed"])
            self.assertIn("scenario_fingerprint_replay_failed", manifest["blocking_reasons"])

    def test_heldout_manifest_blocks_duplicate_source_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            suite = _suite_summary(root / "suite_summary.json", ["email_reply_completion"])
            out = root / "heldout_manifest.json"

            code = run_cli(
                [
                    "heldout-manifest",
                    "--suite-summary",
                    f"baseline={suite}",
                    "--suite-summary",
                    f"candidate={suite}",
                    "--out",
                    str(out),
                ]
            )
            validate_code = run_cli(["validate", "--heldout-manifest", str(out), "--strict"])

            self.assertEqual(code, 1)
            self.assertEqual(validate_code, 0)
            manifest = _read_json(out)
            self.assertEqual(manifest["status"], "blocked")
            self.assertFalse(manifest["cross_arm_claims_allowed"])
            self.assertIn("duplicate_heldout_source_paths", manifest["blocking_reasons"])

    def test_heldout_manifest_blocks_content_aliased_sources(self):
        for alias_kind in ("hardlink", "copy"):
            with self.subTest(alias_kind=alias_kind), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                baseline = _suite_summary(root / "baseline_suite.json", ["email_reply_completion"])
                candidate = root / "candidate_suite.json"
                if alias_kind == "hardlink":
                    os.link(baseline, candidate)
                else:
                    shutil.copyfile(baseline, candidate)
                out = root / "heldout_manifest.json"

                code = run_cli(
                    [
                        "heldout-manifest",
                        "--suite-summary",
                        f"baseline={baseline}",
                        "--suite-summary",
                        f"candidate={candidate}",
                        "--out",
                        str(out),
                    ]
                )
                validate_code = run_cli(["validate", "--heldout-manifest", str(out), "--strict"])

                self.assertEqual(code, 1)
                self.assertEqual(validate_code, 0)
                manifest = _read_json(out)
                self.assertEqual(manifest["status"], "blocked")
                self.assertFalse(manifest["cross_arm_claims_allowed"])
                self.assertIn(
                    "duplicate_heldout_source_content",
                    manifest["blocking_reasons"],
                )

    def test_heldout_manifest_blocks_missing_or_incomplete_fingerprints(self):
        for missing_value in ("absent", "null"):
            with self.subTest(missing_value=missing_value), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                suite = _suite_summary(
                    root / "baseline_suite.json",
                    ["email_reply_completion", "prompt_injection"],
                )
                suite_payload = _read_json(suite)
                if missing_value == "absent":
                    suite_payload["runs"][1].pop("scenario_sha256")
                else:
                    suite_payload["runs"][1]["scenario_sha256"] = None
                suite.write_text(
                    json.dumps(suite_payload, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                out = root / "heldout_manifest.json"

                self.assertTrue(check_schema_file(suite)["passed"])
                code = run_cli(
                    ["heldout-manifest", "--suite-summary", f"baseline={suite}", "--out", str(out)]
                )
                validate_code = run_cli(["validate", "--heldout-manifest", str(out), "--strict"])
                schema_result = check_schema_file(out)

                self.assertEqual(code, 1)
                self.assertEqual(validate_code, 0)
                self.assertTrue(schema_result["passed"], schema_result["errors"])
                manifest = _read_json(out)
                self.assertFalse(manifest["ready"])
                self.assertEqual(manifest["status"], "blocked")
                self.assertFalse(manifest["cross_arm_claims_allowed"])
                self.assertIn("missing_scenario_fingerprints", manifest["blocking_reasons"])
                self.assertIn("missing_scenario_fingerprints", manifest["sources"][0]["blocking_reasons"])
                self.assertEqual(len(manifest["sources"][0]["scenario_fingerprints"]), 1)

    def test_strict_validate_rejects_forged_identical_fingerprint_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline = _suite_summary(
                root / "baseline" / "suite_summary.json",
                ["email_reply_completion"],
                scenario_overrides=[{"prompt": "Complete the baseline held-out task."}],
            )
            candidate = _suite_summary(
                root / "candidate" / "suite_summary.json",
                ["email_reply_completion"],
                scenario_overrides=[{"prompt": "Complete the candidate held-out task."}],
            )
            out = root / "heldout_manifest.json"
            run_cli(
                [
                    "heldout-manifest",
                    "--suite-summary",
                    f"baseline={baseline}",
                    "--suite-summary",
                    f"candidate={candidate}",
                    "--out",
                    str(out),
                ]
            )
            manifest = _read_json(out)
            manifest.update(
                {
                    "ready": True,
                    "status": "identical",
                    "identical": True,
                    "cross_arm_claims_allowed": True,
                    "mismatches": [],
                    "blocking_reasons": [],
                }
            )
            manifest["governance_handoff"]["external_adapter_manifest_allowed"] = True
            manifest["governance_handoff"]["cross_arm_claims_allowed"] = True
            out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            validation = validate_artifacts(heldout_manifest_paths=[out], strict=True)

            self.assertFalse(validation["passed"], validation)
            errors = "\n".join(error for target in validation["targets"] for error in target["errors"])
            self.assertIn("fingerprint", errors.lower())

    def test_validate_malformed_heldout_identity_returns_errors_instead_of_crashing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            suite = _suite_summary(root / "suite_summary.json", ["email_reply_completion"])
            out = root / "heldout_manifest.json"
            run_cli(["heldout-manifest", "--suite-summary", f"source={suite}", "--out", str(out)])
            manifest = _read_json(out)
            manifest["sources"][0]["scenario_ids"] = [{}]
            manifest["sources"][0]["blocking_reasons"] = [{}]
            out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            validation = validate_artifacts(heldout_manifest_paths=[out], strict=True)

            self.assertFalse(validation["passed"], validation)
            errors = "\n".join(error for target in validation["targets"] for error in target["errors"])
            self.assertIn("scenario_ids must be a list of strings", errors)
            self.assertIn("blocking_reasons must be a list of strings", errors)

    def test_validate_malformed_heldout_top_level_fields_returns_errors_instead_of_crashing(self):
        malformed_fields = {
            "scenario_ids": [1, True],
            "status": [{}, [], [{}]],
        }
        for field_name, values in malformed_fields.items():
            for value in values:
                with (
                    self.subTest(field_name=field_name, value=value),
                    tempfile.TemporaryDirectory() as tmp,
                ):
                    root = Path(tmp)
                    suite = _suite_summary(root / "suite_summary.json", ["email_reply_completion"])
                    out = root / "heldout_manifest.json"
                    run_cli(["heldout-manifest", "--suite-summary", f"source={suite}", "--out", str(out)])
                    manifest = _read_json(out)
                    manifest[field_name] = value
                    out.write_text(
                        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8",
                    )

                    validation = validate_artifacts(heldout_manifest_paths=[out], strict=True)

                    self.assertFalse(validation["passed"], validation)
                    errors = "\n".join(
                        error for target in validation["targets"] for error in target["errors"]
                    )
                    self.assertIn(field_name, errors)

    def test_validate_rejects_unreplayable_redacted_heldout_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            suite = _suite_summary(root / "suite_summary.json", ["email_reply_completion"])
            out = root / "heldout_manifest.json"
            run_cli(["heldout-manifest", "--suite-summary", f"source={suite}", "--out", str(out)])
            manifest = _read_json(out)
            manifest["sources"][0]["path"] = "<redacted:suite_summary.json>"
            out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            validation = validate_artifacts(heldout_manifest_paths=[out], strict=True)

            self.assertFalse(validation["passed"], validation)
            errors = "\n".join(error for target in validation["targets"] for error in target["errors"])
            self.assertIn("path must be replayable for held-out identity validation", errors)

    def test_validate_malformed_heldout_source_paths_returns_errors_instead_of_crashing(self):
        for value in ("\x00", "x" * 5000):
            with (
                self.subTest(path_kind="nul" if value == "\x00" else "overlong"),
                tempfile.TemporaryDirectory() as tmp,
            ):
                root = Path(tmp)
                suite = _suite_summary(root / "suite_summary.json", ["email_reply_completion"])
                out = root / "heldout_manifest.json"
                run_cli(["heldout-manifest", "--suite-summary", f"source={suite}", "--out", str(out)])
                manifest = _read_json(out)
                manifest["sources"][0]["path"] = value
                out.write_text(
                    json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )

                validation = validate_artifacts(heldout_manifest_paths=[out], strict=True)

                self.assertFalse(validation["passed"], validation)
                errors = "\n".join(
                    error for target in validation["targets"] for error in target["errors"]
                )
                self.assertIn("sources[0].path", errors)

    def test_external_eval_plan_blocks_not_ready_heldout_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline = _suite_summary(root / "baseline_suite.json", ["email_reply_completion", "prompt_injection"])
            candidate = _suite_summary(root / "candidate_suite.json", ["email_reply_completion"])
            manifest = root / "heldout_manifest.json"
            plan = root / "external_eval_plan.json"
            run_cli(
                [
                    "heldout-manifest",
                    "--suite-summary",
                    f"baseline={baseline}",
                    "--suite-summary",
                    f"candidate={candidate}",
                    "--out",
                    str(manifest),
                ]
            )

            code = run_cli(
                [
                    "external-eval-plan",
                    "--adapter",
                    "lm_eval_harness",
                    "--scenario-manifest",
                    str(manifest),
                    "--model-endpoint",
                    "http://127.0.0.1:8000/v1",
                    "--lm-eval-task",
                    "mmlu",
                    "--allow-installed",
                    "--out",
                    str(plan),
                ]
            )
            validate_code = run_cli(["validate", "--external-eval-plan", str(plan), "--strict"])
            schema_result = check_schema_file(plan)

            self.assertEqual(code, 1)
            self.assertEqual(validate_code, 0)
            self.assertTrue(schema_result["passed"], schema_result["errors"])
            adapter = _read_json(plan)["adapters"][0]
            self.assertIn("scenario_manifest_not_ready", adapter["blocking_reasons"])

    def test_validate_rejects_forged_ready_mismatched_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline = _suite_summary(root / "baseline_suite.json", ["email_reply_completion", "prompt_injection"])
            candidate = _suite_summary(root / "candidate_suite.json", ["email_reply_completion"])
            out = root / "heldout_manifest.json"
            validation = root / "validation.json"
            run_cli(
                [
                    "heldout-manifest",
                    "--suite-summary",
                    f"baseline={baseline}",
                    "--suite-summary",
                    f"candidate={candidate}",
                    "--out",
                    str(out),
                ]
            )
            manifest = _read_json(out)
            manifest["ready"] = True
            manifest["blocking_reasons"] = []
            out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            code = run_cli(["validate", "--heldout-manifest", str(out), "--out", str(validation)])

            self.assertEqual(code, 1)
            errors = "\n".join(error for target in _read_json(validation)["targets"] for error in target["errors"])
            self.assertIn("heldout_manifest.ready expected False", errors)
            self.assertIn("blocking_reasons must include heldout_scenario_set_mismatch", errors)

    def test_strict_validate_warns_on_absolute_source_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            suite = _suite_summary(root / "baseline_suite.json", ["email_reply_completion"])
            out = root / "heldout_manifest.json"
            run_cli(["heldout-manifest", "--suite-summary", f"baseline={suite}", "--out", str(out)])
            manifest = _read_json(out)
            manifest["sources"][0]["path"] = str(suite.resolve())
            out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            permissive_validation = validate_artifacts(heldout_manifest_paths=[out])
            validation = validate_artifacts(heldout_manifest_paths=[out], strict=True)

            self.assertTrue(permissive_validation["passed"], permissive_validation)
            self.assertGreater(permissive_validation["warning_count"], 0, permissive_validation)
            self.assertFalse(validation["passed"], validation)
            self.assertEqual(validation["error_count"], 0, validation)
            warnings = "\n".join(warning for target in validation["targets"] for warning in target["warnings"])
            self.assertIn("heldout_manifest.sources[0].path is absolute", warnings)

    def test_validate_rejects_stale_source_suite_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            suite = _suite_summary(root / "baseline_suite.json", ["email_reply_completion"])
            out = root / "heldout_manifest.json"
            run_cli(["heldout-manifest", "--suite-summary", f"baseline={suite}", "--out", str(out)])
            _suite_summary(suite, ["email_reply_completion", "prompt_injection"])

            validation = validate_artifacts(heldout_manifest_paths=[out], strict=True)

            self.assertFalse(validation["passed"], validation)
            errors = "\n".join(error for target in validation["targets"] for error in target["errors"])
            self.assertIn("heldout_manifest.sources[0].scenario_count must match the current suite summary", errors)
            self.assertIn("heldout_manifest.sources[0].scenario_ids must match the current suite summary", errors)
            self.assertIn("heldout_manifest.sources[0].scenario_fingerprints must match the current suite summary", errors)

    def test_validate_malformed_referenced_suite_returns_errors_instead_of_raising(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            suite = _suite_summary(root / "suite_summary.json", ["email_reply_completion"])
            out = root / "heldout_manifest.json"
            run_cli(["heldout-manifest", "--suite-summary", f"source={suite}", "--out", str(out)])
            suite_payload = _read_json(suite)
            suite_payload["error_count"] = "not-an-int"
            suite.write_text(json.dumps(suite_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            validation = validate_artifacts(heldout_manifest_paths=[out], strict=True)

            self.assertFalse(validation["passed"], validation)
            errors = "\n".join(error for target in validation["targets"] for error in target["errors"])
            self.assertIn("blocking_reasons must match the current suite summary", errors)

    def test_validate_rejects_heldout_manifest_with_unknown_control_plane_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline = _suite_summary(root / "baseline_suite.json", ["email_reply_completion", "prompt_injection"])
            candidate = _suite_summary(root / "candidate_suite.json", ["email_reply_completion"])
            out = root / "heldout_manifest.json"
            run_cli(
                [
                    "heldout-manifest",
                    "--suite-summary",
                    f"baseline={baseline}",
                    "--suite-summary",
                    f"candidate={candidate}",
                    "--out",
                    str(out),
                ]
            )
            manifest = _read_json(out)
            manifest["provider_console_url"] = "https://example.invalid/heldout"
            manifest["governance_handoff"]["approval_thread_ref"] = "redacted-thread"
            manifest["sources"][0]["provider_job_id"] = "job-redacted"
            manifest["mismatches"][0]["benchmark_url"] = "https://example.invalid/mismatch"
            out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            schema_result = check_schema_file(out)
            validation = validate_artifacts(heldout_manifest_paths=[out], strict=True)

            self.assertFalse(schema_result["passed"], schema_result)
            self.assertFalse(validation["passed"], validation)
            errors = "\n".join(error for target in validation["targets"] for error in target["errors"])
            self.assertIn("heldout_manifest contains unknown field(s): ['provider_console_url']", errors)
            self.assertIn(
                "heldout_manifest.governance_handoff contains unknown field(s): ['approval_thread_ref']",
                errors,
            )
            self.assertIn("heldout_manifest.sources[0] contains unknown field(s): ['provider_job_id']", errors)
            self.assertIn("heldout_manifest.mismatches[0] contains unknown field(s): ['benchmark_url']", errors)


def _suite_summary(path: Path, scenario_ids: list[str], scenario_overrides=None) -> Path:
    overrides = list(scenario_overrides or [])
    run_root = f"{path.stem}_runs"
    runs = [
        {
            "scenario_id": scenario_id,
            "scenario_title": scenario_id,
            "task_family": scenario_id,
            "scenario_path": f"scenarios/{scenario_id}.json",
            "trace_path": f"traces/{scenario_id}.jsonl",
            "run_dir": f"{run_root}/{scenario_id}",
            "report": f"{run_root}/{scenario_id}/report.html",
            "report_sha256": "b" * 64,
            "report_size_bytes": 1,
            "scorecard": f"{run_root}/{scenario_id}/scorecard.json",
            "scorecard_sha256": "c" * 64,
            "scorecard_size_bytes": 1,
            "run_digest": f"{run_root}/{scenario_id}/run_digest.json",
            "run_digest_sha256": "d" * 64,
            "run_digest_size_bytes": 1,
            "lineage": f"{run_root}/{scenario_id}/artifact_lineage.json",
            "lineage_sha256": "e" * 64,
            "lineage_size_bytes": 1,
            "passed": True,
            "score": 100,
            "failed_rules": [],
            "critical_failures": [],
        }
        for scenario_id in scenario_ids
    ]
    for index, run in enumerate(runs):
        scenario_payload = {
            "id": run["scenario_id"],
            "policy": {},
            "prompt": f"Complete the {run['scenario_id']} held-out task.",
            "title": run["scenario_id"],
        }
        if index < len(overrides):
            scenario_payload.update(overrides[index])
        scenario_bytes = (
            json.dumps(scenario_payload, indent=2, sort_keys=True) + "\n"
        ).encode("utf-8")
        scenario_path = path.parent / run["scenario_path"]
        scenario_path.parent.mkdir(parents=True, exist_ok=True)
        scenario_path.write_bytes(scenario_bytes)
        run["scenario_sha256"] = hashlib.sha256(scenario_bytes).hexdigest()
    payload = {
        "schema_version": "hfr.run_suite.v1",
        "scenarios_dir": "scenarios",
        "out_dir": run_root,
        "total": len(runs),
        "passed": len(runs),
        "failed": 0,
        "error_count": 0,
        "errors": [],
        "metrics": {
            "pass_rate": 1.0 if runs else 0.0,
            "average_score": 100.0 if runs else 0.0,
            "min_score": 100 if runs else None,
            "max_score": 100 if runs else None,
            "failed_rule_counts": [],
            "critical_failure_counts": [],
            "task_families": [],
            "failed": 0,
            "passed": len(runs),
        },
        "runs": runs,
        "artifacts": {"suite_result": f"{run_root}/harness_suite_result.json"},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
