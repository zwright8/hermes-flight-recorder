import hashlib
import json
import os
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from flightrecorder.cli import main
from flightrecorder.decision_gate import (
    DECISION_GATE_SOURCE_SCHEMAS,
    decision_gate_source_contract_errors,
    evaluate_decision_gate,
)
from flightrecorder.schema_registry import check_schema_file, load_schema


def run_cli(args):
    with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
        return main(args)


def _gate_decision_args(source, output):
    return [
        "gate-decision",
        "--artifact",
        str(source),
        "--expect-recommendation",
        "promote_iteration",
        "--expect-readiness",
        "ready",
        "--require-passed",
        "--out",
        str(output),
    ]


def _write_source_decision(path, *, passed=True, recommendation="promote_iteration", readiness="ready"):
    blocking_check_count = 0 if passed else 1
    recurring_action_count = 0 if passed else 1
    metrics = {
        "bundle_count": 0 if passed else 1,
        "unique_action_count": recurring_action_count,
        "open_action_count": recurring_action_count,
        "new_action_count": 0,
        "recurring_action_count": recurring_action_count,
        "resolved_action_count": 0,
        "open_priority_counts": [],
    }
    checks = []
    blocking_checks = []
    if not passed:
        check = {
            "id": "max_recurring_actions",
            "passed": False,
            "actual": recurring_action_count,
            "expected": {"maximum": 0},
            "summary": "recurring actions exceed the configured maximum",
        }
        checks.append(check)
        blocking_checks.append({"id": check["id"], "summary": check["summary"]})
    path.write_text(
        json.dumps(
            {
                "schema_version": "hfr.action_ledger_gate.v1",
                "action_ledger": "action_ledger.json",
                "passed": passed,
                "check_count": len(checks),
                "failed_check_count": len(checks),
                "checks": checks,
                "metrics": metrics,
                "decision": {
                    "readiness": readiness,
                    "recommendation": recommendation,
                    "summary": "ok" if passed else "blocked",
                    "blocking_check_count": blocking_check_count,
                    "blocking_checks": blocking_checks,
                    "key_metrics": metrics,
                },
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _write_historical_forged_decision_gate(source_path, gate_path):
    source = json.loads(source_path.read_text(encoding="utf-8"))
    decision = source["decision"]
    source_bytes = source_path.read_bytes()
    checks = [
        {
            "id": "recommendation_matches",
            "passed": True,
            "actual": {"recommendation": decision["recommendation"]},
            "expected": {"recommendation": "promote_iteration"},
            "summary": "recommendation matches",
        },
        {
            "id": "readiness_matches",
            "passed": True,
            "actual": {"readiness": decision["readiness"]},
            "expected": {"readiness": "ready"},
            "summary": "readiness matches",
        },
        {
            "id": "source_artifact_passed",
            "passed": True,
            "actual": {"passed": True},
            "expected": {"passed": True},
            "summary": "source artifact passed",
        },
    ]
    gate_path.write_text(
        json.dumps(
            {
                "schema_version": "hfr.decision_gate.v1",
                "artifact": source_path.name,
                "source_artifact": {
                    "path": source_path.name,
                    "kind": "file",
                    "exists": True,
                    "size_bytes": len(source_bytes),
                    "sha256": hashlib.sha256(source_bytes).hexdigest(),
                },
                "passed": True,
                "readiness": "ready",
                "recommendation": "allow_promotion",
                "expected_recommendation": "promote_iteration",
                "expected_readiness": "ready",
                "require_passed": True,
                "check_count": len(checks),
                "failed_check_count": 0,
                "checks": checks,
                "source_decision": {
                    "schema_version": str(source.get("schema_version") or ""),
                    "passed": True,
                    "recommendation": decision["recommendation"],
                    "readiness": decision["readiness"],
                    "summary": decision["summary"],
                    "blocking_check_count": decision["blocking_check_count"],
                    "key_metrics": decision["key_metrics"],
                },
                "notes": ["Historical gate created before source schema contracts were enforced."],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


class DecisionGateTests(unittest.TestCase):
    def test_source_allowlist_matches_public_decision_gate_schema(self):
        schema = load_schema("decision_gate")
        public_versions = schema["$defs"]["source_decision"]["properties"]["schema_version"]["enum"]

        self.assertEqual(set(public_versions), set(DECISION_GATE_SOURCE_SCHEMAS))
        for schema_version, schema_name in DECISION_GATE_SOURCE_SCHEMAS.items():
            with self.subTest(schema_version=schema_version, schema_name=schema_name):
                source_schema = load_schema(schema_name)
                self.assertEqual(source_schema["properties"]["schema_version"]["const"], schema_version)

    def test_source_contract_bounds_unknown_schema_diagnostics(self):
        errors = decision_gate_source_contract_errors({"schema_version": "x" * 10_000})

        self.assertEqual(len(errors), 1)
        self.assertLess(len(errors[0]), 1_024)

    def test_gate_decision_allows_expected_recommendation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "action_ledger_gate.json"
            decision_gate = root / "decision_gate.json"
            summary_path = root / "validation.json"
            strict_summary_path = root / "strict_validation.json"
            _write_source_decision(source)
            source_schema = check_schema_file(source)
            self.assertTrue(source_schema["passed"], source_schema["errors"])

            code = run_cli(
                [
                    "gate-decision",
                    "--artifact",
                    str(source),
                    "--expect-recommendation",
                    "promote_iteration",
                    "--expect-readiness",
                    "ready",
                    "--require-passed",
                    "--preserve-paths",
                    "--out",
                    str(decision_gate),
                ]
            )

            self.assertEqual(code, 0)
            schema_result = check_schema_file(decision_gate, "decision_gate")
            self.assertTrue(schema_result["passed"], schema_result["errors"])
            gate = json.loads(decision_gate.read_text(encoding="utf-8"))
            self.assertEqual(gate["schema_version"], "hfr.decision_gate.v1")
            self.assertTrue(gate["passed"])
            self.assertEqual(gate["recommendation"], "allow_promotion")
            self.assertEqual(gate["artifact"], str(source))
            self.assertEqual(gate["source_artifact"]["path"], str(source))
            self.assertTrue(gate["source_artifact"]["exists"])
            self.assertEqual(len(gate["source_artifact"]["sha256"]), 64)
            self.assertEqual(gate["source_decision"]["recommendation"], "promote_iteration")
            self.assertEqual(gate["source_decision"]["key_metrics"]["recurring_action_count"], 0)
            self.assertEqual(run_cli(["validate", "--decision-gate", str(decision_gate), "--out", str(summary_path)]), 0)
            self.assertEqual(run_cli(["validate", "--decision-gate", str(decision_gate), "--strict", "--out", str(strict_summary_path)]), 1)
            summary = json.loads(strict_summary_path.read_text(encoding="utf-8"))
            warnings = "\n".join(warning for target in summary["targets"] for warning in target["warnings"])
            self.assertIn("decision_gate.artifact is absolute", warnings)
            self.assertIn("decision_gate.source_artifact.path is absolute", warnings)

            tampered = json.loads(json.dumps(gate))
            tampered["source_decision"]["recommendation"] = "block_iteration"
            decision_gate.write_text(json.dumps(tampered, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            self.assertEqual(run_cli(["validate", "--decision-gate", str(decision_gate)]), 1)

            decision_gate.write_text(json.dumps(gate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            self.assertEqual(run_cli(["validate", "--decision-gate", str(decision_gate)]), 0)

            source.write_text(source.read_text(encoding="utf-8").replace("promote_iteration", "block_iteration"), encoding="utf-8")
            self.assertEqual(run_cli(["validate", "--decision-gate", str(decision_gate)]), 1)

    def test_gate_decision_blocks_unexpected_recommendation_but_validates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "action_ledger_gate.json"
            decision_gate = root / "decision_gate.json"
            _write_source_decision(source, passed=False, recommendation="block_iteration", readiness="blocked")
            source_schema = check_schema_file(source)
            self.assertTrue(source_schema["passed"], source_schema["errors"])

            code = run_cli(
                [
                    "gate-decision",
                    "--artifact",
                    str(source),
                    "--expect-recommendation",
                    "promote_iteration",
                    "--expect-readiness",
                    "ready",
                    "--require-passed",
                    "--out",
                    str(decision_gate),
                ]
            )

            self.assertEqual(code, 1)
            gate = json.loads(decision_gate.read_text(encoding="utf-8"))
            self.assertFalse(gate["passed"])
            self.assertEqual(gate["recommendation"], "block_promotion")
            self.assertEqual(len(gate["source_artifact"]["sha256"]), 64)
            self.assertEqual(gate["source_decision"]["recommendation"], "block_iteration")
            self.assertEqual(run_cli(["validate", "--decision-gate", str(decision_gate), "--strict"]), 0)

            gate["failed_check_count"] = 0
            decision_gate.write_text(json.dumps(gate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            self.assertEqual(run_cli(["validate", "--decision-gate", str(decision_gate)]), 1)

    def test_gate_decision_rejects_uncontracted_source_artifacts(self):
        decision = {
            "readiness": "ready",
            "recommendation": "promote_iteration",
            "summary": "forged promotion recommendation",
            "blocking_check_count": 0,
            "key_metrics": {},
        }
        cases = {
            "missing_schema_version": {"passed": True, "decision": decision},
            "unknown_schema_version": {
                "schema_version": "hfr.attacker_gate.v1",
                "passed": True,
                "decision": decision,
            },
            "registered_but_unsupported_schema": {
                "schema_version": "hfr.state_diff.v1",
                "passed": True,
                "decision": decision,
            },
            "schema_invalid_supported_source": {
                "schema_version": "hfr.action_ledger_gate.v1",
                "passed": True,
                "decision": decision,
            },
        }

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for case_name, payload in cases.items():
                with self.subTest(case=case_name):
                    source = root / f"{case_name}.json"
                    decision_gate = root / f"{case_name}_decision_gate.json"
                    source.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")

                    with self.assertRaises(SystemExit) as raised:
                        run_cli(
                            [
                                "gate-decision",
                                "--artifact",
                                str(source),
                                "--expect-recommendation",
                                "promote_iteration",
                                "--expect-readiness",
                                "ready",
                                "--require-passed",
                                "--out",
                                str(decision_gate),
                            ]
                        )

                    self.assertEqual(raised.exception.code, 2)
                    self.assertFalse(decision_gate.exists())

    def test_validate_rejects_historical_gate_over_uncontracted_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "arbitrary.json"
            decision_gate = root / "decision_gate.json"
            summary_path = root / "validation.json"
            source.write_text(
                json.dumps(
                    {
                        "schema_version": "hfr.action_ledger_gate.v1",
                        "passed": True,
                        "decision": {
                            "readiness": "ready",
                            "recommendation": "promote_iteration",
                            "summary": "forged promotion recommendation",
                            "blocking_check_count": 0,
                            "key_metrics": {},
                        },
                    },
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            _write_historical_forged_decision_gate(source, decision_gate)
            outer_schema = check_schema_file(decision_gate, "decision_gate")
            self.assertTrue(outer_schema["passed"], outer_schema["errors"])

            code = run_cli(["validate", "--decision-gate", str(decision_gate), "--out", str(summary_path)])

            self.assertEqual(code, 1)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            errors = [error for target in summary["targets"] for error in target["errors"]]
            self.assertTrue(any("source_artifact contract error" in error for error in errors), errors)

    def test_gate_decision_writes_output_relative_source_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runs = root / "runs"
            runs.mkdir()
            source = runs / "action_ledger_gate.json"
            decision_gate = runs / "decision_gate.json"
            summary_path = runs / "validation.json"
            _write_source_decision(source)

            previous_cwd = Path.cwd()
            try:
                os.chdir(root)
                code = run_cli(
                    [
                        "gate-decision",
                        "--artifact",
                        str(source.relative_to(root)),
                        "--expect-recommendation",
                        "promote_iteration",
                        "--expect-readiness",
                        "ready",
                        "--require-passed",
                        "--out",
                        str(decision_gate),
                    ]
                )
                self.assertEqual(code, 0)
                gate = json.loads(decision_gate.read_text(encoding="utf-8"))
                self.assertEqual(gate["artifact"], "action_ledger_gate.json")
                self.assertEqual(gate["source_artifact"]["path"], "action_ledger_gate.json")
                code = run_cli(["validate", "--decision-gate", str(decision_gate), "--strict", "--out", str(summary_path)])
            finally:
                os.chdir(previous_cwd)

            self.assertEqual(code, 0)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            errors = "\n".join(error for target in summary["targets"] for error in target["errors"])
            self.assertEqual(errors, "")

    def test_gate_decision_rejects_symlinked_artifact_parent(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_dir = root / "source"
            source_dir.mkdir()
            source = source_dir / "action_ledger_gate.json"
            linked_parent = root / "linked_source"
            decision_gate = root / "decision_gate.json"
            _write_source_decision(source)
            try:
                linked_parent.symlink_to(source_dir, target_is_directory=True)
            except (NotImplementedError, OSError) as exc:
                self.skipTest(f"symlink unavailable: {exc}")

            with self.assertRaises(SystemExit) as raised:
                run_cli(
                    [
                        "gate-decision",
                        "--artifact",
                        str(linked_parent / source.name),
                        "--expect-recommendation",
                        "promote_iteration",
                        "--expect-readiness",
                        "ready",
                        "--require-passed",
                        "--out",
                        str(decision_gate),
                    ]
                )

            self.assertEqual(raised.exception.code, 2)
            self.assertFalse(decision_gate.exists())

    def test_gate_decision_rejects_source_as_output_without_modifying_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "action_ledger_gate.json"
            _write_source_decision(source)
            source_before = source.read_bytes()

            with self.assertRaises(SystemExit) as raised:
                run_cli(_gate_decision_args(source, source))

            self.assertEqual(raised.exception.code, 2)
            self.assertEqual(source.read_bytes(), source_before)

    def test_gate_decision_rejects_hardlink_output_without_modifying_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "action_ledger_gate.json"
            output = root / "decision_gate.json"
            _write_source_decision(source)
            try:
                os.link(source, output)
            except (NotImplementedError, OSError) as exc:
                self.skipTest(f"hardlinks unavailable: {exc}")
            source_before = source.read_bytes()

            with self.assertRaises(SystemExit) as raised:
                run_cli(_gate_decision_args(source, output))

            self.assertEqual(raised.exception.code, 2)
            self.assertEqual(source.read_bytes(), source_before)
            self.assertEqual(output.read_bytes(), source_before)
            self.assertEqual(source.stat().st_ino, output.stat().st_ino)

    def test_gate_decision_rejects_leaf_output_symlink_without_modifying_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "action_ledger_gate.json"
            target = root / "existing.json"
            output = root / "decision_gate.json"
            _write_source_decision(source)
            target.write_bytes(b'{"owner":"competing-writer"}\n')
            try:
                output.symlink_to(target)
            except (NotImplementedError, OSError) as exc:
                self.skipTest(f"symlinks unavailable: {exc}")
            source_before = source.read_bytes()
            target_before = target.read_bytes()

            with self.assertRaises(SystemExit) as raised:
                run_cli(_gate_decision_args(source, output))

            self.assertEqual(raised.exception.code, 2)
            self.assertEqual(source.read_bytes(), source_before)
            self.assertEqual(target.read_bytes(), target_before)
            self.assertTrue(output.is_symlink())

    def test_gate_decision_rejects_symlinked_output_parent_without_modifying_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "action_ledger_gate.json"
            real_parent = root / "real-output"
            linked_parent = root / "linked-output"
            real_parent.mkdir()
            target = real_parent / "decision_gate.json"
            output = linked_parent / target.name
            _write_source_decision(source)
            target.write_bytes(b'{"owner":"competing-writer"}\n')
            try:
                linked_parent.symlink_to(real_parent, target_is_directory=True)
            except (NotImplementedError, OSError) as exc:
                self.skipTest(f"symlinks unavailable: {exc}")
            source_before = source.read_bytes()
            target_before = target.read_bytes()

            with self.assertRaises(SystemExit) as raised:
                run_cli(_gate_decision_args(source, output))

            self.assertEqual(raised.exception.code, 2)
            self.assertEqual(source.read_bytes(), source_before)
            self.assertEqual(target.read_bytes(), target_before)
            self.assertTrue(linked_parent.is_symlink())

    def test_gate_decision_rejects_output_changed_after_initial_digest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "action_ledger_gate.json"
            output = root / "decision_gate.json"
            _write_source_decision(source)
            output.write_bytes(b'{"generation":"initial"}\n')
            competing_bytes = b'{"generation":"competing"}\n'

            def mutate_output_then_evaluate(*args, **kwargs):
                output.write_bytes(competing_bytes)
                return evaluate_decision_gate(*args, **kwargs)

            with patch(
                "flightrecorder.cli.evaluate_decision_gate",
                side_effect=mutate_output_then_evaluate,
            ):
                with self.assertRaises(SystemExit) as raised:
                    run_cli(_gate_decision_args(source, output))

            self.assertEqual(raised.exception.code, 2)
            self.assertEqual(output.read_bytes(), competing_bytes)

    def test_validate_rejects_decision_gate_missing_source_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "action_ledger_gate.json"
            decision_gate = root / "decision_gate.json"
            summary_path = root / "validation.json"
            _write_source_decision(source)
            code = run_cli(
                [
                    "gate-decision",
                    "--artifact",
                    str(source),
                    "--expect-recommendation",
                    "promote_iteration",
                    "--expect-readiness",
                    "ready",
                    "--require-passed",
                    "--out",
                    str(decision_gate),
                ]
            )
            self.assertEqual(code, 0)
            source.unlink()

            code = run_cli(["validate", "--decision-gate", str(decision_gate), "--strict", "--out", str(summary_path)])

            self.assertEqual(code, 1)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            errors = "\n".join(error for target in summary["targets"] for error in target["errors"])
            self.assertIn("decision_gate.source_artifact.path does not resolve to an existing file", errors)

    def test_validate_rejects_decision_gate_with_non_local_source_reference(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "action_ledger_gate.json"
            decision_gate = root / "decision_gate.json"
            summary_path = root / "validation.json"
            _write_source_decision(source)
            self.assertEqual(
                run_cli(
                    [
                        "gate-decision",
                        "--artifact",
                        str(source),
                        "--expect-recommendation",
                        "promote_iteration",
                        "--expect-readiness",
                        "ready",
                        "--require-passed",
                        "--out",
                        str(decision_gate),
                    ]
                ),
                0,
            )
            gate = json.loads(decision_gate.read_text(encoding="utf-8"))
            gate["artifact"] = "https://example.test/action_ledger_gate.json"
            gate["source_artifact"]["path"] = gate["artifact"]
            decision_gate.write_text(json.dumps(gate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            outer_schema = check_schema_file(decision_gate, "decision_gate")
            self.assertTrue(outer_schema["passed"], outer_schema["errors"])

            code = run_cli(
                [
                    "validate",
                    "--decision-gate",
                    str(decision_gate),
                    "--strict",
                    "--out",
                    str(summary_path),
                ]
            )

            self.assertEqual(code, 1)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            errors = "\n".join(error for target in summary["targets"] for error in target["errors"])
            self.assertIn("decision_gate.source_artifact.path must resolve to a local file", errors)

    def test_validate_rejects_decision_gate_cwd_relative_source_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cwd_root = root / "cwd"
            outside_root = root / "outside"
            nested = cwd_root / "nested"
            nested.mkdir(parents=True)
            outside_root.mkdir()
            source = nested / "action_ledger_gate.json"
            decision_gate = cwd_root / "decision_gate.json"
            outside_gate = outside_root / "decision_gate.json"
            summary_path = root / "validation.json"
            _write_source_decision(source)

            previous_cwd = Path.cwd()
            try:
                os.chdir(cwd_root)
                code = run_cli(
                    [
                        "gate-decision",
                        "--artifact",
                        "nested/action_ledger_gate.json",
                        "--expect-recommendation",
                        "promote_iteration",
                        "--expect-readiness",
                        "ready",
                        "--require-passed",
                        "--out",
                        str(decision_gate),
                    ]
                )
                self.assertEqual(code, 0)
                outside_gate.write_text(decision_gate.read_text(encoding="utf-8"), encoding="utf-8")
                code = run_cli(["validate", "--decision-gate", str(outside_gate), "--strict", "--out", str(summary_path)])
            finally:
                os.chdir(previous_cwd)

            self.assertEqual(code, 1)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            errors = "\n".join(error for target in summary["targets"] for error in target["errors"])
            self.assertIn("decision_gate.source_artifact.path does not resolve to an existing file", errors)

    def test_validate_rejects_decision_gate_regular_file_false_directory_bypass(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "action_ledger_gate.json"
            source_dir = root / "source_dir"
            decision_gate = root / "decision_gate.json"
            summary_path = root / "validation.json"
            _write_source_decision(source)
            source_dir.mkdir()
            code = run_cli(
                [
                    "gate-decision",
                    "--artifact",
                    str(source),
                    "--expect-recommendation",
                    "promote_iteration",
                    "--expect-readiness",
                    "ready",
                    "--require-passed",
                    "--out",
                    str(decision_gate),
                ]
            )
            self.assertEqual(code, 0)
            gate = json.loads(decision_gate.read_text(encoding="utf-8"))
            gate["artifact"] = "source_dir"
            gate["source_artifact"]["path"] = "source_dir"
            gate["source_artifact"]["regular_file"] = False
            decision_gate.write_text(json.dumps(gate, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            code = run_cli(["validate", "--decision-gate", str(decision_gate), "--strict", "--out", str(summary_path)])

            self.assertEqual(code, 1)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            errors = "\n".join(error for target in summary["targets"] for error in target["errors"])
            self.assertIn("decision_gate.source_artifact.regular_file must be true when present", errors)
            self.assertIn("decision_gate.source_artifact.path does not resolve to an existing file", errors)

    def test_validate_rejects_decision_gate_regular_file_false_symlink_bypass(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "action_ledger_gate.json"
            source_link = root / "source_link.json"
            decision_gate = root / "decision_gate.json"
            summary_path = root / "validation.json"
            _write_source_decision(source)
            try:
                source_link.symlink_to(source)
            except (NotImplementedError, OSError) as exc:
                self.skipTest(f"symlink unavailable: {exc}")
            code = run_cli(
                [
                    "gate-decision",
                    "--artifact",
                    str(source),
                    "--expect-recommendation",
                    "promote_iteration",
                    "--expect-readiness",
                    "ready",
                    "--require-passed",
                    "--out",
                    str(decision_gate),
                ]
            )
            self.assertEqual(code, 0)
            gate = json.loads(decision_gate.read_text(encoding="utf-8"))
            gate["artifact"] = "source_link.json"
            gate["source_artifact"]["path"] = "source_link.json"
            gate["source_artifact"]["regular_file"] = False
            decision_gate.write_text(json.dumps(gate, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            code = run_cli(["validate", "--decision-gate", str(decision_gate), "--strict", "--out", str(summary_path)])

            self.assertEqual(code, 1)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            errors = "\n".join(error for target in summary["targets"] for error in target["errors"])
            self.assertIn("decision_gate.source_artifact.regular_file must be true when present", errors)
            self.assertIn("decision_gate.source_artifact.path must not resolve to a symlink", errors)

    def test_validate_rejects_decision_gate_parent_symlink_source_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_dir = root / "source"
            source_dir.mkdir()
            source = source_dir / "action_ledger_gate.json"
            linked_target = root / "linked_target"
            linked_target.mkdir()
            linked_source = linked_target / source.name
            linked_parent = root / "linked_source"
            decision_gate = root / "decision_gate.json"
            summary_path = root / "validation.json"
            _write_source_decision(source)
            _write_source_decision(linked_source, passed=False, recommendation="block_iteration", readiness="blocked")
            try:
                linked_parent.symlink_to(linked_target, target_is_directory=True)
            except (NotImplementedError, OSError) as exc:
                self.skipTest(f"symlink unavailable: {exc}")
            code = run_cli(
                [
                    "gate-decision",
                    "--artifact",
                    str(source),
                    "--expect-recommendation",
                    "promote_iteration",
                    "--expect-readiness",
                    "ready",
                    "--require-passed",
                    "--out",
                    str(decision_gate),
                ]
            )
            self.assertEqual(code, 0)
            gate = json.loads(decision_gate.read_text(encoding="utf-8"))
            gate["artifact"] = str(Path(linked_parent.name) / source.name)
            gate["source_artifact"]["path"] = str(Path(linked_parent.name) / source.name)
            decision_gate.write_text(json.dumps(gate, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            code = run_cli(["validate", "--decision-gate", str(decision_gate), "--strict", "--out", str(summary_path)])

            self.assertEqual(code, 1)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            errors = "\n".join(error for target in summary["targets"] for error in target["errors"])
            self.assertIn("decision_gate.source_artifact.path must not traverse symlinked components", errors)
            self.assertNotIn("decision_gate.source_decision.", errors)


if __name__ == "__main__":
    unittest.main()
