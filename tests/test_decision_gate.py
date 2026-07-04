import json
import os
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path

from flightrecorder.cli import main
from flightrecorder.schema_registry import check_schema_file


def run_cli(args):
    with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
        return main(args)


def _write_source_decision(path, *, passed=True, recommendation="promote_iteration", readiness="ready"):
    blocking_check_count = 0 if passed else 1
    path.write_text(
        json.dumps(
            {
                "schema_version": "hfr.action_ledger_gate.v1",
                "passed": passed,
                "decision": {
                    "readiness": readiness,
                    "recommendation": recommendation,
                    "summary": "ok" if passed else "blocked",
                    "blocking_check_count": blocking_check_count,
                    "key_metrics": {"recurring_action_count": blocking_check_count},
                },
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


class DecisionGateTests(unittest.TestCase):
    def test_gate_decision_allows_expected_recommendation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "action_ledger_gate.json"
            decision_gate = root / "decision_gate.json"
            summary_path = root / "validation.json"
            strict_summary_path = root / "strict_validation.json"
            source.write_text(
                json.dumps(
                    {
                        "schema_version": "hfr.action_ledger_gate.v1",
                        "passed": True,
                        "decision": {
                            "readiness": "ready",
                            "recommendation": "promote_iteration",
                            "summary": "ok",
                            "blocking_check_count": 0,
                            "key_metrics": {"recurring_action_count": 0},
                        },
                    },
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )

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
            source.write_text(
                json.dumps(
                    {
                        "schema_version": "hfr.action_ledger_gate.v1",
                        "passed": False,
                        "decision": {
                            "readiness": "blocked",
                            "recommendation": "block_iteration",
                            "summary": "blocked",
                            "blocking_check_count": 1,
                            "key_metrics": {"recurring_action_count": 3},
                        },
                    },
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )

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
