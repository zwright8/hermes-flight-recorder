import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path

from flightrecorder.cli import main

ROOT = Path(__file__).resolve().parents[1]


def run_cli(args):
    with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
        return main(args)


class PromotionLedgerTests(unittest.TestCase):
    def test_promotion_ledger_tracks_allow_and_block_decisions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            allow_source = root / "allow_action_ledger_gate.json"
            block_source = root / "block_action_ledger_gate.json"
            allow_gate = root / "allow_decision_gate.json"
            block_gate = root / "block_decision_gate.json"
            ledger_path = root / "promotion_ledger.json"

            allow_source.write_text(
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
            block_source.write_text(
                json.dumps(
                    {
                        "schema_version": "hfr.action_ledger_gate.v1",
                        "passed": False,
                        "decision": {
                            "readiness": "blocked",
                            "recommendation": "block_iteration",
                            "summary": "repair pressure remains",
                            "blocking_check_count": 1,
                            "key_metrics": {"recurring_action_count": 2},
                        },
                    },
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )

            self.assertEqual(
                run_cli(
                    [
                        "gate-decision",
                        "--artifact",
                        str(allow_source),
                        "--expect-recommendation",
                        "promote_iteration",
                        "--expect-readiness",
                        "ready",
                        "--require-passed",
                        "--preserve-paths",
                        "--out",
                        str(allow_gate),
                    ]
                ),
                0,
            )
            self.assertEqual(
                run_cli(
                    [
                        "gate-decision",
                        "--artifact",
                        str(block_source),
                        "--expect-recommendation",
                        "promote_iteration",
                        "--expect-readiness",
                        "ready",
                        "--require-passed",
                        "--preserve-paths",
                        "--out",
                        str(block_gate),
                    ]
                ),
                1,
            )

            code = run_cli(
                [
                    "promotion-ledger",
                    "--decision-gate",
                    str(allow_gate),
                    "--decision-gate",
                    str(block_gate),
                    "--preserve-paths",
                    "--out",
                    str(ledger_path),
                ]
            )

            self.assertEqual(code, 0)
            ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
            self.assertEqual(ledger["schema_version"], "hfr.promotion_ledger.v1")
            self.assertTrue(ledger["passed"])
            self.assertEqual(ledger["decision_count"], 2)
            self.assertEqual(ledger["metrics"]["decision_count"], 2)
            self.assertEqual(ledger["metrics"]["allowed_count"], 1)
            self.assertEqual(ledger["metrics"]["blocked_count"], 1)
            self.assertEqual(ledger["metrics"]["latest_recommendation"], "block_promotion")
            self.assertEqual(ledger["metrics"]["latest_readiness"], "blocked")
            self.assertFalse(ledger["metrics"]["latest_passed"])
            self.assertEqual(ledger["metrics"]["consecutive_allowed_count"], 0)
            self.assertEqual(ledger["metrics"]["consecutive_blocked_count"], 1)
            self.assertEqual(ledger["metrics"]["unique_source_artifact_count"], 2)
            self.assertEqual(
                ledger["metrics"]["recommendation_counts"],
                [{"count": 1, "id": "allow_promotion"}, {"count": 1, "id": "block_promotion"}],
            )
            self.assertEqual(
                ledger["metrics"]["source_recommendation_counts"],
                [{"count": 1, "id": "block_iteration"}, {"count": 1, "id": "promote_iteration"}],
            )
            self.assertEqual(ledger["records"][0]["source"]["recommendation"], "promote_iteration")
            self.assertEqual(ledger["records"][1]["source"]["recommendation"], "block_iteration")
            self.assertEqual(len(ledger["records"][0]["sha256"]), 64)
            self.assertEqual(len(ledger["records"][0]["source"]["artifact_sha256"]), 64)
            self.assertEqual(run_cli(["validate", "--promotion-ledger", str(ledger_path), "--strict"]), 0)
            self.assertEqual(run_cli(["schemas", "--check", str(ledger_path)]), 0)

            ledger["metrics"]["allowed_count"] = 99
            ledger_path.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            self.assertEqual(run_cli(["validate", "--promotion-ledger", str(ledger_path)]), 1)

    def test_gate_promotion_ledger_allows_clean_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "action_ledger_gate.json"
            decision_gate = root / "decision_gate.json"
            ledger_path = root / "promotion_ledger.json"
            gate_path = root / "promotion_ledger_gate.json"
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
                        "--preserve-paths",
                        "--out",
                        str(decision_gate),
                    ]
                ),
                0,
            )
            run_cli(
                [
                    "promotion-ledger",
                    "--decision-gate",
                    str(decision_gate),
                    "--preserve-paths",
                    "--out",
                    str(ledger_path),
                ]
            )

            code = run_cli(
                [
                    "gate-promotion-ledger",
                    "--promotion-ledger",
                    str(ledger_path),
                    "--policy",
                    str(ROOT / "examples" / "promotion_ledger_gate_policy.demo.json"),
                    "--out",
                    str(gate_path),
                ]
            )

            self.assertEqual(code, 0)
            gate = json.loads(gate_path.read_text(encoding="utf-8"))
            self.assertEqual(gate["schema_version"], "hfr.promotion_ledger_gate.v1")
            self.assertTrue(gate["passed"])
            self.assertEqual(gate["decision"]["recommendation"], "promote_iteration")
            self.assertEqual(gate["decision"]["readiness"], "ready")
            self.assertEqual(gate["metrics"]["blocked_rate"], 0.0)
            self.assertEqual(gate["metrics"]["failed_decision_count"], 0)
            self.assertEqual(gate["policy"]["schema_version"], "hfr.promotion_ledger_gate.policy.v1")
            self.assertEqual(gate["policy"]["effective"]["require_latest_recommendation"], "allow_promotion")
            self.assertTrue(gate["policy"]["effective"]["require_latest_passed"])
            self.assertEqual(run_cli(["validate", "--promotion-ledger-gate", str(gate_path), "--strict"]), 0)
            self.assertEqual(run_cli(["schemas", "--check", str(gate_path)]), 0)

            gate["metrics"]["blocked_rate"] = 1.0
            gate_path.write_text(json.dumps(gate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            self.assertEqual(run_cli(["validate", "--promotion-ledger-gate", str(gate_path)]), 1)

    def test_gate_promotion_ledger_blocks_bad_latest_decision(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "blocked_action_ledger_gate.json"
            decision_gate = root / "decision_gate.json"
            ledger_path = root / "promotion_ledger.json"
            gate_path = root / "promotion_ledger_gate.json"
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
                            "key_metrics": {"recurring_action_count": 5},
                        },
                    },
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
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
                        "--preserve-paths",
                        "--out",
                        str(decision_gate),
                    ]
                ),
                1,
            )
            run_cli(
                [
                    "promotion-ledger",
                    "--decision-gate",
                    str(decision_gate),
                    "--preserve-paths",
                    "--out",
                    str(ledger_path),
                ]
            )

            code = run_cli(
                [
                    "gate-promotion-ledger",
                    "--promotion-ledger",
                    str(ledger_path),
                    "--min-decisions",
                    "1",
                    "--require-latest-recommendation",
                    "allow_promotion",
                    "--require-latest-passed",
                    "--max-blocked-count",
                    "0",
                    "--max-consecutive-blocked",
                    "0",
                    "--max-failed-decisions",
                    "0",
                    "--forbid-source-recommendation",
                    "block_iteration",
                    "--out",
                    str(gate_path),
                ]
            )

            self.assertEqual(code, 1)
            gate = json.loads(gate_path.read_text(encoding="utf-8"))
            self.assertFalse(gate["passed"])
            self.assertEqual(gate["decision"]["recommendation"], "block_iteration")
            self.assertEqual(gate["decision"]["readiness"], "blocked")
            self.assertEqual(gate["metrics"]["blocked_rate"], 1.0)
            self.assertEqual(gate["metrics"]["failed_decision_count"], 1)
            failed_checks = {check["id"] for check in gate["checks"] if not check["passed"]}
            self.assertIn("require_latest_recommendation", failed_checks)
            self.assertIn("require_latest_passed", failed_checks)
            self.assertIn("max_blocked_count", failed_checks)
            self.assertIn("max_consecutive_blocked", failed_checks)
            self.assertIn("max_failed_decisions", failed_checks)
            self.assertIn("forbid_source_recommendation", failed_checks)
            self.assertEqual(run_cli(["validate", "--promotion-ledger-gate", str(gate_path), "--strict"]), 0)

    def test_promotion_archive_remains_valid_after_source_paths_are_removed(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            root = Path(tmp)
            source = root / "action_ledger_gate.json"
            decision_gate = root / "decision_gate.json"
            ledger_path = root / "promotion_ledger.json"
            gate_path = root / "promotion_ledger_gate.json"
            archive_dir = root / "promotion_archive"
            source_ref = str(source.relative_to(ROOT))
            decision_gate_ref = str(decision_gate.relative_to(ROOT))
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
            run_cli(
                [
                    "gate-decision",
                    "--artifact",
                    source_ref,
                    "--expect-recommendation",
                    "promote_iteration",
                    "--expect-readiness",
                    "ready",
                    "--require-passed",
                    "--out",
                    str(decision_gate),
                ]
            )
            run_cli(
                [
                    "promotion-ledger",
                    "--decision-gate",
                    decision_gate_ref,
                    "--out",
                    str(ledger_path),
                ]
            )
            run_cli(
                [
                    "gate-promotion-ledger",
                    "--promotion-ledger",
                    str(ledger_path),
                    "--policy",
                    str(ROOT / "examples" / "promotion_ledger_gate_policy.demo.json"),
                    "--out",
                    str(gate_path),
                ]
            )

            code = run_cli(
                [
                    "promotion-archive",
                    "--promotion-ledger",
                    str(ledger_path),
                    "--promotion-ledger-gate",
                    str(gate_path),
                    "--decision-gate",
                    str(decision_gate),
                    "--out",
                    str(archive_dir),
                    "--require-self-contained",
                ]
            )

            self.assertEqual(code, 0)
            manifest_path = archive_dir / "promotion_archive.json"
            archive = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(archive["schema_version"], "hfr.promotion_archive.v1")
            self.assertTrue(archive["passed"])
            self.assertTrue(archive["self_contained"])
            self.assertEqual(archive["metrics"]["missing_count"], 0)
            roles = {artifact["role"] for artifact in archive["artifacts"]}
            self.assertEqual(roles, {"promotion_ledger", "promotion_ledger_gate", "decision_gate", "source_artifact"})
            self.assertEqual(run_cli(["validate", "--promotion-archive", str(archive_dir), "--strict"]), 0)

            source.unlink()
            decision_gate.unlink()
            ledger_path.unlink()
            gate_path.unlink()
            self.assertEqual(run_cli(["validate", "--promotion-archive", str(archive_dir), "--strict"]), 0)

    def test_promotion_archive_includes_release_record_publication_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ledger_path = root / "promotion_ledger.json"
            release_record_path = root / "promotion_release_record.json"
            archive_dir = root / "promotion_archive"
            ledger_path.write_text(
                json.dumps({"schema_version": "hfr.promotion_ledger.v1", "records": []}) + "\n",
                encoding="utf-8",
            )
            release_record_path.write_text(
                json.dumps(
                    {
                        "schema_version": "hfr.promotion_release_record.v1",
                        "release": {"release_id": "release-2026-07-02"},
                    },
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )

            code = run_cli(
                [
                    "promotion-archive",
                    "--promotion-ledger",
                    str(ledger_path),
                    "--promotion-release-record",
                    str(release_record_path),
                    "--out",
                    str(archive_dir),
                    "--require-self-contained",
                ]
            )

            self.assertEqual(code, 0)
            archive = json.loads((archive_dir / "promotion_archive.json").read_text(encoding="utf-8"))
            release_artifacts = [artifact for artifact in archive["artifacts"] if artifact["role"] == "promotion_release_record"]
            self.assertEqual(len(release_artifacts), 1)
            self.assertEqual(archive["metrics"]["promotion_release_record_count"], 1)
            self.assertIn(
                {"from": "promotion_release_record_000", "to": "promotion_ledger", "type": "release_record"},
                archive["relationships"],
            )
            self.assertTrue((archive_dir / release_artifacts[0]["path"]).is_file())
            self.assertEqual(run_cli(["validate", "--promotion-archive", str(archive_dir), "--strict"]), 0)

            ledger_path.unlink()
            release_record_path.unlink()
            self.assertEqual(run_cli(["validate", "--promotion-archive", str(archive_dir), "--strict"]), 0)

    def test_promotion_archive_requires_self_contained_sources_when_requested(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "action_ledger_gate.json"
            decision_gate = root / "decision_gate.json"
            ledger_path = root / "promotion_ledger.json"
            archive_dir = root / "promotion_archive"
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
                            "key_metrics": {},
                        },
                    },
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
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
            run_cli(["promotion-ledger", "--decision-gate", str(decision_gate), "--out", str(ledger_path)])

            code = run_cli(
                [
                    "promotion-archive",
                    "--promotion-ledger",
                    str(ledger_path),
                    "--out",
                    str(archive_dir),
                    "--require-self-contained",
                ]
            )

            self.assertEqual(code, 1)
            archive = json.loads((archive_dir / "promotion_archive.json").read_text(encoding="utf-8"))
            self.assertFalse(archive["passed"])
            self.assertFalse(archive["self_contained"])
            self.assertEqual(archive["metrics"]["missing_count"], 1)
            self.assertEqual(archive["metrics"]["missing_role_counts"], [{"count": 1, "id": "decision_gate"}])
            self.assertEqual(run_cli(["validate", "--promotion-archive", str(archive_dir), "--strict"]), 0)

    def test_promotion_archive_force_refuses_non_archive_directories(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ledger_path = root / "promotion_ledger.json"
            protected_dir = root / "not_an_archive"
            protected_file = protected_dir / "keep.txt"
            ledger_path.write_text(
                json.dumps({"schema_version": "hfr.promotion_ledger.v1", "records": []}) + "\n",
                encoding="utf-8",
            )
            protected_dir.mkdir()
            protected_file.write_text("do not delete\n", encoding="utf-8")

            with self.assertRaises(SystemExit) as raised:
                run_cli(
                    [
                        "promotion-archive",
                        "--promotion-ledger",
                        str(ledger_path),
                        "--out",
                        str(protected_dir),
                        "--force",
                    ]
                )

            self.assertEqual(raised.exception.code, 2)
            self.assertTrue(protected_file.exists())
            self.assertFalse((protected_dir / "promotion_archive.json").exists())

    def test_promotion_archive_missing_source_indexes_reference_decision_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ledger_path = root / "promotion_ledger.json"
            first_gate = root / "first_decision_gate.json"
            second_gate = root / "second_decision_gate.json"
            archive_dir = root / "promotion_archive"
            ledger_path.write_text(
                json.dumps({"schema_version": "hfr.promotion_ledger.v1", "records": []}) + "\n",
                encoding="utf-8",
            )
            for gate_path in (first_gate, second_gate):
                gate_path.write_text(
                    json.dumps(
                        {
                            "schema_version": "hfr.decision_gate.v1",
                            "id": gate_path.stem,
                            "source_artifact": {"path": f"<redacted:{gate_path.stem}.json>"},
                        },
                        sort_keys=True,
                    )
                    + "\n",
                    encoding="utf-8",
                )

            code = run_cli(
                [
                    "promotion-archive",
                    "--promotion-ledger",
                    str(ledger_path),
                    "--decision-gate",
                    str(first_gate),
                    "--decision-gate",
                    str(second_gate),
                    "--out",
                    str(archive_dir),
                    "--require-self-contained",
                ]
            )

            self.assertEqual(code, 1)
            archive = json.loads((archive_dir / "promotion_archive.json").read_text(encoding="utf-8"))
            self.assertEqual([item["index"] for item in archive["missing"]], [1, 2])
            self.assertEqual([item["role"] for item in archive["missing"]], ["source_artifact", "source_artifact"])
            self.assertEqual(run_cli(["validate", "--promotion-archive", str(archive_dir), "--strict"]), 0)

    def test_promotion_archive_does_not_copy_traversing_recorded_source_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            evidence_dir = root / "evidence"
            secret_dir = root / "secret"
            evidence_dir.mkdir()
            secret_dir.mkdir()
            ledger_path = root / "promotion_ledger.json"
            decision_gate = evidence_dir / "decision_gate.json"
            secret_source = secret_dir / "secret.json"
            archive_dir = root / "promotion_archive"
            ledger_path.write_text(
                json.dumps({"schema_version": "hfr.promotion_ledger.v1", "records": []}) + "\n",
                encoding="utf-8",
            )
            secret_source.write_text(
                json.dumps(
                    {
                        "schema_version": "hfr.action_ledger_gate.v1",
                        "passed": True,
                        "decision": {"recommendation": "promote_iteration"},
                    },
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            decision_gate.write_text(
                json.dumps(
                    {
                        "schema_version": "hfr.decision_gate.v1",
                        "source_artifact": {"path": "../secret/secret.json"},
                    },
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )

            code = run_cli(
                [
                    "promotion-archive",
                    "--promotion-ledger",
                    str(ledger_path),
                    "--decision-gate",
                    str(decision_gate),
                    "--out",
                    str(archive_dir),
                    "--require-self-contained",
                ]
            )

            self.assertEqual(code, 1)
            archive = json.loads((archive_dir / "promotion_archive.json").read_text(encoding="utf-8"))
            self.assertFalse(archive["self_contained"])
            self.assertEqual(archive["metrics"]["source_artifact_count"], 0)
            self.assertEqual(archive["missing"][0]["role"], "source_artifact")
            self.assertIn("parent traversal", archive["missing"][0]["reason"])
            archived_text = "\n".join(path.read_text(encoding="utf-8") for path in (archive_dir / "artifacts").glob("*.json"))
            self.assertNotIn("hfr.action_ledger_gate.v1", archived_text)
            self.assertEqual(run_cli(["validate", "--promotion-archive", str(archive_dir), "--strict"]), 0)

    def test_promotion_archive_does_not_copy_absolute_recorded_source_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ledger_path = root / "promotion_ledger.json"
            decision_gate = root / "decision_gate.json"
            secret_source = root / "secret.json"
            archive_dir = root / "promotion_archive"
            ledger_path.write_text(
                json.dumps({"schema_version": "hfr.promotion_ledger.v1", "records": []}) + "\n",
                encoding="utf-8",
            )
            secret_source.write_text(
                json.dumps({"schema_version": "hfr.action_ledger_gate.v1", "passed": True}) + "\n",
                encoding="utf-8",
            )
            decision_gate.write_text(
                json.dumps(
                    {
                        "schema_version": "hfr.decision_gate.v1",
                        "source_artifact": {"path": str(secret_source)},
                    },
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )

            code = run_cli(
                [
                    "promotion-archive",
                    "--promotion-ledger",
                    str(ledger_path),
                    "--decision-gate",
                    str(decision_gate),
                    "--out",
                    str(archive_dir),
                    "--require-self-contained",
                ]
            )

            self.assertEqual(code, 1)
            archive = json.loads((archive_dir / "promotion_archive.json").read_text(encoding="utf-8"))
            self.assertEqual(archive["metrics"]["source_artifact_count"], 0)
            self.assertIn("absolute recorded paths", archive["missing"][0]["reason"])
            archived_text = "\n".join(path.read_text(encoding="utf-8") for path in (archive_dir / "artifacts").glob("*.json"))
            self.assertNotIn("hfr.action_ledger_gate.v1", archived_text)
            self.assertEqual(run_cli(["validate", "--promotion-archive", str(archive_dir), "--strict"]), 0)

    def test_promotion_archive_validation_rejects_symlinked_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ledger_path = root / "promotion_ledger.json"
            archive_dir = root / "promotion_archive"
            ledger_path.write_text(
                json.dumps({"schema_version": "hfr.promotion_ledger.v1", "records": []}) + "\n",
                encoding="utf-8",
            )
            self.assertEqual(
                run_cli(["promotion-archive", "--promotion-ledger", str(ledger_path), "--out", str(archive_dir)]),
                0,
            )
            archive = json.loads((archive_dir / "promotion_archive.json").read_text(encoding="utf-8"))
            artifact_path = archive_dir / archive["artifacts"][0]["path"]
            external_path = root / "external_ledger.json"
            external_path.write_text(artifact_path.read_text(encoding="utf-8"), encoding="utf-8")
            artifact_path.unlink()
            try:
                artifact_path.symlink_to(external_path)
            except (NotImplementedError, OSError) as exc:
                self.skipTest(f"symlink creation unavailable: {exc}")

            self.assertEqual(run_cli(["validate", "--promotion-archive", str(archive_dir), "--strict"]), 1)

    def test_promotion_archive_validation_rejects_bad_relationship_receipts(self):
        tamper_cases = (("to", "missing_promotion_ledger"), ("type", "source_artifact"))
        for field_name, forged_value in tamper_cases:
            with self.subTest(field_name=field_name):
                with tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp)
                    ledger_path = root / "promotion_ledger.json"
                    release_record_path = root / "promotion_release_record.json"
                    archive_dir = root / "promotion_archive"
                    ledger_path.write_text(
                        json.dumps({"schema_version": "hfr.promotion_ledger.v1", "records": []}) + "\n",
                        encoding="utf-8",
                    )
                    release_record_path.write_text(
                        json.dumps(
                            {
                                "schema_version": "hfr.promotion_release_record.v1",
                                "release": {"release_id": "release-2026-07-02"},
                            },
                            sort_keys=True,
                        )
                        + "\n",
                        encoding="utf-8",
                    )
                    self.assertEqual(
                        run_cli(
                            [
                                "promotion-archive",
                                "--promotion-ledger",
                                str(ledger_path),
                                "--promotion-release-record",
                                str(release_record_path),
                                "--out",
                                str(archive_dir),
                                "--require-self-contained",
                            ]
                        ),
                        0,
                    )
                    manifest_path = archive_dir / "promotion_archive.json"
                    archive = json.loads(manifest_path.read_text(encoding="utf-8"))
                    archive["relationships"][0][field_name] = forged_value
                    manifest_path.write_text(json.dumps(archive, indent=2, sort_keys=True) + "\n", encoding="utf-8")

                    self.assertEqual(run_cli(["validate", "--promotion-archive", str(archive_dir), "--strict"]), 1)

    def test_promotion_archive_redacts_original_paths_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ledger_path = root / "promotion_ledger.json"
            archive_dir = root / "promotion_archive"
            ledger_path.write_text(
                json.dumps({"schema_version": "hfr.promotion_ledger.v1", "records": []}) + "\n",
                encoding="utf-8",
            )

            self.assertEqual(
                run_cli(["promotion-archive", "--promotion-ledger", str(ledger_path), "--out", str(archive_dir)]),
                0,
            )

            archive = json.loads((archive_dir / "promotion_archive.json").read_text(encoding="utf-8"))
            self.assertEqual(archive["artifacts"][0]["original_path"], "<redacted:promotion_ledger.json>")

    def test_promotion_ledger_rejects_wrong_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            wrong_gate = root / "not_a_decision_gate.json"
            wrong_gate.write_text(json.dumps({"schema_version": "hfr.not_a_decision_gate.v1"}) + "\n", encoding="utf-8")

            with self.assertRaises(SystemExit) as raised:
                run_cli(["promotion-ledger", "--decision-gate", str(wrong_gate)])

            self.assertEqual(raised.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
