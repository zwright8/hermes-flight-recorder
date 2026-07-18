from __future__ import annotations

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


class PromotionDecisionTests(unittest.TestCase):
    def test_promotion_decision_allows_complete_governance_packet(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = self._fixture(Path(tmp))
            out = Path(tmp) / "promotion_decision.json"

            code = run_cli(self._decision_args(fixture, out))

            self.assertEqual(code, 0)
            decision = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(decision["schema_version"], "hfr.promotion_decision.v1")
            self.assertTrue(decision["passed"])
            self.assertEqual(decision["decision"]["recommendation"], "promote_candidate")
            self.assertEqual(decision["metrics"]["present_required_artifact_count"], 8)
            self.assertEqual(decision["metrics"]["present_required_eval_count"], 5)
            self.assertEqual(decision["metrics"]["passed_required_gate_count"], 3)
            self.assertEqual(decision["metrics"]["new_critical_failure_count"], 0)
            self.assertEqual(run_cli(["validate", "--promotion-decision", str(out), "--strict"]), 0)

    def test_promotion_decision_blocks_missing_model_card(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = self._fixture(Path(tmp))
            fixture["artifacts"].pop("model_card")
            out = Path(tmp) / "promotion_decision.json"

            code = run_cli(self._decision_args(fixture, out))

            self.assertEqual(code, 1)
            decision = json.loads(out.read_text(encoding="utf-8"))
            failed = {(check["id"], check.get("scope", {}).get("role")) for check in decision["checks"] if not check["passed"]}
            self.assertIn(("required_artifact", "model_card"), failed)
            self.assertIn(("card_required_section", "model_card"), failed)

    def test_promotion_decision_blocks_missing_rollback(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = self._fixture(Path(tmp))
            fixture["artifacts"].pop("rollback")
            out = Path(tmp) / "promotion_decision.json"

            code = run_cli(self._decision_args(fixture, out))

            self.assertEqual(code, 1)
            decision = json.loads(out.read_text(encoding="utf-8"))
            failed = {(check["id"], check.get("scope", {}).get("role")) for check in decision["checks"] if not check["passed"]}
            self.assertIn(("required_artifact", "rollback"), failed)
            self.assertIn(("rollback_target", "rollback"), failed)

    def test_promotion_decision_blocks_unknown_license(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture = self._fixture(root)
            self._write_json(
                fixture["artifacts"]["model_registry_entry"],
                {
                    "schema_version": "hfr.model_registry_entry.v1",
                    "model_id": "candidate-model",
                    "license_status": "unknown",
                },
            )
            out = root / "promotion_decision.json"

            code = run_cli(self._decision_args(fixture, out))

            self.assertEqual(code, 1)
            decision = json.loads(out.read_text(encoding="utf-8"))
            failed_ids = {check["id"] for check in decision["checks"] if not check["passed"]}
            self.assertIn("license_status", failed_ids)

    def test_promotion_decision_blocks_eval_scenario_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture = self._fixture(root)
            self._write_json(fixture["evals"]["champion"], self._eval_summary("champion", ["scenario_a", "scenario_c"], 1.0, 95.0, 2))
            out = root / "promotion_decision.json"

            code = run_cli(self._decision_args(fixture, out))

            self.assertEqual(code, 1)
            decision = json.loads(out.read_text(encoding="utf-8"))
            failed = [
                check
                for check in decision["checks"]
                if not check["passed"] and check["id"] == "eval_scenarios_match"
            ]
            self.assertTrue(any(check.get("scope", {}).get("arm") == "champion" for check in failed))

    def test_promotion_decision_blocks_new_critical_and_secret_exposure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture = self._fixture(root)
            self._write_json(
                fixture["evals"]["candidate"],
                self._eval_summary(
                    "candidate",
                    ["scenario_a", "scenario_b"],
                    1.0,
                    98.0,
                    2,
                    critical_counts={"secret_exposure": 1},
                ),
            )
            out = root / "promotion_decision.json"

            code = run_cli(self._decision_args(fixture, out))

            self.assertEqual(code, 1)
            decision = json.loads(out.read_text(encoding="utf-8"))
            failed_ids = {check["id"] for check in decision["checks"] if not check["passed"]}
            self.assertIn("no_new_critical_failures", failed_ids)
            self.assertIn("forbidden_candidate_critical_rules", failed_ids)

    def test_registry_alias_receipt_allows_passing_promotion_decision_without_mutating_registry(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture = self._fixture(root)
            decision_path = root / "promotion_decision.json"
            registry_path = root / "model_registry.json"
            receipt_path = root / "registry_alias_receipt.json"
            self._write_registry(registry_path)

            decision_args = self._decision_args(fixture, decision_path)
            decision_args.extend(["--metadata", "target_entry_id=candidate-v2"])
            self.assertEqual(run_cli(decision_args), 0)
            registry_before = json.loads(registry_path.read_text(encoding="utf-8"))

            code = run_cli(
                [
                    "model-registry",
                    "alias-receipt",
                    "--registry",
                    str(registry_path),
                    "--promotion-decision",
                    str(decision_path),
                    "--alias",
                    "champion",
                    "--target",
                    "candidate-v2",
                    "--rollback-target",
                    "champion-v1",
                    "--reason",
                    "release candidate-v2",
                    "--out",
                    str(receipt_path),
                    "--preserve-paths",
                ]
            )

            self.assertEqual(code, 0)
            self.assertEqual(json.loads(registry_path.read_text(encoding="utf-8")), registry_before)
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            self.assertEqual(receipt["schema_version"], "hfr.registry_alias_receipt.v1")
            self.assertTrue(receipt["passed"])
            self.assertEqual(receipt["recommendation"], "apply_alias_update")
            self.assertFalse(receipt["alias_update"]["side_effects"])
            self.assertFalse(receipt["alias_update"]["applied"])
            self.assertEqual(receipt["alias_update"]["planned_alias_history"][0]["alias"], "rollback")
            self.assertEqual(receipt["alias_update"]["planned_alias_history"][1]["alias"], "champion")
            self.assertEqual(run_cli(["validate", "--registry-alias-receipt", str(receipt_path), "--strict"]), 0)

    def test_registry_alias_receipt_blocks_failed_promotion_decision(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture = self._fixture(root)
            fixture["artifacts"].pop("model_card")
            decision_path = root / "promotion_decision.json"
            registry_path = root / "model_registry.json"
            receipt_path = root / "registry_alias_receipt.json"
            self._write_registry(registry_path)
            decision_args = self._decision_args(fixture, decision_path)
            decision_args.extend(["--metadata", "target_entry_id=candidate-v2"])
            self.assertEqual(run_cli(decision_args), 1)

            code = run_cli(
                [
                    "model-registry",
                    "alias-receipt",
                    "--registry",
                    str(registry_path),
                    "--promotion-decision",
                    str(decision_path),
                    "--alias",
                    "candidate",
                    "--target",
                    "candidate-v2",
                    "--out",
                    str(receipt_path),
                    "--preserve-paths",
                ]
            )

            self.assertEqual(code, 1)
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            self.assertFalse(receipt["passed"])
            self.assertEqual(receipt["recommendation"], "block_alias_update")
            failed_ids = {check["id"] for check in receipt["checks"] if not check["passed"]}
            self.assertIn("promotion_decision_passed", failed_ids)
            self.assertIn("promotion_decision_no_failed_checks", failed_ids)
            self.assertEqual(run_cli(["validate", "--registry-alias-receipt", str(receipt_path), "--strict"]), 0)

    def test_registry_alias_receipt_requires_rollback_for_champion_alias(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture = self._fixture(root)
            decision_path = root / "promotion_decision.json"
            registry_path = root / "model_registry.json"
            receipt_path = root / "registry_alias_receipt.json"
            self._write_registry(registry_path)
            decision_args = self._decision_args(fixture, decision_path)
            decision_args.extend(["--metadata", "target_entry_id=candidate-v2"])
            self.assertEqual(run_cli(decision_args), 0)

            code = run_cli(
                [
                    "model-registry",
                    "alias-receipt",
                    "--registry",
                    str(registry_path),
                    "--promotion-decision",
                    str(decision_path),
                    "--alias",
                    "champion",
                    "--target",
                    "candidate-v2",
                    "--out",
                    str(receipt_path),
                    "--preserve-paths",
                ]
            )

            self.assertEqual(code, 1)
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            failed_ids = {check["id"] for check in receipt["checks"] if not check["passed"]}
            self.assertIn("champion_has_rollback_target", failed_ids)
            self.assertEqual(run_cli(["validate", "--registry-alias-receipt", str(receipt_path), "--strict"]), 0)

    def test_promotion_cards_generate_valid_cards_for_decision(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture = self._fixture(root)
            fixture["artifacts"].pop("model_card")
            fixture["artifacts"].pop("dataset_card")
            out_dir = root / "cards"
            manifest_path = out_dir / "promotion_cards.json"
            decision_path = root / "promotion_decision.json"

            code = run_cli(self._cards_args(fixture, out_dir))

            self.assertEqual(code, 0)
            self.assertTrue((out_dir / "MODEL_CARD.md").exists())
            self.assertTrue((out_dir / "DATASET_CARD.md").exists())
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["schema_version"], "hfr.promotion_cards.v1")
            self.assertTrue(manifest["passed"])
            self.assertIn("## Intended Use", (out_dir / "MODEL_CARD.md").read_text(encoding="utf-8"))
            self.assertIn("## Boundaries", (out_dir / "DATASET_CARD.md").read_text(encoding="utf-8"))
            self.assertEqual(run_cli(["validate", "--promotion-cards", str(manifest_path), "--strict"]), 0)

            fixture["artifacts"]["model_card"] = out_dir / "MODEL_CARD.md"
            fixture["artifacts"]["dataset_card"] = out_dir / "DATASET_CARD.md"
            self.assertEqual(run_cli(self._decision_args(fixture, decision_path)), 0)
            self.assertEqual(run_cli(["validate", "--promotion-decision", str(decision_path), "--strict"]), 0)

    def test_promotion_cards_block_missing_rollback_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture = self._fixture(root)
            fixture["artifacts"].pop("model_card")
            fixture["artifacts"].pop("dataset_card")
            fixture["artifacts"].pop("rollback")
            out_dir = root / "cards"
            manifest_path = out_dir / "promotion_cards.json"

            code = run_cli(self._cards_args(fixture, out_dir))

            self.assertEqual(code, 1)
            self.assertTrue((out_dir / "MODEL_CARD.md").exists())
            self.assertTrue((out_dir / "DATASET_CARD.md").exists())
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertFalse(manifest["passed"])
            failed_ids = {check["id"] for check in manifest["checks"] if not check["passed"]}
            self.assertIn("card_source_artifact", failed_ids)
            self.assertIn("rollback_target", failed_ids)
            self.assertEqual(run_cli(["validate", "--promotion-cards", str(manifest_path), "--strict"]), 0)

    def test_promotion_release_record_binds_passed_governance_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture = self._fixture(root)
            cards_dir = root / "cards"
            cards_path = cards_dir / "promotion_cards.json"
            decision_path = root / "promotion_decision.json"
            registry_path = root / "model_registry.json"
            alias_receipt_path = root / "registry_alias_receipt.json"
            release_path = root / "promotion_release_record.json"
            notes_path = root / "RELEASE_NOTES.md"
            self._write_registry(registry_path)
            self.assertEqual(run_cli(self._cards_args(fixture, cards_dir)), 0)
            fixture["artifacts"]["model_card"] = cards_dir / "MODEL_CARD.md"
            fixture["artifacts"]["dataset_card"] = cards_dir / "DATASET_CARD.md"
            decision_args = self._decision_args(fixture, decision_path)
            decision_args.extend(["--metadata", "target_entry_id=candidate-v2"])
            self.assertEqual(run_cli(decision_args), 0)
            self.assertEqual(
                run_cli(
                    [
                        "model-registry",
                        "alias-receipt",
                        "--registry",
                        str(registry_path),
                        "--promotion-decision",
                        str(decision_path),
                        "--alias",
                        "champion",
                        "--target",
                        "candidate-v2",
                        "--rollback-target",
                        "champion-v1",
                        "--out",
                        str(alias_receipt_path),
                        "--preserve-paths",
                    ]
                ),
                0,
            )

            code = run_cli(self._release_args(fixture, decision_path, cards_path, alias_receipt_path, release_path, notes_path))

            self.assertEqual(code, 0)
            release = json.loads(release_path.read_text(encoding="utf-8"))
            self.assertEqual(release["schema_version"], "hfr.promotion_release_record.v1")
            self.assertTrue(release["passed"])
            self.assertEqual(release["recommendation"], "record_release")
            self.assertTrue(notes_path.exists())
            self.assertIn("release-001", notes_path.read_text(encoding="utf-8"))
            self.assertEqual(run_cli(["validate", "--promotion-release-record", str(release_path), "--strict"]), 0)

    def test_promotion_release_record_blocks_failed_alias_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture = self._fixture(root)
            cards_dir = root / "cards"
            cards_path = cards_dir / "promotion_cards.json"
            decision_path = root / "promotion_decision.json"
            registry_path = root / "model_registry.json"
            alias_receipt_path = root / "registry_alias_receipt.json"
            release_path = root / "promotion_release_record.json"
            notes_path = root / "RELEASE_NOTES.md"
            self._write_registry(registry_path)
            self.assertEqual(run_cli(self._cards_args(fixture, cards_dir)), 0)
            fixture["artifacts"]["model_card"] = cards_dir / "MODEL_CARD.md"
            fixture["artifacts"]["dataset_card"] = cards_dir / "DATASET_CARD.md"
            decision_args = self._decision_args(fixture, decision_path)
            decision_args.extend(["--metadata", "target_entry_id=candidate-v2"])
            self.assertEqual(run_cli(decision_args), 0)
            self.assertEqual(
                run_cli(
                    [
                        "model-registry",
                        "alias-receipt",
                        "--registry",
                        str(registry_path),
                        "--promotion-decision",
                        str(decision_path),
                        "--alias",
                        "champion",
                        "--target",
                        "candidate-v2",
                        "--out",
                        str(alias_receipt_path),
                        "--preserve-paths",
                    ]
                ),
                1,
            )

            code = run_cli(self._release_args(fixture, decision_path, cards_path, alias_receipt_path, release_path, notes_path))

            self.assertEqual(code, 1)
            release = json.loads(release_path.read_text(encoding="utf-8"))
            self.assertFalse(release["passed"])
            failed_ids = {check["id"] for check in release["checks"] if not check["passed"]}
            self.assertIn("release_component_passed", failed_ids)
            self.assertEqual(run_cli(["validate", "--promotion-release-record", str(release_path), "--strict"]), 0)

    def _fixture(self, root: Path) -> dict[str, dict[str, Path]]:
        artifacts = {
            "evidence_bundle": root / "evidence_bundle.json",
            "dataset_manifest": root / "dataset_manifest.json",
            "dataset_card": root / "DATASET_CARD.md",
            "model_registry_entry": root / "model_registry_entry.json",
            "training_result": root / "training_result.json",
            "serving_profile": root / "serving_profile.json",
            "model_card": root / "MODEL_CARD.md",
            "rollback": root / "rollback.json",
        }
        evals = {
            "base": root / "base_eval.json",
            "trace_only": root / "trace_only_eval.json",
            "frontier": root / "frontier_eval.json",
            "champion": root / "champion_eval.json",
            "candidate": root / "candidate_eval.json",
        }
        gates = {
            "training_gate": root / "training_gate.json",
            "compare_gate": root / "compare_gate.json",
            "safety_gate": root / "safety_gate.json",
        }
        self._write_json(
            artifacts["evidence_bundle"],
            {
                "schema_version": "hfr.evidence_bundle.v1",
                "passed": True,
                "decision": {"readiness": "ready", "recommendation": "handoff_ready"},
            },
        )
        self._write_json(
            artifacts["dataset_manifest"],
            {
                "schema_version": "hfr.rl.manifest.v1",
                "redaction": {"status": "passed"},
            },
        )
        artifacts["dataset_card"].write_text(
            "# Dataset Card\n\n## Summary\n\nDataset summary.\n\n## Boundaries\n\nDataset boundaries.\n",
            encoding="utf-8",
        )
        self._write_json(
            artifacts["model_registry_entry"],
            {
                "schema_version": "hfr.model_registry_entry.v1",
                "model_id": "candidate-model",
                "license_status": "approved",
            },
        )
        self._write_json(artifacts["training_result"], {"schema_version": "hfr.training_result.v1", "passed": True})
        self._write_json(artifacts["serving_profile"], {"schema_version": "hfr.serving_profile.v1", "passed": True})
        artifacts["model_card"].write_text(
            "# Model Card\n\n## Summary\n\nModel summary.\n\n## Intended Use\n\nAgentic task completion.\n\n"
            "## Limitations\n\nLimited to validated scenarios.\n\n## Rollback\n\nUse champion-v1.\n",
            encoding="utf-8",
        )
        self._write_json(
            artifacts["rollback"],
            {"schema_version": "hfr.rollback.v1", "target_model_id": "champion-v1", "alias": "rollback"},
        )
        self._write_json(evals["base"], self._eval_summary("base", ["scenario_a", "scenario_b"], 0.5, 72.0, 1))
        self._write_json(evals["trace_only"], self._eval_summary("trace_only", ["scenario_a", "scenario_b"], 0.5, 70.0, 1))
        self._write_json(evals["frontier"], self._eval_summary("frontier", ["scenario_a", "scenario_b"], 1.0, 92.0, 2))
        self._write_json(evals["champion"], self._eval_summary("champion", ["scenario_a", "scenario_b"], 1.0, 95.0, 2))
        self._write_json(evals["candidate"], self._eval_summary("candidate", ["scenario_a", "scenario_b"], 1.0, 98.0, 2))
        for gate_id, path in gates.items():
            self._write_json(
                path,
                {
                    "schema_version": f"hfr.{gate_id}.v1",
                    "passed": True,
                    "decision": {"readiness": "ready", "recommendation": "allow"},
                },
            )
        return {"artifacts": artifacts, "evals": evals, "gates": gates}

    def _decision_args(self, fixture: dict[str, dict[str, Path]], out: Path) -> list[str]:
        args = [
            "promotion-decision",
            "--policy",
            str(ROOT / "examples" / "promotion_policy.demo.json"),
            "--out",
            str(out),
            "--preserve-paths",
        ]
        for role, path in fixture["artifacts"].items():
            args.extend(["--artifact", f"{role}={path}"])
        for arm, path in fixture["evals"].items():
            args.extend(["--eval", f"{arm}={path}"])
        for gate_id, path in fixture["gates"].items():
            args.extend(["--gate", f"{gate_id}={path}"])
        return args

    def _cards_args(self, fixture: dict[str, dict[str, Path]], out_dir: Path) -> list[str]:
        args = [
            "promotion-cards",
            "--policy",
            str(ROOT / "examples" / "promotion_policy.demo.json"),
            "--out-dir",
            str(out_dir),
            "--preserve-paths",
            "--metadata",
            "release_id=release-001",
        ]
        for role, path in fixture["artifacts"].items():
            args.extend(["--artifact", f"{role}={path}"])
        for arm, path in fixture["evals"].items():
            args.extend(["--eval", f"{arm}={path}"])
        for gate_id, path in fixture["gates"].items():
            args.extend(["--gate", f"{gate_id}={path}"])
        return args

    def _release_args(
        self,
        fixture: dict[str, dict[str, Path]],
        decision_path: Path,
        cards_path: Path,
        alias_receipt_path: Path,
        release_path: Path,
        notes_path: Path,
    ) -> list[str]:
        args = [
            "promotion-release-record",
            "--release-id",
            "release-001",
            "--policy",
            str(ROOT / "examples" / "promotion_policy.demo.json"),
            "--promotion-decision",
            str(decision_path),
            "--promotion-cards",
            str(cards_path),
            "--registry-alias-receipt",
            str(alias_receipt_path),
            "--rollback",
            str(fixture["artifacts"]["rollback"]),
            "--out",
            str(release_path),
            "--notes-out",
            str(notes_path),
            "--preserve-paths",
        ]
        for arm, path in fixture["evals"].items():
            args.extend(["--eval", f"{arm}={path}"])
        return args

    def _eval_summary(
        self,
        arm: str,
        scenario_ids: list[str],
        pass_rate: float,
        average_score: float,
        passed: int,
        *,
        critical_counts: dict[str, int] | None = None,
    ) -> dict[str, object]:
        critical_counts = critical_counts or {}
        failed = len(scenario_ids) - passed
        return {
            "schema_version": "hfr.hermes_heldout_eval_summary.v1",
            "arm": arm,
            "model": f"{arm}-model",
            "scenario_ids": scenario_ids,
            "total": len(scenario_ids),
            "passed": passed,
            "failed": failed,
            "error_count": 0,
            "metrics": {
                "pass_rate": pass_rate,
                "average_score": average_score,
                "passed": passed,
                "failed": failed,
                "critical_failure_counts": [
                    {"id": rule_id, "count": count}
                    for rule_id, count in sorted(critical_counts.items())
                ],
            },
        }

    def _write_json(self, path: Path, payload: dict[str, object]) -> None:
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def _write_registry(self, path: Path) -> None:
        registry = {
            "schema_version": "hfr.model_registry.v1",
            "registry_path": str(path),
            "updated_at": "2026-07-02T13:00:00Z",
            "entries": {
                entry_id: self._registry_entry(entry_id)
                for entry_id in ("rollback-v0", "champion-v1", "candidate-v2")
            },
            "aliases": {
                "candidate": "candidate-v2",
                "champion": "champion-v1",
                "rollback": "rollback-v0",
            },
            "alias_history": [],
            "notes": [],
        }
        self._write_json(path, registry)

    def _registry_entry(self, entry_id: str) -> dict[str, object]:
        candidate = {
            "schema_version": "hfr.model_candidate.v1",
            "candidate_id": entry_id,
            "model_id": f"local/{entry_id}",
            "source": {"type": "local", "url": f"file:///models/{entry_id}"},
            "license": {
                "status": "approved",
                "license_id": "apache-2.0",
                "source_url": "https://example.invalid/license",
                "review_status": "approved",
                "terms_url": "https://example.invalid/terms",
                "accepted_terms": True,
                "training_allowed": True,
                "reviewed_at": "2026-07-02T13:00:00Z",
                "reviewer": "governance-test",
            },
            "compatibility": {
                "context_length": 8192,
                "tokenizer": {},
                "chat_template": {},
                "serving": {},
                "tool_calls": {"supported": True},
                "structured_outputs": {"supported": True},
                "quantization": {},
                "memory": {},
            },
        }
        return {
            "schema_version": "hfr.model_registry_entry.v1",
            "entry_id": entry_id,
            "candidate_id": entry_id,
            "registered_at": "2026-07-02T13:00:00Z",
            "updated_at": "2026-07-02T13:00:00Z",
            "status": "registered",
            "training_eligible": True,
            "license_status": "approved",
            "candidate": candidate,
            "datasets": [],
            "training_runs": [],
            "adapters": [],
            "evals": [],
            "promotion_decisions": [],
            "notes": [],
        }


if __name__ == "__main__":
    unittest.main()
