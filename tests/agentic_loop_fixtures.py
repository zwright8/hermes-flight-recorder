import json
import shutil
from pathlib import Path

from flightrecorder.eval_summary import build_eval_summary
from flightrecorder.governance import build_promotion_decision
from flightrecorder.model_grader import build_model_grader_override_receipt
from flightrecorder.promotion_ledger import build_promotion_ledger


ROOT = Path(__file__).resolve().parents[1]


def copy_valid_loop_artifacts(root: Path) -> dict[str, list[Path]]:
    """Copy the complete schema-valid closed-loop example into an isolated fixture tree."""
    fixture_examples = root / "loop_fixture" / "examples"
    shutil.copytree(ROOT / "examples", fixture_examples)
    fixture_root = fixture_examples / "agentic_training"
    plan = json.loads((fixture_root / "loop_plan.json").read_text(encoding="utf-8"))
    artifacts = {
        role: [fixture_root / ref["path"] for ref in refs]
        for role, refs in plan["source_artifacts"].items()
        if isinstance(refs, list)
    }
    override_rows = fixture_root / "model_grader" / "override_rows.jsonl"
    override_rows.write_text("", encoding="utf-8")
    override_path = fixture_root / "model_grader" / "override_receipt.json"
    override = build_model_grader_override_receipt(
        dry_run_path=artifacts["model_grader_dry_run"][0],
        overrides_path=override_rows,
        out_path=override_path,
        created_at="2026-07-03T00:00:00+00:00",
    )
    override_path.write_text(json.dumps(override, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    artifacts["model_grader_override_receipt"] = [override_path]
    artifacts["next_iteration_schedule"] = [fixture_root / "next_iteration_schedule.json"]
    return artifacts


def write_eval_summary(root: Path) -> Path:
    suite = root / "eval_suite_summary.json"
    _write_suite_summary(suite)
    summary = build_eval_summary(suite_summary_specs=[suite], output_base_dir=root)
    path = root / "eval_summary.json"
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def write_valid_promotion_ledger(root: Path) -> Path:
    decision_gate = root / "decision_gate.json"
    decision_gate.write_text(
        json.dumps(
            {
                "schema_version": "hfr.decision_gate.v1",
                "passed": True,
                "readiness": "ready",
                "recommendation": "allow_promotion",
                "expected_recommendation": "promote_iteration",
                "expected_readiness": "ready",
                "require_passed": True,
                "check_count": 1,
                "failed_check_count": 0,
                "source_decision": {
                    "schema_version": "hfr.action_ledger_gate.v1",
                    "passed": True,
                    "readiness": "ready",
                    "recommendation": "promote_iteration",
                    "blocking_check_count": 0,
                },
                "source_artifact": {},
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    ledger_path = root / "promotion_ledger.json"
    ledger = build_promotion_ledger([decision_gate], out_path=ledger_path)
    ledger_path.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return ledger_path


def write_valid_promotion_decision(root: Path) -> Path:
    sources = root / "promotion_decision_sources"
    sources.mkdir(exist_ok=True)
    artifacts = _write_promotion_decision_sources(sources)
    decision_path = root / "promotion_decision.json"
    decision = build_promotion_decision(
        candidate_id="candidate-v2",
        champion_id="champion-v1",
        rollback_id="champion-v1",
        out_path=decision_path,
        evidence_bundle_path=artifacts["evidence_bundle"],
        promotion_ledger_gate_path=artifacts["promotion_ledger_gate"],
        compare_gate_path=artifacts["compare_gate"],
        trainer_launch_check_path=artifacts["trainer_launch_check"],
        model_registry_entry_path=artifacts["model_registry_entry"],
        agentic_training_result_path=artifacts["agentic_training_result"],
        model_card_path=artifacts["model_card"],
        dataset_card_path=artifacts["dataset_card"],
        rollback_metadata_path=artifacts["rollback_metadata"],
        license_review_path=artifacts["license_review"],
        redaction_check_path=artifacts["redaction_check"],
        safety_gate_path=artifacts["safety_gate"],
        serving_profile_path=artifacts["serving_profile"],
        serving_report_path=artifacts["serving_report"],
    )
    decision_path.write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return decision_path


def _write_suite_summary(path: Path) -> Path:
    payload = {
        "schema_version": "hfr.run_suite.v1",
        "scenarios_dir": "scenarios",
        "out_dir": "runs",
        "total": 1,
        "passed": 1,
        "failed": 0,
        "error_count": 0,
        "errors": [],
        "metrics": {
            "pass_rate": 1.0,
            "average_score": 100.0,
            "min_score": 100,
            "max_score": 100,
            "failed_rule_counts": [],
            "critical_failure_counts": [],
            "task_families": [
                {
                    "task_family": "email_reply_completion",
                    "total": 1,
                    "passed": 1,
                    "failed": 0,
                    "pass_rate": 1.0,
                    "average_score": 100.0,
                    "failed_rule_counts": [],
                    "critical_failure_counts": [],
                }
            ],
            "failed": 0,
            "passed": 1,
        },
        "runs": [
            {
                "scenario_id": "email_reply_completion",
                "scenario_title": "email_reply_completion",
                "task_family": "email_reply_completion",
                "scenario_path": "scenarios/email_reply_completion.json",
                "trace_path": "traces/email_reply_completion.jsonl",
                "run_dir": "runs/email_reply_completion",
                "report": "runs/email_reply_completion/report.html",
                "report_sha256": "b" * 64,
                "report_size_bytes": 1,
                "scorecard": "runs/email_reply_completion/scorecard.json",
                "scorecard_sha256": "c" * 64,
                "scorecard_size_bytes": 1,
                "run_digest": "runs/email_reply_completion/run_digest.json",
                "run_digest_sha256": "d" * 64,
                "run_digest_size_bytes": 1,
                "lineage": "runs/email_reply_completion/artifact_lineage.json",
                "lineage_sha256": "e" * 64,
                "lineage_size_bytes": 1,
                "passed": True,
                "score": 100,
                "failed_rules": [],
                "critical_failures": [],
            }
        ],
        "artifacts": {"suite_result": "runs/harness_suite_result.json"},
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _write_promotion_decision_sources(root: Path) -> dict[str, Path]:
    payloads = {
        "evidence_bundle": {"schema_version": "hfr.evidence_bundle.v1", "passed": True},
        "promotion_ledger_gate": {"schema_version": "hfr.promotion_ledger_gate.v1", "passed": True},
        "compare_gate": {
            "schema_version": "hfr.compare_gate.v1",
            "passed": True,
            "metrics": {
                "baseline_win_count": 0,
                "contract_drift_count": 0,
                "new_critical_failure_counts": {},
                "regressed_rule_counts": {},
                "task_completion_regression_count": 0,
                "unverified_contract_count": 0,
            },
        },
        "trainer_launch_check": {"schema_version": "hfr.trainer_launch_check.v1", "passed": True},
        "model_registry_entry": {
            "schema_version": "hfr.model_registry_entry.v1",
            "candidate_id": "candidate-v2",
            "entry_id": "candidate-v2",
        },
        "agentic_training_result": {"schema_version": "hfr.agentic_training_result.v1", "passed": True},
        "rollback_metadata": {"available": True, "rollback_id": "champion-v1"},
        "license_review": {"passed": True, "license_status": "known", "accepted_terms": True},
        "redaction_check": {"passed": True},
        "safety_gate": {"passed": True},
        "serving_profile": {
            "schema_version": "hfr.serving_profile.v1",
            "eval_preflight": {"ready": True, "readiness": "ready", "failed_checks": []},
        },
        "serving_report": {"passed": True},
    }
    paths = {role: _write_source_json(root / f"{role}.json", payload) for role, payload in payloads.items()}
    paths["model_card"] = root / "MODEL_CARD.md"
    paths["model_card"].write_text("# Model Card\n\nEvidence-backed candidate model.\n", encoding="utf-8")
    paths["dataset_card"] = root / "DATASET_CARD.md"
    paths["dataset_card"].write_text("# Dataset Card\n\nRedacted held-out data.\n", encoding="utf-8")
    return paths


def _write_source_json(path: Path, payload: dict[str, object]) -> Path:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
