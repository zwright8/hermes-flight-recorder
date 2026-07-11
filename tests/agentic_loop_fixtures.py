import hashlib
import json
import shutil
from pathlib import Path

from flightrecorder.bundle import build_evidence_bundle
from flightrecorder.cloud_training_completion import (
    build_cloud_training_completion_receipt,
    write_cloud_training_completion_receipt,
)
from flightrecorder.eval_summary import build_eval_summary
from flightrecorder.external_eval import build_external_eval_plan, write_external_eval_plan
from flightrecorder.external_eval_result import build_external_eval_result, write_external_eval_result
from flightrecorder.governance import build_promotion_decision
from flightrecorder.model_grader import build_model_grader_override_receipt


ROOT = Path(__file__).resolve().parents[1]
PROMOTION_CANDIDATE_ID = "local/mock-candidate"


def write_cloud_completion_fixture(
    root: Path,
    agentic_training_result_path: Path,
    candidate_id: str,
    *,
    status: str = "completed",
) -> Path:
    """Write deterministic import-only cloud completion evidence for tests."""
    cloud_root = root / "cloud_training"
    shutil.copytree(
        ROOT / "examples" / "agentic_training" / "cloud_training",
        cloud_root,
        dirs_exist_ok=True,
    )
    raw_result_path = cloud_root / "raw_provider_result.json"
    launch_plan_path = cloud_root / "launch_plan.json"
    launch_receipt_path = cloud_root / "launch_receipt.json"
    status_receipt_path = cloud_root / "status_receipt.json"
    runner_metadata_path = cloud_root / "runner_metadata.json"
    failed = status != "completed"
    failure_class = "provider" if failed else "none"
    failure_message = "external provider reported failure" if failed else ""
    metadata = {
        "schema_version": "hfr.external_cloud_training_runner.v1",
        "provider_id": "modal",
        "provider_job_id": "fixture-modal-job-001",
        "execution_id": "fixture-cloud-training-001",
        "candidate_model_id": candidate_id,
        "status": status,
        "terminal": status in {"completed", "failed"},
        "failure": {"class": failure_class, "message": failure_message},
        "runner": {"id": "fixture-external-cloud-runner", "version": "1"},
        "started_at": "2026-07-03T00:10:00+00:00",
        "finished_at": "2026-07-03T00:20:00+00:00",
        "exit_code": 1 if failed else 0,
        "provider_constraints": {
            "region": "provider_default",
            "gpu_class": "a100",
            "reported_cost_usd": 0,
        },
        "source_sha256": {
            "launch_plan": _sha256(launch_plan_path),
            "launch_receipt": _sha256(launch_receipt_path),
            "status_receipt": _sha256(status_receipt_path),
            "raw_provider_result": _sha256(raw_result_path),
            "output_artifact_manifest": _sha256(agentic_training_result_path),
        },
        "side_effects": {
            "external_provider_api_called": True,
            "external_cloud_job_started": True,
            "external_artifacts_uploaded": True,
            "external_artifacts_downloaded": True,
            "credential_values_recorded": "not_observed",
            "provider_api_called_by_flight_recorder": False,
            "cloud_job_started_by_flight_recorder": False,
            "provider_status_polled_by_flight_recorder": False,
            "artifacts_uploaded_by_flight_recorder": False,
            "artifacts_downloaded_by_flight_recorder": False,
            "model_downloads_started_by_flight_recorder": False,
            "weights_updated_by_flight_recorder": False,
            "provider_modules_imported_by_flight_recorder": False,
        },
    }
    runner_metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    completion_path = root / "cloud_training_completion_receipt.json"
    receipt = build_cloud_training_completion_receipt(
        launch_plan_path=launch_plan_path,
        launch_receipt_path=launch_receipt_path,
        status_receipt_path=status_receipt_path,
        runner_metadata_path=runner_metadata_path,
        raw_provider_result_path=raw_result_path,
        output_artifact_manifest_path=agentic_training_result_path,
        out_path=completion_path,
        created_at="2026-07-03T00:30:00+00:00",
    )
    write_cloud_training_completion_receipt(receipt, completion_path)
    return completion_path


def copy_valid_loop_artifacts(root: Path) -> dict[str, list[Path]]:
    """Copy the complete schema-valid closed-loop example into an isolated fixture tree."""
    fixture_examples = root / "loop_fixture" / "examples"
    shutil.copytree(ROOT / "examples", fixture_examples)
    shutil.copyfile(ROOT / "pyproject.toml", fixture_examples.parent / "pyproject.toml")
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


def write_valid_promotion_decision(root: Path) -> Path:
    sources = root / "promotion_decision_sources"
    sources.mkdir(exist_ok=True)
    artifacts = _write_promotion_decision_sources(sources)
    decision_path = root / "promotion_decision.json"
    decision = build_promotion_decision(
        candidate_id=PROMOTION_CANDIDATE_ID,
        champion_id="champion-v1",
        rollback_id="champion-v1",
        out_path=decision_path,
        evidence_bundle_path=artifacts["evidence_bundle"],
        eval_summary_path=artifacts["eval_summary"],
        external_eval_result_paths=artifacts["external_eval_results"],
        promotion_ledger_gate_path=artifacts["promotion_ledger_gate"],
        compare_gate_path=artifacts["compare_gate"],
        trainer_launch_check_path=artifacts["trainer_launch_check"],
        model_registry_entry_path=artifacts["model_registry_entry"],
        agentic_training_result_path=artifacts["agentic_training_result"],
        cloud_training_completion_receipt_path=artifacts[
            "cloud_training_completion_receipt"
        ],
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


def _write_promotion_decision_sources(root: Path) -> dict[str, Path | list[Path]]:
    eval_lineage = _write_promotion_eval_lineage(root / "promotion_eval")
    evidence_root = root / "evidence"
    evidence_root.mkdir(exist_ok=True)
    evidence_bundle_path = evidence_root / "evidence_bundle.json"
    evidence_bundle = build_evidence_bundle(
        out_path=evidence_bundle_path,
        eval_summary_path=eval_lineage["eval_summary"],
    )
    evidence_bundle_path.write_text(
        json.dumps(evidence_bundle, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    example_root = ROOT / "examples" / "agentic_training"
    for filename in ("promotion_ledger.json", "promotion_history_decision_gate.json"):
        shutil.copyfile(
            example_root / "promotion_governance" / filename,
            root / filename,
        )
    registry_entry_payload = _read_json_object(
        example_root / "promotion_governance" / "model_registry_entry.json"
    )
    payloads = {
        "promotion_ledger_gate": _read_json_object(
            example_root / "promotion_governance" / "promotion_ledger_gate.json"
        ),
        "compare_gate": _read_json_object(
            example_root / "promotion_governance" / "compare_gate.json"
        ),
        "trainer_launch_check": _read_json_object(
            example_root / "trainer_launch_check.json"
        ),
        "rollback_metadata": {"available": True, "rollback_id": "champion-v1"},
        "license_review": {"passed": True, "license_status": "known", "accepted_terms": True},
        "redaction_check": {"passed": True},
        "safety_gate": {"passed": True},
        "serving_profile": _read_json_object(
            example_root
            / "serving_lifecycle"
            / "managed_mock"
            / "preflight"
            / "serving_profile.json"
        ),
        "serving_report": {"passed": True},
    }
    paths = {role: _write_source_json(root / f"{role}.json", payload) for role, payload in payloads.items()}
    paths["agentic_training_result"] = _write_candidate_training_result(
        root, PROMOTION_CANDIDATE_ID
    )
    paths["cloud_training_completion_receipt"] = write_cloud_completion_fixture(
        root,
        paths["agentic_training_result"],
        PROMOTION_CANDIDATE_ID,
    )
    _bind_registry_entry_links(
        registry_entry_payload,
        {
            "datasets": example_root / "training_export" / "manifest.json",
            "evals": paths["compare_gate"],
            "serving_probes": (
                example_root
                / "serving_lifecycle"
                / "managed_mock"
                / "preflight"
                / "serving_check.json"
            ),
            "training_runs": paths["agentic_training_result"],
        },
    )
    paths["model_registry_entry"] = _write_source_json(
        root / "model_registry_entry.json", registry_entry_payload
    )
    paths["evidence_bundle"] = evidence_bundle_path
    paths["eval_summary"] = eval_lineage["eval_summary"]
    paths["external_eval_results"] = [eval_lineage["external_eval_result"]]
    paths["model_card"] = root / "MODEL_CARD.md"
    paths["model_card"].write_text("# Model Card\n\nEvidence-backed candidate model.\n", encoding="utf-8")
    paths["dataset_card"] = root / "DATASET_CARD.md"
    paths["dataset_card"].write_text("# Dataset Card\n\nRedacted held-out data.\n", encoding="utf-8")
    return paths


def _write_promotion_eval_lineage(root: Path) -> dict[str, Path]:
    root.mkdir(exist_ok=True)
    example_root = ROOT / "examples" / "agentic_training" / "heldout_eval"
    for filename in (
        "baseline_suite_summary.json",
        "candidate_suite_summary.json",
        "heldout_manifest.json",
        "external_eval_raw_result.json",
        "external_eval_runner.json",
    ):
        shutil.copyfile(example_root / filename, root / filename)

    candidate_id = PROMOTION_CANDIDATE_ID
    heldout_path = root / "heldout_manifest.json"
    plan_path = root / "external_eval_plan.json"
    plan = build_external_eval_plan(
        adapters=["local_mock"],
        scenario_manifest=heldout_path,
        model_endpoint=candidate_id,
        model=candidate_id,
        allow_installed=True,
        output_base_dir=root,
    )
    write_external_eval_plan(plan, plan_path)

    execution_id = "loop-fixture-promotion-eval-001"
    runner_path = root / "external_eval_runner.json"
    runner = json.loads(runner_path.read_text(encoding="utf-8"))
    runner["execution_id"] = execution_id
    runner["model_id"] = candidate_id
    runner_path.write_text(json.dumps(runner, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    result_path = root / "external_eval_result.json"
    result = build_external_eval_result(
        plan_path=plan_path,
        heldout_manifest_path=heldout_path,
        raw_result_path=root / "external_eval_raw_result.json",
        runner_metadata_path=runner_path,
        adapter_id="local_mock",
        execution_id=execution_id,
        model_id=candidate_id,
        normalizer_id="hfr.local_mock.per_case_json",
        normalizer_version="1",
        raw_format="json",
        execution_status="completed",
        out_path=result_path,
        created_at="2026-07-10T00:01:00+00:00",
    )
    write_external_eval_result(result, result_path)

    eval_summary_path = root / "eval_summary.json"
    eval_summary = build_eval_summary(
        external_adapter_plan_specs=[f"local_mock={plan_path}"],
        external_adapter_result_specs=[f"local_mock={result_path}"],
        output_base_dir=root,
    )
    eval_summary_path.write_text(
        json.dumps(eval_summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "eval_summary": eval_summary_path,
        "external_eval_result": result_path,
    }


def _write_candidate_training_result(root: Path, candidate_id: str) -> Path:
    example_root = ROOT / "examples" / "agentic_training"
    for dirname in ("plans", "runtime_preflight", "trainer_outputs"):
        shutil.copytree(
            example_root / dirname,
            root / dirname,
            dirs_exist_ok=True,
        )
    for filename in ("agentic_training_flow.json", "trainer_consumer_plan.json"):
        shutil.copyfile(example_root / filename, root / filename)
    payload = _read_json_object(example_root / "completed_result.json")
    result_path = root / "agentic_training_result.json"
    payload["artifact_path"] = result_path.name
    payload["registry_update"]["target_model_id"] = candidate_id
    payload["registry_update"]["links"][0]["path"] = result_path.name
    return _write_source_json(result_path, payload)


def _bind_registry_entry_links(
    payload: dict[str, object], sources: dict[str, Path]
) -> None:
    links = payload["links"]
    if not isinstance(links, dict):
        raise AssertionError("model registry fixture links must be an object")
    for collection, source_path in sources.items():
        rows = links[collection]
        if not isinstance(rows, list) or not rows or not isinstance(rows[0], dict):
            raise AssertionError(f"model registry fixture link missing: {collection}")
        source_bytes = source_path.read_bytes()
        rows[0]["path"] = str(source_path.resolve())
        rows[0]["sha256"] = hashlib.sha256(source_bytes).hexdigest()
        rows[0]["size_bytes"] = len(source_bytes)


def _read_json_object(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise AssertionError(f"expected JSON object fixture: {path}")
    return payload


def _write_source_json(path: Path, payload: dict[str, object]) -> Path:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
