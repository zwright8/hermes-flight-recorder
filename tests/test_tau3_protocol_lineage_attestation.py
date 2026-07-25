from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_tau3_protocol_lineage_attestation.py"
SPEC = importlib.util.spec_from_file_location("tau3_protocol_lineage_attestation_script", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
lineage_script = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(lineage_script)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


class Tau3ProtocolLineageAttestationTests(unittest.TestCase):
    def test_success_writes_create_once_attestation_and_cli_replays(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parent_bundle, child_path, corpus_path, mixture_path, corpus_sha256 = make_fixture(root)
            out_a = root / "attestation-a.json"
            out_b = root / "attestation-b.json"

            attestation = lineage_script.build_tau3_protocol_lineage_attestation(
                parent_bundle=parent_bundle,
                child_protocol=child_path,
                corpus=corpus_path,
                mixture=mixture_path,
                out=out_a,
                created_at="2026-07-24T00:00:00+00:00",
            )
            completed = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--parent-bundle",
                    str(parent_bundle),
                    "--child-protocol",
                    str(child_path),
                    "--corpus",
                    str(corpus_path),
                    "--mixture",
                    str(mixture_path),
                    "--out",
                    str(out_b),
                    "--created-at",
                    "2026-07-24T00:00:00+00:00",
                ],
                check=False,
                capture_output=True,
                text=True,
                cwd=ROOT,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            cli_attestation = json.loads(completed.stdout)
            self.assertEqual(attestation["attestation_sha256"], cli_attestation["attestation_sha256"])
            self.assertEqual(_read_json(out_a), _read_json(out_b))
            self.assertEqual(out_a.stat().st_mode & 0o777, 0o600)
            self.assertTrue(all(check["passed"] for check in attestation["checks"]))
            self.assertEqual(attestation["bindings"]["corpus"]["sha256"], corpus_sha256)
            self.assertFalse(Path(attestation["bindings"]["corpus"]["path"]).is_absolute())
            self.assertEqual(attestation["bindings"]["mixture"]["sha256"], _sha256(mixture_path))
            self.assertFalse(Path(attestation["bindings"]["mixture"]["path"]).is_absolute())
            self.assertEqual(
                attestation["allowed_delta"]["model_freeze.teachers"],
                "exactly one pinned teacher with role teacher_generation_and_review_only and comparator_eligible=false",
            )
            frozen_hashes = {row["field"]: row for row in attestation["frozen_field_hashes"]}
            self.assertEqual(
                frozen_hashes["harness_contract"]["parent_sha256"],
                frozen_hashes["harness_contract"]["child_sha256"],
            )

    def test_output_overwrite_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parent_bundle, child_path, corpus_path, mixture_path, _corpus_sha256 = make_fixture(root)
            out = root / "attestation.json"
            out.write_text("{}", encoding="utf-8")

            with self.assertRaisesRegex(lineage_script.Tau3ProtocolLineageError, "already exists"):
                lineage_script.build_tau3_protocol_lineage_attestation(
                    parent_bundle=parent_bundle,
                    child_protocol=child_path,
                    corpus=corpus_path,
                    mixture=mixture_path,
                    out=out,
                )

    def test_derived_corpus_hash_mismatch_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parent_bundle, child_path, corpus_path, mixture_path, _corpus_sha256 = make_fixture(root)
            corpus_path.write_text("tampered\n", encoding="utf-8")

            with self.assertRaisesRegex(lineage_script.Tau3ProtocolLineageError, "derived corpus file sha256"):
                lineage_script.build_tau3_protocol_lineage_attestation(
                    parent_bundle=parent_bundle,
                    child_protocol=child_path,
                    corpus=corpus_path,
                    mixture=mixture_path,
                    out=root / "attestation.json",
                )

    def test_frozen_harness_drift_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parent_bundle, child_path, corpus_path, mixture_path, _corpus_sha256 = make_fixture(root)
            child = _read_json(child_path)
            child["harness_contract"]["decoding"]["temperature"] = 0.2
            _write_json(child_path, child)

            with self.assertRaisesRegex(lineage_script.Tau3ProtocolLineageError, "harness_contract"):
                lineage_script.build_tau3_protocol_lineage_attestation(
                    parent_bundle=parent_bundle,
                    child_protocol=child_path,
                    corpus=corpus_path,
                    mixture=mixture_path,
                    out=root / "attestation.json",
                )

    def test_teacher_policy_drift_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parent_bundle, child_path, corpus_path, mixture_path, _corpus_sha256 = make_fixture(root)
            child = _read_json(child_path)
            child["model_freeze"]["teacher_policy"] = "Teachers can be comparators."
            _write_json(child_path, child)

            with self.assertRaisesRegex(lineage_script.Tau3ProtocolLineageError, "teacher_policy"):
                lineage_script.build_tau3_protocol_lineage_attestation(
                    parent_bundle=parent_bundle,
                    child_protocol=child_path,
                    corpus=corpus_path,
                    mixture=mixture_path,
                    out=root / "attestation.json",
                )


def make_fixture(root: Path) -> tuple[Path, Path, Path, Path, str]:
    input_dir = root / "inputs"
    input_dir.mkdir()
    corpus_path = input_dir / "corpus.jsonl"
    mixture_path = input_dir / "mixture.json"
    corpus_path.write_text('{"row":1}\n', encoding="utf-8")
    mixture_path.write_text('{"mixture":true}\n', encoding="utf-8")
    corpus_sha256 = _sha256(corpus_path)
    parent = parent_protocol(corpus_sha256)
    child = child_protocol(parent)
    parent_bundle = root / "bundle"
    child_path = root / "child-protocol.json"
    _write_parent_bundle(parent_bundle, parent)
    _write_json(child_path, child)
    return parent_bundle, child_path, corpus_path, mixture_path, corpus_sha256


def parent_protocol(corpus_sha256: str) -> dict[str, Any]:
    return {
        "schema_version": "hfr.tau3_protocol_config.v1",
        "protocol_manifest": {
            "schema_version": "hfr.tau3_protocol_manifest.v1",
            "title": "Tau-3 fixture",
            "claim_scope": "best frozen eligible 7-9B open model under the fixed harness",
            "domains": ["airline", "retail", "telecom"],
            "primary_metric": "macro_pass_1",
            "paired_confidence_procedure": "domain_stratified_paired_bootstrap_95pct",
            "promotion_predicates": ["beat_frozen_strongest_comparator"],
            "secondary_metrics": ["per_domain_pass_1"],
        },
        "tau_revision": {"schema_version": "hfr.tau3_revision.v1", "revision": "1" * 40},
        "split_manifest": {
            "schema_version": "hfr.tau3_split_manifest.v1",
            "domains": ["airline", "retail", "telecom"],
            "training_captures": {
                "local_path": "local/tau3/training_captures.jsonl",
                "sha256": corpus_sha256,
                "row_count": 192,
                "admitted_count": 74,
                "rejected_count": 118,
                "trajectory_ids_sha256": "c" * 64,
            },
        },
        "harness_contract": {
            "schema_version": "hfr.tau3_harness_contract.v1",
            "fixed": True,
            "domains": ["airline", "retail", "telecom"],
            "decoding": {"temperature": 0.0, "top_p": 1.0, "max_output_tokens": 1024},
            "turn_limit": 30,
            "test_time_search": False,
        },
        "model_freeze": {
            "schema_version": "hfr.tau3_model_freeze.v1",
            "selection_rule": "official_upstream_public_ungated_immutable_open_weights_dense_7_to_9b_total_apache_2_pinned_mlx_conversion",
            "benchmark_superiority_claimed": False,
            "base_model": {"name": "mlx-community/Qwen3.5-9B-4bit", "revision": "2" * 40, "local_path": "local/base"},
            "comparators": [
                {"name": "mlx-community/Qwen3-8B-4bit", "revision": "3" * 40},
                {"name": "mlx-community/granite-3.3-8b-instruct-4bit", "revision": "4" * 40},
            ],
            "teachers": [],
        },
        "budget": {"schema_version": "hfr.tau3_budget.v1", "network": False, "local_only": True},
        "sealed_manifest": {
            "schema_version": "hfr.tau3_sealed_manifest.v1",
            "access_count": 0,
            "manifest_sha256": "d" * 64,
            "leakage_blocking_hashes": ["e" * 64],
            "prompt_template_hashes": ["f" * 64],
            "quarantined_at": "2026-07-22T00:00:00+00:00",
        },
        "mlx_qlora_plan": {
            "schema_version": "hfr.tau3_mlx_qlora_plan.v1",
            "passed": True,
            "network": False,
            "command_argv": ["python", "-m", "mlx_lm", "lora", "--train"],
        },
        "recipe_space": {"schema_version": "hfr.tau3_recipe_space.v1", "development_only": True, "sealed_used": False},
        "candidate_selection_contract": {
            "schema_version": "hfr.tau3_candidate_selection.v1",
            "passed": True,
            "development_only": True,
            "sealed_used": False,
        },
        "contamination_attestation": {
            "passed": True,
            "unresolved_leakage": False,
            "leakage_found": False,
            "checks": {
                "exact_duplicate": "passed",
                "near_duplicate": "passed",
                "task_template_overlap": "passed",
                "tool_sequence_overlap": "passed",
                "state_transition_overlap": "passed",
            },
        },
        "redaction_attestation": {
            "passed": True,
            "secrets_found": False,
            "unredacted_sensitive_data": False,
            "reviewed_sources": ["train_tasks", "development_tasks", "captures"],
        },
        "licenses": [
            {"id": "tau2-bench", "license": "MIT", "status": "approved", "training_allowed": True},
            {"id": "mlx-community/Qwen3.5-9B-4bit", "license": "Apache-2.0", "status": "approved", "training_allowed": True},
        ],
        "environment_manifest": {
            "schema_version": "hfr.tau3_environment.v1",
            "network_allowed": False,
            "device_identifiers_recorded": False,
        },
    }


def child_protocol(parent: dict[str, Any]) -> dict[str, Any]:
    child = json.loads(json.dumps(parent))
    child["split_manifest"]["training_captures"]["local_path"] = "local/tau3/captures-v1/captures.jsonl"
    child["sealed_manifest"]["quarantined_at"] = "2026-07-23T00:00:00+00:00"
    teacher = {
        "name": "mlx-community/Qwen3.6-35B-A3B-4bit",
        "revision": "38740b847e4cb78f352aba30aa41c76e08e6eb46",
        "upstream": {"name": "Qwen/Qwen3.6-35B-A3B", "revision": "7c787ca72eef6f25ba9a43a73219a461bf0b304d"},
        "architecture": "35B mixture-of-experts transformer",
        "parameters_billion": 35.0,
        "license": "Apache-2.0",
        "quantization": "mlx-4bit",
        "tokenizer": "Qwen/Qwen3.6-35B-A3B@38740b847e4cb78f352aba30aa41c76e08e6eb46",
        "chat_template": "qwen3.6-tool-use-chat-template",
        "role": lineage_script.TEACHER_ROLE,
        "local_path": "local/tau3/models/teacher-qwen3.6-35b-a3b",
        "local_identity_path": "local/tau3/identities/teacher-qwen3.6-35b-a3b.json",
        "local_identity_sha256": "9" * 64,
        "local_tree_sha256": "8" * 64,
        "pre_run_eligibility": {
            "role": lineage_script.TEACHER_ROLE,
            "comparator_eligible": False,
            "excluded_from_comparator_rule": True,
            "immutable_open_weights": True,
            "license": "Apache-2.0",
        },
    }
    child["model_freeze"]["teachers"] = [teacher]
    child["model_freeze"]["teacher_policy"] = (
        "Teachers are pinned for local generation and review evidence only. "
        "They are excluded from the 7-9B dense comparator eligibility rule and from benchmark superiority claims."
    )
    child["licenses"].append({
        "id": "mlx-community/Qwen3.6-35B-A3B-4bit",
        "license": "Apache-2.0",
        "status": "approved",
        "training_allowed": True,
        "usage": lineage_script.TEACHER_ROLE,
    })
    return child


def _write_parent_bundle(bundle: Path, protocol: dict[str, Any]) -> None:
    artifact_payloads = {
        "protocol_manifest": {
            **protocol["protocol_manifest"],
            "created_at": "2026-07-22T00:00:00+00:00",
            "environment": protocol["environment_manifest"],
            "frozen": True,
            "lineage_rule": "Any contract change requires a new bundle and protocol signature.",
            "signature": "0" * 64,
            "signature_algorithm": "sha256-canonical-json-content-seal",
            "signed": True,
        },
        "tau_revision": protocol["tau_revision"],
        "split_manifest": lineage_script._public_contract(protocol["split_manifest"]),
        "harness_contract": protocol["harness_contract"],
        "model_freeze": lineage_script._public_contract(protocol["model_freeze"]),
        "budget": protocol["budget"],
        "sealed_manifest": protocol["sealed_manifest"],
        "mlx_qlora_plan": {**protocol["mlx_qlora_plan"], "tokenizer_compatibility": {"passed": True}},
        "recipe_space": protocol["recipe_space"],
        "candidate_selection_contract": protocol["candidate_selection_contract"],
        "contamination_report": {
            "schema_version": "hfr.tau3_contamination_report.v1",
            "passed": True,
            "attestation": protocol["contamination_attestation"],
        },
        "redaction_report": {
            "schema_version": "hfr.tau3_redaction_report.v1",
            "passed": True,
            "attestation": protocol["redaction_attestation"],
        },
        "license_report": {
            "schema_version": "hfr.tau3_license_report.v1",
            "passed": True,
            "sources": protocol["licenses"],
        },
    }
    artifacts = []
    for role, parts in lineage_script.PARENT_PROTOCOL_ROLES.items():
        rel_path = Path(*parts)
        path = bundle / rel_path
        _write_json(path, artifact_payloads[role])
        artifacts.append({"role": role, "path": str(rel_path), "sha256": _sha256(path), "size": path.stat().st_size})
    _write_json(bundle / "manifest.json", {
        "schema_version": "hfr.tau3_training_bundle.v1",
        "ready_for_training": True,
        "bundle_mode": "production",
        "artifacts": artifacts,
    })


if __name__ == "__main__":
    unittest.main()
