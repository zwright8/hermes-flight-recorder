#!/usr/bin/env python3
"""Build the immutable, sealed-blind Tau-3 competitive-agent v3 plan bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flightrecorder.tau3_competitive_v3 import PLAN_FILENAME, PLAN_SCHEMA_VERSION  # noqa: E402


MISSION_ID = "tau3-competitive-agent-v3"
RUBRIC_PATH = Path(".omx/specs/tau3-competitive-agent-v3-goal-rubric.md")
MISSION_PATH = Path(".omx/goals/autoresearch/tau3-competitive-agent-v3/mission.json")
PROTOCOL_PATH = Path("local/tau3/protocol-teacher-v1.json")
V2_BLOCKED_PATH = Path(".omx/goals/autoresearch/tau3-core-qlora-training/completion.json")
V2_CANDIDATE_C_TRAINING_PATH = Path(
    "local/tau3/candidate-attempts/policy-complete-v2/"
    "v2-candidate-c-r8-a16-l4-lr1e5-i100-s11200-ga1-prefix-eager-seed8675309/"
    "run/training_receipt.json"
)
V2_CANDIDATE_C_DEVELOPMENT_PATH = Path(
    "local/tau3/development-evals/policy-complete-v2/"
    "candidate-c-prefix-r8-l4-lr1e5-i100-no-thinking/manifest.json"
)
TOOL_CATALOG_PATH = Path("local/tau3/tool-schemas-official.json")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=ROOT)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    plan_path = build_plan_bundle(args.repository_root.resolve(), args.out.resolve())
    print(
        json.dumps(
            {
                "schema_version": "hfr.tau3_competitive_v3_plan_build.v1",
                "passed": True,
                "plan_path": str(plan_path),
                "plan_sha256": sha256_file(plan_path),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def build_plan_bundle(repository_root: Path, out: Path) -> Path:
    """Create a fresh plan bundle without reading sealed task payloads."""

    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output directory: {out}")
    out.mkdir(parents=True, exist_ok=True)

    mission = read_json(required_source(repository_root, MISSION_PATH))
    protocol = read_json(required_source(repository_root, PROTOCOL_PATH))
    topic = require_string(mission, "topic", MISSION_PATH)
    if mission.get("slug") != MISSION_ID:
        raise ValueError(f"{MISSION_PATH} slug must be {MISSION_ID}")

    refs = {
        "rubric": copy_ref(repository_root, RUBRIC_PATH, out, Path("evidence/rubric.md")),
        "protocol": copy_ref(repository_root, PROTOCOL_PATH, out, Path("evidence/protocol.json")),
        "v2_blocked": copy_ref(repository_root, V2_BLOCKED_PATH, out, Path("evidence/v2/blocked-verdict.json")),
        "v2_training": copy_ref(
            repository_root,
            V2_CANDIDATE_C_TRAINING_PATH,
            out,
            Path("evidence/v2/candidate-c-training.json"),
        ),
        "v2_development": copy_ref(
            repository_root,
            V2_CANDIDATE_C_DEVELOPMENT_PATH,
            out,
            Path("evidence/v2/candidate-c-development.json"),
        ),
        "tau_repository": write_ref(out, Path("evidence/tau-revision.json"), require_dict(protocol, "tau_revision")),
        "harness": write_ref(out, Path("evidence/harness.json"), require_dict(protocol, "harness_contract")),
        "tool_catalog": copy_ref(
            repository_root,
            TOOL_CATALOG_PATH,
            out,
            Path("evidence/ordered-tool-catalog.json"),
        ),
    }

    harness = require_dict(protocol, "harness_contract")
    model_freeze = require_dict(protocol, "model_freeze")
    base = require_dict(model_freeze, "base_model")
    comparators = require_list_of_dicts(model_freeze, "comparators")
    if len(comparators) != 2:
        raise ValueError("protocol model_freeze.comparators must contain exactly two records")
    evaluator_model_contract = build_evaluator_model_contract(protocol, model_freeze)

    refs.update(
        {
            "evaluator_models": write_ref(
                out,
                Path("evidence/evaluator-model-contract.json"),
                evaluator_model_contract,
            ),
            "tokenizer": write_ref(
                out,
                Path("evidence/tokenizer-chat-template.json"),
                {
                    "base": {
                        "model": base.get("name"),
                        "revision": base.get("revision"),
                        "tokenizer": base.get("tokenizer"),
                        "chat_template": base.get("chat_template"),
                    },
                    "comparators": [
                        {
                            "model": comparator.get("name"),
                            "revision": comparator.get("revision"),
                            "tokenizer": comparator.get("tokenizer"),
                            "chat_template": comparator.get("chat_template"),
                        }
                        for comparator in comparators
                    ],
                },
            ),
            "policy": write_ref(
                out,
                Path("evidence/policy-prompt-contract.json"),
                {
                    "system_prompt_sha256": harness.get("system_prompt_sha256"),
                    "domain_policy_sha256": {
                        domain: contract.get("policy_sha256")
                        for domain, contract in require_dict(harness, "domain_contracts").items()
                    },
                    "retention": "exact_system_prompt_and_domain_policy",
                },
            ),
            "grid": write_ref(
                out,
                Path("evidence/task-trial-seed-grid.json"),
                {
                    "domains": harness.get("domains"),
                    "seeds": require_dict(harness, "decoding").get("seeds"),
                    "split_hashes": require_dict(protocol, "tau_revision").get("split_hashes"),
                    "development_only_selection": True,
                    "sealed_payload_access_count": 0,
                },
            ),
            "decoding": write_ref(out, Path("evidence/decoding.json"), require_dict(harness, "decoding")),
            "retry": write_ref(
                out,
                Path("evidence/retry-policy.json"),
                {
                    "retry_policy": harness.get("retry_policy"),
                    "no_test_time_search": harness.get("no_test_time_search"),
                    "test_time_search": harness.get("test_time_search"),
                },
            ),
            "safety": write_ref(
                out,
                Path("evidence/safety-policy.json"),
                {
                    "stop_conditions": harness.get("stop_conditions"),
                    "turn_limit": harness.get("turn_limit"),
                    "candidate_selection_contract": protocol.get("candidate_selection_contract"),
                    "promotion_predicates": require_dict(protocol, "protocol_manifest").get("promotion_predicates"),
                },
            ),
        }
    )

    plan = {
        "schema_version": PLAN_SCHEMA_VERSION,
        "mission_contract": {
            "mission_id": MISSION_ID,
            "mission_statement_sha256": sha256_bytes(topic.encode("utf-8")),
            "rubric": refs["rubric"],
            "protocol": refs["protocol"],
        },
        "v2_negative_evidence": {
            "immutable": True,
            "rewrites_v2": False,
            "blocked_verdict": refs["v2_blocked"],
            "candidate_c": {
                "training_receipt": refs["v2_training"],
                "development_result": refs["v2_development"],
            },
        },
        "lineage": {
            "dataset_id": MISSION_ID,
            "version": "v3",
            "predecessor": "v2",
            "new_lineage": True,
        },
        "sealed_access": {"payload_access_count": 0, "materialized_sealed_fields": []},
        "models": {
            "base": {
                "model_id": base.get("name"),
                "parameter_b": base.get("parameters_billion"),
                "selected_by_sealed_blind_preflight": True,
            },
            "comparators": [
                {
                    "arm_id": f"comparator_{index}",
                    "model_id": comparator.get("name"),
                    "parameter_b": comparator.get("parameters_billion"),
                    "same_size_open": True,
                    "frozen": True,
                }
                for index, comparator in enumerate(comparators, start=1)
            ],
            "evaluator_models": refs["evaluator_models"],
        },
        "harness_contract": {
            "identical_for_all_arms": True,
            "text_mode": True,
            "frozen_before_training": True,
            "tau_repository": refs["tau_repository"],
            "harness": refs["harness"],
            "tokenizer_chat_template": refs["tokenizer"],
            "ordered_tool_catalog": refs["tool_catalog"],
            "policy_prompt": refs["policy"],
            "task_trial_seed_grid": refs["grid"],
            "decoding": refs["decoding"],
            "retry_policy": refs["retry"],
            "safety_policy": refs["safety"],
            "evaluator_model_contract": refs["evaluator_models"],
            "context_window": harness.get("context_window"),
        },
        "statistical_contract": {
            "procedure": "paired_bootstrap",
            "confidence_level": 0.95,
            "adapter_improvement_ci_excludes_zero": True,
            "same_size_noninferiority_margin": 0.05,
            "promotion_predicates_fail_closed": True,
        },
        "budget_contract": {
            "unattended_days": 7,
            "separate_infrastructure_attempt_budget": True,
            "separate_qualified_candidate_budget": True,
            "minimum_qualified_candidates": 2,
            "target_qualified_candidates_min": 3,
            "target_qualified_candidates_max": 4,
        },
        "dataset_contract": {
            "coverage_gates": {
                "all_domains_train_and_internal_validation": True,
                "min_train_targets_per_tool": 16,
                "min_internal_validation_targets_per_tool": 4,
                "min_train_argument_payloads_per_tool": 8,
                "min_internal_validation_argument_payloads_per_tool": 2,
                "all_required_behaviors_per_domain": True,
                "telecom_min_training_example_fraction": 0.25,
                "telecom_min_supervised_token_fraction": 0.25,
                "domain_supervised_token_min_fraction": 0.25,
                "domain_supervised_token_max_fraction": 0.40,
                "max_domain_canonical_target_duplication_fraction": 0.20,
                "split_hashes_disjoint": True,
            },
            "sampling_contract": {
                "deterministic_receipt_producing": True,
                "records_per_step_exposure": True,
                "replay_from_dataset_hash_config_seed": True,
            },
            "objective_contract": {
                "supervises_every_eligible_assistant_decision": True,
                "masks_prompt_tool_result_negative_user_private_reference_and_grader_tokens": True,
                "retains_parent_trajectory_and_decision_ordinal": True,
                "negative_actions_masked_with_safe_correction_only": True,
            },
        },
        "recipe_contract": {
            "minimum_rank": 16,
            "minimum_adapted_layers": 8,
            "minimum_effective_batch_examples": 4,
            "minimum_effective_epochs": 2,
            "token_loss_is_selection_metric": False,
            "candidate_c_stronger_search_space": True,
        },
        "development_contract": {
            "development_only_selection": True,
            "identical_harness": True,
            "no_comparator_specific_prompting": True,
            "qualification_gate": {
                "macro_pass1_min": 0.10,
                "per_domain_pass1_min": 0.05,
                "macro_base_improvement_min": 0.05,
            },
        },
        "publication_contract": {
            "competitive_claims_fail_closed": True,
            "sealed_payloads_public": False,
            "redaction_required": True,
        },
    }
    plan_path = out / PLAN_FILENAME
    write_json(plan_path, plan)
    return plan_path


def build_evaluator_model_contract(protocol: dict[str, Any], model_freeze: dict[str, Any]) -> dict[str, Any]:
    """Return a hash-bound user-simulator/reviewer contract from one pinned teacher."""

    teachers = require_list_of_dicts(model_freeze, "teachers")
    if len(teachers) != 1:
        raise ValueError("protocol model_freeze.teachers must contain exactly one pinned teacher record")
    teacher = teachers[0]
    teacher_model_id = require_string(teacher, "name", Path("model_freeze.teachers[0]"))
    teacher_revision = require_string(teacher, "revision", Path("model_freeze.teachers[0]"))
    teacher_license = require_string(teacher, "license", Path("model_freeze.teachers[0]"))
    local_identity_sha256 = require_sha_string(teacher, "local_identity_sha256")
    local_tree_sha256 = require_sha_string(teacher, "local_tree_sha256")
    local_identity_path = require_string(teacher, "local_identity_path", Path("model_freeze.teachers[0]"))
    local_path = require_string(teacher, "local_path", Path("model_freeze.teachers[0]"))
    role = require_string(teacher, "role", Path("model_freeze.teachers[0]"))
    if role != "teacher_generation_and_review_only":
        raise ValueError("teacher role must be teacher_generation_and_review_only")
    eligibility = require_dict(teacher, "pre_run_eligibility")
    if eligibility.get("comparator_eligible") is not False:
        raise ValueError("teacher pre_run_eligibility.comparator_eligible must be false")
    if eligibility.get("excluded_from_comparator_rule") is not True:
        raise ValueError("teacher must be excluded from the comparator rule")
    if eligibility.get("immutable_open_weights") is not True:
        raise ValueError("teacher must be pinned to immutable open weights")
    if eligibility.get("mlx_local_load_compatible") is not True:
        raise ValueError("teacher must be locally loadable by MLX")
    if not local_path.startswith("local/") or not local_identity_path.startswith("local/"):
        raise ValueError("teacher local_path and local_identity_path must be repository-local paths")
    if not has_matching_teacher_license(protocol, teacher_model_id, teacher_license):
        raise ValueError("protocol licenses must include the pinned teacher with matching license")

    teacher_identity = {
        "model_id": teacher_model_id,
        "revision": teacher_revision,
        "upstream": teacher.get("upstream"),
        "license": teacher_license,
        "quantization": teacher.get("quantization"),
        "tokenizer": teacher.get("tokenizer"),
        "chat_template": teacher.get("chat_template"),
        "local_identity_path": local_identity_path,
        "local_identity_sha256": local_identity_sha256,
        "local_path": local_path,
        "local_tree_sha256": local_tree_sha256,
        "local_file_count": teacher.get("local_file_count"),
        "model_card_url": teacher.get("model_card_url"),
        "role": role,
    }
    identity_sha256 = canonical_sha256(teacher_identity)
    role_contract = {
        "model_identity": teacher_identity,
        "model_identity_sha256": identity_sha256,
        "local_only": True,
        "network": False,
        "temperature": 0.0,
        "top_p": 1.0,
        "no_test_time_search": True,
        "comparator_specific_prompting": False,
    }
    return {
        "schema_version": "hfr.tau3_evaluator_model_contract.v1",
        "source": "local/tau3/protocol-teacher-v1.json",
        "teacher_policy": model_freeze.get("teacher_policy"),
        "teacher_record_sha256": canonical_sha256(teacher),
        "roles": {
            "user_simulator": {"role": "user_simulator", **role_contract},
            "reviewer": {"role": "reviewer", **role_contract},
        },
        "roles_share_exact_model": True,
        "identical_for_all_arms": True,
        "applies_to_arms": ["adapter", "base", "comparator_1", "comparator_2"],
        "no_comparator_specific_prompting": True,
        "excluded_from_comparator_claims": True,
        "excluded_from_gradient_data": True,
        "local_only": True,
        "network": False,
    }


def required_source(repository_root: Path, relative: Path) -> Path:
    path = repository_root / relative
    if not path.is_file():
        raise FileNotFoundError(f"required preregistration source is missing: {path}")
    return path


def copy_ref(repository_root: Path, source_relative: Path, out: Path, destination_relative: Path) -> dict[str, str]:
    source = required_source(repository_root, source_relative)
    destination = out / destination_relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    return {"path": destination_relative.as_posix(), "sha256": sha256_file(destination)}


def write_ref(out: Path, relative: Path, payload: Any) -> dict[str, str]:
    path = out / relative
    write_json(path, payload)
    return {"path": relative.as_posix(), "sha256": sha256_file(path)}


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def require_dict(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be an object")
    return value


def require_list_of_dicts(payload: dict[str, Any], key: str) -> list[dict[str, Any]]:
    value = payload.get(key)
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise ValueError(f"{key} must be a list of objects")
    return value


def require_sha_string(payload: dict[str, Any], key: str) -> str:
    value = require_string(payload, key, Path(key))
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{key} must be a lowercase sha256")
    return value


def require_string(payload: dict[str, Any], key: str, source: Path) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{source} {key} must be a non-empty string")
    return value


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return sha256_bytes(encoded)


def has_matching_teacher_license(protocol: dict[str, Any], teacher_model_id: str, teacher_license: str) -> bool:
    licenses = protocol.get("licenses")
    if not isinstance(licenses, list):
        return False
    matches = [
        item
        for item in licenses
        if isinstance(item, dict)
        and item.get("id") == teacher_model_id
        and item.get("license") == teacher_license
        and item.get("status") == "approved"
    ]
    return len(matches) == 1


if __name__ == "__main__":
    raise SystemExit(main())
