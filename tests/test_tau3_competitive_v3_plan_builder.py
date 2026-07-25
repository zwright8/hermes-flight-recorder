from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from flightrecorder.tau3_competitive_v3 import validate_tau3_competitive_v3_bundle
from scripts.build_tau3_competitive_v3_plan import (
    MISSION_ID,
    MISSION_PATH,
    PROTOCOL_PATH,
    RUBRIC_PATH,
    TOOL_CATALOG_PATH,
    V2_BLOCKED_PATH,
    V2_CANDIDATE_C_DEVELOPMENT_PATH,
    V2_CANDIDATE_C_TRAINING_PATH,
    build_evaluator_model_contract,
    build_plan_bundle,
    write_json,
)


class Tau3CompetitiveV3PlanBuilderTests(unittest.TestCase):
    def test_builder_binds_sources_and_passes_plan_validation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repository = Path(tmp) / "repository"
            out = Path(tmp) / "bundle"
            mission = {"slug": MISSION_ID, "topic": "mission statement"}
            protocol = minimal_protocol()
            write_json(repository / MISSION_PATH, mission)
            write_json(repository / PROTOCOL_PATH, protocol)
            write_text(repository / RUBRIC_PATH, "# rubric\n")
            write_json(
                repository / V2_BLOCKED_PATH,
                {
                    "slug": "tau3-core-qlora-training",
                    "verdict": "blocked",
                    "passed": False,
                },
            )
            write_json(repository / V2_CANDIDATE_C_TRAINING_PATH, {"terminal_status": "success"})
            write_json(repository / V2_CANDIDATE_C_DEVELOPMENT_PATH, {"macro_pass1": 0.0})
            write_json(repository / TOOL_CATALOG_PATH, {"tools": []})

            plan_path = build_plan_bundle(repository, out)
            result = validate_tau3_competitive_v3_bundle(out, stage="plan")
            plan = json.loads(plan_path.read_text(encoding="utf-8"))

            self.assertTrue(result["passed"], json.dumps(result, indent=2))
            self.assertEqual(plan["v2_negative_evidence"]["immutable"], True)
            self.assertEqual(plan["v2_negative_evidence"]["rewrites_v2"], False)
            self.assertEqual(plan["sealed_access"]["payload_access_count"], 0)
            self.assertNotIn("evidence_refs", plan)
            self.assertEqual(plan["models"]["evaluator_models"], plan["harness_contract"]["evaluator_model_contract"])
            evaluator_ref = plan["models"]["evaluator_models"]
            self.assertEqual(evaluator_ref["path"], "evidence/evaluator-model-contract.json")
            evaluator = json.loads((out / evaluator_ref["path"]).read_text(encoding="utf-8"))
            self.assertEqual(evaluator["schema_version"], "hfr.tau3_evaluator_model_contract.v1")
            self.assertEqual(evaluator["applies_to_arms"], ["adapter", "base", "comparator_1", "comparator_2"])
            self.assertTrue(evaluator["roles_share_exact_model"])
            self.assertTrue(evaluator["local_only"])
            self.assertFalse(evaluator["network"])
            self.assertTrue(evaluator["excluded_from_comparator_claims"])
            self.assertTrue(evaluator["excluded_from_gradient_data"])
            user_model = evaluator["roles"]["user_simulator"]["model_identity"]
            reviewer_model = evaluator["roles"]["reviewer"]["model_identity"]
            self.assertEqual(user_model, reviewer_model)
            self.assertEqual(user_model["model_id"], "mlx-community/Qwen3.6-35B-A3B-4bit")
            self.assertEqual(user_model["revision"], "c" * 40)
            self.assertEqual(user_model["local_identity_sha256"], "5" * 64)
            self.assertEqual(user_model["local_tree_sha256"], "6" * 64)

    def test_builder_refuses_nonempty_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repository = Path(tmp) / "repository"
            out = Path(tmp) / "bundle"
            out.mkdir(parents=True)
            write_text(out / "preserve.txt", "do not overwrite\n")

            with self.assertRaises(FileExistsError):
                build_plan_bundle(repository, out)

    def test_builder_requires_exactly_one_teacher_record(self) -> None:
        protocol = minimal_protocol()
        model_freeze = protocol["model_freeze"]

        model_freeze["teachers"] = []
        with self.assertRaisesRegex(ValueError, "exactly one pinned teacher"):
            build_evaluator_model_contract(protocol, model_freeze)

        model_freeze["teachers"] = [teacher_record(), teacher_record()]
        with self.assertRaisesRegex(ValueError, "exactly one pinned teacher"):
            build_evaluator_model_contract(protocol, model_freeze)

    def test_builder_rejects_unpinned_teacher_identity(self) -> None:
        protocol = minimal_protocol()
        protocol["model_freeze"]["teachers"][0].pop("local_tree_sha256")

        with self.assertRaisesRegex(ValueError, "local_tree_sha256"):
            build_evaluator_model_contract(protocol, protocol["model_freeze"])

    def test_builder_rejects_teacher_without_matching_license(self) -> None:
        protocol = minimal_protocol()
        protocol["licenses"] = []

        with self.assertRaisesRegex(ValueError, "licenses"):
            build_evaluator_model_contract(protocol, protocol["model_freeze"])


def minimal_protocol() -> dict:
    model = {
        "name": "base-8b",
        "parameters_billion": 8.0,
        "revision": "a" * 40,
        "tokenizer": "tokenizer@revision",
        "chat_template": "template",
    }
    return {
        "tau_revision": {"revision": "b" * 40, "split_hashes": {"train": "1", "development": "2", "sealed": "3"}},
        "harness_contract": {
            "context_window": 16384,
            "domains": ["airline", "retail", "telecom"],
            "decoding": {"temperature": 0.0, "top_p": 1.0, "max_output_tokens": 1024, "seeds": [101, 202]},
            "domain_contracts": {
                domain: {"policy_sha256": str(index) * 64}
                for index, domain in enumerate(("airline", "retail", "telecom"), start=1)
            },
            "system_prompt_sha256": "4" * 64,
            "retry_policy": "none",
            "no_test_time_search": True,
            "test_time_search": False,
            "stop_conditions": ["tau_terminal"],
            "turn_limit": 30,
        },
        "model_freeze": {
            "base_model": model,
            "comparators": [
                {**model, "name": "comparator-one", "parameters_billion": 8.2},
                {**model, "name": "comparator-two", "parameters_billion": 8.0},
            ],
            "teachers": [teacher_record()],
            "teacher_policy": (
                "Teachers are pinned for local generation and review evidence only. "
                "They are excluded from comparator claims."
            ),
        },
        "licenses": [
            {
                "id": "mlx-community/Qwen3.6-35B-A3B-4bit",
                "license": "Apache-2.0",
                "status": "approved",
                "training_allowed": True,
                "usage": "teacher_generation_and_review_only",
            }
        ],
        "candidate_selection_contract": {"development_only": True},
        "protocol_manifest": {"promotion_predicates": ["beat_base"]},
    }


def teacher_record() -> dict:
    return {
        "name": "mlx-community/Qwen3.6-35B-A3B-4bit",
        "revision": "c" * 40,
        "upstream": {"name": "Qwen/Qwen3.6-35B-A3B", "revision": "d" * 40},
        "license": "Apache-2.0",
        "quantization": "mlx-4bit",
        "tokenizer": "Qwen/Qwen3.6-35B-A3B@" + "d" * 40,
        "chat_template": "qwen3.6-tool-use-chat-template",
        "local_identity_path": "local/tau3/identities/teacher.json",
        "local_identity_sha256": "5" * 64,
        "local_path": "local/tau3/models/teacher",
        "local_tree_sha256": "6" * 64,
        "local_file_count": 17,
        "model_card_url": "https://huggingface.co/mlx-community/Qwen3.6-35B-A3B-4bit",
        "role": "teacher_generation_and_review_only",
        "pre_run_eligibility": {
            "comparator_eligible": False,
            "excluded_from_comparator_rule": True,
            "immutable_open_weights": True,
            "mlx_local_load_compatible": True,
        },
    }


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")
