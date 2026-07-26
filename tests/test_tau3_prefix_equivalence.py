from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from flightrecorder.schema_registry import check_schema_contract
from flightrecorder.tau3_prefix_equivalence import (
    REQUIRED_EQUIVALENCE_PROBE_FAMILIES,
    TAU3_PREFIX_EQUIVALENCE_SCHEMA_VERSION,
    validate_tau3_prefix_equivalence,
)


def _sha(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def equivalence_fixture() -> dict[str, object]:
    row_hashes = [f"{index:064x}" for index in range(1, 7)]
    target_rows = [
        {
            "row_sha256": row_sha256,
            "prompt_offset": 100 + index,
            "target_start": 100 + index,
            "target_end": 164 + index,
            "supervised_token_count": 64,
            "loss_mask_sha256": f"{100 + index:064x}",
            "target_tokens_sha256": f"{200 + index:064x}",
        }
        for index, row_sha256 in enumerate(row_hashes)
    ]
    boundaries = [
        {
            "row_sha256": row["row_sha256"],
            "prompt_offset": row["prompt_offset"],
            "target_start": row["target_start"],
            "target_end": row["target_end"],
            "supervised_token_count": row["supervised_token_count"],
        }
        for row in target_rows
    ]
    target = {
        "sample_row_count": len(row_hashes),
        "supervised_token_count": 384,
        "target_boundaries_sha256": _sha(boundaries),
        "loss_mask_sha256": _sha(
            [
                {
                    "row_sha256": row["row_sha256"],
                    "loss_mask_sha256": row["loss_mask_sha256"],
                }
                for row in target_rows
            ]
        ),
        "target_tokens_sha256": _sha(
            [
                {
                    "row_sha256": row["row_sha256"],
                    "target_tokens_sha256": row["target_tokens_sha256"],
                }
                for row in target_rows
            ]
        ),
        "rows": target_rows,
    }
    modules = [
        "model.layers.0.self_attn.q_proj.lora_a",
        "model.layers.0.self_attn.q_proj.lora_b",
    ]
    family_results = {}
    for family in REQUIRED_EQUIVALENCE_PROBE_FAMILIES:
        trials = [
            {
                "probe_id": f"{family}-{index}",
                "full_gradient_passed": True,
                "detached_prefix_passed": True,
            }
            for index in range(4)
        ]
        family_results[family] = {
            "trial_count": len(trials),
            "full_gradient_pass_rate": 1.0,
            "detached_prefix_pass_rate": 1.0,
            "trials": trials,
        }
    full_gradient_modules = [
        {"name": modules[0], "l2_norm": 0.45},
        {"name": modules[1], "l2_norm": 0.6},
    ]
    prefix_modules = [
        {"name": modules[0], "l2_norm": 0.3},
        {"name": modules[1], "l2_norm": 0.4},
    ]
    full_replays = [
        {
            "losses": [1.5, 1.25, 1.0],
            "peak_memory_bytes": 12_000_000,
            "numerical_failure_count": 0,
        },
        {
            "losses": [1.5, 1.25, 1.0],
            "peak_memory_bytes": 11_900_000,
            "numerical_failure_count": 0,
        },
    ]
    prefix_replays = [
        {
            "losses": [1.52, 1.27, 1.02],
            "peak_memory_bytes": 8_000_000,
            "numerical_failure_count": 0,
        },
        {
            "losses": [1.52, 1.27, 1.02],
            "peak_memory_bytes": 7_950_000,
            "numerical_failure_count": 0,
        },
    ]
    return {
        "schema_version": TAU3_PREFIX_EQUIVALENCE_SCHEMA_VERSION,
        "passed": True,
        "method": {
            "reference": "standard_full_gradient_mlx_lora",
            "candidate": "detached_complete_prompt_cache_mlx_lora",
            "max_material_probe_drop": 0.05,
            "max_loss_replay_delta": 0.0001,
        },
        "bindings": {
            "dataset_file_sha256": "1" * 64,
            "protocol_file_sha256": "2" * 64,
            "model_identity_file_sha256": "3" * 64,
            "recipe": {
                "rank": 16,
                "scale": 32.0,
                "learning_rate": 1e-5,
                "num_layers": 8,
                "max_seq_length": 16384,
                "batch_size": 1,
                "grad_accumulation": 4,
                "mask_prompt": True,
                "allowed_seeds": [101, 303],
            },
        },
        "sample": {
            "row_count": len(row_hashes),
            "row_hashes": row_hashes,
            "row_hashes_sha256": _sha(row_hashes),
            "domains": ["airline", "retail", "telecom"],
            "probe_families": list(REQUIRED_EQUIVALENCE_PROBE_FAMILIES),
            "stratified": True,
        },
        "target_accounting": {
            "full_gradient": copy.deepcopy(target),
            "detached_prefix": copy.deepcopy(target),
        },
        "gradient_evidence": {
            "intended_modules": modules,
            "intended_modules_sha256": _sha(modules),
            "full_gradient": {
                "nonzero_module_count": len(modules),
                "gradient_l2_norm": 0.75,
                "finite": True,
                "modules": full_gradient_modules,
            },
            "detached_prefix": {
                "nonzero_module_count": len(modules),
                "gradient_l2_norm": 0.5,
                "finite": True,
                "modules": prefix_modules,
            },
        },
        "behavior_probes": {
            "required_families": list(REQUIRED_EQUIVALENCE_PROBE_FAMILIES),
            "family_results": family_results,
        },
        "stability": {
            "full_gradient": {
                "replay_count": 2,
                "loss_count": 6,
                "finite_loss_count": 6,
                "numerical_failure_count": 0,
                "peak_memory_bytes": 12_000_000,
                "loss_replay_max_abs_delta": 0.0,
                "replay_sha256": _sha(full_replays),
                "replays": full_replays,
            },
            "detached_prefix": {
                "replay_count": 2,
                "loss_count": 6,
                "finite_loss_count": 6,
                "numerical_failure_count": 0,
                "peak_memory_bytes": 8_000_000,
                "loss_replay_max_abs_delta": 0.0,
                "replay_sha256": _sha(prefix_replays),
                "replays": prefix_replays,
            },
        },
        "failures": [],
    }


class Tau3PrefixEquivalenceTests(unittest.TestCase):
    def test_valid_artifact_passes_schema_and_semantic_replay(self) -> None:
        artifact = equivalence_fixture()

        schema = check_schema_contract(
            artifact,
            name_or_id="tau3_prefix_equivalence",
        )
        validation = validate_tau3_prefix_equivalence(artifact)

        self.assertTrue(schema["passed"], schema["errors"])
        self.assertTrue(validation["passed"], validation["errors"])

    def test_target_accounting_mismatch_fails_closed(self) -> None:
        artifact = equivalence_fixture()
        artifact["target_accounting"]["detached_prefix"][  # type: ignore[index]
            "target_boundaries_sha256"
        ] = "0" * 64

        validation = validate_tau3_prefix_equivalence(artifact)

        self.assertFalse(validation["passed"])
        self.assertIn("target accounting", json.dumps(validation))

    def test_material_probe_degradation_fails_closed(self) -> None:
        artifact = equivalence_fixture()
        recovery = artifact["behavior_probes"]["family_results"]["recovery"]  # type: ignore[index]
        recovery["trials"][0]["detached_prefix_passed"] = False
        recovery["detached_prefix_pass_rate"] = 0.75

        validation = validate_tau3_prefix_equivalence(artifact)

        self.assertFalse(validation["passed"])
        self.assertIn("material degradation", json.dumps(validation))

    def test_expected_launch_bindings_must_match(self) -> None:
        artifact = equivalence_fixture()

        validation = validate_tau3_prefix_equivalence(
            artifact,
            expected_bindings={
                "dataset_file_sha256": "f" * 64,
                "protocol_file_sha256": "2" * 64,
                "model_identity_file_sha256": "3" * 64,
                "recipe": {
                    "rank": 16,
                    "scale": 32.0,
                    "learning_rate": 1e-5,
                    "num_layers": 8,
                    "max_seq_length": 16384,
                    "batch_size": 1,
                    "grad_accumulation": 4,
                    "mask_prompt": True,
                    "seed": 101,
                },
            },
        )

        self.assertFalse(validation["passed"])
        self.assertIn("dataset_file_sha256", json.dumps(validation))

    def test_file_hash_tampering_is_detected_by_caller_replay(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "equivalence.json"
            path.write_text(
                json.dumps(equivalence_fixture(), sort_keys=True) + "\n",
                encoding="utf-8",
            )

            validation = validate_tau3_prefix_equivalence(path)

            self.assertTrue(validation["passed"], validation["errors"])
            self.assertEqual(validation["artifact"]["sha256"], hashlib.sha256(path.read_bytes()).hexdigest())


if __name__ == "__main__":
    unittest.main()
