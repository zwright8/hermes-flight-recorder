from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from flightrecorder.schema_registry import check_schema_contract
from flightrecorder.tau3_behavior_probes import (
    EndpointConfig,
    build_tau3_behavior_probes,
)
from flightrecorder.mlx_prefix_equivalence_smoke import (
    PrefixEquivalenceSmokeError,
    _parse_smoke_args,
    supervised_target_preserving_prompt_tail,
    target_accounting_row,
)
from flightrecorder.tau3_prefix_equivalence_sample import (
    SAMPLE_STRATA,
    build_tau3_prefix_equivalence_sample,
)
from flightrecorder.tau3_prefix_equivalence import (
    REQUIRED_EQUIVALENCE_PROBE_FAMILIES,
    TAU3_PREFIX_EQUIVALENCE_SCHEMA_VERSION,
    build_tau3_paired_behavior_trials,
    build_tau3_prefix_equivalence,
    validate_tau3_prefix_equivalence,
)
from tests.test_tau3_behavior_probes import (
    _bindings as behavior_bindings,
    _mock_openai_server,
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
            "source_sample_file_sha256": "4" * 64,
            "derivation": {
                "method": (
                    "supervised_target_preserving_prompt_tail_v1"
                ),
                "prompt_tail_token_limit": 512,
                "proxy_only": True,
                "candidate_training_uses_full_prompt": True,
            },
        },
        "target_accounting": {
            "full_gradient": copy.deepcopy(target),
            "detached_prefix": copy.deepcopy(target),
        },
        "gradient_evidence": {
            "intended_modules": modules,
            "intended_modules_sha256": _sha(modules),
            "full_gradient": {
                "evidence_kind": (
                    "pre_optimizer_accumulated_gradient_l2"
                ),
                "nonzero_module_count": len(modules),
                "gradient_l2_norm": 0.75,
                "finite": True,
                "modules": full_gradient_modules,
            },
            "detached_prefix": {
                "evidence_kind": "per_microbatch_gradient_l2",
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
    def test_paired_behavior_trials_replay_real_probe_bundles(self) -> None:
        with _mock_openai_server() as server:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                full_bindings = behavior_bindings()
                prefix_bindings = behavior_bindings()
                prefix_bindings["training_receipt_sha256"] = "6" * 64
                prefix_bindings["adapter_tree_sha256"] = "7" * 64
                endpoint = EndpointConfig(
                    base_url=server.base_url,
                    model="local-agent",
                )
                build_tau3_behavior_probes(
                    root / "full",
                    endpoint=endpoint,
                    bindings=full_bindings,
                )
                build_tau3_behavior_probes(
                    root / "prefix",
                    endpoint=endpoint,
                    bindings=prefix_bindings,
                )

                artifact = build_tau3_paired_behavior_trials(
                    full_gradient_probe_path=root / "full",
                    detached_prefix_probe_path=root / "prefix",
                )

                self.assertEqual(artifact["trial_count"], 7)
                self.assertEqual(
                    {trial["family"] for trial in artifact["trials"]},
                    set(REQUIRED_EQUIVALENCE_PROBE_FAMILIES),
                )
                self.assertTrue(
                    all(
                        trial["full_gradient_passed"]
                        and trial["detached_prefix_passed"]
                        for trial in artifact["trials"]
                    )
                )

    def test_equivalence_sample_selects_one_deterministic_row_per_stratum(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "train.jsonl"
            rows = [
                {
                    "messages": [
                        {"role": "user", "content": f"{domain}-{behavior}"},
                        {"role": "assistant", "content": "ok"},
                    ],
                    "metadata": {"domain": domain, "behavior": behavior},
                    "tools": [],
                }
                for domain, behavior in SAMPLE_STRATA
                for _ in range(2)
            ]
            source.write_text(
                "".join(
                    json.dumps(row, sort_keys=True) + "\n" for row in rows
                ),
                encoding="utf-8",
            )

            manifest = build_tau3_prefix_equivalence_sample(
                source,
                root / "sample",
            )

            self.assertEqual(manifest["row_count"], len(SAMPLE_STRATA))
            self.assertFalse(manifest["candidate_eligible"])
            selected = [
                json.loads(line)
                for line in (root / "sample" / "train.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual(
                [
                    (
                        row["metadata"]["domain"],
                        row["metadata"]["behavior"],
                    )
                    for row in selected
                ],
                list(SAMPLE_STRATA),
            )

    def test_smoke_target_accounting_preserves_first_target_boundary(self) -> None:
        row = target_accounting_row(
            tokens=[10, 11, 12, 13, 14],
            prompt_offset=3,
            row_sha256="a" * 64,
        )

        self.assertEqual(row["prompt_offset"], 3)
        self.assertEqual(row["target_start"], 3)
        self.assertEqual(row["target_end"], 5)
        self.assertEqual(row["supervised_token_count"], 2)

    def test_smoke_target_accounting_rejects_empty_target(self) -> None:
        with self.assertRaises(PrefixEquivalenceSmokeError):
            target_accounting_row(
                tokens=[10, 11, 12],
                prompt_offset=3,
                row_sha256="a" * 64,
            )

    def test_prompt_tail_proxy_preserves_every_supervised_token(self) -> None:
        tokens, prompt_offset = supervised_target_preserving_prompt_tail(
            list(range(20)),
            prompt_offset=16,
            prompt_tail_token_limit=5,
        )

        self.assertEqual(tokens, list(range(11, 20)))
        self.assertEqual(prompt_offset, 5)
        self.assertEqual(tokens[prompt_offset:], list(range(16, 20)))

    def test_smoke_parser_strips_only_measurement_arguments(self) -> None:
        args, passthrough = _parse_smoke_args(
            [
                "--equivalence-arm",
                "full_gradient",
                "--measurement-out",
                "measurement.json",
                "--protocol",
                "protocol.json",
                "--model-identity",
                "identity.json",
                "--candidate-dataset",
                "candidate.jsonl",
                "--prompt-tail-token-limit",
                "512",
                "--compiled-full-gradient",
                "--standard-full-gradient",
                "--sample-domains",
                "airline,retail,telecom",
                "--sample-probe-families",
                "tool_choice,clarification,recovery,stopping,state_transition",
                "--exposure-dataset",
                "train.jsonl",
                "--exposure-receipt",
                "receipt.json",
                "--exposure-ledger",
                "ledger.jsonl",
                "--batch-size",
                "1",
                "--grad-accumulation-steps",
                "4",
                "--iters",
                "8",
                "--model",
                "base-model",
            ]
        )

        self.assertEqual(args.equivalence_arm, "full_gradient")
        self.assertTrue(args.compiled_full_gradient)
        self.assertTrue(args.standard_full_gradient)
        self.assertEqual(args.model_identity, Path("identity.json"))
        self.assertEqual(args.candidate_dataset, Path("candidate.jsonl"))
        self.assertEqual(args.prompt_tail_token_limit, 512)
        self.assertNotIn("--measurement-out", passthrough)
        self.assertNotIn("--protocol", passthrough)
        self.assertNotIn("--compiled-full-gradient", passthrough)
        self.assertNotIn("--standard-full-gradient", passthrough)
        self.assertNotIn("--candidate-dataset", passthrough)
        self.assertNotIn("--prompt-tail-token-limit", passthrough)
        self.assertIn("--batch-size", passthrough)
        self.assertIn("--iters", passthrough)
        self.assertIn("--model", passthrough)

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

    def test_builder_replays_raw_arm_measurements(self) -> None:
        fixture = equivalence_fixture()
        bindings = fixture["bindings"]
        target_rows = fixture["target_accounting"]["full_gradient"]["rows"]  # type: ignore[index]
        behavior_trials = [
            {"family": family, **trial}
            for family, result in fixture["behavior_probes"]["family_results"].items()  # type: ignore[index]
            for trial in result["trials"]
        ]
        full_modules = fixture["gradient_evidence"]["full_gradient"]["modules"]  # type: ignore[index]
        prefix_modules = fixture["gradient_evidence"]["detached_prefix"]["modules"]  # type: ignore[index]
        full_replays = fixture["stability"]["full_gradient"]["replays"]  # type: ignore[index]
        prefix_replays = fixture["stability"]["detached_prefix"]["replays"]  # type: ignore[index]

        def measurement(
            arm: str,
            modules: object,
            replay: dict[str, object],
            run_id: str,
        ) -> dict[str, object]:
            return {
                "schema_version": "hfr.tau3_prefix_equivalence_run.v1",
                "arm": arm,
                "run_id": run_id,
                "bindings": {
                    "dataset_file_sha256": bindings["dataset_file_sha256"],
                    "protocol_file_sha256": bindings["protocol_file_sha256"],
                    "model_identity_file_sha256": bindings[
                        "model_identity_file_sha256"
                    ],
                    "recipe": {
                        key: value
                        for key, value in bindings["recipe"].items()
                        if key != "allowed_seeds"
                    }
                    | {"seed": 101},
                },
                "sample": {
                    "domains": ["airline", "retail", "telecom"],
                    "probe_families": list(
                        REQUIRED_EQUIVALENCE_PROBE_FAMILIES
                    ),
                    "stratified": True,
                    "source_sample_file_sha256": fixture["sample"][
                        "source_sample_file_sha256"
                    ],
                    "derivation": copy.deepcopy(
                        fixture["sample"]["derivation"]
                    ),
                },
                "execution": {
                    "gradient_evidence_kind": (
                        "pre_optimizer_accumulated_gradient_l2"
                        if arm == "full_gradient"
                        else "per_microbatch_gradient_l2"
                    )
                },
                "target_rows": copy.deepcopy(target_rows),
                "gradient_modules": copy.deepcopy(modules),
                "losses": copy.deepcopy(replay["losses"]),
                "peak_memory_bytes": replay["peak_memory_bytes"],
                "numerical_failure_count": replay[
                    "numerical_failure_count"
                ],
            }

        artifact = build_tau3_prefix_equivalence(
            bindings=bindings,
            full_gradient_runs=[
                measurement(
                    "full_gradient",
                    full_modules,
                    replay,
                    f"full-{index}",
                )
                for index, replay in enumerate(full_replays)
            ],
            detached_prefix_runs=[
                measurement(
                    "detached_prefix",
                    prefix_modules,
                    replay,
                    f"prefix-{index}",
                )
                for index, replay in enumerate(prefix_replays)
            ],
            behavior_trials=behavior_trials,
        )

        validation = validate_tau3_prefix_equivalence(artifact)
        self.assertTrue(artifact["passed"], artifact["failures"])
        self.assertTrue(validation["passed"], validation["errors"])

    def test_builder_records_mismatched_target_boundaries_as_negative_evidence(self) -> None:
        fixture = equivalence_fixture()
        bindings = fixture["bindings"]
        target_rows = fixture["target_accounting"]["full_gradient"]["rows"]  # type: ignore[index]

        def measurement(arm: str, run_id: str) -> dict[str, object]:
            return {
                "schema_version": "hfr.tau3_prefix_equivalence_run.v1",
                "arm": arm,
                "run_id": run_id,
                "bindings": {
                    "dataset_file_sha256": bindings["dataset_file_sha256"],
                    "protocol_file_sha256": bindings["protocol_file_sha256"],
                    "model_identity_file_sha256": bindings[
                        "model_identity_file_sha256"
                    ],
                    "recipe": {
                        key: value
                        for key, value in bindings["recipe"].items()
                        if key != "allowed_seeds"
                    }
                    | {"seed": 101},
                },
                "sample": {
                    "domains": ["airline", "retail", "telecom"],
                    "probe_families": list(
                        REQUIRED_EQUIVALENCE_PROBE_FAMILIES
                    ),
                    "stratified": True,
                    "source_sample_file_sha256": fixture["sample"][
                        "source_sample_file_sha256"
                    ],
                    "derivation": copy.deepcopy(
                        fixture["sample"]["derivation"]
                    ),
                },
                "execution": {
                    "gradient_evidence_kind": (
                        "pre_optimizer_accumulated_gradient_l2"
                        if arm == "full_gradient"
                        else "per_microbatch_gradient_l2"
                    )
                },
                "target_rows": copy.deepcopy(target_rows),
                "gradient_modules": [
                    {"name": "adapter.a", "l2_norm": 0.5}
                ],
                "losses": [1.0, 0.9],
                "peak_memory_bytes": 1000,
                "numerical_failure_count": 0,
            }

        full = [measurement("full_gradient", f"full-{index}") for index in range(2)]
        prefix = [
            measurement("detached_prefix", f"prefix-{index}")
            for index in range(2)
        ]
        prefix[0]["target_rows"][0]["target_start"] += 1  # type: ignore[index]
        trials = [
            {
                "family": family,
                "probe_id": family,
                "full_gradient_passed": True,
                "detached_prefix_passed": True,
            }
            for family in REQUIRED_EQUIVALENCE_PROBE_FAMILIES
        ]

        artifact = build_tau3_prefix_equivalence(
            bindings=bindings,
            full_gradient_runs=full,
            detached_prefix_runs=prefix,
            behavior_trials=trials,
        )

        self.assertFalse(artifact["passed"])
        self.assertIn("target", json.dumps(artifact["failures"]).lower())


if __name__ == "__main__":
    unittest.main()
