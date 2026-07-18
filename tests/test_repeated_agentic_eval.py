import json
import tempfile
import unittest
import urllib.request
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest import mock

from flightrecorder.repeated_eval import (
    RepeatedEvalError,
    build_observation,
    build_promotion_evidence,
    load_arm_identity,
    paired_bootstrap,
    validate_observation,
    validate_promotion_evidence,
    write_json,
)
from flightrecorder.schema_registry import check_schema_contract, list_schema_records
from scripts.compare_agentic_finetune_results import main as comparison_main
from scripts.evaluate_hermes_heldout import (
    _build_request_attestation,
    _start_mock_server,
    _start_request_proxy,
    _validate_serving_profile,
)


ARMS = ("baseline", "trace_only", "flightrecorder")
POOLS = ("frozen", "rolling", "adversarial")


class RepeatedAgenticEvalTests(unittest.TestCase):
    def test_paired_bootstrap_reports_repeated_paired_effect_and_interval(self):
        result = paired_bootstrap([1, 1, 1], [0, 0, 0], samples=200, seed=7)

        self.assertEqual(result["pair_count"], 3)
        self.assertEqual(result["cluster_count"], 3)
        self.assertEqual(result["effective_sample_count"], 3)
        self.assertEqual(result["mean_difference"], 1.0)
        self.assertEqual(result["confidence_interval"], {"lower": 1.0, "upper": 1.0})
        self.assertIsNone(result["paired_standardized_effect"])

        variable = paired_bootstrap([1, 2, 4], [0, 0, 1], samples=200, seed=7)
        self.assertEqual(variable["mean_difference"], 2.0)
        self.assertEqual(variable["paired_standardized_effect"], 2.0)

    def test_paired_bootstrap_clusters_repeated_seeds_by_scenario_and_pool(self):
        result = paired_bootstrap(
            [1.0] * 20 + [0.0],
            [0.0] * 21,
            cluster_ids=[("frozen", "scenario-a")] * 20 + [("frozen", "scenario-b")],
            samples=200,
            seed=7,
        )

        self.assertEqual(result["pair_count"], 21)
        self.assertEqual(result["cluster_count"], 2)
        self.assertEqual(result["effective_sample_count"], 2)
        self.assertEqual(result["resampling_unit"], "scenario_pool_cluster")
        self.assertEqual(result["mean_difference"], 0.5)
        self.assertEqual(result["pairs_per_cluster"], {"min": 1, "max": 20})

    def test_request_proxy_applies_and_observes_exact_inference_configuration(self):
        upstream, upstream_requests, upstream_url = _start_mock_server("ok", "test/model")
        proxy = None
        try:
            proxy, observed, proxy_url = _start_request_proxy(
                upstream_url,
                {
                    "seed": 1729,
                    "temperature": 0.2,
                    "top_p": 0.8,
                    "max_tokens": 77,
                },
            )
            payload = json.dumps(
                {
                    "model": "test/model",
                    "messages": [{"role": "user", "content": "hello"}],
                    "stream": False,
                    "temperature": 1.0,
                    "top_p": 1.0,
                    "max_tokens": 999,
                    "seed": 1,
                }
            ).encode("utf-8")
            request = urllib.request.Request(
                proxy_url + "/chat/completions",
                data=payload,
                headers={"content-type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=5) as response:
                self.assertEqual(response.status, 200)

            self.assertEqual(len(observed), 1)
            self.assertTrue(observed[0]["matched"])
            self.assertEqual(
                {key: observed[0][key] for key in ("seed", "temperature", "top_p", "max_tokens")},
                {"seed": 1729, "temperature": 0.2, "top_p": 0.8, "max_tokens": 77},
            )
            chat_request = next(row for row in upstream_requests if row["method"] == "POST")
            self.assertEqual(chat_request["seed"], 1729)
            self.assertEqual(chat_request["temperature"], 0.2)
            self.assertEqual(chat_request["top_p"], 0.8)
            self.assertEqual(chat_request["max_tokens"], 77)
            attestation = _build_request_attestation(
                request_config={"seed": 1729, "temperature": 0.2, "top_p": 0.8, "max_tokens": 77},
                expected_model="test/model",
                endpoint_base_url=upstream_url,
                records=observed,
            )
            self.assertTrue(attestation["passed"])
            self.assertEqual(attestation["matching_request_count"], 1)
        finally:
            if proxy is not None:
                proxy.shutdown()
                proxy.server_close()
            upstream.shutdown()
            upstream.server_close()

    def test_arm_identity_rejects_mutable_revisions(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "identity.json"
            identity = _identity("flightrecorder")
            identity["model"]["revision"] = "main"
            write_json(path, identity)

            with self.assertRaisesRegex(RepeatedEvalError, "immutable"):
                load_arm_identity(path)

    def test_three_arm_evidence_passes_with_improvement_and_tied_non_regressions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = _observation_fixture(root)
            evidence_path = root / "promotion_evidence.json"

            evidence = build_promotion_evidence(
                observation_paths=paths,
                policy={"bootstrap_samples": 200, "bootstrap_seed": 19},
                out_path=evidence_path,
            )
            write_json(evidence_path, evidence)

            self.assertTrue(evidence["passed"])
            self.assertTrue(evidence["promotion_ready"])
            self.assertEqual(evidence["paired_observation_count"], 9)
            self.assertEqual(set(evidence["effects"]), {"baseline", "trace_only"})
            self.assertEqual(evidence["pool_coverage"]["minimum_observed_repeats"], 3)
            non_regression = [check for check in evidence["checks"] if check["semantics"] == "non_regression"]
            self.assertTrue(non_regression)
            self.assertTrue(all(check["passed"] for check in non_regression))
            self.assertTrue(validate_promotion_evidence(evidence_path)["passed"])
            for observation_paths in paths.values():
                self.assertTrue(validate_observation(observation_paths[0])["passed"])

    def test_effect_tie_is_valid_non_regression_but_not_promotion_improvement(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = _observation_fixture(root, candidate={"passed": False, "score": 40.0})

            evidence = build_promotion_evidence(
                observation_paths=paths,
                policy={"bootstrap_samples": 200},
                out_path=root / "promotion_evidence.json",
            )

            self.assertFalse(evidence["passed"])
            self.assertIn("primary_effect_vs_baseline", evidence["blocking_reasons"])
            self.assertIn("primary_effect_vs_trace_only", evidence["blocking_reasons"])
            non_regression = [check for check in evidence["checks"] if check["semantics"] == "non_regression"]
            self.assertTrue(all(check["passed"] for check in non_regression))

    def test_coverage_identity_and_pairing_fail_closed(self):
        cases = (
            ("missing_pool", {"pools": ("frozen", "rolling")}, "required_pool_coverage"),
            ("too_few_repeats", {"repeats": 2}, "minimum_repeats_per_scenario"),
            (
                "runtime_drift",
                {"identity_overrides": {"flightrecorder": {"runtime_sha256": _digest("different-runtime")}}},
                "immutable_arm_identity",
            ),
            (
                "scenario_drift",
                {"arm_overrides": {"flightrecorder": {"scenario_sha256": _digest("different-scenario")}}},
                "identical_paired_observations",
            ),
        )
        for label, fixture_kwargs, expected_check in cases:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                paths = _observation_fixture(root, **fixture_kwargs)

                evidence = build_promotion_evidence(
                    observation_paths=paths,
                    policy={"bootstrap_samples": 200},
                    out_path=root / "promotion_evidence.json",
                )

                self.assertFalse(evidence["passed"])
                self.assertIn(expected_check, evidence["blocking_reasons"])

    def test_each_non_regression_dimension_can_block_promotion(self):
        cases = (
            ("critical", {"critical_failures": ["new_critical"]}, "critical_safety_non_regression_vs_baseline"),
            ("tool", {"tool_schema_valid": False}, "tool_schema_non_regression_vs_baseline"),
            ("cost", {"cost_usd": 2.0}, "cost_non_regression_vs_baseline"),
            ("latency", {"latency_seconds": 2.0}, "latency_non_regression_vs_baseline"),
            ("family", {"task_family": "candidate_only_family"}, "family_non_regression_vs_baseline"),
            ("risk", {"risk_tier": "candidate_only_risk"}, "risk_non_regression_vs_baseline"),
        )
        for label, candidate, expected_check in cases:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                paths = _observation_fixture(root, candidate=candidate)

                evidence = build_promotion_evidence(
                    observation_paths=paths,
                    policy={"bootstrap_samples": 200},
                    out_path=root / "promotion_evidence.json",
                )

                self.assertFalse(evidence["passed"])
                self.assertIn(expected_check, evidence["blocking_reasons"])

    def test_source_mutation_invalidates_observation_and_promotion_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = _observation_fixture(root)
            evidence_path = root / "promotion_evidence.json"
            evidence = build_promotion_evidence(
                observation_paths=paths,
                policy={"bootstrap_samples": 200},
                out_path=evidence_path,
            )
            write_json(evidence_path, evidence)
            observation_path = paths["baseline"][0]
            observation = json.loads(observation_path.read_text(encoding="utf-8"))
            suite_path = observation_path.parent / observation["source_artifacts"]["suite_summary"]["path"]
            suite = json.loads(suite_path.read_text(encoding="utf-8"))
            suite["runs"][0]["score"] = 99.0
            write_json(suite_path, suite)

            self.assertFalse(validate_observation(observation_path)["passed"])
            self.assertFalse(validate_promotion_evidence(evidence_path)["passed"])

    def test_tuned_observation_requires_matching_serving_adapter_attestation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            identity = _identity("flightrecorder")
            paths = _write_observation_sources(root, identity, "flightrecorder")
            profile = json.loads(paths["serving_profile"].read_text(encoding="utf-8"))
            profile["model_identity"]["adapter"]["sha256"] = _digest("wrong-adapter")
            write_json(paths["serving_profile"], profile)

            observation = build_observation(
                arm_identity_path=paths["identity"],
                evaluation_summary_path=paths["evaluation"],
                suite_summary_path=paths["suite"],
                request_attestation_path=paths["request_attestation"],
                serving_profile_path=paths["serving_profile"],
                repeat_index=0,
                seed=1000,
                decoding={"temperature": 0.0, "top_p": 1.0, "max_tokens": 256},
                pool_type="frozen",
                pool_id="frozen-v1",
                risk_tier="standard",
                out_path=root / "observation.json",
            )

            self.assertFalse(observation["passed"])
            self.assertIn("serving_adapter_identity_mismatch", observation["blocking_reasons"])

    def test_evaluator_requires_exact_tuned_serving_adapter_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            identity = _identity("flightrecorder")
            profile_path = root / "serving_profile.json"
            write_json(profile_path, _serving_profile(identity))

            summary = _validate_serving_profile(
                profile_path,
                expected_model=identity["model"]["id"],
                expected_base_url="http://127.0.0.1:8000/v1",
                expected_identity=identity,
            )
            self.assertTrue(summary["adapter_attested"])

            profile = json.loads(profile_path.read_text(encoding="utf-8"))
            profile["model_identity"]["adapter"]["revision"] = "different-revision"
            write_json(profile_path, profile)
            with self.assertRaisesRegex(SystemExit, "exactly match"):
                _validate_serving_profile(
                    profile_path,
                    expected_model=identity["model"]["id"],
                    expected_base_url="http://127.0.0.1:8000/v1",
                    expected_identity=identity,
                )

    def test_evaluator_rejects_declared_only_remote_adapter_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            identity = _identity("flightrecorder")
            profile = _serving_profile(identity)
            profile["model_identity"]["adapter"]["observation_source"] = "declared_cli"
            profile_path = root / "serving_profile.json"
            write_json(profile_path, profile)

            with self.assertRaisesRegex(SystemExit, "endpoint-observed"):
                _validate_serving_profile(
                    profile_path,
                    expected_model=identity["model"]["id"],
                    expected_base_url="http://127.0.0.1:8000/v1",
                    expected_identity=identity,
                )

    def test_tuned_observation_cannot_use_unattested_base_endpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            identity = _identity("trace_only")
            paths = _write_observation_sources(root, identity, "trace_only")

            observation = build_observation(
                arm_identity_path=paths["identity"],
                evaluation_summary_path=paths["evaluation"],
                suite_summary_path=paths["suite"],
                request_attestation_path=paths["request_attestation"],
                serving_profile_path=None,
                repeat_index=0,
                seed=1000,
                decoding={"temperature": 0.0, "top_p": 1.0, "max_tokens": 256},
                pool_type="frozen",
                pool_id="frozen-v1",
                risk_tier="standard",
                out_path=root / "observation.json",
            )

            self.assertFalse(observation["passed"])
            self.assertIn("serving_profile_required_for_tuned_arm", observation["blocking_reasons"])

    def test_declared_decoding_without_matching_observed_requests_is_blocked(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            identity = _identity("baseline")
            paths = _write_observation_sources(root, identity, "baseline")
            attestation = json.loads(paths["request_attestation"].read_text(encoding="utf-8"))
            attestation["configured"]["seed"] = 999
            write_json(paths["request_attestation"], attestation)

            observation = build_observation(
                arm_identity_path=paths["identity"],
                evaluation_summary_path=paths["evaluation"],
                suite_summary_path=paths["suite"],
                request_attestation_path=paths["request_attestation"],
                serving_profile_path=None,
                repeat_index=0,
                seed=1000,
                decoding={"temperature": 0.0, "top_p": 1.0, "max_tokens": 256},
                pool_type="frozen",
                pool_id="frozen-v1",
                risk_tier="standard",
                out_path=root / "observation.json",
            )

            self.assertFalse(observation["passed"])
            self.assertIn("request_configuration_mismatch", observation["blocking_reasons"])

    def test_new_contracts_are_registered_and_strict(self):
        names = {record["name"] for record in list_schema_records()}
        self.assertTrue(
            {"eval_arm_identity", "agentic_eval_observation", "agentic_eval_promotion_evidence"}.issubset(names)
        )
        invalid = _identity("baseline")
        invalid["unexpected"] = True
        result = check_schema_contract(invalid, name_or_id="eval_arm_identity")
        self.assertFalse(result["passed"])

    def test_comparison_cli_writes_and_validates_canonical_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = _observation_fixture(root)
            summary_paths = {
                arm: paths[arm][0].parent / "suite_summary.json"
                for arm in ARMS
            }
            evidence_path = root / "promotion_evidence.json"
            argv = [
                "compare_agentic_finetune_results.py",
                "--baseline", str(summary_paths["baseline"]),
                "--trace-only", str(summary_paths["trace_only"]),
                "--flightrecorder", str(summary_paths["flightrecorder"]),
                "--out", str(root / "legacy_comparison.json"),
                "--report", str(root / "legacy_report.md"),
                "--promotion-evidence-out", str(evidence_path),
                "--bootstrap-samples", "200",
            ]
            for arm, flag in (
                ("baseline", "--baseline-observation"),
                ("trace_only", "--trace-only-observation"),
                ("flightrecorder", "--flightrecorder-observation"),
            ):
                for path in paths[arm]:
                    argv.extend([flag, str(path)])

            with mock.patch("sys.argv", argv), redirect_stdout(StringIO()):
                exit_code = comparison_main()

            self.assertEqual(exit_code, 0)
            self.assertTrue(evidence_path.exists())
            self.assertTrue(validate_promotion_evidence(evidence_path)["passed"])
            self.assertIn("Canonical repeated evidence passed: True", (root / "legacy_report.md").read_text(encoding="utf-8"))


def _observation_fixture(
    root: Path,
    *,
    pools: tuple[str, ...] = POOLS,
    repeats: int = 3,
    candidate: dict | None = None,
    arm_overrides: dict[str, dict] | None = None,
    identity_overrides: dict[str, dict] | None = None,
) -> dict[str, list[Path]]:
    candidate_case = {
        "passed": True,
        "score": 90.0,
        "critical_failures": [],
        "tool_schema_valid": True,
        "cost_usd": 1.0,
        "latency_seconds": 1.0,
        "task_family": "agentic_tool_use",
        "risk_tier": "standard",
        **(candidate or {}),
    }
    arm_cases = {
        "baseline": {
            "passed": False,
            "score": 20.0,
            "critical_failures": [],
            "tool_schema_valid": True,
            "cost_usd": 1.0,
            "latency_seconds": 1.0,
            "task_family": "agentic_tool_use",
            "risk_tier": "standard",
        },
        "trace_only": {
            "passed": False,
            "score": 40.0,
            "critical_failures": [],
            "tool_schema_valid": True,
            "cost_usd": 1.0,
            "latency_seconds": 1.0,
            "task_family": "agentic_tool_use",
            "risk_tier": "standard",
        },
        "flightrecorder": candidate_case,
    }
    for arm, override in (arm_overrides or {}).items():
        arm_cases[arm].update(override)

    identities = {arm: _identity(arm) for arm in ARMS}
    for arm, override in (identity_overrides or {}).items():
        if "runtime_sha256" in override:
            identities[arm]["runtime"]["sha256"] = override["runtime_sha256"]

    paths: dict[str, list[Path]] = {arm: [] for arm in ARMS}
    for arm in ARMS:
        for pool in pools:
            for repeat_index in range(repeats):
                observation_dir = root / arm / pool / str(repeat_index)
                identity_path = observation_dir / "arm_identity.json"
                evaluation_path = observation_dir / "evaluation_summary.json"
                suite_path = observation_dir / "suite_summary.json"
                request_attestation_path = observation_dir / "request_attestation.json"
                serving_profile_path = observation_dir / "serving_profile.json"
                observation_path = observation_dir / "observation.json"
                case = arm_cases[arm]
                scenario_sha256 = str(case.get("scenario_sha256") or _digest(f"scenario-{pool}"))
                write_json(identity_path, identities[arm])
                write_json(
                    evaluation_path,
                    {
                        "schema_version": "hfr.hermes_heldout_eval_summary.v1",
                        "arm": arm,
                        "model": identities[arm]["model"]["id"],
                        "base_url": "http://127.0.0.1:8000/v1",
                    },
                )
                write_json(
                    suite_path,
                    {
                        "schema_version": "hfr.run_suite.v1",
                        "scenarios_dir": "scenarios",
                        "out_dir": ".",
                        "total": 1,
                        "passed": 1 if case["passed"] else 0,
                        "failed": 0 if case["passed"] else 1,
                        "error_count": 0,
                        "errors": [],
                        "metrics": {
                            "pass_rate": 1.0 if case["passed"] else 0.0,
                            "average_score": float(case["score"]),
                            "min_score": int(case["score"]),
                            "max_score": int(case["score"]),
                            "failed_rule_counts": [],
                            "critical_failure_counts": [],
                            "task_families": [
                                {
                                    "task_family": case["task_family"],
                                    "total": 1,
                                    "passed": 1 if case["passed"] else 0,
                                    "failed": 0 if case["passed"] else 1,
                                    "pass_rate": 1.0 if case["passed"] else 0.0,
                                    "average_score": float(case["score"]),
                                    "failed_rule_counts": [],
                                    "critical_failure_counts": [],
                                }
                            ],
                            "failed": 0 if case["passed"] else 1,
                            "passed": 1 if case["passed"] else 0,
                        },
                        "runs": [
                            {
                                "scenario_id": f"scenario-{pool}",
                                "scenario_title": f"Scenario {pool}",
                                "scenario_path": f"scenarios/scenario-{pool}.json",
                                "scenario_sha256": scenario_sha256,
                                "trace_path": "trace.json",
                                "trace_sha256": _digest(f"trace-{arm}-{pool}-{repeat_index}"),
                                "run_dir": ".",
                                "report": "report.md",
                                "report_sha256": _digest("report"),
                                "report_size_bytes": 1,
                                "scorecard": "scorecard.json",
                                "scorecard_sha256": _digest("scorecard"),
                                "scorecard_size_bytes": 1,
                                "run_digest": "run_digest.json",
                                "run_digest_sha256": _digest("run-digest"),
                                "run_digest_size_bytes": 1,
                                "lineage": "lineage.json",
                                "lineage_sha256": _digest("lineage"),
                                "lineage_size_bytes": 1,
                                "passed": case["passed"],
                                "score": int(case["score"]),
                                "failed_rules": [],
                                "critical_failures": case["critical_failures"],
                                "tool_schema_valid": case["tool_schema_valid"],
                                "cost_usd": case["cost_usd"],
                                "latency_seconds": case["latency_seconds"],
                                "task_family": case["task_family"],
                                "risk_tier": case["risk_tier"],
                            }
                        ],
                        "artifacts": {},
                    },
                )
                write_json(
                    request_attestation_path,
                    _request_attestation(
                        model=identities[arm]["model"]["id"],
                        seed=1000 + repeat_index,
                        decoding={"temperature": 0.0, "top_p": 1.0, "max_tokens": 256},
                    ),
                )
                if arm != "baseline":
                    write_json(serving_profile_path, _serving_profile(identities[arm]))
                observation = build_observation(
                    arm_identity_path=identity_path,
                    evaluation_summary_path=evaluation_path,
                    suite_summary_path=suite_path,
                    request_attestation_path=request_attestation_path,
                    serving_profile_path=serving_profile_path if arm != "baseline" else None,
                    repeat_index=repeat_index,
                    seed=1000 + repeat_index,
                    decoding={"temperature": 0.0, "top_p": 1.0, "max_tokens": 256},
                    pool_type=pool,
                    pool_id=f"{pool}-v1",
                    risk_tier="standard",
                    out_path=observation_path,
                    created_at=f"2026-07-18T00:00:0{repeat_index}+00:00",
                )
                write_json(observation_path, observation)
                paths[arm].append(observation_path)
    return paths


def _write_observation_sources(root: Path, identity: dict, arm: str) -> dict[str, Path]:
    identity_path = root / "arm_identity.json"
    evaluation_path = root / "evaluation_summary.json"
    suite_path = root / "suite_summary.json"
    request_attestation_path = root / "request_attestation.json"
    serving_profile_path = root / "serving_profile.json"
    write_json(identity_path, identity)
    write_json(
        evaluation_path,
        {
            "schema_version": "hfr.hermes_heldout_eval_summary.v1",
            "arm": arm,
            "model": identity["model"]["id"],
            "base_url": "http://127.0.0.1:8000/v1",
        },
    )
    write_json(
        suite_path,
        {
            "schema_version": "hfr.run_suite.v1",
            "scenarios_dir": "scenarios",
            "out_dir": ".",
            "total": 1,
            "passed": 1,
            "failed": 0,
            "error_count": 0,
            "errors": [],
            "metrics": {
                "pass_rate": 1.0,
                "average_score": 1.0,
                "min_score": 1,
                "max_score": 1,
                "failed_rule_counts": [],
                "critical_failure_counts": [],
                "task_families": [],
                "failed": 0,
                "passed": 1,
            },
            "runs": [
                {
                    "scenario_id": "scenario-frozen",
                    "scenario_title": "Scenario frozen",
                    "scenario_path": "scenarios/scenario-frozen.json",
                    "scenario_sha256": _digest("scenario-frozen"),
                    "trace_path": "trace.json",
                    "trace_sha256": _digest("trace"),
                    "run_dir": ".",
                    "report": "report.md",
                    "report_sha256": _digest("report"),
                    "report_size_bytes": 1,
                    "scorecard": "scorecard.json",
                    "scorecard_sha256": _digest("scorecard"),
                    "scorecard_size_bytes": 1,
                    "run_digest": "run_digest.json",
                    "run_digest_sha256": _digest("run-digest"),
                    "run_digest_size_bytes": 1,
                    "lineage": "lineage.json",
                    "lineage_sha256": _digest("lineage"),
                    "lineage_size_bytes": 1,
                    "passed": True,
                    "score": 1,
                    "failed_rules": [],
                    "critical_failures": [],
                    "tool_schema_valid": True,
                    "cost_usd": 1.0,
                    "latency_seconds": 1.0,
                    "task_family": "agentic_tool_use",
                    "risk_tier": "standard",
                }
            ],
            "artifacts": {},
        },
    )
    write_json(
        request_attestation_path,
        _request_attestation(
            model=identity["model"]["id"],
            seed=1000,
            decoding={"temperature": 0.0, "top_p": 1.0, "max_tokens": 256},
        ),
    )
    if arm != "baseline":
        write_json(serving_profile_path, _serving_profile(identity))
    return {
        "identity": identity_path,
        "evaluation": evaluation_path,
        "suite": suite_path,
        "request_attestation": request_attestation_path,
        "serving_profile": serving_profile_path,
    }


def _request_attestation(*, model: str, seed: int, decoding: dict) -> dict:
    configured = {"seed": seed, **decoding}
    config_sha256 = _digest(json.dumps(configured, sort_keys=True, separators=(",", ":")))
    request = {
        "request_index": 0,
        "path": "/v1/chat/completions",
        "model": model,
        **configured,
        "config_sha256": config_sha256,
        "body_sha256": _digest("request-body"),
        "matched": True,
    }
    return {
        "schema_version": "hfr.eval_request_attestation.v1",
        "endpoint_base_url": "http://127.0.0.1:8000/v1",
        "configured": {**configured, "config_sha256": config_sha256},
        "request_count": 1,
        "matching_request_count": 1,
        "observed_models": [model],
        "requests": [request],
        "passed": True,
        "blocking_reasons": [],
    }


def _serving_profile(identity: dict) -> dict:
    adapter = identity["adapter"]
    model = identity["model"]["id"]
    return {
        "schema_version": "hfr.serving_profile.v1",
        "generated_at": "2026-07-18T00:00:00+00:00",
        "profile_id": f"profile-{identity['arm']}",
        "arm": identity["arm"],
        "provider": "custom",
        "engine": "vllm",
        "endpoint": {"base_url": "http://127.0.0.1:8000/v1"},
        "model_identity": {
            "requested_model": model,
            "served_model_id": f"{model}+{adapter['id']}",
            "observed_model_ids": [f"{model}+{adapter['id']}"],
            "metadata_model": f"{model}+{adapter['id']}",
            "chat_response_model": f"{model}+{adapter['id']}",
            "adapter": {
                "present": True,
                "local": False,
                "immutable": True,
                "observation_source": "endpoint_model_metadata",
                **adapter,
            },
        },
        "capabilities": {"chat_completions": True},
        "eval_preflight": {"ready": True, "readiness": "ready", "failed_checks": []},
    }


def _identity(arm: str) -> dict:
    adapter = None
    if arm != "baseline":
        adapter = {"id": f"adapter/{arm}", "revision": f"adapter-{arm}-r1", "sha256": _digest(f"adapter-{arm}")}
    return {
        "schema_version": "hfr.eval_arm_identity.v1",
        "arm": arm,
        "model": {"id": "Qwen/Qwen3-4B-Instruct-2507", "revision": "model-r1", "sha256": _digest("model")},
        "adapter": adapter,
        "runtime": {"id": "vllm", "revision": "0.9.0", "sha256": _digest("runtime")},
        "tools": {"id": "hermes-tools-v1", "sha256": _digest("tools")},
        "environment": {"id": "eval-env-v1", "sha256": _digest("environment")},
    }


def _digest(value: str) -> str:
    import hashlib

    return hashlib.sha256(value.encode("utf-8")).hexdigest()


if __name__ == "__main__":
    unittest.main()
