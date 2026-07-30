import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from flightrecorder.schema_registry import check_schema_contract, check_schema_file, list_schema_records
from flightrecorder.tau3_evaluation import (
    REQUIRED_ARMS,
    TAU3_EVALUATION_SCHEMA_VERSION,
    Tau3EvaluationError,
    analyze_tau3_evaluation,
    validate_tau3_evaluation_report,
)


REV = "1" * 40
ROOT = Path(__file__).resolve().parents[1]


class Tau3EvaluationTests(unittest.TestCase):
    def test_builds_public_safe_passing_sealed_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            arms = _write_all_results(root)
            out = root / "evaluation.json"

            report = analyze_tau3_evaluation(
                arm_result_paths=arms,
                out_path=out,
                mode="sealed",
                expected_tau_revision=REV,
                bootstrap_samples=200,
                bootstrap_seed=7,
                created_at="2026-07-22T00:00:00+00:00",
            )

            self.assertEqual(report["schema_version"], TAU3_EVALUATION_SCHEMA_VERSION)
            self.assertTrue(report["passed"], report["blocking_reasons"])
            self.assertTrue(report["promotion_ready"])
            self.assertEqual(report["metrics"]["macro_pass1"]["adapter"], 1.0)
            self.assertEqual(report["metrics"]["macro_pass1"]["base"], 0.0)
            self.assertEqual(report["pairing"]["domain_counts"], {"airline": 2, "retail": 2, "telecom": 2})
            self.assertGreater(
                report["effects"]["base"]["domain_stratified_macro_pass1"]["confidence_interval"]["lower"],
                0.0,
            )
            self.assertTrue(check_schema_file(out, "tau3_evaluation")["passed"])
            self.assertTrue(validate_tau3_evaluation_report(out)["passed"])
            encoded = json.dumps(report, sort_keys=True)
            for forbidden in (
                "messages",
                "user_scenario",
                "evaluation_criteria",
                "raw_data",
                "policy",
                "fixed-user-sim",
                "127.0.0.1",
                "localhost",
                "local/tau3",
                "/Users/",
            ):
                self.assertNotIn(f'"{forbidden}"', encoded)

    def test_builds_from_registered_hash_only_blind_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            arms = _write_all_blind_results(root)

            report = analyze_tau3_evaluation(
                arm_result_paths=arms,
                mode="sealed",
                expected_tau_revision=REV,
                bootstrap_samples=200,
                bootstrap_seed=7,
                created_at="2026-07-22T00:00:00+00:00",
            )

            self.assertTrue(report["passed"], report["blocking_reasons"])
            self.assertTrue(report["promotion_ready"])
            self.assertEqual(
                report["metrics"]["macro_pass1"]["adapter"], 1.0
            )
            self.assertTrue(
                check_schema_file(
                    arms["adapter"][0],
                    "tau3_blind_benchmark_result",
                )["passed"]
            )
            encoded = json.dumps(report, sort_keys=True)
            self.assertNotIn('"tasks"', encoded)
            self.assertNotIn('"messages"', encoded)

    def test_rejects_non_identical_harness(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            arms = _write_all_results(root)
            payload = _read(arms["base"][0])
            payload["info"]["max_steps"] = 31
            _write(arms["base"][0], payload)

            report = analyze_tau3_evaluation(
                arm_result_paths=arms,
                expected_tau_revision=REV,
                bootstrap_samples=200,
                created_at="2026-07-22T00:00:00+00:00",
            )

            self.assertFalse(report["passed"])
            self.assertIn("source_results_valid", report["blocking_reasons"])
            self.assertIn("identical_harness", report["blocking_reasons"])

    def test_rejects_unpaired_and_duplicate_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            arms = _write_all_results(root)
            payload = _read(arms["comparator_1"][0])
            payload["simulations"][0]["seed"] = 999
            payload["simulations"][1]["seed"] = 999
            _write(arms["comparator_1"][0], payload)

            report = analyze_tau3_evaluation(
                arm_result_paths=arms,
                expected_tau_revision=REV,
                bootstrap_samples=200,
                created_at="2026-07-22T00:00:00+00:00",
            )

            self.assertFalse(report["passed"])
            self.assertIn("unique_paired_results", report["blocking_reasons"])

    def test_fails_closed_when_safety_is_not_provable(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            arms = _write_all_results(root)
            payload = _read(arms["adapter"][0])
            payload["simulations"][0]["reward_info"]["db_check"] = {}
            _write(arms["adapter"][0], payload)

            report = analyze_tau3_evaluation(
                arm_result_paths=arms,
                expected_tau_revision=REV,
                bootstrap_samples=200,
                created_at="2026-07-22T00:00:00+00:00",
            )

            self.assertFalse(report["passed"])
            self.assertIn("safety_metrics_provable", report["blocking_reasons"])

    def test_requires_full_policy_review_and_counts_guideline_violations(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            arms = _write_all_results(root)
            adapter = _read(arms["adapter"][0])
            adapter["simulations"][0]["review"] = None
            _write(arms["adapter"][0], adapter)
            report = analyze_tau3_evaluation(
                arm_result_paths=arms,
                expected_tau_revision=REV,
                bootstrap_samples=200,
                created_at="2026-07-22T00:00:00+00:00",
            )
            self.assertFalse(report["passed"])
            self.assertEqual(report["metrics"]["safety"]["missing_policy_review_counts"]["adapter"], 1)

            adapter["simulations"][0]["review"] = {
                "errors": [
                    {
                        "source": "agent",
                        "severity": "critical",
                        "error_tags": ["guideline_violation"],
                    }
                ]
            }
            _write(arms["adapter"][0], adapter)
            report = analyze_tau3_evaluation(
                arm_result_paths=arms,
                expected_tau_revision=REV,
                bootstrap_samples=200,
                created_at="2026-07-22T00:00:00+00:00",
            )
            self.assertEqual(report["metrics"]["safety"]["policy_violation_counts"]["adapter"], 1)
            self.assertIn("safety_non_inferiority_vs_base", report["blocking_reasons"])

    def test_missing_db_check_is_allowed_for_non_db_reward_basis(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            arms = _write_all_results(root)
            for arm in REQUIRED_ARMS:
                payload = _read(arms[arm][0])
                payload["simulations"][0]["reward_info"]["reward_basis"] = None
                payload["simulations"][0]["reward_info"]["db_check"] = None
                _write(arms[arm][0], payload)

            report = analyze_tau3_evaluation(
                arm_result_paths=arms,
                mode="sealed",
                expected_tau_revision=REV,
                bootstrap_samples=200,
                created_at="2026-07-22T00:00:00+00:00",
            )

            self.assertTrue(report["passed"], report["blocking_reasons"])
            self.assertEqual(report["metrics"]["safety"]["missing_db_evidence_counts"]["adapter"], 0)
            first_row = report["per_task_hashed"][0]["arms"]["adapter"]
            self.assertFalse(first_row["db_evaluated"])

    def test_failed_or_premature_runs_are_pass1_zero_not_source_invalid(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            arms = _write_all_results(root)
            payload = _read(arms["adapter"][0])
            payload["simulations"][0]["termination_reason"] = "max_steps"
            payload["simulations"][0]["reward_info"]["reward"] = 0.0
            _write(arms["adapter"][0], payload)

            report = analyze_tau3_evaluation(
                arm_result_paths=arms,
                expected_tau_revision=REV,
                bootstrap_samples=200,
                created_at="2026-07-22T00:00:00+00:00",
            )

            source_check = next(check for check in report["checks"] if check["id"] == "source_results_valid")
            self.assertTrue(source_check["passed"], source_check["details"])
            failed_row = next(row for row in report["per_task_hashed"] if row["key"]["domain"] == "airline")
            self.assertEqual(failed_row["arms"]["adapter"]["pass1"], 0.0)

    def test_primary_improvement_requires_strict_positive_ci_lower_bound(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            arms = _write_all_results(root)
            for domain_path in arms["adapter"]:
                payload = _read(domain_path)
                for simulation in payload["simulations"]:
                    simulation["reward_info"]["reward"] = 0.0
                if payload["info"]["environment_info"]["domain_name"] == "airline":
                    payload["simulations"][0]["reward_info"]["reward"] = 1.0
                _write(domain_path, payload)

            report = analyze_tau3_evaluation(
                arm_result_paths=arms,
                expected_tau_revision=REV,
                bootstrap_samples=400,
                created_at="2026-07-22T00:00:00+00:00",
            )

            effect = report["effects"]["base"]["domain_stratified_macro_pass1"]
            self.assertEqual(effect["confidence_interval"]["lower"], 0.0)
            self.assertFalse(report["effects"]["base"]["primary_improvement_passed"])
            self.assertIn("primary_macro_improvement_vs_base", report["blocking_reasons"])

    def test_macro_bootstrap_averages_repeats_within_task_clusters(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            arms = _write_all_results(root)
            for arm in REQUIRED_ARMS:
                payload = _read(arms[arm][0])
                repeat = _simulation("airline", 0, reward=0.0 if arm == "adapter" else 0.0, db_match=True)
                repeat["seed"] = 2000
                payload["simulations"].append(repeat)
                _write(arms[arm][0], payload)

            report = analyze_tau3_evaluation(
                arm_result_paths=arms,
                mode="sealed",
                expected_tau_revision=REV,
                bootstrap_samples=200,
                created_at="2026-07-22T00:00:00+00:00",
            )

            domain_means = report["effects"]["base"]["domain_stratified_macro_pass1"]["domain_means"]
            self.assertEqual(domain_means["airline"], 0.75)

    def test_new_only_output_and_public_raw_payload_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            arms = _write_all_results(root)
            out = root / "evaluation.json"
            out.write_text("{}\n", encoding="utf-8")

            with self.assertRaisesRegex(Tau3EvaluationError, "output already exists"):
                analyze_tau3_evaluation(arm_result_paths=arms, out_path=out, expected_tau_revision=REV, bootstrap_samples=200)

            bad = {
                "schema_version": TAU3_EVALUATION_SCHEMA_VERSION,
                "created_at": "2026-07-22T00:00:00+00:00",
                "mode": "sealed",
                "tau_revision": REV,
                "passed": True,
                "promotion_ready": True,
                "harness": {},
                "pairing": {"passed": True, "paired_count": 1},
                "metrics": {},
                "effects": {},
                "checks": [],
                "blocking_reasons": [],
                "public_payload_scan": {"passed": True},
                "messages": [{"role": "user", "content": "raw"}],
            }
            bad_path = root / "bad.json"
            _write(bad_path, bad)
            validation = validate_tau3_evaluation_report(bad_path)
            self.assertFalse(validation["passed"])
            self.assertIn("forbidden raw payload", "; ".join(validation["errors"]))

    def test_validation_rejects_local_path_or_loopback_leak(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            arms = _write_all_results(root)
            out = root / "evaluation.json"
            report = analyze_tau3_evaluation(
                arm_result_paths=arms,
                out_path=out,
                mode="sealed",
                expected_tau_revision=REV,
                bootstrap_samples=200,
                created_at="2026-07-22T00:00:00+00:00",
            )
            report["harness"]["normalized_by_domain"]["airline"]["user"]["llm_sha256"] = "/Users/zachary/local/tau3"
            leaked = root / "leaked.json"
            _write(leaked, report)

            validation = validate_tau3_evaluation_report(leaked)

            self.assertFalse(validation["passed"])
            self.assertIn("local/private text", "; ".join(validation["errors"]))


    def test_cli_writes_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            arms = _write_all_results(root)
            out = root / "cli.json"
            cmd = [
                sys.executable,
                str(ROOT / "scripts" / "analyze_tau3_evaluation.py"),
                "--out",
                str(out),
                "--mode",
                "sealed",
                "--expected-tau-revision",
                REV,
                "--bootstrap-samples",
                "200",
            ]
            for arm in REQUIRED_ARMS:
                for path in arms[arm]:
                    cmd.extend(["--arm", f"{arm}={path}"])

            completed = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, check=False)

            self.assertEqual(completed.returncode, 0, completed.stderr + completed.stdout)
            self.assertTrue(check_schema_file(out, "tau3_evaluation")["passed"])

    def test_schema_is_registered(self):
        names = {record["name"] for record in list_schema_records()}
        self.assertIn("tau3_evaluation", names)
        with tempfile.TemporaryDirectory() as tmp:
            report = analyze_tau3_evaluation(
                arm_result_paths=_write_all_results(Path(tmp)),
                mode="sealed",
                expected_tau_revision=REV,
                bootstrap_samples=200,
                created_at="2026-07-22T00:00:00+00:00",
            )
        self.assertTrue(check_schema_contract(report, name_or_id="tau3_evaluation")["passed"])


def _write_all_results(root: Path) -> dict[str, list[Path]]:
    arms: dict[str, list[Path]] = {arm: [] for arm in REQUIRED_ARMS}
    for arm in REQUIRED_ARMS:
        for domain in ("airline", "retail", "telecom"):
            path = root / arm / f"{domain}.json"
            rows = []
            for index in range(2):
                if arm == "adapter":
                    reward, db_match = 1.0, True
                elif arm == "base":
                    reward, db_match = 0.0, False
                else:
                    reward, db_match = 0.0, True
                rows.append(_simulation(domain, index, reward=reward, db_match=db_match))
            _write(path, _result(domain, rows))
            arms[arm].append(path)
    return arms


def _write_all_blind_results(root: Path) -> dict[str, list[Path]]:
    arms: dict[str, list[Path]] = {
        arm: [] for arm in REQUIRED_ARMS
    }
    for arm in REQUIRED_ARMS:
        for domain in ("airline", "retail", "telecom"):
            path = root / arm / f"{domain}.json"
            simulations = []
            for index in range(2):
                if arm == "adapter":
                    reward, db_match = 1.0, True
                elif arm == "base":
                    reward, db_match = 0.0, False
                else:
                    reward, db_match = 0.0, True
                simulations.append(
                    _blind_simulation(
                        domain,
                        index,
                        reward=reward,
                        db_match=db_match,
                    )
                )
            _write(path, _blind_result(domain, simulations))
            arms[arm].append(path)
    return arms


def _blind_result(
    domain: str,
    simulations: list[dict],
) -> dict:
    return {
        "schema_version": "hfr.tau3_blind_benchmark_result.v1",
        "created_at": "2026-07-22T00:00:00Z",
        "source_revision": REV,
        "sealed_source_manifest_sha256": "a" * 64,
        "domain": domain,
        "run_seed": 101,
        "task_count": len(simulations),
        "hashes_only": True,
        "local_paths_included": False,
        "raw_payload_included": False,
        "harness": {
            "git_commit": REV,
            "max_steps": 30,
            "max_errors": 10,
            "num_trials": 1,
            "max_retries": 0,
            "auto_resume": False,
            "auto_review": True,
            "review_mode": "full",
            "review_model_sha256": "b" * 64,
            "hallucination_retries": 0,
            "text_streaming_config_sha256": "c" * 64,
            "retrieval_config_sha256": "d" * 64,
            "domain_name": domain,
            "policy_sha256": "e" * 64,
            "agent": {
                "implementation": "llm_agent",
                "llm_args": {
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "max_tokens": 1024,
                    "num_retries": 0,
                },
            },
            "user": {
                "implementation": "user_simulator",
                "llm_sha256": "f" * 64,
                "llm_args": {
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "max_tokens": 1024,
                    "num_retries": 0,
                },
            },
        },
        "simulations": simulations,
    }


def _blind_simulation(
    domain: str,
    index: int,
    *,
    reward: float,
    db_match: bool,
) -> dict:
    return {
        "task_id_sha256": _hash(f"{domain}-id-{index}"),
        "task_sha256": _hash(f"{domain}-task-{index}"),
        "prompt_sha256": _hash(f"{domain}-prompt-{index}"),
        "trial": 0,
        "seed": _tau_trial_seed(101),
        "termination_reason": "user_stop",
        "reward_info": {
            "reward": reward,
            "reward_basis": ["DB", "COMMUNICATE"],
            "db_check": {"db_match": db_match},
        },
        "safety": {
            "policy_provable": True,
            "policy_violation": False,
            "harmful_mutation": not db_match,
        },
        "usage": {"prompt_tokens": 42},
    }


def _hash(value: str) -> str:
    import hashlib

    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _tau_trial_seed(run_seed: int) -> int:
    import random

    return random.Random(run_seed).randint(0, 1000000)


def _result(domain: str, simulations: list[dict]) -> dict:
    return {
        "timestamp": "2026-07-22T00:00:00",
        "info": {
            "git_commit": REV,
            "num_trials": 1,
            "max_steps": 30,
            "max_errors": 10,
            "max_retries": 0,
            "auto_resume": False,
            "auto_review": True,
            "review_mode": "full",
            "review_model": "fixed-reviewer",
            "hallucination_retries": 0,
            "seed": 101,
            "text_streaming_config": {"chunk_by": "words", "chunk_size": 1},
            "retrieval_config": None,
            "user_info": {
                "implementation": "user_simulator",
                "llm": "fixed-user-sim",
                "llm_args": {
                    "api_base": "http://127.0.0.1:18081/v1",
                    "api_key": "local",
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "max_tokens": 1024,
                    "num_retries": 0,
                },
                "global_simulation_guidelines": "raw hidden guideline that must not be output",
            },
            "agent_info": {
                "implementation": "llm_agent",
                "llm": "arm-specific-model",
                "llm_args": {
                    "api_base": "http://127.0.0.1:18080/v1",
                    "api_key": "local",
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "max_tokens": 1024,
                    "num_retries": 0,
                },
            },
            "environment_info": {
                "domain_name": domain,
                "policy": f"{domain} raw policy that must not be output",
                "tool_defs": [{"name": "raw_tool"}],
            },
        },
        "tasks": [_task(domain, 0), _task(domain, 1)],
        "simulations": simulations,
    }


def _task(domain: str, index: int) -> dict:
    return {
        "id": f"{domain}-{index}",
        "user_scenario": {"instructions": {"task_instructions": "raw hidden user-sim task"}},
        "evaluation_criteria": {"nl_assertions": ["raw grader"]},
    }


def _simulation(domain: str, index: int, *, reward: float, db_match: bool) -> dict:
    return {
        "id": f"sim-{domain}-{index}",
        "task_id": f"{domain}-{index}",
        "trial": 0,
        "seed": 1000 + index,
        "termination_reason": "user_stop",
        "reward_info": {
            "reward": reward,
            "db_check": {"db_match": db_match, "db_reward": 1.0 if db_match else 0.0},
            "reward_basis": ["DB", "COMMUNICATE"],
        },
        "messages": [{"role": "user", "content": "raw transcript"}],
        "raw_data": {"secretish": "provider payload"},
        "review": {"errors": []},
    }


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
