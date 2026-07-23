from __future__ import annotations

import copy
import json
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

from flightrecorder.schema_registry import check_schema_contract
from flightrecorder.tau3_candidate_attempts import (
    CAMPAIGN_MARKER,
    Tau3CandidateAttemptError,
    build_candidate_attempt_ledger,
    run_candidate_attempt,
)
from tests.test_tau3_mlx_training import (
    _fake_model,
    _install_fake_python,
    _mixture_variant,
    _protocol_config,
    _runner_bundle,
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _ledger_strings(value: Any) -> list[str]:
    found: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            found.append(str(key))
            found.extend(_ledger_strings(item))
    elif isinstance(value, list):
        for item in value:
            found.extend(_ledger_strings(item))
    elif isinstance(value, str):
        found.append(value)
    return found


class Tau3CandidateAttemptTests(unittest.TestCase):
    def test_candidate_attempt_wrapper_writes_intent_outcome_and_public_safe_ledger(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _install_fake_python(root, "success")
            bundle = _runner_bundle(root)
            campaign = root / "candidate_attempts"
            outcome = run_candidate_attempt(
                campaign_root=campaign,
                attempt_id="candidate-a",
                workspace_root=root,
                training_args=["--bundle", str(bundle), "--iters", "2", "--timeout-seconds", "5"],
            )
            self.assertEqual(outcome["status"], "completed")
            self.assertTrue((campaign / "candidate-a" / "attempt_intent.json").is_file())
            self.assertTrue((campaign / "candidate-a" / "attempt_outcome.json").is_file())
            self.assertTrue((campaign / "candidate-a" / "run" / "training_receipt.json").is_file())

            ledger_path = root / "candidate_attempt_ledger.json"
            ledger = build_candidate_attempt_ledger(
                campaign_root=campaign,
                out_path=ledger_path,
                workspace_root=root,
                created_at="2026-07-23T00:00:00Z",
            )

            self.assertEqual(ledger["schema_version"], "hfr.tau3_candidate_attempt_ledger.v1")
            self.assertEqual(ledger["attempt_count"], 1)
            self.assertEqual(ledger["successful_attempt_count"], 1)
            self.assertEqual(ledger["attempts"][0]["status"], "completed")
            self.assertTrue(ledger["attempts"][0]["bindings"]["config_sha256"])
            self.assertTrue(ledger["attempts"][0]["bindings"]["adapter_tree_sha256"])
            self.assertTrue(ledger["attempts"][0]["metrics"]["weights_updated"])
            self.assertEqual(ledger["attempts"][0]["metrics"]["last_train_loss"], 1.25)
            self.assertTrue(
                all(
                    not item.startswith(str(root)) and not item.startswith("/Users/")
                    for item in _ledger_strings(ledger)
                )
            )
            schema = check_schema_contract(_read_json(ledger_path), name_or_id="tau3_candidate_attempt_ledger")
            self.assertTrue(schema["passed"], schema["errors"])

    def test_malformed_or_partial_receipt_cannot_prevent_immutable_outcome(self) -> None:
        class ReceiptWritingProcess:
            def __init__(self, command: list[str], receipt_text: str, **_kwargs: Any) -> None:
                self.returncode: int | None = None
                run_dir = Path(command[command.index("--out") + 1])
                run_dir.mkdir(parents=True)
                (run_dir / "training_receipt.json").write_text(receipt_text, encoding="utf-8")

            def wait(self) -> int:
                self.returncode = 0
                return 0

            def poll(self) -> int | None:
                return self.returncode

            def terminate(self) -> None:
                self.returncode = -signal.SIGTERM

        prelaunch_receipt = json.dumps(
            {
                "schema_version": "hfr.tau3_mlx_training_run.v1",
                "phase": "prelaunch",
                "created_at": "2026-07-23T00:00:00Z",
                "bundle": {},
                "output_dir": ".",
                "command": [],
                "config": {},
                "checks": [],
                "weights_updated": False,
                "terminal_status": "prelaunch",
            }
        )
        cases = (
            ("{", "receipt_parse_error"),
            ("{}", "receipt_schema_invalid"),
            (prelaunch_receipt, "receipt_schema_invalid"),
        )
        for receipt_text, expected_reason in cases:
            with self.subTest(receipt_text=receipt_text), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                bundle = _runner_bundle(root)
                campaign = root / "campaign"

                def fake_popen(command: list[str], **kwargs: Any) -> ReceiptWritingProcess:
                    return ReceiptWritingProcess(command, receipt_text, **kwargs)

                with mock.patch(
                    "flightrecorder.tau3_candidate_attempts.subprocess.Popen",
                    side_effect=fake_popen,
                ):
                    outcome = run_candidate_attempt(
                        campaign_root=campaign,
                        attempt_id="malformed",
                        workspace_root=root,
                        training_args=["--bundle", str(bundle)],
                    )

                outcome_path = campaign / "malformed" / "attempt_outcome.json"
                self.assertEqual(outcome["status"], "malformed-receipt")
                self.assertEqual(outcome["failure_reasons"], [expected_reason])
                self.assertTrue(outcome_path.is_file())
                self.assertEqual(_read_json(outcome_path)["status"], "malformed-receipt")
                self.assertEqual(outcome_path.stat().st_mode & 0o222, 0)
                ledger = build_candidate_attempt_ledger(
                    campaign_root=campaign,
                    out_path=root / "ledger.json",
                    workspace_root=root,
                )
                self.assertEqual(ledger["attempts"][0]["status"], "malformed-receipt")
                self.assertIn("malformed_receipt", ledger["attempts"][0]["failure_reasons"])

    def test_wrapper_sigterm_is_recorded_as_interrupted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _install_fake_python(root, "sleep")
            bundle = _runner_bundle(root)
            campaign = root / "campaign"
            wrapper = Path(__file__).resolve().parents[1] / "scripts" / "run_tau3_candidate_attempt.py"
            process = subprocess.Popen(
                [
                    sys.executable,
                    str(wrapper),
                    "--campaign-root",
                    str(campaign),
                    "--attempt-id",
                    "sigterm",
                    "--",
                    "--bundle",
                    str(bundle),
                    "--iters",
                    "2",
                    "--timeout-seconds",
                    "10",
                ],
                cwd=root,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            prelaunch_path = campaign / "sigterm" / "run" / "prelaunch_receipt.json"
            deadline = time.monotonic() + 8
            while not prelaunch_path.exists() and process.poll() is None and time.monotonic() < deadline:
                time.sleep(0.02)
            self.assertTrue(prelaunch_path.is_file(), "wrapper did not launch training before SIGTERM")
            process.send_signal(signal.SIGTERM)
            stdout, stderr = process.communicate(timeout=10)

            self.assertEqual(process.returncode, 1, (stdout, stderr))
            outcome = _read_json(campaign / "sigterm" / "attempt_outcome.json")
            self.assertEqual(outcome["status"], "interrupted")
            self.assertTrue(outcome["interrupted"])
            self.assertLess(outcome["exit_code"], 0)

    def test_ledger_censuses_orphan_and_missing_outcome_attempts_as_failed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            campaign = root / "campaign"
            campaign.mkdir()
            (campaign / CAMPAIGN_MARKER).write_text("hfr.tau3_candidate_attempt_campaign.v1\n", encoding="utf-8")
            orphan = campaign / "orphan-dir"
            orphan.mkdir()
            missing_outcome = campaign / "missing-outcome"
            missing_outcome.mkdir()
            _write_json(
                missing_outcome / "attempt_intent.json",
                {
                    "schema_version": "hfr.tau3_candidate_attempt_intent.v1",
                    "attempt_id": "missing-outcome",
                    "created_at": "2026-07-23T00:00:00Z",
                },
            )

            ledger = build_candidate_attempt_ledger(
                campaign_root=campaign,
                out_path=root / "ledger.json",
                workspace_root=root,
            )

            self.assertEqual(ledger["attempt_count"], 2)
            self.assertEqual(ledger["failed_attempt_count"], 2)
            statuses = {entry["attempt_id"]: entry for entry in ledger["attempts"]}
            self.assertIn("missing_intent", statuses["orphan-dir"]["failure_reasons"])
            self.assertIn("missing_outcome", statuses["missing-outcome"]["failure_reasons"])
            self.assertIn("missing_receipt", statuses["missing-outcome"]["failure_reasons"])

    def test_wrapper_rejects_forwarded_out_sealed_refs_and_symlink_sources(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _install_fake_python(root, "success")
            bundle = _runner_bundle(root)
            campaign = root / "campaign"
            with self.assertRaisesRegex(Tau3CandidateAttemptError, "owns --out"):
                run_candidate_attempt(
                    campaign_root=campaign,
                    attempt_id="bad-out",
                    workspace_root=root,
                    training_args=["--bundle", str(bundle), "--out", str(root / "out")],
                )
            with self.assertRaisesRegex(Tau3CandidateAttemptError, "owns --out"):
                run_candidate_attempt(
                    campaign_root=campaign,
                    attempt_id="bad-out-equals",
                    workspace_root=root,
                    training_args=["--bundle", str(bundle), f"--out={root / 'out'}"],
                )
            sealed = root / "sealed_bundle"
            sealed.mkdir()
            with self.assertRaisesRegex(Tau3CandidateAttemptError, "sealed/test"):
                run_candidate_attempt(
                    campaign_root=campaign,
                    attempt_id="sealed-ref",
                    workspace_root=root,
                    training_args=["--bundle", str(sealed)],
                )
            link = root / "linked_bundle"
            link.symlink_to(bundle, target_is_directory=True)
            with self.assertRaisesRegex(Tau3CandidateAttemptError, "symlink"):
                run_candidate_attempt(
                    campaign_root=campaign,
                    attempt_id="linked-source",
                    workspace_root=root,
                    training_args=["--bundle", str(link)],
                )

    def test_mixture_attempt_ledger_binds_protocol_dataset_recipe_and_adapter_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _install_fake_python(root, "success")
            model, identity = _fake_model(root)
            protocol = _protocol_config(root, identity)
            mixture = _mixture_variant(root, protocol_path=protocol)
            campaign = root / "campaign"
            outcome = run_candidate_attempt(
                campaign_root=campaign,
                attempt_id="candidate-mix",
                workspace_root=root,
                training_args=[
                    "--mixture-dir",
                    str(mixture),
                    "--protocol",
                    str(protocol),
                    "--model-identity",
                    str(identity),
                    "--model-path",
                    str(model),
                    "--iters",
                    "2",
                    "--timeout-seconds",
                    "5",
                ],
            )
            self.assertEqual(outcome["status"], "completed")
            ledger = build_candidate_attempt_ledger(
                campaign_root=campaign,
                out_path=root / "ledger.json",
                workspace_root=root,
            )
            bindings = ledger["attempts"][0]["bindings"]
            self.assertRegex(bindings["protocol_sha256"], r"^[0-9a-f]{64}$")
            self.assertRegex(bindings["dataset_manifest_sha256"], r"^[0-9a-f]{64}$")
            self.assertRegex(bindings["recipe_sha256"], r"^[0-9a-f]{64}$")
            self.assertRegex(bindings["adapter_tree_sha256"], r"^[0-9a-f]{64}$")

    def test_public_ledger_schema_rejects_paths_raw_logs_and_unknown_keys(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            campaign = root / "campaign"
            campaign.mkdir()
            (campaign / CAMPAIGN_MARKER).write_text(
                "hfr.tau3_candidate_attempt_campaign.v1\n",
                encoding="utf-8",
            )
            (campaign / "orphan").mkdir()
            ledger = build_candidate_attempt_ledger(
                campaign_root=campaign,
                out_path=root / "ledger.json",
                workspace_root=root,
            )

            invalid_payloads: list[dict[str, Any]] = []
            local_path = copy.deepcopy(ledger)
            local_path["campaign"]["root_ref"] = "/Users/private/campaign"
            invalid_payloads.append(local_path)
            raw_log = copy.deepcopy(ledger)
            raw_log["attempts"][0]["metrics"]["raw_log"] = "unredacted output"
            invalid_payloads.append(raw_log)
            unknown_binding = copy.deepcopy(ledger)
            unknown_binding["attempts"][0]["bindings"]["local_path"] = "/private/model"
            invalid_payloads.append(unknown_binding)
            unknown_attempt = copy.deepcopy(ledger)
            unknown_attempt["attempts"][0]["unknown"] = True
            invalid_payloads.append(unknown_attempt)
            unknown_status_count = copy.deepcopy(ledger)
            unknown_status_count["status_counts"]["cancelled"] = 0
            invalid_payloads.append(unknown_status_count)

            for payload in invalid_payloads:
                with self.subTest(payload=payload):
                    schema = check_schema_contract(payload, name_or_id="tau3_candidate_attempt_ledger")
                    self.assertFalse(schema["passed"], schema["errors"])

    def test_ledger_refuses_attempts_modified_after_lock_timestamp(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            campaign = root / "campaign"
            campaign.mkdir()
            (campaign / CAMPAIGN_MARKER).write_text("hfr.tau3_candidate_attempt_campaign.v1\n", encoding="utf-8")
            attempt = campaign / "late"
            attempt.mkdir()
            _write_json(
                attempt / "attempt_intent.json",
                {"schema_version": "hfr.tau3_candidate_attempt_intent.v1", "attempt_id": "late"},
            )
            time.sleep(0.01)
            with self.assertRaisesRegex(Tau3CandidateAttemptError, "after candidate lock"):
                build_candidate_attempt_ledger(
                    campaign_root=campaign,
                    out_path=root / "ledger.json",
                    workspace_root=root,
                    lock_created_at="2000-01-01T00:00:00Z",
                    lock_sha256="a" * 64,
                )


if __name__ == "__main__":
    unittest.main()
