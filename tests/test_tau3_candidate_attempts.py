from __future__ import annotations

import copy
import fcntl
import json
import os
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
    _acquire_attempt_lease,
    _freeze_regular_file_record,
    _publish_new_json_readonly,
    _snapshot_regular_file,
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

    def test_wrapper_resumes_existing_attempt_without_rewriting_evidence(
        self,
    ) -> None:
        class ReceiptWritingProcess:
            def __init__(
                self,
                command: list[str],
                receipt_bytes: bytes,
                calls: list[list[str]],
                **_kwargs: Any,
            ) -> None:
                self.returncode: int | None = None
                calls.append(command)
                run_dir = Path(command[command.index("--out") + 1])
                run_dir.mkdir(parents=True, exist_ok=True)
                (run_dir / "training_receipt.json").write_bytes(receipt_bytes)

            def wait(self) -> int:
                self.returncode = 0
                return 0

            def poll(self) -> int | None:
                return self.returncode

            def terminate(self) -> None:
                self.returncode = -signal.SIGTERM

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _install_fake_python(root, "success")
            bundle = _runner_bundle(root)
            campaign = root / "campaign"
            template = run_candidate_attempt(
                campaign_root=campaign,
                attempt_id="template",
                workspace_root=root,
                training_args=[
                    "--bundle",
                    str(bundle),
                    "--iters",
                    "2",
                    "--timeout-seconds",
                    "5",
                ],
            )
            self.assertEqual(template["status"], "completed")
            receipt_bytes = (
                campaign / "template/run/training_receipt.json"
            ).read_bytes()
            training_args = [
                "--bundle",
                str(bundle),
                "--iters",
                "4",
                "--process-segment-iters",
                "2",
                "--timeout-seconds",
                "5",
            ]
            setup_calls: list[list[str]] = []

            def setup_popen(
                command: list[str],
                **kwargs: Any,
            ) -> ReceiptWritingProcess:
                return ReceiptWritingProcess(
                    command,
                    receipt_bytes,
                    setup_calls,
                    **kwargs,
                )

            with mock.patch(
                "flightrecorder.tau3_candidate_attempts.subprocess.Popen",
                side_effect=setup_popen,
            ):
                setup = run_candidate_attempt(
                    campaign_root=campaign,
                    attempt_id="candidate-a",
                    workspace_root=root,
                    training_args=training_args,
                )
            self.assertEqual(setup["status"], "completed")
            attempt = campaign / "candidate-a"
            committed = attempt / "run/process_segments/segments/segment-0001"
            committed.mkdir(parents=True)
            (committed / "state.bin").write_bytes(b"committed-state")
            (committed / "state.bin").chmod(0o444)
            partial = attempt / "run/process_segments/.segment-0002.partial"
            partial.mkdir()
            (partial / "work.bin").write_bytes(b"unpublished")
            intent_bytes = (attempt / "attempt_intent.json").read_bytes()
            committed_bytes = (committed / "state.bin").read_bytes()
            original_stdout = (attempt / "child.stdout.log").read_bytes()
            original_stderr = (attempt / "child.stderr.log").read_bytes()
            (attempt / "attempt_outcome.json").unlink()
            (attempt / "run/training_receipt.json").unlink()

            resume_calls: list[list[str]] = []

            def resume_popen(
                command: list[str],
                **kwargs: Any,
            ) -> ReceiptWritingProcess:
                return ReceiptWritingProcess(
                    command,
                    receipt_bytes,
                    resume_calls,
                    **kwargs,
                )

            with mock.patch(
                "flightrecorder.tau3_candidate_attempts.subprocess.Popen",
                side_effect=resume_popen,
            ):
                outcome = run_candidate_attempt(
                    campaign_root=campaign,
                    attempt_id="candidate-a",
                    workspace_root=root,
                    training_args=training_args,
                    resume_existing_attempt=True,
                )

            self.assertEqual(outcome["status"], "completed")
            self.assertTrue(outcome["resume_process_segments"])
            self.assertEqual(len(resume_calls), 1)
            command = resume_calls[0]
            self.assertEqual(command.count("--resume-process-segments"), 1)
            self.assertEqual(
                Path(command[command.index("--out") + 1]).resolve(),
                (attempt / "run").resolve(),
            )
            self.assertEqual(
                (attempt / "attempt_intent.json").read_bytes(),
                intent_bytes,
            )
            self.assertEqual(
                (committed / "state.bin").read_bytes(),
                committed_bytes,
            )
            self.assertTrue((partial / "work.bin").is_file())
            self.assertEqual(
                (attempt / "child.stdout.log").read_bytes(),
                original_stdout,
            )
            self.assertEqual(
                (attempt / "child.stderr.log").read_bytes(),
                original_stderr,
            )
            recovery_stdout = attempt / "child.stdout.recovery-0001.log"
            recovery_stderr = attempt / "child.stderr.recovery-0001.log"
            self.assertTrue(recovery_stdout.is_file())
            self.assertTrue(recovery_stderr.is_file())
            self.assertEqual(recovery_stdout.stat().st_mode & 0o222, 0)
            self.assertEqual(recovery_stderr.stat().st_mode & 0o222, 0)
            self.assertEqual(
                len(list(attempt.glob("attempt_outcome.json"))),
                1,
            )

    def test_wrapper_resume_rejects_drift_and_existing_outcome(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _install_fake_python(root, "success")
            bundle = _runner_bundle(root)
            campaign = root / "campaign"
            training_args = [
                "--bundle",
                str(bundle),
                "--iters",
                "2",
                "--timeout-seconds",
                "5",
            ]
            run_candidate_attempt(
                campaign_root=campaign,
                attempt_id="candidate-a",
                workspace_root=root,
                training_args=training_args,
            )
            child = mock.Mock()
            with (
                mock.patch(
                    "flightrecorder.tau3_candidate_attempts.subprocess.Popen",
                    child,
                ),
                self.assertRaisesRegex(
                    Tau3CandidateAttemptError,
                    "existing outcome",
                ),
            ):
                run_candidate_attempt(
                    campaign_root=campaign,
                    attempt_id="candidate-a",
                    workspace_root=root,
                    training_args=training_args,
                    resume_existing_attempt=True,
                )
            child.assert_not_called()

            attempt = campaign / "candidate-a"
            (attempt / "attempt_outcome.json").unlink()
            child.reset_mock()
            with (
                mock.patch(
                    "flightrecorder.tau3_candidate_attempts.subprocess.Popen",
                    child,
                ),
                self.assertRaisesRegex(
                    Tau3CandidateAttemptError,
                    "does not match supplied training args",
                ),
            ):
                run_candidate_attempt(
                    campaign_root=campaign,
                    attempt_id="candidate-a",
                    workspace_root=root,
                    training_args=[
                        "--bundle",
                        str(bundle),
                        "--iters",
                        "3",
                        "--timeout-seconds",
                        "5",
                    ],
                    resume_existing_attempt=True,
                )
            child.assert_not_called()

    def test_attempt_lease_remains_exclusive_while_inherited_child_lives(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            attempt = Path(tmp) / "attempt"
            attempt.mkdir()
            lease_fd = _acquire_attempt_lease(attempt)
            child = subprocess.Popen(
                [sys.executable, "-c", "import time; time.sleep(5)"],
                pass_fds=(lease_fd,),
            )
            os.close(lease_fd)
            try:
                with self.assertRaisesRegex(
                    Tau3CandidateAttemptError,
                    "already active",
                ):
                    _acquire_attempt_lease(attempt)
            finally:
                child.terminate()
                child.wait(timeout=5)
            replacement_fd = _acquire_attempt_lease(attempt)
            os.close(replacement_fd)

    def test_attempt_lease_rejects_path_replacement_during_acquisition(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            attempt = Path(tmp) / "attempt"
            attempt.mkdir()
            lease = attempt / ".attempt.lock"
            original_flock = fcntl.flock

            def replace_then_lock(fd: int, operation: int) -> None:
                lease.unlink()
                lease.write_bytes(b"replacement")
                original_flock(fd, operation)

            with (
                mock.patch(
                    "flightrecorder.tau3_candidate_attempts.fcntl.flock",
                    side_effect=replace_then_lock,
                ),
                self.assertRaisesRegex(
                    Tau3CandidateAttemptError,
                    "changed during acquisition",
                ),
            ):
                _acquire_attempt_lease(attempt)
            replacement_fd = _acquire_attempt_lease(attempt)
            os.close(replacement_fd)

    def test_outcome_publication_never_exposes_a_partial_final(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            outcome = root / "attempt_outcome.json"
            original_write = os.write
            interrupted = False

            def interrupted_write(fd: int, data: Any) -> int:
                nonlocal interrupted
                if not interrupted:
                    interrupted = True
                    original_write(fd, bytes(data[:1]))
                    raise SystemExit(137)
                return original_write(fd, data)

            with (
                mock.patch(
                    "flightrecorder.tau3_candidate_attempts.os.write",
                    side_effect=interrupted_write,
                ),
                self.assertRaises(SystemExit),
            ):
                _publish_new_json_readonly(outcome, {"status": "completed"})

            self.assertFalse(outcome.exists())
            partials = list(root.glob(".attempt_outcome.json.partial-*"))
            self.assertEqual(len(partials), 1)
            self.assertEqual(partials[0].stat().st_mode & 0o222, 0)

            _publish_new_json_readonly(outcome, {"status": "completed"})
            self.assertEqual(_read_json(outcome), {"status": "completed"})
            self.assertEqual(outcome.stat().st_mode & 0o222, 0)
            self.assertTrue(partials[0].is_file())

    def test_log_freeze_detects_an_existing_writer_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            log = root / "child.stdout.log"
            log.write_bytes(b"before")
            writer = log.open("ab")
            original_read = os.read
            mutated = False

            def mutating_read(fd: int, count: int) -> bytes:
                nonlocal mutated
                chunk = original_read(fd, count)
                if chunk and not mutated:
                    writer.write(b"-after")
                    writer.flush()
                    mutated = True
                return chunk

            try:
                with (
                    mock.patch(
                        "flightrecorder.tau3_candidate_attempts.os.read",
                        side_effect=mutating_read,
                    ),
                    self.assertRaisesRegex(
                        Tau3CandidateAttemptError,
                        "changed while being frozen",
                    ),
                ):
                    _freeze_regular_file_record(log, root)
            finally:
                writer.close()

    def test_log_snapshot_remains_immutable_after_legacy_writer_appends(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            log = root / "child.stdout.log"
            log.write_bytes(b"before")
            writer = log.open("ab")
            try:
                snapshot = _snapshot_regular_file(log, root)
                snapshot_path = root / snapshot["path"]
                writer.write(b"-after")
                writer.flush()
            finally:
                writer.close()
            self.assertEqual(snapshot_path.read_bytes(), b"before")
            self.assertEqual(snapshot_path.stat().st_mode & 0o222, 0)
            self.assertEqual(log.read_bytes(), b"before-after")
            self.assertEqual(snapshot["source_path"], "child.stdout.log")

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
            with self.assertRaisesRegex(
                Tau3CandidateAttemptError,
                "owns --resume-process-segments",
            ):
                run_candidate_attempt(
                    campaign_root=campaign,
                    attempt_id="bad-resume-flag",
                    workspace_root=root,
                    training_args=[
                        "--bundle",
                        str(bundle),
                        "--resume-process-segments",
                    ],
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
