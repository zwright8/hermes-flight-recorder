import json
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

from flightrecorder.tau3_capture_generation import (
    Tau3CaptureGenerationError,
    generate_tau3_training_captures,
)
from flightrecorder.tau3_capture import capture_to_hfr


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "generate_tau3_training_captures.py"
DOMAINS = ("airline", "retail", "telecom")


class Tau3CaptureGenerationTests(unittest.TestCase):
    def test_deterministic_success_and_cli_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = make_fake_tau_repo(root / "tau")
            revision = git(repo, "rev-parse", "HEAD")
            train = root / "train.jsonl"
            dev = root / "development.jsonl"
            write_jsonl(train, [
                envelope("airline", "train", revision, task("airline", "airline-1")),
                envelope("retail", "train", revision, task("retail", "retail-1")),
                envelope("telecom", "train", revision, task("telecom", "[mobile_data_issue]telecom-train-1")),
            ])
            write_jsonl(dev, [
                envelope("airline", "development", revision, task("airline", "airline-dev-1")),
                envelope("retail", "development", revision, task("retail", "retail-dev-1")),
                envelope("telecom", "development", revision, task("telecom", "[mobile_data_issue]telecom-dev-1")),
            ])
            out_a = root / "captures-a"
            out_b = root / "captures-b"

            summary = generate_tau3_training_captures(
                tau_repo=repo,
                expected_revision=revision,
                train_tasks=train,
                development_tasks=dev,
                out=out_a,
                tau_python=sys.executable,
                train_domain_quotas=one_each(),
                development_domain_quotas=one_each(),
            )
            completed = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--tau-repo",
                    str(repo),
                    "--expected-revision",
                    revision,
                    "--train-tasks",
                    str(train),
                    "--development-tasks",
                    str(dev),
                    "--out",
                    str(out_b),
                    "--tau-python",
                    sys.executable,
                    "--train-domain-quotas",
                    "airline=1,retail=1,telecom=1",
                    "--development-domain-quotas",
                    "airline=1,retail=1,telecom=1",
                ],
                cwd=ROOT,
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            cli_summary = json.loads(completed.stdout)
            self.assertEqual(cli_summary["capture_count"], summary["capture_count"])
            self.assertEqual((out_a / "captures.jsonl").read_bytes(), (out_b / "captures.jsonl").read_bytes())
            self.assertEqual((out_a / "manifest.json").read_bytes(), (out_b / "manifest.json").read_bytes())
            captures = read_jsonl(out_a / "captures.jsonl")
            self.assertEqual(len(captures), 48)
            behavior_names = {capture["behavior"] for capture in captures}
            self.assertEqual(
                behavior_names,
                {
                    "success",
                    "correction",
                    "clarification_refusal",
                    "recovery",
                    "policy_failure",
                    "harmful_mutation",
                    "hallucinated_tool",
                    "premature_completion",
                },
            )
            self.assertEqual((out_a / "captures.jsonl").stat().st_mode & 0o777, 0o600)
            self.assertEqual((out_a / "manifest.json").stat().st_mode & 0o777, 0o600)
            for capture in captures:
                self.assertEqual(capture["schema_version"], "hfr.tau3_capture.v1")
                self.assertIn("source_task_sha256", capture)
                self.assertIn("source_prompt_sha256", capture)
                self.assertTrue(all(event.get("tool_call_id") for event in capture["events"] if event["type"].startswith("tool_")))
                artifacts = capture_to_hfr(capture)
                self.assertEqual(artifacts["scorecard"]["passed"], capture["outcome"]["success"])
                self.assertIsInstance(capture["token_count"], int)
                self.assertGreater(capture["token_count"], 0)
                self.assertIn("policy_evidence", capture)
                for change in capture["state_transition"]["changes"]:
                    self.assertEqual(change["kind"], "changed")
                if capture["behavior"] == "success":
                    self.assertTrue(any(event["type"] == "tool_call" for event in capture["events"]))
            manifest = read_json(out_a / "manifest.json")
            self.assertFalse(manifest["sealed_payload_accessed"])
            self.assertFalse(manifest["sealed_manifest_accessed"])
            self.assertEqual(manifest["split_counts"], {"development": 24, "train": 24})
            self.assertTrue(manifest["domain_balance_passed"])
            self.assertEqual(
                manifest["token_count_by_domain"],
                token_count_by_domain(captures),
            )
            self.assertEqual(manifest["sampling"]["quotas"]["train"], one_each())
            self.assertEqual(manifest["sampling"]["quotas"]["development"], one_each())
            for split in ("train", "development"):
                for domain in DOMAINS:
                    self.assertEqual(manifest["sampling"]["splits"][split][domain]["selected_count"], 1)
                    self.assertGreaterEqual(manifest["sampling"]["splits"][split][domain]["eligible_count"], 1)
            self.assertTrue(manifest["worker"]["tool_schemas_recorded_exact"])

    def test_wrong_revision_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = make_fake_tau_repo(root / "tau")
            revision = git(repo, "rev-parse", "HEAD")
            train, dev = write_inputs(root, revision)

            with self.assertRaisesRegex(Tau3CaptureGenerationError, "revision mismatch"):
                generate_tau3_training_captures(
                    tau_repo=repo,
                    expected_revision="0" * 40,
                    train_tasks=train,
                    development_tasks=dev,
                    out=root / "out",
                    tau_python=sys.executable,
                    train_domain_quotas=one_each(),
                    development_domain_quotas=one_each(),
                )

    def test_dirty_repo_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = make_fake_tau_repo(root / "tau")
            revision = git(repo, "rev-parse", "HEAD")
            (repo / "dirty.txt").write_text("dirty\n", encoding="utf-8")
            train, dev = write_inputs(root, revision)

            with self.assertRaisesRegex(Tau3CaptureGenerationError, "clean"):
                generate_tau3_training_captures(
                    tau_repo=repo,
                    expected_revision=revision,
                    train_tasks=train,
                    development_tasks=dev,
                    out=root / "out",
                    tau_python=sys.executable,
                    train_domain_quotas=one_each(),
                    development_domain_quotas=one_each(),
                )

    def test_symlink_input_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = make_fake_tau_repo(root / "tau")
            revision = git(repo, "rev-parse", "HEAD")
            train, dev = write_inputs(root, revision)
            link = root / "train-link.jsonl"
            link.symlink_to(train)

            with self.assertRaisesRegex(Tau3CaptureGenerationError, "symlink"):
                generate_tau3_training_captures(
                    tau_repo=repo,
                    expected_revision=revision,
                    train_tasks=link,
                    development_tasks=dev,
                    out=root / "out",
                    tau_python=sys.executable,
                    train_domain_quotas=one_each(),
                    development_domain_quotas=one_each(),
                )

    def test_output_overwrite_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = make_fake_tau_repo(root / "tau")
            revision = git(repo, "rev-parse", "HEAD")
            train, dev = write_inputs(root, revision)
            out = root / "out"
            out.mkdir()

            with self.assertRaisesRegex(Tau3CaptureGenerationError, "already exist"):
                generate_tau3_training_captures(
                    tau_repo=repo,
                    expected_revision=revision,
                    train_tasks=train,
                    development_tasks=dev,
                    out=out,
                    tau_python=sys.executable,
                    train_domain_quotas=one_each(),
                    development_domain_quotas=one_each(),
                )

    def test_envelope_revision_hash_and_sealed_row_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = make_fake_tau_repo(root / "tau")
            revision = git(repo, "rev-parse", "HEAD")
            train = root / "train.jsonl"
            dev = root / "development.jsonl"
            write_jsonl(train, [envelope("airline", "train", "0" * 40, task("airline", "airline-1"))])
            write_jsonl(dev, [envelope("retail", "development", revision, task("retail", "retail-1"))])
            with self.assertRaisesRegex(Tau3CaptureGenerationError, "source_revision mismatch"):
                generate_tau3_training_captures(
                    tau_repo=repo,
                    expected_revision=revision,
                    train_tasks=train,
                    development_tasks=dev,
                    out=root / "revision-out",
                    tau_python=sys.executable,
                    train_domain_quotas=one_each(),
                    development_domain_quotas=one_each(),
                )

            bad = envelope("airline", "train", revision, task("airline", "airline-1"))
            bad["task_sha256"] = "0" * 64
            write_jsonl(train, [bad])
            with self.assertRaisesRegex(Tau3CaptureGenerationError, "task_sha256 mismatch"):
                generate_tau3_training_captures(
                    tau_repo=repo,
                    expected_revision=revision,
                    train_tasks=train,
                    development_tasks=dev,
                    out=root / "hash-out",
                    tau_python=sys.executable,
                    train_domain_quotas=one_each(),
                    development_domain_quotas=one_each(),
                )

            sealed = envelope("airline", "train", revision, task("airline", "sealed-row"))
            sealed["task"]["official_split"] = "official_test"
            sealed["task_sha256"] = generation_hash(sealed["task"])
            write_jsonl(train, [sealed])
            with self.assertRaisesRegex(Tau3CaptureGenerationError, "sealed/test"):
                generate_tau3_training_captures(
                    tau_repo=repo,
                    expected_revision=revision,
                    train_tasks=train,
                    development_tasks=dev,
                    out=root / "sealed-out",
                    tau_python=sys.executable,
                    train_domain_quotas=one_each(),
                    development_domain_quotas=one_each(),
                )

    def test_duplicate_missing_action_and_tool_failure_rejections(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = make_fake_tau_repo(root / "tau")
            revision = git(repo, "rev-parse", "HEAD")
            train = root / "train.jsonl"
            dev = root / "development.jsonl"
            write_jsonl(train, [envelope("airline", "train", revision, task("airline", "same"))])
            write_jsonl(dev, [envelope("airline", "development", revision, task("airline", "same"))])
            with self.assertRaisesRegex(Tau3CaptureGenerationError, "duplicate"):
                generate_tau3_training_captures(
                    tau_repo=repo,
                    expected_revision=revision,
                    train_tasks=train,
                    development_tasks=dev,
                    out=root / "dup-out",
                    tau_python=sys.executable,
                    train_domain_quotas=one_each(),
                    development_domain_quotas=one_each(),
                )

            write_jsonl(train, [
                envelope("airline", "train", revision, task("airline", "no-actions", actions=[], communicate_info=["Please provide the missing confirmation."])),
                envelope("retail", "train", revision, task("retail", "retail-train-1")),
                envelope("telecom", "train", revision, task("telecom", "[mobile_data_issue]telecom-train-1")),
            ])
            write_jsonl(dev, [
                envelope("airline", "development", revision, task("airline", "airline-dev-1")),
                envelope("retail", "development", revision, task("retail", "retail-dev-1")),
                envelope("telecom", "development", revision, task("telecom", "[mobile_data_issue]telecom-dev-1")),
            ])
            summary = generate_tau3_training_captures(
                tau_repo=repo,
                expected_revision=revision,
                train_tasks=train,
                development_tasks=dev,
                out=root / "missing-actions",
                tau_python=sys.executable,
                train_domain_quotas=one_each(),
                development_domain_quotas=one_each(),
            )
            self.assertEqual(summary["capture_count"], 48)
            rows = read_jsonl(root / "missing-actions" / "captures.jsonl")
            grounded = [
                row for row in rows
                if row["task_id"] == "no-actions" and row["behavior"] == "clarification_refusal"
            ][0]
            self.assertTrue(grounded["outcome"]["success"])
            self.assertEqual(grounded["review"]["disposition"], "admit")
            action_required = [
                row for row in rows
                if row["task_id"] == "retail-train-1" and row["behavior"] == "clarification_refusal"
            ][0]
            self.assertFalse(action_required["outcome"]["success"])
            self.assertEqual(action_required["review"]["disposition"], "reject")

            write_jsonl(train, [
                envelope("airline", "train", revision, task("airline", "bad-tool", actions=[{"name": "fail_tool", "arguments": {}}])),
                envelope("retail", "train", revision, task("retail", "retail-train-1")),
                envelope("telecom", "train", revision, task("telecom", "[mobile_data_issue]telecom-train-1")),
            ])
            with self.assertRaisesRegex(Tau3CaptureGenerationError, "tool execution failed"):
                generate_tau3_training_captures(
                    tau_repo=repo,
                    expected_revision=revision,
                    train_tasks=train,
                    development_tasks=dev,
                    out=root / "bad-tool",
                    tau_python=sys.executable,
                    train_domain_quotas=one_each(),
                    development_domain_quotas=one_each(),
                )

            bad_candidate = envelope("airline", "train", revision, task("airline", "bad-tool", actions=[{"name": "fail_tool", "arguments": {}}]))
            good_candidate = envelope("airline", "train", revision, task("airline", "fallback-good"))
            write_jsonl(train, [
                bad_candidate,
                good_candidate,
                envelope("retail", "train", revision, task("retail", "retail-train-1")),
                envelope("telecom", "train", revision, task("telecom", "[mobile_data_issue]telecom-train-1")),
            ])
            import flightrecorder.tau3_capture_generation as generation
            fallback_salt = next(
                f"fallback-{index}"
                for index in range(1000)
                if generation._sha256_text(
                    f"fallback-{index}\0train\0airline\0{bad_candidate['task_family']}\0{bad_candidate['task_sha256']}"
                )
                < generation._sha256_text(
                    f"fallback-{index}\0train\0airline\0{good_candidate['task_family']}\0{good_candidate['task_sha256']}"
                )
            )
            fallback = generate_tau3_training_captures(
                tau_repo=repo,
                expected_revision=revision,
                train_tasks=train,
                development_tasks=dev,
                out=root / "fallback-good",
                tau_python=sys.executable,
                train_domain_quotas=one_each(),
                development_domain_quotas=one_each(),
                sample_salt=fallback_salt,
            )
            self.assertEqual(fallback["source_rejection_count"], 1)
            fallback_manifest = read_json(root / "fallback-good" / "manifest.json")
            self.assertEqual(fallback_manifest["sampling"]["splits"]["train"]["airline"]["rejected_candidate_count"], 1)
            self.assertEqual(fallback_manifest["sampling"]["splits"]["train"]["airline"]["selected_count"], 1)

    def test_paired_invalid_call_result_and_imbalance_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = make_fake_tau_repo(root / "tau")
            revision = git(repo, "rev-parse", "HEAD")
            train, dev = write_inputs(root, revision)
            out = root / "out"

            generate_tau3_training_captures(
                tau_repo=repo,
                expected_revision=revision,
                train_tasks=train,
                development_tasks=dev,
                out=out,
                tau_python=sys.executable,
                train_domain_quotas=one_each(),
                development_domain_quotas=one_each(),
            )

            rows = read_jsonl(out / "captures.jsonl")
            for behavior in ("hallucinated_tool", "recovery"):
                row = next(item for item in rows if item["behavior"] == behavior)
                invalid = [event for event in row["events"] if event.get("tool_name") == "invented_tau_tool"]
                self.assertEqual([event["type"] for event in invalid], ["tool_call", "tool_result"])
                self.assertEqual(invalid[0]["tool_call_id"], invalid[1]["tool_call_id"])

            train2 = root / "train-imbalanced.jsonl"
            dev2 = root / "dev-imbalanced.jsonl"
            write_jsonl(train2, [
                envelope("airline", "train", revision, task("airline", "airline-long", known_info=" ".join(["airline"] * 4000))),
                envelope("retail", "train", revision, task("retail", "retail-short")),
                envelope("telecom", "train", revision, task("telecom", "[mobile_data_issue]telecom-short")),
            ])
            write_jsonl(dev2, [
                envelope("airline", "development", revision, task("airline", "airline-dev-long", known_info=" ".join(["airline"] * 4000))),
                envelope("retail", "development", revision, task("retail", "retail-dev-short")),
                envelope("telecom", "development", revision, task("telecom", "[mobile_data_issue]telecom-dev-short")),
            ])
            with self.assertRaisesRegex(Tau3CaptureGenerationError, "domain token share exceeds"):
                generate_tau3_training_captures(
                    tau_repo=repo,
                    expected_revision=revision,
                    train_tasks=train2,
                    development_tasks=dev2,
                    out=root / "imbalanced",
                    tau_python=sys.executable,
                    train_domain_quotas=one_each(),
                    development_domain_quotas=one_each(),
                )


def write_inputs(root: Path, revision: str) -> tuple[Path, Path]:
    train = root / "train.jsonl"
    dev = root / "development.jsonl"
    write_jsonl(train, [
        envelope("airline", "train", revision, task("airline", "airline-1")),
        envelope("retail", "train", revision, task("retail", "retail-1")),
        envelope("telecom", "train", revision, task("telecom", "[mobile_data_issue]telecom-1")),
    ])
    write_jsonl(dev, [
        envelope("airline", "development", revision, task("airline", "airline-dev-1")),
        envelope("retail", "development", revision, task("retail", "retail-dev-1")),
        envelope("telecom", "development", revision, task("telecom", "[mobile_data_issue]telecom-dev-1")),
    ])
    return train, dev


def make_fake_tau_repo(repo: Path) -> Path:
    package = repo / "src" / "tau2"
    for sub in [
        package,
        package / "data_model",
        package / "environment",
        package / "domains",
        package / "domains" / "airline",
        package / "domains" / "retail",
        package / "domains" / "telecom",
    ]:
        sub.mkdir(parents=True, exist_ok=True)
        (sub / "__init__.py").write_text("", encoding="utf-8")
    (package / "data_model" / "message.py").write_text(
        textwrap.dedent(
            """
            from dataclasses import dataclass
            @dataclass
            class ToolCall:
                id: str
                name: str
                arguments: dict
                requestor: str = "assistant"
            @dataclass
            class AssistantMessage:
                role: str
                content: str | None = None
                tool_calls: list | None = None
            @dataclass
            class ToolMessage:
                id: str
                content: str
                requestor: str = "assistant"
                role: str = "tool"
                error: bool = False
            """
        ),
        encoding="utf-8",
    )
    (package / "data_model" / "tasks.py").write_text(
        textwrap.dedent(
            """
            from dataclasses import dataclass
            @dataclass
            class Action:
                action_id: str
                name: str
                arguments: dict
                requestor: str = "assistant"
            class Criteria:
                def __init__(self, actions):
                    self.actions = actions
            class Scenario:
                def __init__(self, raw):
                    self.raw = raw
                def __str__(self):
                    return str(self.raw)
            class Task:
                def __init__(self, raw):
                    self.id = raw["id"]
                    self.user_scenario = Scenario(raw.get("user_scenario", {}))
                    self.initial_state = None
                    actions = raw.get("evaluation_criteria", {}).get("actions") or []
                    self.evaluation_criteria = Criteria([
                        Action(
                            action_id=item.get("action_id", f"{self.id}-{i}"),
                            name=item["name"],
                            arguments=item.get("arguments", {}),
                            requestor=item.get("requestor", "assistant"),
                        )
                        for i, item in enumerate(actions)
                    ])
                @classmethod
                def model_validate(cls, raw):
                    return cls(raw)
            """
        ),
        encoding="utf-8",
    )
    (package / "fake_env.py").write_text(
        textwrap.dedent(
            """
            import json
            from types import SimpleNamespace
            from tau2.data_model.message import ToolMessage
            class FakeEnv:
                def __init__(self, domain):
                    self.domain = domain
                    self.state = {"calls": []}
                def set_state(self, initialization_data, initialization_actions, message_history, strict=True):
                    self.state["initialized"] = True
                def get_db_hash(self):
                    return json.dumps(self.state, sort_keys=True)
                def get_user_db_hash(self):
                    return None
                def get_policy(self):
                    return f"{self.domain} policy"
                def get_info(self, include_tool_info=False):
                    return SimpleNamespace(model_dump=lambda mode="json": {"tool_defs": {
                        "update_state": {"name": "update_state", "parameters": {"type": "object"}},
                        "fail_tool": {"name": "fail_tool", "parameters": {"type": "object"}},
                    }})
                def get_response(self, message):
                    if message.name == "fail_tool":
                        return ToolMessage(id=message.id, content="failed", requestor=message.requestor, error=True)
                    self.state["calls"].append({"name": message.name, "arguments": message.arguments})
                    return ToolMessage(id=message.id, content=json.dumps({"ok": True, "name": message.name}), requestor=message.requestor, error=False)
            """
        ),
        encoding="utf-8",
    )
    for domain in DOMAINS:
        (package / "domains" / domain / "environment.py").write_text(
            f"from tau2.fake_env import FakeEnv\n\ndef get_environment():\n    return FakeEnv({domain!r})\n",
            encoding="utf-8",
        )
    git(repo, "init")
    git(repo, "config", "user.email", "tau@example.test")
    git(repo, "config", "user.name", "Tau Test")
    git(repo, "add", "src")
    git(repo, "commit", "-m", "fixture")
    return repo


def task(domain: str, task_id: str, *, actions=None, communicate_info=None, known_info: str | None = None) -> dict:
    if actions is None:
        actions = [{"name": "update_state", "arguments": {"id": task_id}}]
    if communicate_info is None:
        communicate_info = []
    return {
        "id": task_id,
        "user_scenario": {
            "instructions": {
                "domain": domain,
                "reason_for_call": f"reason {task_id}",
                "known_info": known_info,
                "task_instructions": "do it",
            }
        },
        "evaluation_criteria": {
            "actions": [
                {"action_id": f"{task_id}-{index}", "requestor": "assistant", **action}
                for index, action in enumerate(actions)
            ],
            "communicate_info": communicate_info,
            "nl_assertions": [],
            "reward_basis": ["DB"],
        },
    }


def envelope(domain: str, split: str, revision: str, payload: dict) -> dict:
    import flightrecorder.tau3_capture_generation as generation

    return {
        "schema_version": "hfr.tau3_training_source.v1",
        "source_revision": revision,
        "domain": domain,
        "split": split,
        "task_family": generation._hash_json([action["name"] for action in payload["evaluation_criteria"]["actions"]]),
        "task_sha256": generation._hash_json(payload),
        "prompt_sha256": generation._hash_json(payload["user_scenario"]["instructions"]),
        "task": payload,
    }


def generation_hash(payload: dict) -> str:
    import flightrecorder.tau3_capture_generation as generation

    return generation._hash_json(payload)


def one_each() -> dict[str, int]:
    return {"airline": 1, "retail": 1, "telecom": 1}


def token_count_by_domain(captures: list[dict]) -> dict[str, int]:
    totals: dict[str, int] = {}
    for capture in captures:
        totals[capture["domain"]] = totals.get(capture["domain"], 0) + int(capture["token_count"])
    return {domain: totals[domain] for domain in sorted(totals)}


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def git(repo: Path, *args: str) -> str:
    completed = subprocess.run(["git", "-C", str(repo), *args], check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        raise AssertionError(completed.stderr or completed.stdout)
    return completed.stdout.strip()


if __name__ == "__main__":
    unittest.main()
