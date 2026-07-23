import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from flightrecorder.tau3_source_partition import (
    Tau3SourcePartitionError,
    prepare_tau3_training_sources,
)
from flightrecorder.schema_registry import check_schema_contract


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "prepare_tau3_training_sources.py"
DOMAINS = ("airline", "retail", "telecom")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


class Tau3SourcePartitionTests(unittest.TestCase):
    def test_deterministic_success_and_cli_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = make_tau_repo(Path(tmp) / "tau")
            revision = git(repo, "rev-parse", "HEAD")
            out_a = Path(tmp) / "out-a"
            out_b = Path(tmp) / "out-b"

            summary = prepare_tau3_training_sources(repo, revision, out_a)
            completed = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--tau-repo",
                    str(repo),
                    "--expected-revision",
                    revision,
                    "--out",
                    str(out_b),
                ],
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            cli_summary = json.loads(completed.stdout)
            self.assertEqual(cli_summary["train_task_count"], summary["train_task_count"])
            self.assertEqual((out_a / "manifest.json").read_bytes(), (out_b / "manifest.json").read_bytes())
            self.assertEqual((out_a / "train.json").read_bytes(), (out_b / "train.json").read_bytes())
            self.assertEqual((out_a / "development.json").read_bytes(), (out_b / "development.json").read_bytes())

            manifest = read_json(out_a / "manifest.json")
            train = read_json(out_a / "train.json")
            development = read_json(out_a / "development.json")
            train_payloads = read_jsonl(out_a / "training_source" / "train_tasks.jsonl")
            development_payloads = read_jsonl(out_a / "training_source" / "development_tasks.jsonl")

            self.assertTrue(manifest["proofs"]["train_development_family_disjoint"])
            self.assertTrue(manifest["proofs"]["sealed_payload_non_materialization"])
            self.assertEqual(manifest["proofs"]["sealed_payload_files"], [])
            self.assertEqual(train["task_count"], len(train_payloads))
            self.assertEqual(development["task_count"], len(development_payloads))
            self.assertGreater(train["task_count"], 0)
            self.assertGreater(development["task_count"], 0)
            self.assertTrue(set(train["family_ids"]).isdisjoint(development["family_ids"]))
            self.assertTrue(all(row["split"] == "train" for row in train_payloads))
            self.assertTrue(all(row["split"] == "development" for row in development_payloads))
            self.assertEqual({row["domain"] for row in train_payloads}, set(DOMAINS))
            self.assertEqual({row["domain"] for row in development_payloads}, set(DOMAINS))
            self.assertTrue(all(isinstance(row["task"], dict) for row in train_payloads))
            self.assertTrue(all(row["source_revision"] == revision for row in train_payloads))
            self.assertTrue(check_schema_contract(manifest, name_or_id="tau3_source_partition")["passed"])
            self.assertTrue(check_schema_contract(train, name_or_id="tau3_source_split")["passed"])
            self.assertTrue(check_schema_contract(development, name_or_id="tau3_source_split")["passed"])
            self.assertTrue(check_schema_contract(train_payloads[0], name_or_id="tau3_training_source")["passed"])
            for rel, record in manifest["artifacts"].items():
                path = out_a / rel
                self.assertEqual(path.stat().st_size, record["size"])
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), record["sha256"])
                self.assertEqual(path.stat().st_mode & 0o777, 0o600)

    def test_sealed_manifest_does_not_leak_payload_or_raw_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = make_tau_repo(Path(tmp) / "tau")
            revision = git(repo, "rev-parse", "HEAD")
            out = Path(tmp) / "out"

            prepare_tau3_training_sources(repo, revision, out)

            sealed_text = (out / "sealed.json").read_text(encoding="utf-8")
            self.assertNotIn("airline-test-raw-id", sealed_text)
            self.assertNotIn("secret sealed prompt", sealed_text)
            self.assertNotIn("expected_action", sealed_text)
            self.assertFalse((out / "training_source" / "sealed_tasks.jsonl").exists())
            sealed = read_json(out / "sealed.json")
            self.assertTrue(
                check_schema_contract(sealed, name_or_id="tau3_sealed_source_manifest")["passed"]
            )
            self.assertTrue(sealed["hashes_only"])
            self.assertEqual(sealed["task_count"], len(sealed["entries"]))
            self.assertEqual(
                set(sealed["entries"][0]),
                {"prompt_sha256", "task_id_sha256", "task_sha256"},
            )

    def test_wrong_revision_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = make_tau_repo(Path(tmp) / "tau")
            wrong = "0" * 40
            with self.assertRaisesRegex(Tau3SourcePartitionError, "revision mismatch"):
                prepare_tau3_training_sources(repo, wrong, Path(tmp) / "out")

    def test_unrelated_official_domains_are_ignored(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = make_tau_repo(root / "tau")
            extra = repo / "data" / "tau2" / "domains" / "banking_knowledge"
            extra.mkdir()
            (extra / "README.md").write_text("unrelated domain\n", encoding="utf-8")
            git(repo, "add", "data")
            git(repo, "commit", "-m", "add unrelated domain")
            revision = git(repo, "rev-parse", "HEAD")

            summary = prepare_tau3_training_sources(repo, revision, root / "out")

            self.assertEqual(summary["official_train_task_count"], 12)

    def test_dirty_checkout_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = make_tau_repo(Path(tmp) / "tau")
            revision = git(repo, "rev-parse", "HEAD")
            (repo / "data" / "tau2" / "domains" / "airline" / "tasks.json").write_text("[]\n", encoding="utf-8")

            with self.assertRaisesRegex(Tau3SourcePartitionError, "clean"):
                prepare_tau3_training_sources(repo, revision, Path(tmp) / "out")

    def test_split_overlap_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = make_tau_repo(Path(tmp) / "tau", mutate_split=lambda split: split["test"].append(split["train"][0]))
            revision = git(repo, "rev-parse", "HEAD")

            with self.assertRaisesRegex(Tau3SourcePartitionError, "overlap"):
                prepare_tau3_training_sources(repo, revision, Path(tmp) / "out")

    def test_unresolved_id_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = make_tau_repo(
                Path(tmp) / "tau",
                mutate_split=lambda split: (split["train"].append("missing"), split["base"].append("missing")),
            )
            revision = git(repo, "rev-parse", "HEAD")

            with self.assertRaisesRegex(Tau3SourcePartitionError, "unresolved"):
                prepare_tau3_training_sources(repo, revision, Path(tmp) / "out")

    def test_output_overwrite_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = make_tau_repo(Path(tmp) / "tau")
            revision = git(repo, "rev-parse", "HEAD")
            out = Path(tmp) / "out"
            out.mkdir()

            with self.assertRaisesRegex(Tau3SourcePartitionError, "already exist"):
                prepare_tau3_training_sources(repo, revision, out)

    def test_symlink_output_component_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = make_tau_repo(root / "tau")
            revision = git(repo, "rev-parse", "HEAD")
            real_parent = root / "real"
            real_parent.mkdir()
            link_parent = root / "link"
            link_parent.symlink_to(real_parent, target_is_directory=True)

            with self.assertRaisesRegex(Tau3SourcePartitionError, "symlink"):
                prepare_tau3_training_sources(repo, revision, link_parent / "out")

    def test_symlink_repository_component_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = make_tau_repo(root / "tau")
            revision = git(repo, "rev-parse", "HEAD")
            link = root / "tau-link"
            link.symlink_to(repo, target_is_directory=True)

            with self.assertRaisesRegex(Tau3SourcePartitionError, "symlink"):
                prepare_tau3_training_sources(link, revision, root / "out")

    def test_empty_development_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = make_tau_repo(Path(tmp) / "tau", empty_train=True)
            revision = git(repo, "rev-parse", "HEAD")

            with self.assertRaisesRegex(Tau3SourcePartitionError, "at least two task families"):
                prepare_tau3_training_sources(repo, revision, Path(tmp) / "out")


def make_tau_repo(
    repo: Path,
    *,
    mutate_split=None,
    one_family: bool = False,
    empty_train: bool = False,
) -> Path:
    for domain in DOMAINS:
        domain_root = repo / "data" / "tau2" / "domains" / domain
        domain_root.mkdir(parents=True, exist_ok=True)
        tasks = fake_tasks(domain, one_family=one_family)
        if empty_train:
            split = {"train": [], "test": [task["id"] for task in tasks], "base": [task["id"] for task in tasks]}
        else:
            split = {"train": [task["id"] for task in tasks[:4]], "test": [task["id"] for task in tasks[4:]], "base": [task["id"] for task in tasks]}
        if domain == "airline" and mutate_split is not None:
            mutate_split(split)
        write_json(domain_root / "tasks.json", tasks)
        write_json(domain_root / "split_tasks.json", split)
    git(repo, "init")
    git(repo, "config", "user.email", "tau@example.test")
    git(repo, "config", "user.name", "Tau Test")
    git(repo, "add", "data")
    git(repo, "commit", "-m", "fixture")
    return repo


def fake_tasks(domain: str, *, one_family: bool = False) -> list[dict]:
    if domain == "telecom":
        ids = [
            "[mobile_data_issue]slow_none[PERSONA:None]",
            "[mobile_data_issue]slow_easy[PERSONA:Easy]",
            "[service_issue]sim_none[PERSONA:None]",
            "[service_issue]sim_easy[PERSONA:Easy]",
            "[mms_issue]mms_none[PERSONA:None]",
            "[mms_issue]mms_easy[PERSONA:Easy]",
        ]
        if one_family:
            ids = [f"[mobile_data_issue]case_{index}[PERSONA:None]" for index in range(6)]
        return [task(domain, task_id, actions=[f"telecom_tool_{index % 2}"]) for index, task_id in enumerate(ids)]
    prefix = f"{domain}-"
    if one_family:
        action_sets = [["same_tool"], ["same_tool"], ["same_tool"], ["same_tool"], ["same_tool"], ["same_tool"]]
    else:
        action_sets = [
            [],
            ["lookup_user"],
            ["lookup_user", "update_order"],
            ["lookup_user", "refund"],
            ["expected_action"],
            ["lookup_user", "seal_only"],
        ]
    ids = [f"{prefix}{index}" for index in range(6)]
    if domain == "airline":
        ids[4] = "airline-test-raw-id"
    return [task(domain, task_id, actions=actions) for task_id, actions in zip(ids, action_sets)]


def task(domain: str, task_id: str, *, actions: list[str]) -> dict:
    return {
        "id": task_id,
        "description": {"purpose": f"{domain} purpose", "relevant_policies": None, "notes": None},
        "user_scenario": {
            "persona": None,
            "instructions": {
                "domain": domain,
                "reason_for_call": "secret sealed prompt" if task_id == "airline-test-raw-id" else f"{domain} reason {task_id}",
                "known_info": f"{domain} known {task_id}",
                "unknown_info": None,
                "task_instructions": f"{domain} instructions {task_id}",
            },
        },
        "initial_state": None,
        "evaluation_criteria": {
            "actions": [
                {"action_id": f"{task_id}_{index}", "name": name, "arguments": {"id": task_id}, "info": None}
                for index, name in enumerate(actions)
            ],
            "communicate_info": [],
            "nl_assertions": ["Agent should refuse."] if not actions else ["Agent should act."],
            "reward_basis": ["DB", "COMMUNICATE"],
        },
        "annotations": None,
    }


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise AssertionError(completed.stderr or completed.stdout)
    return completed.stdout.strip()


if __name__ == "__main__":
    unittest.main()
