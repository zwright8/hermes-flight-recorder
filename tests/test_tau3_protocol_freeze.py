import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from flightrecorder.tau3_capture import canonical_sha256
from flightrecorder.tau3_model_identity import build_tau3_model_identity
from flightrecorder.tau3_protocol_freeze import (
    MODEL_SPECS,
    Tau3ProtocolFreezeError,
    freeze_tau3_training_protocol,
)


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "freeze_tau3_training_protocol.py"


class Tau3ProtocolFreezeTests(unittest.TestCase):
    def test_deterministic_success_and_cli(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = make_fixture(Path(tmp))
            out_a = Path(tmp) / "protocol-a.json"
            out_b = Path(tmp) / "protocol-b.json"

            summary_a = freeze_tau3_training_protocol(out=out_a, **fixture)
            completed = subprocess.run(
                [sys.executable, str(SCRIPT), *cli_args(fixture, out_b)],
                check=False,
                capture_output=True,
                text=True,
                cwd=ROOT,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            summary_b = json.loads(completed.stdout)
            self.assertEqual(summary_a["protocol_sha256"], summary_b["protocol_sha256"])
            self.assertEqual(out_a.read_bytes(), out_b.read_bytes())
            self.assertEqual(out_a.stat().st_mode & 0o777, 0o600)

            protocol = read_json(out_a)
            self.assertNotIn("REPLACE_WITH_", json.dumps(protocol, sort_keys=True))
            self.assertTrue(protocol["mlx_qlora_plan"]["passed"])
            self.assertTrue(protocol["candidate_selection_contract"]["passed"])
            self.assertEqual(protocol["harness_contract"]["context_window"], 16384)
            self.assertEqual(protocol["harness_contract"]["decoding"]["temperature"], 0.0)
            self.assertEqual(protocol["harness_contract"]["decoding"]["top_p"], 1.0)
            self.assertEqual(protocol["harness_contract"]["decoding"]["max_output_tokens"], 1024)
            self.assertEqual(protocol["harness_contract"]["decoding"]["seeds"], [101, 202, 303, 404])
            self.assertEqual(protocol["harness_contract"]["turn_limit"], 30)
            self.assertFalse(protocol["harness_contract"]["test_time_search"])
            self.assertEqual(protocol["model_freeze"]["base_model"]["name"], MODEL_SPECS["base"]["name"])
            self.assertEqual(protocol["model_freeze"]["base_model"]["revision"], MODEL_SPECS["base"]["revision"])
            self.assertEqual(
                [row["name"] for row in protocol["model_freeze"]["comparators"]],
                [MODEL_SPECS["comparator-1"]["name"], MODEL_SPECS["comparator-2"]["name"]],
            )
            self.assertEqual(
                protocol["mlx_qlora_plan"]["command_argv"],
                [
                    "python",
                    "-m",
                    "mlx_lm",
                    "lora",
                    "--train",
                    "--fine-tune-type",
                    "lora",
                    "--model",
                    "model_input",
                    "--data",
                    "input_export",
                    "--adapter-path",
                    "adapter_output",
                    "--batch-size",
                    "1",
                    "--grad-accumulation-steps",
                    "8",
                    "--max-seq-length",
                    "12288",
                    "--learning-rate",
                    "5e-5",
                    "--iters",
                    "200",
                    "--seed",
                    "8675309",
                    "--grad-checkpoint",
                ],
            )
            self.assertEqual(protocol["schema_version"], "hfr.tau3_protocol_config.v1")
            self.assertEqual(protocol["mlx_qlora_plan"]["mlx_lm_version"], "0.31.3")
            self.assertEqual(protocol["mlx_qlora_plan"]["mlx_version"], "0.32.0")
            self.assertEqual(protocol["mlx_qlora_plan"]["data_layout"]["required_files"], ["train.jsonl", "valid.jsonl"])
            self.assertFalse(protocol["mlx_qlora_plan"]["data_layout"]["test_file_required"])
            self.assertEqual(set(protocol["harness_contract"]["domain_contracts"]), {"airline", "retail", "telecom"})
            self.assertIn("--test", protocol["mlx_qlora_plan"]["forbidden_flags"])
            self.assertIn("pre_run_eligibility_rule", protocol["model_freeze"])
            self.assertFalse(protocol["model_freeze"]["benchmark_superiority_claimed"])
            self.assertIn("evidence_urls", protocol["model_freeze"]["base_model"])
            self.assertEqual(protocol["model_freeze"]["base_model"]["upstream"]["name"], "Qwen/Qwen3.5-9B")
            capture_freeze = protocol["split_manifest"]["training_captures"]
            self.assertEqual(capture_freeze["row_count"], 8)
            self.assertEqual(capture_freeze["sha256"], artifact_record(Path(fixture["captures"]))["sha256"])
            self.assertEqual(capture_freeze["admitted_count"], 8)

    def test_output_overwrite_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = make_fixture(Path(tmp))
            out = Path(tmp) / "protocol.json"
            out.write_text("{}", encoding="utf-8")

            with self.assertRaisesRegex(Tau3ProtocolFreezeError, "already exist"):
                freeze_tau3_training_protocol(out=out, **fixture)

    def test_dirty_tau_checkout_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = make_fixture(Path(tmp))
            Path(fixture["tau_repo"], "dirty.txt").write_text("dirty\n", encoding="utf-8")

            with self.assertRaisesRegex(Tau3ProtocolFreezeError, "clean"):
                freeze_tau3_training_protocol(out=Path(tmp) / "protocol.json", **fixture)

    def test_wrong_tau_revision_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = make_fixture(Path(tmp))
            fixture["tau_revision"] = "0" * 40

            with self.assertRaisesRegex(Tau3ProtocolFreezeError, "revision mismatch"):
                freeze_tau3_training_protocol(out=Path(tmp) / "protocol.json", **fixture)

    def test_model_identity_tamper_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = make_fixture(Path(tmp))
            identity = read_json(Path(fixture["base_identity"]))
            identity["revision"] = "0" * 40
            write_json(Path(fixture["base_identity"]), identity)

            with self.assertRaisesRegex(Tau3ProtocolFreezeError, "identity does not replay"):
                freeze_tau3_training_protocol(out=Path(tmp) / "protocol.json", **fixture)

    def test_symlink_identity_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture = make_fixture(root)
            link = root / "base-link.json"
            link.symlink_to(Path(fixture["base_identity"]))
            fixture["base_identity"] = link

            with self.assertRaisesRegex(Tau3ProtocolFreezeError, "symlink"):
                freeze_tau3_training_protocol(out=root / "protocol.json", **fixture)

    def test_sealed_raw_payload_leak_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = make_fixture(Path(tmp))
            sealed = read_json(Path(fixture["sealed_split"]))
            sealed["entries"][0]["raw_id"] = "sealed-task-raw-id"
            write_json(Path(fixture["sealed_split"]), sealed)
            refresh_source_manifest_hash(fixture, "sealed.json", Path(fixture["sealed_split"]))

            with self.assertRaisesRegex(Tau3ProtocolFreezeError, "hash"):
                freeze_tau3_training_protocol(out=Path(tmp) / "protocol.json", **fixture)

    def test_sealed_task_identity_overlap_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = make_fixture(Path(tmp))
            train = read_json(Path(fixture["train_split"]))
            sealed = read_json(Path(fixture["sealed_split"]))
            sealed["entries"][0]["task_sha256"] = train["tasks"][0]["task_sha256"]
            write_json(Path(fixture["sealed_split"]), sealed)
            refresh_source_manifest_hash(fixture, "sealed.json", Path(fixture["sealed_split"]))

            with self.assertRaisesRegex(Tau3ProtocolFreezeError, "contamination"):
                freeze_tau3_training_protocol(out=Path(tmp) / "protocol.json", **fixture)

    def test_shared_official_prompt_template_is_reported_but_not_task_leakage(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = make_fixture(Path(tmp))
            train = read_json(Path(fixture["train_split"]))
            sealed = read_json(Path(fixture["sealed_split"]))
            sealed["entries"][0]["prompt_sha256"] = train["tasks"][0]["prompt_sha256"]
            write_json(Path(fixture["sealed_split"]), sealed)
            refresh_source_manifest_hash(fixture, "sealed.json", Path(fixture["sealed_split"]))

            freeze_tau3_training_protocol(out=Path(tmp) / "protocol.json", **fixture)
            protocol = read_json(Path(tmp) / "protocol.json")
            contamination = protocol["contamination_attestation"]
            self.assertTrue(contamination["passed"])
            self.assertEqual(contamination["evidence"]["sealed_prompt_template_overlap_count"], 1)
            self.assertTrue(contamination["evidence"]["sealed_prompt_template_overlap_resolved"])

    def test_duplicate_capture_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = make_fixture(Path(tmp))
            rows = read_jsonl(Path(fixture["captures"]))
            rows.append(dict(rows[0]))
            write_jsonl(Path(fixture["captures"]), rows)

            with self.assertRaisesRegex(Tau3ProtocolFreezeError, "duplicate"):
                freeze_tau3_training_protocol(out=Path(tmp) / "protocol.json", **fixture)

    def test_secret_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = make_fixture(Path(tmp))
            rows = read_jsonl(Path(fixture["captures"]))
            rows[0]["prompt"] += " api_key=sk-testsecretsecretsecret"
            rows[0]["prompt_hash"] = canonical_sha256(rows[0]["prompt"])
            write_jsonl(Path(fixture["captures"]), rows)

            with self.assertRaisesRegex(Tau3ProtocolFreezeError, "redaction"):
                freeze_tau3_training_protocol(out=Path(tmp) / "protocol.json", **fixture)

    def test_capture_source_mismatch_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = make_fixture(Path(tmp))
            rows = read_jsonl(Path(fixture["captures"]))
            rows[0]["source_task_sha256"] = canonical_sha256("wrong-source")
            write_jsonl(Path(fixture["captures"]), rows)

            with self.assertRaisesRegex(Tau3ProtocolFreezeError, "permitted source"):
                freeze_tau3_training_protocol(out=Path(tmp) / "protocol.json", **fixture)

    def test_source_schema_mismatch_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = make_fixture(Path(tmp))
            train = read_json(Path(fixture["train_split"]))
            train["schema_version"] = "old"
            write_json(Path(fixture["train_split"]), train)
            refresh_source_manifest_hash(fixture, "train.json", Path(fixture["train_split"]))

            with self.assertRaisesRegex(Tau3ProtocolFreezeError, "schema_version"):
                freeze_tau3_training_protocol(out=Path(tmp) / "protocol.json", **fixture)

    def test_cross_family_template_collision_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = make_fixture(Path(tmp))
            rows = read_jsonl(Path(fixture["captures"]))
            rows[1]["prompt"] = rows[0]["prompt"]
            rows[1]["prompt_hash"] = canonical_sha256(rows[1]["prompt"])
            write_jsonl(Path(fixture["captures"]), rows)

            with self.assertRaisesRegex(Tau3ProtocolFreezeError, "contamination"):
                freeze_tau3_training_protocol(out=Path(tmp) / "protocol.json", **fixture)

    def test_missing_behavior_and_domain_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = make_fixture(Path(tmp))
            rows = [row for row in read_jsonl(Path(fixture["captures"])) if row["domain"] != "telecom" and row["behavior"] != "recovery"]
            write_jsonl(Path(fixture["captures"]), rows)

            with self.assertRaisesRegex(Tau3ProtocolFreezeError, "domains"):
                freeze_tau3_training_protocol(out=Path(tmp) / "protocol.json", **fixture)


def make_fixture(root: Path) -> dict:
    tau_repo = make_tau_repo(root / "tau")
    tau_revision = git(tau_repo, "rev-parse", "HEAD")
    sources = root / "sources"
    make_sources(sources, tau_revision)
    models = root / "models"
    identities = root / "identities"
    identities.mkdir()
    model_args = {}
    for key, arg_prefix in (
        ("base", "base"),
        ("comparator-1", "comparator1"),
        ("comparator-2", "comparator2"),
    ):
        model_path = models / key
        make_model(model_path)
        spec = MODEL_SPECS[key]
        identity = build_tau3_model_identity(model_path, model_id=spec["name"], revision=spec["revision"])
        identity_path = identities / f"{key}.json"
        write_json(identity_path, identity)
        model_args[f"{arg_prefix}_identity"] = identity_path
        model_args[f"{arg_prefix}_model_path"] = model_path
    captures = root / "captures.jsonl"
    write_jsonl(captures, fake_captures())
    return {
        "tau_repo": tau_repo,
        "tau_revision": tau_revision,
        "source_manifest": sources / "manifest.json",
        "train_split": sources / "train.json",
        "development_split": sources / "development.json",
        "sealed_split": sources / "sealed.json",
        "train_tasks": sources / "training_source" / "train_tasks.jsonl",
        "development_tasks": sources / "training_source" / "development_tasks.jsonl",
        "captures": captures,
        **model_args,
    }


def make_tau_repo(repo: Path) -> Path:
    repo.mkdir()
    git(repo, "init")
    git(repo, "config", "user.email", "tau@example.test")
    git(repo, "config", "user.name", "Tau Test")
    (repo / "README.md").write_text("tau fixture\n", encoding="utf-8")
    git(repo, "add", "README.md")
    git(repo, "commit", "-m", "fixture")
    return repo


def make_sources(out: Path, revision: str) -> None:
    (out / "training_source").mkdir(parents=True)
    train_tasks = [
        {"id": "airline-train-1", "user_scenario": {"instructions": "Book a safe flight for customer A"}},
        {"id": "retail-train-1", "user_scenario": {"instructions": "Return a damaged lamp for customer B"}},
    ]
    development_tasks = [{"id": "telecom-dev-1", "user_scenario": {"instructions": "Fix voicemail issue for customer C"}}]
    train = split_manifest("train", revision, train_tasks, ["family-a", "family-b"])
    development = split_manifest("development", revision, development_tasks, ["family-c"])
    write_jsonl(out / "training_source" / "train_tasks.jsonl", envelopes("train", revision, train_tasks, ["family-a", "family-b"]))
    write_jsonl(out / "training_source" / "development_tasks.jsonl", envelopes("development", revision, development_tasks, ["family-c"]))
    sealed = {
        "schema_version": "hfr.tau3_sealed_source_manifest.v1",
        "source_revision": revision,
        "hashes_only": True,
        "task_count": 1,
        "entries": [
            {
                "task_id_sha256": canonical_sha256("sealed-id"),
                "prompt_sha256": canonical_sha256("sealed prompt"),
                "task_sha256": canonical_sha256({"sealed": True}),
            }
        ],
    }
    write_json(out / "train.json", train)
    write_json(out / "development.json", development)
    write_json(out / "sealed.json", sealed)
    artifacts = {
        rel: artifact_record(out / rel)
        for rel in (
            "train.json",
            "development.json",
            "training_source/train_tasks.jsonl",
            "training_source/development_tasks.jsonl",
            "sealed.json",
        )
    }
    write_json(
        out / "manifest.json",
        {
            "schema_version": "hfr.tau3_source_partition.v1",
            "source_revision": revision,
            "task_schema_version": "tau2.tasks.v1",
            "proofs": {
                "train_development_family_disjoint": True,
                "sealed_payload_non_materialization": True,
                "sealed_payload_files": [],
                "official_test_sealed": True,
            },
            "artifacts": artifacts,
        },
    )


def split_manifest(split: str, revision: str, tasks: list[dict], families: list[str]) -> dict:
    rows = []
    for task, family in zip(tasks, families):
        rows.append(
            {
                "domain": task["id"].split("-", 1)[0],
                "raw_id": task["id"],
                "raw_id_sha256": canonical_sha256(task["id"]),
                "prompt_sha256": canonical_sha256(task["user_scenario"]["instructions"]),
                "task_sha256": canonical_sha256(task),
                "family_id": family,
            }
        )
    return {
        "schema_version": "hfr.tau3_source_split.v1",
        "split": split,
        "source_revision": revision,
        "task_schema_version": "tau2.tasks.v1",
        "task_count": len(tasks),
        "family_count": len(families),
        "family_ids": families,
        "tasks": rows,
    }


def envelopes(split: str, revision: str, tasks: list[dict], families: list[str]) -> list[dict]:
    rows = []
    for task, family in zip(tasks, families):
        rows.append(
            {
                "schema_version": "hfr.tau3_training_source.v1",
                "source_revision": revision,
                "domain": task["id"].split("-", 1)[0],
                "split": split,
                "task_family": family,
                "task_sha256": canonical_sha256(task),
                "prompt_sha256": canonical_sha256(task["user_scenario"]["instructions"]),
                "task": task,
            }
        )
    return rows


def make_model(path: Path) -> None:
    path.mkdir(parents=True)
    (path / "config.json").write_text('{"model_type":"fixture"}\n', encoding="utf-8")
    (path / "tokenizer.json").write_text('{"tokenizer":"fixture"}\n', encoding="utf-8")
    (path / "weights.safetensors").write_bytes(b"fixture-weights")


def fake_captures() -> list[dict]:
    source = {
        "airline-train-1": source_hashes({"id": "airline-train-1", "user_scenario": {"instructions": "Book a safe flight for customer A"}}),
        "retail-train-1": source_hashes({"id": "retail-train-1", "user_scenario": {"instructions": "Return a damaged lamp for customer B"}}),
        "telecom-dev-1": source_hashes({"id": "telecom-dev-1", "user_scenario": {"instructions": "Fix voicemail issue for customer C"}}),
    }
    return [
        capture("airline", "airline-train-1", "family-a", "success", "Book a safe flight for customer A", "lookup_airline", "before-a", "after-a", **source["airline-train-1"]),
        capture("retail", "retail-train-1", "family-b", "correction", "Return a damaged lamp for customer B", "lookup_order", "before-b", "after-b", **source["retail-train-1"]),
        capture("telecom", "telecom-dev-1", "family-c", "clarification_refusal", "Fix voicemail issue for customer C", "lookup_line", "before-c", "after-c", split="development", **source["telecom-dev-1"]),
        capture("airline", "airline-train-1", "family-a", "recovery", "Book a safe flight for customer A", "lookup_airline", "before-a2", "after-a2", **source["airline-train-1"]),
        capture("retail", "retail-train-1", "family-b", "policy_failure", "Return a damaged lamp for customer B", "lookup_order", "before-b2", "after-b2", **source["retail-train-1"]),
        capture("telecom", "telecom-dev-1", "family-c", "harmful_mutation", "Fix voicemail issue for customer C", "lookup_line", "before-c2", "after-c2", split="development", **source["telecom-dev-1"]),
        capture("airline", "airline-train-1", "family-a", "hallucinated_tool", "Book a safe flight for customer A", "lookup_airline", "before-a3", "after-a3", **source["airline-train-1"]),
        capture("retail", "retail-train-1", "family-b", "premature_completion", "Return a damaged lamp for customer B", "lookup_order", "before-b3", "after-b3", **source["retail-train-1"]),
    ]


def source_hashes(task: dict) -> dict:
    return {
        "source_task_sha256": canonical_sha256(task),
        "source_prompt_sha256": canonical_sha256(task["user_scenario"]["instructions"]),
    }


def capture(
    domain: str,
    task_id: str,
    family: str,
    behavior: str,
    prompt: str,
    tool: str,
    before: str,
    after: str,
    *,
    source_task_sha256: str,
    source_prompt_sha256: str,
    split: str = "train",
) -> dict:
    before_hash = canonical_sha256(before)
    after_hash = canonical_sha256(after)
    trajectory_id = f"{domain}-{task_id}-{split}-{behavior}"
    return {
        "schema_version": "hfr.tau3_capture.v1",
        "trajectory_id": trajectory_id,
        "task_id": task_id,
        "task_family": family,
        "domain": domain,
        "split": split,
        "behavior": behavior,
        "prompt": prompt,
        "prompt_hash": canonical_sha256(prompt),
        "source_task_sha256": source_task_sha256,
        "source_prompt_sha256": source_prompt_sha256,
        "seed": len(prompt),
        "generator_id": "fixture-generator",
        "generator_revision": "1" * 40,
        "policy_revision": f"{domain}-policy-v1",
        "policy_hash": canonical_sha256(f"{domain}-policy-v1"),
        "tool_schema_revision": f"{domain}-tools-v1",
        "starting_state_hash": before_hash,
        "token_count": 100,
        "tools": [{"name": tool, "description": "fixture tool", "parameters": {"type": "object"}}],
        "events": [
            {"type": "user_message", "role": "user", "content": prompt, "text": prompt},
            {"type": "tool_call", "role": "assistant", "tool_name": tool, "tool_call_id": f"call-{task_id}", "args": {"id": task_id}, "status": "requested"},
            {"type": "tool_result", "role": "tool", "tool_name": tool, "tool_call_id": f"call-{task_id}", "result": {"ok": True}, "status": "ok"},
            {"type": "assistant_message", "role": "assistant", "content": "Done with verified state.", "text": "Done with verified state."},
        ],
        "state_transition": {
            "before_hash": before_hash,
            "after_hash": after_hash,
            "changes": [{"path": task_id, "kind": "changed", "before": before_hash, "after": after_hash}],
            "executable": True,
        },
        "outcome": {"success": True, "executable_label": "success", "policy_violation": False, "harmful_mutation": False, "evidence_refs": ["tool_result"]},
        "review": {"reviewer": "fixture-reviewer", "verifier": "fixture-verifier", "disposition": "admit", "reason": "fixture"},
    }


def cli_args(fixture: dict, out: Path) -> list[str]:
    mapping = {
        "tau_repo": "--tau-repo",
        "tau_revision": "--tau-revision",
        "source_manifest": "--source-manifest",
        "train_split": "--train-split",
        "development_split": "--development-split",
        "sealed_split": "--sealed-split",
        "train_tasks": "--train-tasks",
        "development_tasks": "--development-tasks",
        "base_identity": "--base-identity",
        "base_model_path": "--base-model-path",
        "comparator1_identity": "--comparator1-identity",
        "comparator1_model_path": "--comparator1-model-path",
        "comparator2_identity": "--comparator2-identity",
        "comparator2_model_path": "--comparator2-model-path",
        "captures": "--captures",
    }
    args: list[str] = []
    for key, flag in mapping.items():
        args.extend([flag, str(fixture[key])])
    args.extend(["--out", str(out)])
    return args


def artifact_record(path: Path) -> dict:
    data = path.read_bytes()
    return {"size": len(data), "sha256": __import__("hashlib").sha256(data).hexdigest()}


def refresh_source_manifest_hash(fixture: dict, rel: str, path: Path) -> None:
    manifest_path = Path(fixture["source_manifest"])
    manifest = read_json(manifest_path)
    manifest["artifacts"][rel] = artifact_record(path)
    write_json(manifest_path, manifest)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def git(repo: Path, *args: str) -> str:
    completed = subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True)
    return completed.stdout.strip()


if __name__ == "__main__":
    unittest.main()
