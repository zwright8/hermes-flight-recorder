from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from flightrecorder.schema_registry import check_schema_contract
from flightrecorder.tau3_model_identity import (
    Tau3ModelIdentityError,
    build_tau3_model_identity,
    validate_tau3_model_identity,
)
from scripts.build_tau3_training_artifacts import _local_model_identity_matches, _production_source_checks
from scripts.check_tau3_training_sources import check_tau3_training_sources


ROOT = Path(__file__).resolve().parents[1]


class Tau3ModelIdentityTests(unittest.TestCase):
    def _model_tree(self, root: Path) -> Path:
        model = root / "model"
        model.mkdir()
        (model / "config.json").write_text('{"model_type":"fixture"}\n', encoding="utf-8")
        (model / "tokenizer.json").write_text('{"version":"1"}\n', encoding="utf-8")
        (model / "model-00001-of-00001.safetensors").write_bytes(b"fixture-weights")
        return model

    def test_identity_replays_complete_model_tree_and_schema(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model = self._model_tree(Path(tmp))
            identity = build_tau3_model_identity(
                model,
                model_id="local/fixture-8b",
                revision="a" * 40,
            )

            self.assertEqual(
                validate_tau3_model_identity(
                    identity,
                    model,
                    expected_model_id="local/fixture-8b",
                    expected_revision="a" * 40,
                ),
                [],
            )
            self.assertTrue(check_schema_contract(identity, name_or_id="tau3_model_identity")["passed"])
            self.assertEqual(identity["file_count"], 3)

    def test_identity_rejects_changed_missing_and_unrecorded_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model = self._model_tree(Path(tmp))
            identity = build_tau3_model_identity(
                model,
                model_id="local/fixture-8b",
                revision="b" * 40,
            )
            (model / "model-00001-of-00001.safetensors").write_bytes(b"changed")
            (model / "added.json").write_text("{}\n", encoding="utf-8")

            errors = validate_tau3_model_identity(
                identity,
                model,
                expected_model_id="local/fixture-8b",
                expected_revision="b" * 40,
            )

            self.assertTrue(any("omits model files" in error for error in errors))
            self.assertTrue(any("SHA-256 does not replay" in error for error in errors))

    def test_identity_rejects_mutable_revision_and_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model = self._model_tree(Path(tmp))
            with self.assertRaisesRegex(Tau3ModelIdentityError, "immutable"):
                build_tau3_model_identity(model, model_id="local/fixture-8b", revision="main")
            try:
                (model / "linked-config.json").symlink_to(model / "config.json")
            except OSError:
                self.skipTest("symlinks unavailable")
            with self.assertRaisesRegex(Tau3ModelIdentityError, "symlinks"):
                build_tau3_model_identity(model, model_id="local/fixture-8b", revision="c" * 40)
            (model / "linked-config.json").unlink()
            model_link = Path(tmp) / "model-link"
            model_link.symlink_to(model, target_is_directory=True)
            with self.assertRaisesRegex(Tau3ModelIdentityError, "must not contain symlink"):
                build_tau3_model_identity(model_link, model_id="local/fixture-8b", revision="c" * 40)

    def test_tokenizer_config_without_payload_is_incomplete(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / "model"
            model.mkdir()
            (model / "config.json").write_text("{}\n", encoding="utf-8")
            (model / "tokenizer_config.json").write_text("{}\n", encoding="utf-8")
            (model / "model.safetensors").write_bytes(b"weights")

            with self.assertRaisesRegex(Tau3ModelIdentityError, "actual tokenizer payload"):
                build_tau3_model_identity(model, model_id="local/incomplete-8b", revision="f" * 40)

    def test_registered_schema_rejects_mutable_revision(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model = self._model_tree(Path(tmp))
            identity = build_tau3_model_identity(
                model,
                model_id="local/fixture-8b",
                revision="1" * 40,
            )
            for revision in ("main", "LATEST", "Unknown"):
                identity["revision"] = revision
                schema = check_schema_contract(identity, name_or_id="tau3_model_identity")
                self.assertFalse(schema["passed"], revision)

    def test_registered_schema_rejects_unsafe_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model = self._model_tree(Path(tmp))
            identity = build_tau3_model_identity(
                model,
                model_id="local/fixture-8b",
                revision="2" * 40,
            )
            for unsafe_path in ("../private/config.json", "/Users/example/config.json", "dir\\config.json"):
                identity["files"][0]["path"] = unsafe_path
                schema = check_schema_contract(identity, name_or_id="tau3_model_identity")
                self.assertFalse(schema["passed"], unsafe_path)


    def test_builder_source_check_requires_replaying_tree_identity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = self._model_tree(root)
            identity = build_tau3_model_identity(
                model,
                model_id="local/fixture-8b",
                revision="d" * 40,
            )
            identity_path = root / "identity.json"
            identity_path.write_text(json.dumps(identity, sort_keys=True) + "\n", encoding="utf-8")
            digest = hashlib.sha256(identity_path.read_bytes()).hexdigest()
            entry = {
                "name": "local/fixture-8b",
                "revision": "d" * 40,
                "local_identity_path": str(identity_path),
                "local_identity_sha256": digest,
            }

            self.assertTrue(_local_model_identity_matches(entry, model, "d" * 40))
            identity_link = root / "identity-link.json"
            try:
                identity_link.symlink_to(identity_path)
            except OSError:
                pass
            else:
                linked_entry = {**entry, "local_identity_path": str(identity_link)}
                self.assertFalse(_local_model_identity_matches(linked_entry, model, "d" * 40))
            (model / "tokenizer.json").write_text("tampered\n", encoding="utf-8")
            self.assertFalse(_local_model_identity_matches(entry, model, "d" * 40))

    def test_cli_writes_new_private_identity_outside_model_tree(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = self._model_tree(root)
            out = root / "identity.json"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "build_tau3_model_identity.py"),
                    "--model-path",
                    str(model),
                    "--model-id",
                    "local/fixture-8b",
                    "--revision",
                    "e" * 40,
                    "--out",
                    str(out),
                ],
                cwd=ROOT,
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(json.loads(completed.stdout)["identity_file"], "identity.json")
            self.assertEqual(os.stat(out).st_mode & 0o777, 0o600)
            repeated = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "build_tau3_model_identity.py"),
                    "--model-path",
                    str(model),
                    "--model-id",
                    "local/fixture-8b",
                    "--revision",
                    "e" * 40,
                    "--out",
                    str(out),
                ],
                cwd=ROOT,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(repeated.returncode, 2)
            self.assertIn("refusing to overwrite", repeated.stderr)

    def test_checked_in_protocol_template_is_deliberately_not_ready(self) -> None:
        template = json.loads(
            (ROOT / "examples" / "tau3_training" / "protocol_config.template.json").read_text(encoding="utf-8")
        )

        receipt = check_tau3_training_sources(template)

        failed = {check["id"] for check in receipt["checks"] if check["passed"] is not True}
        self.assertFalse(receipt["passed"])
        self.assertIn("no_unresolved_template_placeholders", failed)
        self.assertIn("base_model_revision_matches_local_identity", failed)
        self.assertIn("contamination_attestation_passed", failed)
        self.assertIn("redaction_attestation_passed", failed)
        self.assertIn("all_licenses_allow_training", failed)

    def test_source_preflight_requires_every_contamination_subcheck(self) -> None:
        template = json.loads(
            (ROOT / "examples" / "tau3_training" / "protocol_config.template.json").read_text(encoding="utf-8")
        )
        template["contamination_attestation"]["passed"] = True
        template["contamination_attestation"]["unresolved_leakage"] = False

        checks = {check["id"]: check for check in _production_source_checks(template)}

        self.assertFalse(checks["contamination_attestation_passed"]["passed"])
        self.assertEqual(
            set(checks["contamination_attestation_passed"]["actual"]["pending_subchecks"]),
            {
                "exact_duplicate",
                "near_duplicate",
                "task_template_overlap",
                "tool_sequence_overlap",
                "state_transition_overlap",
            },
        )

    def test_builder_requires_supplied_captures_to_match_frozen_hash(self) -> None:
        template = json.loads(
            (ROOT / "examples" / "tau3_training" / "protocol_config.template.json").read_text(encoding="utf-8")
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frozen = root / "frozen.jsonl"
            supplied = root / "supplied.jsonl"
            frozen.write_text('{"trajectory_id":"frozen"}\n', encoding="utf-8")
            supplied.write_text('{"trajectory_id":"other"}\n', encoding="utf-8")
            template["split_manifest"]["training_captures"] = {
                "local_path": str(frozen),
                "sha256": hashlib.sha256(frozen.read_bytes()).hexdigest(),
                "row_count": 1,
            }

            checks = {
                check["id"]: check
                for check in _production_source_checks(
                    template,
                    supplied_captures_path=supplied,
                )
            }

            self.assertFalse(checks["supplied_training_captures_match_frozen_input"]["passed"])


if __name__ == "__main__":
    unittest.main()
