from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from flightrecorder.data_governance import (
    DataGovernanceError,
    apply_deletion_request,
    assign_clustered_splits,
    build_contamination_report,
    build_dataset_derivation,
    build_governance_receipt,
    scan_personal_data,
)
from flightrecorder.schema_registry import check_schema_contract


def _governance(subject: str = "subject-1") -> dict[str, object]:
    return {
        "owner": "example-owner",
        "tenant": "tenant-a",
        "legal_basis": "consent",
        "consent_id": "consent-1",
        "allowed_purposes": ["agent_training"],
        "sensitivity": "confidential",
        "jurisdiction": "US",
        "retention_expires_at": "2030-01-01T00:00:00+00:00",
        "license": "internal-training-only",
        "provenance": {"source": "unit-test", "source_revision": "fixture-v1"},
        "deletion_subject_ids": [subject],
    }


class DataGovernanceTests(unittest.TestCase):
    def test_personal_data_scanner_detects_structured_and_free_text_values(self) -> None:
        payload = {
            "customer_name": "Ada Lovelace",
            "body": (
                "Contact Grace Hopper at grace@example.com, +1 (212) 555-0123, "
                "10 Main Street, account CUST-12345, from 192.0.2.8."
            ),
            "organization_note": "Escalate to Example Health patient desk.",
        }
        findings = scan_personal_data(
            payload,
            organization_entities=["Example Health"],
        )
        kinds = {finding["kind"] for finding in findings}
        self.assertTrue(
            {"person_name", "email", "phone", "postal_address", "customer_id", "ip_address", "organization_entity"}
            <= kinds
        )
        self.assertTrue(all("matched_value" not in finding for finding in findings))

    def test_redaction_marker_does_not_hide_other_personal_data(self) -> None:
        findings = scan_personal_data(
            {"body": "Customer [REDACTED] can still be reached at visible@example.com."}
        )
        self.assertEqual({finding["kind"] for finding in findings}, {"email"})

    def test_governance_receipt_fails_closed_and_passes_authorized_clean_data(self) -> None:
        missing = build_governance_receipt(
            [{"episode_id": "e-missing", "messages": []}],
            purpose="agent_training",
            now="2028-01-01T00:00:00+00:00",
        )
        self.assertFalse(missing["passed"])
        self.assertIn("missing_governance_metadata", missing["blocked_reasons"])

        clean = build_governance_receipt(
            [
                {
                    "episode_id": "e-clean",
                    "governance": _governance(),
                    "messages": [{"role": "user", "content": "Summarize the approved test fixture."}],
                }
            ],
            purpose="agent_training",
            now="2028-01-01T00:00:00+00:00",
        )
        self.assertTrue(clean["passed"])
        self.assertEqual(clean["authorized_record_count"], 1)
        self.assertRegex(clean["policy_fingerprint"], r"^[0-9a-f]{64}$")
        self.assertTrue(check_schema_contract(clean, name_or_id="data_governance_receipt")["passed"])

        pii = build_governance_receipt(
            [
                {
                    "episode_id": "e-pii",
                    "governance": _governance(),
                    "messages": [{"role": "user", "content": "Email grace@example.com"}],
                }
            ],
            purpose="agent_training",
            now="2028-01-01T00:00:00+00:00",
        )
        self.assertFalse(pii["passed"])
        self.assertEqual(pii["pii_finding_count"], 1)

    def test_governance_policy_cannot_disable_admission_safeguards(self) -> None:
        record = {
            "episode_id": "e-policy-weakening",
            "governance": _governance(),
            "messages": [{"role": "user", "content": "Email visible@example.com"}],
        }
        receipt = build_governance_receipt(
            [record],
            purpose="agent_training",
            now="2028-01-01T00:00:00+00:00",
            policy={
                "purpose": "unrelated-purpose",
                "pii_policy": "allow",
                "unknown_license_allowed": True,
                "required_fields": [],
            },
        )

        self.assertFalse(receipt["passed"])
        self.assertEqual(receipt["policy"]["pii_policy"], "block_unredacted")
        self.assertFalse(receipt["policy"]["unknown_license_allowed"])
        self.assertEqual(receipt["policy"]["purpose"], "agent_training")
        self.assertIn("pii_policy_weakening_forbidden", receipt["blocked_reasons"])
        self.assertIn("required_governance_fields_weakening_forbidden", receipt["blocked_reasons"])
        self.assertIn("unredacted_personal_data", receipt["blocked_reasons"])
        self.assertTrue(check_schema_contract(receipt, name_or_id="data_governance_receipt")["passed"])

    def test_duplicate_clusters_are_atomic_and_protected_matches_block(self) -> None:
        rows = [
            {"row_id": "a", "task_family": "mail", "prompt": "Find the latest invoice and summarize it."},
            {"row_id": "b", "task_family": "calendar", "prompt": "Find the latest invoice and summarize it!"},
            {"row_id": "c", "task_family": "code", "prompt": "Repair the parser and run tests."},
        ]
        assigned = assign_clustered_splits(
            rows,
            similarity_threshold=0.85,
            seed="fixture-seed",
        )
        by_id = {row["row_id"]: row["split"] for row in assigned}
        self.assertEqual(by_id["a"], by_id["b"])

        report = build_contamination_report(
            assigned,
            protected_rows=[{"row_id": "benchmark-1", "prompt": "Find latest invoice and summarize it"}],
            similarity_threshold=0.75,
        )
        self.assertFalse(report["passed"])
        self.assertGreater(report["protected_match_count"], 0)
        self.assertEqual(report["cross_split_cluster_count"], 0)
        self.assertRegex(report["report_fingerprint"], r"^[0-9a-f]{64}$")
        self.assertTrue(check_schema_contract(report, name_or_id="dataset_contamination_report")["passed"])

    def test_deletion_rebuilds_descendants_and_quarantines_models(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "dataset.jsonl"
            source.write_text(
                "\n".join(
                    json.dumps(row, sort_keys=True)
                    for row in [
                        {"row_id": "keep", "governance": _governance("subject-keep"), "prompt": "keep"},
                        {"row_id": "remove", "governance": _governance("subject-delete"), "prompt": "remove"},
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            descendant = root / "descendant.jsonl"
            descendant.write_text(
                json.dumps(
                    {
                        "row_id": "derived-from-deleted-parent",
                        "governance": _governance("subject-keep"),
                        "prompt": "derived from the deleted parent row",
                    },
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            grandchild = root / "grandchild.jsonl"
            grandchild.write_text(
                json.dumps(
                    {
                        "row_id": "derived-from-deleted-descendant",
                        "governance": _governance("subject-keep"),
                        "prompt": "derived from the affected descendant row",
                    },
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            receipt = apply_deletion_request(
                deletion_subject_ids=["subject-delete"],
                dataset_entries=[
                    {
                        "dataset_version": "hfrds-grandchild",
                        "parent_versions": ["hfrds-descendant"],
                        "path": str(grandchild),
                    },
                    {
                        "dataset_version": "hfrds-descendant",
                        "parent_versions": ["hfrds-parent"],
                        "path": str(descendant),
                    },
                    {"dataset_version": "hfrds-parent", "path": str(source)},
                ],
                model_entries=[
                    {"model_id": "adapter-a", "dataset_versions": ["hfrds-parent"], "status": "candidate"},
                    {"model_id": "adapter-b", "dataset_versions": ["hfrds-descendant"], "status": "candidate"},
                    {"model_id": "adapter-c", "dataset_versions": ["hfrds-grandchild"], "status": "candidate"},
                ],
                output_dir=root / "deletion",
                request_id="delete-1",
                erase_sources=True,
                created_at="2028-01-02T00:00:00+00:00",
            )
            self.assertTrue(receipt["passed"])
            self.assertEqual(receipt["removed_row_count"], 3)
            self.assertEqual(
                receipt["rebuild_order"],
                ["hfrds-parent", "hfrds-descendant", "hfrds-grandchild"],
            )
            rebuilt_by_parent = {row["parent_dataset_version"]: row for row in receipt["datasets"]}
            parent_receipt = rebuilt_by_parent["hfrds-parent"]
            descendant_receipt = rebuilt_by_parent["hfrds-descendant"]
            grandchild_receipt = rebuilt_by_parent["hfrds-grandchild"]
            rebuilt_parent = root / "deletion" / parent_receipt["output_path"]
            rebuilt_descendant = root / "deletion" / descendant_receipt["output_path"]
            rebuilt_grandchild = root / "deletion" / grandchild_receipt["output_path"]
            self.assertEqual([json.loads(line)["row_id"] for line in rebuilt_parent.read_text().splitlines()], ["keep"])
            self.assertEqual(rebuilt_descendant.read_text(encoding="utf-8"), "")
            self.assertEqual(rebuilt_grandchild.read_text(encoding="utf-8"), "")
            self.assertEqual(parent_receipt["direct_subject_removed_row_count"], 1)
            self.assertEqual(parent_receipt["ancestor_lineage_removed_row_count"], 0)
            self.assertEqual(descendant_receipt["direct_subject_removed_row_count"], 0)
            self.assertEqual(descendant_receipt["ancestor_lineage_removed_row_count"], 1)
            self.assertEqual(grandchild_receipt["ancestor_lineage_removed_row_count"], 1)
            self.assertEqual(descendant_receipt["source_parent_versions"], ["hfrds-parent"])
            self.assertEqual(descendant_receipt["parent_versions"], [parent_receipt["dataset_version"]])
            self.assertEqual(grandchild_receipt["parent_versions"], [descendant_receipt["dataset_version"]])
            self.assertTrue(receipt["affected_lineage_absent"])
            self.assertEqual(receipt["affected_lineage_survivor_count"], 0)
            self.assertTrue(all(row["affected_lineage_absent"] for row in receipt["datasets"]))
            self.assertEqual(
                {row["model_id"] for row in receipt["quarantined_models"]},
                {"adapter-a", "adapter-b", "adapter-c"},
            )
            self.assertTrue(all(row["status"] == "quarantined" for row in receipt["quarantined_models"]))
            quarantine_by_model = {row["model_id"]: row for row in receipt["quarantined_models"]}
            self.assertEqual(quarantine_by_model["adapter-a"]["quarantine_scope"], "direct")
            self.assertEqual(quarantine_by_model["adapter-b"]["quarantine_scope"], "transitive")
            self.assertEqual(quarantine_by_model["adapter-c"]["quarantine_scope"], "transitive")
            self.assertFalse(source.exists())
            self.assertFalse(descendant.exists())
            self.assertFalse(grandchild.exists())
            self.assertEqual(receipt["source_erasure_count"], 3)
            self.assertTrue((root / "deletion" / receipt["model_quarantine_path"]).is_file())
            self.assertTrue(check_schema_contract(receipt, name_or_id="dataset_deletion_receipt")["passed"])
            quarantine = json.loads(
                (root / "deletion" / receipt["model_quarantine_path"]).read_text(encoding="utf-8")
            )
            self.assertTrue(check_schema_contract(quarantine, name_or_id="model_quarantine")["passed"])

    def test_deletion_rejects_cyclic_dataset_lineage_before_rebuild(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            left = root / "left.jsonl"
            right = root / "right.jsonl"
            left.write_text(json.dumps({"row_id": "left", "governance": _governance("subject-delete")}) + "\n")
            right.write_text(json.dumps({"row_id": "right", "governance": _governance("subject-keep")}) + "\n")

            with self.assertRaisesRegex(DataGovernanceError, "parent_versions contain a cycle"):
                apply_deletion_request(
                    deletion_subject_ids=["subject-delete"],
                    dataset_entries=[
                        {"dataset_version": "hfrds-left", "parent_versions": ["hfrds-right"], "path": str(left)},
                        {"dataset_version": "hfrds-right", "parent_versions": ["hfrds-left"], "path": str(right)},
                    ],
                    model_entries=[],
                    output_dir=root / "deletion",
                    request_id="cyclic-lineage",
                    erase_sources=True,
                )

            self.assertTrue(left.exists())
            self.assertTrue(right.exists())
            self.assertEqual(list((root / "deletion").glob("*.jsonl")), [])

    def test_deletion_rejects_malformed_lineage_before_rebuild(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "dataset.jsonl"
            source.write_text(
                json.dumps({"row_id": "remove", "governance": _governance("subject-delete")}) + "\n",
                encoding="utf-8",
            )

            with self.subTest("dataset parents"):
                with self.assertRaisesRegex(DataGovernanceError, "parent_versions must be a list"):
                    apply_deletion_request(
                        deletion_subject_ids=["subject-delete"],
                        dataset_entries=[
                            {
                                "dataset_version": "hfrds-parent",
                                "parent_versions": "hfrds-ambiguous-parent",
                                "path": str(source),
                            }
                        ],
                        model_entries=[],
                        output_dir=root / "bad-dataset-lineage",
                        request_id="bad-dataset-lineage",
                        erase_sources=True,
                    )

            with self.subTest("model datasets"):
                with self.assertRaisesRegex(DataGovernanceError, "dataset_versions must be a list"):
                    apply_deletion_request(
                        deletion_subject_ids=["subject-delete"],
                        dataset_entries=[{"dataset_version": "hfrds-parent", "path": str(source)}],
                        model_entries=[
                            {
                                "model_id": "adapter-a",
                                "dataset_versions": "hfrds-parent",
                            }
                        ],
                        output_dir=root / "bad-model-lineage",
                        request_id="bad-model-lineage",
                        erase_sources=True,
                    )

            self.assertTrue(source.exists())
            self.assertEqual(list(root.rglob("*-after-*.jsonl")), [])

    def test_deletion_receipt_fails_and_preserves_sources_if_affected_lineage_survives(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "dataset.jsonl"
            source.write_text(
                json.dumps({"row_id": "remove", "governance": _governance("subject-delete")}) + "\n",
                encoding="utf-8",
            )

            def retain_affected_row(path: Path, rows: list[dict[str, object]]) -> None:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(
                    json.dumps({"row_id": "survivor", "governance": _governance("subject-delete")}) + "\n",
                    encoding="utf-8",
                )

            with mock.patch("flightrecorder.data_governance._atomic_write_jsonl", side_effect=retain_affected_row):
                receipt = apply_deletion_request(
                    deletion_subject_ids=["subject-delete"],
                    dataset_entries=[{"dataset_version": "hfrds-parent", "path": str(source)}],
                    model_entries=[],
                    output_dir=root / "deletion",
                    request_id="surviving-lineage",
                    erase_sources=True,
                )

            self.assertFalse(receipt["passed"])
            self.assertFalse(receipt["affected_lineage_absent"])
            self.assertGreater(receipt["affected_lineage_survivor_count"], 0)
            self.assertIn("affected_lineage_survived_rebuild", receipt["blocked_reasons"])
            self.assertIn("source_erasure_incomplete", receipt["blocked_reasons"])
            self.assertTrue(source.exists())
            self.assertTrue(check_schema_contract(receipt, name_or_id="dataset_deletion_receipt")["passed"])

    def test_deletion_fails_closed_without_source_erasure_authorization(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "dataset.jsonl"
            source.write_text(
                json.dumps(
                    {"row_id": "remove", "governance": _governance("subject-delete"), "prompt": "remove"}
                )
                + "\n",
                encoding="utf-8",
            )
            receipt = apply_deletion_request(
                deletion_subject_ids=["subject-delete"],
                dataset_entries=[{"dataset_version": "hfrds-parent", "path": str(source)}],
                model_entries=[],
                output_dir=root / "deletion",
                request_id="delete-without-erasure",
            )
            self.assertFalse(receipt["passed"])
            self.assertIn("source_erasure_not_authorized", receipt["blocked_reasons"])
            self.assertTrue(source.exists())

    def test_derivation_identity_binds_recipe_policy_and_parents(self) -> None:
        base = build_dataset_derivation(
            parent_versions=["hfrds-parent"],
            recipe={"seed": 7, "selector": "accepted-only"},
            selected_rows=[{"row_id": "a"}],
            policy_fingerprint="a" * 64,
            scan_fingerprint="b" * 64,
        )
        changed = build_dataset_derivation(
            parent_versions=["hfrds-parent"],
            recipe={"seed": 8, "selector": "accepted-only"},
            selected_rows=[{"row_id": "a"}],
            policy_fingerprint="a" * 64,
            scan_fingerprint="b" * 64,
        )
        self.assertNotEqual(base["derivation_id"], changed["derivation_id"])
        self.assertEqual(base["parent_versions"], ["hfrds-parent"])
        self.assertTrue(check_schema_contract(base, name_or_id="dataset_derivation")["passed"])


if __name__ == "__main__":
    unittest.main()
