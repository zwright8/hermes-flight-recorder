from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

from flightrecorder.runtime_adapter_router import (
    ApprovalStore,
    build_adapter_route_decision,
    build_tool_capability_selection,
    build_write_approval,
    canonical_sha256,
    dispatch_tool_call,
    validate_adapter_route_decision,
    validate_json_schema_subset,
    validate_tool_capability_selection,
)
from flightrecorder.schema_registry import _validate_value


ROOT = Path(__file__).resolve().parents[1]
SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64
BASE_ENV = {
    "base_model_id": "Qwen/Qwen3-0.6B",
    "base_revision": "c1899de289a04d12100db370d81485cdf75e47ca",
    "tokenizer_revision": "tok-rev-1",
    "chat_template_sha256": SHA_D,
}


def schema_object() -> dict[str, object]:
    return {
        "type": "object",
        "properties": {
            "query": {"type": "string", "minLength": 1},
            "limit": {"type": "integer", "minimum": 1, "maximum": 10},
        },
        "required": ["query"],
        "additionalProperties": False,
    }


def task(
    *,
    domains: list[str] | None = None,
    required: list[str] | None = None,
    task_id: str = "task-1",
) -> dict[str, object]:
    return {
        "task_id": task_id,
        "contract_fingerprint": SHA_D,
        "prompt_sha256": SHA_A,
        "system_contract_sha256": SHA_B,
        "developer_contract_sha256": SHA_C,
        "tool_contract_sha256": SHA_A,
        "environment_contract_sha256": SHA_B,
        "policy_contract_sha256": SHA_C,
        "scenario_contract_sha256": SHA_D,
        "capabilities": {
            "required": required or ["browser.search"],
            "optional": ["browser.read"],
        },
        "domains": domains or ["browser"],
        "requires_tools": bool(required or ["browser.search"]),
        "allow_no_tools": False,
    }


def tool(
    name: str,
    capability: str,
    sha: str,
    *,
    risk: str = "read",
    write: bool = False,
    version: str = "1",
) -> dict[str, object]:
    return {
        "name": name,
        "version": version,
        "definition_sha256": sha,
        "capabilities": [capability, "browser.read"]
        if capability.startswith("browser")
        else [capability],
        "risk_class": risk,
        "write_capable": write,
        "parameters_schema": schema_object(),
    }


def tool_policy(**overrides: object) -> dict[str, object]:
    policy: dict[str, object] = {
        "policy_id": "tool-policy-1",
        "known_capabilities": [
            "browser.search",
            "browser.read",
            "database.query",
            "code.execute",
        ],
        "deny_tools": [],
        "allow_write_tools": False,
        "allowed_risk_classes": ["read", "low", "medium"],
    }
    policy.update(overrides)
    return policy


def selection_for_browser() -> dict[str, object]:
    return build_tool_capability_selection(
        task(domains=["browser"], required=["browser.search"]),
        [
            tool("browser.search", "browser.search", SHA_A),
            tool("database.query", "database.query", SHA_B),
        ],
        tool_policy(),
        environment={"runtime": "local"},
    )


def promotion(
    candidate: dict[str, object], *, passed: bool = True
) -> dict[str, object]:
    return {
        "schema_version": "hfr.promotion_decision.v1",
        "passed": passed,
        "recommendation": "apply_alias_update" if passed else "block_promotion",
        "alias_update": {
            "authorized": passed,
            "recommendation": "apply_alias_update" if passed else "hold_aliases",
            "aliases": [],
        },
        "models": {
            "candidate": {"id": candidate["candidate_id"], "class": "lora_adapter"},
            "champion": {"id": "champion", "class": "base_model"},
            "rollback": {"id": "rollback", "class": "base_model"},
        },
    }


def candidate(
    candidate_id: str,
    kind: str,
    domains: list[str],
    adapter_sha: str,
    *,
    capabilities: list[str] | None = None,
    active_adapter_ids: list[str] | None = None,
    members: list[str] | None = None,
    promoted: bool = True,
) -> dict[str, object]:
    row: dict[str, object] = {
        "candidate_id": candidate_id,
        "candidate_kind": kind,
        "adapter_id": f"adapter-{candidate_id}",
        "active_adapter_ids": active_adapter_ids or [f"adapter-{candidate_id}"],
        "adapter_revision": f"rev-{candidate_id}",
        "adapter_sha256": adapter_sha,
        "base_model_id": BASE_ENV["base_model_id"],
        "base_revision": BASE_ENV["base_revision"],
        "tokenizer_revision": BASE_ENV["tokenizer_revision"],
        "chat_template_sha256": BASE_ENV["chat_template_sha256"],
        "domains": domains,
        "capabilities": capabilities
        or ["browser.search", "database.query", "code.execute"],
        "registry_entry_id": f"registry-{candidate_id}",
        "composition_members": members or [],
        "promotion_evidence_ref": {
            "path": f"evidence/promotions/{candidate_id}.promotion.json",
            "sha256": adapter_sha,
            "size_bytes": 512,
        },
        "promotion_binding": {
            "independent_evidence": True,
            "candidate_id": candidate_id,
            "registry_entry_id": f"registry-{candidate_id}",
            "adapter_id": f"adapter-{candidate_id}",
            "adapter_revision": f"rev-{candidate_id}",
            "adapter_sha256": adapter_sha,
            "base_model_id": BASE_ENV["base_model_id"],
            "base_revision": BASE_ENV["base_revision"],
            "tokenizer_revision": BASE_ENV["tokenizer_revision"],
            "chat_template_sha256": BASE_ENV["chat_template_sha256"],
            "training_result_ref": {
                "path": f"evidence/training/{candidate_id}.training.json",
                "sha256": SHA_B,
                "size_bytes": 1024,
            },
            "evaluation_result_ref": {
                "path": f"evidence/eval/{candidate_id}.eval.json",
                "sha256": SHA_C,
                "size_bytes": 2048,
            },
        },
    }
    row["promotion_decision"] = promotion(row, passed=promoted)
    return row


def route_policy(**overrides: object) -> dict[str, object]:
    policy: dict[str, object] = {
        "policy_id": "route-policy-1",
        "allow_generalist_fallback": True,
        "required_base_model_id": BASE_ENV["base_model_id"],
        "required_base_revision": BASE_ENV["base_revision"],
        "required_tokenizer_revision": BASE_ENV["tokenizer_revision"],
        "required_chat_template_sha256": BASE_ENV["chat_template_sha256"],
    }
    policy.update(overrides)
    return policy


class RuntimeAdapterRouterTests(unittest.TestCase):
    def test_tool_selection_is_deterministic_and_records_rejections(self) -> None:
        catalog_a = [
            tool("database.query", "database.query", SHA_B),
            tool("browser.search", "browser.search", SHA_A),
            {
                **tool(
                    "browser.write",
                    "browser.search",
                    SHA_C,
                    risk="write_capable",
                    write=True,
                ),
                "name": "browser.write",
            },
        ]
        catalog_b = list(reversed(catalog_a))

        first = build_tool_capability_selection(
            task(), catalog_a, tool_policy(), environment={"runtime": "local"}
        )
        second = build_tool_capability_selection(
            task(), catalog_b, tool_policy(), environment={"runtime": "local"}
        )

        self.assertTrue(first["passed"])
        self.assertEqual(first["decision_fingerprint"], second["decision_fingerprint"])
        self.assertEqual(
            [row["name"] for row in first["selected_tools"]], ["browser.search"]
        )
        rejected = {
            row["name"]: row["reason_codes"]
            for row in first["evaluated_tools"]
            if not row["eligible"]
        }
        self.assertIn("missing_required_capability", rejected["database.query"])
        self.assertIn("write_tools_not_allowed", rejected["browser.write"])
        self.assertEqual(validate_tool_capability_selection(first), [])

    def test_task_contract_preserves_full_binding_fingerprints(self) -> None:
        selection = selection_for_browser()

        self.assertEqual(selection["task_contract_fingerprint"], SHA_D)
        bindings = selection["task_contract"]["binding_fingerprints"]
        self.assertEqual(bindings["prompt_sha256"], SHA_A)
        self.assertEqual(bindings["system_contract_sha256"], SHA_B)
        self.assertEqual(bindings["developer_contract_sha256"], SHA_C)
        self.assertEqual(bindings["tool_contract_sha256"], SHA_A)
        self.assertEqual(bindings["environment_contract_sha256"], SHA_B)
        self.assertEqual(bindings["policy_contract_sha256"], SHA_C)
        self.assertEqual(bindings["scenario_contract_sha256"], SHA_D)
        self.assertEqual(validate_tool_capability_selection(selection), [])

    def test_tool_selection_blocks_unknown_duplicate_and_no_route(self) -> None:
        duplicate = tool("browser.search", "browser.search", SHA_A)
        selection = build_tool_capability_selection(
            task(required=["unknown.capability"]),
            [duplicate, copy.deepcopy(duplicate)],
            tool_policy(known_capabilities=["browser.search"]),
        )

        self.assertFalse(selection["passed"])
        self.assertIn(
            "unknown_required_capability:unknown.capability",
            selection["blocking_reasons"],
        )
        reason_sets = [row["reason_codes"] for row in selection["evaluated_tools"]]
        self.assertTrue(
            any("duplicate_tool_identity" in reasons for reasons in reason_sets)
        )

    def test_tool_selection_uses_union_coverage_for_cross_domain_contract(self) -> None:
        contract = task(
            domains=["browser", "database"],
            required=["browser.search", "database.query"],
            task_id="task-cross-tools",
        )
        selection = build_tool_capability_selection(
            contract,
            [
                tool("database.query", "database.query", SHA_B),
                tool("browser.search", "browser.search", SHA_A),
            ],
            tool_policy(),
        )

        self.assertTrue(selection["passed"])
        self.assertEqual(
            [row["name"] for row in selection["selected_tools"]],
            ["browser.search", "database.query"],
        )
        self.assertEqual(validate_tool_capability_selection(selection), [])

        missing_database = build_tool_capability_selection(
            contract,
            [tool("browser.search", "browser.search", SHA_A)],
            tool_policy(),
        )
        self.assertFalse(missing_database["passed"])
        self.assertIn(
            "required_capability_unavailable:database.query",
            missing_database["blocking_reasons"],
        )
        self.assertEqual(validate_tool_capability_selection(missing_database), [])

    def test_no_tool_clarification_task_can_pass_when_allowed(self) -> None:
        contract = {
            "task_id": "clarify-1",
            "capabilities": {"required": [], "optional": []},
            "domains": [],
            "requires_tools": False,
            "allow_no_tools": True,
        }
        selection = build_tool_capability_selection(contract, [], tool_policy())
        self.assertTrue(selection["passed"])
        self.assertEqual(selection["selected_tools"], [])

    def test_adapter_routing_selects_specialist_then_generalist_fallback(self) -> None:
        selection = selection_for_browser()
        browser = candidate(
            "browser", "specialist", ["browser"], SHA_A, capabilities=["browser.search"]
        )
        generalist = candidate(
            "generalist", "generalist", ["browser", "database", "code_terminal"], SHA_B
        )

        route = build_adapter_route_decision(
            task(domains=["browser"], required=["browser.search"]),
            selection,
            [generalist, browser],
            route_policy(),
            runtime_environment=BASE_ENV,
        )

        self.assertTrue(route["passed"])
        self.assertEqual(route["selected_candidate"]["candidate_id"], "browser")
        self.assertTrue(route["exact_one_adapter"])
        self.assertEqual(
            validate_adapter_route_decision(route, capability_selection=selection), []
        )

        stale = copy.deepcopy(browser)
        stale["adapter_sha256"] = SHA_C
        stale["promotion_decision"] = promotion(browser)
        fallback = build_adapter_route_decision(
            task(domains=["browser"], required=["browser.search"]),
            selection,
            [stale, generalist],
            route_policy(),
            runtime_environment=BASE_ENV,
        )
        self.assertTrue(fallback["passed"])
        self.assertEqual(fallback["selected_candidate"]["candidate_id"], "generalist")
        self.assertIn(
            "specialist_unavailable_generalist_fallback", fallback["fallback_reasons"]
        )
        self.assertEqual(
            validate_adapter_route_decision(fallback, capability_selection=selection),
            [],
        )

    def test_cross_domain_requires_promoted_generalist_and_can_block(self) -> None:
        selection = build_tool_capability_selection(
            task(
                domains=["browser", "database"],
                required=["browser.search"],
                task_id="task-cross",
            ),
            [tool("browser.search", "browser.search", SHA_A)],
            tool_policy(),
        )
        generalist = candidate(
            "generalist",
            "generalist",
            ["browser", "database"],
            SHA_B,
            capabilities=["browser.search"],
        )
        route = build_adapter_route_decision(
            task(
                domains=["database", "browser"],
                required=["browser.search"],
                task_id="task-cross",
            ),
            selection,
            [generalist],
            route_policy(),
            runtime_environment=BASE_ENV,
        )
        self.assertTrue(route["passed"])
        self.assertEqual(route["route_kind"], "generalist")

        blocked = build_adapter_route_decision(
            task(
                domains=["database", "browser"],
                required=["browser.search"],
                task_id="task-cross",
            ),
            selection,
            [
                candidate(
                    "browser",
                    "specialist",
                    ["browser"],
                    SHA_A,
                    capabilities=["browser.search"],
                )
            ],
            route_policy(allow_generalist_fallback=False),
            runtime_environment=BASE_ENV,
        )
        self.assertFalse(blocked["passed"])
        self.assertIsNone(blocked["selected_candidate"])
        self.assertIn("generalist_unavailable", blocked["blocking_reasons"])

    def test_adapter_rejects_unpromoted_identity_mismatch_and_multi_adapter_activation(
        self,
    ) -> None:
        selection = selection_for_browser()
        unpromoted = candidate(
            "unpromoted",
            "specialist",
            ["browser"],
            SHA_A,
            capabilities=["browser.search"],
            promoted=False,
        )
        multi = candidate(
            "multi",
            "specialist",
            ["browser"],
            SHA_B,
            capabilities=["browser.search"],
            active_adapter_ids=["adapter-multi", "adapter-other"],
        )
        stale_base = candidate(
            "stale", "specialist", ["browser"], SHA_C, capabilities=["browser.search"]
        )
        stale_base["base_revision"] = "different"
        stale_base["promotion_decision"] = promotion(stale_base)

        route = build_adapter_route_decision(
            task(domains=["browser"], required=["browser.search"]),
            selection,
            [unpromoted, multi, stale_base],
            route_policy(allow_generalist_fallback=False),
            runtime_environment=BASE_ENV,
        )

        self.assertFalse(route["passed"])
        rejections = {
            row["candidate_id"]: row["reason_codes"]
            for row in route["evaluated_candidates"]
        }
        self.assertIn("promotion_not_passed", rejections["unpromoted"])
        self.assertIn("multiple_active_adapters_requested", rejections["multi"])
        self.assertIn("base_revision_mismatch", rejections["stale"])

    def test_adapter_requires_complete_promotion_evidence_projection(self) -> None:
        selection = selection_for_browser()
        missing_ref = candidate(
            "missing-ref",
            "specialist",
            ["browser"],
            SHA_A,
            capabilities=["browser.search"],
        )
        missing_ref["promotion_evidence_ref"] = {}
        bad_alias = candidate(
            "bad-alias",
            "specialist",
            ["browser"],
            SHA_B,
            capabilities=["browser.search"],
        )
        bad_alias["promotion_decision"]["alias_update"] = {
            "authorized": False,
            "recommendation": "hold_aliases",
            "aliases": [],
        }

        route = build_adapter_route_decision(
            task(domains=["browser"], required=["browser.search"]),
            selection,
            [missing_ref, bad_alias],
            route_policy(allow_generalist_fallback=False),
            runtime_environment=BASE_ENV,
        )

        self.assertFalse(route["passed"])
        rejections = {
            row["candidate_id"]: row["reason_codes"]
            for row in route["evaluated_candidates"]
        }
        self.assertIn("promotion_evidence_ref_path_invalid", rejections["missing-ref"])
        self.assertIn("promotion_alias_update_not_authorized", rejections["bad-alias"])

        valid = build_adapter_route_decision(
            task(domains=["browser"], required=["browser.search"]),
            selection,
            [
                candidate(
                    "browser",
                    "specialist",
                    ["browser"],
                    SHA_A,
                    capabilities=["browser.search"],
                )
            ],
            route_policy(),
            runtime_environment=BASE_ENV,
        )
        self.assertEqual(
            validate_adapter_route_decision(valid, capability_selection=selection), []
        )
        summary = valid["selected_candidate"]["promotion_evidence"]["decision_summary"]
        self.assertEqual(summary["models_candidate_id"], "browser")
        self.assertNotIn("adapter_sha256", summary)
        self.assertNotIn("base_revision", summary)
        self.assertEqual(
            valid["selected_candidate"]["promotion_binding"]["adapter_sha256"], SHA_A
        )
        self.assertEqual(
            valid["selected_candidate"]["promotion_binding"]["base_revision"],
            BASE_ENV["base_revision"],
        )
        tampered = copy.deepcopy(valid)
        tampered["selected_candidate"]["promotion_binding"]["adapter_sha256"] = SHA_C
        self.assertIn(
            "candidate:browser:promotion_binding_adapter_sha256_mismatch",
            validate_adapter_route_decision(tampered, capability_selection=selection),
        )

    def test_composition_is_atomic_and_independently_promoted(self) -> None:
        selection = selection_for_browser()
        composition = candidate(
            "browser-db-composition",
            "composition",
            ["browser", "database"],
            SHA_C,
            capabilities=["browser.search"],
            members=["adapter-browser", "adapter-database"],
        )
        generalist = candidate(
            "generalist",
            "generalist",
            ["browser", "database"],
            SHA_B,
            capabilities=["browser.search"],
        )
        route = build_adapter_route_decision(
            task(domains=["browser"], required=["browser.search"]),
            selection,
            [composition, generalist],
            route_policy(),
            runtime_environment=BASE_ENV,
        )
        self.assertTrue(route["passed"])
        self.assertEqual(
            route["selected_candidate"]["candidate_id"], "browser-db-composition"
        )
        composition_row = next(
            row
            for row in route["evaluated_candidates"]
            if row["candidate_id"] == "browser-db-composition"
        )
        self.assertNotIn(
            "multiple_active_adapters_requested", composition_row["reason_codes"]
        )
        self.assertEqual(
            composition_row["active_adapter_ids"], ["adapter-browser-db-composition"]
        )

    def test_cross_domain_composition_precedes_generalist_and_ambiguity_blocks(
        self,
    ) -> None:
        contract = task(
            domains=["browser", "database"],
            required=["browser.search", "database.query"],
            task_id="task-cross-composition",
        )
        selection = build_tool_capability_selection(
            contract,
            [
                tool("browser.search", "browser.search", SHA_A),
                tool("database.query", "database.query", SHA_B),
            ],
            tool_policy(),
        )
        composition = candidate(
            "browser-db-composition",
            "composition",
            ["browser", "database"],
            SHA_C,
            capabilities=["browser.search", "database.query"],
            members=["adapter-browser", "adapter-database"],
        )
        generalist = candidate(
            "generalist",
            "generalist",
            ["browser", "database", "code_terminal"],
            SHA_B,
            capabilities=["browser.search", "database.query"],
        )

        route = build_adapter_route_decision(
            contract,
            selection,
            [generalist, composition],
            route_policy(),
            runtime_environment=BASE_ENV,
        )
        self.assertTrue(route["passed"])
        self.assertEqual(route["route_kind"], "generalist")
        self.assertEqual(
            route["selected_candidate"]["candidate_id"], "browser-db-composition"
        )

        second = candidate(
            "browser-db-composition-2",
            "composition",
            ["browser", "database"],
            SHA_D,
            capabilities=["browser.search", "database.query"],
            members=["adapter-browser-2", "adapter-database-2"],
        )
        ambiguous = build_adapter_route_decision(
            contract,
            selection,
            [composition, second, generalist],
            route_policy(),
            runtime_environment=BASE_ENV,
        )
        self.assertFalse(ambiguous["passed"])
        self.assertIn("ambiguous_composition_candidates", ambiguous["blocking_reasons"])

    def test_semantic_validators_catch_tampering(self) -> None:
        selection = selection_for_browser()
        tampered = copy.deepcopy(selection)
        tampered["selected_tools"][0]["definition_sha256"] = SHA_C
        self.assertIn(
            "selected_tool_set_fingerprint_mismatch",
            validate_tool_capability_selection(tampered),
        )

        write_requirement_tamper = copy.deepcopy(selection)
        write_requirement_tamper["selected_requires_external_write_authorization"] = (
            True
        )
        self.assertIn(
            "selected_write_authorization_requirement_mismatch",
            validate_tool_capability_selection(write_requirement_tamper),
        )

        route = build_adapter_route_decision(
            task(domains=["browser"], required=["browser.search"]),
            selection,
            [
                candidate(
                    "generalist",
                    "generalist",
                    ["browser"],
                    SHA_B,
                    capabilities=["browser.search"],
                )
            ],
            route_policy(),
            runtime_environment=BASE_ENV,
        )
        bad_route = copy.deepcopy(route)
        bad_route["selected_candidate"]["adapter_id"] = "changed"
        self.assertIn(
            "route_fingerprint_mismatch",
            validate_adapter_route_decision(bad_route, capability_selection=selection),
        )
        ref_tampered = copy.deepcopy(route)
        ref_tampered["capability_selection_ref"]["sha256"] = SHA_C
        errors = validate_adapter_route_decision(
            ref_tampered, capability_selection=selection
        )
        self.assertIn("capability_selection_ref_sha256_mismatch", errors)
        self.assertIn("route_fingerprint_mismatch", errors)

    def test_readonly_dispatch_validates_selected_tool_and_arguments(self) -> None:
        selection = selection_for_browser()
        route = build_adapter_route_decision(
            task(domains=["browser"], required=["browser.search"]),
            selection,
            [
                candidate(
                    "browser",
                    "specialist",
                    ["browser"],
                    SHA_A,
                    capabilities=["browser.search"],
                )
            ],
            route_policy(),
            runtime_environment=BASE_ENV,
        )
        calls: list[dict[str, object]] = []

        def handler(arguments: dict[str, object]) -> dict[str, object]:
            calls.append(arguments)
            return {"ok": True}

        identity = selection["selected_tools"][0]
        result = dispatch_tool_call(
            capability_selection=selection,
            route_decision=route,
            tool_call={
                "call_id": "call-1",
                "tool": identity,
                "arguments": {"query": "status", "limit": 3},
            },
            handlers={
                f"{identity['name']}@{identity['version']}#{identity['definition_sha256']}": handler
            },
            policy={"policy_id": "dispatch-1", "allow_readonly_dispatch": True},
        )
        self.assertEqual(result["status"], "dispatched")
        self.assertEqual(len(calls), 1)

        denied = dispatch_tool_call(
            capability_selection=selection,
            route_decision=route,
            tool_call={
                "call_id": "call-2",
                "tool": identity,
                "arguments": {"query": "status", "extra": True},
            },
            handlers={
                f"{identity['name']}@{identity['version']}#{identity['definition_sha256']}": handler
            },
            policy={"policy_id": "dispatch-1", "allow_readonly_dispatch": True},
        )
        self.assertEqual(denied["status"], "denied")
        self.assertEqual(denied["reason_code"], "arguments_schema_invalid")
        self.assertEqual(len(calls), 1)

    def test_write_dispatch_requires_content_bound_single_use_approval(self) -> None:
        write_tool = tool(
            "browser.write", "browser.search", SHA_A, risk="write_capable", write=True
        )
        selection = build_tool_capability_selection(
            task(domains=["browser"], required=["browser.search"]),
            [write_tool],
            tool_policy(allow_write_tools=True, allowed_risk_classes=["write_capable"]),
        )
        route = build_adapter_route_decision(
            task(domains=["browser"], required=["browser.search"]),
            selection,
            [
                candidate(
                    "browser",
                    "specialist",
                    ["browser"],
                    SHA_A,
                    capabilities=["browser.search"],
                )
            ],
            route_policy(),
            runtime_environment=BASE_ENV,
        )
        identity = selection["selected_tools"][0]
        calls: list[dict[str, object]] = []

        def handler(arguments: dict[str, object]) -> dict[str, object]:
            calls.append(arguments)
            return {"written": True}

        handlers = {
            f"{identity['name']}@{identity['version']}#{identity['definition_sha256']}": handler
        }
        missing = dispatch_tool_call(
            capability_selection=selection,
            route_decision=route,
            tool_call={
                "call_id": "write-1",
                "tool": identity,
                "arguments": {"query": "save"},
            },
            handlers=handlers,
            policy={"policy_id": "dispatch-write", "allow_readonly_dispatch": True},
        )
        self.assertEqual(missing["reason_code"], "approval_missing")
        self.assertEqual(calls, [])

        store = ApprovalStore()
        approval = build_write_approval(
            approval_id="approval-1",
            task_fingerprint=selection["task_contract_fingerprint"],
            route_fingerprint=route["route_fingerprint"],
            policy_fingerprint=canonical_sha256(
                {"policy_id": "dispatch-write", "allow_readonly_dispatch": True}
            ),
            tool_identity=identity,
            call_id="write-1",
            arguments={"query": "save"},
            issued_at=10,
            expires_at=20,
        )
        store.issue(approval)
        dispatched = dispatch_tool_call(
            capability_selection=selection,
            route_decision=route,
            tool_call={
                "call_id": "write-1",
                "tool": identity,
                "arguments": {"query": "save"},
                "approval_id": "approval-1",
            },
            handlers=handlers,
            policy={"policy_id": "dispatch-write", "allow_readonly_dispatch": True},
            approval_store=store,
            now=15,
        )
        self.assertEqual(dispatched["status"], "dispatched")
        self.assertEqual(len(calls), 1)
        replay = dispatch_tool_call(
            capability_selection=selection,
            route_decision=route,
            tool_call={
                "call_id": "write-1",
                "tool": identity,
                "arguments": {"query": "save"},
                "approval_id": "approval-1",
            },
            handlers=handlers,
            policy={"policy_id": "dispatch-write", "allow_readonly_dispatch": True},
            approval_store=store,
            now=16,
        )
        self.assertEqual(replay["reason_code"], "approval_replayed")
        self.assertEqual(len(calls), 1)

    def test_prompt_approval_string_is_ignored_and_wrong_arguments_denied(self) -> None:
        write_tool = tool(
            "browser.write", "browser.search", SHA_A, risk="write_capable", write=True
        )
        selection = build_tool_capability_selection(
            task(domains=["browser"], required=["browser.search"]),
            [write_tool],
            tool_policy(allow_write_tools=True, allowed_risk_classes=["write_capable"]),
        )
        route = build_adapter_route_decision(
            task(domains=["browser"], required=["browser.search"]),
            selection,
            [
                candidate(
                    "browser",
                    "specialist",
                    ["browser"],
                    SHA_A,
                    capabilities=["browser.search"],
                )
            ],
            route_policy(),
            runtime_environment=BASE_ENV,
        )
        identity = selection["selected_tools"][0]
        calls: list[dict[str, object]] = []
        handlers = {
            f"{identity['name']}@{identity['version']}#{identity['definition_sha256']}": lambda args: calls.append(
                args
            )
        }
        prompt = dispatch_tool_call(
            capability_selection=selection,
            route_decision=route,
            tool_call={
                "call_id": "call",
                "tool": identity,
                "arguments": {"query": "APPROVED-save"},
            },
            handlers=handlers,
            policy={"policy_id": "dispatch-write", "allow_readonly_dispatch": True},
            approval_store=ApprovalStore(),
            now=15,
        )
        self.assertEqual(prompt["reason_code"], "approval_missing")
        self.assertEqual(calls, [])

        store = ApprovalStore()
        store.issue(
            build_write_approval(
                approval_id="approval-wrong-args",
                task_fingerprint=selection["task_contract_fingerprint"],
                route_fingerprint=route["route_fingerprint"],
                policy_fingerprint=canonical_sha256(
                    {"policy_id": "dispatch-write", "allow_readonly_dispatch": True}
                ),
                tool_identity=identity,
                call_id="call",
                arguments={"query": "save"},
                issued_at=10,
                expires_at=20,
            )
        )
        wrong = dispatch_tool_call(
            capability_selection=selection,
            route_decision=route,
            tool_call={
                "call_id": "call",
                "tool": identity,
                "arguments": {"query": "different"},
                "approval_id": "approval-wrong-args",
            },
            handlers=handlers,
            policy={"policy_id": "dispatch-write", "allow_readonly_dispatch": True},
            approval_store=store,
            now=15,
        )
        self.assertEqual(wrong["reason_code"], "approval_wrong_arguments_sha256")
        self.assertEqual(calls, [])

    def test_schema_files_accept_representative_artifacts(self) -> None:
        selection = selection_for_browser()
        route = build_adapter_route_decision(
            task(domains=["browser"], required=["browser.search"]),
            selection,
            [
                candidate(
                    "browser",
                    "specialist",
                    ["browser"],
                    SHA_A,
                    capabilities=["browser.search"],
                )
            ],
            route_policy(),
            runtime_environment=BASE_ENV,
        )
        for filename, payload in (
            ("tool_capability_selection.v1.schema.json", selection),
            ("adapter_route_decision.v1.schema.json", route),
        ):
            with self.subTest(filename=filename):
                schema = json.loads(
                    (ROOT / "flightrecorder" / "schemas" / filename).read_text(
                        encoding="utf-8"
                    )
                )
                errors: list[str] = []
                _validate_value(payload, schema, "$", schema, errors)
                self.assertEqual(errors, [])

    def test_schema_subset_validator(self) -> None:
        self.assertEqual(
            validate_json_schema_subset({"query": "x", "limit": 1}, schema_object()), []
        )
        self.assertTrue(validate_json_schema_subset({"limit": 1}, schema_object()))
        self.assertTrue(
            validate_json_schema_subset({"query": "x", "limit": 11}, schema_object())
        )


if __name__ == "__main__":
    unittest.main()
