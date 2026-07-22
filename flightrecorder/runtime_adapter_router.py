"""Governed tool selection, adapter routing, and pre-dispatch authorization."""

from __future__ import annotations

import copy
import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable, cast

TOOL_CAPABILITY_SELECTION_SCHEMA_VERSION = "hfr.tool_capability_selection.v1"
ADAPTER_ROUTE_DECISION_SCHEMA_VERSION = "hfr.adapter_route_decision.v1"
ROUTER_VERSION = "hfr.runtime_adapter_router.core.v1"

READ_RISKS = {"read", "low", "medium", "high"}
WRITE_RISKS = {"write", "write_capable"}
KNOWN_RISKS = READ_RISKS | WRITE_RISKS


class RuntimeAdapterRouterError(ValueError):
    """Raised when router inputs or guarded dispatch calls are malformed."""


class JsonSchemaSubsetError(RuntimeAdapterRouterError):
    """Raised when a value fails the router's supported JSON Schema subset."""


class DispatchDenied(RuntimeAdapterRouterError):
    """Raised when a guarded tool dispatch is denied before handler execution."""


def build_tool_capability_selection(
    task_contract: dict[str, Any],
    tool_catalog: list[dict[str, Any]],
    policy: dict[str, Any],
    *,
    environment: dict[str, Any] | None = None,
    router_version: str = ROUTER_VERSION,
) -> dict[str, Any]:
    """Build a deterministic governed tool-capability decision artifact."""

    task = _normalize_task_contract(task_contract)
    normalized_policy = _normalize_policy(policy)
    env = _normalize_mapping(environment or {})
    required = sorted(task["required_capabilities"])
    optional = sorted(task["optional_capabilities"])
    known_capabilities = set(normalized_policy.get("known_capabilities", []))
    if not known_capabilities:
        known_capabilities = set(required) | set(optional)
        for tool in tool_catalog:
            known_capabilities.update(_string_list(tool.get("capabilities")))

    evaluated: list[dict[str, Any]] = []
    selected_tools: list[dict[str, Any]] = []
    blocking_reasons: list[str] = []
    seen_identities: dict[tuple[str, str, str], int] = {}
    selected_requires_write_auth = False

    for index, raw_tool in enumerate(tool_catalog):
        normalized, errors = _normalize_tool(raw_tool, known_capabilities)
        identity_key = (
            normalized.get("name", ""),
            normalized.get("version", ""),
            normalized.get("definition_sha256", ""),
        )
        if identity_key in seen_identities:
            errors.append("duplicate_tool_identity")
        else:
            seen_identities[identity_key] = index
        reasons = sorted(
            set(
                errors
                + _tool_policy_rejections(normalized, required, normalized_policy, env)
            )
        )
        eligible = not reasons
        evaluated_tool = {
            "name": normalized.get("name", ""),
            "version": normalized.get("version", ""),
            "definition_sha256": normalized.get("definition_sha256", ""),
            "capabilities": sorted(normalized.get("capabilities", [])),
            "risk_class": normalized.get("risk_class", ""),
            "write_capable": bool(normalized.get("write_capable")),
            "requires_write_authorization": bool(
                normalized.get("requires_write_authorization")
            ),
            "parameters_schema": normalized.get("parameters_schema", {}),
            "eligible": eligible,
            "reason_codes": reasons or ["eligible"],
        }
        evaluated.append(evaluated_tool)
        if eligible:
            selected_tools.append(_selected_tool_projection(evaluated_tool))
            selected_requires_write_auth = selected_requires_write_auth or bool(
                evaluated_tool["requires_write_authorization"]
            )

    unknown_required = sorted(set(required) - known_capabilities)
    unknown_optional = sorted(set(optional) - known_capabilities)
    for capability in unknown_required:
        blocking_reasons.append(f"unknown_required_capability:{capability}")
    for capability in unknown_optional:
        blocking_reasons.append(f"unknown_optional_capability:{capability}")
    covered_capabilities = {
        capability
        for tool in selected_tools
        for capability in _string_list(tool.get("capabilities"))
    }
    for capability in sorted(set(required) - covered_capabilities):
        blocking_reasons.append(f"required_capability_unavailable:{capability}")
    if required and not selected_tools:
        blocking_reasons.append("required_tools_unavailable")
    if task["requires_tools"] and not selected_tools:
        blocking_reasons.append("task_requires_tools")
    if not task["requires_tools"] and not selected_tools and not task["allow_no_tools"]:
        blocking_reasons.append("no_tools_not_allowed")

    evaluated.sort(
        key=lambda row: (row["name"], row["version"], row["definition_sha256"])
    )
    selected_tools.sort(
        key=lambda row: (row["name"], row["version"], row["definition_sha256"])
    )
    selected_fingerprint = canonical_sha256(selected_tools)
    fingerprint_input = {
        "router_version": router_version,
        "task_contract": task,
        "tool_catalog_fingerprint": canonical_sha256(evaluated),
        "policy": normalized_policy,
        "environment": env,
        "selected_tool_set_fingerprint": selected_fingerprint,
        "blocking_reasons": sorted(set(blocking_reasons)),
    }
    decision_fingerprint = canonical_sha256(fingerprint_input)
    passed = not blocking_reasons and (bool(selected_tools) or task["allow_no_tools"])
    return {
        "schema_version": TOOL_CAPABILITY_SELECTION_SCHEMA_VERSION,
        "router_version": router_version,
        "task_contract_id": task["task_id"],
        "task_contract": task,
        "task_contract_fingerprint": _task_fingerprint(task),
        "tool_catalog_fingerprint": canonical_sha256(evaluated),
        "policy": normalized_policy,
        "policy_fingerprint": canonical_sha256(normalized_policy),
        "environment": env,
        "required_capabilities": required,
        "optional_capabilities": optional,
        "evaluated_tools": evaluated,
        "selected_tools": selected_tools,
        "selected_tool_set_fingerprint": selected_fingerprint,
        "selected_requires_external_write_authorization": selected_requires_write_auth,
        "passed": passed,
        "blocking_reasons": sorted(set(blocking_reasons)),
        "decision_fingerprint": decision_fingerprint,
    }


def validate_tool_capability_selection(artifact: dict[str, Any]) -> list[str]:
    """Return semantic validation errors for a tool-capability artifact."""

    errors: list[str] = []
    if artifact.get("schema_version") != TOOL_CAPABILITY_SELECTION_SCHEMA_VERSION:
        errors.append("schema_version_mismatch")
    task = artifact.get("task_contract")
    policy = artifact.get("policy")
    evaluated = artifact.get("evaluated_tools")
    selected = artifact.get("selected_tools")
    if not isinstance(task, dict):
        errors.append("task_contract_missing")
        task = {}
    if not isinstance(policy, dict):
        errors.append("policy_missing")
        policy = {}
    if not isinstance(evaluated, list):
        errors.append("evaluated_tools_missing")
        evaluated = []
    if not isinstance(selected, list):
        errors.append("selected_tools_missing")
        selected = []
    if artifact.get("task_contract_fingerprint") != _task_fingerprint(task):
        errors.append("task_contract_fingerprint_mismatch")
    if artifact.get("policy_fingerprint") != canonical_sha256(policy):
        errors.append("policy_fingerprint_mismatch")
    if artifact.get("tool_catalog_fingerprint") != canonical_sha256(evaluated):
        errors.append("tool_catalog_fingerprint_mismatch")
    if artifact.get("selected_tool_set_fingerprint") != canonical_sha256(selected):
        errors.append("selected_tool_set_fingerprint_mismatch")
    selected_keys = {
        (row.get("name"), row.get("version"), row.get("definition_sha256"))
        for row in selected
        if isinstance(row, dict)
    }
    eligible_keys = {
        (row.get("name"), row.get("version"), row.get("definition_sha256"))
        for row in evaluated
        if isinstance(row, dict) and row.get("eligible") is True
    }
    if selected_keys != eligible_keys:
        errors.append("selected_tools_do_not_match_eligible_tools")
    selected_requires_write_authorization = any(
        isinstance(row, dict)
        and (
            row.get("write_capable") is True
            or row.get("requires_write_authorization") is True
        )
        for row in selected
    )
    if (
        artifact.get("selected_requires_external_write_authorization")
        is not selected_requires_write_authorization
    ):
        errors.append("selected_write_authorization_requirement_mismatch")
    required = set(_string_list(artifact.get("required_capabilities")))
    covered = {
        capability
        for row in selected
        if isinstance(row, dict)
        for capability in _string_list(row.get("capabilities"))
    }
    missing = required - covered
    declared_missing = {
        reason.removeprefix("required_capability_unavailable:")
        for reason in _string_list(artifact.get("blocking_reasons"))
        if reason.startswith("required_capability_unavailable:")
    }
    if declared_missing != missing:
        errors.append("required_capability_coverage_mismatch")
    expected = _selection_fingerprint_input(artifact)
    if artifact.get("decision_fingerprint") != canonical_sha256(expected):
        errors.append("decision_fingerprint_mismatch")
    blocking = artifact.get("blocking_reasons")
    if artifact.get("passed") is True and blocking:
        errors.append("passed_artifact_has_blocking_reasons")
    if artifact.get("passed") is False and not blocking:
        errors.append("blocked_artifact_missing_blocking_reasons")
    return errors


def build_adapter_route_decision(
    task_contract: dict[str, Any],
    capability_selection: dict[str, Any],
    candidates: list[dict[str, Any]],
    routing_policy: dict[str, Any],
    *,
    runtime_environment: dict[str, Any],
    router_version: str = ROUTER_VERSION,
    capability_selection_sha256: str | None = None,
    capability_selection_ref: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a deterministic exact-one-adapter route decision."""

    task = _normalize_task_contract(task_contract)
    policy = _normalize_routing_policy(routing_policy)
    env = _normalize_mapping(runtime_environment)
    required_domains = sorted(task["domains"])
    route_kind = "specialist" if len(required_domains) == 1 else "generalist"
    fallback_reasons: list[str] = []
    evaluated: list[dict[str, Any]] = []
    selected: dict[str, Any] | None = None
    specialist_matches: list[dict[str, Any]] = []
    composition_matches: list[dict[str, Any]] = []
    generalist_matches: list[dict[str, Any]] = []

    for raw_candidate in candidates:
        candidate = _normalize_candidate(raw_candidate)
        reasons = _candidate_rejections(
            candidate, task, capability_selection, policy, env
        )
        projection = _candidate_projection(candidate)
        projection["eligible"] = not reasons
        projection["reason_codes"] = sorted(set(reasons)) or ["eligible"]
        evaluated.append(projection)
        if not reasons:
            if candidate["candidate_kind"] == "generalist":
                generalist_matches.append(projection)
            elif candidate["candidate_kind"] == "composition":
                composition_matches.append(projection)
            elif (
                route_kind == "specialist"
                and required_domains
                and required_domains[0] in candidate["domains"]
            ):
                specialist_matches.append(projection)

    blocking_reasons: list[str] = []
    if route_kind == "specialist":
        if len(specialist_matches) == 1:
            selected = specialist_matches[0]
        elif len(specialist_matches) > 1:
            blocking_reasons.append("ambiguous_specialist_candidates")
        elif len(composition_matches) == 1:
            selected = composition_matches[0]
        elif len(composition_matches) > 1:
            blocking_reasons.append("ambiguous_composition_candidates")
        elif policy["allow_generalist_fallback"]:
            fallback_reasons.append("specialist_unavailable_generalist_fallback")
        else:
            blocking_reasons.append("specialist_unavailable")
    else:
        if len(composition_matches) == 1:
            selected = composition_matches[0]
        elif len(composition_matches) > 1:
            blocking_reasons.append("ambiguous_composition_candidates")
        else:
            fallback_reasons.append("cross_domain_generalist_required")

    if selected is None and not blocking_reasons:
        if len(generalist_matches) == 1 and (
            route_kind != "specialist" or policy["allow_generalist_fallback"]
        ):
            selected = generalist_matches[0]
        elif len(generalist_matches) > 1:
            blocking_reasons.append("ambiguous_generalist_candidates")
        else:
            blocking_reasons.append("generalist_unavailable")

    evaluated.sort(
        key=lambda row: (row["candidate_id"], row["adapter_id"], row["adapter_sha256"])
    )
    capability_hash = capability_selection_sha256 or canonical_sha256(
        capability_selection
    )
    selection_ref = _normalize_artifact_ref(
        capability_selection_ref
        or {
            "path": "tool_capability_selection.json",
            "sha256": capability_hash,
            "size_bytes": len(canonical_json(capability_selection).encode("utf-8")),
        }
    )
    if selection_ref.get("sha256") != capability_hash:
        blocking_reasons.append("capability_selection_ref_sha256_mismatch")
    fingerprint_input = {
        "router_version": router_version,
        "task_contract": task,
        "capability_selection_fingerprint": capability_selection.get(
            "decision_fingerprint"
        ),
        "capability_selection_sha256": capability_hash,
        "capability_selection_ref": selection_ref,
        "routing_policy": policy,
        "runtime_environment": env,
        "evaluated_candidates": evaluated,
        "selected_candidate": selected,
        "fallback_reasons": sorted(set(fallback_reasons)),
        "blocking_reasons": sorted(set(blocking_reasons)),
    }
    route_fingerprint = canonical_sha256(fingerprint_input)
    return {
        "schema_version": ADAPTER_ROUTE_DECISION_SCHEMA_VERSION,
        "router_version": router_version,
        "task_contract_id": task["task_id"],
        "task_contract": task,
        "task_contract_fingerprint": _task_fingerprint(task),
        "capability_selection_fingerprint": capability_selection.get(
            "decision_fingerprint"
        ),
        "capability_selection_sha256": capability_hash,
        "capability_selection_ref": selection_ref,
        "routing_policy": policy,
        "routing_policy_fingerprint": canonical_sha256(policy),
        "runtime_environment": env,
        "route_kind": route_kind,
        "evaluated_candidates": evaluated,
        "selected_candidate": selected,
        "fallback_reasons": sorted(set(fallback_reasons)),
        "blocking_reasons": sorted(set(blocking_reasons)),
        "exact_one_adapter": selected is not None,
        "external_write_safety_boundary": "pre_dispatch_authorization_required_for_write_capable_tools",
        "confidence": "contract_match" if selected is not None else "blocked",
        "passed": selected is not None and not blocking_reasons,
        "route_fingerprint": route_fingerprint,
    }


def validate_adapter_route_decision(
    artifact: dict[str, Any],
    *,
    capability_selection: dict[str, Any] | None = None,
) -> list[str]:
    """Return semantic validation errors for an adapter-route artifact."""

    errors: list[str] = []
    if artifact.get("schema_version") != ADAPTER_ROUTE_DECISION_SCHEMA_VERSION:
        errors.append("schema_version_mismatch")
    task = (
        cast(dict[str, Any], artifact.get("task_contract"))
        if isinstance(artifact.get("task_contract"), dict)
        else {}
    )
    policy = (
        cast(dict[str, Any], artifact.get("routing_policy"))
        if isinstance(artifact.get("routing_policy"), dict)
        else {}
    )
    evaluated = (
        cast(list[Any], artifact.get("evaluated_candidates"))
        if isinstance(artifact.get("evaluated_candidates"), list)
        else []
    )
    selected = artifact.get("selected_candidate")
    if artifact.get("task_contract_fingerprint") != _task_fingerprint(task):
        errors.append("task_contract_fingerprint_mismatch")
    if artifact.get("routing_policy_fingerprint") != canonical_sha256(policy):
        errors.append("routing_policy_fingerprint_mismatch")
    if capability_selection is not None:
        if artifact.get("capability_selection_sha256") != canonical_sha256(
            capability_selection
        ):
            errors.append("capability_selection_sha256_mismatch")
        if artifact.get("capability_selection_fingerprint") != capability_selection.get(
            "decision_fingerprint"
        ):
            errors.append("capability_selection_fingerprint_mismatch")
    ref_errors = _artifact_ref_errors(artifact.get("capability_selection_ref"))
    errors.extend(f"capability_selection_ref_{error}" for error in ref_errors)
    if isinstance(artifact.get("capability_selection_ref"), dict) and artifact.get(
        "capability_selection_ref", {}
    ).get("sha256") != artifact.get("capability_selection_sha256"):
        errors.append("capability_selection_ref_sha256_mismatch")
    if selected is None:
        if artifact.get("passed") is True or artifact.get("exact_one_adapter") is True:
            errors.append("blocked_route_claims_adapter")
    elif not isinstance(selected, dict):
        errors.append("selected_candidate_malformed")
    else:
        selected_count = len(
            [
                row
                for row in evaluated
                if isinstance(row, dict)
                and row.get("eligible") is True
                and row.get("candidate_id") == selected.get("candidate_id")
                and row.get("adapter_id") == selected.get("adapter_id")
            ]
        )
        if selected_count != 1:
            errors.append("selected_candidate_not_unique_eligible")
    for row in evaluated:
        if isinstance(row, dict):
            declared_rejections = (
                set(_string_list(row.get("reason_codes")))
                if row.get("eligible") is False
                else set()
            )
            errors.extend(
                f"candidate:{row.get('candidate_id')}:{error}"
                for error in _projection_promotion_errors(row)
                if error not in declared_rejections
            )
    if artifact.get("passed") is True and artifact.get("blocking_reasons"):
        errors.append("passed_route_has_blocking_reasons")
    if artifact.get("passed") is False and not artifact.get("blocking_reasons"):
        errors.append("blocked_route_missing_blocking_reasons")
    if artifact.get("route_fingerprint") != canonical_sha256(
        _route_fingerprint_input(artifact)
    ):
        errors.append("route_fingerprint_mismatch")
    return errors


@dataclass
class ApprovalStore:
    """In-memory content-bound single-use write-approval store."""

    approvals: dict[str, dict[str, Any]] = field(default_factory=dict)
    consumed: set[str] = field(default_factory=set)
    revoked: set[str] = field(default_factory=set)

    def issue(self, approval: dict[str, Any]) -> dict[str, Any]:
        normalized = copy.deepcopy(approval)
        approval_id = _required_nonempty_string(normalized, "approval_id")
        if approval_id in self.approvals:
            raise RuntimeAdapterRouterError("duplicate approval_id")
        normalized["consumed"] = False
        self.approvals[approval_id] = normalized
        return copy.deepcopy(normalized)

    def revoke(self, approval_id: str) -> None:
        self.revoked.add(approval_id)

    def consume(self, approval_id: str) -> dict[str, Any]:
        approval = self.approvals.get(approval_id)
        if approval is None:
            raise DispatchDenied("approval_missing")
        if approval_id in self.revoked:
            raise DispatchDenied("approval_revoked")
        if approval_id in self.consumed or approval.get("consumed") is True:
            raise DispatchDenied("approval_replayed")
        self.consumed.add(approval_id)
        approval["consumed"] = True
        return copy.deepcopy(approval)


def build_write_approval(
    *,
    approval_id: str,
    task_fingerprint: str,
    route_fingerprint: str,
    policy_fingerprint: str,
    tool_identity: dict[str, str],
    call_id: str,
    arguments: dict[str, Any],
    issued_at: int,
    expires_at: int,
) -> dict[str, Any]:
    """Build a content-bound approval payload; callers still store it externally."""

    argument_hash = canonical_sha256(arguments)
    binding = {
        "task_fingerprint": task_fingerprint,
        "route_fingerprint": route_fingerprint,
        "policy_fingerprint": policy_fingerprint,
        "tool_identity": _tool_identity_projection(tool_identity),
        "call_id": call_id,
        "arguments_sha256": argument_hash,
        "issued_at": issued_at,
        "expires_at": expires_at,
    }
    return {
        "approval_id": approval_id,
        **binding,
        "approval_fingerprint": canonical_sha256(binding),
    }


def dispatch_tool_call(
    *,
    capability_selection: dict[str, Any],
    route_decision: dict[str, Any],
    tool_call: dict[str, Any],
    handlers: dict[str, Callable[[dict[str, Any]], Any]],
    policy: dict[str, Any],
    approval_store: ApprovalStore | None = None,
    now: int | None = None,
) -> dict[str, Any]:
    """Authorize and dispatch one tool call; denied calls never invoke handlers."""

    now = int(time.time()) if now is None else now
    call_id = _required_nonempty_string(tool_call, "call_id")
    raw_tool_identity = tool_call.get("tool")
    tool_identity = _tool_identity_projection(
        cast(dict[str, Any], raw_tool_identity)
        if isinstance(raw_tool_identity, dict)
        else tool_call
    )
    arguments = tool_call.get("arguments")
    if not isinstance(arguments, dict):
        return _dispatch_denial(call_id, tool_identity, "arguments_must_be_object")
    selected_tool = _find_selected_tool(capability_selection, tool_identity)
    if selected_tool is None:
        return _dispatch_denial(call_id, tool_identity, "tool_not_selected")
    if route_decision.get("passed") is not True:
        return _dispatch_denial(call_id, tool_identity, "route_not_passed")
    schema_errors = validate_json_schema_subset(
        arguments, selected_tool.get("parameters_schema", {})
    )
    if schema_errors:
        return _dispatch_denial(
            call_id, tool_identity, "arguments_schema_invalid", details=schema_errors
        )
    dispatch_policy = _normalize_dispatch_policy(policy)
    handler = handlers.get(_tool_key(tool_identity))
    if handler is None:
        return _dispatch_denial(call_id, tool_identity, "handler_missing")
    if selected_tool.get("write_capable") is True:
        approval_id = tool_call.get("approval_id")
        if not isinstance(approval_id, str) or not approval_id:
            return _dispatch_denial(call_id, tool_identity, "approval_missing")
        if approval_store is None:
            return _dispatch_denial(call_id, tool_identity, "approval_store_missing")
        try:
            approval = approval_store.consume(approval_id)
            _validate_approval(
                approval,
                task_fingerprint=str(
                    capability_selection.get("task_contract_fingerprint") or ""
                ),
                route_fingerprint=str(route_decision.get("route_fingerprint") or ""),
                policy_fingerprint=canonical_sha256(dispatch_policy),
                tool_identity=tool_identity,
                call_id=call_id,
                arguments=arguments,
                now=now,
            )
        except DispatchDenied as exc:
            return _dispatch_denial(call_id, tool_identity, str(exc))
    elif dispatch_policy["allow_readonly_dispatch"] is not True:
        return _dispatch_denial(call_id, tool_identity, "readonly_dispatch_not_allowed")

    try:
        result = handler(copy.deepcopy(arguments))
    except (
        Exception
    ) as exc:  # pragma: no cover - exercised by callers needing failure capture.
        return {
            "status": "handler_failed",
            "call_id": call_id,
            "tool": tool_identity,
            "handler_called": True,
            "error": type(exc).__name__,
            "message": str(exc),
        }
    return {
        "status": "dispatched",
        "call_id": call_id,
        "tool": tool_identity,
        "handler_called": True,
        "result": result,
    }


def validate_json_schema_subset(value: Any, schema: dict[str, Any]) -> list[str]:
    """Validate a value against the exact JSON Schema subset supported here."""

    errors: list[str] = []
    _validate_subset_value(value, schema, "$", errors)
    return errors


def canonical_json(value: Any) -> str:
    """Return deterministic JSON for hashes and persisted semantic comparisons."""

    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    )


def canonical_sha256(value: Any) -> str:
    """Return the SHA-256 of a JSON-compatible value's canonical representation."""

    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _selection_fingerprint_input(artifact: dict[str, Any]) -> dict[str, Any]:
    return {
        "router_version": artifact.get("router_version"),
        "task_contract": artifact.get("task_contract"),
        "tool_catalog_fingerprint": artifact.get("tool_catalog_fingerprint"),
        "policy": artifact.get("policy"),
        "environment": artifact.get("environment"),
        "selected_tool_set_fingerprint": artifact.get("selected_tool_set_fingerprint"),
        "blocking_reasons": artifact.get("blocking_reasons"),
    }


def _route_fingerprint_input(artifact: dict[str, Any]) -> dict[str, Any]:
    return {
        "router_version": artifact.get("router_version"),
        "task_contract": artifact.get("task_contract"),
        "capability_selection_fingerprint": artifact.get(
            "capability_selection_fingerprint"
        ),
        "capability_selection_sha256": artifact.get("capability_selection_sha256"),
        "capability_selection_ref": artifact.get("capability_selection_ref"),
        "routing_policy": artifact.get("routing_policy"),
        "runtime_environment": artifact.get("runtime_environment"),
        "evaluated_candidates": artifact.get("evaluated_candidates"),
        "selected_candidate": artifact.get("selected_candidate"),
        "fallback_reasons": artifact.get("fallback_reasons"),
        "blocking_reasons": artifact.get("blocking_reasons"),
    }


def _normalize_task_contract(task_contract: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(task_contract, dict):
        raise RuntimeAdapterRouterError("task_contract must be an object")
    task_id = _required_nonempty_string(task_contract, "task_id")
    capabilities = task_contract.get("capabilities", {})
    if not isinstance(capabilities, dict):
        capabilities = {}
    required = _string_list(
        capabilities.get("required") or task_contract.get("required_capabilities")
    )
    optional = _string_list(
        capabilities.get("optional") or task_contract.get("optional_capabilities")
    )
    domains = _string_list(
        task_contract.get("domains") or task_contract.get("task_domains")
    )
    allow_no_tools = bool(task_contract.get("allow_no_tools", False))
    requires_tools = bool(task_contract.get("requires_tools", bool(required)))
    return {
        "task_id": task_id,
        "required_capabilities": required,
        "optional_capabilities": optional,
        "domains": domains,
        "allow_no_tools": allow_no_tools,
        "requires_tools": requires_tools,
        "contract_fingerprint": _optional_sha256(
            task_contract.get("contract_fingerprint")
            or task_contract.get("task_contract_fingerprint")
        ),
        "binding_fingerprints": _task_binding_fingerprints(task_contract),
        "metadata": _normalize_mapping(task_contract.get("metadata", {})),
    }


def _normalize_policy(policy: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(policy, dict):
        raise RuntimeAdapterRouterError("policy must be an object")
    return {
        "policy_id": str(policy.get("policy_id") or "runtime-router-policy"),
        "known_capabilities": _string_list(policy.get("known_capabilities")),
        "allow_tools": _string_list(policy.get("allow_tools")),
        "deny_tools": _string_list(policy.get("deny_tools")),
        "allow_write_tools": bool(policy.get("allow_write_tools", False)),
        "allowed_risk_classes": _string_list(
            policy.get("allowed_risk_classes") or sorted(READ_RISKS)
        ),
        "required_environment": _normalize_mapping(
            policy.get("required_environment", {})
        ),
    }


def _normalize_routing_policy(policy: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(policy, dict):
        raise RuntimeAdapterRouterError("routing_policy must be an object")
    return {
        "policy_id": str(policy.get("policy_id") or "runtime-router-policy"),
        "allow_generalist_fallback": bool(
            policy.get("allow_generalist_fallback", False)
        ),
        "required_base_model_id": str(policy.get("required_base_model_id") or ""),
        "required_base_revision": str(policy.get("required_base_revision") or ""),
        "required_tokenizer_revision": str(
            policy.get("required_tokenizer_revision") or ""
        ),
        "required_chat_template_sha256": str(
            policy.get("required_chat_template_sha256") or ""
        ),
    }


def _normalize_dispatch_policy(policy: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(policy, dict):
        raise RuntimeAdapterRouterError("dispatch policy must be an object")
    return {
        "policy_id": str(policy.get("policy_id") or "runtime-dispatch-policy"),
        "allow_readonly_dispatch": bool(policy.get("allow_readonly_dispatch", True)),
    }


def _normalize_tool(
    raw_tool: Any, known_capabilities: set[str]
) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    if not isinstance(raw_tool, dict):
        return {"name": "", "version": "", "definition_sha256": ""}, ["tool_not_object"]
    name = str(raw_tool.get("name") or "").strip()
    version = str(raw_tool.get("version") or "").strip()
    definition_sha256 = str(
        raw_tool.get("definition_sha256") or raw_tool.get("definition_hash") or ""
    ).strip()
    capabilities = _string_list(raw_tool.get("capabilities"))
    risk_class = str(raw_tool.get("risk_class") or "").strip()
    write_capable = bool(raw_tool.get("write_capable", risk_class in WRITE_RISKS))
    parameters_schema = raw_tool.get("parameters_schema")
    if not name:
        errors.append("missing_tool_name")
    if not version:
        errors.append("missing_tool_version")
    if not _is_sha256(definition_sha256):
        errors.append("invalid_definition_sha256")
    if not capabilities:
        errors.append("missing_capabilities")
    unknown_capabilities = sorted(set(capabilities) - known_capabilities)
    errors.extend(
        f"unknown_capability:{capability}" for capability in unknown_capabilities
    )
    if risk_class not in KNOWN_RISKS:
        errors.append("unknown_risk_class")
    if not isinstance(parameters_schema, dict) or _schema_subset_errors(
        parameters_schema
    ):
        errors.append("invalid_parameters_schema")
        parameters_schema = {}
    return {
        "name": name,
        "version": version,
        "definition_sha256": definition_sha256,
        "capabilities": capabilities,
        "risk_class": risk_class,
        "write_capable": write_capable,
        "requires_write_authorization": write_capable,
        "parameters_schema": _normalize_mapping(parameters_schema),
        "environment": _normalize_mapping(raw_tool.get("environment", {})),
    }, errors


def _tool_policy_rejections(
    tool: dict[str, Any],
    required: list[str],
    policy: dict[str, Any],
    env: dict[str, Any],
) -> list[str]:
    reasons: list[str] = []
    if required and set(required).isdisjoint(set(tool.get("capabilities", []))):
        reasons.append("missing_required_capability")
    allow_tools = set(policy.get("allow_tools", []))
    deny_tools = set(policy.get("deny_tools", []))
    if allow_tools and tool.get("name") not in allow_tools:
        reasons.append("tool_not_allowed")
    if tool.get("name") in deny_tools:
        reasons.append("tool_denied")
    if tool.get("risk_class") not in set(policy.get("allowed_risk_classes", [])):
        reasons.append("risk_class_not_allowed")
    if tool.get("write_capable") and not policy.get("allow_write_tools"):
        reasons.append("write_tools_not_allowed")
    for key, expected in policy.get("required_environment", {}).items():
        if env.get(key) != expected:
            reasons.append(f"environment_mismatch:{key}")
    for key, expected in tool.get("environment", {}).items():
        if env.get(key) != expected:
            reasons.append(f"tool_environment_mismatch:{key}")
    return reasons


def _normalize_candidate(raw_candidate: Any) -> dict[str, Any]:
    if not isinstance(raw_candidate, dict):
        raw_candidate = {}
    return {
        "candidate_id": str(raw_candidate.get("candidate_id") or ""),
        "candidate_kind": str(
            raw_candidate.get("candidate_kind") or raw_candidate.get("kind") or ""
        ),
        "adapter_id": str(raw_candidate.get("adapter_id") or ""),
        "active_adapter_ids": _string_list(
            raw_candidate.get("active_adapter_ids")
            or (
                [raw_candidate.get("adapter_id")]
                if raw_candidate.get("adapter_id")
                else []
            )
        ),
        "adapter_revision": str(raw_candidate.get("adapter_revision") or ""),
        "adapter_sha256": str(raw_candidate.get("adapter_sha256") or ""),
        "base_model_id": str(raw_candidate.get("base_model_id") or ""),
        "base_revision": str(raw_candidate.get("base_revision") or ""),
        "tokenizer_revision": str(raw_candidate.get("tokenizer_revision") or ""),
        "chat_template_sha256": str(raw_candidate.get("chat_template_sha256") or ""),
        "domains": _string_list(raw_candidate.get("domains")),
        "capabilities": _string_list(raw_candidate.get("capabilities")),
        "registry_entry_id": str(raw_candidate.get("registry_entry_id") or ""),
        "promotion_decision": copy.deepcopy(
            raw_candidate.get("promotion_decision", {})
        ),
        "promotion_evidence_ref": _normalize_artifact_ref(
            raw_candidate.get("promotion_evidence_ref", {})
        ),
        "promotion_binding": _normalize_promotion_binding(
            raw_candidate.get("promotion_binding", {})
        ),
        "composition_members": _string_list(raw_candidate.get("composition_members")),
    }


def _candidate_rejections(
    candidate: dict[str, Any],
    task: dict[str, Any],
    capability_selection: dict[str, Any],
    policy: dict[str, Any],
    env: dict[str, Any],
) -> list[str]:
    reasons: list[str] = []
    for field_name in (
        "candidate_id",
        "candidate_kind",
        "adapter_id",
        "adapter_revision",
        "adapter_sha256",
        "base_model_id",
        "base_revision",
        "tokenizer_revision",
        "chat_template_sha256",
        "registry_entry_id",
    ):
        if not candidate.get(field_name):
            reasons.append(f"missing_{field_name}")
    if candidate["candidate_kind"] not in {"specialist", "generalist", "composition"}:
        reasons.append("unknown_candidate_kind")
    if (
        len(candidate["active_adapter_ids"]) != 1
        or candidate["adapter_id"] not in candidate["active_adapter_ids"]
    ):
        reasons.append("multiple_active_adapters_requested")
    if (
        candidate["candidate_kind"] == "composition"
        and not candidate["composition_members"]
    ):
        reasons.append("composition_missing_members")
    if not _is_sha256(candidate["adapter_sha256"]):
        reasons.append("invalid_adapter_sha256")
    if not _is_sha256(candidate["chat_template_sha256"]):
        reasons.append("invalid_chat_template_sha256")
    required_capabilities = set(task.get("required_capabilities", []))
    if required_capabilities and not required_capabilities.issubset(
        set(candidate["capabilities"])
    ):
        reasons.append("capability_scope_mismatch")
    required_domains = set(task.get("domains", []))
    if candidate["candidate_kind"] == "specialist" and len(candidate["domains"]) != 1:
        reasons.append("specialist_scope_malformed")
    if (
        candidate["candidate_kind"] == "specialist"
        and required_domains
        and not required_domains.intersection(candidate["domains"])
    ):
        reasons.append("task_scope_mismatch")
    if (
        candidate["candidate_kind"] == "generalist"
        and required_domains
        and not required_domains.issubset(set(candidate["domains"]))
    ):
        reasons.append("generalist_scope_mismatch")
    if (
        candidate["candidate_kind"] == "composition"
        and required_domains
        and not required_domains.issubset(set(candidate["domains"]))
    ):
        reasons.append("composition_scope_mismatch")
    if candidate["base_model_id"] != (
        policy["required_base_model_id"] or env.get("base_model_id", "")
    ):
        reasons.append("base_model_mismatch")
    if candidate["base_revision"] != (
        policy["required_base_revision"] or env.get("base_revision", "")
    ):
        reasons.append("base_revision_mismatch")
    if candidate["tokenizer_revision"] != (
        policy["required_tokenizer_revision"] or env.get("tokenizer_revision", "")
    ):
        reasons.append("tokenizer_revision_mismatch")
    if candidate["chat_template_sha256"] != (
        policy["required_chat_template_sha256"] or env.get("chat_template_sha256", "")
    ):
        reasons.append("chat_template_hash_mismatch")
    if capability_selection.get("passed") is not True:
        reasons.append("capability_selection_blocked")
    reasons.extend(_promotion_rejections(candidate))
    return reasons


def _promotion_rejections(candidate: dict[str, Any]) -> list[str]:
    promotion = candidate.get("promotion_decision")
    if not isinstance(promotion, dict) or not promotion:
        return ["promotion_decision_missing"]
    reasons: list[str] = []
    if promotion.get("schema_version") != "hfr.promotion_decision.v1":
        reasons.append("promotion_schema_version_mismatch")
    if promotion.get("passed") is not True:
        reasons.append("promotion_not_passed")
    if promotion.get("recommendation") != "apply_alias_update":
        reasons.append("promotion_recommendation_not_apply_alias_update")
    alias_update = (
        cast(dict[str, Any], promotion.get("alias_update"))
        if isinstance(promotion.get("alias_update"), dict)
        else {}
    )
    alias_authorized = (
        promotion.get("alias_update_authorized") is True
        or alias_update.get("authorized") is True
    )
    if not alias_authorized:
        reasons.append("promotion_alias_update_not_authorized")
    models = (
        cast(dict[str, Any], promotion.get("models"))
        if isinstance(promotion.get("models"), dict)
        else {}
    )
    promoted_candidate = (
        cast(dict[str, Any], models.get("candidate"))
        if isinstance(models.get("candidate"), dict)
        else {}
    )
    if promoted_candidate.get("id") != candidate.get("candidate_id"):
        reasons.append("promotion_models_candidate_id_mismatch")
    reasons.extend(
        f"promotion_evidence_ref_{error}"
        for error in _artifact_ref_errors(candidate.get("promotion_evidence_ref"))
    )
    reasons.extend(_promotion_binding_rejections(candidate))
    return reasons


def _selected_tool_projection(tool: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": tool["name"],
        "version": tool["version"],
        "definition_sha256": tool["definition_sha256"],
        "capabilities": list(tool["capabilities"]),
        "risk_class": tool["risk_class"],
        "write_capable": tool["write_capable"],
        "requires_write_authorization": tool["requires_write_authorization"],
        "parameters_schema": copy.deepcopy(tool["parameters_schema"]),
    }


def _candidate_projection(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "candidate_id": candidate["candidate_id"],
        "candidate_kind": candidate["candidate_kind"],
        "adapter_id": candidate["adapter_id"],
        "active_adapter_ids": list(candidate["active_adapter_ids"]),
        "adapter_revision": candidate["adapter_revision"],
        "adapter_sha256": candidate["adapter_sha256"],
        "base_model_id": candidate["base_model_id"],
        "base_revision": candidate["base_revision"],
        "tokenizer_revision": candidate["tokenizer_revision"],
        "chat_template_sha256": candidate["chat_template_sha256"],
        "domains": list(candidate["domains"]),
        "capabilities": list(candidate["capabilities"]),
        "registry_entry_id": candidate["registry_entry_id"],
        "promotion_evidence": _promotion_projection(candidate),
        "promotion_binding": copy.deepcopy(candidate["promotion_binding"]),
        "composition_members": list(candidate["composition_members"]),
    }


def _promotion_projection(candidate: dict[str, Any]) -> dict[str, Any]:
    promotion = (
        cast(dict[str, Any], candidate.get("promotion_decision"))
        if isinstance(candidate.get("promotion_decision"), dict)
        else {}
    )
    alias_update = (
        cast(dict[str, Any], promotion.get("alias_update"))
        if isinstance(promotion.get("alias_update"), dict)
        else {}
    )
    return {
        "artifact_ref": _normalize_artifact_ref(
            candidate.get("promotion_evidence_ref")
        ),
        "decision_summary": {
            "schema_version": str(promotion.get("schema_version") or ""),
            "passed": promotion.get("passed") is True,
            "recommendation": str(promotion.get("recommendation") or ""),
            "alias_update_authorized": promotion.get("alias_update_authorized") is True
            or alias_update.get("authorized") is True,
            "models_candidate_id": _promotion_models_candidate_id(promotion),
        },
        "decision_sha256": canonical_sha256(promotion),
    }


def _projection_promotion_errors(row: dict[str, Any]) -> list[str]:
    evidence = row.get("promotion_evidence")
    if not isinstance(evidence, dict):
        return ["promotion_evidence_missing"]
    decision = evidence.get("decision_summary")
    if not isinstance(decision, dict):
        return ["promotion_decision_summary_missing"]
    errors: list[str] = []
    ref_errors = _artifact_ref_errors(evidence.get("artifact_ref"))
    errors.extend(f"promotion_evidence_ref_{error}" for error in ref_errors)
    expected = {
        "schema_version": "hfr.promotion_decision.v1",
        "models_candidate_id": row.get("candidate_id"),
    }
    for field_name, expected_value in expected.items():
        if decision.get(field_name) != expected_value:
            errors.append(f"promotion_{field_name}_mismatch")
    if row.get("eligible") is True:
        required_routable = {
            "passed": True,
            "recommendation": "apply_alias_update",
            "alias_update_authorized": True,
        }
        for field_name, expected_value in required_routable.items():
            if decision.get(field_name) != expected_value:
                errors.append(f"promotion_{field_name}_mismatch")
    if not isinstance(evidence.get("decision_sha256"), str) or not _is_sha256(
        evidence.get("decision_sha256")
    ):
        errors.append("promotion_decision_sha256_invalid")
    errors.extend(_projection_binding_errors(row))
    return errors


def _promotion_models_candidate_id(promotion: dict[str, Any]) -> str:
    models = (
        cast(dict[str, Any], promotion.get("models"))
        if isinstance(promotion.get("models"), dict)
        else {}
    )
    candidate = (
        cast(dict[str, Any], models.get("candidate"))
        if isinstance(models.get("candidate"), dict)
        else {}
    )
    return str(candidate.get("id") or "")


def _normalize_promotion_binding(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        value = {}
    return {
        "independent_evidence": value.get("independent_evidence") is True,
        "candidate_id": str(value.get("candidate_id") or ""),
        "registry_entry_id": str(value.get("registry_entry_id") or ""),
        "adapter_id": str(value.get("adapter_id") or ""),
        "adapter_revision": str(value.get("adapter_revision") or ""),
        "adapter_sha256": str(value.get("adapter_sha256") or ""),
        "base_model_id": str(value.get("base_model_id") or ""),
        "base_revision": str(value.get("base_revision") or ""),
        "tokenizer_revision": str(value.get("tokenizer_revision") or ""),
        "chat_template_sha256": str(value.get("chat_template_sha256") or ""),
        "training_result_ref": _normalize_artifact_ref(
            value.get("training_result_ref", {})
        ),
        "evaluation_result_ref": _normalize_artifact_ref(
            value.get("evaluation_result_ref", {})
        ),
    }


def _promotion_binding_rejections(candidate: dict[str, Any]) -> list[str]:
    binding = candidate.get("promotion_binding")
    if not isinstance(binding, dict):
        return ["promotion_binding_missing"]
    errors: list[str] = []
    expected = {
        "independent_evidence": True,
        "candidate_id": candidate.get("candidate_id"),
        "registry_entry_id": candidate.get("registry_entry_id"),
        "adapter_id": candidate.get("adapter_id"),
        "adapter_revision": candidate.get("adapter_revision"),
        "adapter_sha256": candidate.get("adapter_sha256"),
        "base_model_id": candidate.get("base_model_id"),
        "base_revision": candidate.get("base_revision"),
        "tokenizer_revision": candidate.get("tokenizer_revision"),
        "chat_template_sha256": candidate.get("chat_template_sha256"),
    }
    for field_name, expected_value in expected.items():
        if binding.get(field_name) != expected_value:
            errors.append(f"promotion_binding_{field_name}_mismatch")
    errors.extend(
        f"promotion_binding_training_result_ref_{error}"
        for error in _artifact_ref_errors(binding.get("training_result_ref"))
    )
    errors.extend(
        f"promotion_binding_evaluation_result_ref_{error}"
        for error in _artifact_ref_errors(binding.get("evaluation_result_ref"))
    )
    return errors


def _projection_binding_errors(row: dict[str, Any]) -> list[str]:
    binding = row.get("promotion_binding")
    if not isinstance(binding, dict):
        return ["promotion_binding_missing"]
    errors: list[str] = []
    expected = {
        "independent_evidence": True,
        "candidate_id": row.get("candidate_id"),
        "registry_entry_id": row.get("registry_entry_id"),
        "adapter_id": row.get("adapter_id"),
        "adapter_revision": row.get("adapter_revision"),
        "adapter_sha256": row.get("adapter_sha256"),
        "base_model_id": row.get("base_model_id"),
        "base_revision": row.get("base_revision"),
        "tokenizer_revision": row.get("tokenizer_revision"),
        "chat_template_sha256": row.get("chat_template_sha256"),
    }
    for field_name, expected_value in expected.items():
        if binding.get(field_name) != expected_value:
            errors.append(f"promotion_binding_{field_name}_mismatch")
    errors.extend(
        f"promotion_binding_training_result_ref_{error}"
        for error in _artifact_ref_errors(binding.get("training_result_ref"))
    )
    errors.extend(
        f"promotion_binding_evaluation_result_ref_{error}"
        for error in _artifact_ref_errors(binding.get("evaluation_result_ref"))
    )
    return errors


def _validate_approval(
    approval: dict[str, Any],
    *,
    task_fingerprint: str,
    route_fingerprint: str,
    policy_fingerprint: str,
    tool_identity: dict[str, str],
    call_id: str,
    arguments: dict[str, Any],
    now: int,
) -> None:
    if approval.get("expires_at") is None or not isinstance(
        approval.get("expires_at"), int
    ):
        raise DispatchDenied("approval_malformed")
    if approval.get("issued_at") is None or not isinstance(
        approval.get("issued_at"), int
    ):
        raise DispatchDenied("approval_malformed")
    if now > approval["expires_at"]:
        raise DispatchDenied("approval_expired")
    checks = {
        "task_fingerprint": task_fingerprint,
        "route_fingerprint": route_fingerprint,
        "policy_fingerprint": policy_fingerprint,
        "call_id": call_id,
        "arguments_sha256": canonical_sha256(arguments),
    }
    for field_name, expected in checks.items():
        if approval.get(field_name) != expected:
            raise DispatchDenied(f"approval_wrong_{field_name}")
    if approval.get("tool_identity") != tool_identity:
        raise DispatchDenied("approval_wrong_tool")
    binding = {
        "task_fingerprint": approval["task_fingerprint"],
        "route_fingerprint": approval["route_fingerprint"],
        "policy_fingerprint": approval["policy_fingerprint"],
        "tool_identity": approval["tool_identity"],
        "call_id": approval["call_id"],
        "arguments_sha256": approval["arguments_sha256"],
        "issued_at": approval["issued_at"],
        "expires_at": approval["expires_at"],
    }
    if approval.get("approval_fingerprint") != canonical_sha256(binding):
        raise DispatchDenied("approval_fingerprint_mismatch")


def _dispatch_denial(
    call_id: str,
    tool_identity: dict[str, str],
    reason: str,
    *,
    details: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "status": "denied",
        "call_id": call_id,
        "tool": tool_identity,
        "handler_called": False,
        "reason_code": reason,
        "details": details or [],
    }


def _find_selected_tool(
    capability_selection: dict[str, Any], identity: dict[str, str]
) -> dict[str, Any] | None:
    for tool in capability_selection.get("selected_tools", []):
        if not isinstance(tool, dict):
            continue
        if _tool_identity_projection(tool) == identity:
            return tool
    return None


def _tool_identity_projection(value: dict[str, Any]) -> dict[str, str]:
    return {
        "name": str(value.get("name") or ""),
        "version": str(value.get("version") or ""),
        "definition_sha256": str(
            value.get("definition_sha256") or value.get("definition_hash") or ""
        ),
    }


def _tool_key(identity: dict[str, str]) -> str:
    return f"{identity['name']}@{identity['version']}#{identity['definition_sha256']}"


def _validate_subset_value(
    value: Any, schema: Any, path: str, errors: list[str]
) -> None:
    if not isinstance(schema, dict):
        errors.append(f"{path}: schema must be an object")
        return
    if "const" in schema and value != schema["const"]:
        errors.append(f"{path}: expected constant {schema['const']!r}")
    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path}: expected one of {schema['enum']!r}")
    expected_type = schema.get("type")
    if expected_type is not None and not _matches_type(value, expected_type):
        errors.append(f"{path}: expected type {expected_type}")
        return
    if isinstance(value, dict):
        properties = (
            schema.get("properties", {})
            if isinstance(schema.get("properties"), dict)
            else {}
        )
        required = (
            schema.get("required", [])
            if isinstance(schema.get("required"), list)
            else []
        )
        for field_name in required:
            if field_name not in value:
                errors.append(f"{path}.{field_name}: missing required property")
        for field_name, field_value in value.items():
            if field_name in properties:
                _validate_subset_value(
                    field_value, properties[field_name], f"{path}.{field_name}", errors
                )
            elif schema.get("additionalProperties") is False:
                errors.append(
                    f"{path}.{field_name}: additional property is not allowed"
                )
    if isinstance(value, list) and isinstance(schema.get("items"), dict):
        for index, item in enumerate(value):
            _validate_subset_value(item, schema["items"], f"{path}[{index}]", errors)
    if isinstance(value, str):
        if (
            isinstance(schema.get("minLength"), int)
            and len(value) < schema["minLength"]
        ):
            errors.append(f"{path}: string shorter than {schema['minLength']}")
        if (
            isinstance(schema.get("maxLength"), int)
            and len(value) > schema["maxLength"]
        ):
            errors.append(f"{path}: string longer than {schema['maxLength']}")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if (
            isinstance(schema.get("minimum"), (int, float))
            and value < schema["minimum"]
        ):
            errors.append(f"{path}: number below minimum {schema['minimum']}")
        if (
            isinstance(schema.get("maximum"), (int, float))
            and value > schema["maximum"]
        ):
            errors.append(f"{path}: number above maximum {schema['maximum']}")


def _schema_subset_errors(schema: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if schema.get("type") != "object":
        errors.append("root_schema_must_be_object")
    allowed = {
        "type",
        "properties",
        "required",
        "additionalProperties",
        "enum",
        "const",
        "items",
        "minLength",
        "maxLength",
        "minimum",
        "maximum",
    }
    _walk_schema_subset(schema, "$", allowed, errors)
    return errors


def _walk_schema_subset(
    schema: Any, path: str, allowed: set[str], errors: list[str]
) -> None:
    if not isinstance(schema, dict):
        errors.append(f"{path}: schema node must be object")
        return
    for key, value in schema.items():
        if key not in allowed:
            errors.append(f"{path}: unsupported keyword {key}")
        if key == "properties" and isinstance(value, dict):
            for field_name, subschema in value.items():
                _walk_schema_subset(
                    subschema, f"{path}.properties.{field_name}", allowed, errors
                )
        if key == "items" and isinstance(value, dict):
            _walk_schema_subset(value, f"{path}.items", allowed, errors)


def _matches_type(value: Any, expected_type: Any) -> bool:
    if isinstance(expected_type, list):
        return any(_matches_type(value, item) for item in expected_type)
    if expected_type == "object":
        return isinstance(value, dict)
    if expected_type == "array":
        return isinstance(value, list)
    if expected_type == "string":
        return isinstance(value, str)
    if expected_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected_type == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected_type == "boolean":
        return isinstance(value, bool)
    if expected_type == "null":
        return value is None
    return False


def _normalize_mapping(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    return json.loads(canonical_json(value))


def _task_fingerprint(task: dict[str, Any]) -> str:
    explicit = task.get("contract_fingerprint")
    if _is_sha256(explicit):
        return str(explicit)
    return canonical_sha256(task)


def _task_binding_fingerprints(task_contract: dict[str, Any]) -> dict[str, str]:
    fields = (
        "prompt_sha256",
        "system_contract_sha256",
        "developer_contract_sha256",
        "tool_contract_sha256",
        "environment_contract_sha256",
        "policy_contract_sha256",
        "scenario_contract_sha256",
    )
    bindings: dict[str, str] = {}
    raw_bindings = task_contract.get("binding_fingerprints")
    if isinstance(raw_bindings, dict):
        for key, value in raw_bindings.items():
            if isinstance(key, str) and _is_sha256(value):
                bindings[key] = str(value)
    for field_name in fields:
        if _is_sha256(task_contract.get(field_name)):
            bindings[field_name] = str(task_contract[field_name])
    return dict(sorted(bindings.items()))


def _optional_sha256(value: Any) -> str | None:
    if value is None:
        return None
    if not _is_sha256(value):
        raise RuntimeAdapterRouterError(
            "contract_fingerprint must be a SHA-256 hex digest"
        )
    return str(value)


def _normalize_artifact_ref(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {"path": "", "sha256": "", "size_bytes": -1}
    size = value.get("size_bytes")
    if isinstance(size, bool) or not isinstance(size, int):
        size = -1
    return {
        "path": str(value.get("path") or ""),
        "sha256": str(value.get("sha256") or ""),
        "size_bytes": size,
    }


def _artifact_ref_errors(value: Any) -> list[str]:
    ref = _normalize_artifact_ref(value)
    errors: list[str] = []
    path = ref.get("path")
    if (
        not isinstance(path, str)
        or not path
        or path.startswith("/")
        or path.startswith("~")
        or ".." in path.split("/")
    ):
        errors.append("path_invalid")
    if not _is_sha256(ref.get("sha256")):
        errors.append("sha256_invalid")
    size = ref.get("size_bytes")
    if isinstance(size, bool) or not isinstance(size, int) or size < 0:
        errors.append("size_bytes_invalid")
    return errors


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return sorted(
        {item.strip() for item in value if isinstance(item, str) and item.strip()}
    )


def _required_nonempty_string(value: dict[str, Any], field_name: str) -> str:
    item = value.get(field_name)
    if not isinstance(item, str) or not item:
        raise RuntimeAdapterRouterError(f"{field_name} must be a non-empty string")
    return item


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )
