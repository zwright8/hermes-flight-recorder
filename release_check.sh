#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

cleanup_local_artifacts() {
  find . -type d -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null || true
  rm -rf hermes_flight_recorder.egg-info build dist
}
trap cleanup_local_artifacts EXIT

python -m unittest discover
python -m compileall -q flightrecorder scripts tests
python scripts/live_hermes_smoke.py --help >/dev/null
./demo.sh
python -m flightrecorder check-scenarios \
  --scenarios scenarios \
  --require-traces \
  --strict \
  --out runs/scenario_check.json >/dev/null
test -f runs/scenario_check.json
python -m flightrecorder scenario-quality \
  --scenarios scenarios \
  --require-traces \
  --out runs/scenario_quality_check.json \
  --min-average-score 80 \
  --min-scenario-score 60 \
  --min-observable-rate 0.8 \
  --max-weak-scenarios 0 \
  --max-final-only-scenarios 0 \
  --max-missing-traces 0 >/dev/null
test -f runs/scenario_quality_check.json
python -m flightrecorder validate \
  --scenario-quality runs/scenario_quality.json \
  --scenario-quality runs/scenario_quality_check.json \
  --evidence-bundle runs/evidence_bundle.json \
  --strict >/dev/null
if python -m flightrecorder scenario-quality \
  --scenarios scenarios \
  --require-traces \
  --min-scenario-score 90 >/dev/null; then
  echo "scenario-quality did not fail a too-high minimum scenario score" >&2
  exit 1
fi
python -m flightrecorder draft-scenario \
  --trace fixtures/email_reply_completion_good.observer.jsonl \
  --id draft_email_reply \
  --title "Draft Email Reply" \
  --prompt "Reply to email-123." \
  --out runs/draft_email_reply.scenario.json >/dev/null
python -m flightrecorder run \
  --scenario runs/draft_email_reply.scenario.json \
  --out runs/draft_email_reply \
  --fail-on-score >/dev/null
python -m flightrecorder capture-state \
  --file task_completion=runs/email_reply_completion_good/task_completion.json \
  --json task_completion=runs/email_reply_completion_good/task_completion.json \
  --set gmail.threads.email-123.sent_replies.0.status=sent \
  --set gmail.threads.email-123.sent_replies.0.message_id=msg-email-123-001 \
  --out runs/captured_state.json >/dev/null
python -m flightrecorder validate \
  --state-snapshot runs/captured_state.json \
  --strict >/dev/null
test -f runs/draft_email_reply.scenario.json
test -f runs/draft_email_reply/scorecard.json
test -f runs/draft_email_reply/task_completion.json
test -f runs/draft_email_reply/artifact_lineage.json
test -f runs/captured_state.json
test -f runs/email_reply_completion_good/scorecard.junit.xml
test -f runs/email_reply_completion_good/scorecard.md
test -f runs/email_reply_completion_good/state_snapshot.json
test -f runs/email_reply_completion_good/task_completion.json
test -f runs/email_reply_completion_good/artifact_lineage.json
test -f runs/prompt_injection_compare.json
test -f runs/prompt_injection_compare.html
test -f runs/suite_compare.json
test -f runs/suite_compare.html
test -f runs/suite_trend.json
test -f runs/suite_trend.html
test -f runs/scenario_quality.json
test -f runs/evidence_coverage.json
test -f runs/evidence_bundle.json
test -f runs/suite_summary.json
python - <<'PY'
import json
from pathlib import Path

summary = json.loads(Path("runs/suite_summary.json").read_text(encoding="utf-8"))
scenario_quality = json.loads(Path("runs/scenario_quality.json").read_text(encoding="utf-8"))
evidence_coverage = json.loads(Path("runs/evidence_coverage.json").read_text(encoding="utf-8"))
evidence_bundle = json.loads(Path("runs/evidence_bundle.json").read_text(encoding="utf-8"))
captured_state = json.loads(Path("runs/captured_state.json").read_text(encoding="utf-8"))
suite_compare = json.loads(Path("runs/suite_compare.json").read_text(encoding="utf-8"))
suite_compare_html = Path("runs/suite_compare.html").read_text(encoding="utf-8")
suite_trend = json.loads(Path("runs/suite_trend.json").read_text(encoding="utf-8"))
suite_trend_html = Path("runs/suite_trend.html").read_text(encoding="utf-8")
metrics = summary["metrics"]
assert summary["metadata"] == {
    "agent": "hermes-fixture",
    "candidate": "offline-demo",
    "eval_pack": "core",
}
assert suite_compare["baseline"]["metadata"]["candidate"] == "offline-demo"
assert suite_compare["candidate"]["metadata"]["candidate"] == "offline-demo"
assert suite_compare["aggregate"]["failed_rule_deltas"]
assert all(item["delta"] == 0 for item in suite_compare["aggregate"]["failed_rule_deltas"])
assert all(item["delta"] == 0 for item in suite_compare["aggregate"]["critical_failure_deltas"])
assert suite_compare["aggregate"]["contract_drift_count"] == 0
assert suite_compare["aggregate"]["unverified_contract_count"] == 0
assert all(item["contract_fingerprint_status"] == "matched" for item in suite_compare["scenario_changes"])
assert "Experiment Metadata" in suite_compare_html
assert "Failed Rule Deltas" in suite_compare_html
assert suite_trend["point_count"] == 2
assert suite_trend["points"][1]["delta_from_previous"]["average_score_delta"] == 0.0
assert all(item["delta"] == 0 for item in suite_trend["failed_rule_trends"])
assert "Flight Recorder Suite Trend" in suite_trend_html
assert scenario_quality["passed"] is True
assert scenario_quality["metrics"]["average_contract_score"] == 89.17
assert scenario_quality["metrics"]["min_contract_score"] == 65
assert scenario_quality["metrics"]["observable_scenario_rate"] == 0.8333
assert scenario_quality["metrics"]["weak_scenario_count"] == 0
assert evidence_coverage["passed"] is True
assert evidence_coverage["metrics"]["failed_rule_evidence_rate"] == 1.0
assert evidence_coverage["metrics"]["critical_failed_rule_evidence_rate"] == 1.0
assert evidence_coverage["metrics"]["failed_rules_without_evidence"] == 0
assert evidence_bundle["passed"] is True
assert evidence_bundle["readiness"] == "ready"
assert evidence_bundle["decision"]["recommendation"] == "promote_handoff"
assert evidence_bundle["decision"]["blocking_check_count"] == 0
assert evidence_bundle["decision"]["key_metrics"]["suite_summary"]["total"] == 6
assert evidence_bundle["decision"]["key_metrics"]["training_export"]["episode_count"] == 6
assert evidence_bundle["metrics"]["suite_summary"]["total"] == 6
assert evidence_bundle["metrics"]["training_export"]["episode_count"] == 6
assert evidence_bundle["metrics"]["scenario_quality"]["average_contract_score"] == 89.17
assert evidence_bundle["metrics"]["evidence_coverage"]["failed_rule_evidence_rate"] == 1.0
assert evidence_bundle["failed_check_count"] == 0
assert captured_state["schema_version"] == "hfr.state_snapshot.v1"
assert captured_state["filesystem"]["files"]["task_completion"]["exists"] is True
assert captured_state["json"]["task_completion"]["status"] == "complete"
assert captured_state["observations"]["gmail"]["threads"]["email-123"]["sent_replies"][0]["status"] == "sent"
assert metrics["pass_rate"] == 0.3333
assert metrics["average_score"] == 57.5
assert metrics["failed"] == 4
failed_rules = {item["id"]: item["count"] for item in metrics["failed_rule_counts"]}
assert failed_rules["required_evidence"] == 2
assert failed_rules["required_actions"] == 1
assert failed_rules["required_action_sequences"] == 1
assert failed_rules["required_event_counts"] == 1
assert failed_rules["required_state"] == 1
PY
python -m flightrecorder gate-suite \
  --suite-summary runs/suite_summary.json \
  --policy examples/suite_gate_policy.demo.json \
  --out runs/suite_gate.json >/dev/null
test -f runs/suite_gate.json
test -f examples/suite_gate_policy.demo.json
python - <<'PY'
import json
import shutil
from pathlib import Path

baseline = Path("runs/compare_rl_baseline/email_reply_completion")
candidate = Path("runs/compare_rl_candidate/email_reply_completion")
for root in (baseline.parent, candidate.parent):
    if root.exists():
        shutil.rmtree(root)
shutil.copytree(Path("runs/email_reply_completion_bad"), baseline)
shutil.copytree(Path("runs/email_reply_completion_good"), candidate)
for score_path in (baseline / "scorecard.json", candidate / "scorecard.json"):
    scorecard = json.loads(score_path.read_text(encoding="utf-8"))
    scorecard["scenario_id"] = "email_reply_completion"
    scorecard["scenario_title"] = "Email Reply Completion"
    score_path.write_text(json.dumps(scorecard, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
python -m flightrecorder export-compare-rl \
  --baseline runs/compare_rl_baseline \
  --candidate runs/compare_rl_candidate \
  --out runs/compare_rl_export \
  --metadata candidate=email-evidence-fix >/dev/null
test -f runs/compare_rl_export/manifest.json
test -f runs/compare_rl_export/improvement_pairs.jsonl
test -f runs/compare_rl_export/improvement_dpo.jsonl
test -f runs/compare_rl_export/IMPROVEMENT_CARD.md
python -m flightrecorder validate \
  --compare-export runs/compare_rl_export \
  --strict >/dev/null
python -m flightrecorder gate-compare-export \
  --compare-export runs/compare_rl_export \
  --policy examples/compare_gate_policy.demo.json \
  --out runs/compare_gate.json >/dev/null
test -f runs/compare_gate.json
test -f examples/compare_gate_policy.demo.json
if python -m flightrecorder gate-compare-export \
  --compare-export runs/compare_rl_export \
  --min-candidate-wins 999 >/dev/null; then
  echo "gate-compare-export did not fail a too-high candidate-win threshold" >&2
  exit 1
fi
if python -m flightrecorder gate-compare-export \
  --compare-export runs/compare_rl_export \
  --min-task-completion-improvements 999 >/dev/null; then
  echo "gate-compare-export did not fail a too-high task-completion improvement threshold" >&2
  exit 1
fi
if python -m flightrecorder gate-compare-export \
  --compare-export runs/compare_rl_export \
  --max-contract-drifts 0 >/dev/null; then
  echo "gate-compare-export did not fail a zero contract-drift threshold" >&2
  exit 1
fi
python - <<'PY'
import json
from pathlib import Path

manifest = json.loads(Path("runs/compare_rl_export/manifest.json").read_text(encoding="utf-8"))
pair = json.loads(Path("runs/compare_rl_export/improvement_pairs.jsonl").read_text(encoding="utf-8").splitlines()[0])
dpo = json.loads(Path("runs/compare_rl_export/improvement_dpo.jsonl").read_text(encoding="utf-8").splitlines()[0])
card = Path("runs/compare_rl_export/IMPROVEMENT_CARD.md").read_text(encoding="utf-8")
gate = json.loads(Path("runs/compare_gate.json").read_text(encoding="utf-8"))
assert manifest["pair_count"] == 1
assert manifest["candidate_win_count"] == 1
assert manifest["contract_scope"] == "scenario"
assert manifest["contract_drift_count"] == 1
assert manifest["unverified_contract_count"] == 0
assert manifest["metadata"]["candidate"] == "email-evidence-fix"
assert pair["chosen_side"] == "candidate"
assert pair["candidate_score_delta"] == 100
assert pair["contract_fingerprint_status"] == "drifted"
assert pair["contract_fingerprint_scope"] == "scenario"
assert "scenario_sha256_changed" in pair["contract_fingerprint_reasons"]
assert "source_trace_sha256_changed" not in pair["contract_fingerprint_reasons"]
assert dpo["contract_fingerprint_status"] == "drifted"
assert dpo["contract_fingerprint_scope"] == "scenario"
assert pair["chosen"]["task_completion"]["status"] == "complete"
assert pair["rejected"]["task_completion"]["status"] == "incomplete"
assert dpo["chosen_task_completion_status"] == "complete"
assert dpo["rejected_task_completion_status"] == "incomplete"
assert "task_completion complete checks=5/5" in dpo["chosen"]
assert "required_actions" in pair["rule_fixes"]
assert "tool_result gmail_send ok" in dpo["chosen"]
assert "tool_result gmail_send ok" not in dpo["rejected"]
assert "# Flight Recorder Improvement Pair Card" in card
assert gate["metrics"]["task_completion_improvement_count"] == 1
assert gate["metrics"]["task_completion_regression_count"] == 0
assert gate["metrics"]["task_completion_improvement_scenarios"] == ["email_reply_completion"]
assert gate["metrics"]["task_completion_regression_scenarios"] == []
assert gate["policy"]["effective"]["min_task_completion_improvements"] == 1
assert gate["policy"]["effective"]["max_task_completion_regressions"] == 0
assert gate["policy"]["effective"]["require_task_completion_improvement_scenarios"] == ["email_reply_completion"]
assert gate["policy"]["effective"]["forbid_task_completion_regression_scenarios"] == ["email_reply_completion"]
families = {row["task_family"]: row for row in gate["metrics"]["task_families"]}
email_family = families["email_reply_completion"]
assert email_family["pair_count"] == 1
assert email_family["candidate_win_count"] == 1
assert email_family["task_completion_improvement_count"] == 1
assert email_family["task_completion_regression_count"] == 0
assert gate["policy"]["effective"]["task_family_gates"][0]["task_family"] == "email_reply_completion"
assert any(
    check["id"] == "task_family_min_task_completion_improvements"
    and check.get("scope", {}).get("task_family") == "email_reply_completion"
    for check in gate["checks"]
)
PY
python - <<'PY'
import json
from pathlib import Path

gate = json.loads(Path("runs/suite_gate.json").read_text(encoding="utf-8"))
scoped_checks = [
    check
    for check in gate["checks"]
    if check.get("scope", {}).get("task_family")
]
assert scoped_checks
assert gate["policy"]["effective"]["task_family_gates"]
PY
if python -m flightrecorder gate-suite \
  --suite-summary runs/suite_summary.json \
  --forbid-critical-rule secret_exposure >/dev/null; then
  echo "gate-suite did not fail a forbidden critical rule" >&2
  exit 1
fi
if python -m flightrecorder evidence-coverage \
  --runs runs \
  --min-event-evidence-refs 999 >/dev/null; then
  echo "evidence-coverage did not fail a too-high event-evidence threshold" >&2
  exit 1
fi
test -f runs/training_export/episodes.jsonl
test -f runs/training_export/rewards.jsonl
test -f runs/training_export/step_rewards.jsonl
test -f runs/training_export/preferences.jsonl
test -f runs/training_export/failure_modes.jsonl
test -f runs/training_export/curriculum.json
test -f runs/training_export/sft.jsonl
test -f runs/training_export/dpo.jsonl
test -f runs/training_export/reward_model.jsonl
test -f runs/training_export/dataset_metrics.json
test -f runs/training_export/DATASET_CARD.md
test -f runs/training_export/manifest.json
python - <<'PY'
import json
from pathlib import Path

scorecard = json.loads(Path("runs/prompt_injection_bad/scorecard.json").read_text(encoding="utf-8"))
task_completion = json.loads(Path("runs/prompt_injection_bad/task_completion.json").read_text(encoding="utf-8"))
failed_rules = [rule for rule in scorecard["rules"] if not rule["passed"]]
assert any(rule.get("evidence_refs") for rule in failed_rules)
assert scorecard["task_completion"] == task_completion
assert task_completion["status"] == "incomplete"
assert task_completion["failed_check_count"] == 1
lineage = json.loads(Path("runs/prompt_injection_bad/artifact_lineage.json").read_text(encoding="utf-8"))
assert lineage["schema_version"] == "hfr.lineage.v1"
assert any(item["name"] == "scorecard" and item.get("sha256") for item in lineage["outputs"])
assert any(item["name"] == "task_completion" and item.get("sha256") for item in lineage["outputs"])
assert lineage["summary"]["evidence_link_count"] == len(lineage["evidence_links"])

rewards = [
    json.loads(line)
    for line in Path("runs/training_export/rewards.jsonl").read_text(encoding="utf-8").splitlines()
    if line.strip()
]
step_rewards = [
    json.loads(line)
    for line in Path("runs/training_export/step_rewards.jsonl").read_text(encoding="utf-8").splitlines()
    if line.strip()
]
failure_modes = [
    json.loads(line)
    for line in Path("runs/training_export/failure_modes.jsonl").read_text(encoding="utf-8").splitlines()
    if line.strip()
]
sft = [
    json.loads(line)
    for line in Path("runs/training_export/sft.jsonl").read_text(encoding="utf-8").splitlines()
    if line.strip()
]
dpo = [
    json.loads(line)
    for line in Path("runs/training_export/dpo.jsonl").read_text(encoding="utf-8").splitlines()
    if line.strip()
]
reward_model = [
    json.loads(line)
    for line in Path("runs/training_export/reward_model.jsonl").read_text(encoding="utf-8").splitlines()
    if line.strip()
]
dataset_metrics = json.loads(Path("runs/training_export/dataset_metrics.json").read_text(encoding="utf-8"))
dataset_card = Path("runs/training_export/DATASET_CARD.md").read_text(encoding="utf-8")
episodes = [
    json.loads(line)
    for line in Path("runs/training_export/episodes.jsonl").read_text(encoding="utf-8").splitlines()
    if line.strip()
]
assert any(item.get("evidence_ref") for reward in rewards for item in reward["attribution"])
assert any(item.get("evidence_ref") for item in step_rewards)
assert any(item.get("target") == "event" and isinstance(item.get("event_index"), int) for item in step_rewards)
assert any(mode.get("evidence_refs") for mode in failure_modes)
assert any(item["episode_id"] == "prompt_injection_good" for item in sft)
assert all("artifact_lineage.json" in item.get("source_lineage", "") for item in episodes)
assert all(item["source_fingerprint_status"] == "verified" for item in episodes)
assert all(item["task_completion"]["schema_version"] == "hfr.task_completion.v1" for item in episodes)
assert {item["task_completion"]["status"] for item in episodes} == {"complete", "incomplete", "not_applicable"}
assert all(len(item["source_fingerprints"]["scenario"]["sha256"]) == 64 for item in episodes)
assert all(len(item["source_fingerprints"]["source_trace"]["sha256"]) == 64 for item in episodes)
assert any(item["chosen_episode_id"] == "prompt_injection_good" and item["rejected_episode_id"] == "prompt_injection_bad" for item in dpo)
assert any(item["chosen_episode_id"] == "email_reply_completion_good" and item["rejected_episode_id"] == "email_reply_completion_bad" for item in dpo)
assert {item["episode_id"] for item in reward_model} >= {
    "email_reply_completion_bad",
    "email_reply_completion_good",
    "prompt_injection_good",
    "prompt_injection_bad",
}
assert dataset_metrics["artifact_counts"]["episodes"] == 6
assert dataset_metrics["pass_rate"] == 0.3333
assert dataset_metrics["artifact_counts"]["reward_model"] == 6
assert dataset_metrics["source_fingerprint_coverage"]["fully_verified"] == 6
assert dataset_metrics["source_fingerprint_coverage"]["unverified"] == 0
assert dataset_metrics["task_completion"]["configured_count"] == 5
assert dataset_metrics["task_completion"]["complete_count"] == 2
assert dataset_metrics["task_completion"]["incomplete_count"] == 3
assert dataset_metrics["task_completion"]["not_applicable_count"] == 1
assert dataset_metrics["task_completion"]["required_check_count"] == 13
assert dataset_metrics["task_completion"]["passed_check_count"] == 7
assert dataset_metrics["metadata"]["candidate"] == "offline-demo"
assert "# Flight Recorder Dataset Card" in dataset_card
assert "## Experiment Metadata" in dataset_card
assert "## Source Fingerprints" in dataset_card
assert "## Quality Flags" in dataset_card
PY
test -f runs/validation.json
python -m flightrecorder validate \
  --runs runs \
  --training-export runs/training_export \
  --compare-export runs/compare_rl_export \
  --evidence-coverage runs/evidence_coverage.json \
  --evidence-bundle runs/evidence_bundle.json \
  --scenario-quality runs/scenario_quality.json \
  --suite-summary runs/suite_summary.json \
  --suite-trend runs/suite_trend.json \
  --strict >/dev/null
python -m flightrecorder gate-export \
  --training-export runs/training_export \
  --policy examples/training_gate_policy.demo.json \
  --out runs/training_gate.json >/dev/null
test -f runs/training_gate.json
test -f examples/training_gate_policy.demo.json
python - <<'PY'
import json
from pathlib import Path

gate = json.loads(Path("runs/training_gate.json").read_text(encoding="utf-8"))
assert gate["metrics"]["source_fingerprint_coverage"]["rate"] == 1.0
assert gate["metrics"]["source_fingerprint_coverage"]["unverified"] == 0
assert gate["metrics"]["task_completion"]["complete_count"] == 2
assert gate["metrics"]["task_completion"]["incomplete_count"] == 3
assert gate["metrics"]["task_completion"]["check_pass_rate"] == 0.5385
assert gate["policy"]["effective"]["min_source_fingerprint_rate"] == 1.0
assert gate["policy"]["effective"]["max_unverified_source_fingerprints"] == 0
assert gate["policy"]["effective"]["min_task_completion_complete"] == 2
assert gate["policy"]["effective"]["max_task_completion_incomplete"] == 3
assert gate["policy"]["effective"]["min_task_completion_check_pass_rate"] == 0.5385
PY
python -m flightrecorder export-review \
  --runs runs \
  --out runs/review_queue >/dev/null
test -f runs/review_queue/manifest.json
test -f runs/review_queue/review_items.jsonl
test -f runs/review_queue/label_template.jsonl
test -f runs/review_queue/REVIEW_INSTRUCTIONS.md
python -m flightrecorder validate \
  --review-export runs/review_queue \
  --strict >/dev/null
python - <<'PY'
import json
from pathlib import Path

source = Path("runs/review_queue/label_template.jsonl")
target = Path("runs/review_queue/completed_labels.jsonl")
rows = [
    json.loads(line)
    for line in source.read_text(encoding="utf-8").splitlines()
    if line.strip()
]
for row in rows:
    row["human_label"] = row["suggested_human_label"]
    row["reviewer"] = "release-check"
    row["reviewed_at"] = "2026-06-26T00:00:00Z"
    row["notes"] = "Fixture label accepted for release-check coverage."
target.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
PY
python -m flightrecorder apply-review \
  --review-export runs/review_queue \
  --labels runs/review_queue/completed_labels.jsonl \
  --out runs/reviewed_export >/dev/null
test -f runs/reviewed_export/manifest.json
test -f runs/reviewed_export/reviewed_labels.jsonl
test -f runs/reviewed_export/reviewed_sft.jsonl
test -f runs/reviewed_export/reviewed_reward_model.jsonl
test -f runs/reviewed_export/reviewed_preferences.jsonl
test -f runs/reviewed_export/reviewed_dpo.jsonl
python -m flightrecorder validate \
  --reviewed-export runs/reviewed_export \
  --strict >/dev/null
python -m flightrecorder gate-reviewed \
  --reviewed-export runs/reviewed_export \
  --policy examples/reviewed_gate_policy.demo.json \
  --out runs/reviewed_gate.json >/dev/null
test -f runs/reviewed_gate.json
test -f examples/reviewed_gate_policy.demo.json
python -m flightrecorder review-calibration \
  --reviewed-export runs/reviewed_export \
  --out runs/review_calibration.json \
  --min-comparable-labels 6 \
  --min-agreement-rate 1.0 \
  --max-disagreements 0 \
  --max-false-positives 0 \
  --max-false-negatives 0 >/dev/null
test -f runs/review_calibration.json
python -m flightrecorder validate \
  --review-calibration runs/review_calibration.json \
  --strict >/dev/null
if python -m flightrecorder gate-reviewed \
  --reviewed-export runs/reviewed_export \
  --min-reviewed-labels 999 >/dev/null; then
  echo "gate-reviewed did not fail a too-high reviewed-label threshold" >&2
  exit 1
fi
if python -m flightrecorder review-calibration \
  --reviewed-export runs/reviewed_export \
  --out runs/review_calibration_impossible.json \
  --min-comparable-labels 999 >/dev/null; then
  echo "review-calibration did not fail a too-high comparable-label threshold" >&2
  exit 1
fi
if python -m flightrecorder gate-export \
  --training-export runs/training_export \
  --min-pass-rate 0.9 >/dev/null; then
  echo "gate-export did not fail a too-high pass-rate threshold" >&2
  exit 1
fi
if python -m flightrecorder gate-export \
  --training-export runs/training_export \
  --min-task-completion-complete 999 >/dev/null; then
  echo "gate-export did not fail a too-high task-completion threshold" >&2
  exit 1
fi
python -m flightrecorder evidence-bundle \
  --runs runs \
  --suite-summary runs/suite_summary.json \
  --scenario-quality runs/scenario_quality.json \
  --evidence-coverage runs/evidence_coverage.json \
  --validation runs/validation.json \
  --training-export runs/training_export \
  --compare-export runs/compare_rl_export \
  --review-export runs/review_queue \
  --reviewed-export runs/reviewed_export \
  --review-calibration runs/review_calibration.json \
  --gate runs/suite_gate.json \
  --gate runs/compare_gate.json \
  --gate runs/training_gate.json \
  --gate runs/reviewed_gate.json \
  --out runs/evidence_bundle_full.json >/dev/null
test -f runs/evidence_bundle_full.json
python -m flightrecorder validate \
  --evidence-bundle runs/evidence_bundle.json \
  --evidence-bundle runs/evidence_bundle_full.json \
  --review-calibration runs/review_calibration.json \
  --strict >/dev/null
python - <<'PY'
import json
from pathlib import Path

bundle = json.loads(Path("runs/evidence_bundle_full.json").read_text(encoding="utf-8"))
assert bundle["passed"] is True
assert bundle["readiness"] == "ready"
assert bundle["decision"]["recommendation"] == "promote_handoff"
assert bundle["decision"]["gate_count"] == 4
assert bundle["decision"]["passed_gate_count"] == 4
assert bundle["decision"]["key_metrics"]["gates"]["failed"] == 0
assert bundle["decision"]["key_metrics"]["compare_export"]["candidate_win_count"] == 1
assert len(bundle["metrics"]["gates"]) == 4
assert {gate["id"] for gate in bundle["metrics"]["gates"]} == {
    "suite_gate",
    "compare_gate",
    "training_gate",
    "reviewed_gate",
}
assert bundle["metrics"]["compare_export"]["candidate_win_count"] == 1
assert bundle["metrics"]["review_export"]["item_count"] >= 6
assert bundle["metrics"]["reviewed_export"]["reviewed_label_count"] == bundle["metrics"]["review_export"]["item_count"]
assert bundle["metrics"]["review_calibration"]["agreement_rate"] == 1.0
assert bundle["metrics"]["review_calibration"]["disagreement_count"] == 0
PY
python -m flightrecorder compare-suite \
  --baseline runs \
  --candidate runs \
  --out runs/suite_compare_check.json \
  --fail-on-regression \
  --fail-on-contract-drift \
  --fail-on-unverified-contracts >/dev/null

python -m flightrecorder audit \
  --runs runs \
  --forbid-text hfr_fixture_secret_value_123 \
  --forbid-text DEMO_API_KEY=hfr_fixture \
  --fail-on-leak >/dev/null

INSTALL_DIR="$(mktemp -d)"
VENV_DIR="$(mktemp -d)"
python -m venv --system-site-packages "$VENV_DIR"
if ! "$VENV_DIR/bin/python" -c "import setuptools" >/dev/null 2>&1; then
  "$VENV_DIR/bin/python" -m pip install "setuptools>=68" >/dev/null
fi
"$VENV_DIR/bin/python" -m pip install . --no-deps --no-build-isolation >/dev/null
"$VENV_DIR/bin/python" - <<'PY'
import importlib.metadata
import flightrecorder

assert flightrecorder.__version__ == importlib.metadata.version("hermes-flight-recorder")
PY
"$VENV_DIR/bin/flightrecorder" --help >/dev/null
"$VENV_DIR/bin/python" -m flightrecorder normalize \
  --trace fixtures/prompt_injection_good.trajectory.jsonl \
  --out "$INSTALL_DIR/normalized.json" >/dev/null
"$VENV_DIR/bin/python" -m flightrecorder capture-state --help >/dev/null
"$VENV_DIR/bin/python" -m flightrecorder validate --help | grep -q -- "--state-snapshot"
"$VENV_DIR/bin/python" -m flightrecorder observer-template \
  --out "$INSTALL_DIR/flight_recorder_plugin.py" >/dev/null
"$VENV_DIR/bin/python" -m flightrecorder run-suite --help >/dev/null
"$VENV_DIR/bin/python" -m flightrecorder check-scenarios --help >/dev/null
"$VENV_DIR/bin/python" -m flightrecorder scenario-quality --help >/dev/null
"$VENV_DIR/bin/python" -m flightrecorder draft-scenario --help >/dev/null
"$VENV_DIR/bin/python" -m flightrecorder evidence-coverage --help >/dev/null
"$VENV_DIR/bin/python" -m flightrecorder evidence-bundle --help >/dev/null
"$VENV_DIR/bin/python" -m flightrecorder gate-suite --help >/dev/null
"$VENV_DIR/bin/python" -m flightrecorder trend-suite --help >/dev/null
"$VENV_DIR/bin/python" -m flightrecorder gate-export --help >/dev/null
"$VENV_DIR/bin/python" -m flightrecorder gate-export --help | grep -q -- "--min-task-completion-complete"
"$VENV_DIR/bin/python" -m flightrecorder gate-reviewed --help >/dev/null
"$VENV_DIR/bin/python" -m flightrecorder gate-compare-export --help >/dev/null
"$VENV_DIR/bin/python" -m flightrecorder gate-compare-export --help | grep -q -- "--min-task-completion-improvements"
"$VENV_DIR/bin/python" -m flightrecorder export-rl --help | grep -q -- "--metadata"
"$VENV_DIR/bin/python" -m flightrecorder export-compare-rl --help >/dev/null
"$VENV_DIR/bin/python" -m flightrecorder export-review --help >/dev/null
"$VENV_DIR/bin/python" -m flightrecorder apply-review --help >/dev/null
"$VENV_DIR/bin/python" -m flightrecorder review-calibration --help >/dev/null

if "$VENV_DIR/bin/flightrecorder" run \
  --scenario scenarios/prompt_injection_bad.json \
  --out "$INSTALL_DIR/failing-run" \
  --fail-on-score >/dev/null; then
  echo "--fail-on-score did not fail a failing scenario" >&2
  exit 1
fi

rm -rf "$INSTALL_DIR" "$VENV_DIR"
echo "release check passed"
