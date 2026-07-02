# Autonomous Operations Model

This project should be structured as durable layer goals plus persistent Codex
goal workers. The layer goals define what needs to exist. The autonomous workers
keep advancing those goals continuously, checkpointing progress as they go. A
separate daily reporter reads the checkpoints and sends a morning progress
email.

For copy-pasteable 24/7 worker prompts, see
`docs/agentic-finetune-24-7-goals.md`.

The daily email is only an update surface. It should not control the work loop,
define the work budget, or cause the agents to stop. The workers should keep
running until the goals are complete, blocked, or explicitly cancelled.

Do not run eight broad layer goals as eight uncoordinated agents with no shared
state. That will create duplicated abstractions, conflicting schemas, and
unclear promotion state. Use the layer goals as epics and let a single
supervisor coordinate persistent work packets under them. Add bounded subagents
only when they improve throughput on clearly independent tasks.

Implementation workers should run in dedicated Codex worktrees or dedicated git
worktrees. The shared checkout is a coordination surface, not the place where
multiple persistent workers should write code.

Treat the repository as public. Do not commit personal email addresses,
home-directory paths, machine-specific workspace paths, private Codex state,
API keys, or local automation configuration. Use placeholders such as
`<repo-root>`, `<goal-worktree-root>`, `<main-push-lock>`, and
`<daily-report-recipient>` in reusable docs and prompts.

## Recommended Structure

### 1. Layer Goals Are Epics

The layer goals in `docs/agentic-finetune-autonomous-goals.md` should remain
stable, high-level contracts:

- Evidence
- Harness
- Data
- Model
- Training
- Eval
- Serving/demo
- Governance

Each goal defines target artifacts, acceptance criteria, and verification.

### 2. Work Packets Are Resumable Units

The autonomous runner should convert layer goals into small, resumable work
packets. These are not daily limits; they are checkpoint boundaries that make
nonstop work auditable and restartable:

- add one schema
- add one CLI dry-run command
- add one gate
- add one registry operation
- add one smoke test
- connect two existing artifacts
- write one missing doc section

Each packet should finish with a checkpoint:

- changed files
- artifacts produced
- commands run
- test results
- next packet recommendation
- blockers or risks

### 3. Persistent Supervisor Owns The Backlog

Maintain a local supervisor state file such as:

`experiments/autonomy/supervisor_state.json`

This file is runtime coordination state and should remain ignored by git unless
a sanitized example is intentionally added. Keep private thread ids, local
worktree paths, lock paths, and email recipients out of public commits.

Suggested shape:

```json
{
  "schema_version": "hfr.autonomy.supervisor_state.v1",
  "updated_at": "2026-07-02T00:00:00Z",
  "active_layer": "evidence",
  "current_packet": "add evidence readiness handoff doc",
  "completed_packets": [],
  "blocked_packets": [],
  "next_packets": [],
  "latest_artifacts": [],
  "latest_verification": [],
  "promotion_readiness": {
    "evidence": "in_progress",
    "harness": "not_started",
    "data": "not_started",
    "model": "not_started",
    "training": "not_started",
    "eval": "not_started",
    "serving_demo": "not_started",
    "governance": "not_started"
  }
}
```

This file is the handoff surface between autonomous runs.
Validate it with the bundled schema before treating a checkpoint as resumable:

```bash
flightrecorder schemas --check experiments/autonomy/supervisor_state.json
```

The bundled schema name is `supervisor_state`, with schema version
`hfr.autonomy.supervisor_state.v1`. It requires explicit readiness entries for
Evidence, Harness, Data, Model, Training, Serving/demo, Eval, and Governance so
the supervisor cannot silently drop a layer while resuming work.

The state contract is intentionally the coordination layer, not a workflow
engine. It should record the active layer, current/completed/blocked/next
packets, recent artifacts, verification evidence, promotion readiness, layer
status, and safety guardrails.

The supervisor should:

- keep selecting the next unblocked packet
- launch bounded subagents only for independent work
- run verification before marking packets complete
- avoid unmanaged long-running processes
- checkpoint state after each packet
- require each implementation worker to commit verified changes to a
  goal-specific branch, push the branch, integrate it to `main`, push `main`,
  and close or delete the temporary branch
- serialize `main` pushes through a configured local main-push lock outside the
  repository
- track branch, commit SHA, main SHA, PR/branch closure status, and delivery
  blockers for each packet
- continue until the layer goals are complete, blocked, or cancelled

### 4. Journal Every Checkpoint

Each autonomous run should append dated journal entries:

`experiments/autonomy/journal/YYYY-MM-DD.md`

These journals are local runtime records. Do not commit them unless they have
been scrubbed of personal paths, private thread ids, credentials, and local
automation details.

Suggested sections:

- Objective
- Work completed
- Files changed
- Artifacts produced
- Verification run
- Failures/blockers
- Next recommended packet

The daily email should summarize the last 24 hours of journal entries, not
invent progress from memory.

### 5. Daily Email Is A Report, Not A Control Plane

The email should be short and evidence-backed. It should not contain giant diffs
or full logs.

Suggested subject:

`Hermes fine-tune infra daily update - YYYY-MM-DD`

Suggested body:

```text
Good morning.

Progress in the last 24 hours:
- <completed packet 1>
- <completed packet 2>

Verification:
- <command>: <pass/fail summary>

Artifacts:
- <path or report>

Current layer status:
- Evidence: <status>
- Harness: <status>
- Data: <status>
- Model: <status>
- Training: <status>
- Eval: <status>
- Serving/demo: <status>
- Governance: <status>

Blockers or risks:
- <none or concise list>

Next planned work:
- <next packet>
```

## Proper Autonomy Pattern

Use this pattern for the persistent goal worker:

1. Read `docs/agentic-finetune-autonomous-goals.md`.
2. Read `experiments/autonomy/supervisor_state.json` if it exists.
3. Inspect the repo state.
4. Choose the highest-priority unblocked work packet.
5. Execute one or more bounded packets.
6. Run verification appropriate to the touched layer.
7. Update supervisor state.
8. Append the dated journal.
9. Continue to the next packet unless complete, blocked, cancelled, or a
   configured handoff/checkpoint limit is reached.

Use this pattern for the daily reporter:

1. Read `experiments/autonomy/supervisor_state.json`.
2. Read journal entries since the previous daily report.
3. Summarize completed work, verification, artifacts, blockers, risks, and next
   planned work.
4. Send, draft, or write the email report.
5. Record that the report was produced.

The worker loop should not leave long-running training, serving, or eval
processes unmanaged. Expensive jobs should be launched only through explicit
plans with artifacts and recovery instructions.

The worker loop should also avoid direct writes to the shared checkout. If a
worker is accidentally running in the shared checkout, it should stop
implementation edits, move to a dedicated worktree, and resume there.

Main integration is allowed only after a packet has been verified. Workers must
acquire the configured local main-push lock, fetch current `origin/main`, merge
or cherry-pick their verified branch, rerun relevant verification, push `main`,
verify the remote SHA, then close/delete the temporary branch.

## Scheduling Options

### Option A: Persistent Codex Goal Plus Daily Reporter

Use Codex goals or persistent Codex threads for the actual work, then use a
daily Codex cron automation only for the morning email report. This matches the
desired behavior: Codex keeps working nonstop, and the email is just a status
digest.

The persistent worker prompt should instruct the agent to:

- work from the supervisor state
- keep advancing bounded packets
- run verification
- checkpoint state and journal entries
- commit verified changes to a goal-specific branch
- integrate verified branch changes to `main` under the main-push lock
- close or delete the temporary branch after `main` is verified
- continue until complete, blocked, or cancelled

The daily reporter prompt should instruct the agent to:

- read the supervisor state and last 24 hours of journals
- summarize progress and verification
- send, draft, or write the morning email

### Option B: Frequent Codex Cron Worker Plus Daily Reporter

If persistent goals are not available or reliable enough, schedule a frequent
worker automation, such as every 30 or 60 minutes, plus a separate daily
reporter automation. This approximates nonstop work while preserving clean
checkpoints.

### Option C: Local Cron Or CI

Use local cron, GitHub Actions, or another scheduler to run a script that starts
the supervisor. This is better if the loop needs stable external credentials,
dedicated GPUs, or production-like scheduling.

### Option D: Manual Start During Bootstrap

Start or resume the persistent supervisor manually until the layer contracts and
email reporting are stable.

## Email Delivery Modes

### Draft-First Mode

The runner creates a Gmail draft every morning. You review and send it manually.
This is safest while the project is still taking shape.

### Auto-Send Mode

The runner sends the email automatically after writing the journal and passing
basic verification. Use this after the report format is stable and the recipient
is explicit.

### File-Only Mode

The runner writes only the journal and an email-ready Markdown file. Use this if
mail credentials are not available in the automation environment.

## Suggested Persistent Goal Prompt

```text
You are the autonomous supervisor for the Hermes agentic fine-tuning
infrastructure project.

Work in <repo-root>.
Read docs/agentic-finetune-autonomous-goals.md and
docs/agentic-finetune-autonomous-operations.md. Read
experiments/autonomy/supervisor_state.json if it exists.

Keep working continuously on the layer goals. Choose the highest-priority
unblocked work packet that advances the platform without launching expensive
compute or unmanaged long-running processes. Prefer Evidence, Harness, Data,
Model, Training, Serving/demo, Eval, then Governance unless supervisor_state
says otherwise.

Implement bounded progress. Keep diffs small. Reuse existing Flight Recorder
patterns. Add or update tests and docs when behavior changes. Run appropriate
verification. Do not train on unredacted traces. Do not promote any model
without governance gates.

Do implementation work from a dedicated Codex worktree or dedicated git
worktree, not from the shared checkout. For verified repository changes, create
or reuse a goal-specific branch, stage only intended files, commit with the Lore
Commit Protocol, run the public-repo privacy scan, push the branch to origin,
verify the remote branch, acquire the configured local main-push lock, integrate
the verified branch into current `origin/main`, rerun relevant verification,
push `main`, verify the remote main SHA, close/delete the temporary branch, and
release the lock. Record branch, commit SHA, main SHA, PR/branch closure status
or delivery blocker, and verification evidence in the local journal.

After each completed packet:
1. Update experiments/autonomy/supervisor_state.json.
2. Append experiments/autonomy/journal/YYYY-MM-DD.md.
3. Record changed files, artifacts, verification, blockers, risks, and next
   packet recommendation.

Continue selecting and executing packets until all goals are complete, a real
blocker prevents meaningful progress, or the run is explicitly cancelled. If a
handoff/checkpoint limit is reached, leave the state clean so the next worker
can resume.

Final response should summarize completed work and verification for this
checkpoint only. The daily email is handled by the reporter.
```

## Suggested Daily Reporter Prompt

```text
You are the daily reporter for the Hermes agentic fine-tuning infrastructure
project.

Work in <repo-root>.
Read experiments/autonomy/supervisor_state.json and all journal entries from
the previous 24 hours.

Prepare a concise morning progress email with:
- progress made in the last 24 hours
- verification results
- artifacts and reports produced
- current layer status
- blockers or risks
- next planned work

If email delivery is configured, create a draft or send according to the
configured mode. Otherwise write experiments/autonomy/daily-email-YYYY-MM-DD.md.
Update supervisor_state with the latest report timestamp.

Do not make unrelated code changes. This job reports progress; it does not run
the work backlog.
```

## My Recommendation

Use persistent Codex goal workers for implementation and a separate daily
reporter automation for the email. Use draft-first daily email at first. After
one or two weeks, switch to auto-send if the reports are concise and useful.

Start with a single persistent supervisor, not one independent infinite worker
per layer. Add subagents only for bounded parallel subtasks inside the
supervisor run, such as one agent auditing schemas while another writes tests.
