# Future RL Training Pipeline

Flight Recorder can now export completed run directories into training-ready
JSONL artifacts. This is a bridge from deterministic eval evidence to future
SFT, preference-tuning, reward-modeling, or RL loops.

It is not a trainer. It does not generate rollouts, update model weights, or
guarantee that the reward function is impossible to game. It gives a future
trainer a clean, deterministic data contract grounded in observed traces.

## Export

Generate normal Flight Recorder runs first:

```bash
./demo.sh
```

For your own scenario directory, the suite runner can generate runs, validation,
and training artifacts in one command:

```bash
flightrecorder run-suite \
  --scenarios scenarios \
  --out runs \
  --export-rl \
  --validate \
  --strict
```

Or export training artifacts from an existing runs directory:

```bash
flightrecorder export-rl \
  --runs runs \
  --out runs/training_export
```

`demo.sh` already runs this export for the included scenarios.

Validate the generated dataset before sending it to downstream jobs:

```bash
flightrecorder validate \
  --runs runs \
  --training-export runs/training_export \
  --strict
```

## Artifacts

The export directory contains:

- `episodes.jsonl`: one trace episode per completed run.
- `rewards.jsonl`: scalar terminal rewards, failed rules, and attribution.
- `preferences.jsonl`: chosen/rejected pairs within the same task family.
- `failure_modes.jsonl`: one failed-rule record per episode with evidence and
  attribution.
- `curriculum.json`: task-family and rule-level rollups for prioritizing
  regression work and future training curricula.
- `manifest.json`: generation settings, counts, output paths, and caveats.

All exports are built from `normalized_trace.json` and `scorecard.json`, so they
use the redacted evidence surface rather than raw sensitive traces.
Absolute source/output paths are redacted from exported metadata by default;
use `--preserve-paths` only for private local debugging.
`flightrecorder validate --strict` checks that counts, episode ids, reward
links, preference references, failure-mode links, and curriculum counts are
internally consistent.

## Episode Records

Each episode includes:

- `episode_id` and source run directory,
- scenario id/title and derived `task_family`,
- prompt recovered from the first user-message event,
- normalized events,
- final answer,
- outcome: pass/fail, score, threshold, reward, failed rules, and summary.

This is the right shape for supervised fine-tuning filters, offline RL dataset
construction, replay inspection, and task-family analytics.

## Reward Records

Rewards are terminal labels derived from the deterministic scorecard.

Available reward scales:

- `score`: score divided by 100, yielding `0.0..1.0`.
- `binary`: passing runs get `1.0`, failing runs get `0.0`.
- `signed`: score mapped to `-1.0..1.0`.

Failed rules include approximate attribution:

- `event` when rule evidence mentions `event #N`,
- `final_answer` when the violation is in the final answer,
- `episode` when only run-level attribution is available.

This gives future trainers a starting point for credit assignment, but it should
not be mistaken for a full environment-level step reward.

## Preference Records

Preference pairs are generated inside each derived task family. For example,
`prompt_injection_good` and `prompt_injection_bad` both map to
`prompt_injection`, so the higher-scoring run becomes `chosen` and the
lower-scoring run becomes `rejected`.

Useful options:

```bash
flightrecorder export-rl \
  --runs runs \
  --out runs/training_export \
  --reward-scale binary \
  --min-score-gap 20 \
  --max-pairs-per-family 10
```

Preference records are suitable as a starting point for DPO-style datasets or
reward-model comparisons.

## Failure Modes And Curriculum

`failure_modes.jsonl` makes the negative signal explicit. Each row links a
failed rule back to its episode, scenario, task family, score, reward, evidence,
criticality, and attribution target. This gives future trainers or benchmark
dashboards a direct way to ask which failure class happened in a run.

`curriculum.json` rolls those rows up by task family and rule id. High-count
critical modes are good candidates for new regression scenarios, targeted
synthetic data generation, or focused reward-model review. Passing episodes in
the same family remain useful positive references, but the curriculum file is
metadata only; it does not choose optimizer settings or update a model.

## Future Trainer Shape

A future training loop can consume the artifacts like this:

```python
import json
from pathlib import Path

episodes = [
    json.loads(line)
    for line in Path("runs/training_export/episodes.jsonl").read_text().splitlines()
]
rewards = [
    json.loads(line)
    for line in Path("runs/training_export/rewards.jsonl").read_text().splitlines()
]
preferences = [
    json.loads(line)
    for line in Path("runs/training_export/preferences.jsonl").read_text().splitlines()
]
failure_modes = [
    json.loads(line)
    for line in Path("runs/training_export/failure_modes.jsonl").read_text().splitlines()
]
curriculum = json.loads(Path("runs/training_export/curriculum.json").read_text())
```

Recommended first uses:

- filter passing episodes into SFT candidates,
- convert preference records into chosen/rejected pairs,
- train a small reward model on scorecard-derived labels,
- build failure-mode dashboards or curricula from explicit failed-rule rows,
- gate Hermes skill/model changes by re-exporting and comparing rewards.

## Boundaries

This pipeline is useful only when scenarios are meaningful. Weak scenarios can
produce weak rewards, and any learned policy can overfit or reward-hack shallow
assertions. Keep expanding scenario suites, vary task families, and review
reports alongside aggregate rewards.
