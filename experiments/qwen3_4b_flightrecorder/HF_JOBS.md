# Hugging Face Jobs Launch Notes

The training script is `scripts/train_agentic_lora.py`. It is a UV-compatible
script with dependencies declared in the file header. Training runs report to
Trackio by default with project `hermes-flightrecorder-agentic`; set
`TRACKIO_SPACE_ID` or pass `--trackio-space-id` when the run should sync to a
persistent Space.

## Data Access

HF Jobs must be able to read this experiment directory. Use one of these paths:

1. Push this branch to `https://github.com/zwright8/hermes-flight-recorder.git`
   and clone it in the job before running training.
2. Publish `experiments/qwen3_4b_flightrecorder/data/` as a private Hugging Face
   dataset, then run the script from a checkout that downloads that dataset.
3. For Codex MCP-launched jobs, submit inline job code or a URL to a published
   script. MCP-launched jobs cannot see this workstation's local file paths.

The current local bundle is tiny, so cloning the repo branch is the simplest
proof path once the branch has been pushed. The examples below assume either a
local `hf` CLI invocation that uploads the UV script, or an equivalent MCP job
that first clones a pushed branch.

## Smoke Dry-Run

```bash
hf jobs uv run \
  --timeout 20m \
  scripts/train_agentic_lora.py \
  --mode fr_sft_dpo \
  --experiment-dir experiments/qwen3_4b_flightrecorder \
  --output-dir experiments/qwen3_4b_flightrecorder/adapters \
  --run-name-prefix smoke \
  --dry-run
```

## Trace-Only Adapter

```bash
hf jobs uv run \
  --flavor a100-large \
  --timeout 6h \
  --secrets HF_TOKEN \
  scripts/train_agentic_lora.py \
  --mode trace_sft \
  --experiment-dir experiments/qwen3_4b_flightrecorder \
  --output-dir experiments/qwen3_4b_flightrecorder/adapters/trace_only \
  --hub-model-id zwright/qwen3-4b-hermes-trace-only-lora \
  --run-name-prefix trace-only \
  --push-to-hub
```

## Flight Recorder Adapter

```bash
hf jobs uv run \
  --flavor a100-large \
  --timeout 6h \
  --secrets HF_TOKEN \
  scripts/train_agentic_lora.py \
  --mode fr_sft_dpo \
  --experiment-dir experiments/qwen3_4b_flightrecorder \
  --output-dir experiments/qwen3_4b_flightrecorder/adapters/flightrecorder \
  --hub-model-id zwright/qwen3-4b-hermes-flightrecorder-lora \
  --run-name-prefix flightrecorder \
  --push-to-hub
```

## Required Post-Training Evaluation

After both adapters exist, run the same held-out scenarios against:

1. `Qwen/Qwen3-4B-Instruct-2507`
2. `zwright/qwen3-4b-hermes-trace-only-lora`
3. `zwright/qwen3-4b-hermes-flightrecorder-lora`

Then score every run with Flight Recorder and compare:

- pass rate;
- average score;
- critical failure counts;
- task-completion evidence;
- forbidden-action regressions;
- unsupported-claim regressions.

The goal is not satisfied until the Flight Recorder adapter beats both the
baseline and trace-only adapter on the held-out suite without new forbidden
action or unsupported-claim regressions.
