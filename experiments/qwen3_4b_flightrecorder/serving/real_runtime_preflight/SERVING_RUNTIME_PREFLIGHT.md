# Serving Runtime Preflight

- Passed: True
- Readiness: `ready`
- Model: `Qwen/Qwen3-4B-Instruct-2507`
- Runtime Python: `/Users/zacharywright/Documents/GitHub/hermes_dev/hermes-flight-recorder/.venv/bin/python` (True)
- Model cache: `/Users/zacharywright/.cache/huggingface/hub/models--Qwen--Qwen3-4B-Instruct-2507` (True)
- Blocked checks: none

## Dependencies

| Dependency | Available | Origin |
| --- | ---: | --- |
| `torch` | True | `/Users/zacharywright/Documents/GitHub/hermes_dev/hermes-flight-recorder/.venv/lib/python3.11/site-packages/torch/__init__.py` |
| `transformers` | True | `/Users/zacharywright/Documents/GitHub/hermes_dev/hermes-flight-recorder/.venv/lib/python3.11/site-packages/transformers/__init__.py` |
| `peft` | True | `/Users/zacharywright/Documents/GitHub/hermes_dev/hermes-flight-recorder/.venv/lib/python3.11/site-packages/peft/__init__.py` |
| `accelerate` | True | `/Users/zacharywright/Documents/GitHub/hermes_dev/hermes-flight-recorder/.venv/lib/python3.11/site-packages/accelerate/__init__.py` |

## Adapters

| Arm | Exists | Adapter Config | Adapter Model | Path |
| --- | ---: | ---: | ---: | --- |
| trace_only | True | True | True | `/Users/zacharywright/Documents/GitHub/hermes_dev/hermes-flight-recorder/experiments/qwen3_4b_flightrecorder/adapters/qwen3_4b_local_trace_sft/trace_sft_adapter` |
| flightrecorder | True | True | True | `/Users/zacharywright/Documents/GitHub/hermes_dev/hermes-flight-recorder/experiments/qwen3_4b_flightrecorder/adapters/qwen3_4b_local_fr_sft_dpo/fr_sft_dpo_adapter` |

## Command Refs

- `baseline` server: `/Users/zacharywright/Documents/GitHub/hermes_dev/hermes-flight-recorder/.venv/bin/python /Users/zacharywright/Documents/GitHub/hermes_dev/hermes-flight-recorder/scripts/serve_transformers_openai.py --model Qwen/Qwen3-4B-Instruct-2507 --host 127.0.0.1 --port '{port}' --max-new-tokens 64`
- `baseline` managed eval: `/Users/zacharywright/Documents/GitHub/hermes_dev/hermes-flight-recorder/.venv/bin/python scripts/run_managed_serving_eval.py --server-command '/Users/zacharywright/Documents/GitHub/hermes_dev/hermes-flight-recorder/.venv/bin/python /Users/zacharywright/Documents/GitHub/hermes_dev/hermes-flight-recorder/scripts/serve_transformers_openai.py --model Qwen/Qwen3-4B-Instruct-2507 --host 127.0.0.1 --port '"'"'{port}'"'"' --max-new-tokens 64' --base-url 'http://127.0.0.1:{port}/v1' --model Qwen/Qwen3-4B-Instruct-2507 --arm baseline --out experiments/qwen3_4b_flightrecorder/serving/baseline --require-structured-output --eval-command '/Users/zacharywright/Documents/GitHub/hermes_dev/hermes-flight-recorder/.venv/bin/python scripts/evaluate_hermes_heldout.py --arm baseline --model Qwen/Qwen3-4B-Instruct-2507 --base-url {base_url} --serving-profile {serving_profile} --out experiments/qwen3_4b_flightrecorder/evaluations/baseline --force'`
- `trace_only` server: `/Users/zacharywright/Documents/GitHub/hermes_dev/hermes-flight-recorder/.venv/bin/python /Users/zacharywright/Documents/GitHub/hermes_dev/hermes-flight-recorder/scripts/serve_transformers_openai.py --model Qwen/Qwen3-4B-Instruct-2507 --host 127.0.0.1 --port '{port}' --max-new-tokens 64 --adapter /Users/zacharywright/Documents/GitHub/hermes_dev/hermes-flight-recorder/experiments/qwen3_4b_flightrecorder/adapters/qwen3_4b_local_trace_sft/trace_sft_adapter`
- `trace_only` managed eval: `/Users/zacharywright/Documents/GitHub/hermes_dev/hermes-flight-recorder/.venv/bin/python scripts/run_managed_serving_eval.py --server-command '/Users/zacharywright/Documents/GitHub/hermes_dev/hermes-flight-recorder/.venv/bin/python /Users/zacharywright/Documents/GitHub/hermes_dev/hermes-flight-recorder/scripts/serve_transformers_openai.py --model Qwen/Qwen3-4B-Instruct-2507 --host 127.0.0.1 --port '"'"'{port}'"'"' --max-new-tokens 64 --adapter /Users/zacharywright/Documents/GitHub/hermes_dev/hermes-flight-recorder/experiments/qwen3_4b_flightrecorder/adapters/qwen3_4b_local_trace_sft/trace_sft_adapter' --base-url 'http://127.0.0.1:{port}/v1' --model Qwen/Qwen3-4B-Instruct-2507 --arm trace_only --out experiments/qwen3_4b_flightrecorder/serving/trace_only --require-structured-output --adapter /Users/zacharywright/Documents/GitHub/hermes_dev/hermes-flight-recorder/experiments/qwen3_4b_flightrecorder/adapters/qwen3_4b_local_trace_sft/trace_sft_adapter --eval-command '/Users/zacharywright/Documents/GitHub/hermes_dev/hermes-flight-recorder/.venv/bin/python scripts/evaluate_hermes_heldout.py --arm trace_only --model Qwen/Qwen3-4B-Instruct-2507 --base-url {base_url} --serving-profile {serving_profile} --out experiments/qwen3_4b_flightrecorder/evaluations/trace_only --force'`
- `flightrecorder` server: `/Users/zacharywright/Documents/GitHub/hermes_dev/hermes-flight-recorder/.venv/bin/python /Users/zacharywright/Documents/GitHub/hermes_dev/hermes-flight-recorder/scripts/serve_transformers_openai.py --model Qwen/Qwen3-4B-Instruct-2507 --host 127.0.0.1 --port '{port}' --max-new-tokens 64 --adapter /Users/zacharywright/Documents/GitHub/hermes_dev/hermes-flight-recorder/experiments/qwen3_4b_flightrecorder/adapters/qwen3_4b_local_fr_sft_dpo/fr_sft_dpo_adapter`
- `flightrecorder` managed eval: `/Users/zacharywright/Documents/GitHub/hermes_dev/hermes-flight-recorder/.venv/bin/python scripts/run_managed_serving_eval.py --server-command '/Users/zacharywright/Documents/GitHub/hermes_dev/hermes-flight-recorder/.venv/bin/python /Users/zacharywright/Documents/GitHub/hermes_dev/hermes-flight-recorder/scripts/serve_transformers_openai.py --model Qwen/Qwen3-4B-Instruct-2507 --host 127.0.0.1 --port '"'"'{port}'"'"' --max-new-tokens 64 --adapter /Users/zacharywright/Documents/GitHub/hermes_dev/hermes-flight-recorder/experiments/qwen3_4b_flightrecorder/adapters/qwen3_4b_local_fr_sft_dpo/fr_sft_dpo_adapter' --base-url 'http://127.0.0.1:{port}/v1' --model Qwen/Qwen3-4B-Instruct-2507 --arm flightrecorder --out experiments/qwen3_4b_flightrecorder/serving/flightrecorder --require-structured-output --adapter /Users/zacharywright/Documents/GitHub/hermes_dev/hermes-flight-recorder/experiments/qwen3_4b_flightrecorder/adapters/qwen3_4b_local_fr_sft_dpo/fr_sft_dpo_adapter --eval-command '/Users/zacharywright/Documents/GitHub/hermes_dev/hermes-flight-recorder/.venv/bin/python scripts/evaluate_hermes_heldout.py --arm flightrecorder --model Qwen/Qwen3-4B-Instruct-2507 --base-url {base_url} --serving-profile {serving_profile} --out experiments/qwen3_4b_flightrecorder/evaluations/flightrecorder --force'`
