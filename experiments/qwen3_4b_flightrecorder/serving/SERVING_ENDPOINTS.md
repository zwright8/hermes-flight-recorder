# Serving Endpoint Suite

- Passed: True
- Failed checks: none

## Arms

| Arm | Ready | Requested Model | Served Model | Endpoint | Tool Calls | Structured Outputs | Lifecycle | Failed Checks |
| --- | ---: | --- | --- | --- | --- | --- | --- | --- |
| baseline | True | `Qwen/Qwen3-4B-Instruct-2507` | `Qwen/Qwen3-4B-Instruct-2507` | http://127.0.0.1:18100/v1 | not_verified | supported | [lifecycle](/Users/zacharywright/Documents/GitHub/hermes_dev/hermes-flight-recorder/experiments/qwen3_4b_flightrecorder/serving/baseline/serving_lifecycle_run.json) | none |
| trace_only | True | `Qwen/Qwen3-4B-Instruct-2507` | `Qwen/Qwen3-4B-Instruct-2507+trace_sft_adapter` | http://127.0.0.1:18101/v1 | not_verified | supported | [lifecycle](/Users/zacharywright/Documents/GitHub/hermes_dev/hermes-flight-recorder/experiments/qwen3_4b_flightrecorder/serving/trace_only/serving_lifecycle_run.json) | none |
| flightrecorder | True | `Qwen/Qwen3-4B-Instruct-2507` | `Qwen/Qwen3-4B-Instruct-2507+fr_sft_dpo_adapter` | http://127.0.0.1:18102/v1 | not_verified | supported | [lifecycle](/Users/zacharywright/Documents/GitHub/hermes_dev/hermes-flight-recorder/experiments/qwen3_4b_flightrecorder/serving/flightrecorder/serving_lifecycle_run.json) | none |

## Profile Links

- `baseline`: [serving_profile](/Users/zacharywright/Documents/GitHub/hermes_dev/hermes-flight-recorder/experiments/qwen3_4b_flightrecorder/serving/baseline/serving_profile.json)
- `trace_only`: [serving_profile](/Users/zacharywright/Documents/GitHub/hermes_dev/hermes-flight-recorder/experiments/qwen3_4b_flightrecorder/serving/trace_only/serving_profile.json)
- `flightrecorder`: [serving_profile](/Users/zacharywright/Documents/GitHub/hermes_dev/hermes-flight-recorder/experiments/qwen3_4b_flightrecorder/serving/flightrecorder/serving_profile.json)
