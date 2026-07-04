# Eval Summary

- Status: blocked
- Governance ready: no
- Held-out scenarios: identical (3)
- Cross-arm claims allowed: yes
- Recommendation: Do not promote until the listed eval summary risks are resolved.

## Arms

| Arm | Scenarios | Passed | Failed | Serving | Blockers |
| --- | ---: | ---: | ---: | --- | --- |
| baseline | 3 | 3 | 0 | missing | none |
| candidate | 3 | 3 | 0 | missing | none |

## Comparisons

- No compare exports were provided.

## Compare Gates

- No compare gates were provided.

## Repair And Curriculum

- Work items: 3
- Critical work items: 0

| Priority | Category | Reason | Summary |
| --- | --- | --- | --- |
| medium | eval_harness | adapter_disabled_until_allow_installed | External adapter plan external is blocked by adapter_disabled_until_allow_installed. |
| medium | eval_harness | dependencies_missing | External adapter plan external is blocked by dependencies_missing. |
| medium | eval_harness | no_ready_external_adapters | External adapter plan external is blocked by no_ready_external_adapters. |

## Risks

| Source | Label | Reason |
| --- | --- | --- |
| external_adapter_plan | external | no_ready_external_adapters |
| external_adapter_plan | external | adapter_disabled_until_allow_installed |
| external_adapter_plan | external | dependencies_missing |

## Notes

- Raw movement is reported separately from approved governance claims.
- Candidate wins or task-completion improvements are approved only when the held-out scenario gate allows cross-arm claims.
