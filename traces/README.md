# Agent traces

JSON fixtures for **multi-step tool-using agents**, not ShareGPT chat dumps.

Each file is one task. The schema is OpenAI-style `messages` plus:

| field | meaning |
| --- | --- |
| `id` | stable trace id |
| `task_type` | research_qa, code_fix, multihop_retrieval, constrained_planning, data_analysis |
| `provenance` | `synthetic-but-realistic` on every checked-in fixture |
| `task_success` | gold label: did the trajectory complete the task? |
| `failure_mode` | taxonomy label when `task_success` is false; null on the happy path |
| `tools` | function schemas the agent was allowed to call |
| `success_criteria.required_tools` | tools that must appear for structural success |
| `success_criteria.answer_must_contain` | phrases the final answer must include, or the trace is `off_goal` |
| `success_criteria.required_arg_substrings` | `{tool, contains}` that must show up in that tool's arguments, or the trace is `wrong_source` |
| `expected_success` | kept in sync with `task_success` for older callers |
| `messages[].expected_ok` | gold label on each **assistant** step: should a step acceptor keep it? |

`structural_success()` does not trust the boolean alone. It returns false for an unknown tool, missing required arguments, a consecutive identical tool call, a tool error with no later successful tool result, `rm -rf` / `DROP TABLE` / `TRUNCATE`, a final answer that misses `answer_must_contain`, a read of the wrong document id, a final answer before required tools ran, or a multi-turn stall with no tool calls. On the checked-in files that result matches `task_success`.

Assistant turns may include `tool_calls` with JSON `arguments`. That mix of **free text and schema tokens** is why speculative-decoding accept rate on agents is not the ShareGPT number.

These traces are **synthetic-but-realistic**. Snippets are plausible, not live API results. Five happy-path tasks and ten failure-mode tasks. The failure ids and modes are listed in the repository README.
