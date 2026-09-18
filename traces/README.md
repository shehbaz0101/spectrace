# Agent traces

JSON fixtures for **multi-step tool-using agents**, not ShareGPT chat dumps.

Each file is one task. The schema is OpenAI-style `messages` plus:

| field | meaning |
| --- | --- |
| `id` | stable trace id |
| `task_type` | research_qa, code_fix, multihop_retrieval, constrained_planning, data_analysis |
| `tools` | function schemas the agent was allowed to call |
| `success_criteria.required_tools` | tools that must appear for structural success |
| `expected_success` | fixture ground truth for the mock replay |

Assistant turns may include `tool_calls` with JSON `arguments`. That mix of **free text and schema tokens** is why speculative-decoding accept rate on agents is not the ShareGPT number.

These traces are **synthetic**. Snippets are plausible, not live API results.
