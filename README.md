# spectrace

**Reproducible bake-off harness for speculative decoding and other serving tricks, measured on real multi-step agent traces — not ShareGPT chat.**

[![MIT license](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](pyproject.toml)
[![$0 default](https://img.shields.io/badge/default_cost-%240-green.svg)](#quickstart)

Student-friendly. **No API keys, GPUs, or paid providers required** to run the demo or `pytest`. Optional OpenAI-compatible `SPECTRACE_BASE_URL` is documented for later; secrets are never committed.

| You have | spectrace gives you |
| --- | --- |
| A recorded agent trajectory (tools, retries, growing context) | Wall latency, draft accept rate, tokens in/out |
| Two serving methods (`baseline` vs `mock_speculative`) | Same task, different decode simulation |
| A pluggable price table (default **$0**) | **Cost per successful task**, plus failure modes |

This repository is the public scaffold: fixture traces, a mock speculative decoder, metrics, CLI, and tests. Live vLLM/SGLang backends are a later milestone, not a gate to clone-and-run.

---

## Why agent traces are not chat

ShareGPT-style dumps are (mostly) single-turn or casual multi-turn **prose**. Serving papers report TTFT / TPOT / accept length on that mix because it is easy to replay.

Agents are a different workload:

1. **Context grows every tool round.** Prefill is charged again with the observation appended. Token-in totals explode relative to a one-shot chat completion with the same final answer.
2. **Many assistant turns are JSON tool calls**, not English. Draft models that look strong on chat often reject earlier on schema tokens, quotes, and argument keys.
3. **Success is a task, not a token.** A faster decode that corrupts a `check_constraints` payload can *raise* cost per successful task even when mean latency falls.
4. **Failure modes are structural** — missing tool, broken JSON, recovered test failure — not BLEU against a chat reference.

spectrace treats the trace as the unit of work: replay each assistant turn, accumulate prefix tokens, and score the serving method on the whole trajectory.

The mock decoder encodes that hypothesis directly: per-position draft accept probabilities are **lower for `tool_call` steps than for free text**. That is a simulator knob, not a GPU measurement; the harness exists so a later live backend can confirm or kill the claim.

---

## Quickstart

Requires Python 3.10+. No keys.

```bash
git clone https://github.com/shehbaz0101/spectrace.git
cd spectrace
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# validate fixtures without simulating decode
spectrace run --trace traces/code_fix.json --method baseline --dry-run

# the command that prints the bake-off table
spectrace bakeoff --traces traces
```

Equivalent:

```bash
python examples/bakeoff.py
python benchmarks/run.py
pytest          # zero API keys
```

Write JSON + Markdown reports:

```bash
spectrace bakeoff --traces traces --output reports --format all
```

Optional later (not used by tests):

```bash
export SPECTRACE_BASE_URL=http://127.0.0.1:11434/v1   # any OpenAI-compatible server
export SPECTRACE_MODEL=local-model
# export SPECTRACE_API_KEY=...   # if the server requires one; never commit it
```

Copy `.env.example` rather than inventing filenames for secrets.

---

## Sample results (fixtures, not GPUs)

**These numbers are produced by the offline mock decoder on the synthetic traces in `traces/`. They are not hardware measurements. Do not cite them as speculative-decoding speedups.**

Seed `0`, gamma `5`, price table all zeros. Copied from `spectrace bakeoff --traces traces --format md` on this scaffold (still simulated):

| trace | method | latency_ms | accept_rate | tokens_in | tokens_out | success | cost_usd |
| --- | --- | ---: | ---: | ---: | ---: | --- | ---: |
| code_fix_01 | baseline | 1176.0 | — | 1176 | 192 | yes | 0.0000 |
| code_fix_01 | mock_speculative | 766.8 | 0.441 | 1176 | 192 | yes | 0.0000 |
| data_analysis_01 | baseline | 1696.0 | — | 737 | 280 | yes | 0.0000 |
| data_analysis_01 | mock_speculative | 968.8 | 0.519 | 737 | 280 | yes | 0.0000 |
| multihop_retrieval_01 | baseline | 1312.0 | — | 801 | 216 | yes | 0.0000 |
| multihop_retrieval_01 | mock_speculative | 720.4 | 0.551 | 801 | 216 | yes | 0.0000 |
| research_qa_01 | baseline | 1936.0 | — | 818 | 320 | yes | 0.0000 |
| research_qa_01 | mock_speculative | 1024.0 | 0.571 | 818 | 320 | yes | 0.0000 |
| travel_planner_01 | baseline | 1798.0 | — | 768 | 297 | yes | 0.0000 |
| travel_planner_01 | mock_speculative | 1085.2 | 0.474 | 768 | 297 | yes | 0.0000 |

Cost per successful task: **$0.0000**. Tool-call-heavy traces (`code_fix_01`, `travel_planner_01`) show lower mock accept rates than the more prose-heavy research answer — that is the simulator doing its job, not a GPU finding.

Re-generate the fixture table (still simulated) with:

```bash
spectrace bakeoff --traces traces --format md
```

Cost per successful task is **$0.0000** on the default price table. Pass `--price-table examples/prices.example.json --model example-target` to see a non-zero column; those rates are examples, not a quote from a vendor.

---

## What gets measured

| Metric | Meaning |
| --- | --- |
| `wall_latency_ms` | Simulated decode time (target ms/token vs draft+verify rounds). No `sleep`. |
| `accept_rate` | `draft_tokens_accepted / draft_tokens_proposed` (`—` on baseline). |
| `tokens_in` | **Sum of prefix tokens at every assistant turn** (the agent prefill tax). |
| `tokens_out` | Approximate tokens in recorded assistant text + tool-call JSON. |
| `success` | Structural: required tools fired and a final answer exists. |
| `failure_mode` | `missing_required_tool`, `missing_final_answer`, `tool_error`, … |
| `cost_estimate_usd` | `(in·p_in + out·p_out + draft·p_draft) / 1e6`. Default prices are 0. |

**Cost per successful task** averages cost only over runs with `success=true`, so a method cannot win by aborting early.

Tokenizer: a deterministic ~4 characters / token stand-in. Swap later; do not mix tokenizer families inside one bake-off.

---

## Repository layout

```
spectrace/            # Python package (CLI, replay, metrics, mock decoder, Jev grader)
traces/               # 5 synthetic multi-step agent traces (JSON)
benchmarks/           # runner entry (`python benchmarks/run.py`)
examples/             # bake-off script + example price table
tests/                # pytest — CLI, metrics, fixtures, Jev mock + optional live skip
docs/jev.md           # TypeSafe AI Jev: System-1 step grader (not a draft model)
```

CLI:

```
spectrace run --trace <file-or-dir> --method baseline|mock_speculative
spectrace bakeoff --traces traces --methods baseline,mock_speculative
spectrace grade --traces traces --provider mock|jev
spectrace list-traces --traces traces
```

---

## Methods

**baseline** — sequential target decode. No drafts. `accept_rate` is empty.

**mock_speculative** — Leviathan-style window of `gamma` draft tokens, longest accepted prefix, plus one guaranteed target token per round. Acceptance is sampled from a per-position schedule that is harsher on `tool_call` JSON than on prose. Wall time is `rounds × (γ·draft_ms + verify_ms)`.

Neither method calls a live model on the default path. They *serve the recorded tokens* so the pipeline is demonstrable on a laptop.

---

## Synthetic traces

| id | task type | why it is not chat |
| --- | --- | --- |
| `research_qa_01` | web Q&A | search → fetch a primary table → compare years |
| `code_fix_01` | SWE-style repair | read, failing pytest, patch, passing pytest (recovered error) |
| `multihop_retrieval_01` | RAG | second retrieve is conditioned on the first observation |
| `travel_planner_01` | constraints | nested JSON arguments + explicit `check_constraints` |
| `data_analysis_01` | tools + brief | schema inspect → pandas snippet → artifact → PM paragraph |

All are synthetic. Snippets are plausible, not live API pulls.

---

## Optional: Jev (System-1 grader, not drafting)

[TypeSafe AI Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev) is a System One model: **noul / choice / score**, no string generation. The serving LLM (and `mock_speculative`) **drafts** tokens; Jev **accepts or rejects the step**. It is not a speculative-decoding draft model.

`spectrace grade` scores each assistant step and prints `token_accept` (mock speculative) vs `jev_accept` (Jev) vs `task_success`.

Offline, no key:

```bash
spectrace grade --provider mock --traces traces
spectrace grade --provider mock --traces traces --output reports --format all
```

Live Jev (not used by tests; never commit a key):

```bash
export TYPESAFE_API_KEY=...          # or TYPESAFE_KEY
spectrace grade --provider jev --traces traces
```

`--provider jev` without a key errors clearly. `--provider mock` always works.

`jev_accept` is deterministic: `tool_ok >= 0.6 and progress >= 0.5 and abort < 0.5`. Details: [`docs/jev.md`](docs/jev.md).

---

## Contributing

This is an early public scaffold. Useful follow-ups, in roughly this order:

1. Real OpenAI-compatible replay (`SPECTRACE_BASE_URL`) behind a flag, still defaulting to mock.
2. Import adapters for public agent logs (e.g. SWE-bench traces) with the same schema.
3. Plug-in to a local engine’s spec-decode accept stats instead of the mock sampler.
4. Price tables checked in as **examples only** — never keys.

Please:

- Keep `pytest` green with empty env (no `SPECTRACE_API_KEY`, no `TYPESAFE_API_KEY`).
- Mark any number that is not a real measurement as **fixture / simulated**.
- Do not add Opentrons, lab, or employer branding.
- Do not call paid X/Twitter APIs.

```bash
pip install -e ".[dev]"
pytest
```

Cite via [`CITATION.cff`](CITATION.cff). License: [MIT](LICENSE).
