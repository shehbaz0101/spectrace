# Jev as a System-1 trajectory / step grader

[TypeSafe AI **Jev**](https://typesafe.ai/blog/introducing-system-one-models-and-jev) is a **System One** model: calibrated structured decisions, not generated text.

The dual-brain picture for spectrace:

| Brain | Who | Job on an agent trace |
| --- | --- | --- |
| **System 2 (draft)** | The serving LLM + optional speculative decoder | Propose the next assistant tokens / tool-call JSON |
| **System 1 (accept)** | Jev | Judge whether that *step* is valid and worth keeping |

Speculative decoding accepts or rejects **draft tokens**. Jev accepts or rejects the **step** (tool call or assistant message) against the goal and history. Those are different layers. A high `token_accept` on a looping `search` call can still be a `jev_accept=no`.

`spectrace grade` scores every assistant step with both signals and compares them to structural `task_success`. A tweetable snapshot of one LIVE Jev run (mock `token_accept`, real nouls) lives in [`docs/dual-brain-accept.png`](dual-brain-accept.png).

## What Jev is *not*

| Concern | Speculative decoding | Jev |
| --- | --- | --- |
| Job | Speed up **token generation** on the target LLM | Make a **calibrated decision** about state |
| Output | Tokens (then verified) | noul / choice / score |
| Failure mode | Draft prefix rejected, JSON broken mid-argument | Low noul / abort / escalate-to-human |
| spectrace role | `--method mock_speculative` (`token_accept`) or `--provider openai-compat` (live draft/serve) | `spectrace grade` (`jev_accept`) |

Do not wire Jev as a draft model. A System One model cannot propose the next tool-call JSON. Live token generation is the OpenAI-compatible path (`--provider openai-compat` + `SPECTRACE_BASE_URL`); Jev still only accepts or rejects the step.

## Per-step questions

State sent to Jev:

```json
{
  "goal": "<trace prompt>",
  "history_so_far": [ { "role": "...", "content": "...", "tool_calls": [...] } ],
  "current_step": {
    "assistant": { "role": "assistant", "content": "...", "tool_calls": [...] },
    "tool_results": [ { "role": "tool", "content": "...", "error": false } ]
  },
  "tool_schemas": [ { "type": "function", "function": { "name": "..." } } ]
}
```

Questions (one System One call per step):

| name | type | Asks |
| --- | --- | --- |
| `tool_ok` | noul | Is this tool call / step schema-valid and appropriate? |
| `progress` | noul | Does this step move toward the goal? |
| `abort` | noul | Is this a wasteful loop / should we stop? |
| `escalate` | noul | Should a human review before continuing? |
| `disposition` | choice | `continue` / `accept` / `reject` / `escalate` |

`escalate` and `disposition` are **advisory**. They are stored on the JSON report but they do not change `jev_accept`.

## `jev_accept` rule

A step is accepted iff all three core nouls parse and:

```
tool_ok >= 0.6  AND  progress >= 0.5  AND  abort < 0.5
```

Missing or unparseable nouls **fail closed** (`jev_accept=false`). Constants live in `spectrace.jev` (`TOOL_OK_MIN`, `PROGRESS_MIN`, `ABORT_MAX`) and are printed on every grade report as `jev_accept_rule`.

`token_accept` is the mock speculative decoder's `draft_tokens_accepted / draft_tokens_proposed` for that assistant turn (same seed/gamma as `spectrace bakeoff`). It is **fixture / simulated**, not a GPU measurement.

`task_success` is the trace-level structural check (required tools + final answer). It is copied onto every step row so you can see step-level Jev vs whole-task outcome.

## Live API

```http
POST https://api.typesafe.ai/v1/systemone
Authorization: Bearer $TYPESAFE_API_KEY
Content-Type: application/json

{
  "model": "jev-latest",
  "state": { "...": "see above" },
  "questions": {
    "tool_ok": { "type": "noul", "instructions": "...", "criteria": { "true": "...", "false": "..." } }
  }
}
```

Auth also accepts the alias `TYPESAFE_KEY`. Never commit a key. The client retries `408/429/5xx/529` with exponential backoff and parses nouls defensively (clamp to `[0,1]`, ignore NaN / junk).

Override URL / model with `SPECTRACE_JEV_URL` and `SPECTRACE_JEV_MODEL` if needed (see `.env.example`).

## CLI

Offline (CI, no key):

```bash
spectrace grade --provider mock --traces traces
spectrace grade --provider mock --traces traces --output reports --format all
```

Live Jev (requires a key in the environment; not used by `pytest`):

```bash
export TYPESAFE_API_KEY=...    # never commit this
spectrace grade --provider jev --traces traces
```

`--provider jev` without a key exits 2 with a clear error. `--provider mock` always works.

## Default path stays $0

- `MockJev` answers the step questions locally with deterministic heuristics (schema names, required args, exact-repeat loops). No HTTP.
- `LiveJev.evaluate()` refuses to run without `TYPESAFE_API_KEY` / `TYPESAFE_KEY`.
- Serving bake-offs (`baseline` vs `mock_speculative`) do not call Jev.
- `--provider openai-compat` is System-2 draft/serve, not a Jev provider.
- `@pytest.mark.integration` live tests skip unless a key / `SPECTRACE_BASE_URL` is set.

See `spectrace/jev.py` and `spectrace/grade.py`.
