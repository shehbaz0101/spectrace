# Jev as a trajectory grader (optional)

[TypeSafe AI **Jev**](https://typesafe.ai/blog/introducing-system-one-models-and-jev) is a **System One** model: it returns calibrated structured decisions, not generated text.

Jev's public question types are:

- **noul** — yes/no as a probability in `[0, 1]`
- **choice** — one-of-N with a probability vector
- **score** — a position on a short ordered scale

There is no completion to parse and no draft tokens. That is the opposite of speculative decoding.

## What Jev is *not* (in spectrace)

| Concern | Speculative decoding | Jev |
| --- | --- | --- |
| Job | Speed up **token generation** on the target LLM | Make a **calibrated decision** about state |
| Output | Tokens (then verified) | noul / choice / score |
| Failure mode | Draft prefix rejected, JSON broken mid-argument | Low confidence / abstain-to-human |
| spectrace role | `--method mock_speculative` | Optional grader / step router |

Do not wire Jev as a draft model. A System One model cannot propose the next tool-call JSON.

## Where it *would* plug in later

1. **Trajectory grader** — after a replay, ask `task_complete` (noul), `trace_quality` (score).
2. **Step router** — mid-trace, ask `next_action` (choice: continue / stop_success / stop_fail) instead of hoping an LLM emits a well-formed plan token-by-token.
3. **Failure taxonomy** — noul questions per mode (`missing_required_tool`, `budget_violation`, …) with thresholds, not regex on generated critiques.

Sketch (never called from `pytest`):

```http
POST https://api.typesafe.ai/v1/systemone
{
  "model": "jev-latest",
  "state": { "trace_id": "code_fix_01", "success": true, "failure_mode": null },
  "questions": {
    "task_complete": { "type": "noul", "instructions": "Did the agent finish the user task?" }
  }
}
```

## Default path stays $0

`spectrace.jev.NullJev` answers those questions locally and deterministically.

- Tests **must not** set `TYPESAFE_API_KEY`.
- `LiveJev.evaluate()` refuses to run unless a future caller opts in.
- Serving bake-offs (`baseline` vs `mock_speculative`) do not import Jev.

See `spectrace/jev.py`.
