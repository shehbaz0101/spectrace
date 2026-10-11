# Changelog

## 0.1.0 — October 2026

First release.

- Python package (`spectrace` 0.1.0, Python 3.10+) with no required third-party dependencies. CLI commands: `list-traces`, `run`, `bakeoff`, `grade`, and `demo`.
- Offline mock path compares `baseline` with simulated speculative decoding and grades each step with MockJev. The default price table is zeros. No API key and no network call.
- Optional live providers, off by default: TypeSafe Jev for step grades, and an OpenAI-compatible client that times a chat completion. CI does not call them.
- 15 synthetic traces (49 steps): 5 happy paths and 10 failures, with gold labels. Reports cover latency, draft accept rate, prefill growth, cost per successful task, and whether a step should have been kept.
- On those 15 synthetic traces, a step-level verifier rejected 11/11 harmful steps that token acceptance >= 0.5 would keep. Tool-call drafts accept 0.327 versus 0.694 for chat prose. Draft acceptance and latency are simulated. The 11/11 figure is the step grader on this set, not a production agent eval.
- Static one-file HTML demo, plus a Gradio page on Render's free web service: [https://spectrace-f6lk.onrender.com](https://spectrace-f6lk.onrender.com).
- Committed LIVE Jev grade files for the five happy traces (2026-09-20) and all 15 traces (2026-10-10).
