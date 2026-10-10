# Examples

```bash
# Install editable + pytest, then:
spectrace bakeoff --traces traces

# Same table via the example script:
python examples/bakeoff.py

# JSON + markdown files:
spectrace bakeoff --traces traces --output reports --format all

# Grade each step with the local Jev mock (token_accept vs jev_accept):
spectrace grade --provider mock --traces traces

# Single method:
spectrace run --trace traces/code_fix.json --method mock_speculative --dry-run

# Optional live draft/serve (needs SPECTRACE_BASE_URL; Jev is still grade-only):
# spectrace run --trace traces/code_fix.json --method baseline --provider openai-compat
```

Numbers are **fixture / simulated**. Plug in `examples/prices.example.json` only when you want a non-zero cost column:

```bash
spectrace bakeoff --traces traces --price-table examples/prices.example.json --model example-target
```
