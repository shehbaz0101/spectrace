# Examples

```bash
# Install editable + pytest, then:
spectrace bakeoff --traces traces

# Same table via the example script:
python examples/bakeoff.py

# JSON + markdown files:
spectrace bakeoff --traces traces --output reports --format all

# Single method:
spectrace run --trace traces/code_fix.json --method mock_speculative --dry-run
```

Numbers are **fixture / simulated**. Plug in `examples/prices.example.json` only when you want a non-zero cost column:

```bash
spectrace bakeoff --traces traces --price-table examples/prices.example.json --model example-target
```
