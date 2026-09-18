# Benchmark runners
#
# Default:
#   python benchmarks/run.py
#   python benchmarks/run.py --traces traces --output reports
#
# This is a thin wrapper around `spectrace bakeoff`. Live GPU / vLLM / SGLang
# backends are intentionally not wired yet — the mock decoder is the $0 path.
