# Benchmark runners
#
# Default:
#   python benchmarks/run.py
#   python benchmarks/run.py --traces traces --output reports
#
# This is a thin wrapper around `spectrace bakeoff`. The default is the $0
# mock decoder. Optional live OpenAI-compat draft/serve is
# `spectrace run|bakeoff --provider openai-compat` (requires SPECTRACE_BASE_URL).
