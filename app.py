"""Hugging Face Space entrypoint.

Offline mock path only. Telemetry is disabled before Gradio imports so a
cold start does not call the Hub or Gradio analytics.
"""

from __future__ import annotations

import os

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import gradio as gr

from spectrace.space_view import (
    build_snapshot,
    default_trace_id,
    render_overview,
    render_trace,
    trace_choices,
)

_SNAPSHOT = build_snapshot()
_DEFAULT_TRACE = default_trace_id(_SNAPSHOT)

_CSS = """
.gradio-container {
  background:
    radial-gradient(1000px 360px at 8% -12%, #efe4cf 0%, transparent 55%),
    #f4efe6 !important;
  color: #1c1917 !important;
  color-scheme: light;
  max-width: 1120px !important;
  margin: 0 auto !important;
  padding: 18px 18px 28px !important;
}
.gradio-container .prose { max-width: none !important; }
#sp-overview, #sp-detail {
  border: 0 !important;
  background: transparent !important;
  box-shadow: none !important;
  padding: 0 !important;
}
#sp-picker {
  background: #fffdf8 !important;
  border: 1px solid #e4d9c8 !important;
  border-radius: 12px !important;
}
footer { display: none !important; }
.sp { color: #1c1917; font: 15px/1.45 ui-sans-serif, "Segoe UI", sans-serif; }
.sp-top {
  background: #1c1917;
  color: #f6f1e7;
  border-radius: 14px;
  padding: 16px 18px 14px;
}
.sp-top-row { display: flex; justify-content: space-between; gap: 16px; align-items: center; }
.sp-brand {
  margin: 0;
  font-weight: 750;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  font-size: 13px;
}
.sp-brand span {
  font-weight: 500;
  letter-spacing: 0;
  text-transform: none;
  color: #d6d3d1;
  margin-left: 8px;
}
.sp-repo {
  color: #1c1917;
  background: #f6f1e7;
  text-decoration: none;
  font-weight: 700;
  font-size: 13px;
  border-radius: 999px;
  padding: 6px 12px;
}
.sp-repo:hover { background: #fff; }
.sp-oneline { margin: 10px 0 0; font-size: 16px; max-width: 70ch; }
.sp-finding {
  margin: 16px 0 0;
  font-family: Georgia, Palatino, serif;
  font-size: 20px;
  line-height: 1.35;
}
.sp-cards {
  display: grid;
  grid-template-columns: repeat(5, 1fr);
  gap: 10px;
  margin-top: 14px;
}
.sp-card {
  background: #fffdf8;
  border: 1px solid #e4d9c8;
  border-top: 4px solid #1c1917;
  border-radius: 12px;
  padding: 10px 12px 12px;
}
.sp-lat { border-top-color: #1d4ed8; }
.sp-acc { border-top-color: #b45309; }
.sp-pre { border-top-color: #0f766e; }
.sp-cost { border-top-color: #57534e; }
.sp-jev { border-top-color: #c2410c; }
.sp-kicker {
  margin: 0;
  color: #57534e;
  font-size: 11px;
  font-weight: 750;
  letter-spacing: 0.06em;
  text-transform: uppercase;
}
.sp-stat {
  margin: 4px 0 0;
  font-family: Georgia, Palatino, serif;
  font-size: 30px;
  line-height: 1;
  font-variant-numeric: tabular-nums;
}
.sp-lat .sp-stat { color: #1d4ed8; }
.sp-acc .sp-stat { color: #b45309; }
.sp-pre .sp-stat { color: #0f766e; }
.sp-cost .sp-stat { color: #1c1917; }
.sp-jev .sp-stat { color: #c2410c; }
.sp-label { margin: 6px 0 0; font-weight: 700; font-size: 13.5px; }
.sp-sub { margin: 3px 0 0; color: #57534e; font-size: 12.5px; }
.sp-caveat { margin: 12px 0 0; color: #57534e; font-size: 13px; }
.sp-caveat strong { color: #9a3412; }
.sp-details { margin-top: 10px; }
.sp-details summary { cursor: pointer; font-weight: 700; }
.sp-trace { margin-top: 4px; }
.sp-trace-title { margin: 0; font-family: Georgia, Palatino, serif; font-size: 22px; }
.sp-prompt { margin: 4px 0 8px; color: #44403c; }
.sp-callout {
  background: #fff7ed;
  border: 1px solid #fdba74;
  border-radius: 12px;
  padding: 10px 12px;
  margin: 0 0 10px;
}
.sp-mins { display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; margin-bottom: 10px; }
.sp-mini {
  margin: 0;
  background: #fffdf8;
  border: 1px solid #e4d9c8;
  border-radius: 10px;
  padding: 8px 10px;
  font-variant-numeric: tabular-nums;
  font-size: 13.5px;
}
.sp-mini span {
  display: block;
  color: #57534e;
  font-size: 11px;
  font-weight: 750;
  letter-spacing: 0.05em;
  text-transform: uppercase;
  margin-bottom: 2px;
}
.sp-table {
  overflow-x: auto;
  border: 1px solid #e4d9c8;
  border-radius: 12px;
  background: #fffdf8;
}
.sp-table table { width: 100%; border-collapse: collapse; font-size: 13.5px; }
.sp-table th, .sp-table td {
  padding: 7px 9px;
  text-align: left;
  border-bottom: 1px solid #e4d9c8;
  vertical-align: top;
}
.sp-table thead th { background: #1c1917; color: #f6f1e7; font-size: 12px; font-weight: 650; }
.sp-table td.num, .sp-table th.num { text-align: right; font-variant-numeric: tabular-nums; }
.sp-table tr.sp-caught td { background: #fff7ed; }
.sp-table tr.sp-bad td { background: #fafaf9; }
.sp-table tr.sp-ok td { background: #f0fdfa; }
.pref { min-width: 120px; }
.pref-n { display: block; font-variant-numeric: tabular-nums; }
.bar {
  display: block;
  height: 7px;
  margin-top: 4px;
  background: #e4d9c8;
  border-radius: 99px;
  overflow: hidden;
}
.bar > span { display: block; height: 100%; background: #0f766e; }
.scores { color: #57534e; font-size: 12px; white-space: nowrap; }
.sp-note { color: #57534e; font-size: 12.5px; margin: 8px 0 0; }
.sp-missing { color: #9a3412; font-weight: 700; }
code { font-family: ui-monospace, Menlo, Consolas, monospace; font-size: 0.88em; }
@media (max-width: 980px) {
  .sp-cards { grid-template-columns: 1fr 1fr; }
  .sp-mins { grid-template-columns: 1fr; }
  .sp-finding { font-size: 18px; }
}
@media (max-width: 640px) {
  .sp-cards { grid-template-columns: 1fr; }
  .sp-top-row { flex-direction: column; align-items: flex-start; }
}
"""


def _show(trace_id: str | None) -> str:
    return render_trace(_SNAPSHOT, trace_id or _DEFAULT_TRACE)


def build_demo() -> gr.Blocks:
    theme = gr.themes.Base(
        primary_hue="stone",
        secondary_hue="teal",
        neutral_hue="stone",
        font=["ui-sans-serif", "system-ui", "sans-serif"],
        font_mono=["ui-monospace", "monospace"],
    )
    with gr.Blocks(
        title="spectrace — agent traces are not chat",
        analytics_enabled=False,
        theme=theme,
        css=_CSS,
    ) as demo:
        gr.HTML(render_overview(_SNAPSHOT), elem_id="sp-overview")
        picker = gr.Dropdown(
            label="Pick a trace",
            info="Per-step draft accept, prefill, and whether the verifier keeps the step.",
            choices=trace_choices(_SNAPSHOT),
            value=_DEFAULT_TRACE,
            filterable=True,
            elem_id="sp-picker",
        )
        detail = gr.HTML(render_trace(_SNAPSHOT, _DEFAULT_TRACE), elem_id="sp-detail")
        picker.change(_show, inputs=picker, outputs=detail)
    return demo


demo = build_demo()


def main() -> None:
    demo.launch(ssr_mode=False, show_api=False, enable_monitoring=False)


if __name__ == "__main__":
    main()
