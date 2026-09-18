"""Flask routes only. Simulation lives in simulation_engine, AI in ai_interpreter."""

import json
from datetime import datetime, timedelta

from flask import Flask, render_template, request

from ai_interpreter import interpret_results
from simulation_engine import run_simulation

app = Flask(__name__)


def _parse_input():
    """Return (data dict, error string). Accepts file upload or pasted text."""
    pasted = (request.form.get("json_text") or "").strip()
    f = request.files.get("file")

    raw = ""
    if f and f.filename:
        try:
            raw = f.read().decode("utf-8")
        except Exception:
            return None, "could not read uploaded file, make sure it is JSON text"
    elif pasted:
        raw = pasted
    else:
        return None, "upload a JSON file or paste JSON text"

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        return None, f"invalid JSON: {e.msg} (line {e.lineno})"

    sims_raw = (request.form.get("simulations") or "").strip()
    if sims_raw:
        try:
            data["simulations"] = int(sims_raw)
        except ValueError:
            return None, "simulation count must be a number"

    return data, None


def _add_days(start_str, days):
    if not start_str:
        return None
    try:
        base = datetime.strptime(start_str, "%Y-%m-%d")
        return (base + timedelta(days=round(days))).strftime("%Y-%m-%d")
    except Exception:
        return None


def build_chart(results):
    """Plotly histogram + S-curve. Kept out of the route so the handler stays thin."""
    import numpy as np
    import plotly.graph_objects as go
    from plotly.offline import plot

    comps = np.array(results["completions"])
    x_sorted = np.sort(comps)
    cum = np.arange(1, len(x_sorted) + 1) / len(x_sorted) * 100

    accent = "#0D7377"
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=comps, nbinsx=40, name="Outcomes",
        marker_color=accent, opacity=0.55,
        hovertemplate="days: %{x:.1f}<br>count: %{y}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=x_sorted, y=cum, name="Cumulative %", yaxis="y2",
        line=dict(color="#1a1a1a", width=2),
        hovertemplate="%{x:.1f} days: %{y:.0f}%<extra></extra>",
    ))

    target = results.get("target_completion_days")
    if target is not None:
        fig.add_vline(x=target, line_dash="dash", line_color="#b3261e",
                      annotation_text="target", annotation_position="top right")

    for label, key in (("P50", "p50"), ("P80", "p80"), ("P95", "p95")):
        fig.add_vline(x=results[key], line_dash="dot", line_color=accent, opacity=0.7,
                      annotation_text=f"{label} {results[key]:.1f}d",
                      annotation_position="top")

    fig.update_layout(
        margin=dict(l=40, r=40, t=30, b=40),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="-apple-system, Inter, Segoe UI, sans-serif", size=13, color="#1a1a1a"),
        legend=dict(orientation="h", y=1.08),
        xaxis_title="Completion (days)",
        yaxis_title="Runs",
        yaxis2=dict(title="Cumulative %", overlaying="y", side="right", range=[0, 100]),
        bargap=0.08,
    )
    return plot(fig, output_type="div", include_plotlyjs="cdn")


@app.get("/")
def index():
    with open("sample_input.json", encoding="utf-8") as f:
        example = f.read()
    return render_template("upload.html", error=None, example=example)


@app.post("/analyze")
def analyze():
    data, err = _parse_input()
    if err:
        with open("sample_input.json", encoding="utf-8") as f:
            example = f.read()
        return render_template("upload.html", error=err, example=example), 400
    try:
        results = run_simulation(data)
    except ValueError as e:
        with open("sample_input.json", encoding="utf-8") as f:
            example = f.read()
        return render_template("upload.html", error=str(e), example=example), 400
    except Exception:
        with open("sample_input.json", encoding="utf-8") as f:
            example = f.read()
        return render_template("upload.html", error="simulation failed, check your inputs", example=example), 500

    ai = interpret_results(results)
    chart_div = build_chart(results)

    # calendar dates for the strip
    start = results.get("project_start_date")
    dates = {k: _add_days(start, results[k]) for k in ("p50", "p80", "p95")}

    return render_template("results.html", results=results, ai=ai, chart_div=chart_div, dates=dates)


if __name__ == "__main__":
    app.run(debug=True)
