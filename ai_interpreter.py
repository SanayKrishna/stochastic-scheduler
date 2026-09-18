"""Calls Gemini to explain simulation results. Only module that touches the AI API."""

import json
import os
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

LOG_PATH = Path(__file__).parent / "logs" / "ai_debug.log"


def _log(prompt, raw):
    try:
        LOG_PATH.parent.mkdir(exist_ok=True)
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(f"\n===== {datetime.now().isoformat()} =====\n")
            f.write("--- PROMPT ---\n")
            f.write(prompt + "\n")
            f.write("--- RAW RESPONSE ---\n")
            f.write(str(raw) + "\n")
    except OSError:
        pass


def build_prompt(results):
    drivers = results.get("risk_drivers", [])
    lines = []
    for d in drivers:
        lines.append(
            f"- {d['id']} {d['name']}: O={d['optimistic']} M={d['most_likely']} "
            f"P={d['pessimistic']} corr={d['risk_contribution']:.3f}"
        )
    gap = results.get("p95", 0) - results.get("p50", 0)
    return f"""You are a project schedule risk analyst. Explain Monte Carlo results plainly.

Project: {results.get('project_name')}
Target: {results.get('target_completion_days')} days
P50: {results.get('p50'):.2f} days, P80: {results.get('p80'):.2f} days, P95: {results.get('p95'):.2f} days
Chance of hitting target: {results.get('prob_on_time')}
Schedule buffer needed (P95 - P50): {gap:.2f} days

Top risk drivers:
{chr(10).join(lines)}

Reply with JSON only, exactly this shape:
{{"summary": "2-3 sentences on overall risk",
"mitigations": [{{"task_id": "...", "task_name": "...", "suggestion": "one specific action tied to that task's O/M/P spread"}} for the top 3 drivers],
"recommended_buffer_days": <number to add to P50 for ~80% confidence>}}

Be specific. Tie each suggestion to the task's spread, not generic advice."""


def _extract_json(text):
    text = text.strip()
    if text.startswith("```"):
        # strip code fences if model adds them
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        text = text[start:end + 1]
    return json.loads(text)


def interpret_results(results):
    prompt = build_prompt(results)
    api_key = os.getenv("GEMINI_API_KEY", "")

    if not api_key:
        _log(prompt, "MISSING GEMINI_API_KEY")
        return {
            "available": False,
            "message": "AI interpretation unavailable (missing API key).",
            "summary": None,
            "mitigations": [],
            "recommended_buffer_days": None,
        }

    try:
        from google import genai

        client = genai.Client(api_key=api_key)
        resp = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
            config={"response_mime_type": "application/json"},
        )
        raw = getattr(resp, "text", str(resp))
        _log(prompt, raw)
        parsed = _extract_json(raw)
        return {
            "available": True,
            "message": None,
            "summary": str(parsed.get("summary", "")),
            "mitigations": parsed.get("mitigations", [])[:3],
            "recommended_buffer_days": parsed.get("recommended_buffer_days"),
        }
    except Exception as e:
        _log(prompt, f"ERROR: {e!r}")
        return {
            "available": False,
            "message": "AI interpretation unavailable. Showing simulation results only.",
            "summary": None,
            "mitigations": [],
            "recommended_buffer_days": None,
        }
