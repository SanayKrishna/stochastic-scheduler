"""Calls Gemini to explain simulation results. Only module that touches the AI API."""

import json
import os
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

LOG_PATH = Path(__file__).parent / "logs" / "ai_debug.log"

PLACEHOLDERS = {"", "your_gemini_key_here", "changeme", "xxx", "none", "null"}


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


def get_api_key():
    """Read the key from the environment. Accepts GEMINI_API_KEY or GOOGLE_API_KEY.

    Returns "" when no usable key is configured (missing .env, blank value,
    or the placeholder from .env.example left unfilled).
    """
    for var in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        raw = os.getenv(var, "").strip()
        if raw.lower().startswith("export "):
            raw = raw[7:].strip()
        key = raw.strip("\"'")
        if key.lower() not in PLACEHOLDERS:
            return key
    return ""


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


def _unavailable(reason, message):
    return {
        "available": False,
        "reason": reason,
        "message": message,
        "summary": None,
        "mitigations": [],
        "recommended_buffer_days": None,
    }


PRIMARY_MODEL = "gemini-3.5-flash-lite"


def _pick_fallback_model(client):
    """Ask the API which models exist and prefer a flash one.

    Model names retire over time, so on a 404 we look up what is actually
    available instead of guessing another hardcoded name.
    """
    names = []
    for m in client.models.list():
        actions = getattr(m, "supported_actions", None) or []
        if "generateContent" not in actions:
            continue
        name = getattr(m, "name", "")
        if name.startswith("models/"):
            name = name[len("models/"):]
        names.append(name)
    for n in names:
        if "flash" in n and "image" not in n and "tts" not in n:
            return n
    return names[0] if names else None


def _suggested_model(err_text):
    """Pull the replacement model out of a retirement error.

    The API names it directly, e.g. "update your code to use
    models/gemini-3.6-flash". Listing lies (retired names still appear),
    so the message is the trustworthy source.
    """
    import re
    found = re.findall(r"models/([\w\-.]+)", err_text)
    for name in found:
        if name != PRIMARY_MODEL:
            return name
    return None


def _generate(client, prompt, model):
    resp = client.models.generate_content(
        model=model,
        contents=prompt,
        config={"response_mime_type": "application/json"},
    )
    return getattr(resp, "text", str(resp))


def interpret_results(results):
    prompt = build_prompt(results)
    api_key = get_api_key()

    if not api_key:
        _log(prompt, "MISSING GEMINI_API_KEY")
        return _unavailable(
            "missing",
            "No Gemini API key found. Copy .env.example to .env and set "
            "GEMINI_API_KEY to a key from Google AI Studio.",
        )

    try:
        from google import genai

        client = genai.Client(api_key=api_key)
        try:
            raw = _generate(client, prompt, PRIMARY_MODEL)
        except Exception as first_e:
            if "404" not in repr(first_e) and "not_found" not in repr(first_e).lower():
                raise
            tried = {PRIMARY_MODEL}
            candidates = []
            suggested = _suggested_model(repr(first_e))
            if suggested:
                candidates.append(suggested)
            try:
                listed = _pick_fallback_model(client)
                if listed:
                    candidates.append(listed)
            except Exception:
                pass
            raw = None
            last_e = first_e
            for cand in candidates:
                if cand in tried:
                    continue
                tried.add(cand)
                try:
                    _log(prompt, f"MODEL {PRIMARY_MODEL} RETIRED, RETRYING WITH {cand}")
                    raw = _generate(client, prompt, cand)
                    break
                except Exception as next_e:
                    last_e = next_e
                    _log(prompt, f"RETRY WITH {cand} FAILED: {next_e!r}")
                    newer = _suggested_model(repr(next_e))
                    if newer and newer not in tried and newer not in candidates:
                        candidates.append(newer)
            if raw is None:
                raise last_e
        _log(prompt, raw)
    except Exception as e:
        _log(prompt, f"ERROR: {e!r}")
        text = repr(e).lower()
        if any(s in text for s in ("401", "403", "api_key_invalid", "invalid api key",
                                   "permission_denied", "unauthenticated")):
            return _unavailable(
                "auth",
                "The Gemini API key was rejected. Check that GEMINI_API_KEY in "
                ".env is a valid key from Google AI Studio.",
            )
        return _unavailable(
            "error",
            "AI interpretation unavailable (API call failed). "
            "Showing simulation results only.",
        )

    try:
        parsed = _extract_json(raw)
        return {
            "available": True,
            "reason": None,
            "message": None,
            "summary": str(parsed.get("summary", "")),
            "mitigations": parsed.get("mitigations", [])[:3],
            "recommended_buffer_days": parsed.get("recommended_buffer_days"),
        }
    except Exception as e:
        _log(prompt, f"PARSE ERROR: {e!r} / RAW WAS: {raw}")
        return _unavailable(
            "parse",
            "AI interpretation unavailable (unparseable response). "
            "Showing simulation results only.",
        )
