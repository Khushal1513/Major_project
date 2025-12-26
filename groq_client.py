"""
Minimal Groq API client helper.

Usage: set `GROQ_API_KEY` and `GROQ_MODEL` (optional) in your .env file.
This module exposes `generate(prompt, max_tokens, temperature, model)` which
will call the Groq HTTP inference endpoint. The exact response shape may vary
by Groq model; this helper attempts common patterns and returns the generated
text or raises an informative exception.
"""
from __future__ import annotations

import os
import time
import requests
from typing import Optional

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "groq-instruct")
GROQ_URL = os.getenv("GROQ_URL") or "https://api.groq.ai/v1/models"


class GroqError(RuntimeError):
    pass


def _default_headers():
    if not GROQ_API_KEY:
        raise GroqError("GROQ_API_KEY not set in environment")
    return {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }


def _parse_response_json(j: dict) -> Optional[str]:
    # Try several common response shapes
    if not isinstance(j, dict):
        return None

    # 1) top-level 'output' string
    if "output" in j and isinstance(j["output"], str):
        return j["output"]

    # 2) 'outputs' list with text/content fields
    outs = j.get("outputs") or j.get("results") or j.get("choices")
    if isinstance(outs, list) and len(outs) > 0:
        first = outs[0]
        # check common fields
        for key in ("text", "content", "message", "output"):
            if isinstance(first.get(key), str):
                return first.get(key)
        # sometimes items are dicts with nested 'content' list
        if isinstance(first.get("content"), list):
            parts = []
            for p in first.get("content"):
                if isinstance(p, dict) and isinstance(p.get("text"), str):
                    parts.append(p.get("text"))
            if parts:
                return "".join(parts)

    # 3) fallback: join any string values
    strings = []
    def _collect(obj):
        if isinstance(obj, str):
            strings.append(obj)
        elif isinstance(obj, dict):
            for v in obj.values():
                _collect(v)
        elif isinstance(obj, list):
            for v in obj:
                _collect(v)

    _collect(j)
    return "\n".join(strings) if strings else None


def generate(prompt: str, max_tokens: int = 512, temperature: float = 0.3, model: Optional[str] = None, timeout: int = 60) -> str:
    """Generate text from Groq.

    This is a best-effort generic client. Set `GROQ_MODEL` in .env or pass
    `model`. Returns generated text.
    """
    if not model:
        model = GROQ_MODEL

    if not model:
        raise GroqError("No model specified for Groq generate call")

    url = f"{GROQ_URL}/{model}/infer"
    headers = _default_headers()

    payload = {
        "input": prompt,
        "max_tokens": int(max_tokens),
        "temperature": float(temperature)
    }

    # Simple retry
    last_exc = None
    for attempt in range(3):
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=timeout)
            if resp.status_code >= 400:
                # include body for better diagnostics
                raise GroqError(f"Groq API error {resp.status_code}: {resp.text}")

            j = resp.json()
            out = _parse_response_json(j)
            if out is None:
                # fallback to raw text
                text = resp.text
                if text:
                    return text
                raise GroqError("Unable to parse Groq response JSON")
            return out

        except Exception as e:
            last_exc = e
            time.sleep(1 + attempt * 2)
            continue

    raise GroqError(f"Groq generate failed after retries: {last_exc}")
