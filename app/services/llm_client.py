import os
import json
import logging
from typing import Any, Dict, Optional

import requests

logger = logging.getLogger(__name__)

# For example, if you use Ollama, default URL is something like:
# http://localhost:11434/api/chat
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:11434/api/chat")


def generate_tailored_cv_text(
    model: str,
    job_description: str,
    original_cv_text: str,
) -> str:
    """
    Calls a local LLM server to rewrite the CV text for the job description.

    This is written for an Ollama-style /api/chat endpoint.
    If you use something else, adjust the payload accordingly.
    """

    system_prompt = (
        "You are an assistant that rewrites CVs. "
        "You will be given a job description and a CV text. "
        "Your job is to rewrite ONLY the CV content to better match the job description, "
        "while keeping the overall structure (headings, role names, company names, and dates) as similar as possible.\n\n"
        "Return ONLY the rewritten CV text. Do not add explanations."
    )

    user_prompt = (
        f"JOB DESCRIPTION:\n{job_description}\n\n"
        f"ORIGINAL CV TEXT:\n{original_cv_text}\n\n"
        "Now return the tailored CV text."
    )

    payload = {
        "model": model,
        "stream": False,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }

    try:
        response = requests.post(LLM_BASE_URL, json=payload, timeout=120)
        response.raise_for_status()
        data = response.json()
        # This assumes Ollama-like response structure:
        # {"message": {"content": "..."}}
        message = data.get("message") or {}
        content: Optional[str] = message.get("content")

        if not content:
            logger.warning("LLM response missing 'message.content'. Falling back to original text.")
            return original_cv_text

        return content.strip()
    except Exception as e:
        logger.error(f"Error calling LLM server: {e}")
        # Fallback: return original text so app still works
        return original_cv_text


def _extract_json_payload(content: str) -> Optional[Dict[str, Any]]:
    """Attempt to parse JSON output, handling optional code fences."""
    text = content.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines:
            # Drop opening fence
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        logger.warning("Failed to decode LLM JSON payload. Raw content preserved.")
        return None


def analyze_cv_for_ats(
    model: str,
    job_description: str,
    original_cv_text: str,
) -> Dict[str, Any]:
    """
    Calls the LLM to critique a CV, simulate an ATS scan, and suggest improvements.
    Returns both parsed fields and the raw model output for transparency.
    """

    system_prompt = (
        "You are an expert resume reviewer and ATS (Applicant Tracking System) consultant. "
        "Provide an ATS readiness assessment, highlight issues, and propose concrete improvements."
    )

    user_prompt = (
        "Analyze the following CV against the given job description. "
        "Respond in strict JSON with the keys: "
        "'ats_score' (0-100 integer), 'summary' (string), 'issues' (list of strings), "
        "'recommendations' (list of strings).\n\n"
        f"JOB DESCRIPTION:\n{job_description}\n\n"
        f"CV TEXT:\n{original_cv_text}\n\n"
        "Return ONLY the JSON object without additional commentary."
    )

    payload = {
        "model": model,
        "stream": False,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }

    raw_content: Optional[str] = None
    try:
        response = requests.post(LLM_BASE_URL, json=payload, timeout=120)
        response.raise_for_status()
        data = response.json()
        message = data.get("message") or {}
        raw_content = (message.get("content") or "").strip()
    except Exception as e:
        logger.error(f"Error calling LLM for ATS analysis: {e}")

    parsed_payload = _extract_json_payload(raw_content or "") or {}
    try:
        ats_score = int(parsed_payload.get("ats_score", 60))
    except (TypeError, ValueError):
        ats_score = 60

    summary = parsed_payload.get("summary") or "Unable to generate a detailed summary."

    issues = parsed_payload.get("issues")
    if isinstance(issues, list):
        issues_list = [str(item) for item in issues if item]
    elif issues:
        issues_list = [str(issues)]
    else:
        issues_list = []

    recommendations = parsed_payload.get("recommendations")
    if isinstance(recommendations, list):
        recs_list = [str(item) for item in recommendations if item]
    elif recommendations:
        recs_list = [str(recommendations)]
    else:
        recs_list = []

    result = {
        "ats_score": max(0, min(100, ats_score)),
        "summary": summary,
        "issues": issues_list,
        "recommendations": recs_list,
        "raw_report": raw_content or "No analysis available.",
    }
    return result
