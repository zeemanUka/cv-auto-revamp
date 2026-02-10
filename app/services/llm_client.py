import os
import json
import logging
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

GEMINI_BASE_URL = os.getenv("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com")
GEMINI_API_VERSION = os.getenv("GEMINI_API_VERSION", "v1beta")
GEMINI_DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
GEMINI_FALLBACK_MODEL = os.getenv("GEMINI_FALLBACK_MODEL", "gemini-1.5-flash")


def _get_gemini_api_key() -> Optional[str]:
    # Support either env var name.
    return os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")


def _normalize_model(model: str) -> str:
    candidate = (model or "").strip()
    if not candidate:
        candidate = GEMINI_DEFAULT_MODEL
    if candidate.startswith("models/"):
        candidate = candidate.split("models/", 1)[1]
    # Keep Gemini as the only provider target for this backend.
    if "gemini" not in candidate.lower():
        logger.warning(
            "Non-Gemini model '%s' requested; using default Gemini model '%s'.",
            candidate,
            GEMINI_DEFAULT_MODEL,
        )
        candidate = GEMINI_DEFAULT_MODEL
    return candidate


def resolve_gemini_model(model: str) -> str:
    """Public helper for routes that need to store the final model used."""
    return _normalize_model(model)


def _build_gemini_generate_content_url(model: str) -> str:
    base = GEMINI_BASE_URL.rstrip("/")
    for suffix in ("/v1beta", "/v1"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            break
    version = GEMINI_API_VERSION.strip("/")
    return f"{base}/{version}/models/{model}:generateContent"


def _ordered_candidate_models(requested_model: str) -> List[str]:
    # Always prefer the configured default model, then try the requested model,
    # then fallback model, removing duplicates while preserving order.
    ordered: List[str] = []
    for candidate in (GEMINI_DEFAULT_MODEL, requested_model, GEMINI_FALLBACK_MODEL):
        normalized = _normalize_model(candidate)
        if normalized not in ordered:
            ordered.append(normalized)
    return ordered


def _extract_first_candidate_text(payload: Dict[str, Any]) -> Optional[str]:
    candidates = payload.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        return None

    first_candidate = candidates[0] if isinstance(candidates[0], dict) else {}
    content = first_candidate.get("content") if isinstance(first_candidate, dict) else {}
    if not isinstance(content, dict):
        return None

    parts = content.get("parts")
    if not isinstance(parts, list):
        return None

    texts = []
    for part in parts:
        if isinstance(part, dict):
            text_value = part.get("text")
            if isinstance(text_value, str) and text_value.strip():
                texts.append(text_value.strip())

    if not texts:
        return None
    return "\n".join(texts)


def _call_gemini(
    model: str,
    system_prompt: str,
    user_prompt: str,
    timeout: int = 120,
) -> Optional[str]:
    api_key = _get_gemini_api_key()
    if not api_key:
        logger.error("Missing Gemini API key. Set GEMINI_API_KEY (or GOOGLE_API_KEY).")
        return None

    prompt = f"{system_prompt}\n\n{user_prompt}"
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": prompt}],
            }
        ]
    }

    candidate_models = _ordered_candidate_models(model)

    try:
        for idx, candidate_model in enumerate(candidate_models):
            url = _build_gemini_generate_content_url(candidate_model)
            response = requests.post(
                url,
                headers={
                    "x-goog-api-key": api_key,
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=timeout,
            )

            if response.status_code >= 400:
                has_more_candidates = idx < len(candidate_models) - 1
                should_try_next = response.status_code in {404, 429, 500, 503}
                if has_more_candidates and should_try_next:
                    logger.warning(
                        "Gemini request failed for model '%s' with status %s. Trying next model.",
                        candidate_model,
                        response.status_code,
                    )
                    continue

                logger.warning(
                    "Gemini request failed for model '%s' with status %s.",
                    candidate_model,
                    response.status_code,
                )
                response.raise_for_status()

            data = response.json()
            content = _extract_first_candidate_text(data)
            if content:
                return content

            logger.warning(
                "Gemini response missing candidate text for model '%s'.",
                candidate_model,
            )
            if idx < len(candidate_models) - 1:
                continue

        logger.error("Gemini request failed for all candidate models: %s", candidate_models)
        return None
    except Exception as e:
        logger.error(f"Error calling Gemini API: {e}")
        return None


def generate_tailored_cv_text(
    model: str,
    job_description: str,
    original_cv_text: str,
) -> str:
    """
    Calls Gemini to rewrite the CV text for the job description.
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

    content = _call_gemini(
        model=model,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        timeout=120,
    )
    if not content:
        logger.warning("Gemini response missing content. Falling back to original text.")
        return original_cv_text
    return content.strip()


def _extract_json_payload(content: str) -> Optional[Dict[str, Any]]:
    """Attempt to parse JSON output, handling optional code fences."""
    text = content.strip()
    if not text:
        return None
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
        # Fallback for model outputs that wrap JSON with extra text.
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = text[start : end + 1]
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                pass

        logger.warning("Failed to decode LLM JSON payload. Raw content preserved.")
        return None


def analyze_cv_for_ats(
    model: str,
    job_description: str,
    original_cv_text: str,
) -> Dict[str, Any]:
    """
    Calls Gemini to critique a CV, simulate an ATS scan, and suggest improvements.
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

    raw_content = _call_gemini(
        model=model,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        timeout=120,
    )
    if raw_content:
        raw_content = raw_content.strip()

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
