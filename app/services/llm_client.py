import os
import logging
from typing import Optional

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
