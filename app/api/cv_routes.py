import os
from datetime import datetime
from pathlib import Path
from uuid import uuid4
from typing import List, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi import status
from pydantic import BaseModel

from app.services.pdf_parser import extract_text_from_pdf
from app.services.cv_renderer import generate_pdf_from_text
from app.services.data_store import store
from app.services.llm_client import (
    generate_tailored_cv_text,
    analyze_cv_for_ats,
    resolve_gemini_model,
)

router = APIRouter()
DEFAULT_GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
configured_files_root = Path(os.getenv("FILES_ROOT", "files"))
if not configured_files_root.is_absolute():
    configured_files_root = PROJECT_ROOT / configured_files_root
FILES_ROOT = configured_files_root.resolve()


# ---------- Pydantic schemas ----------

class CVUploadResponse(BaseModel):
    cv_id: int
    job_requirement_id: int
    message: str


class TailorRequest(BaseModel):
    job_requirement_id: int
    model: str = DEFAULT_GEMINI_MODEL


class TailoredCVResponse(BaseModel):
    id: int
    cv_id: int
    job_requirement_id: int
    tailored_text: str
    tailored_pdf_url: str
    model_used: str


class CVDetail(BaseModel):
    id: int
    title: str
    original_pdf_url: str
    original_text: str


class TailoredSummary(BaseModel):
    id: int
    job_requirement_id: int
    tailored_pdf_url: str
    model_used: str


class CVWithTailored(BaseModel):
    cv: CVDetail
    tailored_versions: List[TailoredSummary]


class ATSAnalysisRequest(BaseModel):
    job_requirement_id: int
    model: str = DEFAULT_GEMINI_MODEL


class ATSInsights(BaseModel):
    ats_score: int
    summary: str
    issues: List[str]
    recommendations: List[str]


class ATSAnalysisResponse(BaseModel):
    insights: ATSInsights
    raw_report: str


class TailoredHistoryItem(BaseModel):
    id: int
    cv_id: int
    cv_title: str
    job_requirement_id: int
    job_title: str
    tailored_pdf_url: str
    model_used: str
    created_at: datetime


class ATSAnalysisHistoryItem(BaseModel):
    id: int
    cv_id: int
    cv_title: str
    job_requirement_id: int
    job_title: str
    model_used: str
    ats_score: int
    summary: str
    issues: List[str]
    recommendations: List[str]
    raw_report: str
    created_at: datetime


class HistoryResponse(BaseModel):
    tailored_cv_history: List[TailoredHistoryItem]
    ats_analysis_history: List[ATSAnalysisHistoryItem]


class TaskHistoryItem(BaseModel):
    task_type: str
    task_id: int
    cv_id: int
    cv_title: str
    job_requirement_id: int
    job_title: str
    model_used: str
    created_at: datetime
    tailored_pdf_url: Optional[str] = None
    ats_score: Optional[int] = None
    summary: Optional[str] = None


# ---------- Helpers ----------

ORIGINAL_DIR = FILES_ROOT / "original"
TAILORED_DIR = FILES_ROOT / "tailored"


def build_file_url(path: Path) -> str:
    # The StaticFiles mount in main.py will serve /files/<relative-path>
    candidate = Path(path)
    if candidate.is_absolute():
        try:
            relative = candidate.relative_to(FILES_ROOT)
        except ValueError:
            # Fallback to filename if path is outside current files root.
            relative = Path(candidate.name)
    else:
        # Handle legacy records that may have "files/" prefix.
        if candidate.parts and candidate.parts[0] == "files":
            relative = Path(*candidate.parts[1:])
        else:
            relative = candidate
    return f"/files/{relative.as_posix()}"


def _lookup_maps():
    snapshot = store.get_snapshot()
    cv_map = {int(c["id"]): c for c in snapshot.get("cv_documents", [])}
    job_map = {int(j["id"]): j for j in snapshot.get("job_requirements", [])}
    return cv_map, job_map


# ---------- Routes ----------

@router.post("/cv/upload", response_model=CVUploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_cv(
    job_title: str = Form(...),
    job_description: str = Form(...),
    cv_title: str = Form("My CV"),
    file: UploadFile = File(...),
):
    if file.content_type not in ("application/pdf", "application/x-pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF files are supported.",
        )

    ORIGINAL_DIR.mkdir(parents=True, exist_ok=True)

    filename = f"{uuid4()}.pdf"
    output_path = ORIGINAL_DIR / filename

    # Save the uploaded file
    contents = await file.read()
    with open(output_path, "wb") as f:
        f.write(contents)

    # Extract text from PDF
    try:
        extracted_text = extract_text_from_pdf(str(output_path))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error extracting text from PDF: {e}",
        )

    cv_doc, job_req = store.create_cv_and_job(
        cv_title=cv_title,
        original_pdf_path=str(output_path),
        original_text=extracted_text,
        job_title=job_title,
        job_description=job_description,
    )

    return CVUploadResponse(
        cv_id=int(cv_doc["id"]),
        job_requirement_id=int(job_req["id"]),
        message="CV and job requirement uploaded successfully.",
    )


@router.post("/cv/{cv_id}/tailor", response_model=TailoredCVResponse)
def tailor_cv(
    cv_id: int,
    payload: TailorRequest,
):
    cv_doc = store.get_cv(cv_id)
    if not cv_doc:
        raise HTTPException(status_code=404, detail="CV not found.")

    job_req = store.get_job_requirement(payload.job_requirement_id)
    if not job_req:
        raise HTTPException(status_code=404, detail="Job requirement not found.")

    resolved_model = resolve_gemini_model(payload.model)

    # Call Gemini to get tailored text
    tailored_text = generate_tailored_cv_text(
        model=resolved_model,
        job_description=job_req["description"],
        original_cv_text=cv_doc["original_text"],
    )

    # Generate tailored PDF
    TAILORED_DIR.mkdir(parents=True, exist_ok=True)
    tailored_filename = f"{uuid4()}.pdf"
    tailored_path = TAILORED_DIR / tailored_filename
    generate_pdf_from_text(tailored_text, str(tailored_path))

    tailored_record = store.create_tailored_cv(
        cv_id=int(cv_doc["id"]),
        job_requirement_id=int(job_req["id"]),
        tailored_text=tailored_text,
        tailored_pdf_path=str(tailored_path),
        model_used=resolved_model,
    )

    return TailoredCVResponse(
        id=int(tailored_record["id"]),
        cv_id=int(tailored_record["cv_id"]),
        job_requirement_id=int(tailored_record["job_requirement_id"]),
        tailored_text=tailored_record["tailored_text"],
        tailored_pdf_url=build_file_url(Path(tailored_record["tailored_pdf_path"])),
        model_used=tailored_record["model_used"],
    )


@router.post("/cv/{cv_id}/analyze", response_model=ATSAnalysisResponse)
def analyze_cv(
    cv_id: int,
    payload: ATSAnalysisRequest,
):
    cv_doc = store.get_cv(cv_id)
    if not cv_doc:
        raise HTTPException(status_code=404, detail="CV not found.")

    job_req = store.get_job_requirement(payload.job_requirement_id)
    if not job_req:
        raise HTTPException(status_code=404, detail="Job requirement not found.")

    resolved_model = resolve_gemini_model(payload.model)

    analysis = analyze_cv_for_ats(
        model=resolved_model,
        job_description=job_req["description"],
        original_cv_text=cv_doc["original_text"],
    )

    store.create_ats_analysis(
        cv_id=int(cv_doc["id"]),
        job_requirement_id=int(job_req["id"]),
        model_used=resolved_model,
        ats_score=int(analysis["ats_score"]),
        summary=str(analysis["summary"]),
        issues=[str(x) for x in analysis["issues"]],
        recommendations=[str(x) for x in analysis["recommendations"]],
        raw_report=str(analysis["raw_report"]),
    )

    return ATSAnalysisResponse(
        insights=ATSInsights(
            ats_score=analysis["ats_score"],
            summary=analysis["summary"],
            issues=analysis["issues"],
            recommendations=analysis["recommendations"],
        ),
        raw_report=analysis["raw_report"],
    )


@router.get("/cv/{cv_id}", response_model=CVWithTailored)
def get_cv_with_tailored(
    cv_id: int,
):
    cv_doc = store.get_cv(cv_id)
    if not cv_doc:
        raise HTTPException(status_code=404, detail="CV not found.")

    tailored = store.list_tailored_cvs(cv_id=cv_id)

    cv_detail = CVDetail(
        id=int(cv_doc["id"]),
        title=cv_doc["title"],
        original_pdf_url=build_file_url(Path(cv_doc["original_pdf_path"])),
        original_text=cv_doc["original_text"],
    )

    tailored_list = [
        TailoredSummary(
            id=int(t["id"]),
            job_requirement_id=int(t["job_requirement_id"]),
            tailored_pdf_url=build_file_url(Path(t["tailored_pdf_path"])),
            model_used=t["model_used"],
        )
        for t in tailored
    ]

    return CVWithTailored(cv=cv_detail, tailored_versions=tailored_list)


@router.get("/tailored/{tailored_id}", response_model=TailoredCVResponse)
def get_tailored_cv(
    tailored_id: int,
):
    record = store.get_tailored_cv(tailored_id)
    if not record:
        raise HTTPException(status_code=404, detail="Tailored CV not found.")

    return TailoredCVResponse(
        id=int(record["id"]),
        cv_id=int(record["cv_id"]),
        job_requirement_id=int(record["job_requirement_id"]),
        tailored_text=record["tailored_text"],
        tailored_pdf_url=build_file_url(Path(record["tailored_pdf_path"])),
        model_used=record["model_used"],
    )


@router.get("/history", response_model=HistoryResponse)
def get_history(
    cv_id: Optional[int] = None,
):
    tailored_records = store.list_tailored_cvs(cv_id=cv_id)
    ats_records = store.list_ats_analyses(cv_id=cv_id)
    cv_map, job_map = _lookup_maps()

    tailored_history = [
        TailoredHistoryItem(
            id=int(record["id"]),
            cv_id=int(record["cv_id"]),
            cv_title=(cv_map.get(int(record["cv_id"])) or {}).get("title", ""),
            job_requirement_id=int(record["job_requirement_id"]),
            job_title=(job_map.get(int(record["job_requirement_id"])) or {}).get("title", ""),
            tailored_pdf_url=build_file_url(Path(record["tailored_pdf_path"])),
            model_used=record["model_used"],
            created_at=record["created_at"],
        )
        for record in tailored_records
    ]

    ats_history = [
        ATSAnalysisHistoryItem(
            id=int(record["id"]),
            cv_id=int(record["cv_id"]),
            cv_title=(cv_map.get(int(record["cv_id"])) or {}).get("title", ""),
            job_requirement_id=int(record["job_requirement_id"]),
            job_title=(job_map.get(int(record["job_requirement_id"])) or {}).get("title", ""),
            model_used=record["model_used"],
            ats_score=int(record["ats_score"]),
            summary=record["summary"],
            issues=[str(x) for x in record.get("issues", [])],
            recommendations=[str(x) for x in record.get("recommendations", [])],
            raw_report=record["raw_report"],
            created_at=record["created_at"],
        )
        for record in ats_records
    ]

    return HistoryResponse(
        tailored_cv_history=tailored_history,
        ats_analysis_history=ats_history,
    )


@router.get("/tasks", response_model=List[TaskHistoryItem])
def get_all_tasks():
    tailored_records = store.list_tailored_cvs()
    ats_records = store.list_ats_analyses()
    cv_map, job_map = _lookup_maps()

    tasks: List[TaskHistoryItem] = []

    for record in tailored_records:
        tasks.append(
            TaskHistoryItem(
                task_type="cv_tailor",
                task_id=int(record["id"]),
                cv_id=int(record["cv_id"]),
                cv_title=(cv_map.get(int(record["cv_id"])) or {}).get("title", ""),
                job_requirement_id=int(record["job_requirement_id"]),
                job_title=(job_map.get(int(record["job_requirement_id"])) or {}).get("title", ""),
                model_used=record["model_used"],
                created_at=record["created_at"],
                tailored_pdf_url=build_file_url(Path(record["tailored_pdf_path"])),
            )
        )

    for record in ats_records:
        tasks.append(
            TaskHistoryItem(
                task_type="ats_analysis",
                task_id=int(record["id"]),
                cv_id=int(record["cv_id"]),
                cv_title=(cv_map.get(int(record["cv_id"])) or {}).get("title", ""),
                job_requirement_id=int(record["job_requirement_id"]),
                job_title=(job_map.get(int(record["job_requirement_id"])) or {}).get("title", ""),
                model_used=record["model_used"],
                created_at=record["created_at"],
                ats_score=int(record["ats_score"]),
                summary=record["summary"],
            )
        )

    tasks.sort(key=lambda item: item.created_at, reverse=True)
    return tasks
