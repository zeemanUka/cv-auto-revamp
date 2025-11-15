from pathlib import Path
from uuid import uuid4
from typing import List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi import status
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.db import models
from app.services.pdf_parser import extract_text_from_pdf
from app.services.cv_renderer import generate_pdf_from_text
from app.services.llm_client import generate_tailored_cv_text, analyze_cv_for_ats

router = APIRouter()


# ---------- Pydantic schemas ----------

class CVUploadResponse(BaseModel):
    cv_id: int
    job_requirement_id: int
    message: str


class TailorRequest(BaseModel):
    job_requirement_id: int
    model: str = "llama3.2"  # adjust to match your local model name


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

    class Config:
        orm_mode = True


class TailoredSummary(BaseModel):
    id: int
    job_requirement_id: int
    tailored_pdf_url: str
    model_used: str

    class Config:
        orm_mode = True


class CVWithTailored(BaseModel):
    cv: CVDetail
    tailored_versions: List[TailoredSummary]


class ATSAnalysisRequest(BaseModel):
    job_requirement_id: int
    model: str = "llama3.2"


class ATSInsights(BaseModel):
    ats_score: int
    summary: str
    issues: List[str]
    recommendations: List[str]


class ATSAnalysisResponse(BaseModel):
    insights: ATSInsights
    raw_report: str


# ---------- Helpers ----------

FILES_ROOT = Path("files")
ORIGINAL_DIR = FILES_ROOT / "original"
TAILORED_DIR = FILES_ROOT / "tailored"


def build_file_url(path: Path) -> str:
    # The StaticFiles mount in main.py will serve /files/<relative-path>
    relative = path.relative_to(FILES_ROOT)
    return f"/files/{relative.as_posix()}"


# ---------- Routes ----------

@router.post("/cv/upload", response_model=CVUploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_cv(
    job_title: str = Form(...),
    job_description: str = Form(...),
    cv_title: str = Form("My CV"),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    if file.content_type not in ("application/pdf", "application/x-pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF files are supported.",
        )

    ORIGINAL_DIR.mkdir(parents=True, exist_ok=True)

    ext = ".pdf"
    filename = f"{uuid4()}{ext}"
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

    # Create CVDocument
    cv_doc = models.CVDocument(
        title=cv_title,
        original_pdf_path=str(output_path),
        original_text=extracted_text,
    )
    db.add(cv_doc)
    db.commit()
    db.refresh(cv_doc)

    # Create JobRequirement
    job_req = models.JobRequirement(
        cv_id=cv_doc.id,
        title=job_title,
        description=job_description,
    )
    db.add(job_req)
    db.commit()
    db.refresh(job_req)

    return CVUploadResponse(
        cv_id=cv_doc.id,
        job_requirement_id=job_req.id,
        message="CV and job requirement uploaded successfully.",
    )


@router.post("/cv/{cv_id}/tailor", response_model=TailoredCVResponse)
def tailor_cv(
    cv_id: int,
    payload: TailorRequest,
    db: Session = Depends(get_db),
):
    cv_doc = db.query(models.CVDocument).filter(models.CVDocument.id == cv_id).first()
    if not cv_doc:
        raise HTTPException(status_code=404, detail="CV not found.")

    job_req = (
        db.query(models.JobRequirement)
        .filter(models.JobRequirement.id == payload.job_requirement_id)
        .first()
    )
    if not job_req:
        raise HTTPException(status_code=404, detail="Job requirement not found.")

    # Call LLM to get tailored text
    tailored_text = generate_tailored_cv_text(
        model=payload.model,
        job_description=job_req.description,
        original_cv_text=cv_doc.original_text,
    )

    # Generate tailored PDF
    TAILORED_DIR.mkdir(parents=True, exist_ok=True)
    tailored_filename = f"{uuid4()}.pdf"
    tailored_path = TAILORED_DIR / tailored_filename
    generate_pdf_from_text(tailored_text, str(tailored_path))

    tailored_record = models.TailoredCV(
        cv_id=cv_doc.id,
        job_requirement_id=job_req.id,
        tailored_text=tailored_text,
        tailored_pdf_path=str(tailored_path),
        model_used=payload.model,
    )
    db.add(tailored_record)
    db.commit()
    db.refresh(tailored_record)

    return TailoredCVResponse(
        id=tailored_record.id,
        cv_id=cv_doc.id,
        job_requirement_id=job_req.id,
        tailored_text=tailored_record.tailored_text,
        tailored_pdf_url=build_file_url(Path(tailored_record.tailored_pdf_path)),
        model_used=tailored_record.model_used,
    )


@router.post("/cv/{cv_id}/analyze", response_model=ATSAnalysisResponse)
def analyze_cv(
    cv_id: int,
    payload: ATSAnalysisRequest,
    db: Session = Depends(get_db),
):
    cv_doc = db.query(models.CVDocument).filter(models.CVDocument.id == cv_id).first()
    if not cv_doc:
        raise HTTPException(status_code=404, detail="CV not found.")

    job_req = (
        db.query(models.JobRequirement)
        .filter(models.JobRequirement.id == payload.job_requirement_id)
        .first()
    )
    if not job_req:
        raise HTTPException(status_code=404, detail="Job requirement not found.")

    analysis = analyze_cv_for_ats(
        model=payload.model,
        job_description=job_req.description,
        original_cv_text=cv_doc.original_text,
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
    db: Session = Depends(get_db),
):
    cv_doc = db.query(models.CVDocument).filter(models.CVDocument.id == cv_id).first()
    if not cv_doc:
        raise HTTPException(status_code=404, detail="CV not found.")

    tailored = (
        db.query(models.TailoredCV)
        .filter(models.TailoredCV.cv_id == cv_id)
        .order_by(models.TailoredCV.created_at.desc())
        .all()
    )

    cv_detail = CVDetail(
        id=cv_doc.id,
        title=cv_doc.title,
        original_pdf_url=build_file_url(Path(cv_doc.original_pdf_path)),
        original_text=cv_doc.original_text,
    )

    tailored_list = [
        TailoredSummary(
            id=t.id,
            job_requirement_id=t.job_requirement_id,
            tailored_pdf_url=build_file_url(Path(t.tailored_pdf_path)),
            model_used=t.model_used,
        )
        for t in tailored
    ]

    return CVWithTailored(cv=cv_detail, tailored_versions=tailored_list)


@router.get("/tailored/{tailored_id}", response_model=TailoredCVResponse)
def get_tailored_cv(
    tailored_id: int,
    db: Session = Depends(get_db),
):
    t = (
        db.query(models.TailoredCV)
        .filter(models.TailoredCV.id == tailored_id)
        .first()
    )
    if not t:
        raise HTTPException(status_code=404, detail="Tailored CV not found.")

    return TailoredCVResponse(
        id=t.id,
        cv_id=t.cv_id,
        job_requirement_id=t.job_requirement_id,
        tailored_text=t.tailored_text,
        tailored_pdf_url=build_file_url(Path(t.tailored_pdf_path)),
        model_used=t.model_used,
    )
