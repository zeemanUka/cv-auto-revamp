from datetime import datetime
from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey
from sqlalchemy.orm import relationship

from .session import Base


class CVDocument(Base):
    __tablename__ = "cv_documents"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(255), nullable=False)
    original_pdf_path = Column(String(512), nullable=False)
    original_text = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    job_requirements = relationship(
        "JobRequirement", back_populates="cv", cascade="all, delete-orphan"
    )
    tailored_versions = relationship(
        "TailoredCV", back_populates="cv", cascade="all, delete-orphan"
    )
    ats_analyses = relationship(
        "ATSAnalysis", back_populates="cv", cascade="all, delete-orphan"
    )


class JobRequirement(Base):
    __tablename__ = "job_requirements"

    id = Column(Integer, primary_key=True, index=True)
    cv_id = Column(Integer, ForeignKey("cv_documents.id"), nullable=True)
    title = Column(String(255), nullable=False)
    description = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    cv = relationship("CVDocument", back_populates="job_requirements")
    tailored_versions = relationship(
        "TailoredCV", back_populates="job_requirement", cascade="all, delete-orphan"
    )
    ats_analyses = relationship(
        "ATSAnalysis", back_populates="job_requirement", cascade="all, delete-orphan"
    )


class TailoredCV(Base):
    __tablename__ = "tailored_cvs"

    id = Column(Integer, primary_key=True, index=True)
    cv_id = Column(Integer, ForeignKey("cv_documents.id"), nullable=False)
    job_requirement_id = Column(Integer, ForeignKey("job_requirements.id"), nullable=False)
    tailored_text = Column(Text, nullable=False)
    tailored_pdf_path = Column(String(512), nullable=False)
    model_used = Column(String(100), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    cv = relationship("CVDocument", back_populates="tailored_versions")
    job_requirement = relationship("JobRequirement", back_populates="tailored_versions")


class ATSAnalysis(Base):
    __tablename__ = "ats_analyses"

    id = Column(Integer, primary_key=True, index=True)
    cv_id = Column(Integer, ForeignKey("cv_documents.id"), nullable=False)
    job_requirement_id = Column(Integer, ForeignKey("job_requirements.id"), nullable=False)
    model_used = Column(String(100), nullable=False)
    ats_score = Column(Integer, nullable=False)
    summary = Column(Text, nullable=False)
    issues_json = Column(Text, nullable=False, default="[]")
    recommendations_json = Column(Text, nullable=False, default="[]")
    raw_report = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    cv = relationship("CVDocument", back_populates="ats_analyses")
    job_requirement = relationship("JobRequirement", back_populates="ats_analyses")
