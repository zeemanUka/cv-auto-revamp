import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# Load variables from .env before importing modules that read env at import time.
load_dotenv()

from app.db.session import Base, engine
from app.api.cv_routes import router as cv_router

Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="CV Revamp Backend",
    description="Backend API for uploading CV PDFs, job requirements, and generating tailored CVs.",
    version="0.1.0",
)

origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://192.168.1.238:3000",
    "https://cv-revamp-frontend.vercel.app"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
configured_files_root = Path(os.getenv("FILES_ROOT", "files"))
if not configured_files_root.is_absolute():
    configured_files_root = PROJECT_ROOT / configured_files_root
FILES_ROOT = configured_files_root.resolve()

# Ensure static directories always exist (local and containerized runs).
FILES_ROOT.mkdir(parents=True, exist_ok=True)
(FILES_ROOT / "original").mkdir(parents=True, exist_ok=True)
(FILES_ROOT / "tailored").mkdir(parents=True, exist_ok=True)

app.mount("/files", StaticFiles(directory=str(FILES_ROOT)), name="files")

app.include_router(cv_router, prefix="/api")

## Create a copy of this file and name it Verticul Data.
