from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

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
    "http://192.168.1.238:3000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/files", StaticFiles(directory="files"), name="files")

app.include_router(cv_router, prefix="/api")
