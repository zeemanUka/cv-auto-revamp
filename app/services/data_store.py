import json
import os
import threading
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
configured_store_path = Path(os.getenv("DATA_STORE_PATH", "data/store.json"))
if not configured_store_path.is_absolute():
    configured_store_path = PROJECT_ROOT / configured_store_path
STORE_PATH = configured_store_path.resolve()


class JsonDataStore:
    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.Lock()
        self._ensure_store_file()

    @staticmethod
    def _default_payload() -> Dict[str, Any]:
        return {
            "next_ids": {
                "cv_documents": 1,
                "job_requirements": 1,
                "tailored_cvs": 1,
                "ats_analyses": 1,
            },
            "cv_documents": [],
            "job_requirements": [],
            "tailored_cvs": [],
            "ats_analyses": [],
        }

    def _ensure_store_file(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self._write_payload(self._default_payload())

    def _read_payload(self) -> Dict[str, Any]:
        self._ensure_store_file()
        with self.path.open("r", encoding="utf-8") as f:
            try:
                payload = json.load(f)
            except json.JSONDecodeError:
                payload = self._default_payload()

        default_payload = self._default_payload()
        for key, value in default_payload.items():
            payload.setdefault(key, value)

        for key, value in default_payload["next_ids"].items():
            payload["next_ids"].setdefault(key, value)

        return payload

    def _write_payload(self, payload: Dict[str, Any]) -> None:
        temp_path = self.path.with_suffix(f"{self.path.suffix}.tmp")
        with temp_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=True, indent=2)
        temp_path.replace(self.path)

    @staticmethod
    def _find_by_id(records: List[Dict[str, Any]], record_id: int) -> Optional[Dict[str, Any]]:
        for record in records:
            if int(record.get("id", -1)) == int(record_id):
                return record
        return None

    @staticmethod
    def _next_id(payload: Dict[str, Any], bucket: str) -> int:
        current = int(payload["next_ids"].get(bucket, 1))
        payload["next_ids"][bucket] = current + 1
        return current

    @staticmethod
    def _now_iso() -> str:
        return datetime.utcnow().isoformat()

    def get_snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return deepcopy(self._read_payload())

    def create_cv_and_job(
        self,
        cv_title: str,
        original_pdf_path: str,
        original_text: str,
        job_title: str,
        job_description: str,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        with self._lock:
            payload = self._read_payload()

            now = self._now_iso()
            cv_record = {
                "id": self._next_id(payload, "cv_documents"),
                "title": cv_title,
                "original_pdf_path": original_pdf_path,
                "original_text": original_text,
                "created_at": now,
                "updated_at": now,
            }
            payload["cv_documents"].append(cv_record)

            job_record = {
                "id": self._next_id(payload, "job_requirements"),
                "cv_id": cv_record["id"],
                "title": job_title,
                "description": job_description,
                "created_at": now,
            }
            payload["job_requirements"].append(job_record)

            self._write_payload(payload)
            return deepcopy(cv_record), deepcopy(job_record)

    def get_cv(self, cv_id: int) -> Optional[Dict[str, Any]]:
        with self._lock:
            payload = self._read_payload()
            record = self._find_by_id(payload["cv_documents"], cv_id)
            return deepcopy(record) if record else None

    def get_job_requirement(self, job_requirement_id: int) -> Optional[Dict[str, Any]]:
        with self._lock:
            payload = self._read_payload()
            record = self._find_by_id(payload["job_requirements"], job_requirement_id)
            return deepcopy(record) if record else None

    def create_tailored_cv(
        self,
        cv_id: int,
        job_requirement_id: int,
        tailored_text: str,
        tailored_pdf_path: str,
        model_used: str,
    ) -> Dict[str, Any]:
        with self._lock:
            payload = self._read_payload()
            record = {
                "id": self._next_id(payload, "tailored_cvs"),
                "cv_id": cv_id,
                "job_requirement_id": job_requirement_id,
                "tailored_text": tailored_text,
                "tailored_pdf_path": tailored_pdf_path,
                "model_used": model_used,
                "created_at": self._now_iso(),
            }
            payload["tailored_cvs"].append(record)
            self._write_payload(payload)
            return deepcopy(record)

    def get_tailored_cv(self, tailored_id: int) -> Optional[Dict[str, Any]]:
        with self._lock:
            payload = self._read_payload()
            record = self._find_by_id(payload["tailored_cvs"], tailored_id)
            return deepcopy(record) if record else None

    def list_tailored_cvs(self, cv_id: Optional[int] = None) -> List[Dict[str, Any]]:
        with self._lock:
            payload = self._read_payload()
            records = payload["tailored_cvs"]
            if cv_id is not None:
                records = [r for r in records if int(r.get("cv_id", -1)) == int(cv_id)]
            return sorted(deepcopy(records), key=lambda r: r.get("created_at", ""), reverse=True)

    def create_ats_analysis(
        self,
        cv_id: int,
        job_requirement_id: int,
        model_used: str,
        ats_score: int,
        summary: str,
        issues: List[str],
        recommendations: List[str],
        raw_report: str,
    ) -> Dict[str, Any]:
        with self._lock:
            payload = self._read_payload()
            record = {
                "id": self._next_id(payload, "ats_analyses"),
                "cv_id": cv_id,
                "job_requirement_id": job_requirement_id,
                "model_used": model_used,
                "ats_score": int(ats_score),
                "summary": summary,
                "issues": list(issues),
                "recommendations": list(recommendations),
                "raw_report": raw_report,
                "created_at": self._now_iso(),
            }
            payload["ats_analyses"].append(record)
            self._write_payload(payload)
            return deepcopy(record)

    def list_ats_analyses(self, cv_id: Optional[int] = None) -> List[Dict[str, Any]]:
        with self._lock:
            payload = self._read_payload()
            records = payload["ats_analyses"]
            if cv_id is not None:
                records = [r for r in records if int(r.get("cv_id", -1)) == int(cv_id)]
            return sorted(deepcopy(records), key=lambda r: r.get("created_at", ""), reverse=True)


store = JsonDataStore(STORE_PATH)
