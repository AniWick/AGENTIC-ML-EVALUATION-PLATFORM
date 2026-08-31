from fastapi import APIRouter, UploadFile, File, HTTPException
import shutil
import tempfile
from pathlib import Path
from datetime import datetime
from app.services.project_service import project_service
from app.datasets.versioning import dataset_version_manager
from app.core.database import db

router = APIRouter(tags=["datasets"])

@router.get("/datasets")
def list_datasets():
    with db._get_connection() as conn:
        return [dict(row) for row in conn.execute("SELECT id, name, current_version, format, project_id, created_at FROM datasets ORDER BY created_at DESC")]

@router.post("/projects/{project_id}/datasets/{dataset_id}/select")
def select_existing_dataset(project_id: str, dataset_id: str):
    if not project_service.get_project(project_id):
        raise HTTPException(status_code=404, detail="Project not found")
    with db._get_connection() as conn:
        if not conn.execute("SELECT id FROM datasets WHERE id = ?", (dataset_id,)).fetchone():
            raise HTTPException(status_code=404, detail="Dataset not found")
        conn.execute("UPDATE projects SET dataset_id = ?, updated_at = ? WHERE id = ?", (dataset_id, datetime.utcnow().isoformat(), project_id))
        conn.commit()
    return {"status": "success", "selected_dataset_id": dataset_id}

@router.post("/projects/{project_id}/datasets")
def upload_project_dataset(project_id: str, file: UploadFile = File(...)):
    proj = project_service.get_project(project_id)
    if not proj:
        raise HTTPException(status_code=404, detail="Project not found")

    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        res = project_service.upload_dataset(project_id, tmp_path, file.filename)
        return {"status": "success", "dataset": res}
    finally:
        if Path(tmp_path).exists():
            Path(tmp_path).unlink()

@router.get("/datasets/{dataset_id}/versions")
def get_dataset_versions(dataset_id: str):
    return dataset_version_manager.list_versions(dataset_id)
