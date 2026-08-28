from fastapi import APIRouter, UploadFile, File, HTTPException
import shutil
import tempfile
from pathlib import Path
from app.services.project_service import project_service
from app.datasets.versioning import dataset_version_manager

router = APIRouter(tags=["datasets"])

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

