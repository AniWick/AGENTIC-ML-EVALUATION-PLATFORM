from fastapi import APIRouter, UploadFile, File, HTTPException
import shutil
import tempfile
from pathlib import Path
from datetime import datetime
from app.services.project_service import project_service
from app.core.database import db

router = APIRouter(tags=["models"])

@router.get("/models")
def list_models():
    """Return all stored models across global memory for dropdown selection."""
    with db._get_connection() as conn:
        rows = conn.execute(
            """
            SELECT 
                id, 
                COALESCE(name, id) as name, 
                framework, 
                project_id, 
                created_at 
            FROM models 
            ORDER BY created_at DESC
            """
        ).fetchall()
        return [dict(row) for row in rows]

@router.post("/projects/{project_id}/models/{model_id}/select")
def select_existing_model(project_id: str, model_id: str):
    if not project_service.get_project(project_id):
        raise HTTPException(status_code=404, detail="Project not found")
        
    with db._get_connection() as conn:
        model_row = conn.execute("SELECT id FROM models WHERE id = ?", (model_id,)).fetchone()
        if not model_row:
            raise HTTPException(status_code=404, detail="Model not found")
            
        conn.execute(
            "UPDATE projects SET model_id = ?, updated_at = ? WHERE id = ?",
            (model_id, datetime.utcnow().isoformat(), project_id)
        )
        conn.commit()
        
    return {"status": "success", "selected_model_id": model_id}

@router.post("/projects/{project_id}/models")
def upload_project_model(project_id: str, file: UploadFile = File(...)):
    proj = project_service.get_project(project_id)
    if not proj:
        raise HTTPException(status_code=404, detail="Project not found")

    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        res = project_service.upload_model(project_id, tmp_path, file.filename)
        
        # Ensure model is registered in global memory
        if isinstance(res, dict) and "id" in res:
            db.register_model_if_not_exists(
                model_id=res["id"],
                project_id=project_id,
                name=res.get("name", file.filename),
                file_path=res.get("file_path", tmp_path),
                framework=res.get("framework", "sklearn")
            )
            
        return {"status": "success", "model": res}
    finally:
        if Path(tmp_path).exists():
            Path(tmp_path).unlink()