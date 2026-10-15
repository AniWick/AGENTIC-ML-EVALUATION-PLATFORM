from fastapi import APIRouter, HTTPException
from typing import List, Dict, Any
from app.services.preset_service import preset_service
from app.services.project_service import project_service

router = APIRouter(tags=["presets"])

@router.get("/presets")
def get_presets() -> List[Dict[str, Any]]:
    return preset_service.list_available_presets()

@router.post("/projects/{project_id}/load-preset/{preset_id}")
def load_preset_into_project(project_id: str, preset_id: str):
    proj = project_service.get_project(project_id)
    if not proj:
        raise HTTPException(status_code=404, detail="Project not found")

    try:
        res = preset_service.load_preset_into_project(project_id, preset_id)
        return res
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

