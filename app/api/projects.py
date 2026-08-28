from fastapi import APIRouter, HTTPException
from typing import List
from app.schemas.project import ProjectCreate, ProjectResponse
from app.services.project_service import project_service

router = APIRouter(prefix="/projects", tags=["projects"])

@router.post("", response_model=ProjectResponse)
def create_project(payload: ProjectCreate):
    return project_service.create_project(name=payload.name, description=payload.description)

@router.get("", response_model=List[ProjectResponse])
def list_projects():
    return project_service.list_projects()

@router.get("/{project_id}", response_model=ProjectResponse)
def get_project(project_id: str):
    proj = project_service.get_project(project_id)
    if not proj:
        raise HTTPException(status_code=404, detail="Project not found")
    return proj

