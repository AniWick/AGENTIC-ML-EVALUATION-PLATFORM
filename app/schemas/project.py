from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime

class ProjectCreate(BaseModel):
    name: str
    description: Optional[str] = None

class ProjectResponse(BaseModel):
    id: str
    name: str
    description: Optional[str] = None
    model_id: Optional[str] = None
    dataset_id: Optional[str] = None
    created_at: str
    updated_at: str

class ProjectSummary(BaseModel):
    id: str
    name: str
    description: Optional[str] = None
    model_count: int = 0
    dataset_count: int = 0
    experiment_count: int = 0
    created_at: str

