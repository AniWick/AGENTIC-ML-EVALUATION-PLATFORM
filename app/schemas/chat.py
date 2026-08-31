from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

class ChatRequest(BaseModel):
    message: str
    allow_transformations: bool = True

class ChatResponse(BaseModel):
    message: str
    status: str # 'completed', 'waiting_for_approval', 'running', 'failed'
    actions_taken: List[str] = []
    current_dataset_version: Optional[str] = None
    experiment_id: Optional[str] = None
    report_id: Optional[str] = None
    metrics_summary: Optional[Dict[str, Any]] = None
    robustness_metrics: Optional[Dict[str, Any]] = None
