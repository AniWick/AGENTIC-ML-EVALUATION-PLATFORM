from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

class ExperimentRecord(BaseModel):
    experiment_id: str
    project_id: str
    model_id: str
    operation: str
    parameters: Dict[str, Any]
    input_dataset_version: str
    output_dataset_version: str
    metrics_before: Dict[str, Any]
    metrics_after: Dict[str, Any]
    comparison: Dict[str, Any]
    agent_reasoning: str
    status: str = "completed"
    created_at: str

