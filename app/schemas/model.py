from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

class ModelMetadata(BaseModel):
    model_id: str
    name: str
    framework: str # 'sklearn', 'pytorch', 'onnx'
    status: str = "loaded"
    file_path: str
    architecture: Optional[str] = None
    input_shape: Optional[List[int]] = None
    output_shape: Optional[List[int]] = None
    parameter_count: Optional[int] = None
    model_type: Optional[str] = None # 'classification', 'regression'
    feature_names: Optional[List[str]] = None
    target_classes: Optional[List[Any]] = None
    created_at: str

class ModelInspectResult(BaseModel):
    model_id: str
    framework: str
    model_type: str
    input_shape: Optional[List[Any]] = None
    output_classes: Optional[int] = None
    output_shape: Optional[List[Any]] = None
    parameter_count: Optional[int] = None
    feature_names: Optional[List[str]] = None
    status: str

