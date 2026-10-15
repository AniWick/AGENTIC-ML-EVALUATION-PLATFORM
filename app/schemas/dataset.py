from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

class DatasetVersionInfo(BaseModel):
    version: str
    parent_version: Optional[str] = None
    file_path: str
    operation: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None
    row_count: int
    column_count: int
    created_at: str

class DatasetInspectResult(BaseModel):
    dataset_id: str
    version: str
    rows: int
    columns: int
    column_names: List[str]
    numerical_features: List[str]
    categorical_features: List[str]
    target_column: Optional[str] = None
    missing_values_count: int
    missing_values_by_column: Dict[str, int]
    duplicate_rows_count: int
    class_distribution: Optional[Dict[str, int]] = None
    data_types: Dict[str, str]
    outliers_count: Optional[int] = None
    statistics: Optional[Dict[str, Dict[str, float]]] = None

class ValidationResult(BaseModel):
    is_compatible: bool
    model_id: str
    dataset_id: str
    dataset_version: str
    checks: Dict[str, bool] # {"input_shape": True, "feature_types": True, "target_compatible": True}
    issues: List[str]
    warnings: List[str]

