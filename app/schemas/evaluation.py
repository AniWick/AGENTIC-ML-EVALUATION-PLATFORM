from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

class PerformanceMetrics(BaseModel):
    task_type: str # 'classification', 'regression'
    accuracy: Optional[float] = None
    precision_macro: Optional[float] = None
    recall_macro: Optional[float] = None
    f1_macro: Optional[float] = None
    f1_minority: Optional[float] = None
    per_class_f1: Optional[Dict[str, float]] = None
    confusion_matrix: Optional[List[List[int]]] = None
    roc_auc: Optional[float] = None
    mae: Optional[float] = None
    mse: Optional[float] = None
    rmse: Optional[float] = None
    r2_score: Optional[float] = None

class DataQualityMetrics(BaseModel):
    missing_values_count: int
    missing_values_ratio: float
    duplicate_rows_count: int
    class_imbalance_detected: bool
    imbalance_ratio: Optional[float] = None
    outliers_count: int
    outliers_ratio: float
    constant_columns: List[str]
    high_cardinality_columns: List[str]
    status_summary: Dict[str, str] # e.g. {"missing_values": "warning", "class_imbalance": "problem"}

class RobustnessMetrics(BaseModel):
    perturbation_type: str
    noise_level: float
    original_score: float
    perturbed_score: float
    score_drop: float
    drop_percentage: float
    is_sensitive: bool
    # Full clean/noisy metric pairs power the dashboard comparison chart.
    clean_metrics: Dict[str, float] = Field(default_factory=dict)
    noisy_metrics: Dict[str, float] = Field(default_factory=dict)

class EvaluationResult(BaseModel):
    evaluation_id: str
    project_id: str
    model_id: str
    dataset_version: str
    performance: PerformanceMetrics
    data_quality: DataQualityMetrics
    robustness: Optional[Dict[str, RobustnessMetrics]] = None
    warnings: List[str] = []
    created_at: str

class ComparisonResult(BaseModel):
    baseline_version: str
    experiment_version: str
    baseline_metrics: Dict[str, Any]
    experiment_metrics: Dict[str, Any]
    metric_deltas: Dict[str, float]
    summary_of_changes: List[str]
    is_improved: bool
