from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
from datetime import datetime


class Observation(BaseModel):
    tool: str
    parameters: Dict[str, Any]
    result: Any
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class ActionRecord(BaseModel):
    action: str
    reason: str
    status: str = "completed"
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class AgentState(BaseModel):
    """Structured, explicit state maintained across the entire agent lifecycle."""

    project_id: str
    user_goal: str
    model_id: str
    model_path: str
    dataset_id: str
    dataset_path: str
    current_dataset_version: str = "v001"
    target_column: Optional[str] = None
    task_type: str = "classification"

    is_model_inspected: bool = False
    is_dataset_inspected: bool = False
    is_compatibility_validated: bool = False
    is_baseline_evaluated: bool = False
    is_data_quality_evaluated: bool = False
    is_robustness_evaluated: bool = False

    model_metadata: Optional[Dict[str, Any]] = None
    dataset_metadata: Optional[Dict[str, Any]] = None
    baseline_metrics: Optional[Dict[str, Any]] = None
    data_quality_metrics: Optional[Dict[str, Any]] = None
    robustness_metrics: Optional[Dict[str, Any]] = None
    latest_metrics: Optional[Dict[str, Any]] = None

    # Which dataset version the latest performance metrics actually belong to.
    evaluated_dataset_version: Optional[str] = None

    transformations_applied: List[str] = Field(default_factory=list)
    experiments: List[Dict[str, Any]] = Field(default_factory=list)
    observations: List[Observation] = Field(default_factory=list)
    actions_taken: List[ActionRecord] = Field(default_factory=list)
    messages_history: List[str] = Field(default_factory=list)

    iteration: int = 0
    max_iterations: int = 25
    status: str = "idle"
    stop_reason: Optional[str] = None

    def add_observation(self, tool: str, parameters: Dict[str, Any], result: Any):
        self.observations.append(
            Observation(tool=tool, parameters=parameters, result=result)
        )

    def record_action(self, action: str, reason: str, status: str = "completed"):
        self.actions_taken.append(
            ActionRecord(action=action, reason=reason, status=status)
        )

    def is_complete(self) -> bool:
        return self.status in [
            "completed", "failed", "waiting_for_approval"
        ] or self.iteration >= self.max_iterations

    def get_summary_context(self) -> str:
        lines = [
            f"PROJECT: {self.project_id}",
            f"USER GOAL: {self.user_goal}",
            f"MODEL: {self.model_id} ({self.task_type})",
            f"CURRENT DATASET: {self.current_dataset_version}",
            f"EVALUATED DATASET VERSION: {self.evaluated_dataset_version or 'none'}",
            f"ITERATION: {self.iteration}/{self.max_iterations}",
        ]

        if self.baseline_metrics:
            lines.append(
                f"BASELINE PERFORMANCE: Accuracy: {self.baseline_metrics.get('accuracy', 'N/A')}, "
                f"Macro F1: {self.baseline_metrics.get('f1_macro', 'N/A')}, "
                f"Minority F1: {self.baseline_metrics.get('f1_minority', 'N/A')}"
            )

        if self.data_quality_metrics:
            lines.append(
                f"DATA QUALITY: Imbalance: "
                f"{self.data_quality_metrics.get('class_imbalance_detected')} "
                f"(ratio: {self.data_quality_metrics.get('imbalance_ratio')}), "
                f"Missing Values: {self.data_quality_metrics.get('missing_values_count', 0)}"
            )

        if self.latest_metrics and self.latest_metrics != self.baseline_metrics:
            lines.append(f"LATEST PERFORMANCE: {self.latest_metrics}")

        if self.transformations_applied:
            lines.append(
                f"TRANSFORMATIONS APPLIED: {', '.join(self.transformations_applied)}"
            )

        return "\n".join(lines)
