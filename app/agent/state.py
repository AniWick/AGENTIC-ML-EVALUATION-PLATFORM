# =====================================================================
# Updated state_2.py
# =====================================================================
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from datetime import datetime


@dataclass
class AgentState:
    """
    Central authoritative state for the agentic evaluation workflow.
    """

    project_id: str
    user_goal: str

    requested_workflow: str = "full_evaluation"
    workflow_objective: Optional[str] = None

    model_id: Optional[str] = None
    model_path: Optional[str] = None

    dataset_id: Optional[str] = None
    dataset_path: Optional[str] = None

    current_dataset_version: Optional[str] = None
    evaluated_dataset_version: Optional[str] = None

    target_column: Optional[str] = None
    task_type: Optional[str] = None

    model_inspected: bool = False
    dataset_inspected: bool = False
    compatibility_validated: bool = False
    compatibility_result: Optional[bool] = None

    baseline_evaluated: bool = False
    data_quality_evaluated: bool = False
    robustness_evaluated: bool = False

    adversarial_evaluated: bool = False
    attack_type: Optional[str] = None
    clean_accuracy: Optional[float] = None
    adversarial_accuracy: Optional[float] = None
    attack_results: Dict[str, Any] = field(default_factory=dict)

    defended: bool = False
    defense_results: Dict[str, Any] = field(default_factory=dict)

    defended_attack_evaluated: bool = False
    defended_attack_results: Optional[Dict[str, Any]] = None

    #: Multi-step comparative sweep (Design A). ``defended_attack_results`` is
    #: still populated with the last step so older report paths keep working.
    attack_defense_sequence_evaluated: bool = False
    attack_defense_sequence_results: Optional[Dict[str, Any]] = None

    #: Cumulative hardening loop (Design B).
    hardening_loop_evaluated: bool = False
    hardening_loop_results: Optional[Dict[str, Any]] = None

    model_hardened: bool = False
    hardened_model_info: Optional[Dict[str, Any]] = None

    model_metadata: Dict[str, Any] = field(default_factory=dict)
    dataset_metadata: Dict[str, Any] = field(default_factory=dict)

    baseline_metrics: Dict[str, Any] = field(default_factory=dict)
    data_quality: Dict[str, Any] = field(default_factory=dict)
    robustness_results: Dict[str, Any] = field(default_factory=dict)

    latest_metrics: Dict[str, Any] = field(default_factory=dict)

    transformations_applied: List[Dict[str, Any]] = field(default_factory=list)
    experiments: List[Dict[str, Any]] = field(default_factory=list)

    observations: List[Dict[str, Any]] = field(default_factory=list)
    actions_taken: List[Dict[str, Any]] = field(default_factory=list)
    messages_history: List[Dict[str, Any]] = field(default_factory=list)

    iteration: int = 0
    max_iterations: int = 20

    status: str = "initialized"
    stop_reason: Optional[str] = None

    created_at: str = field(
        default_factory=lambda: datetime.utcnow().isoformat()
    )

    updated_at: str = field(
        default_factory=lambda: datetime.utcnow().isoformat()
    )

    def update_timestamp(self) -> None:
        self.updated_at = datetime.utcnow().isoformat()

    def is_complete(self) -> bool:
        if self.status in {
            "completed",
            "failed",
            "waiting_for_approval",
        }:
            return True

        if self.iteration >= self.max_iterations:
            return True

        return False

    def workflow_complete(self) -> bool:
        workflow = (self.requested_workflow or "").lower().strip()

        if workflow in {
            "attack",
            "adversarial_attack",
            "boundary_attack",
            "hopskipjump",
        }:
            return self.adversarial_evaluated

        if workflow in {
            "defend",
            "defense",
            "gaussian_augmentation",
        }:
            return self.defended

        if workflow in {"defended_attack", "attack_defended_model"}:
            return (
                self.defended_attack_evaluated
                or self.attack_defense_sequence_evaluated
            )

        if workflow in {"hardening_loop", "harden_defense_loop"}:
            return self.hardening_loop_evaluated

        if workflow in {"harden_model", "harden"}:
            return self.model_hardened

        if workflow in {
            "inspection",
            "inspect",
            "inspect_model_dataset",
        }:
            return (
                self.model_inspected
                and self.dataset_inspected
                and self.compatibility_validated
            )

        if workflow in {
            "baseline",
            "baseline_metrics",
            "baseline_evaluation",
        }:
            return self.baseline_evaluated

        if workflow in {
            "full_evaluation",
            "evaluate",
            "evaluation",
        }:
            return self.full_evaluation_complete()

        if workflow in {
            "improve_dataset",
            "dataset_improvement",
            "improvement",
        }:
            return self.improvement_complete()

        if workflow in {
            "targeted_transform",
            "transform",
            "transformation",
            "clean_dataset",
            "balance_dataset",
            "handle_outliers",
            "scale_dataset",
            "add_noise",
        }:
            return len(self.transformations_applied) > 0

        if workflow in {
            "report",
            "generate_report",
            "contextual",
            "comparison",
        }:
            return self.status == "completed"

        return False

    def full_evaluation_complete(self) -> bool:
        return (
            self.model_inspected
            and self.dataset_inspected
            and self.compatibility_validated
            and self.baseline_evaluated
            and self.data_quality_evaluated
            and self.robustness_evaluated
            and (
                self.evaluated_dataset_version
                == self.current_dataset_version
            )
        )

    def improvement_complete(self) -> bool:
        return (
            len(self.transformations_applied) > 0
            and bool(self.latest_metrics)
            and (
                self.evaluated_dataset_version
                == self.current_dataset_version
            )
        )

    def get_workflow_context(self) -> Dict[str, Any]:
        return {
            "requested_workflow": self.requested_workflow,
            "workflow_objective": self.workflow_objective,
            "workflow_complete": self.workflow_complete(),
            "model_inspected": self.model_inspected,
            "dataset_inspected": self.dataset_inspected,
            "compatibility_validated": self.compatibility_validated,
            "compatibility_result": self.compatibility_result,
            "baseline_evaluated": self.baseline_evaluated,
            "data_quality_evaluated": self.data_quality_evaluated,
            "robustness_evaluated": self.robustness_evaluated,
            "adversarial_evaluated": self.adversarial_evaluated,
            "attack_type": self.attack_type,
            "clean_accuracy": self.clean_accuracy,
            "adversarial_accuracy": self.adversarial_accuracy,
            "defended": self.defended,
            "defended_attack_evaluated": self.defended_attack_evaluated,
            "attack_defense_sequence_evaluated": self.attack_defense_sequence_evaluated,
            "attack_defense_sequence_results": self.attack_defense_sequence_results,
            "model_hardened": self.model_hardened,
            "current_dataset_version": self.current_dataset_version,
            "evaluated_dataset_version": self.evaluated_dataset_version,
            "transformations_count": len(self.transformations_applied),
            "experiments_count": len(self.experiments),
        }

    def get_summary_context(self) -> Dict[str, Any]:
        return {
            "project_id": self.project_id,
            "user_goal": self.user_goal,
            "requested_workflow": self.requested_workflow,
            "workflow_objective": self.workflow_objective,
            "workflow_complete": self.workflow_complete(),
            "model_id": self.model_id,
            "dataset_id": self.dataset_id,
            "current_dataset_version": self.current_dataset_version,
            "evaluated_dataset_version": self.evaluated_dataset_version,
            "target_column": self.target_column,
            "task_type": self.task_type,
            "model_inspected": self.model_inspected,
            "dataset_inspected": self.dataset_inspected,
            "compatibility_validated": self.compatibility_validated,
            "compatibility_result": self.compatibility_result,
            "baseline_evaluated": self.baseline_evaluated,
            "data_quality_evaluated": self.data_quality_evaluated,
            "robustness_evaluated": self.robustness_evaluated,
            "adversarial_evaluated": self.adversarial_evaluated,
            "attack_type": self.attack_type,
            "clean_accuracy": self.clean_accuracy,
            "adversarial_accuracy": self.adversarial_accuracy,
            "attack_results": self.attack_results,
            "defended": self.defended,
            "defense_results": self.defense_results,
            "defended_attack_evaluated": self.defended_attack_evaluated,
            "attack_defense_sequence_evaluated": self.attack_defense_sequence_evaluated,
            "attack_defense_sequence_results": self.attack_defense_sequence_results,
            "defended_attack_results": self.defended_attack_results,
            "hardening_loop_evaluated": self.hardening_loop_evaluated,
            "hardening_loop_results": self.hardening_loop_results,
            "model_hardened": self.model_hardened,
            "hardened_model_info": self.hardened_model_info,
            "model_metadata": self.model_metadata,
            "dataset_metadata": self.dataset_metadata,
            "baseline_metrics": self.baseline_metrics,
            "data_quality": self.data_quality,
            "robustness_results": self.robustness_results,
            "latest_metrics": self.latest_metrics,
            "transformations_applied": self.transformations_applied,
            "experiments": self.experiments,
            "iteration": self.iteration,
            "status": self.status,
            "stop_reason": self.stop_reason,
        }

    def add_observation(self, action: str, result: Any) -> None:
        self.observations.append(
            {
                "timestamp": datetime.utcnow().isoformat(),
                "action": action,
                "result": result,
            }
        )
        self.update_timestamp()

    def add_action(self, action: str, arguments: Optional[Dict[str, Any]] = None) -> None:
        self.actions_taken.append(
            {
                "timestamp": datetime.utcnow().isoformat(),
                "action": action,
                "arguments": arguments or {},
            }
        )
        self.update_timestamp()

    def add_message(self, role: str, content: str) -> None:
        self.messages_history.append(
            {
                "timestamp": datetime.utcnow().isoformat(),
                "role": role,
                "content": content,
            }
        )
        self.update_timestamp()

    def mark_dataset_changed(self, new_version: str) -> None:
        self.current_dataset_version = new_version
        if self.evaluated_dataset_version != new_version:
            self.latest_metrics = {}
        self.update_timestamp()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "project_id": self.project_id,
            "user_goal": self.user_goal,
            "requested_workflow": self.requested_workflow,
            "workflow_objective": self.workflow_objective,
            "model_id": self.model_id,
            "model_path": self.model_path,
            "dataset_id": self.dataset_id,
            "dataset_path": self.dataset_path,
            "current_dataset_version": self.current_dataset_version,
            "evaluated_dataset_version": self.evaluated_dataset_version,
            "target_column": self.target_column,
            "task_type": self.task_type,
            "model_inspected": self.model_inspected,
            "dataset_inspected": self.dataset_inspected,
            "compatibility_validated": self.compatibility_validated,
            "compatibility_result": self.compatibility_result,
            "baseline_evaluated": self.baseline_evaluated,
            "data_quality_evaluated": self.data_quality_evaluated,
            "robustness_evaluated": self.robustness_evaluated,
            "adversarial_evaluated": self.adversarial_evaluated,
            "attack_type": self.attack_type,
            "clean_accuracy": self.clean_accuracy,
            "adversarial_accuracy": self.adversarial_accuracy,
            "attack_results": self.attack_results,
            "defended": self.defended,
            "defense_results": self.defense_results,
            "defended_attack_evaluated": self.defended_attack_evaluated,
            "attack_defense_sequence_evaluated": self.attack_defense_sequence_evaluated,
            "attack_defense_sequence_results": self.attack_defense_sequence_results,
            "defended_attack_results": self.defended_attack_results,
            "hardening_loop_evaluated": self.hardening_loop_evaluated,
            "hardening_loop_results": self.hardening_loop_results,
            "model_hardened": self.model_hardened,
            "hardened_model_info": self.hardened_model_info,
            "model_metadata": self.model_metadata,
            "dataset_metadata": self.dataset_metadata,
            "baseline_metrics": self.baseline_metrics,
            "data_quality": self.data_quality,
            "robustness_results": self.robustness_results,
            "latest_metrics": self.latest_metrics,
            "transformations_applied": self.transformations_applied,
            "experiments": self.experiments,
            "observations": self.observations,
            "actions_taken": self.actions_taken,
            "messages_history": self.messages_history,
            "iteration": self.iteration,
            "max_iterations": self.max_iterations,
            "status": self.status,
            "stop_reason": self.stop_reason,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }