import json
from typing import Dict, Any, Optional
from app.agent.state import AgentState
from app.agent.tools import tool_registry
from app.llm.ollama_client import ollama_client
from app.llm.prompts import AGENT_SYSTEM_PROMPT

class Decision:
    def __init__(self, action: str, parameters: Dict[str, Any], reason: str, status: str = "continue"):
        self.action = action
        self.parameters = parameters
        self.reason = reason
        self.status = status

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "action": self.action,
            "parameters": self.parameters,
            "reason": self.reason
        }

class AgentPlanner:
    """Decides the next tool to run based on current observations, goals, and state."""

    @staticmethod
    def decide_next_step(state: AgentState) -> Decision:
        # Build compact context
        context_str = state.get_summary_context()
        tool_defs = json.dumps(tool_registry.get_tool_definitions(), indent=2)

        prompt = (
            f"CURRENT STATE CONTEXT:\n{context_str}\n\n"
            f"AVAILABLE TOOLS:\n{tool_defs}\n\n"
            f"USER OBJECTIVE: {state.user_goal}\n\n"
            f"Determine the single next best action to perform. Respond with valid JSON only."
        )

        # 1. Try Ollama LLM structured decision
        llm_decision = ollama_client.generate_json(prompt=prompt, system=AGENT_SYSTEM_PROMPT)
        if llm_decision and "action" in llm_decision and llm_decision["action"] in tool_registry.tools:
            action = llm_decision["action"]
            params = llm_decision.get("parameters", {})
            reason = llm_decision.get("reason", "LLM reasoning step")
            status = llm_decision.get("status", "continue")

            # Validate parameters against state
            validated_params = AgentPlanner._validate_and_fill_params(action, params, state)
            return Decision(action=action, parameters=validated_params, reason=reason, status=status)

        # 2. Fallback to Deterministic Expert Planning System
        return AgentPlanner._deterministic_fallback_plan(state)

    @staticmethod
    def _validate_and_fill_params(action: str, params: Dict[str, Any], state: AgentState) -> Dict[str, Any]:
        """Ensures tool parameters match state (e.g. paths, versions)."""
        p = dict(params)
        if action in ["inspect_model", "validate_model_dataset", "evaluate_performance", "evaluate_robustness"]:
            p.setdefault("model_path", state.model_path)
            p.setdefault("model_id", state.model_id)

        if action in ["inspect_dataset", "validate_model_dataset", "evaluate_performance", "evaluate_data_quality", "evaluate_robustness"]:
            p.setdefault("dataset_path", state.dataset_path)
            p.setdefault("dataset_id", state.dataset_id)
            p.setdefault("version", state.current_dataset_version)
            if state.target_column:
                p.setdefault("target_column", state.target_column)

        if action in ["handle_missing_values", "balance_data", "handle_outliers", "scale_data", "add_gaussian_noise"]:
            p.setdefault("dataset_id", state.dataset_id)
            p.setdefault("current_version", state.current_dataset_version)
            if state.target_column:
                p.setdefault("target_column", state.target_column)

        if action == "save_experiment":
            p.setdefault("project_id", state.project_id)
            p.setdefault("model_id", state.model_id)

        if action == "generate_report":
            p.setdefault("project_id", state.project_id)

        return p

    @staticmethod
    def _deterministic_fallback_plan(state: AgentState) -> Decision:
        """Deterministic state-machine planner guaranteeing end-to-end completion."""
        # 1. Inspect Model
        if not state.is_model_inspected:
            return Decision(
                action="inspect_model",
                parameters={"model_path": state.model_path, "model_id": state.model_id},
                reason="Inspect model framework, input requirements, and parameter architecture first."
            )

        # 2. Inspect Dataset
        if not state.is_dataset_inspected:
            return Decision(
                action="inspect_dataset",
                parameters={"dataset_path": state.dataset_path, "dataset_id": state.dataset_id, "version": state.current_dataset_version, "target_col": state.target_column},
                reason="Inspect dataset features, missingness, duplicates, and initial class balance."
            )

        # 3. Validate Compatibility
        if not state.is_compatibility_validated:
            return Decision(
                action="validate_model_dataset",
                parameters={"model_path": state.model_path, "dataset_path": state.dataset_path, "model_id": state.model_id, "dataset_id": state.dataset_id, "version": state.current_dataset_version},
                reason="Verify feature shape and target compatibility between model and dataset."
            )

        # 4. Evaluate Baseline Performance
        if not state.is_baseline_evaluated:
            return Decision(
                action="evaluate_performance",
                parameters={"model_path": state.model_path, "dataset_path": state.dataset_path, "target_column": state.target_column, "task_type": state.task_type},
                reason="Establish baseline model performance metrics on original unperturbed dataset."
            )

        # 5. Evaluate Data Quality
        if not state.is_data_quality_evaluated:
            return Decision(
                action="evaluate_data_quality",
                parameters={"dataset_path": state.dataset_path, "target_column": state.target_column},
                reason="Perform formal data quality audit to detect issues needing transformation."
            )

        # 6. Evaluate Robustness
        if not state.is_robustness_evaluated:
            return Decision(
                action="evaluate_robustness",
                parameters={"model_path": state.model_path, "dataset_path": state.dataset_path, "target_column": state.target_column, "noise_level": 0.10},
                reason="Benchmark model robustness against input perturbations."
            )

        # 7. Check if problems require transformation
        dq = state.data_quality_metrics or {}
        
        # Check Missing Values
        if dq.get("missing_values_count", 0) > 0 and "handle_missing_values" not in state.transformations_applied:
            return Decision(
                action="handle_missing_values",
                parameters={"dataset_id": state.dataset_id, "current_version": state.current_dataset_version, "strategy": "median", "target_column": state.target_column},
                reason="Impute missing feature values detected in the data quality audit."
            )

        # Check Class Imbalance
        if dq.get("class_imbalance_detected", False) and "balance_data" not in state.transformations_applied:
            return Decision(
                action="balance_data",
                parameters={"dataset_id": state.dataset_id, "current_version": state.current_dataset_version, "method": "smote", "target_column": state.target_column},
                reason="Detected severe class imbalance. Applying SMOTE to rebalance classes and improve minority F1."
            )

        # 8. Re-evaluate if we applied transformations and haven't evaluated latest version
        if len(state.transformations_applied) > 0 and state.current_dataset_version != "v001" and state.latest_metrics == state.baseline_metrics:
            return Decision(
                action="evaluate_performance",
                parameters={"model_path": state.model_path, "dataset_path": state.dataset_path, "target_column": state.target_column, "task_type": state.task_type},
                reason="Re-evaluate model performance on newly transformed dataset version."
            )

        # 9. Compare results if experiment occurred
        if state.latest_metrics and state.latest_metrics != state.baseline_metrics and len(state.experiments) == 0:
            return Decision(
                action="save_experiment",
                parameters={
                    "project_id": state.project_id,
                    "model_id": state.model_id,
                    "operation": state.transformations_applied[-1] if state.transformations_applied else "transformation",
                    "parameters": {},
                    "input_dataset_version": "v001",
                    "output_dataset_version": state.current_dataset_version,
                    "metrics_before": state.baseline_metrics,
                    "metrics_after": state.latest_metrics,
                    "agent_reasoning": "Transformation was performed to resolve detected data quality issues."
                },
                reason="Persist experiment comparison and outcome to memory."
            )

        # 10. Finish and Generate Report
        return Decision(
            action="generate_report",
            parameters={"project_id": state.project_id, "title": f"Evaluation Report - {state.project_id}"},
            reason="User objective satisfied. Compiling final evaluation report.",
            status="complete"
        )

agent_planner = AgentPlanner()

