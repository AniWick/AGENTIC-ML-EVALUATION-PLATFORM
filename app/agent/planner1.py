import json
from typing import Dict, Any
from app.agent.state import AgentState
from app.agent.tools import tool_registry
from app.llm.ollama_client import ollama_client
from app.llm.prompts import AGENT_SYSTEM_PROMPT


class Decision:
    def __init__(self, action: str, parameters: Dict[str, Any],
                 reason: str, status: str = "continue"):
        self.action = action
        self.parameters = parameters
        self.reason = reason
        self.status = status

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "action": self.action,
            "parameters": self.parameters,
            "reason": self.reason,
        }


class AgentPlanner:
    """State-gated planner.

    The LLM can recommend an action, but it cannot bypass workflow
    prerequisites or repeatedly execute a completed action.
    """

    @staticmethod
    def decide_next_step(state: AgentState) -> Decision:
        context_str = state.get_summary_context()
        tool_defs = json.dumps(tool_registry.get_tool_definitions(), indent=2)

        prompt = (
            f"CURRENT STATE CONTEXT:\n{context_str}\n\n"
            f"AVAILABLE TOOLS:\n{tool_defs}\n\n"
            f"USER OBJECTIVE: {state.user_goal}\n\n"
            "Determine the single next best action. Respond with valid JSON only."
        )

        llm_decision = ollama_client.generate_json(
            prompt=prompt,
            system=AGENT_SYSTEM_PROMPT
        )

        # FIX: the old code trusted any valid LLM action. If Ollama kept
        # returning inspect_model, the same tool ran until max_iterations.
        # The LLM is now advisory; state-machine legality is authoritative.
        if llm_decision and llm_decision.get("action") in tool_registry.tools:
            action = llm_decision["action"]
            if AgentPlanner._is_action_legal(action, state):
                params = AgentPlanner._validate_and_fill_params(
                    action, llm_decision.get("parameters", {}), state
                )
                return Decision(
                    action=action,
                    parameters=params,
                    reason=llm_decision.get("reason", "LLM recommended next step."),
                    status=llm_decision.get("status", "continue")
                )

        return AgentPlanner._deterministic_fallback_plan(state)

    @staticmethod
    def _is_action_legal(action: str, state: AgentState) -> bool:
        rules = {
            "inspect_model": not state.is_model_inspected,

            "inspect_dataset": (
                state.is_model_inspected and not state.is_dataset_inspected
            ),

            "validate_model_dataset": (
                state.is_model_inspected
                and state.is_dataset_inspected
                and not state.is_compatibility_validated
            ),

            "evaluate_performance": (
                state.is_compatibility_validated
                and (
                    not state.is_baseline_evaluated
                    or state.evaluated_dataset_version != state.current_dataset_version
                )
            ),

            "evaluate_data_quality": (
                state.is_baseline_evaluated
                and not state.is_data_quality_evaluated
            ),

            "evaluate_robustness": (
                state.is_data_quality_evaluated
                and not state.is_robustness_evaluated
            ),

            "handle_missing_values": (
                state.is_robustness_evaluated
                and (state.data_quality_metrics or {}).get("missing_values_count", 0) > 0
                and "handle_missing_values" not in state.transformations_applied
            ),

            "balance_data": (
                state.is_robustness_evaluated
                and (state.data_quality_metrics or {}).get(
                    "class_imbalance_detected", False
                )
                and "balance_data" not in state.transformations_applied
            ),

            "save_experiment": (
                bool(state.transformations_applied)
                and state.latest_metrics is not None
                and state.evaluated_dataset_version == state.current_dataset_version
                and not state.experiments
            ),

            "generate_report": (
                state.is_baseline_evaluated
                and state.is_data_quality_evaluated
                and state.is_robustness_evaluated
                and state.evaluated_dataset_version == state.current_dataset_version
                and (
                    not state.transformations_applied or bool(state.experiments)
                )
            ),
        }
        return rules.get(action, False)

    @staticmethod
    def _validate_and_fill_params(action: str, params: Dict[str, Any],
                                  state: AgentState) -> Dict[str, Any]:
        p = dict(params)

        if action in [
            "inspect_model", "validate_model_dataset",
            "evaluate_performance", "evaluate_robustness"
        ]:
            p.setdefault("model_path", state.model_path)
            p.setdefault("model_id", state.model_id)

        if action in [
            "inspect_dataset", "validate_model_dataset",
            "evaluate_performance", "evaluate_data_quality",
            "evaluate_robustness"
        ]:
            p.setdefault("dataset_path", state.dataset_path)
            p.setdefault("dataset_id", state.dataset_id)
            p.setdefault("version", state.current_dataset_version)
            if state.target_column:
                p.setdefault("target_column", state.target_column)

        if action in [
            "handle_missing_values", "balance_data",
            "handle_outliers", "scale_data", "add_gaussian_noise"
        ]:
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
        if not state.is_model_inspected:
            return Decision(
                "inspect_model",
                {"model_path": state.model_path, "model_id": state.model_id},
                "Inspect model first."
            )

        if not state.is_dataset_inspected:
            return Decision(
                "inspect_dataset",
                {
                    "dataset_path": state.dataset_path,
                    "dataset_id": state.dataset_id,
                    "version": state.current_dataset_version,
                    "target_col": state.target_column,
                },
                "Inspect dataset and determine its structure and target."
            )

        if not state.is_compatibility_validated:
            return Decision(
                "validate_model_dataset",
                {
                    "model_path": state.model_path,
                    "dataset_path": state.dataset_path,
                    "model_id": state.model_id,
                    "dataset_id": state.dataset_id,
                    "version": state.current_dataset_version,
                },
                "Validate model/dataset compatibility."
            )

        if not state.is_baseline_evaluated:
            return Decision(
                "evaluate_performance",
                {
                    "model_path": state.model_path,
                    "dataset_path": state.dataset_path,
                    "target_column": state.target_column,
                    "task_type": state.task_type,
                },
                "Calculate deterministic baseline metrics."
            )

        if not state.is_data_quality_evaluated:
            return Decision(
                "evaluate_data_quality",
                {
                    "dataset_path": state.dataset_path,
                    "target_column": state.target_column,
                },
                "Audit missing values, imbalance, duplicates and outliers."
            )

        if not state.is_robustness_evaluated:
            return Decision(
                "evaluate_robustness",
                {
                    "model_path": state.model_path,
                    "dataset_path": state.dataset_path,
                    "target_column": state.target_column,
                    "noise_level": 0.10,
                },
                "Benchmark robustness to Gaussian noise."
            )

        dq = state.data_quality_metrics or {}

        if (
            dq.get("missing_values_count", 0) > 0
            and "handle_missing_values" not in state.transformations_applied
        ):
            return Decision(
                "handle_missing_values",
                {
                    "dataset_id": state.dataset_id,
                    "current_version": state.current_dataset_version,
                    "strategy": "median",
                    "target_column": state.target_column,
                },
                "Create a new version with missing values imputed."
            )

        if (
            dq.get("class_imbalance_detected", False)
            and "balance_data" not in state.transformations_applied
        ):
            return Decision(
                "balance_data",
                {
                    "dataset_id": state.dataset_id,
                    "current_version": state.current_dataset_version,
                    "method": "smote",
                    "target_column": state.target_column,
                },
                "Create a new version with class imbalance addressed using SMOTE."
            )

        # FIX: compare versions, not dictionaries. Every newly transformed
        # version must receive fresh performance metrics.
        if state.evaluated_dataset_version != state.current_dataset_version:
            return Decision(
                "evaluate_performance",
                {
                    "model_path": state.model_path,
                    "dataset_path": state.dataset_path,
                    "target_column": state.target_column,
                    "task_type": state.task_type,
                },
                f"Re-evaluate performance on {state.current_dataset_version}."
            )

        if state.transformations_applied and not state.experiments:
            return Decision(
                "save_experiment",
                {
                    "project_id": state.project_id,
                    "model_id": state.model_id,
                    "operation": ", ".join(state.transformations_applied),
                    "parameters": {},
                    "input_dataset_version": "v001",
                    "output_dataset_version": state.current_dataset_version,
                    "metrics_before": state.baseline_metrics or {},
                    "metrics_after": state.latest_metrics or {},
                    "agent_reasoning": (
                        "Transformations were selected from the data-quality audit "
                        "and the resulting dataset version was re-evaluated."
                    ),
                },
                "Persist the experiment and exact metric deltas."
            )

        return Decision(
            "generate_report",
            {
                "project_id": state.project_id,
                "title": f"Evaluation Report - {state.project_id}",
            },
            "All evaluation and improvement steps are complete; generate the final report.",
            status="complete"
        )


agent_planner = AgentPlanner()
