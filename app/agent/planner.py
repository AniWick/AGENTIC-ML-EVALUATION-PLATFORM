import json
from typing import Dict, Any

from app.agent.state import AgentState
from app.agent.tools import tool_registry
from app.llm.qwen_client import qwen_client
from app.llm.prompts import AGENT_SYSTEM_PROMPT


class Decision:
    def __init__(
        self,
        action: str,
        parameters: Dict[str, Any],
        reason: str,
        status: str = "continue"
    ):
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
    """
    State-gated planner.

    Qwen recommends the next action, but Python remains authoritative
    about which tools are legal.

    The planner also understands conversational information requests.
    """

    @staticmethod
    def decide_next_step(state: AgentState) -> Decision:
        context_str = state.get_summary_context()

        tool_defs = json.dumps(
            tool_registry.get_tool_definitions(),
            indent=2
        )

        prompt = f"""
CURRENT AGENT STATE
===================

{context_str}

USER OBJECTIVE
==============
{state.user_goal}

AVAILABLE TOOLS
===============
{tool_defs}

PLANNING RULES
==============

1. Use the CURRENT AGENT STATE as the primary source of context.

2. If the user is asking for information about the current model,
   use the existing model metadata if it is already available.

3. If model metadata is not available and the user is asking about
   the model, choose inspect_model.

4. Do NOT use search_project_history merely because the user asks
   a question about the current model.

5. search_project_history is only appropriate when the user explicitly
   asks about previous experiments, historical results, past decisions,
   or project history.

6. For the normal evaluation workflow, follow the state-gated sequence:
   inspect model -> inspect dataset -> validate compatibility ->
   baseline evaluation -> data quality -> robustness ->
   transformations when justified -> re-evaluation ->
   save experiment -> report.

7. Never repeat an inspection that has already been completed unless
   the user explicitly requests a fresh inspection.

8. Never apply a transformation unless the current state indicates
   that it is justified.

9. After transforming a dataset, performance must be re-evaluated
   before saving an experiment or generating the final report.

10. Respond with VALID JSON ONLY.

The JSON must have exactly this structure:

{{
    "action": "tool_name",
    "parameters": {{}},
    "reason": "short explanation"
}}
"""

        try:
            llm_decision = qwen_client.generate_json(
                prompt=prompt,
                system=AGENT_SYSTEM_PROMPT
            )
        except Exception:
            llm_decision = None

        if (
            isinstance(llm_decision, dict)
            and llm_decision.get("action") in tool_registry.tools
        ):
            action = llm_decision["action"]

            if AgentPlanner._is_action_legal(action, state):
                params = AgentPlanner._validate_and_fill_params(
                    action,
                    llm_decision.get("parameters", {}),
                    state
                )

                return Decision(
                    action=action,
                    parameters=params,
                    reason=llm_decision.get(
                        "reason",
                        "Qwen recommended the next step."
                    ),
                    status="continue"
                )

        return AgentPlanner._deterministic_fallback_plan(state)

    @staticmethod
    def _is_action_legal(
        action: str,
        state: AgentState
    ) -> bool:

        dq = state.data_quality_metrics or {}

        rules = {

            # ------------------------------------------------------
            # Inspection
            # ------------------------------------------------------

            "inspect_model": (
                not state.is_model_inspected
            ),

            "inspect_dataset": (
                state.is_model_inspected
                and not state.is_dataset_inspected
            ),

            # ------------------------------------------------------
            # Validation
            # ------------------------------------------------------

            "validate_model_dataset": (
                state.is_model_inspected
                and state.is_dataset_inspected
                and not state.is_compatibility_validated
            ),

            # ------------------------------------------------------
            # Evaluation
            # ------------------------------------------------------

            "evaluate_performance": (
                state.is_compatibility_validated
                and (
                    not state.is_baseline_evaluated
                    or state.evaluated_dataset_version
                    != state.current_dataset_version
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

            # ------------------------------------------------------
            # Transformations
            # ------------------------------------------------------

            "handle_missing_values": (
                state.is_robustness_evaluated
                and dq.get("missing_values_count", 0) > 0
                and "handle_missing_values"
                not in state.transformations_applied
            ),

            "balance_data": (
                state.is_robustness_evaluated
                and dq.get("class_imbalance_detected", False)
                and "balance_data"
                not in state.transformations_applied
            ),

            "handle_outliers": (
                state.is_robustness_evaluated
                and dq.get("outliers_count", 0) > 0
                and "handle_outliers"
                not in state.transformations_applied
            ),

            "scale_data": (
                state.is_robustness_evaluated
                and "scale_data"
                not in state.transformations_applied
                and AgentPlanner._scaling_is_needed(state)
            ),

            # Noise is a benchmarking transformation. It should not
            # automatically be applied as a normal cleaning step.
            "add_gaussian_noise": False,

            # ------------------------------------------------------
            # Experiment persistence
            # ------------------------------------------------------

            "save_experiment": (
                bool(state.transformations_applied)
                and state.latest_metrics is not None
                and (
                    state.evaluated_dataset_version
                    == state.current_dataset_version
                )
                and not state.experiments
            ),

            # ------------------------------------------------------
            # Final report
            # ------------------------------------------------------

            "generate_report": (
                state.is_baseline_evaluated
                and state.is_data_quality_evaluated
                and state.is_robustness_evaluated
                and (
                    state.evaluated_dataset_version
                    == state.current_dataset_version
                )
                and (
                    not state.transformations_applied
                    or bool(state.experiments)
                )
            ),
        }

        return rules.get(action, False)

    @staticmethod
    def _scaling_is_needed(state: AgentState) -> bool:
        """
        Scaling is intentionally conservative.

        If the model metadata explicitly indicates that scaling is
        required, allow scale_data. Otherwise don't invent the need
        for scaling.
        """

        metadata = state.model_metadata or {}

        if not isinstance(metadata, dict):
            return False

        scaling_required = metadata.get("scaling_required")

        if scaling_required is True:
            return True

        preprocessing = metadata.get("preprocessing")

        if isinstance(preprocessing, str):
            text = preprocessing.lower()
            return any(
                keyword in text
                for keyword in [
                    "standardscaler",
                    "minmaxscaler",
                    "normalization",
                    "scaling required",
                ]
            )

        return False

    @staticmethod
    def _validate_and_fill_params(
        action: str,
        params: Dict[str, Any],
        state: AgentState
    ) -> Dict[str, Any]:

        p = dict(params)

        # ----------------------------------------------------------
        # Model-related tools
        # ----------------------------------------------------------

        if action in [
            "inspect_model",
            "validate_model_dataset",
            "evaluate_performance",
            "evaluate_robustness",
        ]:
            p.setdefault(
                "model_path",
                state.model_path
            )
            p.setdefault(
                "model_id",
                state.model_id
            )

        # ----------------------------------------------------------
        # Dataset-related tools
        # ----------------------------------------------------------

        if action in [
            "inspect_dataset",
            "validate_model_dataset",
            "evaluate_performance",
            "evaluate_data_quality",
            "evaluate_robustness",
        ]:
            p.setdefault(
                "dataset_path",
                state.dataset_path
            )

            p.setdefault(
                "dataset_id",
                state.dataset_id
            )

            p.setdefault(
                "version",
                state.current_dataset_version
            )

            if state.target_column:
                p.setdefault(
                    "target_column",
                    state.target_column
                )

        # inspect_dataset uses target_col rather than target_column.
        if action == "inspect_dataset":
            if state.target_column:
                p.setdefault(
                    "target_col",
                    state.target_column
                )

        # ----------------------------------------------------------
        # Dataset transformations
        # ----------------------------------------------------------

        if action in [
            "handle_missing_values",
            "balance_data",
            "handle_outliers",
            "scale_data",
            "add_gaussian_noise",
        ]:
            p.setdefault(
                "dataset_id",
                state.dataset_id
            )

            p.setdefault(
                "current_version",
                state.current_dataset_version
            )

            if state.target_column:
                p.setdefault(
                    "target_column",
                    state.target_column
                )

        # ----------------------------------------------------------
        # Experiment persistence
        # ----------------------------------------------------------

        if action == "save_experiment":
            p.setdefault(
                "project_id",
                state.project_id
            )

            p.setdefault(
                "model_id",
                state.model_id
            )

            if not p.get("input_dataset_version"):
                p["input_dataset_version"] = "v001"

            if not p.get("output_dataset_version"):
                p["output_dataset_version"] = (
                    state.current_dataset_version
                )

            if not p.get("metrics_before"):
                p["metrics_before"] = (
                    state.baseline_metrics or {}
                )

            if not p.get("metrics_after"):
                p["metrics_after"] = (
                    state.latest_metrics or {}
                )

            if not p.get("parameters"):
                p["parameters"] = {}

            if not p.get("operation"):
                p["operation"] = ", ".join(
                    state.transformations_applied
                )

            if not p.get("agent_reasoning"):
                p["agent_reasoning"] = (
                    "Transformations were selected from the "
                    "data-quality evaluation and the resulting "
                    "dataset version was re-evaluated."
                )

        # ----------------------------------------------------------
        # Report
        # ----------------------------------------------------------

        if action == "generate_report":
            p.setdefault(
                "project_id",
                state.project_id
            )

            p.setdefault(
                "title",
                f"Evaluation Report - {state.project_id}"
            )

        return p

    @staticmethod
    def _deterministic_fallback_plan(
        state: AgentState
    ) -> Decision:

        # ----------------------------------------------------------
        # 1. Inspect model
        # ----------------------------------------------------------

        if not state.is_model_inspected:
            return Decision(
                "inspect_model",
                {
                    "model_path": state.model_path,
                    "model_id": state.model_id,
                },
                "Model metadata is not available, so inspect the model first."
            )

        # ----------------------------------------------------------
        # 2. Inspect dataset
        # ----------------------------------------------------------

        if not state.is_dataset_inspected:
            return Decision(
                "inspect_dataset",
                {
                    "dataset_path": state.dataset_path,
                    "dataset_id": state.dataset_id,
                    "version": state.current_dataset_version,
                    "target_col": state.target_column,
                },
                "Inspect the dataset and determine its structure and target."
            )

        # ----------------------------------------------------------
        # 3. Validate compatibility
        # ----------------------------------------------------------

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
                "Validate model and dataset compatibility."
            )

        # ----------------------------------------------------------
        # 4. Baseline performance
        # ----------------------------------------------------------

        if not state.is_baseline_evaluated:
            return Decision(
                "evaluate_performance",
                {
                    "model_path": state.model_path,
                    "dataset_path": state.dataset_path,
                    "target_column": state.target_column,
                    "task_type": state.task_type,
                },
                "Calculate deterministic baseline performance metrics."
            )

        # ----------------------------------------------------------
        # 5. Data quality
        # ----------------------------------------------------------

        if not state.is_data_quality_evaluated:
            return Decision(
                "evaluate_data_quality",
                {
                    "dataset_path": state.dataset_path,
                    "target_column": state.target_column,
                },
                "Audit missing values, imbalance, duplicates, outliers, "
                "and constant columns."
            )

        # ----------------------------------------------------------
        # 6. Robustness
        # ----------------------------------------------------------

        if not state.is_robustness_evaluated:
            return Decision(
                "evaluate_robustness",
                {
                    "model_path": state.model_path,
                    "dataset_path": state.dataset_path,
                    "target_column": state.target_column,
                    "noise_level": 0.10,
                },
                "Benchmark model robustness to Gaussian noise."
            )

        dq = state.data_quality_metrics or {}

        # ----------------------------------------------------------
        # 7. Missing values
        # ----------------------------------------------------------

        if (
            dq.get("missing_values_count", 0) > 0
            and "handle_missing_values"
            not in state.transformations_applied
        ):
            return Decision(
                "handle_missing_values",
                {
                    "dataset_id": state.dataset_id,
                    "current_version": (
                        state.current_dataset_version
                    ),
                    "strategy": "median",
                    "target_column": state.target_column,
                },
                "Create a new immutable dataset version with "
                "missing values imputed."
            )

        # ----------------------------------------------------------
        # 8. Class imbalance
        # ----------------------------------------------------------

        if (
            dq.get("class_imbalance_detected", False)
            and "balance_data"
            not in state.transformations_applied
        ):
            return Decision(
                "balance_data",
                {
                    "dataset_id": state.dataset_id,
                    "current_version": (
                        state.current_dataset_version
                    ),
                    "method": "smote",
                    "target_column": state.target_column,
                },
                "Create a new dataset version with class imbalance "
                "addressed using SMOTE."
            )

        # ----------------------------------------------------------
        # 9. Outliers
        # ----------------------------------------------------------

        if (
            dq.get("outliers_count", 0) > 0
            and "handle_outliers"
            not in state.transformations_applied
        ):
            return Decision(
                "handle_outliers",
                {
                    "dataset_id": state.dataset_id,
                    "current_version": (
                        state.current_dataset_version
                    ),
                    "method": "clip",
                    "target_column": state.target_column,
                },
                "Handle detected numerical outliers by clipping "
                "and create a new immutable dataset version."
            )

        # ----------------------------------------------------------
        # 10. Scaling
        # ----------------------------------------------------------

        if (
            "scale_data"
            not in state.transformations_applied
            and AgentPlanner._scaling_is_needed(state)
        ):
            return Decision(
                "scale_data",
                {
                    "dataset_id": state.dataset_id,
                    "current_version": (
                        state.current_dataset_version
                    ),
                    "method": "standard",
                    "target_column": state.target_column,
                },
                "The model metadata indicates that feature scaling "
                "is required."
            )

        # ----------------------------------------------------------
        # 11. Re-evaluate transformed dataset
        # ----------------------------------------------------------

        if (
            state.evaluated_dataset_version
            != state.current_dataset_version
        ):
            return Decision(
                "evaluate_performance",
                {
                    "model_path": state.model_path,
                    "dataset_path": state.dataset_path,
                    "target_column": state.target_column,
                    "task_type": state.task_type,
                },
                (
                    f"Re-evaluate model performance on dataset "
                    f"version {state.current_dataset_version}."
                )
            )

        # ----------------------------------------------------------
        # 12. Save experiment
        # ----------------------------------------------------------

        if (
            state.transformations_applied
            and not state.experiments
        ):
            return Decision(
                "save_experiment",
                {
                    "project_id": state.project_id,
                    "model_id": state.model_id,
                    "operation": ", ".join(
                        state.transformations_applied
                    ),
                    "parameters": {},
                    "input_dataset_version": "v001",
                    "output_dataset_version": (
                        state.current_dataset_version
                    ),
                    "metrics_before": (
                        state.baseline_metrics or {}
                    ),
                    "metrics_after": (
                        state.latest_metrics or {}
                    ),
                    "agent_reasoning": (
                        "Transformations were selected from the "
                        "data-quality audit and the resulting "
                        "dataset version was re-evaluated."
                    ),
                },
                "Persist the experiment and exact metric deltas."
            )

        # ----------------------------------------------------------
        # 13. Generate report
        # ----------------------------------------------------------

        return Decision(
            "generate_report",
            {
                "project_id": state.project_id,
                "title": (
                    f"Evaluation Report - {state.project_id}"
                ),
            },
            (
                "All required evaluation and improvement steps "
                "are complete; generate the final report."
            ),
            status="complete"
        )


agent_planner = AgentPlanner()
