import json
from typing import Dict, Any, Optional

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
        status: str = "continue",
    ):
        self.action = action
        self.parameters = parameters
        self.reason = reason
        self.status = status

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "action": self.action if self.action else None,
            "parameters": self.parameters,
            "reason": self.reason,
        }


class AgentPlanner:
    """
    Workflow-aware state-gated planner.

    Qwen recommends the next action, but Python is authoritative.

    The requested_workflow determines the maximum scope of the workflow.
    Narrow requests therefore cannot accidentally expand into the full
    evaluation pipeline.
    """

    # ------------------------------------------------------------------
    # Main planner
    # ------------------------------------------------------------------

    @staticmethod
    def decide_next_step(state: AgentState) -> Decision:

        # Python is always authoritative about completion.
        if state.workflow_complete():
            return Decision(
                action="",
                parameters={},
                reason=(
                    f"The requested workflow "
                    f"'{state.requested_workflow}' is complete."
                ),
                status="complete",
            )

        context_str = state.get_summary_context()

        tool_defs = json.dumps(
            tool_registry.get_tool_definitions(),
            indent=2,
            default=str,
        )

        prompt = f"""
CURRENT AGENT STATE
===================

{context_str}

USER OBJECTIVE
==============

{state.user_goal}

REQUESTED WORKFLOW
==================

{state.requested_workflow}

WORKFLOW OBJECTIVE
==================

{state.workflow_objective}

AVAILABLE TOOLS
===============

{tool_defs}

IMPORTANT PLANNING RULES
========================

1. Follow requested_workflow exactly.

2. NEVER expand a narrow workflow into full_evaluation.

3. Workflow boundaries:

   inspection:
       inspect_model
       inspect_dataset
       validate_model_dataset
       STOP

   baseline_metrics:
       inspect_model if needed
       inspect_dataset if needed
       validate_model_dataset if needed
       evaluate_performance
       STOP

   targeted_transform:
       prerequisites if required
       ONLY the requested transformation
       STOP

   improve_dataset:
       inspection
       validation
       baseline
       data quality
       robustness
       justified transformations
       reevaluation
       comparison
       experiment storage
       STOP

   full_evaluation:
       inspection
       validation
       baseline
       data quality
       robustness
       justified transformations
       reevaluation
       comparison
       experiment storage
       report

   report:
       obtain required evaluation information
       generate report
       STOP

4. Python state is authoritative.

5. Never select an action that is illegal for the current state.

6. Never repeat a successfully completed action unnecessarily.

7. Never apply a transformation automatically unless:
   - it is explicitly requested by a targeted_transform workflow, OR
   - deterministic evaluation evidence justifies it in an improvement/full
     evaluation workflow.

8. After a transformation creates a new dataset version, performance must
   be reevaluated before comparison, experiment storage, or final reporting.

9. A narrow workflow must stop immediately after its objective is satisfied.

10. Never generate a report for inspection, baseline_metrics, or
    targeted_transform unless the workflow explicitly requires it.

11. Never use search_project_history for a deterministic workflow unless
    explicitly required.

12. Numerical metrics must come only from deterministic Python tools.

13. If the requested workflow is complete, return:

{{
    "status": "complete",
    "action": null,
    "parameters": {{}},
    "reason": "The requested workflow is complete."
}}

14. Otherwise return:

{{
    "status": "continue",
    "action": "<tool_name>",
    "parameters": {{}},
    "reason": "<short explanation>"
}}

Respond with VALID JSON ONLY.
"""

        try:
            llm_decision = qwen_client.generate_json(
                prompt=prompt,
                system=AGENT_SYSTEM_PROMPT,
            )
        except Exception:
            llm_decision = None

        # --------------------------------------------------------------
        # Accept Qwen only when Python confirms the action is legal.
        # --------------------------------------------------------------

        if isinstance(llm_decision, dict):

            llm_status = llm_decision.get(
                "status",
                "continue",
            )

            action = llm_decision.get("action")

            # Qwen is not allowed to declare completion unless Python
            # independently confirms the workflow is complete.
            if llm_status == "complete":

                if state.workflow_complete():
                    return Decision(
                        action="",
                        parameters={},
                        reason=llm_decision.get(
                            "reason",
                            "The requested workflow is complete.",
                        ),
                        status="complete",
                    )

                return AgentPlanner._deterministic_fallback_plan(state)

            if (
                action
                and action in tool_registry.tools
                and AgentPlanner._is_action_legal(
                    action,
                    state,
                )
            ):
                params = AgentPlanner._validate_and_fill_params(
                    action,
                    llm_decision.get("parameters", {}),
                    state,
                )

                return Decision(
                    action=action,
                    parameters=params,
                    reason=llm_decision.get(
                        "reason",
                        "Qwen selected the next legal workflow step.",
                    ),
                    status="continue",
                )

        # --------------------------------------------------------------
        # Deterministic fallback
        # --------------------------------------------------------------

        return AgentPlanner._deterministic_fallback_plan(state)

    # ------------------------------------------------------------------
    # Workflow legality
    # ------------------------------------------------------------------

    @staticmethod
    def _is_action_legal(
        action: str,
        state: AgentState,
    ) -> bool:

        workflow = state.requested_workflow

        # --------------------------------------------------------------
        # INSPECTION
        # --------------------------------------------------------------

        if workflow == "inspection":

            if action == "inspect_model":
                return not state.model_inspected

            if action == "inspect_dataset":
                return (
                    state.model_inspected
                    and not state.dataset_inspected
                )

            if action == "validate_model_dataset":
                return (
                    state.model_inspected
                    and state.dataset_inspected
                    and not state.compatibility_validated
                )

            return False

        # --------------------------------------------------------------
        # BASELINE METRICS
        # --------------------------------------------------------------

        if workflow == "baseline_metrics":

            if action == "inspect_model":
                return not state.model_inspected

            if action == "inspect_dataset":
                return (
                    state.model_inspected
                    and not state.dataset_inspected
                )

            if action == "validate_model_dataset":
                return (
                    state.model_inspected
                    and state.dataset_inspected
                    and not state.compatibility_validated
                )

            if action == "evaluate_performance":
                return (
                    state.compatibility_validated
                    and not state.baseline_evaluated
                )

            return False

        # --------------------------------------------------------------
        # TARGETED TRANSFORMATION
        # --------------------------------------------------------------

        if workflow == "targeted_transform":

            target_action = AgentPlanner._get_target_action(state)

            if action == "inspect_model":
                return not state.model_inspected

            if action == "inspect_dataset":
                return (
                    state.model_inspected
                    and not state.dataset_inspected
                )

            if action == "validate_model_dataset":
                return (
                    state.model_inspected
                    and state.dataset_inspected
                    and not state.compatibility_validated
                )

            if action in {
                "handle_missing_values",
                "balance_data",
                "handle_outliers",
                "scale_data",
                "add_gaussian_noise",
            }:

                if not state.compatibility_validated:
                    return False

                if not target_action:
                    return False

                if action != target_action:
                    return False

                if action in AgentPlanner._applied_operation_names(state):
                    return False

                return True

            return False

        # --------------------------------------------------------------
        # IMPROVE DATASET
        # --------------------------------------------------------------

        if workflow == "improve_dataset":

            if action == "inspect_model":
                return not state.model_inspected

            if action == "inspect_dataset":
                return (
                    state.model_inspected
                    and not state.dataset_inspected
                )

            if action == "validate_model_dataset":
                return (
                    state.model_inspected
                    and state.dataset_inspected
                    and not state.compatibility_validated
                )

            if action == "evaluate_performance":
                return (
                    state.compatibility_validated
                    and (
                        not state.baseline_evaluated
                        or state.evaluated_dataset_version
                        != state.current_dataset_version
                    )
                )

            if action == "evaluate_data_quality":
                return (
                    state.baseline_evaluated
                    and not state.data_quality_evaluated
                )

            if action == "evaluate_robustness":
                return (
                    state.data_quality_evaluated
                    and not state.robustness_evaluated
                )

            if action in {
                "handle_missing_values",
                "balance_data",
                "handle_outliers",
                "scale_data",
            }:
                return AgentPlanner._transformation_is_justified(
                    action,
                    state,
                )

            if action == "add_gaussian_noise":
                return False

            if action == "compare_results":
                return (
                    bool(state.transformations_applied)
                    and state.latest_metrics is not None
                    and (
                        state.evaluated_dataset_version
                        == state.current_dataset_version
                    )
                    and not AgentPlanner._action_was_taken(
                        state,
                        "compare_results",
                    )
                )

            if action == "save_experiment":
                return (
                    bool(state.transformations_applied)
                    and state.latest_metrics is not None
                    and (
                        state.evaluated_dataset_version
                        == state.current_dataset_version
                    )
                    and AgentPlanner._action_was_taken(
                        state,
                        "compare_results",
                    )
                    and not state.experiments
                )

            return False

        # --------------------------------------------------------------
        # FULL EVALUATION
        # --------------------------------------------------------------

        if workflow == "full_evaluation":

            if action == "inspect_model":
                return not state.model_inspected

            if action == "inspect_dataset":
                return (
                    state.model_inspected
                    and not state.dataset_inspected
                )

            if action == "validate_model_dataset":
                return (
                    state.model_inspected
                    and state.dataset_inspected
                    and not state.compatibility_validated
                )

            if action == "evaluate_performance":
                return (
                    state.compatibility_validated
                    and (
                        not state.baseline_evaluated
                        or state.evaluated_dataset_version
                        != state.current_dataset_version
                    )
                )

            if action == "evaluate_data_quality":
                return (
                    state.baseline_evaluated
                    and not state.data_quality_evaluated
                )

            if action == "evaluate_robustness":
                return (
                    state.data_quality_evaluated
                    and not state.robustness_evaluated
                )

            if action in {
                "handle_missing_values",
                "balance_data",
                "handle_outliers",
                "scale_data",
            }:
                return AgentPlanner._transformation_is_justified(
                    action,
                    state,
                )

            # Noise is a robustness-testing operation, not an automatic
            # dataset-improvement operation.
            if action == "add_gaussian_noise":
                return False

            if action == "compare_results":
                return (
                    bool(state.transformations_applied)
                    and state.latest_metrics is not None
                    and (
                        state.evaluated_dataset_version
                        == state.current_dataset_version
                    )
                    and not AgentPlanner._action_was_taken(
                        state,
                        "compare_results",
                    )
                )

            if action == "save_experiment":
                return (
                    bool(state.transformations_applied)
                    and state.latest_metrics is not None
                    and (
                        state.evaluated_dataset_version
                        == state.current_dataset_version
                    )
                    and AgentPlanner._action_was_taken(
                        state,
                        "compare_results",
                    )
                    and not state.experiments
                )

            if action == "generate_report":
                return (
                    state.baseline_evaluated
                    and state.data_quality_evaluated
                    and state.robustness_evaluated
                    and (
                        state.evaluated_dataset_version
                        == state.current_dataset_version
                    )
                    and (
                        not state.transformations_applied
                        or bool(state.experiments)
                    )
                )

            return False

        # --------------------------------------------------------------
        # REPORT
        # --------------------------------------------------------------

        if workflow == "report":

            if action == "inspect_model":
                return not state.model_inspected

            if action == "inspect_dataset":
                return (
                    state.model_inspected
                    and not state.dataset_inspected
                )

            if action == "validate_model_dataset":
                return (
                    state.model_inspected
                    and state.dataset_inspected
                    and not state.compatibility_validated
                )

            if action == "evaluate_performance":
                return (
                    state.compatibility_validated
                    and not state.baseline_evaluated
                )

            if action == "evaluate_data_quality":
                return (
                    state.baseline_evaluated
                    and not state.data_quality_evaluated
                )

            if action == "evaluate_robustness":
                return (
                    state.data_quality_evaluated
                    and not state.robustness_evaluated
                )

            if action == "generate_report":
                return (
                    state.baseline_evaluated
                    and state.data_quality_evaluated
                    and state.robustness_evaluated
                    and (
                        state.evaluated_dataset_version
                        == state.current_dataset_version
                    )
                )

            return False

        # Contextual and comparison workflows do not execute tools.
        if workflow in {
            "contextual",
            "comparison",
        }:
            return False

        return False

    # ------------------------------------------------------------------
    # Transformation justification
    # ------------------------------------------------------------------

    @staticmethod
    def _transformation_is_justified(
        action: str,
        state: AgentState,
    ) -> bool:

        if not state.compatibility_validated:
            return False

        if not state.data_quality_evaluated:
            return False

        if action in AgentPlanner._applied_operation_names(state):
            return False

        dq = state.data_quality or {}

        if not isinstance(dq, dict):
            return False

        if action == "handle_missing_values":
            return (
                dq.get("missing_values_count", 0) > 0
                or dq.get("missing_values", 0) > 0
                or bool(
                    dq.get(
                        "missing_values_detected",
                        False,
                    )
                )
            )

        if action == "balance_data":
            return bool(
                dq.get(
                    "class_imbalance_detected",
                    False,
                )
                or dq.get(
                    "imbalance_detected",
                    False,
                )
                or dq.get(
                    "is_imbalanced",
                    False,
                )
            )

        if action == "handle_outliers":
            return (
                dq.get("outliers_count", 0) > 0
                or dq.get("outlier_count", 0) > 0
                or bool(
                    dq.get(
                        "outliers_detected",
                        False,
                    )
                )
            )

        if action == "scale_data":
            return AgentPlanner._scaling_is_needed(state)

        return False

    # ------------------------------------------------------------------
    # Target action extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _get_target_action(
        state: AgentState,
    ) -> Optional[str]:

        objective = (
            state.workflow_objective
            or state.user_goal
            or ""
        ).lower()

        mapping = {
            "handle_missing_values": [
                "handle_missing_values",
                "missing values",
                "missing data",
                "clean missing",
                "clean the missing",
            ],
            "balance_data": [
                "balance_data",
                "class imbalance",
                "balance the dataset",
                "balance dataset",
                "balance data",
                "balance classes",
                "smote",
            ],
            "handle_outliers": [
                "handle_outliers",
                "outliers",
                "remove outliers",
                "handle outlier",
            ],
            "scale_data": [
                "scale_data",
                "scale the dataset",
                "scale dataset",
                "scale data",
                "normalize data",
                "normalize the dataset",
                "feature scaling",
            ],
            "add_gaussian_noise": [
                "add_gaussian_noise",
                "gaussian noise",
                "add noise",
                "noise augmentation",
            ],
        }

        for action, phrases in mapping.items():
            if any(
                phrase in objective
                for phrase in phrases
            ):
                return action

        return None

    # ------------------------------------------------------------------
    # Applied transformation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _applied_operation_names(
        state: AgentState,
    ) -> set:
        """
        transformations_applied is currently a list of dictionaries.

        Legacy string values are also tolerated.
        """

        names = set()

        for item in state.transformations_applied:

            if isinstance(item, str):
                names.add(item)
                continue

            if isinstance(item, dict):

                operation = item.get("operation")

                if operation:
                    names.add(operation)

                action = item.get("action")

                if action:
                    names.add(action)

        return names

    # ------------------------------------------------------------------
    # Action history
    # ------------------------------------------------------------------

    @staticmethod
    def _action_was_taken(
        state: AgentState,
        action: str,
    ) -> bool:

        for item in state.actions_taken:

            if isinstance(item, dict):

                if item.get("action") == action:
                    return True

            elif isinstance(item, str):

                if item == action:
                    return True

        return False

    # ------------------------------------------------------------------
    # Scaling
    # ------------------------------------------------------------------

    @staticmethod
    def _scaling_is_needed(
        state: AgentState,
    ) -> bool:

        metadata = state.model_metadata or {}

        if not isinstance(metadata, dict):
            return False

        if metadata.get("scaling_required") is True:
            return True

        preprocessing = metadata.get(
            "preprocessing"
        )

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

    # ------------------------------------------------------------------
    # Parameter validation / defaults
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_and_fill_params(
        action: str,
        params: Dict[str, Any],
        state: AgentState,
    ) -> Dict[str, Any]:

        p = dict(params or {})

        # --------------------------------------------------------------
        # Model tools
        # --------------------------------------------------------------

        if action in {
            "inspect_model",
            "validate_model_dataset",
            "evaluate_performance",
            "evaluate_robustness",
        }:

            p.setdefault(
                "model_path",
                state.model_path,
            )

            p.setdefault(
                "model_id",
                state.model_id,
            )

        # --------------------------------------------------------------
        # Dataset tools
        # --------------------------------------------------------------

        if action in {
            "inspect_dataset",
            "validate_model_dataset",
            "evaluate_performance",
            "evaluate_data_quality",
            "evaluate_robustness",
        }:

            p.setdefault(
                "dataset_path",
                state.dataset_path,
            )

            p.setdefault(
                "dataset_id",
                state.dataset_id,
            )

            p.setdefault(
                "version",
                state.current_dataset_version,
            )

            if state.target_column:
                p.setdefault(
                    "target_column",
                    state.target_column,
                )

        # inspect_dataset uses target_col.
        if action == "inspect_dataset":

            if state.target_column:
                p.setdefault(
                    "target_col",
                    state.target_column,
                )

        # --------------------------------------------------------------
        # Transformations
        # --------------------------------------------------------------

        if action in {
            "handle_missing_values",
            "balance_data",
            "handle_outliers",
            "scale_data",
            "add_gaussian_noise",
        }:

            p.setdefault(
                "dataset_id",
                state.dataset_id,
            )

            p.setdefault(
                "current_version",
                state.current_dataset_version,
            )

            if state.target_column:
                p.setdefault(
                    "target_column",
                    state.target_column,
                )

        # --------------------------------------------------------------
        # Experiment
        # --------------------------------------------------------------

        if action == "save_experiment":

            p.setdefault(
                "project_id",
                state.project_id,
            )

            p.setdefault(
                "model_id",
                state.model_id,
            )

            if not p.get("input_dataset_version"):

                input_version = (
                    AgentPlanner._first_transformation_input_version(
                        state
                    )
                )

                p["input_dataset_version"] = (
                    input_version or "v001"
                )

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

                operation_names = (
                    AgentPlanner._applied_operation_names(
                        state
                    )
                )

                p["operation"] = ", ".join(
                    sorted(operation_names)
                )

            if not p.get("agent_reasoning"):
                p["agent_reasoning"] = (
                    "Transformations were selected based on "
                    "deterministic dataset-quality evidence and "
                    "the resulting dataset version was reevaluated."
                )

        # --------------------------------------------------------------
        # Comparison
        # --------------------------------------------------------------

        if action == "compare_results":

            p.setdefault(
                "project_id",
                state.project_id,
            )

            p.setdefault(
                "model_id",
                state.model_id,
            )

            if not p.get("input_dataset_version"):

                input_version = (
                    AgentPlanner._first_transformation_input_version(
                        state
                    )
                )

                p["input_dataset_version"] = (
                    input_version or "v001"
                )

            p.setdefault(
                "output_dataset_version",
                state.current_dataset_version,
            )

            p.setdefault(
                "metrics_before",
                state.baseline_metrics or {},
            )

            p.setdefault(
                "metrics_after",
                state.latest_metrics or {},
            )

        # --------------------------------------------------------------
        # Report
        # --------------------------------------------------------------

        if action == "generate_report":

            p.setdefault(
                "project_id",
                state.project_id,
            )

            p.setdefault(
                "title",
                f"Evaluation Report - {state.project_id}",
            )

        return p

    # ------------------------------------------------------------------
    # First transformation input version
    # ------------------------------------------------------------------

    @staticmethod
    def _first_transformation_input_version(
        state: AgentState,
    ) -> Optional[str]:

        for item in state.transformations_applied:

            if isinstance(item, dict):

                version = item.get(
                    "input_version"
                )

                if version:
                    return version

        return None

    # ------------------------------------------------------------------
    # Deterministic fallback planner
    # ------------------------------------------------------------------

    @staticmethod
    def _deterministic_fallback_plan(
        state: AgentState,
    ) -> Decision:

        workflow = state.requested_workflow

        # ==============================================================
        # INSPECTION
        # ==============================================================

        if workflow == "inspection":

            if not state.model_inspected:
                return Decision(
                    "inspect_model",
                    {
                        "model_path": state.model_path,
                        "model_id": state.model_id,
                    },
                    "Inspect the model as requested.",
                )

            if not state.dataset_inspected:
                return Decision(
                    "inspect_dataset",
                    {
                        "dataset_path": state.dataset_path,
                        "dataset_id": state.dataset_id,
                        "version": state.current_dataset_version,
                        "target_col": state.target_column,
                    },
                    "Inspect the current dataset version.",
                )

            if not state.compatibility_validated:
                return Decision(
                    "validate_model_dataset",
                    {
                        "model_path": state.model_path,
                        "dataset_path": state.dataset_path,
                        "model_id": state.model_id,
                        "dataset_id": state.dataset_id,
                        "version": state.current_dataset_version,
                    },
                    "Validate model and dataset compatibility.",
                )

            return Decision(
                "",
                {},
                "Inspection workflow is complete.",
                status="complete",
            )

        # ==============================================================
        # BASELINE METRICS
        # ==============================================================

        if workflow == "baseline_metrics":

            if not state.model_inspected:
                return Decision(
                    "inspect_model",
                    {
                        "model_path": state.model_path,
                        "model_id": state.model_id,
                    },
                    "Inspect the model before obtaining baseline metrics.",
                )

            if not state.dataset_inspected:
                return Decision(
                    "inspect_dataset",
                    {
                        "dataset_path": state.dataset_path,
                        "dataset_id": state.dataset_id,
                        "version": state.current_dataset_version,
                        "target_col": state.target_column,
                    },
                    "Inspect the dataset before baseline evaluation.",
                )

            if not state.compatibility_validated:
                return Decision(
                    "validate_model_dataset",
                    {
                        "model_path": state.model_path,
                        "dataset_path": state.dataset_path,
                        "model_id": state.model_id,
                        "dataset_id": state.dataset_id,
                        "version": state.current_dataset_version,
                    },
                    "Validate compatibility before evaluating baseline performance.",
                )

            if not state.baseline_evaluated:
                return Decision(
                    "evaluate_performance",
                    {
                        "model_path": state.model_path,
                        "dataset_path": state.dataset_path,
                        "target_column": state.target_column,
                        "task_type": state.task_type,
                    },
                    "Calculate deterministic baseline performance metrics.",
                )

            return Decision(
                "",
                {},
                "Baseline metrics workflow is complete.",
                status="complete",
            )

        # ==============================================================
        # TARGETED TRANSFORMATION
        # ==============================================================

        if workflow == "targeted_transform":

            target_action = AgentPlanner._get_target_action(
                state
            )

            if not target_action:
                return Decision(
                    "",
                    {},
                    "No specific transformation was identified.",
                    status="complete",
                )

            if not state.model_inspected:
                return Decision(
                    "inspect_model",
                    {
                        "model_path": state.model_path,
                        "model_id": state.model_id,
                    },
                    "Inspect the model before applying the requested transformation.",
                )

            if not state.dataset_inspected:
                return Decision(
                    "inspect_dataset",
                    {
                        "dataset_path": state.dataset_path,
                        "dataset_id": state.dataset_id,
                        "version": state.current_dataset_version,
                        "target_col": state.target_column,
                    },
                    "Inspect the dataset before applying the requested transformation.",
                )

            if not state.compatibility_validated:
                return Decision(
                    "validate_model_dataset",
                    {
                        "model_path": state.model_path,
                        "dataset_path": state.dataset_path,
                        "model_id": state.model_id,
                        "dataset_id": state.dataset_id,
                        "version": state.current_dataset_version,
                    },
                    "Validate compatibility before transforming the dataset.",
                )

            if target_action in AgentPlanner._applied_operation_names(
                state
            ):
                return Decision(
                    "",
                    {},
                    "The requested transformation has already been applied.",
                    status="complete",
                )

            params = {
                "dataset_id": state.dataset_id,
                "current_version": state.current_dataset_version,
            }

            if state.target_column:
                params["target_column"] = state.target_column

            if target_action == "handle_missing_values":
                params["strategy"] = "median"

            elif target_action == "balance_data":
                params["method"] = "smote"

            elif target_action == "handle_outliers":
                params["method"] = "clip"

            elif target_action == "scale_data":
                params["method"] = "standard"

            elif target_action == "add_gaussian_noise":
                params["noise_level"] = 0.10

            return Decision(
                target_action,
                params,
                f"Apply the specifically requested transformation: "
                f"{target_action}.",
            )

        # ==============================================================
        # IMPROVE DATASET
        # ==============================================================

        if workflow == "improve_dataset":

            if not state.model_inspected:
                return Decision(
                    "inspect_model",
                    {
                        "model_path": state.model_path,
                        "model_id": state.model_id,
                    },
                    "Inspect the model before improving the dataset.",
                )

            if not state.dataset_inspected:
                return Decision(
                    "inspect_dataset",
                    {
                        "dataset_path": state.dataset_path,
                        "dataset_id": state.dataset_id,
                        "version": state.current_dataset_version,
                        "target_col": state.target_column,
                    },
                    "Inspect the dataset before improving it.",
                )

            if not state.compatibility_validated:
                return Decision(
                    "validate_model_dataset",
                    {
                        "model_path": state.model_path,
                        "dataset_path": state.dataset_path,
                        "model_id": state.model_id,
                        "dataset_id": state.dataset_id,
                        "version": state.current_dataset_version,
                    },
                    "Validate compatibility before dataset improvement.",
                )

            if not state.baseline_evaluated:
                return Decision(
                    "evaluate_performance",
                    {
                        "model_path": state.model_path,
                        "dataset_path": state.dataset_path,
                        "target_column": state.target_column,
                        "task_type": state.task_type,
                    },
                    "Obtain baseline performance before improving the dataset.",
                )

            if not state.data_quality_evaluated:
                return Decision(
                    "evaluate_data_quality",
                    {
                        "dataset_path": state.dataset_path,
                        "target_column": state.target_column,
                    },
                    "Identify deterministic dataset-quality issues.",
                )

            if not state.robustness_evaluated:
                return Decision(
                    "evaluate_robustness",
                    {
                        "model_path": state.model_path,
                        "dataset_path": state.dataset_path,
                        "target_column": state.target_column,
                        "noise_level": 0.10,
                    },
                    "Evaluate model robustness before deciding on improvements.",
                )

            # ----------------------------------------------------------
            # Apply only justified transformations.
            # ----------------------------------------------------------

            for action in [
                "handle_missing_values",
                "balance_data",
                "handle_outliers",
                "scale_data",
            ]:

                if AgentPlanner._transformation_is_justified(
                    action,
                    state,
                ):

                    params = {
                        "dataset_id": state.dataset_id,
                        "current_version": state.current_dataset_version,
                    }

                    if state.target_column:
                        params["target_column"] = (
                            state.target_column
                        )

                    if action == "handle_missing_values":
                        params["strategy"] = "median"

                    elif action == "balance_data":
                        params["method"] = "smote"

                    elif action == "handle_outliers":
                        params["method"] = "clip"

                    elif action == "scale_data":
                        params["method"] = "standard"

                    return Decision(
                        action,
                        params,
                        f"Apply {action} because deterministic evaluation "
                        "identified evidence supporting the transformation.",
                    )

            # ----------------------------------------------------------
            # Re-evaluate transformed dataset.
            # ----------------------------------------------------------

            if (
                state.transformations_applied
                and state.evaluated_dataset_version
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
                        "A new dataset version was created; reevaluate "
                        "model performance on the transformed dataset."
                    ),
                )

            # ----------------------------------------------------------
            # Compare results.
            # ----------------------------------------------------------

            if (
                state.transformations_applied
                and state.latest_metrics is not None
                and state.evaluated_dataset_version
                == state.current_dataset_version
                and not AgentPlanner._action_was_taken(
                    state,
                    "compare_results",
                )
            ):
                return Decision(
                    "compare_results",
                    {},
                    "Compare baseline and transformed evaluation results.",
                )

            # ----------------------------------------------------------
            # Save experiment.
            # ----------------------------------------------------------

            if (
                state.transformations_applied
                and state.latest_metrics is not None
                and state.evaluated_dataset_version
                == state.current_dataset_version
                and AgentPlanner._action_was_taken(
                    state,
                    "compare_results",
                )
                and not state.experiments
            ):
                return Decision(
                    "save_experiment",
                    {},
                    "Persist the dataset improvement experiment and metric changes.",
                )

            return Decision(
                "",
                {},
                "Dataset improvement workflow is complete.",
                status="complete",
            )

        # ==============================================================
        # FULL EVALUATION
        # ==============================================================

        if workflow == "full_evaluation":

            if not state.model_inspected:
                return Decision(
                    "inspect_model",
                    {
                        "model_path": state.model_path,
                        "model_id": state.model_id,
                    },
                    "Inspect the model before running the full evaluation.",
                )

            if not state.dataset_inspected:
                return Decision(
                    "inspect_dataset",
                    {
                        "dataset_path": state.dataset_path,
                        "dataset_id": state.dataset_id,
                        "version": state.current_dataset_version,
                        "target_col": state.target_column,
                    },
                    "Inspect the dataset before running the full evaluation.",
                )

            if not state.compatibility_validated:
                return Decision(
                    "validate_model_dataset",
                    {
                        "model_path": state.model_path,
                        "dataset_path": state.dataset_path,
                        "model_id": state.model_id,
                        "dataset_id": state.dataset_id,
                        "version": state.current_dataset_version,
                    },
                    "Validate model and dataset compatibility.",
                )

            if not state.baseline_evaluated:
                return Decision(
                    "evaluate_performance",
                    {
                        "model_path": state.model_path,
                        "dataset_path": state.dataset_path,
                        "target_column": state.target_column,
                        "task_type": state.task_type,
                    },
                    "Calculate deterministic baseline performance metrics.",
                )

            if not state.data_quality_evaluated:
                return Decision(
                    "evaluate_data_quality",
                    {
                        "dataset_path": state.dataset_path,
                        "target_column": state.target_column,
                    },
                    "Evaluate dataset quality and identify improvement opportunities.",
                )

            if not state.robustness_evaluated:
                return Decision(
                    "evaluate_robustness",
                    {
                        "model_path": state.model_path,
                        "dataset_path": state.dataset_path,
                        "target_column": state.target_column,
                        "noise_level": 0.10,
                    },
                    "Evaluate model robustness.",
                )

            # ----------------------------------------------------------
            # Apply only justified transformations.
            # ----------------------------------------------------------

            for action in [
                "handle_missing_values",
                "balance_data",
                "handle_outliers",
                "scale_data",
            ]:

                if AgentPlanner._transformation_is_justified(
                    action,
                    state,
                ):

                    params = {
                        "dataset_id": state.dataset_id,
                        "current_version": state.current_dataset_version,
                    }

                    if state.target_column:
                        params["target_column"] = (
                            state.target_column
                        )

                    if action == "handle_missing_values":
                        params["strategy"] = "median"

                    elif action == "balance_data":
                        params["method"] = "smote"

                    elif action == "handle_outliers":
                        params["method"] = "clip"

                    elif action == "scale_data":
                        params["method"] = "standard"

                    return Decision(
                        action,
                        params,
                        f"Apply {action} because deterministic data-quality "
                        "results justify it.",
                    )

            # ----------------------------------------------------------
            # Re-evaluation.
            # ----------------------------------------------------------

            if (
                state.transformations_applied
                and state.evaluated_dataset_version
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
                        f"Re-evaluate performance on transformed dataset "
                        f"version {state.current_dataset_version}."
                    ),
                )

            # ----------------------------------------------------------
            # Compare results.
            # ----------------------------------------------------------

            if (
                state.transformations_applied
                and state.latest_metrics is not None
                and state.evaluated_dataset_version
                == state.current_dataset_version
                and not AgentPlanner._action_was_taken(
                    state,
                    "compare_results",
                )
            ):
                return Decision(
                    "compare_results",
                    {},
                    "Compare baseline and transformed evaluation results.",
                )

            # ----------------------------------------------------------
            # Save experiment.
            # ----------------------------------------------------------

            if (
                state.transformations_applied
                and state.latest_metrics is not None
                and state.evaluated_dataset_version
                == state.current_dataset_version
                and AgentPlanner._action_was_taken(
                    state,
                    "compare_results",
                )
                and not state.experiments
            ):
                return Decision(
                    "save_experiment",
                    {},
                    "Persist the completed transformation experiment.",
                )

            # ----------------------------------------------------------
            # Final report.
            # ----------------------------------------------------------

            if (
                state.baseline_evaluated
                and state.data_quality_evaluated
                and state.robustness_evaluated
                and state.evaluated_dataset_version
                == state.current_dataset_version
                and (
                    not state.transformations_applied
                    or bool(state.experiments)
                )
            ):
                return Decision(
                    "generate_report",
                    {
                        "project_id": state.project_id,
                        "title": (
                            f"Evaluation Report - {state.project_id}"
                        ),
                    },
                    "The full evaluation is complete; generate the final report.",
                )

            return Decision(
                "",
                {},
                "Full evaluation workflow is complete.",
                status="complete",
            )

        # ==============================================================
        # REPORT
        # ==============================================================

        if workflow == "report":

            if not state.model_inspected:
                return Decision(
                    "inspect_model",
                    {
                        "model_path": state.model_path,
                        "model_id": state.model_id,
                    },
                    "Inspect the model before generating the report.",
                )

            if not state.dataset_inspected:
                return Decision(
                    "inspect_dataset",
                    {
                        "dataset_path": state.dataset_path,
                        "dataset_id": state.dataset_id,
                        "version": state.current_dataset_version,
                        "target_col": state.target_column,
                    },
                    "Inspect the dataset before generating the report.",
                )

            if not state.compatibility_validated:
                return Decision(
                    "validate_model_dataset",
                    {
                        "model_path": state.model_path,
                        "dataset_path": state.dataset_path,
                        "model_id": state.model_id,
                        "dataset_id": state.dataset_id,
                        "version": state.current_dataset_version,
                    },
                    "Validate compatibility before generating the report.",
                )

            if not state.baseline_evaluated:
                return Decision(
                    "evaluate_performance",
                    {
                        "model_path": state.model_path,
                        "dataset_path": state.dataset_path,
                        "target_column": state.target_column,
                        "task_type": state.task_type,
                    },
                    "Obtain baseline metrics required by the report.",
                )

            if not state.data_quality_evaluated:
                return Decision(
                    "evaluate_data_quality",
                    {
                        "dataset_path": state.dataset_path,
                        "target_column": state.target_column,
                    },
                    "Obtain data-quality information required by the report.",
                )

            if not state.robustness_evaluated:
                return Decision(
                    "evaluate_robustness",
                    {
                        "model_path": state.model_path,
                        "dataset_path": state.dataset_path,
                        "target_column": state.target_column,
                        "noise_level": 0.10,
                    },
                    "Obtain robustness information required by the report.",
                )

            return Decision(
                "generate_report",
                {
                    "project_id": state.project_id,
                    "title": (
                        f"Evaluation Report - {state.project_id}"
                    ),
                },
                "Generate the requested evaluation report.",
            )

        # ==============================================================
        # UNKNOWN WORKFLOW
        # ==============================================================

        return Decision(
            "",
            {},
            f"Unknown workflow: {workflow}",
            status="complete",
        )


agent_planner = AgentPlanner()