import math
import logging
from typing import Dict, Any, Optional, Callable

from app.agent.state import AgentState
from app.agent.planner import agent_planner
from app.agent.security_aliases import IncompatibleSecurityRequest
from app.agent.tools import tool_registry
from app.reports.generator import report_generator

logger = logging.getLogger(__name__)


class AgentRunner:
    """
    Executes the closed-loop agentic evaluation cycle.

    The planner selects actions, while Python tools perform the actual work.

    The requested_workflow in AgentState determines the scope of execution.
    """

    def __init__(self, state: AgentState):
        self.state = state

    # ------------------------------------------------------------------
    # Execute one agent step
    # ------------------------------------------------------------------

    def run_step(self) -> Dict[str, Any]:

        # --------------------------------------------------------------
        # Stop if state is already complete
        # --------------------------------------------------------------

        if self.state.is_complete():
            return {
                "status": self.state.status,
                "message": f"Agent is already {self.state.status}",
                "actions": self._action_names(),
            }

        self.state.iteration += 1

        # --------------------------------------------------------------
        # Ask planner for next action
        # --------------------------------------------------------------

        # A capability refusal is a planned, user-facing outcome: surface the
        # reason verbatim instead of retrying or leaking a traceback.
        try:
            decision = agent_planner.decide_next_step(
                self.state
            )
        except IncompatibleSecurityRequest as exc:
            logger.info(
                "Security request refused for project %s: %s",
                self.state.project_id,
                exc,
            )
            self.state.status = "failed"
            self.state.stop_reason = str(exc)
            return {
                "iteration": self.state.iteration,
                "action": "",
                "reason": str(exc),
                "result_status": "error",
                "agent_status": self.state.status,
                "error": str(exc),
            }

        action = decision.action
        params = decision.parameters or {}
        reason = decision.reason

        # --------------------------------------------------------------
        # Planner has no further action
        #
        # IMPORTANT:
        # Do not stop merely because decision.status says "complete"
        # if the planner supplied an executable action.
        # --------------------------------------------------------------

        if not action:

            self.state.status = "completed"
            self.state.stop_reason = reason

            return {
                "iteration": self.state.iteration,
                "action": "",
                "reason": reason,
                "result_status": "complete",
                "agent_status": self.state.status,
            }

        # --------------------------------------------------------------
        # Record planned action
        # --------------------------------------------------------------

        self.state.add_action(
            action,
            params,
        )

        # --------------------------------------------------------------
        # Execute selected tool
        # --------------------------------------------------------------
        try:
            tool_res = tool_registry.execute_tool(
                action,
                params,
            )
        except Exception as exc:
            tool_res = {
                "status": "error",
                "message": str(exc),
            }

        # --------------------------------------------------------------
        # Store observation (never persist full adversarial tensors)
        # --------------------------------------------------------------

        tool_res = self._sanitize_observation(tool_res)

        self.state.add_observation(
            action,
            tool_res,
        )

        # --------------------------------------------------------------
        # Successful tool execution
        # --------------------------------------------------------------

        if tool_res.get("status") == "success":

            res_data = tool_res.get(
                "result",
                {},
            )

            # ----------------------------------------------------------
            # Report generation
            #
            # generate_report in tools.py authorizes the operation.
            # Actual report generation happens here.
            # ----------------------------------------------------------

            if action == "generate_report":

                try:

                    report_result = (
                        report_generator.generate_and_save(
                            self.state
                        )
                    )

                    if isinstance(
                        report_result,
                        dict,
                    ):
                        res_data = {
                            "status": "report_ready",
                            **report_result,
                        }
                    else:
                        res_data = {
                            "status": "report_ready",
                        }

                    self.state.add_observation(
                        "report_generator",
                        {
                            "status": "success",
                            "result": res_data,
                        },
                    )

                    self.state.status = "completed"
                    self.state.stop_reason = (
                        "Requested report was successfully generated."
                    )

                except Exception as exc:

                    logger.exception("Report generation failed for project %s", self.state.project_id)


                    error_message = (
                        f"Report generation failed: {str(exc)}"
                    )

                    self._mark_last_action_failed(
                        error_message
                    )

                    if self._too_many_consecutive_failures(
                        action
                    ):
                        self.state.status = "failed"
                        self.state.stop_reason = (
                            f"Tool '{action}' failed repeatedly; "
                            "stopping to avoid an infinite loop."
                        )

                    return {
                        "iteration": self.state.iteration,
                        "action": action,
                        "reason": reason,
                        "result_status": "error",
                        "agent_status": self.state.status,
                        "error": error_message,
                    }

            # ----------------------------------------------------------
            # Normal tool execution
            # ----------------------------------------------------------

            else:

                self._update_state_from_result(
                    action,
                    res_data,
                    params,
                )

            # ----------------------------------------------------------
            # Workflow completion
            #
            # Only immediately finalize workflows whose completion
            # criteria are actually the end of execution.
            #
            # improve_dataset and full_evaluation may still require:
            #   compare_results
            #   save_experiment
            #   generate_report
            #
            # Therefore workflow_complete() must not prematurely stop
            # those workflows here.
            # ----------------------------------------------------------

            if self._can_finalize_after_successful_action(
                action
            ):
                self.state.status = "completed"

                self.state.stop_reason = (
                    f"Requested workflow "
                    f"'{self.state.requested_workflow}' completed."
                )

            return {
                "iteration": self.state.iteration,
                "action": action,
                "reason": reason,
                "result_status": "success",
                "agent_status": self.state.status,
            }

        # --------------------------------------------------------------
        # Failed tool execution
        # --------------------------------------------------------------

        error_message = tool_res.get(
            "message",
            tool_res.get(
                "error",
                "Unknown tool error",
            ),
        )

        self._mark_last_action_failed(
            f"Failed: {error_message}"
        )

        # A failed tool leaves its state flags unset, so re-planning would
        # replay the identical deterministic call. Fail fast with the real
        # cause instead of retrying and burying it under a generic message.
        self.state.status = "failed"
        self.state.stop_reason = str(error_message)

        return {
            "iteration": self.state.iteration,
            "action": action,
            "reason": reason,
            "result_status": "error",
            "agent_status": self.state.status,
            "error": str(error_message),
        }

    # ------------------------------------------------------------------
    # Determine whether the runner may finalize immediately
    # ------------------------------------------------------------------

    def _can_finalize_after_successful_action(
        self,
        action: str,
    ) -> bool:

        workflow = self.state.requested_workflow

        # --------------------------------------------------------------
        # Report is finalized explicitly after report generation.
        # --------------------------------------------------------------

        if action == "generate_report":
            return True

        # --------------------------------------------------------------
        # These workflows end as soon as their required state is met.
        # --------------------------------------------------------------

        if workflow in {
            "inspection",
            "baseline_metrics",
            "targeted_transform",
            "contextual",
            "comparison",
        }:
            return self.state.workflow_complete()

        if workflow in {
            "improve_dataset",
            "full_evaluation",
            "attack",
            "defend",
            "boundary_attack", "hopskipjump", "zoo", "fgsm", "pgd", "cw",
            "gaussian_augmentation", "spatial_smoothing", "feature_squeezing",
            "clip_values", "reverse_sigmoid", "class_labels", "high_confidence_masking",
            "attack_defended_model", "defended_attack", "harden_model", "harden",
            "hardening_loop", "harden_defense_loop",
        }:
            return False
        # --------------------------------------------------------------
        # For unknown workflows, fall back to the state's completion
        # logic.
        # --------------------------------------------------------------

        return self.state.workflow_complete()

    # ------------------------------------------------------------------
    # Run complete workflow
    # ------------------------------------------------------------------

    def run_until_complete(
        self,
        callback: Optional[
            Callable[[Dict[str, Any]], None]
        ] = None,
    ) -> AgentState:

        self.state.status = "running"

        while not self.state.is_complete():

            step_summary = self.run_step()

            if callback:
                callback(step_summary)

            # ----------------------------------------------------------
            # Stop if completed or failed
            # ----------------------------------------------------------

            if self.state.is_complete():
                break

            # ----------------------------------------------------------
            # Maximum iteration protection
            # ----------------------------------------------------------

            if (
                self.state.iteration
                >= self.state.max_iterations
            ):
                self.state.status = "failed"
                self.state.stop_reason = (
                    f"Maximum iterations "
                    f"({self.state.max_iterations}) reached "
                    "before the workflow completed."
                )
                break

        return self.state

    # ------------------------------------------------------------------
    # State update after successful tool
    # ------------------------------------------------------------------

    def _update_state_from_result(
        self,
        action: str,
        result: Any,
        params: Dict[str, Any],
    ):

        # --------------------------------------------------------------
        # Model inspection
        # --------------------------------------------------------------

        if action == "inspect_model":

            self.state.model_inspected = True

            self.state.model_metadata = (
                result
                if isinstance(result, dict)
                else {}
            )

            if (
                isinstance(result, dict)
                and result.get("model_type")
            ):
                self.state.task_type = (
                    result["model_type"]
                )

            return

        # --------------------------------------------------------------
        # Dataset inspection
        # --------------------------------------------------------------

        if action == "inspect_dataset":

            self.state.dataset_inspected = True

            self.state.dataset_metadata = (
                result
                if isinstance(result, dict)
                else {}
            )

            if (
                isinstance(result, dict)
                and result.get("target_column")
                and not self.state.target_column
            ):
                self.state.target_column = (
                    result["target_column"]
                )

            return

        # --------------------------------------------------------------
        # Compatibility validation
        # --------------------------------------------------------------

        if action == "validate_model_dataset":
            # Mark validation as executed regardless of compatibility outcome
            self.state.compatibility_validated = True
            if isinstance(result, dict):
                # Only an explicit pass counts as compatible; a missing or
                # non-boolean verdict must not unlock security actions.
                self.state.compatibility_result = (
                    result.get("is_compatible") is True
                )
            else:
                self.state.compatibility_result = False
            return

        # --------------------------------------------------------------
        # Performance evaluation
        # --------------------------------------------------------------

        if action == "evaluate_performance":

            # First successful performance evaluation is the baseline.
            if not self.state.baseline_evaluated:

                self.state.baseline_evaluated = True

                self.state.baseline_metrics = (
                    result
                    if isinstance(result, dict)
                    else {}
                )

            # Latest metrics always represent the most recent evaluation.
            self.state.latest_metrics = (
                result
                if isinstance(result, dict)
                else {}
            )

            # Record exactly which dataset version produced
            # these metrics.
            self.state.evaluated_dataset_version = (
                self.state.current_dataset_version
            )

            return

        # --------------------------------------------------------------
        # Data quality evaluation
        # --------------------------------------------------------------

        if action == "evaluate_data_quality":

            self.state.data_quality_evaluated = True

            self.state.data_quality = (
                result
                if isinstance(result, dict)
                else {}
            )

            return

        # --------------------------------------------------------------
        # Robustness evaluation
        # --------------------------------------------------------------

        if action == "evaluate_robustness":

            self.state.robustness_evaluated = True

            self.state.robustness_results = (
                result
                if isinstance(result, dict)
                else {}
            )

            return

        # --------------------------------------------------------------
        # Adversarial attacks (ART Evasion)
        # --------------------------------------------------------------

        if action in {"boundary_attack", "hopskipjump", "zoo", "fgsm", "pgd", "cw"}:

            if not isinstance(result, dict):
                return

            slim = {
                key: value
                for key, value in result.items()
                if key not in {"adversarial_samples", "x_adv"}
            }

            self.state.adversarial_evaluated = True
            self.state.attack_type = slim.get("attack_type") or action
            self.state.clean_accuracy = slim.get("clean_accuracy")
            self.state.adversarial_accuracy = slim.get("adversarial_accuracy")
            self.state.attack_results = slim

            return

        if action in {
            "gaussian_augmentation",
            "spatial_smoothing",
            "feature_squeezing",
            "clip_values",
            "reverse_sigmoid",
            "class_labels",
            "high_confidence_masking",
        }:

            if not isinstance(result, dict):
                return

            slim = {
                key: value
                for key, value in result.items()
                if key not in {"augmented_samples", "x_aug"}
            }

            self.state.defended = True
            self.state.defense_results = slim
            self.state.clean_accuracy = slim.get(
                "clean_accuracy",
                self.state.clean_accuracy,
            )

            return

        if action == "attack_defended_model":
            if not isinstance(result, dict):
                return

            if result.get("status") == "error":
                self.state.defended_attack_evaluated = False
                self.state.defended_attack_results = result
                self.state.attack_defense_sequence_evaluated = False
                self.state.attack_defense_sequence_results = None
                self.state.stop_reason = (
                    f"Defended-model attack failed: {result.get('error') or result.get('message')}"
                )
                return

            # A multi-step sweep keeps only its last step in the single-result
            # slot so the existing report section 8.1/8.2 logic still renders.
            if result.get("mode") == "sweep":
                self.state.attack_defense_sequence_evaluated = True
                self.state.attack_defense_sequence_results = result
                steps = result.get("steps") or []
                if steps:
                    self.state.defended_attack_results = steps[-1]
                    self.state.defended_attack_evaluated = True
                return

            self.state.defended_attack_evaluated = True
            self.state.defended_attack_results = result
            return

        if action == "harden_defense_loop":
            if not isinstance(result, dict):
                return

            if result.get("status") == "error":
                self.state.hardening_loop_evaluated = False
                self.state.hardening_loop_results = None
                self.state.stop_reason = (
                    f"Hardening loop failed: {result.get('error') or result.get('message')}"
                )
                return

            self.state.hardening_loop_evaluated = True
            self.state.hardening_loop_results = result
            if result.get("final_model_id") and result.get("final_model_path"):
                self.state.model_hardened = True
                self.state.hardened_model_info = result
                self.state.model_id = result["final_model_id"]
                self.state.model_path = result["final_model_path"]
            return

        if action == "harden_model":
            if not isinstance(result, dict) or not result.get("model_id") or not result.get("file_path"):
                return
            self.state.model_hardened = True
            self.state.hardened_model_info = result
            self.state.model_id = result["model_id"]
            self.state.model_path = result["file_path"]
            return

        # --------------------------------------------------------------
        # Dataset transformations
        # --------------------------------------------------------------

        if action in {
            "handle_missing_values",
            "balance_data",
            "handle_outliers",
            "scale_data",
            "add_gaussian_noise",
            "inject_missing_values",
            "inject_label_noise",
        }:

            if not isinstance(result, dict):
                return

            new_version = result.get(
                "version"
            )

            if not new_version:
                return

            old_version = (
                self.state.current_dataset_version
            )

            # ----------------------------------------------------------
            # Update current dataset version
            # ----------------------------------------------------------

            self.state.current_dataset_version = (
                new_version
            )

            # ----------------------------------------------------------
            # Update dataset path if tool returned one
            # ----------------------------------------------------------

            file_path = result.get(
                "file_path"
            )

            if file_path:
                self.state.dataset_path = file_path

            # ----------------------------------------------------------
            # Record transformation.
            # ----------------------------------------------------------

            transformation_record = {
                "action": action,
                "operation": action,
                "input_version": old_version,
                "output_version": new_version,
                "parameters": params,
                "result": result,
            }

            self.state.transformations_applied.append(
                transformation_record
            )

            # ----------------------------------------------------------
            # A transformed dataset needs fresh evaluation.
            # ----------------------------------------------------------

            if (
                self.state.evaluated_dataset_version
                != self.state.current_dataset_version
            ):
                self.state.latest_metrics = None

            return

        # --------------------------------------------------------------
        # Experiment persistence
        # --------------------------------------------------------------

        if action == "save_experiment":

            if isinstance(result, dict):

                self.state.experiments.append(
                    result
                )

            return

        # --------------------------------------------------------------
        # Comparison
        # --------------------------------------------------------------

        if action == "compare_results":
            return

        # --------------------------------------------------------------
        # Report
        # --------------------------------------------------------------

        if action == "generate_report":
            return

    # ------------------------------------------------------------------
    # Action helpers
    # ------------------------------------------------------------------

    def _action_names(self):

        names = []

        for record in self.state.actions_taken:

            if isinstance(record, dict):

                action = record.get(
                    "action"
                )

                if action:
                    names.append(action)

            elif isinstance(record, str):

                names.append(record)

        return names

    @staticmethod
    def _sanitize_observation(value: Any) -> Any:
        """Keep persisted agent observations JSON-sized and serializable."""
        bulky_keys = {"adversarial_samples", "x_adv", "augmented_samples", "x_aug"}
        if isinstance(value, dict):
            return {str(k): AgentRunner._sanitize_observation(v) for k, v in value.items() if k not in bulky_keys}
        if isinstance(value, (list, tuple)):
            if len(value) > 100:
                return {"summary": "sequence omitted", "count": len(value)}
            return [AgentRunner._sanitize_observation(v) for v in value]
        if hasattr(value, "shape"):
            return {"summary": type(value).__name__, "shape": list(value.shape)}
        if isinstance(value, float) and not math.isfinite(value):
            return None
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return str(value)

    def _mark_last_action_failed(
        self,
        reason: str,
    ):

        if not self.state.actions_taken:
            return

        last = self.state.actions_taken[-1]

        if isinstance(last, dict):

            last["status"] = "failed"
            last["reason"] = reason

            return

        # Compatibility with any legacy string records.
        self.state.actions_taken[-1] = {
            "action": last,
            "status": "failed",
            "reason": reason,
        }

    # ------------------------------------------------------------------
    # Consecutive failure protection
    # ------------------------------------------------------------------

    def _too_many_consecutive_failures(
        self,
        action: str,
        limit: int = 3,
    ) -> bool:

        failures = 0

        for record in reversed(
            self.state.actions_taken
        ):

            if isinstance(record, dict):

                record_action = record.get(
                    "action"
                )

                record_status = record.get(
                    "status"
                )

            else:

                record_action = record
                record_status = None

            if record_action != action:
                break

            if record_status == "failed":

                failures += 1

            else:

                break

        return failures >= limit
