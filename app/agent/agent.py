from typing import Dict, Any, Optional, Callable

from app.agent.state import AgentState
from app.agent.planner import agent_planner
from app.agent.tools import tool_registry
from app.reports.generator import report_generator


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

        decision = agent_planner.decide_next_step(
            self.state
        )

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
        # Store observation
        # --------------------------------------------------------------

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

        # --------------------------------------------------------------
        # improve_dataset and full_evaluation have post-processing
        # stages controlled by the planner.
        #
        # Do NOT finalize here even if workflow_complete() temporarily
        # evaluates to True.
        # --------------------------------------------------------------

        if workflow in {
            "improve_dataset",
            "full_evaluation",
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

            self.state.compatibility_validated = True

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
        # Dataset transformations
        # --------------------------------------------------------------

        if action in {
            "handle_missing_values",
            "balance_data",
            "handle_outliers",
            "scale_data",
            "add_gaussian_noise",
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