from typing import Dict, Any, Optional, Callable

from app.agent.state import AgentState
from app.agent.planner import agent_planner
from app.agent.tools import tool_registry
from app.reports.generator import report_generator


class AgentRunner:
    """Executes the closed-loop agentic evaluation cycle."""

    def __init__(self, state: AgentState):
        self.state = state

    def run_step(self) -> Dict[str, Any]:
        # ---------------------------------------------------------
        # Stop if already finished
        # ---------------------------------------------------------
        if self.state.is_complete():
            return {
                "status": self.state.status,
                "message": f"Agent is already {self.state.status}",
                "actions": [
                    a.action for a in self.state.actions_taken
                ],
            }

        self.state.iteration += 1

        # ---------------------------------------------------------
        # Ask planner for next action
        # ---------------------------------------------------------
        decision = agent_planner.decide_next_step(self.state)

        action = decision.action
        params = decision.parameters or {}
        reason = decision.reason

        # ---------------------------------------------------------
        # Execute selected tool
        # ---------------------------------------------------------
        tool_res = tool_registry.execute_tool(
            action,
            params,
        )

        self.state.add_observation(
            tool=action,
            parameters=params,
            result=tool_res,
        )

        # ---------------------------------------------------------
        # Successful tool execution
        # ---------------------------------------------------------
        if tool_res.get("status") == "success":

            res_data = tool_res.get("result", {})

            # -----------------------------------------------------
            # generate_report is special.
            #
            # The registry tool authorizes the operation.
            # The actual report is generated here from AgentState.
            # -----------------------------------------------------
            if action == "generate_report":

                try:
                    report_result = (
                        report_generator.generate_and_save(
                            self.state
                        )
                    )

                    res_data = {
                        "status": "report_ready",
                        **(
                            report_result
                            if isinstance(report_result, dict)
                            else {}
                        ),
                    }

                    self.state.add_observation(
                        tool="report_generator",
                        parameters={},
                        result={
                            "status": "success",
                            "result": res_data,
                        },
                    )

                    self.state.record_action(
                        action=action,
                        reason=reason,
                        status="success",
                    )

                    self.state.status = "completed"
                    self.state.stop_reason = reason

                except Exception as exc:

                    error_message = (
                        f"Report generation failed: {str(exc)}"
                    )

                    self.state.record_action(
                        action=action,
                        reason=error_message,
                        status="failed",
                    )

                    if self._too_many_consecutive_failures(action):
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
                    }

            else:
                # -------------------------------------------------
                # Normal tool result
                # -------------------------------------------------
                self._update_state_from_result(
                    action,
                    res_data,
                    params,
                )

                self.state.record_action(
                    action=action,
                    reason=reason,
                    status="success",
                )

        # ---------------------------------------------------------
        # Failed tool execution
        # ---------------------------------------------------------
        else:

            error_message = tool_res.get(
                "message",
                "Unknown tool error",
            )

            self.state.record_action(
                action=action,
                reason=f"Failed: {error_message}",
                status="failed",
            )

            if self._too_many_consecutive_failures(action):
                self.state.status = "failed"
                self.state.stop_reason = (
                    f"Tool '{action}' failed repeatedly; "
                    "stopping to avoid an infinite loop."
                )

        # ---------------------------------------------------------
        # Step summary
        # ---------------------------------------------------------
        return {
            "iteration": self.state.iteration,
            "action": action,
            "reason": reason,
            "result_status": tool_res.get("status"),
            "agent_status": self.state.status,
        }

    def _too_many_consecutive_failures(
        self,
        action: str,
        limit: int = 3,
    ) -> bool:
        failures = 0

        for record in reversed(
            self.state.actions_taken
        ):
            if record.action != action:
                break

            if record.status == "failed":
                failures += 1
            else:
                break

        return failures >= limit

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

            # Do not overwrite a completed/failed state.
            if self.state.is_complete():
                break

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

    def _update_state_from_result(
        self,
        action: str,
        result: Any,
        params: Dict[str, Any],
    ):
        # ---------------------------------------------------------
        # Model inspection
        # ---------------------------------------------------------
        if action == "inspect_model":

            self.state.is_model_inspected = True
            self.state.model_metadata = result

            if (
                isinstance(result, dict)
                and result.get("model_type")
            ):
                self.state.task_type = result["model_type"]

        # ---------------------------------------------------------
        # Dataset inspection
        # ---------------------------------------------------------
        elif action == "inspect_dataset":

            self.state.is_dataset_inspected = True
            self.state.dataset_metadata = result

            if (
                isinstance(result, dict)
                and result.get("target_column")
            ):
                if not self.state.target_column:
                    self.state.target_column = (
                        result["target_column"]
                    )

        # ---------------------------------------------------------
        # Compatibility validation
        # ---------------------------------------------------------
        elif action == "validate_model_dataset":

            self.state.is_compatibility_validated = True

        # ---------------------------------------------------------
        # Performance evaluation
        # ---------------------------------------------------------
        elif action == "evaluate_performance":

            # First successful performance evaluation is the
            # baseline.
            if not self.state.is_baseline_evaluated:

                self.state.is_baseline_evaluated = True
                self.state.baseline_metrics = result

                # IMPORTANT:
                # Do NOT write baseline_dataset_version here.
                # AgentState does not define that field.
                #
                # evaluated_dataset_version already records which
                # dataset produced the current metrics.

            self.state.latest_metrics = result

            # Remember exactly which dataset version produced
            # these metrics.
            self.state.evaluated_dataset_version = (
                self.state.current_dataset_version
            )

        # ---------------------------------------------------------
        # Data quality evaluation
        # ---------------------------------------------------------
        elif action == "evaluate_data_quality":

            self.state.is_data_quality_evaluated = True
            self.state.data_quality_metrics = result

        # ---------------------------------------------------------
        # Robustness evaluation
        # ---------------------------------------------------------
        elif action == "evaluate_robustness":

            self.state.is_robustness_evaluated = True
            self.state.robustness_metrics = result

        # ---------------------------------------------------------
        # Dataset transformations
        # ---------------------------------------------------------
        elif action in {
            "handle_missing_values",
            "balance_data",
            "handle_outliers",
            "scale_data",
            "add_gaussian_noise",
        }:

            if (
                isinstance(result, dict)
                and result.get("version")
            ):

                self.state.current_dataset_version = (
                    result["version"]
                )

                self.state.dataset_path = result.get(
                    "file_path",
                    self.state.dataset_path,
                )

                self.state.transformations_applied.append(
                    action
                )

                # The dataset changed, so the previous evaluation
                # is no longer the evaluation of the current
                # dataset version.
                if (
                    self.state.evaluated_dataset_version
                    != self.state.current_dataset_version
                ):
                    self.state.latest_metrics = None

        # ---------------------------------------------------------
        # Experiment persistence
        # ---------------------------------------------------------
        elif action == "save_experiment":

            if isinstance(result, dict):
                self.state.experiments.append(result)

        # ---------------------------------------------------------
        # Report generation
        #
        # Actual report generation is handled in run_step().
        # ---------------------------------------------------------
        elif action == "generate_report":
            pass
