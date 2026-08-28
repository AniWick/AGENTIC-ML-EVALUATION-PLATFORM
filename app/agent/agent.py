from typing import Dict, Any, List, Optional, Callable
from app.agent.state import AgentState
from app.agent.planner import agent_planner, Decision
from app.agent.tools import tool_registry
from app.datasets.versioning import dataset_version_manager
from app.reports.generator import report_generator

class AgentRunner:
    """Executes the closed-loop agentic evaluation cycle."""

    def __init__(self, state: AgentState):
        self.state = state

    def run_step(self) -> Dict[str, Any]:
        """Executes a single iteration of the agentic loop."""
        if self.state.is_complete():
            return {
                "status": self.state.status,
                "message": f"Agent is already {self.state.status}",
                "actions": [a.action for a in self.state.actions_taken]
            }

        self.state.iteration += 1
        decision = agent_planner.decide_next_step(self.state)

        action = decision.action
        params = decision.parameters
        reason = decision.reason

        # Execute tool
        tool_res = tool_registry.execute_tool(action, params)
        self.state.add_observation(tool=action, parameters=params, result=tool_res)

        if tool_res.get("status") == "success":
            res_data = tool_res.get("result", {})
            self._update_state_from_result(action, res_data, params)
            self.state.record_action(action=action, reason=reason, status="success")
        else:
            # Handle error as observation
            self.state.record_action(action=action, reason=f"Failed: {tool_res.get('message')}", status="failed")

        if decision.status == "complete" or action == "generate_report":
            self.state.status = "completed"
            self.state.stop_reason = reason

        return {
            "iteration": self.state.iteration,
            "action": action,
            "reason": reason,
            "result_status": tool_res.get("status"),
            "agent_status": self.state.status
        }

    def run_until_complete(self, callback: Optional[Callable[[Dict[str, Any]], None]] = None) -> AgentState:
        """Executes consecutive agent steps until goal completion or max iterations reached."""
        self.state.status = "running"
        while not self.state.is_complete():
            step_summary = self.run_step()
            if callback:
                callback(step_summary)
            if self.state.iteration >= self.state.max_iterations:
                self.state.status = "completed"
                self.state.stop_reason = f"Reached maximum allowed iterations ({self.state.max_iterations})"
                break
        return self.state

    def _update_state_from_result(self, action: str, result: Any, params: Dict[str, Any]):
        """Updates structured agent state from tool result payloads."""
        if action == "inspect_model":
            self.state.is_model_inspected = True
            self.state.model_metadata = result
            if "model_type" in result:
                self.state.task_type = result["model_type"]

        elif action == "inspect_dataset":
            self.state.is_dataset_inspected = True
            self.state.dataset_metadata = result
            if "target_column" in result and not self.state.target_column:
                self.state.target_column = result["target_column"]

        elif action == "validate_model_dataset":
            self.state.is_compatibility_validated = True

        elif action == "evaluate_performance":
            if not self.state.is_baseline_evaluated:
                self.state.is_baseline_evaluated = True
                self.state.baseline_metrics = result
                self.state.latest_metrics = result
            else:
                self.state.latest_metrics = result

        elif action == "evaluate_data_quality":
            self.state.is_data_quality_evaluated = True
            self.state.data_quality_metrics = result

        elif action == "evaluate_robustness":
            self.state.is_robustness_evaluated = True
            self.state.robustness_metrics = result

        elif action in ["handle_missing_values", "balance_data", "handle_outliers", "scale_data", "add_gaussian_noise"]:
            if isinstance(result, dict) and "version" in result:
                new_version = result["version"]
                self.state.current_dataset_version = new_version
                self.state.dataset_path = result.get("file_path", self.state.dataset_path)
                self.state.transformations_applied.append(action)

        elif action == "save_experiment":
            if isinstance(result, dict):
                self.state.experiments.append(result)

        elif action == "generate_report":
            report_generator.generate_and_save(self.state)


