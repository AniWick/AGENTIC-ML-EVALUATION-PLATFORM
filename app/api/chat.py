from fastapi import APIRouter, HTTPException
from typing import Dict, Any, List
import json

from app.schemas.chat import ChatRequest, ChatResponse
from app.services.project_service import project_service
from app.core.database import db

from app.agent.state import AgentState
from app.agent.agent import AgentRunner

from app.memory.chroma_memory import project_memory
from app.llm.qwen_client import qwen_client

from app.llm.prompts import (
    INTENT_CLASSIFICATION_PROMPT,
    CHAT_RESPONSE_PROMPT,
    CHAT_COMPARISON_PROMPT,
)


router = APIRouter()


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

def _parse_router_response(raw: Any, message: str) -> Dict[str, str]:
    """
    Parse Qwen's workflow-routing response.

    Expected:
    {
        "intent": "deterministic",
        "workflow": "inspection"
    }

    Falls back to deterministic keyword routing if Qwen fails.
    """

    if isinstance(raw, dict):
        data = raw
    else:
        try:
            data = json.loads(raw)
        except Exception:
            data = {}

    intent = data.get("intent")
    workflow = data.get("workflow")

    valid_intents = {
        "deterministic",
        "contextual",
        "comparison",
    }

    valid_workflows = {
        "inspection",
        "baseline_metrics",
        "full_evaluation",
        "improve_dataset",
        "targeted_transform",
        "report",
        "contextual",
        "comparison",
    }

    if intent not in valid_intents:
        intent = None

    if workflow not in valid_workflows:
        workflow = None

    if intent and workflow:
        return {
            "intent": intent,
            "workflow": workflow,
        }

    # ---------------------------------------------------------------
    # Deterministic fallback routing
    # ---------------------------------------------------------------

    msg = message.lower().strip()

    # Report
    if any(
        phrase in msg
        for phrase in [
            "generate report",
            "create report",
            "make a report",
            "evaluation report",
            "show report",
        ]
    ):
        return {
            "intent": "deterministic",
            "workflow": "report",
        }

    # Inspection
    if (
        ("inspect" in msg and ("model" in msg or "dataset" in msg))
        or "inspect my model" in msg
        or "inspect my dataset" in msg
        or "model and dataset information" in msg
        or "model and dataset details" in msg
    ):
        return {
            "intent": "deterministic",
            "workflow": "inspection",
        }

    # Baseline metrics
    if (
        "baseline" in msg
        or "before transformation" in msg
        or "before any transformation" in msg
        or "original metrics" in msg
    ):
        return {
            "intent": "deterministic",
            "workflow": "baseline_metrics",
        }

    # Targeted transformations
    if any(
        phrase in msg
        for phrase in [
            "clean missing",
            "handle missing",
            "remove missing",
            "missing values",
        ]
    ):
        return {
            "intent": "deterministic",
            "workflow": "targeted_transform",
        }

    if any(
        phrase in msg
        for phrase in [
            "balance dataset",
            "balance the dataset",
            "handle imbalance",
            "fix class imbalance",
            "balance classes",
            "smote",
        ]
    ):
        return {
            "intent": "deterministic",
            "workflow": "targeted_transform",
        }

    if any(
        phrase in msg
        for phrase in [
            "handle outliers",
            "remove outliers",
            "fix outliers",
        ]
    ):
        return {
            "intent": "deterministic",
            "workflow": "targeted_transform",
        }

    if any(
        phrase in msg
        for phrase in [
            "scale dataset",
            "scale the dataset",
            "apply scaling",
            "normalize dataset",
        ]
    ):
        return {
            "intent": "deterministic",
            "workflow": "targeted_transform",
        }

    if any(
        phrase in msg
        for phrase in [
            "add noise",
            "gaussian noise",
            "noise augmentation",
        ]
    ):
        return {
            "intent": "deterministic",
            "workflow": "targeted_transform",
        }

    # Dataset improvement
    if any(
        phrase in msg
        for phrase in [
            "improve my dataset",
            "improve the dataset",
            "improve dataset",
            "optimize my dataset",
            "improve data quality",
        ]
    ):
        return {
            "intent": "deterministic",
            "workflow": "improve_dataset",
        }

    # Full evaluation
    if any(
        phrase in msg
        for phrase in [
            "evaluate my model",
            "evaluate the model",
            "evaluate model",
            "evaluate my dataset",
            "evaluate the dataset",
            "run evaluation",
            "run full evaluation",
            "full evaluation",
            "complete evaluation",
        ]
    ):
        return {
            "intent": "deterministic",
            "workflow": "full_evaluation",
        }

    # Comparison
    if any(
        phrase in msg
        for phrase in [
            "compare",
            "better",
            "did smote help",
            "was balancing worth it",
            "before and after",
            "before vs after",
            "which version performed",
            "should i apply",
            "is my model robust",
        ]
    ):
        return {
            "intent": "comparison",
            "workflow": "comparison",
        }

    # Default: contextual question
    return {
        "intent": "contextual",
        "workflow": "contextual",
    }


def _extract_metrics(state: AgentState) -> Dict[str, Any]:
    """Build the metrics summary expected by ChatResponse."""

    if state.latest_metrics:
        return state.latest_metrics

    if state.baseline_metrics:
        return state.baseline_metrics

    return {}


def _extract_robustness(state: AgentState) -> Dict[str, Any]:
    """Return robustness results if available."""

    if state.robustness_results:
        return state.robustness_results

    return {}


def _action_names(state: AgentState) -> List[str]:
    """Extract action names from the current state's action history."""

    result = []

    for item in state.actions_taken:
        if isinstance(item, dict):
            action = item.get("action")
            if action:
                result.append(action)

        elif isinstance(item, str):
            result.append(item)

    return result


# -------------------------------------------------------------------
# Chat endpoint
# -------------------------------------------------------------------

@router.post(
    "/projects/{project_id}/chat",
    response_model=ChatResponse,
)
async def chat_with_project(
    project_id: str,
    request: ChatRequest,
):
    user_msg = request.message.strip()

    if not user_msg:
        raise HTTPException(
            status_code=400,
            detail="Message cannot be empty.",
        )

    # ---------------------------------------------------------------
    # Verify project
    # ---------------------------------------------------------------

    project = project_service.get_project(project_id)

    if not project:
        raise HTTPException(
            status_code=404,
            detail="Project not found.",
        )

    # ---------------------------------------------------------------
    # Save user message
    # ---------------------------------------------------------------

    db.insert_chat_message(
        project_id=project_id,
        role="user",
        content=user_msg,
    )

    # ---------------------------------------------------------------
    # Simple greetings
    # ---------------------------------------------------------------

    greetings = {
        "hi",
        "hello",
        "hey",
        "hi there",
        "hello there",
        "good morning",
        "good afternoon",
        "good evening",
    }

    if user_msg.lower() in greetings:
        response_text = (
            "Hello! I can inspect your model and dataset, run evaluations, "
            "improve the dataset, compare results, or answer questions about "
            "your evaluation history."
        )

        db.insert_chat_message(
            project_id=project_id,
            role="assistant",
            content=response_text,
        )

        return ChatResponse(
            message=response_text,
            status="completed",
            actions_taken=[],
        )

    # ---------------------------------------------------------------
    # Intent / workflow routing
    # ---------------------------------------------------------------

    try:
        routing_prompt = INTENT_CLASSIFICATION_PROMPT.format(
            message=user_msg
        )

        routing_result = qwen_client.generate_json(
            prompt=routing_prompt
        )

        route = _parse_router_response(
            routing_result,
            user_msg,
        )

    except Exception:
        route = _parse_router_response(
            {},
            user_msg,
        )

    intent = route["intent"]
    workflow = route["workflow"]

    # ---------------------------------------------------------------
    # Contextual / comparison questions
    #
    # These do NOT start the deterministic evaluation agent.
    # ---------------------------------------------------------------

    if intent in {"contextual", "comparison"}:
        try:
            history = project_memory.query_history(
                project_id,
                user_msg,
            )
        except Exception:
            history = []

        context = json.dumps(
            history,
            default=str,
            indent=2,
        )

        if intent == "comparison":
            prompt = CHAT_COMPARISON_PROMPT.format(
                context=context,
                question=user_msg,
            )
        else:
            prompt = CHAT_RESPONSE_PROMPT.format(
                context=context,
                question=user_msg,
            )

        try:
            response_text = qwen_client.generate(
                prompt=prompt
            )
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to generate chat response: {exc}",
            )

        db.insert_chat_message(
            project_id=project_id,
            role="assistant",
            content=response_text,
        )

        return ChatResponse(
            message=response_text,
            status="completed",
            actions_taken=["search_project_history"],
        )

    # ---------------------------------------------------------------
    # Deterministic workflows require model + dataset
    # ---------------------------------------------------------------

    if not getattr(project, "model_id", None):
        raise HTTPException(
            status_code=400,
            detail="No model is configured for this project.",
        )

    if not getattr(project, "dataset_id", None):
        raise HTTPException(
            status_code=400,
            detail="No dataset is configured for this project.",
        )

    # ---------------------------------------------------------------
    # Resolve model and dataset directly from authoritative DB state
    #
    # ProjectService currently does not expose get_model/get_dataset.
    # The database contains the authoritative file paths and versions.
    # ---------------------------------------------------------------

    model = db.get_model(project.model_id)

    dataset = db.get_dataset(project.dataset_id)

    if not model:
        raise HTTPException(
            status_code=404,
            detail="Configured model not found.",
        )

    if not dataset:
        raise HTTPException(
            status_code=404,
            detail="Configured dataset not found.",
        )

    model_path = model.get("file_path")
    dataset_path = dataset.get("file_path")
    current_dataset_version = dataset.get("current_version")

    if not model_path:
        raise HTTPException(
            status_code=400,
            detail="Model path is not configured.",
        )

    if not dataset_path:
        raise HTTPException(
            status_code=400,
            detail="Dataset path is not configured.",
        )

    # ---------------------------------------------------------------
    # Create workflow-specific agent state
    # ---------------------------------------------------------------

    state = AgentState(
        project_id=project_id,
        user_goal=user_msg,
        requested_workflow=workflow,
        workflow_objective=user_msg,
        model_id=project.model_id,
        model_path=model_path,
        dataset_id=project.dataset_id,
        dataset_path=dataset_path,
        current_dataset_version=current_dataset_version,
    )

    # ---------------------------------------------------------------
    # Transformation permission
    # ---------------------------------------------------------------

    if not request.allow_transformations:
        state.workflow_objective = (
            f"{user_msg}\n"
            "IMPORTANT: transformations are not allowed for this request."
        )

    # ---------------------------------------------------------------
    # Run deterministic workflow
    # ---------------------------------------------------------------

    try:
        runner = AgentRunner(state)
        final_state = runner.run_until_complete()

    except Exception as exc:
        db.insert_chat_message(
            project_id=project_id,
            role="assistant",
            content=f"Workflow failed: {str(exc)}",
        )

        raise HTTPException(
            status_code=500,
            detail=f"Agent workflow failed: {str(exc)}",
        )

    # ---------------------------------------------------------------
    # Collect results
    # ---------------------------------------------------------------

    actions = _action_names(final_state)

    metrics_summary = _extract_metrics(final_state)

    robustness_metrics = _extract_robustness(final_state)

    # ---------------------------------------------------------------
    # Generate workflow-specific natural-language response
    # ---------------------------------------------------------------

    if workflow == "inspection":
        response_text = (
            "Model and dataset inspection completed."
        )

    elif workflow == "baseline_metrics":
        if metrics_summary:
            response_text = (
                "Baseline evaluation completed. "
                "The metrics above represent the model performance "
                "before any dataset transformation."
            )
        else:
            response_text = (
                "Baseline evaluation completed, but no metrics were returned."
            )

    elif workflow == "targeted_transform":
        response_text = (
            "The requested dataset transformation was completed."
        )

    elif workflow == "improve_dataset":
        response_text = (
            "Dataset improvement workflow completed."
        )

    elif workflow == "report":
        response_text = (
            "The evaluation report workflow has been completed."
        )

    elif workflow == "full_evaluation":
        response_text = (
            "Full evaluation completed, including model and dataset "
            "inspection, baseline evaluation, data-quality analysis, "
            "robustness evaluation, applicable transformations, and "
            "final evaluation."
        )

    else:
        response_text = (
            "The requested workflow has been completed."
        )

    if final_state.status == "failed":
        response_text = (
            f"Workflow failed: "
            f"{final_state.stop_reason or 'Unknown error'}"
        )

    # ---------------------------------------------------------------
    # Save assistant message
    # ---------------------------------------------------------------

    db.insert_chat_message(
        project_id=project_id,
        role="assistant",
        content=response_text,
    )

    # ---------------------------------------------------------------
    # Experiment / report IDs
    # ---------------------------------------------------------------

    experiment_id = None
    report_id = None

    if final_state.experiments:
        last_experiment = final_state.experiments[-1]

        if isinstance(last_experiment, dict):
            experiment_id = (
                last_experiment.get("experiment_id")
                or last_experiment.get("id")
            )

    # Search observations for report information
    for observation in final_state.observations:
        if not isinstance(observation, dict):
            continue

        result = observation.get("result", {})

        if not isinstance(result, dict):
            continue

        if result.get("report_id"):
            report_id = result.get("report_id")

        nested_result = result.get("result")

        if isinstance(nested_result, dict):
            if nested_result.get("report_id"):
                report_id = nested_result.get("report_id")

    # ---------------------------------------------------------------
    # Final response
    # ---------------------------------------------------------------

    return ChatResponse(
        message=response_text,
        status=final_state.status,
        actions_taken=actions,
        current_dataset_version=final_state.current_dataset_version,
        experiment_id=experiment_id,
        report_id=report_id,
        metrics_summary=metrics_summary or None,
        robustness_metrics=robustness_metrics or None,
    )


# -------------------------------------------------------------------
# Chat history
# -------------------------------------------------------------------

@router.get(
    "/projects/{project_id}/chat/history"
)
async def get_chat_history(project_id: str):
    project = project_service.get_project(project_id)

    if not project:
        raise HTTPException(
            status_code=404,
            detail="Project not found.",
        )

    messages = db.get_chat_messages(project_id)

    return {
        "project_id": project_id,
        "messages": messages,
    }