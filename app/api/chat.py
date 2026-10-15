# =====================================================================
# Updated chat_4.py
# =====================================================================
from fastapi import APIRouter, HTTPException
from typing import Dict, Any, List
import json
import logging

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
logger = logging.getLogger(__name__)


from app.agent.security_aliases import (
    is_hardening_request as _is_hardening_request,
    is_compound_defended_attack_request as _is_compound_defended_attack_request,
    is_hardening_loop_request as _is_hardening_loop_request,
)


def _is_report_request(message: str) -> bool:
    normalized = (message or "").lower()
    return "report" in normalized and any(
        word in normalized
        for word in ("generate", "create", "make", "show", "write", "detailed")
    )


def _get_active_session_context(project_id: str) -> Dict[str, Any]:
    project = project_service.get_project(project_id)
    if not project:
        return {}

    active_model = None
    active_dataset = None
    dataset_metadata = {}

    if getattr(project, "model_id", None):
        try:
            active_model = db.get_model(project.model_id)
        except Exception:
            active_model = None

    if getattr(project, "dataset_id", None):
        try:
            active_dataset = db.get_dataset(project.dataset_id)
            if active_dataset and active_dataset.get("id"):
                with db._get_connection() as conn:
                    row = conn.execute(
                        """
                        SELECT row_count, column_count, operation, parameters 
                        FROM dataset_versions 
                        WHERE dataset_id = ? 
                        ORDER BY created_at DESC LIMIT 1
                        """,
                        (active_dataset["id"],)
                    ).fetchone()
                    if row:
                        dataset_metadata = dict(row)
        except Exception:
            active_dataset = None

    return {
        "active_model": active_model,
        "active_dataset": active_dataset,
        "active_dataset_metadata": dataset_metadata,
    }


def _get_global_project_memory(project_id: str) -> Dict[str, Any]:
    models = db.list_models()
    datasets = db.list_datasets()

    try:
        chat_history = db.get_chat_messages(project_id)
    except Exception:
        chat_history = []

    try:
        experiments = db.get_all_experiments_for_project(project_id)
    except Exception:
        experiments = []

    reports = []
    try:
        with db._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT id, title, summary, content_markdown, content_json, created_at
                FROM reports
                WHERE project_id = ?
                ORDER BY created_at DESC
                """,
                (project_id,)
            )
            rows = cursor.fetchall()
            for row in rows:
                reports.append({
                    "id": row["id"],
                    "title": row["title"],
                    "summary": row["summary"],
                    "content_markdown": row["content_markdown"],
                    "created_at": row["created_at"],
                })
    except Exception:
        reports = []

    return {
        "registered_models": models,
        "registered_datasets": datasets,
        "experiment_lineage": experiments,
        "project_reports": reports,
        "recent_chat_turns": chat_history[-6:] if chat_history else [],
    }


def _is_too_vague(message: str) -> bool:
    msg = message.lower().strip()
    words = msg.split()

    if len(words) > 6:
        return False

    concrete_terms = {
        "inspect", "baseline", "evaluate", "improve", "compare",
        "report", "clean", "balance", "outlier", "outliers", "scale",
        "noise", "smote", "accuracy", "f1", "robust", "robustness",
        "missing", "dataset", "model", "metrics", "version", "why", "how",
        "attack", "defend", "gaussian_augmentation", "spatial_smoothing", "feature_squeezing", "clip_values", "reverse_sigmoid", "class_labels", "high_confidence_masking", "boundary_attack", "hopskipjump", "zoo", "fgsm", "pgd", "cw", "adversarial", "boundary", "hopskipjump", "zoo",
        "fgsm", "pgd", "cw", "gaussian", "augmentation", "spatial", "smoothing",
        "feature", "squeezing", "clip_values", "reverse_sigmoid", "class_labels",
        "high_confidence_masking"
    }
    if any(term in msg for term in concrete_terms):
        return False

    vague_terms = {
        "help", "do something", "check", "look", "fix",
        "improve it", "what's up", "analyze", "hmm", "idk",
    }

    return any(term in msg for term in vague_terms) or len(words) <= 2


def _deterministic_history_summary(
    history: List[Dict[str, Any]],
    question: str,
) -> str:
    if not history:
        return (
            "No recorded experiment history was found for this project. "
            "To get a detailed explanation, run an evaluation, attack, defense, "
            "or transformation workflow first."
        )

    msg_lower = question.lower()

    target_ops = [
        op for op in ["fgsm", "pgd", "cw", "zoo", "boundary_attack", "hopskipjump",
                      "gaussian_augmentation", "spatial_smoothing", "feature_squeezing",
                      "clip_values", "reverse_sigmoid", "class_labels", "high_confidence_masking"]
        if op in msg_lower
    ]

    records_to_process = history
    if target_ops:
        filtered = [
            r for r in history
            if (r.get("metadata", r) if isinstance(r.get("metadata"), dict) else r).get("operation") in target_ops
        ]
        if filtered:
            records_to_process = filtered

    seen_ops = set()
    deduped_history = []
    for r in reversed(records_to_process):
        if not isinstance(r, dict):
            continue
        meta = r.get("metadata", r) if isinstance(r.get("metadata"), dict) else r
        op = meta.get("operation") or meta.get("action") or "dataset_transformation"

        if op not in seen_ops:
            seen_ops.add(op)
            deduped_history.append(r)

    deduped_history.reverse()
    lines = ["Here is the detailed evaluation analysis based on recorded project experiments:\n"]

    for i, record in enumerate(deduped_history, 1):
        meta = record.get("metadata", record) if isinstance(record.get("metadata"), dict) else record

        op = meta.get("operation") or meta.get("action") or "dataset_transformation"
        inp = meta.get("input_dataset_version") or meta.get("input_version") or "v001"
        out = meta.get("output_dataset_version") or meta.get("output_version") or "v002"

        before = meta.get("metrics_before", {})
        after = meta.get("metrics_after", {})

        if isinstance(before, str):
            try: before = json.loads(before)
            except Exception: before = {}
        if isinstance(after, str):
            try: after = json.loads(after)
            except Exception: after = {}

        lines.append(f"{i}. Experiment Operation: {op}")
        lines.append(f"   Version Transition: {inp} -> {out}")

        if before and after:
            acc_b = float(before.get("accuracy", 0))
            acc_a = float(after.get("accuracy", 0))
            delta_acc = acc_a - acc_b

            f1_b = float(before.get("f1_macro", before.get("macro_f1", 0)))
            f1_a = float(after.get("f1_macro", after.get("macro_f1", 0)))
            delta_f1 = f1_a - f1_b

            is_security_op = op in {
                "boundary_attack", "hopskipjump", "zoo", "fgsm", "pgd", "cw",
                "gaussian_augmentation", "spatial_smoothing", "feature_squeezing",
                "clip_values", "reverse_sigmoid", "class_labels", "high_confidence_masking"
            }

            if is_security_op:
                lines.append(f"   Accuracy Impact: {acc_b:.3f} -> {acc_a:.3f} ({delta_acc:+.3f})")
                is_defense = op in {
                    "gaussian_augmentation", "spatial_smoothing", "feature_squeezing",
                    "clip_values", "reverse_sigmoid", "class_labels", "high_confidence_masking"
                } or any(k in op for k in ["augmentation", "smoothing", "squeezing", "clip", "sigmoid", "labels", "masking"])
                if is_defense:
                    lines.append(
                        f"   Analysis: This was a defensive measure ({op}). "
                        "The accuracy shown reflects the model's performance on "
                        "defense-perturbed inputs versus clean inputs."
                    )
                else:
                    lines.append(
                        f"   Analysis: This was an adversarial attack ({op}). The accuracy drop "
                        "reflects the model's vulnerability to crafted adversarial inputs, not "
                        "ordinary noise or data-quality issues."
                    )
            else:
                lines.append(f"   Accuracy Impact: {acc_b:.3f} -> {acc_a:.3f} ({delta_acc:+.3f})")
                lines.append(f"   Macro F1 Impact: {f1_b:.3f} -> {f1_a:.3f} ({delta_f1:+.3f})")

                if delta_acc < 0:
                    lines.append(
                        "   Analysis: Performance decreased slightly. On small or easily separable datasets, "
                        "applying transformations like synthetic oversampling or aggressive filtering can introduce noise "
                        "or blur decision boundaries."
                    )
                elif delta_acc > 0:
                    lines.append(
                        "   Analysis: Performance improved. The applied transformation successfully resolved "
                        "data quality issues and enhanced overall model generalization."
                    )
                else:
                    lines.append(
                        "   Analysis: Performance remained unchanged. The transformation had a neutral effect."
                    )
        lines.append("")

    return "\n".join(lines)


def _parse_router_response(raw: Any, message: str) -> Dict[str, str]:
    if isinstance(raw, dict):
        data = raw
    else:
        try:
            data = json.loads(raw)
        except Exception:
            data = {}

    intent = data.get("intent")
    workflow = data.get("workflow")
    msg = message.lower().strip()

    if _is_hardening_loop_request(msg):
        return {"intent": "deterministic", "workflow": "harden_defense_loop"}

    # These compound intents have deterministic semantics and must not be
    # overridden by a valid but conflicting LLM route.
    if _is_hardening_request(msg):
        return {"intent": "deterministic", "workflow": "harden_model"}

    if _is_compound_defended_attack_request(msg):
        return {
            "intent": "deterministic",
            "workflow": "attack_defended_model",
        }

    if _is_report_request(msg):
        return {"intent": "deterministic", "workflow": "report"}

    valid_intents = {"deterministic", "contextual", "comparison"}
    valid_workflows = {
        "inspection", "baseline_metrics", "full_evaluation", "improve_dataset",
        "targeted_transform", "report", "contextual", "comparison", "attack", "defend", "gaussian_augmentation", "spatial_smoothing", "feature_squeezing", "clip_values", "reverse_sigmoid", "class_labels", "high_confidence_masking", "boundary_attack", "hopskipjump", "zoo", "fgsm", "pgd", "cw",
        "boundary_attack", "hopskipjump", "zoo", "fgsm", "pgd", "cw",
        "gaussian_augmentation", "spatial_smoothing", "feature_squeezing",
        "clip_values", "reverse_sigmoid", "class_labels", "high_confidence_masking",
        "attack_defended_model", "defended_attack", "harden_model", "harden",
        "harden_defense_loop", "hardening_loop",
    }
    if intent not in valid_intents:
        intent = None

    if workflow not in valid_workflows:
        workflow = None

    if intent and workflow:
        return {"intent": intent, "workflow": workflow}

    if _is_hardening_loop_request(msg):
        return {"intent": "deterministic", "workflow": "harden_defense_loop"}

    if _is_hardening_request(msg):
        return {"intent": "deterministic", "workflow": "harden_model"}

    if _is_compound_defended_attack_request(msg):
        return {
            "intent": "deterministic",
            "workflow": "attack_defended_model",
        }

    if _is_report_request(msg):
        return {"intent": "deterministic", "workflow": "report"}

    if (
        ("inspect" in msg and ("model" in msg or "dataset" in msg))
        or "inspect my model" in msg
        or "inspect my dataset" in msg
        or "model and dataset details" in msg
    ):
        return {"intent": "deterministic", "workflow": "inspection"}

    if ("baseline" in msg or "before transformation" in msg or "original metrics" in msg):
        return {"intent": "deterministic", "workflow": "baseline_metrics"}

    if any(phrase in msg for phrase in ["clean missing", "handle missing", "missing values"]):
        return {"intent": "deterministic", "workflow": "targeted_transform"}

    if any(phrase in msg for phrase in ["balance dataset", "handle imbalance", "smote"]):
        return {"intent": "deterministic", "workflow": "targeted_transform"}

    if any(phrase in msg for phrase in ["handle outliers", "remove outliers"]):
        return {"intent": "deterministic", "workflow": "targeted_transform"}

    if any(phrase in msg for phrase in ["scale dataset", "apply scaling", "normalize dataset"]):
        return {"intent": "deterministic", "workflow": "targeted_transform"}

    if any(phrase in msg for phrase in ["add noise", "gaussian noise"]):
        return {"intent": "deterministic", "workflow": "targeted_transform"}

    if any(phrase in msg for phrase in ["improve my dataset", "improve dataset", "optimize dataset"]):
        return {"intent": "deterministic", "workflow": "improve_dataset"}

    if any(phrase in msg for phrase in ["evaluate my model", "evaluate model", "run evaluation", "full evaluation"]):
        return {"intent": "deterministic", "workflow": "full_evaluation"}

    if any(phrase in msg for phrase in [
        "compare", "better", "did smote help", "was balancing worth it", "before and after", 
        "explain evaluation", "explain metrics", "what happened", "explain the attack", "explain the defense"
    ]):
        return {"intent": "comparison", "workflow": "comparison"}
    
    if any(phrase in msg for phrase in [
        "boundary_attack", "hopskipjump", "zoo", "fgsm", "pgd", "cw",
        "boundary attack", "hop skip jump", "zoo attack", "execute attack", "run attack"
    ]):
        return {"intent": "deterministic", "workflow": "attack"}

    if any(phrase in msg for phrase in [
        "defend", "gaussian_augmentation", "gaussian augmentation", "spatial_smoothing", "feature_squeezing",
        "clip_values", "clip values", "reverse_sigmoid", "reverse sigmoid", "class_labels", "class labels",
        "high_confidence_masking", "high confidence masking", "high confidence"
    ]):
        return {"intent": "deterministic", "workflow": "defend"}

    return {"intent": "contextual", "workflow": "contextual"}


def _extract_metrics(state: AgentState) -> Dict[str, Any]:
    if state.latest_metrics:
        return state.latest_metrics
    if state.baseline_metrics:
        return state.baseline_metrics
    return {}


def _extract_robustness(state: AgentState) -> Dict[str, Any]:
    if state.robustness_results:
        return state.robustness_results
    if getattr(state, "attack_results", None):
        return state.attack_results
    if getattr(state, "defense_results", None):
        return state.defense_results
    return {}


def _action_names(state: AgentState) -> List[str]:
    result = []
    for item in state.actions_taken:
        if isinstance(item, dict):
            action = item.get("action")
            if action:
                result.append(action)
        elif isinstance(item, str):
            result.append(item)
    return result


def _format_attack_response(state: AgentState) -> str:
    ar = getattr(state, "attack_results", None) or {}

    if ar.get("status") == "error":
        return (
            f"The {ar.get('attack_type', 'adversarial')} attack failed to complete: "
            f"{ar.get('error', 'unknown error')}."
        )

    if not state.adversarial_evaluated:
        return "The adversarial attack workflow has been completed."

    attack_label = (state.attack_type or "adversarial").replace("_", " ").title()
    clean = state.clean_accuracy
    adv = state.adversarial_accuracy
    drop = ar.get("drop_percentage")

    if clean is None or adv is None:
        return f"{attack_label} attack completed."

    parts = [
        f"{attack_label} attack completed.",
        f"Clean accuracy: {clean * 100:.2f}%.",
        f"Adversarial accuracy: {adv * 100:.2f}%.",
    ]
    if drop is not None:
        parts.append(f"Performance dropped by {drop:.2f}% under attack.")

    return " ".join(parts)


def _format_defend_response(state: AgentState) -> str:
    dr = getattr(state, "defense_results", None) or {}

    if dr.get("status") == "error":
        return f"The defense failed to complete: {dr.get('error', 'unknown error')}."

    if not state.defended:
        return "The defense workflow has been completed."

    clean = dr.get("clean_accuracy")
    defended = dr.get("defended_accuracy")

    if clean is None or defended is None:
        return "Defense applied successfully."

    return (
        f"Defense applied. "
        f"Clean accuracy: {clean * 100:.2f}%. "
        f"Defended accuracy: {defended * 100:.2f}%."
    )


@router.post(
    "/projects/{project_id}/chat",
    response_model=ChatResponse,
)
async def chat_with_project(
    project_id: str,
    request: ChatRequest,
):
    user_msg = request.message.strip()
    response_text = ""

    if not user_msg:
        raise HTTPException(
            status_code=400,
            detail="Message cannot be empty.",
        )

    project = project_service.get_project(project_id)

    if not project:
        raise HTTPException(
            status_code=404,
            detail="Project not found.",
        )

    # Every accepted request is retained before any workflow branch executes.
    db.insert_chat_message(project_id=project_id, role="user", content=user_msg)

    greetings = {
        "hi", "hello", "hey", "hi there", "hello there",
        "good morning", "good afternoon", "good evening",
    }

    if user_msg.lower() in greetings:
        response_text = (
            "Hello! I can inspect your model and dataset, run evaluations, "
            "improve the dataset, run adversarial attacks or defenses, compare "
            "results, or answer questions about your evaluation history."
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

    if _is_too_vague(user_msg):
        clarification_text = (
            "Could you be more specific? For example, I can:\n"
            "1. Inspect your model and dataset\n"
            "2. Run a full evaluation\n"
            "3. Improve the dataset (clean/balance/scale as needed)\n"
            "4. Run an adversarial attack or a defense\n"
            "5. Compare previous results\n"
            "6. Generate a report\n"
            "Which of these did you mean?"
        )

        db.insert_chat_message(
            project_id=project_id,
            role="assistant",
            content=clarification_text,
        )

        return ChatResponse(
            message=clarification_text,
            status="completed",
            actions_taken=[],
        )

    try:
        routing_prompt = INTENT_CLASSIFICATION_PROMPT.format(message=user_msg)
        routing_result = qwen_client.generate_json(prompt=routing_prompt)
        route = _parse_router_response(routing_result, user_msg)
    except Exception:
        route = _parse_router_response({}, user_msg)

    intent = route["intent"]
    workflow = route["workflow"]

    if intent in {"contextual", "comparison"}:
        session_ctx = _get_active_session_context(project_id)
        global_mem = _get_global_project_memory(project_id)
        sqlite_history = db.get_all_experiments_for_project(project_id)

        msg_lower = user_msg.lower()

        ALL_SECURITY_OPS = [
            "fgsm", "pgd", "cw", "zoo", "boundary_attack", "hopskipjump",
            "gaussian_augmentation", "spatial_smoothing", "feature_squeezing",
            "clip_values", "reverse_sigmoid", "class_labels", "high_confidence_masking",
        ]

        mentioned_op = next((op for op in ALL_SECURITY_OPS if op in msg_lower), None)

        recency_phrases = [
            "above", "that attack", "this attack", "that defense", "this defense",
            "just ran", "just did", "just now", "last attack", "last defense",
            "most recent", "that one", "this one", "the last one",
        ]
        is_recency_reference = (
            mentioned_op is None
            and any(p in msg_lower for p in recency_phrases)
        )

        is_past_run_query = is_recency_reference or any(p in msg_lower for p in [
            "experiment history","past attack result","summarize my runs",
            "past run results","runs until now","list my past attacks",
        ]) or (mentioned_op and any(w in msg_lower for w in ["result","ran","performed"]))
        if is_past_run_query:
            src = sqlite_history[-1:] if is_recency_reference else sqlite_history
            response_text = _deterministic_history_summary(src or [], user_msg)
            db.insert_chat_message(project_id=project_id, role="assistant", content=response_text)
            return ChatResponse(message=response_text, status="completed", actions_taken=["search_project_history"])

        try:
            chroma_history = project_memory.query_history(project_id, user_msg)
        except Exception:
            chroma_history = []

        active_model = session_ctx.get("active_model")
        active_dataset = session_ctx.get("active_dataset")
        active_dataset_meta = session_ctx.get("active_dataset_metadata", {})

        global_models_list = []
        for m in global_mem.get("registered_models", []):
            if isinstance(m, dict):
                name = m.get("name") or m.get("id")
                framework = m.get("framework", "sklearn")
                global_models_list.append(f"- Name: {name} (ID: {m.get('id')}, Framework: {framework})")

        global_datasets_list = []
        for d in global_mem.get("registered_datasets", []):
            if isinstance(d, dict):
                name = d.get("name") or d.get("id")
                version = d.get("current_version", "v001")
                global_datasets_list.append(f"- Name: {name} (ID: {d.get('id')}, Version: {version})")

        context_payload = {
            "currently_selected_in_session": {
                "model": f"{active_model.get('name')} ({active_model.get('id')})" if isinstance(active_model, dict) else "None selected",
                "dataset": f"{active_dataset.get('name')} ({active_dataset.get('id')})" if isinstance(active_dataset, dict) else "None selected",
                "dataset_dimensions": active_dataset_meta,
            },
            "global_memory_registered_models": global_models_list,
            "global_memory_registered_datasets": global_datasets_list,
            "recorded_experiments": sqlite_history,
            "evaluation_reports": global_mem.get("project_reports", []),
            "recent_chat_turn_history": global_mem.get("recent_chat_turns", []),
            "semantic_vector_matches": chroma_history,
        }

        context_str = json.dumps(context_payload, default=str, indent=2)

        if intent == "comparison":
            prompt = CHAT_COMPARISON_PROMPT.format(context=context_str, question=user_msg)
        else:
            prompt = CHAT_RESPONSE_PROMPT.format(context=context_str, question=user_msg)

        try:
            response_text = qwen_client.generate(prompt=prompt)
        except Exception:
            response_text = ""

        if not response_text or not response_text.strip():
            response_text = _deterministic_history_summary(sqlite_history, user_msg)

        db.insert_chat_message(project_id=project_id, role="assistant", content=response_text)

        return ChatResponse(
            message=response_text,
            status="completed",
            actions_taken=["search_project_history"],
        )

    if not getattr(project, "model_id", None):
        raise HTTPException(status_code=400, detail="No model is configured for this project.")

    if not getattr(project, "dataset_id", None):
        raise HTTPException(status_code=400, detail="No dataset is configured for this project.")

    model = db.get_model(project.model_id)
    dataset = db.get_dataset(project.dataset_id)

    if not model or not dataset:
        raise HTTPException(status_code=404, detail="Configured model or dataset not found.")

    model_path = model.get("file_path")
    dataset_path = dataset.get("file_path")
    current_dataset_version = dataset.get("current_version")

    if not model_path or not dataset_path:
        raise HTTPException(status_code=400, detail="Model or dataset path is not configured.")

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

    if not request.allow_transformations:
        state.workflow_objective = (
            f"{user_msg}\nIMPORTANT: transformations are not allowed for this request."
        )

    try:
        runner = AgentRunner(state)
        final_state = runner.run_until_complete()
    except Exception:
        logger.exception("Workflow failed for project %s", project_id)
        safe_error = "The workflow could not be completed. Please try again later."
        db.insert_chat_message(project_id=project_id, role="assistant", content=safe_error)
        raise HTTPException(status_code=500, detail=safe_error)

    actions = _action_names(final_state)
    metrics_summary = _extract_metrics(final_state)
    robustness_metrics = _extract_robustness(final_state)

    if workflow == "inspection":
        response_text = "Model and dataset inspection completed."
    elif workflow == "baseline_metrics":
        response_text = "Baseline evaluation completed." if metrics_summary else "Baseline evaluation completed, but no metrics were returned."
    elif workflow == "targeted_transform":
        response_text = "The requested dataset transformation was completed."
    elif workflow == "improve_dataset":
        response_text = "Dataset improvement workflow completed."
    elif workflow == "report":
        response_text = "The evaluation report workflow has been completed."
    elif workflow == "full_evaluation":
        response_text = "Full evaluation completed."
    elif workflow in {"attack_defended_model", "defended_attack"}:
        response_text = _format_defended_attack_response(final_state)
    elif workflow in {"harden_defense_loop", "hardening_loop"}:
        response_text = _format_hardening_loop_response(final_state)
    elif workflow in {"harden_model", "harden"}:
        response_text = _format_harden_response(final_state)
    elif workflow in {"attack", "boundary_attack", "hopskipjump", "zoo", "fgsm", "pgd", "cw"}:
        response_text = _format_attack_response(final_state)
    elif workflow in {
        "defend", "gaussian_augmentation", "spatial_smoothing", "feature_squeezing",
        "clip_values", "reverse_sigmoid", "class_labels", "high_confidence_masking",
    }:
        response_text = _format_defend_response(final_state)
    else:
        response_text = "The requested workflow has been completed."

    if final_state.status == "failed":
        response_text = f"Workflow failed: {final_state.stop_reason or 'Unknown error'}"

    db.insert_chat_message(project_id=project_id, role="assistant", content=response_text)

    experiment_id = None
    report_id = None

    if final_state.experiments:
        last_experiment = final_state.experiments[-1]
        if isinstance(last_experiment, dict):
            experiment_id = last_experiment.get("experiment_id") or last_experiment.get("id")

    for observation in final_state.observations:
        if isinstance(observation, dict):
            result = observation.get("result", {})
            if isinstance(result, dict):
                if result.get("report_id"):
                    report_id = result.get("report_id")
                nested = result.get("result")
                if isinstance(nested, dict) and nested.get("report_id"):
                    report_id = nested.get("report_id")

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


@router.get("/projects/{project_id}/chat/history")
async def get_chat_history(project_id: str):
    project = project_service.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found.")
    messages = db.get_chat_messages(project_id)
    return {"project_id": project_id, "messages": messages}

def _format_defended_attack_response(state: AgentState) -> str:
    sequence = getattr(state, "attack_defense_sequence_results", None) or {}
    if sequence.get("steps"):
        return _format_attack_defense_sequence_response(sequence)

    res = getattr(state, "defended_attack_results", None) or {}
    if res.get("status") == "error":
        return f"Defended attack failed: {res.get('error', 'unknown error')}."

    clean = res.get("clean_accuracy")
    defended = res.get("defended_accuracy")
    adv_def = res.get("adversarial_defended_accuracy")

    parts = ["Defended model attack completed."]
    if clean is not None:
        parts.append(f"Clean accuracy: {clean * 100:.2f}%.")
    if defended is not None:
        parts.append(f"Defended baseline accuracy: {defended * 100:.2f}%.")
    if adv_def is not None:
        parts.append(f"Adversarial accuracy under defense: {adv_def * 100:.2f}%.")
    return " ".join(parts)


def _pct(value) -> str:
    if value is None:
        return "n/a"
    return f"{min(100.0, max(0.0, float(value) * 100)):.2f}%"


def _format_attack_defense_sequence_response(sequence: Dict[str, Any]) -> str:
    steps = sequence.get("steps") or []
    lines = [
        f"Compared {sequence.get('step_count')} attack/defense combinations "
        f"against the same {sequence.get('framework', 'model')} model "
        f"(up to {sequence.get('max_steps')} allowed)."
    ]
    if sequence.get("requested_steps") not in (None, sequence.get("step_count")):
        lines.append(
            f"Only {sequence.get('step_count')} of {sequence.get('requested_steps')} "
            f"requested combinations were named."
        )
    for step in steps:
        lines.append(
            f"Step {step.get('step_index')} - {step.get('attack_type') or 'no attack'} "
            f"vs {step.get('defense_type') or 'no defense'}: "
            f"clean {_pct(step.get('clean_accuracy'))}, "
            f"defended {_pct(step.get('defended_accuracy'))}, "
            f"adversarial {_pct(step.get('adversarial_accuracy'))} "
            f"(P {_pct(step.get('adversarial_precision'))}, "
            f"R {_pct(step.get('adversarial_recall'))}, "
            f"F1 {_pct(step.get('adversarial_f1'))})."
        )
    ranking = sequence.get("ranking") or {}
    strongest = ranking.get("strongest_attack") or {}
    best = ranking.get("best_defense") or {}
    if strongest.get("attack_type"):
        lines.append(
            f"Strongest attack: {strongest['attack_type']} at "
            f"{_pct(strongest.get('mean_adversarial_accuracy'))} mean adversarial accuracy."
        )
    if best.get("defense_type"):
        lines.append(
            f"Best defense: {best['defense_type']} at "
            f"{_pct(best.get('mean_defended_accuracy'))} mean defended accuracy."
        )
    lines.append(
        "These combinations are independent measurements against the same model; "
        "the defense is an input preprocessor and is not written back, so they "
        "do not stack."
    )
    return "\n".join(lines)


def _format_hardening_loop_response(state: AgentState) -> str:
    loop = getattr(state, "hardening_loop_results", None) or {}
    if loop.get("status") == "error":
        return f"Hardening loop failed: {loop.get('error', 'unknown error')}."
    rounds = loop.get("rounds") or []
    if not rounds:
        return "The hardening loop did not produce any rounds."

    lines = [
        f"Ran {loop.get('round_count')} cumulative hardening rounds using "
        f"{loop.get('retrain_method')} adversarial retraining."
    ]
    for entry in rounds:
        lines.append(
            f"Round {entry.get('round_index')} - {entry.get('attack_type') or 'no attack'} "
            f"vs {entry.get('defense_type') or 'no defense'}: "
            f"clean {_pct(entry.get('clean_accuracy'))}, "
            f"adversarial {_pct(entry.get('adversarial_accuracy'))} "
            f"(P {_pct(entry.get('adversarial_precision'))}, "
            f"R {_pct(entry.get('adversarial_recall'))}, "
            f"F1 {_pct(entry.get('adversarial_f1'))})."
        )
    improvement = loop.get("improvement") or {}
    delta = improvement.get("delta")
    if delta is not None and not improvement.get("comparable", True):
        lines.append(
            "This was a mixed-threat run: each round measured a different attack, "
            "so the first and final numbers are not directly comparable."
        )
        for name, stats in (loop.get("per_attack_change") or {}).items():
            if stats.get("rounds", 0) > 1:
                lines.append(
                    f"  {name}: {_pct(stats.get('first_adversarial_accuracy'))} -> "
                    f"{_pct(stats.get('last_adversarial_accuracy'))} "
                    f"({stats.get('delta', 0) * 100:+.2f} points)."
                )
    elif delta is not None:
        direction = "improved" if delta > 0 else "declined" if delta < 0 else "held steady"
        lines.append(
            f"Adversarial accuracy {direction} from "
            f"{_pct(improvement.get('first_adversarial_accuracy'))} to "
            f"{_pct(improvement.get('final_adversarial_accuracy'))} "
            f"({delta * 100:+.2f} points)."
        )
    if loop.get("final_model_id"):
        lines.append(f"Final hardened model registered: {loop.get('final_model_id')}.")
    return "\n".join(lines)

def format_agent_chat_response(state, step_results) -> str:
    # Check if report generation was executed in the actions taken
    report_generated = any(
        action.get("action") == "generate_report"
        for action in state.actions_taken
    )

    # Get standard security/attack summary
    base_response = _format_defended_attack_response(state)

    if report_generated:
        base_response += (
            "\n\n📄 **Evaluation Report Generated Successfully!** "
            "Your comprehensive markdown report, including model architecture, "
            "dataset features, and the defended attack matrix, is now ready "
            "in the **Reports** tab."
        )

    return base_response

def _format_harden_response(state: AgentState) -> str:
    hr = getattr(state, "hardened_model_info", None) or {}
    if hr.get("status") == "error":
        return f"Model hardening failed: {hr.get('error', 'unknown error')}."

    model_id = hr.get("model_id", "New Model")
    summary = hr.get("training_summary", {})
    method = summary.get("method", "adversarial retraining")
    samples = summary.get("samples_generated", 0)

    return (
        f"Model hardening completed using {method.upper()} adversarial retraining. "
        f"Generated {samples} adversarial samples. "
        f"Registered hardened model asset: {model_id}."
    )

@router.get("/projects/{project_id}/security-experiments")
async def get_security_experiments(project_id: str):
    project = project_service.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found.")
    
    raw_experiments = db.get_all_experiments_for_project(project_id)
    experiments = []
    
    for exp in raw_experiments:
        exp_dict = dict(exp) if hasattr(exp, "_asdict") or isinstance(exp, dict) else dict(exp.__dict__)
        for field in ["metrics_before", "metrics_after", "comparison", "parameters"]:
            if field in exp_dict and isinstance(exp_dict[field], str):
                try:
                    exp_dict[field] = json.loads(exp_dict[field])
                except Exception:
                    pass
        experiments.append(exp_dict)
        
    return {"project_id": project_id, "experiments": experiments}
