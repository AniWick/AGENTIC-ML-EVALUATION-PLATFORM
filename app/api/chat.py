from fastapi import APIRouter, HTTPException
from typing import Dict, Any, List
import json
from datetime import datetime
from app.schemas.chat import ChatRequest, ChatResponse
from app.services.project_service import project_service
from app.core.database import db
from app.agent.state import AgentState
from app.agent.agent import AgentRunner
from app.memory.chroma_memory import project_memory
from app.llm.qwen_client import qwen_client
from app.llm.prompts import (
    CHAT_RESPONSE_PROMPT,
    CHAT_COMPARISON_PROMPT,
    INTENT_CLASSIFICATION_PROMPT,
)

router = APIRouter(tags=["chat"])

@router.post("/projects/{project_id}/chat", response_model=ChatResponse)
def handle_project_chat(project_id: str, payload: ChatRequest):
    proj = project_service.get_project(project_id)
    if not proj:
        raise HTTPException(status_code=404, detail="Project not found")

    user_msg = payload.message.strip()

    # Record user message in DB
    msg_id = f"msg_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    with db._get_connection() as conn:
        conn.execute(
            "INSERT INTO chat_messages (id, project_id, sender, message, created_at) VALUES (?, ?, ?, ?, ?)",
            (msg_id, project_id, "user", user_msg, datetime.utcnow().isoformat())
        )
        conn.commit()

    # A greeting is deliberately handled locally: it should be immediate and
    # always present the same actionable choices, even when the LLM is offline.
    if user_msg.lower() in {"hi", "hello", "hey", "hii"}:
        reply_text = (
            "Hi! I can help you with:\n"
            "• Evaluate your model and dataset\n"
            "• Inspect model/dataset details\n"
            "• Clean missing values\n"
            "• Balance an imbalanced dataset\n"
            "• Review project history\n\n"
            "Choose one of the action buttons below, or type what you need."
        )
        with db._get_connection() as conn:
            conn.execute(
                "INSERT INTO chat_messages (id, project_id, sender, message, created_at) VALUES (?, ?, ?, ?, ?)",
                (f"msg_bot_{datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')}", project_id, "assistant", reply_text, datetime.utcnow().isoformat())
            )
            conn.commit()
        return ChatResponse(message=reply_text, status="completed", actions_taken=[])

    # ------------------------------------------------------------------
    # Step 1: Classify intent with Qwen
    # ------------------------------------------------------------------
    intent_result = qwen_client.generate_json(
        prompt=INTENT_CLASSIFICATION_PROMPT.format(message=user_msg),
    )
    intent = (intent_result or {}).get("intent", "contextual")

    # Fallback: if Qwen is unreachable, use keyword heuristic
    if not intent_result:
        action_keywords = {
            "evaluate", "improve", "balance", "transform", "run",
            "clean", "fix", "inspect", "start", "do it", "generate report",
        }
        intent = (
            "deterministic"
            if any(k in user_msg.lower() for k in action_keywords)
            else "contextual"
        )

    # ------------------------------------------------------------------
    # Step 2: Route by intent
    # ------------------------------------------------------------------

    # --- DETERMINISTIC: run the agentic workflow ---
    if intent == "deterministic" and proj.model_id and proj.dataset_id:
        with db._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT file_path FROM models WHERE id = ?", (proj.model_id,))
            m_row = cursor.fetchone()
            cursor.execute("SELECT file_path, current_version FROM datasets WHERE id = ?", (proj.dataset_id,))
            d_row = cursor.fetchone()

        if m_row and d_row:
            state = AgentState(
                project_id=project_id,
                user_goal=user_msg,
                model_id=proj.model_id,
                model_path=m_row["file_path"],
                dataset_id=proj.dataset_id,
                dataset_path=d_row["file_path"],
                current_dataset_version=d_row["current_version"]
            )
            runner = AgentRunner(state)
            final_state = runner.run_until_complete()

            action_names = [a.action for a in final_state.actions_taken]
            reply_text = "I evaluated your model and dataset.\n"
            if final_state.baseline_metrics:
                reply_text += (
                    f"- Baseline Accuracy: {final_state.baseline_metrics.get('accuracy', 'N/A')}, "
                    f"Macro F1: {final_state.baseline_metrics.get('f1_macro', 'N/A')}\n"
                )
            if final_state.transformations_applied:
                reply_text += f"- Transformations Applied: {', '.join(final_state.transformations_applied)}\n"
            if final_state.latest_metrics and final_state.latest_metrics != final_state.baseline_metrics:
                reply_text += (
                    f"- Post-Transformation Accuracy: {final_state.latest_metrics.get('accuracy', 'N/A')}, "
                    f"Macro F1: {final_state.latest_metrics.get('f1_macro', 'N/A')}\n"
                )
            reply_text += "A full evaluation report has been compiled and saved."

            with db._get_connection() as conn:
                conn.execute(
                    "INSERT INTO chat_messages (id, project_id, sender, message, created_at) VALUES (?, ?, ?, ?, ?)",
                    (f"msg_bot_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}", project_id, "assistant", reply_text, datetime.utcnow().isoformat())
                )
                conn.commit()

            return ChatResponse(
                message=reply_text,
                status="completed",
                actions_taken=action_names,
                current_dataset_version=final_state.current_dataset_version,
                metrics_summary=final_state.latest_metrics,
                robustness_metrics=final_state.robustness_metrics
            )

    # --- CONTEXTUAL or COMPARISON: answer from report/memory context ---
    memory_results = project_memory.query_history(project_id, user_msg, n_results=5)
    memory_context = (
        "\n".join([m["content"] for m in memory_results])
        if memory_results
        else "No previous experiments recorded."
    )

    if intent == "comparison":
        prompt = CHAT_COMPARISON_PROMPT.format(context=memory_context, question=user_msg)
    else:
        prompt = CHAT_RESPONSE_PROMPT.format(context=memory_context, question=user_msg)

    llm_reply = qwen_client.generate(prompt=prompt, temperature=0.3, max_new_tokens=512)
    if not llm_reply or len(llm_reply.strip()) == 0:
        llm_reply = f"Based on project memory:\n{memory_context}"

    with db._get_connection() as conn:
        conn.execute(
            "INSERT INTO chat_messages (id, project_id, sender, message, created_at) VALUES (?, ?, ?, ?, ?)",
            (f"msg_bot_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}", project_id, "assistant", llm_reply, datetime.utcnow().isoformat())
        )
        conn.commit()

    return ChatResponse(
        message=llm_reply,
        status="completed",
        actions_taken=["search_project_history"]
    )

@router.get("/projects/{project_id}/chat/history")
def get_chat_history(project_id: str):
    with db._get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM chat_messages WHERE project_id = ? ORDER BY created_at ASC", (project_id,))
        rows = cursor.fetchall()
        return [{"id": r["id"], "sender": r["sender"], "message": r["message"], "created_at": r["created_at"]} for r in rows]
