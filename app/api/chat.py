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
from app.llm.ollama_client import ollama_client
from app.llm.prompts import CHAT_RESPONSE_PROMPT

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

    # Determine if this is an execution request or conversational query
    is_action_prompt = any(k in user_msg.lower() for k in [
        "evaluate", "improve", "balance", "transform", "run", "clean", "fix", "inspect", "start", "do it"
    ])

    if is_action_prompt and proj.model_id and proj.dataset_id:
        # Retrieve model and dataset paths
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

            # Compile conversational response
            action_names = [a.action for a in final_state.actions_taken]
            reply_text = f"I evaluated your model and dataset.\n"
            if final_state.baseline_metrics:
                reply_text += f"- Baseline Accuracy: {final_state.baseline_metrics.get('accuracy', 'N/A')}, Macro F1: {final_state.baseline_metrics.get('f1_macro', 'N/A')}\n"
            if final_state.transformations_applied:
                reply_text += f"- Transformations Applied: {', '.join(final_state.transformations_applied)}\n"
            if final_state.latest_metrics and final_state.latest_metrics != final_state.baseline_metrics:
                reply_text += f"- Post-Transformation Accuracy: {final_state.latest_metrics.get('accuracy', 'N/A')}, Macro F1: {final_state.latest_metrics.get('f1_macro', 'N/A')}\n"
            
            reply_text += "A full evaluation report has been compiled and saved."

            # Save assistant message
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
                metrics_summary=final_state.latest_metrics
            )

    # Conversational Q&A / Memory Retrieval
    memory_results = project_memory.query_history(project_id, user_msg, n_results=3)
    memory_context = "\n".join([m["content"] for m in memory_results]) if memory_results else "No previous experiments recorded."

    prompt = CHAT_RESPONSE_PROMPT.format(context=memory_context, question=user_msg)
    llm_reply = ollama_client.generate(prompt=prompt)
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

