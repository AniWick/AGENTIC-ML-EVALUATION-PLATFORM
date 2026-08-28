from fastapi import APIRouter, HTTPException
from typing import Dict, Any, Optional
from app.services.project_service import project_service
from app.core.database import db
from app.agent.state import AgentState
from app.agent.agent import AgentRunner
from app.datasets.versioning import dataset_version_manager

router = APIRouter(tags=["evaluations"])

@router.post("/projects/{project_id}/evaluate")
def run_project_evaluation(project_id: str):
    proj = project_service.get_project(project_id)
    if not proj:
        raise HTTPException(status_code=404, detail="Project not found")
    if not proj.model_id or not proj.dataset_id:
        raise HTTPException(status_code=400, detail="Project requires both an uploaded model and dataset to run evaluation.")

    # Retrieve model and dataset paths
    with db._get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT file_path FROM models WHERE id = ?", (proj.model_id,))
        m_row = cursor.fetchone()
        cursor.execute("SELECT file_path, current_version FROM datasets WHERE id = ?", (proj.dataset_id,))
        d_row = cursor.fetchone()

    if not m_row or not d_row:
        raise HTTPException(status_code=400, detail="Model or dataset files not found on record")

    state = AgentState(
        project_id=project_id,
        user_goal="Run full evaluation and optimize data quality",
        model_id=proj.model_id,
        model_path=m_row["file_path"],
        dataset_id=proj.dataset_id,
        dataset_path=d_row["file_path"],
        current_dataset_version=d_row["current_version"]
    )

    runner = AgentRunner(state)
    final_state = runner.run_until_complete()

    return {
        "status": final_state.status,
        "baseline_metrics": final_state.baseline_metrics,
        "latest_metrics": final_state.latest_metrics,
        "transformations": final_state.transformations_applied,
        "actions_taken": [a.action for a in final_state.actions_taken],
        "current_dataset_version": final_state.current_dataset_version
    }

