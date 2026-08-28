from fastapi import APIRouter, HTTPException
from typing import List, Dict, Any
import json
from app.core.database import db

router = APIRouter(tags=["experiments"])

@router.get("/projects/{project_id}/experiments")
def get_project_experiments(project_id: str):
    with db._get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM experiments WHERE project_id = ? ORDER BY created_at DESC", (project_id,))
        rows = cursor.fetchall()
        experiments = []
        for r in rows:
            experiments.append({
                "id": r["id"],
                "project_id": r["project_id"],
                "model_id": r["model_id"],
                "input_dataset_version": r["input_dataset_version"],
                "output_dataset_version": r["output_dataset_version"],
                "operation": r["operation"],
                "parameters": json.loads(r["parameters"] or "{}"),
                "metrics_before": json.loads(r["metrics_before"] or "{}"),
                "metrics_after": json.loads(r["metrics_after"] or "{}"),
                "comparison": json.loads(r["comparison"] or "{}"),
                "agent_reasoning": r["agent_reasoning"],
                "created_at": r["created_at"]
            })
        return experiments

@router.get("/experiments/{experiment_id}")
def get_experiment_detail(experiment_id: str):
    with db._get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM experiments WHERE id = ?", (experiment_id,))
        r = cursor.fetchone()
        if not r:
            raise HTTPException(status_code=404, detail="Experiment not found")
        return {
            "id": r["id"],
            "project_id": r["project_id"],
            "model_id": r["model_id"],
            "input_dataset_version": r["input_dataset_version"],
            "output_dataset_version": r["output_dataset_version"],
            "operation": r["operation"],
            "parameters": json.loads(r["parameters"] or "{}"),
            "metrics_before": json.loads(r["metrics_before"] or "{}"),
            "metrics_after": json.loads(r["metrics_after"] or "{}"),
            "comparison": json.loads(r["comparison"] or "{}"),
            "agent_reasoning": r["agent_reasoning"],
            "created_at": r["created_at"]
        }

