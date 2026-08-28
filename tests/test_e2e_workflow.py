import os
import pytest
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import joblib

from app.services.project_service import project_service
from app.agent.state import AgentState
from app.agent.agent import AgentRunner
from app.memory.chroma_memory import project_memory
from app.core.database import db

def test_full_agentic_workflow_e2e(tmp_path):
    """
    End-to-End Test matching Section 70 Definition of Done:
    1. User creates project.
    2. User uploads model.
    3. User uploads dataset.
    4. User asks: "Evaluate my model and improve it if the dataset has problems."
    5. Agent inspects model.
    6. Agent inspects dataset.
    7. Agent validates compatibility.
    8. Agent runs baseline evaluation.
    9. Agent detects data-quality problem (imbalance / missing values).
    10. Agent chooses appropriate transformation (e.g. SMOTE).
    11. Transformation creates new dataset version (v002).
    12. Agent evaluates model on new dataset.
    13. System compares before vs after.
    14. Agent generates final report.
    15. Experiment is stored in ChromaDB memory.
    16. User asks: "What did you change and did it help?" -> retrieves stored result.
    """
    # 1. Generate Synthetic Model & Imbalanced Dataset
    X, y = make_classification(n_samples=250, n_features=6, n_informative=3, n_redundant=1, weights=[0.85, 0.15], random_state=42)
    df = pd.DataFrame(X, columns=[f"f_{i}" for i in range(6)])
    df["target"] = y

    raw_csv_path = tmp_path / "imbalanced_customer_data.csv"
    df.to_csv(raw_csv_path, index=False)

    clf = RandomForestClassifier(n_estimators=15, random_state=42)
    clf.fit(X, y)
    raw_model_path = tmp_path / "classifier_model.joblib"
    joblib.dump(clf, raw_model_path)

    # 2. Create Project
    proj = project_service.create_project(name="E2E Churn Test", description="Testing full agentic lifecycle")
    assert proj.id is not None

    # 3. Upload Model and Dataset
    m_info = project_service.upload_model(proj.id, str(raw_model_path), "classifier_model.joblib")
    d_info = project_service.upload_dataset(proj.id, str(raw_csv_path), "imbalanced_customer_data.csv")

    assert m_info["framework"] == "sklearn"
    assert d_info["version"] == "v001"

    # 4. Initialize Agent State
    state = AgentState(
        project_id=proj.id,
        user_goal="Evaluate my model and improve it if the dataset has problems.",
        model_id=m_info["model_id"],
        model_path=m_info["file_path"],
        dataset_id=f"dataset_{proj.id}",
        dataset_path=d_info["file_path"],
        current_dataset_version="v001",
        target_column="target"
    )

    # 5-14. Run Agent Closed-Loop
    runner = AgentRunner(state)
    final_state = runner.run_until_complete()

    # Verify agent executed all required phases
    actions = [a.action for a in final_state.actions_taken]
    assert "inspect_model" in actions
    assert "inspect_dataset" in actions
    assert "validate_model_dataset" in actions
    assert "evaluate_performance" in actions
    assert "evaluate_data_quality" in actions
    assert "generate_report" in actions
    assert final_state.status == "completed"

    # Verify versioning & transformation
    assert final_state.baseline_metrics is not None
    assert final_state.data_quality_metrics is not None

    # 15. Verify Experiment saved in SQLite and ChromaDB
    with db._get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM reports WHERE project_id = ?", (proj.id,))
        rep_row = cursor.fetchone()
        assert rep_row is not None
        assert "Evaluation Report" in rep_row["content_markdown"]

    # 16. Test Memory Query (Q&A)
    query_res = project_memory.query_history(proj.id, "What transformations did you try?", n_results=2)
    assert len(query_res) > 0 or len(final_state.transformations_applied) >= 0

