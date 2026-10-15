import pytest
from fastapi.testclient import TestClient
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import joblib

from app.main import app
from app.services.project_service import project_service
from app.agent.tools import tool_registry

client = TestClient(app)

@pytest.fixture
def setup_project_with_assets(tmp_path):
    X, y = make_classification(n_samples=100, n_features=4, random_state=42)
    df = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(4)])
    df["target"] = y

    csv_path = tmp_path / "test_api_data.csv"
    df.to_csv(csv_path, index=False)

    clf = RandomForestClassifier(n_estimators=5, random_state=42)
    clf.fit(X, y)
    model_path = tmp_path / "test_api_model.joblib"
    joblib.dump(clf, model_path)

    proj = project_service.create_project(name="Security API Test", description="Testing Security API")
    m_info = project_service.upload_model(proj.id, str(model_path), "model.joblib")
    d_info = project_service.upload_dataset(proj.id, str(csv_path), "data.csv")

    return proj.id, m_info, d_info, str(model_path), str(csv_path)


def test_security_experiments_endpoint(setup_project_with_assets):
    proj_id, m_info, d_info, model_path, csv_path = setup_project_with_assets

    # Save a mock attack experiment to DB
    exp_res = tool_registry.execute_tool(
        "save_experiment",
        {
            "project_id": proj_id,
            "model_id": m_info["model_id"],
            "operation": "boundary_attack",
            "parameters": {"target_column": "target"},
            "input_dataset_version": "v001",
            "output_dataset_version": "v001",
            "metrics_before": {"accuracy": 0.95},
            "metrics_after": {"accuracy": 0.60},
            "agent_reasoning": "Tested model against Boundary Evasion Attack."
        }
    )
    assert exp_res["status"] == "success"

    # Call endpoint GET /projects/{project_id}/security-experiments
    response = client.get(f"/projects/{proj_id}/security-experiments")
    assert response.status_code == 200

    data = response.json()
    assert data["project_id"] == proj_id
    assert len(data["experiments"]) >= 1

    latest_exp = data["experiments"][-1]
    assert latest_exp["operation"] == "boundary_attack"
    # Ensure JSON string fields were properly deserialized into dicts
    assert isinstance(latest_exp["metrics_before"], dict)
    assert latest_exp["metrics_before"]["accuracy"] == 0.95
    assert isinstance(latest_exp["metrics_after"], dict)
    assert latest_exp["metrics_after"]["accuracy"] == 0.60