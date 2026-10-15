import json
import logging
import sqlite3
from datetime import datetime as RealDatetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
import joblib

from app.agent.agent import AgentRunner
from app.agent.planner import AgentPlanner, Decision, agent_planner
from app.agent.state import AgentState
from app.agent.security_aliases import IncompatibleSecurityRequest
from app.agent.tools import ToolRegistry, tool_registry
from app.api import chat as chat_api
from app.core.database import db
from app.evaluation import robustness as robustness_module
from app.main import app
from app.reports import generator as report_module
from app.services.project_service import project_service


client = TestClient(app)


def test_router_deterministic_rules_override_valid_llm_routes():
    compound = (
        "Run FGSM with feature squeezing defense and generate a report"
    )
    assert chat_api._parse_router_response(
        {"intent": "deterministic", "workflow": "report"}, compound
    ) == {
        "intent": "deterministic",
        "workflow": "attack_defended_model",
    }

    hardening = "Adversarial retrain and harden the model, then create a report"
    assert chat_api._parse_router_response(
        {"intent": "deterministic", "workflow": "attack"}, hardening
    ) == {
        "intent": "deterministic",
        "workflow": "harden_model",
    }


def test_defended_attack_deterministic_order_persists_before_report():
    state = AgentState(
        project_id="project_security",
        user_goal=(
            "Run FGSM with feature squeezing defense and generate a report"
        ),
        requested_workflow="attack_defended_model",
        model_id="model_original",
        model_path="model.joblib",
        dataset_id="dataset_original",
        dataset_path="data.csv",
        current_dataset_version="v001",
    )

    decision = AgentPlanner._deterministic_fallback_plan(state)
    assert decision.action == "inspect_model"

    state.model_inspected = True
    assert AgentPlanner._deterministic_fallback_plan(state).action == "inspect_dataset"

    state.dataset_inspected = True
    assert AgentPlanner._deterministic_fallback_plan(state).action == "validate_model_dataset"

    # A check that ran but found the pair incompatible must not unlock the attack.
    state.compatibility_validated = True
    state.compatibility_result = False
    state.model_metadata = {"framework": "pytorch"}
    with pytest.raises(IncompatibleSecurityRequest):
        AgentPlanner._deterministic_fallback_plan(state)

    state.compatibility_result = True
    decision = AgentPlanner._deterministic_fallback_plan(state)
    assert decision.action == "attack_defended_model"
    assert decision.parameters["attack_name"] == "fgsm"
    assert decision.parameters["defense_name"] == "feature_squeezing"

    state.defended_attack_evaluated = True
    state.defended_attack_results = {
        "status": "success",
        "attack_type": "fgsm",
        "defense_type": "feature_squeezing",
        "defended_accuracy": 0.8,
        "adversarial_accuracy": 0.7,
    }
    assert AgentPlanner._deterministic_fallback_plan(state).action == "save_experiment"

    state.experiments.append({"experiment_id": "exp_1"})
    assert AgentPlanner._deterministic_fallback_plan(state).action == "generate_report"

    state.actions_taken.append({"action": "generate_report"})
    decision = AgentPlanner._deterministic_fallback_plan(state)
    assert decision.action == ""
    assert decision.status == "complete"


def test_hardening_hands_new_asset_to_active_agent_state(monkeypatch):
    state = AgentState(
        project_id="project_hardening",
        user_goal="Harden this model with PGD adversarial training",
        requested_workflow="harden_model",
        model_id="model_original",
        model_path="original.joblib",
        dataset_id="dataset_original",
        dataset_path="data.csv",
        current_dataset_version="v001",
    )

    def inspect_model(**kwargs):
        return {"framework": "pytorch", "model_type": "classification"}

    def inspect_dataset(**kwargs):
        return {"rows": 20, "columns": 3, "target_column": "target"}

    def validate_compatibility(**kwargs):
        return {"is_compatible": True}

    def harden_model(**kwargs):
        return {
            "status": "success",
            "model_id": "model_hardened_new",
            "file_path": "hardened.pt",
            "training_summary": {"method": "pgd", "samples_generated": 12},
        }

    def save_experiment(**kwargs):
        return {"experiment_id": "exp_hardened", **kwargs}

    replacements = {
        "inspect_model": inspect_model,
        "inspect_dataset": inspect_dataset,
        "validate_model_dataset": validate_compatibility,
        "harden_model": harden_model,
        "save_experiment": save_experiment,
    }
    for name, handler in replacements.items():
        monkeypatch.setitem(tool_registry.tools[name], "handler", handler)

    final_state = AgentRunner(state).run_until_complete()
    actions = [item["action"] for item in final_state.actions_taken]

    assert actions == [
        "inspect_model",
        "inspect_dataset",
        "validate_model_dataset",
        "harden_model",
        "save_experiment",
    ]
    assert final_state.model_hardened is True
    assert final_state.model_id == "model_hardened_new"
    assert final_state.model_path == "hardened.pt"
    assert final_state.hardened_model_info["model_id"] == "model_hardened_new"
    assert final_state.status == "completed"


def test_tool_registry_propagates_nested_handler_errors():
    registry = ToolRegistry()
    registry.register(
        name="nested_failure_test",
        description="test",
        parameters_schema={"type": "object"},
        handler=lambda: {"status": "error", "error": "nested failure"},
    )

    try:
        result = registry.execute_tool("nested_failure_test", {})
    finally:
        registry.tools.pop("nested_failure_test", None)

    assert result["status"] == "error"
    assert "nested failure" in result["message"]


def test_runner_strips_bulky_tensors_from_observations(monkeypatch):
    tool_registry.register(
        name="bulky_result_test",
        description="test",
        parameters_schema={"type": "object"},
        handler=lambda: {
            "status": "success",
            "x_adv": np.zeros((100, 100), dtype=np.float32).tolist(),
            "adversarial_sample_count": 100,
        },
    )
    monkeypatch.setattr(
        agent_planner,
        "decide_next_step",
        lambda state: Decision("bulky_result_test", {}, "test"),
    )

    state = AgentState(project_id="p", user_goal="test")
    try:
        AgentRunner(state).run_step()
    finally:
        tool_registry.tools.pop("bulky_result_test", None)

    result = state.observations[-1]["result"]["result"]
    assert "x_adv" not in result
    assert result["adversarial_sample_count"] == 100


def test_defended_attack_result_does_not_serialize_adversarial_tensor(monkeypatch):
    class FakeAttack:
        def generate(self, x):
            return np.asarray(x, dtype=np.float32) + 1.0

    class FakeClassifier:
        def predict(self, x):
            values = np.asarray(x)[:, 0]
            return np.column_stack([values, 1.0 - values])

    class FakeModel:
        classes_ = np.array([0, 1])

    # feature_squeezing needs enough features that the squeezed representation
    # stays smaller than the original, so use a realistic width.
    X = np.asarray([
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        [1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
    ])
    y = np.asarray([0, 1, 0, 1])
    dataset_info = {
        "total_rows": 4,
        "total_features": 6,
        "target_column": "target",
        "num_classes": 2,
        "feature_names": [f"f{i}" for i in range(6)],
    }

    monkeypatch.setattr(
        robustness_module.RobustnessEvaluator,
        "_resolve_inputs",
        staticmethod(
            lambda model_path, dataset_path, target_column=None: (
                FakeModel(),
                object(),
                "sklearn",
                X,
                y,
                (-2.0, 3.0),
                dataset_info,
            )
        ),
    )
    monkeypatch.setattr(
        robustness_module,
        "_build_art_classifier",
        lambda *args, **kwargs: FakeClassifier(),
    )
    monkeypatch.setattr(
        robustness_module,
        "_build_attack",
        lambda attack_name, classifier, params: (FakeAttack(), {"eps": 0.1}),
    )

    result = robustness_module.robustness_evaluator.run_attack_on_defended_model(
        attack_name="boundary_attack",
        defense_name="feature_squeezing",
        model_path="model.joblib",
        dataset_path="data.csv",
        target_column="target",
    )

    assert result["status"] if "status" in result else "success" == "success"
    assert "adversarial_samples" not in result
    assert result["adversarial_sample_count"] == len(X)


def _insert_report_assets(project_id: str):
    now = RealDatetime.utcnow().isoformat()
    model_id = f"model_{project_id}"
    dataset_id = f"dataset_{project_id}"
    with db._get_connection() as conn:
        conn.execute(
            """
            INSERT INTO models
            (id, project_id, name, framework, file_path, architecture,
             input_shape, output_shape, parameter_count, status, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                model_id,
                project_id,
                "AuditClassifier",
                "sklearn",
                "model.joblib",
                "RandomForestClassifier",
                "[5]",
                "[2]",
                1234,
                "ready",
                now,
            ),
        )
        conn.execute(
            """
            INSERT INTO datasets
            (id, project_id, name, file_path, current_version, format, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (dataset_id, project_id, "AuditData", "data.csv", "v001", ".csv", now),
        )
        conn.execute(
            """
            INSERT INTO dataset_versions
            (id, dataset_id, version, file_path, operation, parameters,
             row_count, column_count, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                f"{dataset_id}_v001",
                dataset_id,
                "v001",
                "data.csv",
                "initial_upload",
                "{}",
                100,
                6,
                now,
            ),
        )
    return model_id, dataset_id


def test_report_contains_authoritative_metadata_and_bounded_percentages(
    monkeypatch, tmp_path
):
    project = project_service.create_project("Report regression")
    model_id, dataset_id = _insert_report_assets(project.id)
    monkeypatch.setattr(report_module.storage_manager, "reports_dir", tmp_path)
    monkeypatch.setattr(
        report_module.qwen_client,
        "generate",
        lambda *args, **kwargs: "",
    )

    state = AgentState(
        project_id=project.id,
        user_goal="Defended attack report",
        requested_workflow="attack_defended_model",
        model_id=model_id,
        model_path="model.joblib",
        dataset_id=dataset_id,
        dataset_path="data.csv",
        current_dataset_version="v001",
        target_column="target",
        model_inspected=True,
        dataset_inspected=True,
        compatibility_validated=True,
        defended_attack_evaluated=True,
    )
    state.dataset_metadata = {
        "rows": 100,
        "columns": 6,
        "column_names": ["age", "income", "score", "region", "flag", "target"],
        "numerical_features": ["age", "income", "score"],
        "categorical_features": ["region", "flag"],
        "target_column": "target",
    }
    state.baseline_metrics = {
        "accuracy": 0.875,
        "precision_macro": 1.5,
        "recall_macro": -0.2,
        "f1_macro": 0.8,
    }
    state.data_quality = {
        "missing_values_count": 3,
        "missing_values_ratio": 0.005,
        "duplicate_rows_count": 2,
        "class_imbalance_detected": True,
        "imbalance_ratio": 3.0,
        "outliers_count": 4,
        "outliers_ratio": 0.01,
        "constant_columns": [],
        "high_cardinality_columns": [],
    }
    state.defended_attack_results = {
        "status": "success",
        "attack_type": "fgsm",
        "defense_type": "feature_squeezing",
        "clean_accuracy": 0.9,
        "defended_accuracy": 0.85,
        "adversarial_accuracy": -0.1,
        "clean_precision": 0.9,
        "defended_precision": 0.84,
        "adversarial_precision": 0.7,
        "clean_recall": 0.9,
        "defended_recall": 0.83,
        "adversarial_recall": 0.69,
        "clean_f1": 0.9,
        "defended_f1": 0.835,
        "adversarial_f1": 0.695,
        "attack_parameters": {"eps": 0.2, "max_iter": 5},
        "defense_parameters": {"bit_depth": 4},
        "dataset_info": {
            "total_rows": 100,
            "total_features": 5,
            "feature_names": ["age", "income", "score", "region", "flag"],
        },
    }

    result = report_module.report_generator.generate_and_save(state)
    markdown = Path(result["file_path"]).read_text(encoding="utf-8")

    assert "AuditClassifier" in markdown
    assert "RandomForestClassifier" in markdown
    assert "100 rows" in markdown
    assert "6 total columns" in markdown
    assert "5 features + 1 target" in markdown
    assert "`age`, `income`, `score`" in markdown
    assert "87.50%" in markdown
    assert "100.00%" in markdown
    assert "0.00%" in markdown
    assert "Adversarial under Defense" in markdown
    assert "eps" in markdown and "bit_depth" in markdown
    assert "150.00%" not in markdown


def test_report_ids_are_unique_even_within_same_second(monkeypatch, tmp_path):
    project = project_service.create_project("Collision report")
    model_id, dataset_id = _insert_report_assets(project.id)
    monkeypatch.setattr(report_module.storage_manager, "reports_dir", tmp_path)

    class FixedDateTime:
        @classmethod
        def utcnow(cls):
            return RealDatetime(2026, 1, 1, 0, 0, 0)

    monkeypatch.setattr(report_module, "datetime", FixedDateTime)
    state = AgentState(
        project_id=project.id,
        user_goal="report",
        requested_workflow="report",
        model_id=model_id,
        model_path="model.joblib",
        dataset_id=dataset_id,
        dataset_path="data.csv",
        current_dataset_version="v001",
    )

    first = report_module.report_generator.generate_and_save(state)
    second = report_module.report_generator.generate_and_save(state)

    assert first["report_id"] != second["report_id"]
    assert (tmp_path / f"{first['report_id']}.md").exists()
    assert (tmp_path / f"{second['report_id']}.md").exists()


def test_report_database_failure_does_not_leave_markdown_file(
    monkeypatch, tmp_path
):
    project = project_service.create_project("Atomic report")
    model_id, dataset_id = _insert_report_assets(project.id)
    monkeypatch.setattr(report_module.storage_manager, "reports_dir", tmp_path)

    class FailingConnection:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def execute(self, *args, **kwargs):
            raise sqlite3.OperationalError("database unavailable")

        def commit(self):
            raise AssertionError("commit must not run after execute fails")

    monkeypatch.setattr(db, "_get_connection", lambda: FailingConnection())
    state = AgentState(
        project_id=project.id,
        user_goal="report",
        requested_workflow="report",
        model_id=model_id,
        model_path="model.joblib",
        dataset_id=dataset_id,
        dataset_path="data.csv",
        current_dataset_version="v001",
    )

    with pytest.raises(sqlite3.OperationalError):
        report_module.report_generator.generate_and_save(state)

    assert list(tmp_path.iterdir()) == []


@pytest.fixture
def chat_project(tmp_path):
    X, y = make_classification(
        n_samples=30, n_features=3, n_informative=2, n_redundant=0, random_state=7
    )
    frame = pd.DataFrame(X, columns=["f0", "f1", "f2"])
    frame["target"] = y
    dataset_path = tmp_path / "chat_data.csv"
    model_path = tmp_path / "chat_model.joblib"
    frame.to_csv(dataset_path, index=False)
    model = RandomForestClassifier(n_estimators=3, random_state=7)
    model.fit(X, y)
    joblib.dump(model, model_path)

    project = project_service.create_project("Chat error regression")
    project_service.upload_model(project.id, str(model_path), "chat_model.joblib")
    project_service.upload_dataset(project.id, str(dataset_path), "chat_data.csv")
    return project.id


def test_chat_workflow_error_is_clean_json_and_history_is_paired(
    monkeypatch, caplog, chat_project
):
    monkeypatch.setattr(
        chat_api.qwen_client,
        "generate_json",
        lambda *args, **kwargs: {
            "intent": "deterministic",
            "workflow": "full_evaluation",
        },
    )

    def fail_workflow(self):
        raise RuntimeError("secret-internal-details/model/C:/private/path")

    monkeypatch.setattr(chat_api.AgentRunner, "run_until_complete", fail_workflow)

    with caplog.at_level(logging.ERROR, logger="app.api.chat"):
        response = client.post(
            f"/projects/{chat_project}/chat",
            json={"message": "Run a full evaluation"},
        )

    assert response.status_code == 500
    body = response.json()
    assert body["detail"] == "The workflow could not be completed. Please try again later."
    assert "secret-internal-details" not in response.text
    assert "Traceback" not in response.text
    assert "secret-internal-details" in caplog.text
    assert "Traceback" in caplog.text

    history = client.get(f"/projects/{chat_project}/chat/history").json()["messages"]
    assert [entry["sender"] for entry in history[-2:]] == ["user", "assistant"]
    assert history[-2]["message"] == "Run a full evaluation"
    assert history[-1]["message"] == body["detail"]
