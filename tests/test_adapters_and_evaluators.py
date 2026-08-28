import os
import pytest
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import joblib

from app.models.loader import model_loader
from app.models.inspector import model_inspector
from app.datasets.inspector import dataset_inspector
from app.datasets.validator import compatibility_validator
from app.evaluation.performance import performance_evaluator
from app.evaluation.data_quality import data_quality_evaluator
from app.evaluation.robustness import robustness_evaluator
from app.evaluation.comparison import evaluation_comparator
from app.transformations.cleaner import data_cleaner
from app.transformations.balancer import data_balancer
from app.transformations.scaler import data_scaler

@pytest.fixture
def sample_data_and_model(tmp_path):
    X, y = make_classification(n_samples=200, n_features=5, n_informative=3, weights=[0.85, 0.15], random_state=42)
    df = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(5)])
    df["target"] = y

    # Inject 5 missing values
    df.loc[0:4, "feat_0"] = np.nan

    csv_path = tmp_path / "test_dataset.csv"
    df.to_csv(csv_path, index=False)

    clf = RandomForestClassifier(n_estimators=10, random_state=42)
    clf.fit(X, y)
    model_path = tmp_path / "test_model.joblib"
    joblib.dump(clf, model_path)

    return str(model_path), str(csv_path), df

def test_model_loading_and_inspection(sample_data_and_model):
    model_path, csv_path, _ = sample_data_and_model
    info = model_inspector.inspect(model_path)
    assert info["framework"] == "sklearn"
    assert info["model_type"] == "classification"
    assert info["input_shape"] == [5]

def test_dataset_inspection_and_profiling(sample_data_and_model):
    model_path, csv_path, _ = sample_data_and_model
    ds_info = dataset_inspector.inspect(csv_path)
    assert ds_info["rows"] == 200
    assert ds_info["columns"] == 6
    assert ds_info["missing_values_count"] == 5
    assert ds_info["target_column"] == "target"

def test_compatibility_validator(sample_data_and_model):
    model_path, csv_path, _ = sample_data_and_model
    val = compatibility_validator.validate_compatibility(model_path, csv_path)
    assert val.is_compatible is True
    assert val.checks["input_shape"] is True

def test_performance_evaluation(sample_data_and_model):
    model_path, csv_path, _ = sample_data_and_model
    metrics = performance_evaluator.evaluate(model_path, csv_path, target_column="target")
    assert metrics.accuracy is not None
    assert metrics.f1_macro is not None
    assert metrics.accuracy >= 0.0 and metrics.accuracy <= 1.0

def test_data_quality_and_robustness(sample_data_and_model):
    model_path, csv_path, _ = sample_data_and_model
    dq = data_quality_evaluator.evaluate(csv_path, target_column="target")
    assert dq.missing_values_count == 5
    assert dq.class_imbalance_detected is True

    rob = robustness_evaluator.evaluate_noise_robustness(model_path, csv_path, target_column="target", noise_level=0.10)
    assert rob.perturbed_score is not None

def test_transformations(sample_data_and_model):
    model_path, csv_path, df = sample_data_and_model
    # Cleaner
    df_clean, meta_clean = data_cleaner.handle_missing_values(df, strategy="median", target_column="target")
    assert df_clean["feat_0"].isna().sum() == 0

    # Balancer
    df_bal, meta_bal = data_balancer.balance_data(df_clean, method="smote", target_column="target")
    counts = df_bal["target"].value_counts()
    assert counts.iloc[0] == counts.iloc[1] # SMOTE balanced classes

    # Comparison
    base_metrics = {"accuracy": 0.90, "f1_macro": 0.70, "f1_minority": 0.40}
    exp_metrics = {"accuracy": 0.88, "f1_macro": 0.82, "f1_minority": 0.75}
    comp = evaluation_comparator.compare("v001", "v002", base_metrics, exp_metrics)
    assert comp.is_improved is True
    assert comp.metric_deltas["f1_macro"] > 0

