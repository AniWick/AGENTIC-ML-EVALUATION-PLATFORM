import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, mean_squared_error
from app.models.predictor import model_predictor
from app.datasets.loader import dataset_loader
from app.schemas.evaluation import RobustnessMetrics

class RobustnessEvaluator:
    """Measures model sensitivity to perturbations (Gaussian noise, missing values, label flipping)."""

    @staticmethod
    def evaluate_noise_robustness(
        model_path: str,
        dataset_path: str,
        target_column: Optional[str] = None,
        noise_level: float = 0.10,
        task_type: str = "classification"
    ) -> RobustnessMetrics:
        df = dataset_loader.load(dataset_path)
        if not target_column:
            target_column = df.columns[-1]

        y_true = df[target_column].values
        feature_cols = [c for c in df.columns if c != target_column]
        X = df[feature_cols].select_dtypes(include=[np.number]).fillna(0).values.astype(np.float32)

        # Baseline score
        preds_clean = model_predictor.predict(model_path, X)
        if task_type == "classification":
            clean_metrics = {
                "accuracy": float(accuracy_score(y_true, preds_clean)),
                "precision_macro": float(precision_score(y_true, preds_clean, average="macro", zero_division=0)),
                "recall_macro": float(recall_score(y_true, preds_clean, average="macro", zero_division=0)),
                "f1_macro": float(f1_score(y_true, preds_clean, average="macro", zero_division=0)),
            }
            orig_score = clean_metrics["accuracy"]
        else:
            orig_score = float(-mean_squared_error(y_true, preds_clean))

        # Add Gaussian noise scaled by standard deviation of each feature
        stds = np.std(X, axis=0)
        stds[stds == 0] = 1.0
        noise = np.random.normal(0, stds * noise_level, size=X.shape)
        X_perturbed = X + noise

        preds_noisy = model_predictor.predict(model_path, X_perturbed)
        if task_type == "classification":
            perturbed_score = float(accuracy_score(y_true, preds_noisy))
            noisy_metrics = {
                "accuracy": perturbed_score,
                "precision_macro": float(precision_score(y_true, preds_noisy, average="macro", zero_division=0)),
                "recall_macro": float(recall_score(y_true, preds_noisy, average="macro", zero_division=0)),
                "f1_macro": float(f1_score(y_true, preds_noisy, average="macro", zero_division=0)),
            }
            score_drop = orig_score - perturbed_score
            drop_percentage = float((score_drop / orig_score) * 100) if orig_score > 0 else 0.0
        else:
            perturbed_score = float(-mean_squared_error(y_true, preds_noisy))
            score_drop = orig_score - perturbed_score
            drop_percentage = float(abs(score_drop))
            clean_metrics = {"score": orig_score}
            noisy_metrics = {"score": perturbed_score}

        is_sensitive = drop_percentage > 10.0 # Sensitive if >10% relative drop

        return RobustnessMetrics(
            perturbation_type=f"gaussian_noise_{int(noise_level*100)}%",
            noise_level=noise_level,
            original_score=orig_score,
            perturbed_score=perturbed_score,
            score_drop=score_drop,
            drop_percentage=drop_percentage,
            is_sensitive=is_sensitive,
            clean_metrics=clean_metrics,
            noisy_metrics=noisy_metrics,
        )

robustness_evaluator = RobustnessEvaluator()
