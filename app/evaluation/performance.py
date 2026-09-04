import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, mean_absolute_error, mean_squared_error,
    r2_score, roc_auc_score
)

from app.models.predictor import model_predictor
from app.datasets.loader import dataset_loader
from app.schemas.evaluation import PerformanceMetrics


class PerformanceEvaluator:
    """Calculates objective, deterministic ML performance metrics for classification and regression."""

    @staticmethod
    def evaluate(
        model_path: str,
        dataset_path: str,
        target_column: Optional[str] = None,
        task_type: str = "classification"
    ) -> PerformanceMetrics:

        df = dataset_loader.load(dataset_path)

        if not target_column:
            target_column = df.columns[-1]

        y_true = df[target_column].values

        feature_cols = [c for c in df.columns if c != target_column]

        # Keep X as a DataFrame so sklearn models that were trained
        # with feature names receive the same feature names.
        X_df = df[feature_cols].copy()

        # Ensure numeric type while preserving DataFrame column names.
        try:
            X_numeric = X_df.astype(np.float32)
        except Exception:
            # Auto-encode non-numeric columns while preserving
            # the original feature names.
            X_numeric = X_df.copy()

            for col in X_numeric.select_dtypes(
                include=["object", "category"]
            ).columns:
                X_numeric[col] = pd.factorize(X_numeric[col])[0]

            X_numeric = X_numeric.fillna(0).astype(np.float32)

        # Handle NaNs while preserving DataFrame structure and column names.
        if X_numeric.isna().any().any():
            col_means = X_numeric.mean()

            for col in X_numeric.columns:
                if pd.isna(col_means[col]):
                    col_means[col] = 0

            X_numeric = X_numeric.fillna(col_means)

        # Keep X_numeric as a DataFrame.
        # This prevents sklearn warnings for models trained with
        # pandas feature names.
        y_pred = model_predictor.predict(model_path, X_numeric)

        if task_type == "classification":
            # Map string labels if necessary
            unique_classes = np.unique(y_true)

            acc = float(
                accuracy_score(y_true, y_pred)
            )

            prec = float(
                precision_score(
                    y_true,
                    y_pred,
                    average="macro",
                    zero_division=0
                )
            )

            rec = float(
                recall_score(
                    y_true,
                    y_pred,
                    average="macro",
                    zero_division=0
                )
            )

            f1_macro = float(
                f1_score(
                    y_true,
                    y_pred,
                    average="macro",
                    zero_division=0
                )
            )

            # Per-class F1
            per_class_f1_arr = f1_score(
                y_true,
                y_pred,
                average=None,
                zero_division=0
            )

            per_class_f1 = {
                str(c): float(f)
                for c, f in zip(unique_classes, per_class_f1_arr)
            }

            # Minority class F1
            counts = pd.Series(y_true).value_counts()

            minority_class = counts.index[-1]

            minority_idx = (
                list(unique_classes).index(minority_class)
                if minority_class in unique_classes
                else 0
            )

            f1_minority = (
                float(per_class_f1_arr[minority_idx])
                if len(per_class_f1_arr) > minority_idx
                else f1_macro
            )

            cm = confusion_matrix(
                y_true,
                y_pred
            ).tolist()

            # ROC-AUC if probabilities are available
            roc_auc = None

            proba = model_predictor.predict_proba(
                model_path,
                X_numeric
            )

            if proba is not None and len(unique_classes) == 2:
                try:
                    roc_auc = float(
                        roc_auc_score(
                            y_true,
                            proba[:, 1]
                        )
                    )
                except Exception:
                    pass

            return PerformanceMetrics(
                task_type="classification",
                accuracy=acc,
                precision_macro=prec,
                recall_macro=rec,
                f1_macro=f1_macro,
                f1_minority=f1_minority,
                per_class_f1=per_class_f1,
                confusion_matrix=cm,
                roc_auc=roc_auc
            )

        else:
            # Regression
            mae = float(
                mean_absolute_error(
                    y_true,
                    y_pred
                )
            )

            mse = float(
                mean_squared_error(
                    y_true,
                    y_pred
                )
            )

            rmse = float(
                np.sqrt(mse)
            )

            r2 = float(
                r2_score(
                    y_true,
                    y_pred
                )
            )

            return PerformanceMetrics(
                task_type="regression",
                mae=mae,
                mse=mse,
                rmse=rmse,
                r2_score=r2
            )


performance_evaluator = PerformanceEvaluator()