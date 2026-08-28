import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
from app.datasets.loader import dataset_loader
from app.schemas.evaluation import DataQualityMetrics

class DataQualityEvaluator:
    """Detects data quality problems: missing values, duplicates, class imbalance, outliers, constant columns."""

    @staticmethod
    def evaluate(dataset_path: str, target_column: Optional[str] = None) -> DataQualityMetrics:
        df = dataset_loader.load(dataset_path)
        rows, cols = df.shape

        if not target_column:
            target_column = df.columns[-1]

        missing_count = int(df.isna().sum().sum())
        missing_ratio = float(missing_count / (rows * cols)) if (rows * cols) > 0 else 0.0
        duplicate_count = int(df.duplicated().sum())

        # Class imbalance detection
        class_imbalance_detected = False
        imbalance_ratio = None
        if target_column in df.columns:
            val_counts = df[target_column].value_counts()
            if len(val_counts) > 1:
                majority = val_counts.iloc[0]
                minority = val_counts.iloc[-1]
                imbalance_ratio = float(majority / minority) if minority > 0 else float("inf")
                if imbalance_ratio >= 2.0:
                    class_imbalance_detected = True

        # Constant or near constant columns
        constant_cols = []
        for col in df.columns:
            if df[col].nunique(dropna=False) <= 1:
                constant_cols.append(col)

        # High cardinality categorical columns
        high_cardinality = []
        for col in df.select_dtypes(include=["object", "category"]).columns:
            if col != target_column and df[col].nunique() > 50:
                high_cardinality.append(col)

        # Outlier detection on numerical features
        outliers_count = 0
        num_cols = df.select_dtypes(include=[np.number]).columns
        num_cols = [c for c in num_cols if c != target_column]
        for col in num_cols:
            s = df[col].dropna()
            if len(s) > 4:
                q25 = s.quantile(0.25)
                q75 = s.quantile(0.75)
                iqr = q75 - q25
                if iqr > 0:
                    lower = q25 - 1.5 * iqr
                    upper = q75 + 1.5 * iqr
                    outliers_count += int(((s < lower) | (s > upper)).sum())

        total_num_cells = len(num_cols) * rows
        outliers_ratio = float(outliers_count / total_num_cells) if total_num_cells > 0 else 0.0

        # Status summary ratings
        status_summary = {}
        status_summary["missing_values"] = "problem" if missing_ratio > 0.05 else ("warning" if missing_count > 0 else "good")
        status_summary["duplicates"] = "warning" if duplicate_count > 0 else "good"
        status_summary["class_imbalance"] = "problem" if class_imbalance_detected else "good"
        status_summary["outliers"] = "warning" if outliers_ratio > 0.03 else "good"

        return DataQualityMetrics(
            missing_values_count=missing_count,
            missing_values_ratio=missing_ratio,
            duplicate_rows_count=duplicate_count,
            class_imbalance_detected=class_imbalance_detected,
            imbalance_ratio=imbalance_ratio,
            outliers_count=outliers_count,
            outliers_ratio=outliers_ratio,
            constant_columns=constant_cols,
            high_cardinality_columns=high_cardinality,
            status_summary=status_summary
        )

data_quality_evaluator = DataQualityEvaluator()

