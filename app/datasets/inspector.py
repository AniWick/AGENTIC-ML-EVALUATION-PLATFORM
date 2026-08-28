import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from app.datasets.loader import dataset_loader

class DatasetInspector:
    """Inspects dataset structure, feature types, missingness, duplicates, and class distributions."""

    @staticmethod
    def inspect(file_path: str, dataset_id: str = "dataset_001", version: str = "v001", target_col: Optional[str] = None) -> Dict[str, Any]:
        df = dataset_loader.load(file_path)
        return DatasetInspector.inspect_dataframe(df, dataset_id, version, target_col)

    @staticmethod
    def inspect_dataframe(df: pd.DataFrame, dataset_id: str = "dataset_001", version: str = "v001", target_col: Optional[str] = None) -> Dict[str, Any]:
        rows, cols = df.shape
        col_names = list(df.columns)
        
        # Auto-detect target column if not provided
        if not target_col:
            candidate_targets = ["target", "label", "class", "y", "outcome", "status"]
            for candidate in candidate_targets:
                for c in col_names:
                    if c.lower() == candidate:
                        target_col = c
                        break
                if target_col:
                    break
            if not target_col and len(col_names) > 0:
                target_col = col_names[-1] # Fallback to last column

        numerical_features = []
        categorical_features = []
        data_types = {}

        for col in col_names:
            dtype_str = str(df[col].dtype)
            data_types[col] = dtype_str
            if col == target_col:
                continue
            if pd.api.types.is_numeric_dtype(df[col]):
                numerical_features.append(col)
            else:
                categorical_features.append(col)

        missing_by_col = {col: int(df[col].isna().sum()) for col in col_names}
        total_missing = int(sum(missing_by_col.values()))
        duplicate_rows = int(df.duplicated().sum())

        # Class distribution if target exists
        class_distribution = None
        if target_col and target_col in df.columns:
            counts = df[target_col].value_counts(dropna=False).to_dict()
            class_distribution = {str(k): int(v) for k, v in counts.items()}

        # Simple IQR outlier count on numerical features
        outliers_count = 0
        if numerical_features:
            for col in numerical_features:
                s = df[col].dropna()
                if len(s) > 4:
                    q25 = s.quantile(0.25)
                    q75 = s.quantile(0.75)
                    iqr = q75 - q25
                    if iqr > 0:
                        lower_bound = q25 - 1.5 * iqr
                        upper_bound = q75 + 1.5 * iqr
                        outliers_count += int(((s < lower_bound) | (s > upper_bound)).sum())

        return {
            "dataset_id": dataset_id,
            "version": version,
            "rows": rows,
            "columns": cols,
            "column_names": col_names,
            "numerical_features": numerical_features,
            "categorical_features": categorical_features,
            "target_column": target_col,
            "missing_values_count": total_missing,
            "missing_values_by_column": missing_by_col,
            "duplicate_rows_count": duplicate_rows,
            "class_distribution": class_distribution,
            "data_types": data_types,
            "outliers_count": outliers_count
        }

dataset_inspector = DatasetInspector()

