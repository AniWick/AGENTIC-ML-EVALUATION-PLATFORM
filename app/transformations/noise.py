import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, Optional

class DataCorruption:
    """Injects perturbations for robustness testing: Gaussian noise, missing values, label noise."""

    @staticmethod
    def add_gaussian_noise(df: pd.DataFrame, noise_level: float = 0.10, target_column: Optional[str] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        df_corrupt = df.copy()
        target = target_column or df_corrupt.columns[-1]
        num_cols = [c for c in df_corrupt.select_dtypes(include=[np.number]).columns if c != target]

        for col in num_cols:
            std = df_corrupt[col].std()
            if np.isnan(std) or std == 0:
                std = 1.0
            noise = np.random.normal(0, std * noise_level, size=len(df_corrupt))
            df_corrupt[col] = df_corrupt[col] + noise

        return df_corrupt, {"operation": "add_gaussian_noise", "noise_level": noise_level, "affected_columns": num_cols}

    @staticmethod
    def inject_missing_values(df: pd.DataFrame, missing_ratio: float = 0.10, target_column: Optional[str] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        df_corrupt = df.copy()
        target = target_column or df_corrupt.columns[-1]
        cols_to_corrupt = [c for c in df_corrupt.columns if c != target]

        total_cells = len(df_corrupt) * len(cols_to_corrupt)
        num_missing = int(total_cells * missing_ratio)

        for _ in range(num_missing):
            r = np.random.randint(0, len(df_corrupt))
            c = np.random.choice(cols_to_corrupt)
            df_corrupt.at[r, c] = np.nan

        return df_corrupt, {"operation": "inject_missing_values", "missing_ratio": missing_ratio, "injected_count": num_missing}

    @staticmethod
    def inject_label_noise(df: pd.DataFrame, noise_ratio: float = 0.10, target_column: Optional[str] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        df_corrupt = df.copy()
        target = target_column or df_corrupt.columns[-1]

        unique_labels = df_corrupt[target].unique()
        if len(unique_labels) < 2:
            return df_corrupt, {"warning": "Cannot inject label noise for single-class target"}

        num_corrupt = int(len(df_corrupt) * noise_ratio)
        indices = np.random.choice(df_corrupt.index, size=num_corrupt, replace=False)

        for idx in indices:
            curr_label = df_corrupt.at[idx, target]
            other_labels = [l for l in unique_labels if l != curr_label]
            df_corrupt.at[idx, target] = np.random.choice(other_labels)

        return df_corrupt, {"operation": "inject_label_noise", "noise_ratio": noise_ratio, "flipped_labels_count": num_corrupt}

data_corruption = DataCorruption()

