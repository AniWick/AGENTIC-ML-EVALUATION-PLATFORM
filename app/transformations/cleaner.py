import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, Optional

class DataCleaner:
    """Handles missing values, deduplication, and basic data cleaning."""

    @staticmethod
    def handle_missing_values(df: pd.DataFrame, strategy: str = "median", target_column: Optional[str] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        df_clean = df.copy()
        target = target_column or df_clean.columns[-1]
        
        # If target has missing values, drop those rows
        if target in df_clean.columns and df_clean[target].isna().sum() > 0:
            df_clean = df_clean.dropna(subset=[target])

        num_cols = df_clean.select_dtypes(include=[np.number]).columns
        cat_cols = df_clean.select_dtypes(include=["object", "category"]).columns

        imputed_count = 0
        for col in num_cols:
            if col == target:
                continue
            if df_clean[col].isna().sum() > 0:
                imputed_count += int(df_clean[col].isna().sum())
                if strategy == "mean":
                    df_clean[col] = df_clean[col].fillna(df_clean[col].mean())
                elif strategy == "mode":
                    df_clean[col] = df_clean[col].fillna(df_clean[col].mode().iloc[0] if not df_clean[col].mode().empty else 0)
                else: # Default median
                    df_clean[col] = df_clean[col].fillna(df_clean[col].median())

        for col in cat_cols:
            if col == target:
                continue
            if df_clean[col].isna().sum() > 0:
                imputed_count += int(df_clean[col].isna().sum())
                df_clean[col] = df_clean[col].fillna(df_clean[col].mode().iloc[0] if not df_clean[col].mode().empty else "missing")

        return df_clean, {"strategy": strategy, "imputed_values_count": imputed_count}

    @staticmethod
    def remove_duplicates(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        initial_rows = len(df)
        df_clean = df.drop_duplicates()
        dropped = initial_rows - len(df_clean)
        return df_clean, {"removed_duplicates_count": dropped}

data_cleaner = DataCleaner()

