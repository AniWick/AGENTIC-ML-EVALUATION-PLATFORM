import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, Optional

class DataBalancer:
    """Balances class distributions using SMOTE, RandomOverSampler, or RandomUnderSampler."""

    @staticmethod
    def balance_data(df: pd.DataFrame, method: str = "smote", target_column: Optional[str] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        df_bal = df.copy()
        target = target_column or df_bal.columns[-1]

        if target not in df_bal.columns:
            return df_bal, {"error": "Target column not found"}

        X_cols = [c for c in df_bal.columns if c != target]
        X = df_bal[X_cols]
        y = df_bal[target]

        initial_dist = {str(k): int(v) for k, v in y.value_counts().items()}

        # Factorize / ensure numeric for SMOTE
        X_num = X.copy()
        for col in X_num.select_dtypes(include=["object", "category"]).columns:
            X_num[col] = pd.factorize(X_num[col])[0]
        X_num = X_num.fillna(0)

        # Check minority count for k_neighbors
        min_class_count = y.value_counts().min()
        k_neighbors = min(5, max(1, min_class_count - 1))

        if method == "smote" and min_class_count > 1:
            try:
                from imblearn.over_sampling import SMOTE
                sampler = SMOTE(k_neighbors=k_neighbors, random_state=42)
                X_res, y_res = sampler.fit_resample(X_num, y)
            except Exception:
                from imblearn.over_sampling import RandomOverSampler
                sampler = RandomOverSampler(random_state=42)
                X_res, y_res = sampler.fit_resample(X_num, y)
        elif method == "undersample":
            from imblearn.under_sampling import RandomUnderSampler
            sampler = RandomUnderSampler(random_state=42)
            X_res, y_res = sampler.fit_resample(X_num, y)
        else: # Default RandomOverSampler
            from imblearn.over_sampling import RandomOverSampler
            sampler = RandomOverSampler(random_state=42)
            X_res, y_res = sampler.fit_resample(X_num, y)

        df_balanced = pd.DataFrame(X_res, columns=X_cols)
        df_balanced[target] = y_res.values
        new_dist = {str(k): int(v) for k, v in df_balanced[target].value_counts().items()}

        return df_balanced, {
            "method": method,
            "initial_distribution": initial_dist,
            "balanced_distribution": new_dist,
            "rows_before": len(df),
            "rows_after": len(df_balanced)
        }

data_balancer = DataBalancer()

