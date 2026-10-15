import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, Optional

class OutlierHandler:
    """Handles outliers via clipping or removal based on IQR."""

    @staticmethod
    def handle_outliers(df: pd.DataFrame, method: str = "clip", target_column: Optional[str] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        df_out = df.copy()
        target = target_column or df_out.columns[-1]
        num_cols = [c for c in df_out.select_dtypes(include=[np.number]).columns if c != target]

        modified_count = 0
        if method == "clip":
            for col in num_cols:
                q25 = df_out[col].quantile(0.25)
                q75 = df_out[col].quantile(0.75)
                iqr = q75 - q25
                if iqr > 0:
                    lower = q25 - 1.5 * iqr
                    upper = q75 + 1.5 * iqr
                    clipped_mask = (df_out[col] < lower) | (df_out[col] > upper)
                    modified_count += int(clipped_mask.sum())
                    df_out[col] = df_out[col].clip(lower, upper)
            return df_out, {"method": "clip", "clipped_outliers_count": modified_count}
        else:
            # Remove
            mask = pd.Series(True, index=df_out.index)
            for col in num_cols:
                q25 = df_out[col].quantile(0.25)
                q75 = df_out[col].quantile(0.75)
                iqr = q75 - q25
                if iqr > 0:
                    lower = q25 - 1.5 * iqr
                    upper = q75 + 1.5 * iqr
                    col_mask = (df_out[col] >= lower) & (df_out[col] <= upper)
                    mask = mask & col_mask
            df_cleaned = df_out[mask].reset_index(drop=True)
            removed = len(df) - len(df_cleaned)
            return df_cleaned, {"method": "remove", "removed_rows_count": removed}

outlier_handler = OutlierHandler()
