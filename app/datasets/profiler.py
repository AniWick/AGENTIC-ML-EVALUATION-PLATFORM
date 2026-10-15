import numpy as np
import pandas as pd
from typing import Dict, Any
from app.datasets.loader import dataset_loader

class DatasetProfiler:
    """Computes comprehensive statistical summary of columns, distributions, and quantiles."""

    @staticmethod
    def profile(file_path: str) -> Dict[str, Any]:
        df = dataset_loader.load(file_path)
        stats = {}
        for col in df.columns:
            col_data = df[col]
            if pd.api.types.is_numeric_dtype(col_data):
                clean_s = col_data.dropna()
                if len(clean_s) > 0:
                    stats[col] = {
                        "type": "numeric",
                        "mean": float(clean_s.mean()),
                        "std": float(clean_s.std()) if len(clean_s) > 1 else 0.0,
                        "min": float(clean_s.min()),
                        "q25": float(clean_s.quantile(0.25)),
                        "median": float(clean_s.median()),
                        "q75": float(clean_s.quantile(0.75)),
                        "max": float(clean_s.max()),
                        "skew": float(clean_s.skew()) if len(clean_s) > 2 else 0.0,
                        "missing_count": int(col_data.isna().sum()),
                        "missing_ratio": float(col_data.isna().mean())
                    }
            else:
                vc = col_data.value_counts(dropna=False).head(10).to_dict()
                stats[col] = {
                    "type": "categorical",
                    "unique_count": int(col_data.nunique(dropna=True)),
                    "top_values": {str(k): int(v) for k, v in vc.items()},
                    "missing_count": int(col_data.isna().sum()),
                    "missing_ratio": float(col_data.isna().mean())
                }
        return stats

dataset_profiler = DatasetProfiler()

