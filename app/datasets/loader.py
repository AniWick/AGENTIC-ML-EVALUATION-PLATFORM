import os
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, Any

class DatasetLoader:
    """Loads CSV, Parquet, and JSON tabular datasets into pandas DataFrame."""

    @staticmethod
    def load(file_path: str) -> pd.DataFrame:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset file not found: {file_path}")
        
        ext = Path(file_path).suffix.lower()
        if ext in [".csv", ".txt"]:
            # Auto-detect delimiters
            return pd.read_csv(file_path)
        elif ext in [".parquet", ".pq"]:
            return pd.read_parquet(file_path)
        elif ext in [".json"]:
            return pd.read_json(file_path)
        elif ext in [".xlsx", ".xls"]:
            return pd.read_excel(file_path)
        else:
            # Try CSV fallback
            return pd.read_csv(file_path)

    @staticmethod
    def save(df: pd.DataFrame, file_path: str) -> None:
        p = Path(file_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        ext = p.suffix.lower()
        if ext in [".parquet", ".pq"]:
            df.to_parquet(file_path, index=False)
        else:
            df.to_csv(file_path, index=False)

dataset_loader = DatasetLoader()

