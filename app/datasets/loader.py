import os
import pandas as pd
from pathlib import Path

from app.core.storage import storage_manager


class DatasetLoader:
    """Loads CSV, Parquet, JSON, and Excel tabular datasets into pandas DataFrame."""

    @staticmethod
    def load(file_path: str) -> pd.DataFrame:

        # Resolve old absolute paths / relative storage paths.
        resolved_path = storage_manager.resolve_path(file_path)

        if not resolved_path.exists():
            raise FileNotFoundError(
                f"Dataset file not found: {resolved_path}"
            )

        file_path = str(resolved_path)

        ext = resolved_path.suffix.lower()

        if ext in [".csv", ".txt"]:
            return pd.read_csv(file_path)

        elif ext in [".parquet", ".pq"]:
            return pd.read_parquet(file_path)

        elif ext == ".json":
            return pd.read_json(file_path)

        elif ext in [".xlsx", ".xls"]:
            return pd.read_excel(file_path)

        else:
            # CSV fallback
            return pd.read_csv(file_path)

    @staticmethod
    def save(df: pd.DataFrame, file_path: str) -> None:

        # Always save to the resolved/current storage location.
        resolved_path = storage_manager.resolve_path(file_path)

        resolved_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        ext = resolved_path.suffix.lower()

        if ext in [".parquet", ".pq"]:
            df.to_parquet(
                str(resolved_path),
                index=False
            )

        else:
            df.to_csv(
                str(resolved_path),
                index=False
            )


dataset_loader = DatasetLoader()