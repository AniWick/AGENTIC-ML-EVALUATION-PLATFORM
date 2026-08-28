import os
import json
import shutil
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
from app.core.storage import storage_manager
from app.datasets.loader import dataset_loader
from app.schemas.dataset import DatasetVersionInfo

class DatasetVersionManager:
    """Manages immutable dataset versions (v001, v002, ...) and lineage tracking."""

    @staticmethod
    def create_initial_version(dataset_id: str, original_file_path: str) -> DatasetVersionInfo:
        """Stores the original dataset as immutable v001."""
        ds_dir = storage_manager.get_dataset_dir(dataset_id)
        original_dest = ds_dir / "original" / Path(original_file_path).name
        shutil.copy2(original_file_path, original_dest)

        v001_dir = storage_manager.get_dataset_version_dir(dataset_id, "v001")
        v001_file = v001_dir / f"dataset{Path(original_file_path).suffix}"
        shutil.copy2(original_file_path, v001_file)

        df = dataset_loader.load(str(v001_file))
        rows, cols = df.shape

        info = DatasetVersionInfo(
            version="v001",
            parent_version=None,
            file_path=str(v001_file),
            operation="initial_upload",
            parameters={},
            row_count=rows,
            column_count=cols,
            created_at=datetime.utcnow().isoformat()
        )

        storage_manager.save_json(v001_dir / "metadata.json", info.model_dump())
        return info

    @staticmethod
    def create_new_version(
        dataset_id: str,
        df_transformed: pd.DataFrame,
        parent_version: str,
        operation: str,
        parameters: Dict[str, Any],
        file_format: str = ".csv"
    ) -> DatasetVersionInfo:
        """Generates the next sequential version (e.g. v002, v003) from a transformed DataFrame."""
        ds_dir = storage_manager.get_dataset_dir(dataset_id)
        versions_dir = ds_dir / "versions"
        
        # Determine next version number
        existing_versions = [d.name for d in versions_dir.iterdir() if d.is_dir() and d.name.startswith("v")]
        existing_nums = [int(v.replace("v", "")) for v in existing_versions if v.replace("v", "").isdigit()]
        next_num = max(existing_nums, default=0) + 1
        new_version_tag = f"v{next_num:03d}"

        new_v_dir = storage_manager.get_dataset_version_dir(dataset_id, new_version_tag)
        new_file_path = new_v_dir / f"dataset{file_format}"

        dataset_loader.save(df_transformed, str(new_file_path))
        rows, cols = df_transformed.shape

        info = DatasetVersionInfo(
            version=new_version_tag,
            parent_version=parent_version,
            file_path=str(new_file_path),
            operation=operation,
            parameters=parameters,
            row_count=rows,
            column_count=cols,
            created_at=datetime.utcnow().isoformat()
        )

        storage_manager.save_json(new_v_dir / "metadata.json", info.model_dump())
        return info

    @staticmethod
    def get_version_info(dataset_id: str, version: str) -> Optional[DatasetVersionInfo]:
        v_dir = storage_manager.get_dataset_version_dir(dataset_id, version)
        meta_file = v_dir / "metadata.json"
        if not meta_file.exists():
            return None
        data = storage_manager.load_json(meta_file)
        return DatasetVersionInfo(**data)

    @staticmethod
    def get_version_path(dataset_id: str, version: str) -> Optional[str]:
        info = DatasetVersionManager.get_version_info(dataset_id, version)
        return info.file_path if info else None

    @staticmethod
    def list_versions(dataset_id: str) -> List[DatasetVersionInfo]:
        ds_dir = storage_manager.get_dataset_dir(dataset_id)
        versions_dir = ds_dir / "versions"
        versions = []
        if not versions_dir.exists():
            return []
        for d in sorted(versions_dir.iterdir()):
            if d.is_dir() and d.name.startswith("v"):
                meta_file = d / "metadata.json"
                if meta_file.exists():
                    data = storage_manager.load_json(meta_file)
                    versions.append(DatasetVersionInfo(**data))
        return versions

dataset_version_manager = DatasetVersionManager()

