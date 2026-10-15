import os
import json
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, List
from app.core.config import settings


class StorageManager:
    """Manages file storage hierarchy for models, datasets, versions, experiments, and reports."""

    def __init__(self, base_storage_dir: Optional[Path] = None):
        self.base_dir = Path(base_storage_dir or settings.storage_path).resolve()

        self.projects_dir = self.base_dir / "projects"
        self.models_dir = self.base_dir / "models"
        self.datasets_dir = self.base_dir / "datasets"
        self.experiments_dir = self.base_dir / "experiments"
        self.reports_dir = self.base_dir / "reports"

        for d in [
            self.projects_dir,
            self.models_dir,
            self.datasets_dir,
            self.experiments_dir,
            self.reports_dir
        ]:
            d.mkdir(parents=True, exist_ok=True)

    def get_project_dir(self, project_id: str) -> Path:
        p = self.projects_dir / project_id
        p.mkdir(parents=True, exist_ok=True)
        return p

    def get_model_dir(self, model_id: str) -> Path:
        p = self.models_dir / model_id
        p.mkdir(parents=True, exist_ok=True)
        return p

    def get_dataset_dir(self, dataset_id: str) -> Path:
        p = self.datasets_dir / dataset_id
        p.mkdir(parents=True, exist_ok=True)

        (p / "original").mkdir(parents=True, exist_ok=True)
        (p / "versions").mkdir(parents=True, exist_ok=True)

        return p

    def get_dataset_version_dir(self, dataset_id: str, version: str) -> Path:
        p = self.datasets_dir / dataset_id / "versions" / version
        p.mkdir(parents=True, exist_ok=True)
        return p

    def get_experiment_dir(self, experiment_id: str) -> Path:
        p = self.experiments_dir / experiment_id
        p.mkdir(parents=True, exist_ok=True)
        return p

    def save_json(self, path: Path, data: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)

    def load_json(self, path: Path) -> Optional[Any]:
        if not path.exists():
            return None

        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def resolve_path(self, file_path: str) -> Path:
        """
        Resolve a stored file path safely.

        Supports:
        1. Current absolute paths.
        2. Old absolute paths that used a previous project directory name.
        3. Relative paths such as:
           storage/models/model_x/model.joblib
        4. Paths relative to the storage directory.
        """

        if not file_path:
            return Path(file_path)

        path = Path(file_path)

        # ---------------------------------------------------------
        # Case 1: Absolute path that still exists
        # ---------------------------------------------------------
        if path.is_absolute() and path.exists():
            return path

        # ---------------------------------------------------------
        # Case 2: Old absolute path.
        #
        # Example:
        # C:\...\Project1_CDAC\storage\models\model_x\model.joblib
        #
        # Current:
        # C:\...\Project_CDAC\storage\models\model_x\model.joblib
        #
        # We recover everything after "storage".
        # ---------------------------------------------------------
        if path.is_absolute():
            parts = path.parts

            storage_index = None

            for i, part in enumerate(parts):
                if part.lower() == "storage":
                    storage_index = i
                    break

            if storage_index is not None:
                relative_inside_storage = Path(
                    *parts[storage_index + 1:]
                )

                candidate = self.base_dir / relative_inside_storage

                if candidate.exists():
                    return candidate

        # ---------------------------------------------------------
        # Case 3: Path already begins with "storage"
        # ---------------------------------------------------------
        normalized = str(file_path).replace("\\", "/")

        if normalized.lower().startswith("storage/"):
            relative_path = normalized.split("/", 1)[1]
            candidate = self.base_dir / relative_path

            if candidate.exists():
                return candidate

        # ---------------------------------------------------------
        # Case 4: Relative path.
        # ---------------------------------------------------------
        candidate = self.base_dir / path

        if candidate.exists():
            return candidate

        # ---------------------------------------------------------
        # Return the best resolved candidate even if it does not
        # exist, so the caller can produce the proper error.
        # ---------------------------------------------------------
        return candidate

    def to_relative_storage_path(self, file_path: str) -> str:
        """
        Convert an absolute storage path into a portable path.

        Example:
        C:/.../Project_CDAC/storage/models/model_x/model.joblib

        becomes:

        storage/models/model_x/model.joblib
        """

        path = Path(file_path)

        try:
            relative = path.resolve().relative_to(self.base_dir)
            return str(Path("storage") / relative)
        except ValueError:
            return str(path)


storage_manager = StorageManager()