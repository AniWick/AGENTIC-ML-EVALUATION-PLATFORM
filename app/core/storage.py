import os
import json
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, List
from app.core.config import settings

class StorageManager:
    """Manages file storage hierarchy for models, datasets, versions, experiments, and reports."""

    def __init__(self, base_storage_dir: Optional[Path] = None):
        self.base_dir = base_storage_dir or settings.storage_path
        self.projects_dir = self.base_dir / "projects"
        self.models_dir = self.base_dir / "models"
        self.datasets_dir = self.base_dir / "datasets"
        self.experiments_dir = self.base_dir / "experiments"
        self.reports_dir = self.base_dir / "reports"

        for d in [self.projects_dir, self.models_dir, self.datasets_dir, self.experiments_dir, self.reports_dir]:
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
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)

    def load_json(self, path: Path) -> Optional[Any]:
        if not path.exists():
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

storage_manager = StorageManager()

