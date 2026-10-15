import os
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from typing import Dict, Any, List
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from app.core.config import settings
from app.services.project_service import project_service

class PresetService:
    """Manages pre-built benchmark datasets and pre-trained models with caching."""

    def __init__(self):
        self.presets_dir = settings.storage_path / "preset_downloads"
        self.datasets_dir = self.presets_dir / "datasets"
        self.models_dir = self.presets_dir / "models"
        self.datasets_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)

    def list_available_presets(self) -> List[Dict[str, Any]]:
        return [
            {
                "id": "iris",
                "name": "Iris Flower Classification",
                "description": "Classic 3-class benchmark with 4 morphological features. Includes slight class imbalance to test agentic SMOTE balancing.",
                "features_count": 4,
                "target_classes": ["setosa", "versicolor", "virginica"],
                "task_type": "classification"
            },
            {
                "id": "breast_cancer",
                "name": "Breast Cancer Diagnostic",
                "description": "Binary classification dataset with 30 numerical cellular features. Tests precision, recall, and missing value imputation.",
                "features_count": 30,
                "target_classes": ["malignant", "benign"],
                "task_type": "classification"
            },
            {
                "id": "wine",
                "name": "Wine Cultivar Recognition",
                "description": "13 continuous chemical attributes for 3 wine cultivars. Ideal for multi-class robustness and scaling evaluation.",
                "features_count": 13,
                "target_classes": ["class_0", "class_1", "class_2"],
                "task_type": "classification"
            }
        ]

    def ensure_preset_downloaded(self, preset_id: str) -> Dict[str, str]:
        """Generates/downloads dataset and model to persistent cache folder if not already cached."""
        ds_file = self.datasets_dir / f"{preset_id}.csv"
        model_file = self.models_dir / f"{preset_id}_model.joblib"

        if ds_file.exists() and model_file.exists():
            return {
                "dataset_path": str(ds_file),
                "model_path": str(model_file),
                "cached": True
            }

        if preset_id == "iris":
            data = load_iris(as_frame=True)
            df = data.frame.copy()
            # Rename target column and add class labels
            df.rename(columns={"target": "species"}, inplace=True)
            # Create a realistic slight class imbalance (e.g. 50, 45, 15) and inject 3 missing values to showcase agent
            imbalanced_indices = list(range(0, 50)) + list(range(50, 95)) + list(range(100, 115))
            df = df.iloc[imbalanced_indices].reset_index(drop=True)
            df.loc[2:4, "sepal length (cm)"] = np.nan

            X = df.drop(columns=["species"]).fillna(df.mean(numeric_only=True))
            y = df["species"]
            clf = RandomForestClassifier(n_estimators=25, random_state=42)
            clf.fit(X, y)

        elif preset_id == "breast_cancer":
            data = load_breast_cancer(as_frame=True)
            df = data.frame.copy()
            df.rename(columns={"target": "diagnosis"}, inplace=True)
            # Inject a few missing values
            df.loc[1:5, "mean radius"] = np.nan

            X = df.drop(columns=["diagnosis"]).fillna(df.mean(numeric_only=True))
            y = df["diagnosis"]
            clf = RandomForestClassifier(n_estimators=30, random_state=42)
            clf.fit(X, y)

        elif preset_id == "wine":
            data = load_wine(as_frame=True)
            df = data.frame.copy()
            df.rename(columns={"target": "cultivar"}, inplace=True)
            X = df.drop(columns=["cultivar"])
            y = df["cultivar"]
            clf = RandomForestClassifier(n_estimators=20, random_state=42)
            clf.fit(X, y)

        else:
            raise ValueError(f"Unknown preset dataset: {preset_id}")

        # Save to cached folder
        df.to_csv(ds_file, index=False)
        joblib.dump(clf, model_file)

        return {
            "dataset_path": str(ds_file),
            "model_path": str(model_file),
            "cached": False
        }

    def load_preset_into_project(self, project_id: str, preset_id: str) -> Dict[str, Any]:
        """Ensures preset is cached, then uploads both model and dataset directly to project."""
        cached_paths = self.ensure_preset_downloaded(preset_id)
        
        m_info = project_service.upload_model(
            project_id=project_id,
            file_path_source=cached_paths["model_path"],
            filename=f"{preset_id}_model.joblib"
        )
        d_info = project_service.upload_dataset(
            project_id=project_id,
            file_path_source=cached_paths["dataset_path"],
            filename=f"{preset_id}.csv"
        )

        return {
            "status": "success",
            "preset_id": preset_id,
            "was_cached": cached_paths["cached"],
            "model": m_info,
            "dataset": d_info
        }

preset_service = PresetService()

