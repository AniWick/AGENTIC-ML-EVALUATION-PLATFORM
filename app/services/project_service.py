import os
import shutil
import sqlite3
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
from app.core.database import db
from app.core.storage import storage_manager
from app.schemas.project import ProjectResponse
from app.models.loader import model_loader
from app.datasets.versioning import dataset_version_manager

class ProjectService:
    """Service handling project lifecycles, file uploads, and state synchronization."""

    @staticmethod
    def create_project(name: str, description: Optional[str] = None) -> ProjectResponse:
        project_id = f"proj_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        now = datetime.utcnow().isoformat()
        with db._get_connection() as conn:
            conn.execute(
                "INSERT INTO projects (id, name, description, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
                (project_id, name, description, now, now)
            )
            conn.commit()

        storage_manager.get_project_dir(project_id)
        return ProjectResponse(
            id=project_id,
            name=name,
            description=description,
            created_at=now,
            updated_at=now
        )

    @staticmethod
    def get_project(project_id: str) -> Optional[ProjectResponse]:
        with db._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM projects WHERE id = ?", (project_id,))
            row = cursor.fetchone()
            if not row:
                return None
            return ProjectResponse(
                id=row["id"],
                name=row["name"],
                description=row["description"],
                model_id=row["model_id"],
                dataset_id=row["dataset_id"],
                created_at=row["created_at"],
                updated_at=row["updated_at"]
            )

    @staticmethod
    def list_projects() -> List[ProjectResponse]:
        with db._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM projects ORDER BY created_at DESC")
            rows = cursor.fetchall()
            return [
                ProjectResponse(
                    id=row["id"],
                    name=row["name"],
                    description=row["description"],
                    model_id=row["model_id"],
                    dataset_id=row["dataset_id"],
                    created_at=row["created_at"],
                    updated_at=row["updated_at"]
                )
                for row in rows
            ]

    @staticmethod
    def upload_model(project_id: str, file_path_source: str, filename: str) -> Dict[str, Any]:
        model_id = f"model_{project_id}"
        m_dir = storage_manager.get_model_dir(model_id)
        dest = m_dir / filename
        shutil.copy2(file_path_source, dest)

        model, adapter, framework = model_loader.load_model(str(dest))
        inspect_info = adapter.inspect(model)

        now = datetime.utcnow().isoformat()
        with db._get_connection() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO models (id, project_id, name, framework, file_path, architecture, input_shape, output_shape, parameter_count, status, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                model_id, project_id, filename, framework, str(dest),
                inspect_info.get("model_class"),
                str(inspect_info.get("input_shape")),
                str(inspect_info.get("output_shape")),
                inspect_info.get("parameter_count"),
                "loaded", now
            ))
            conn.execute("UPDATE projects SET model_id = ?, updated_at = ? WHERE id = ?", (model_id, now, project_id))
            conn.commit()

        inspect_info["model_id"] = model_id
        inspect_info["file_path"] = str(dest)
        return inspect_info

    @staticmethod
    def upload_dataset(project_id: str, file_path_source: str, filename: str) -> Dict[str, Any]:
        dataset_id = f"dataset_{project_id}"
        v001_info = dataset_version_manager.create_initial_version(dataset_id, file_path_source)

        now = datetime.utcnow().isoformat()
        with db._get_connection() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO datasets (id, project_id, name, file_path, current_version, format, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                dataset_id, project_id, filename, v001_info.file_path, "v001", Path(filename).suffix, now
            ))
            conn.execute("""
                INSERT OR REPLACE INTO dataset_versions (id, dataset_id, version, parent_version, file_path, operation, parameters, row_count, column_count, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                f"{dataset_id}_v001", dataset_id, "v001", None, v001_info.file_path,
                "initial_upload", "{}", v001_info.row_count, v001_info.column_count, now
            ))
            conn.execute("UPDATE projects SET dataset_id = ?, updated_at = ? WHERE id = ?", (dataset_id, now, project_id))
            conn.commit()

        return v001_info.model_dump()

project_service = ProjectService()

