import sqlite3
import json
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime

from app.core.config import settings


class Database:
    """Relational SQLite database for authoritative project state, metadata, and history."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or (settings.storage_path / "platform.db")
        self.init_db()

    def _get_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def init_db(self):
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Projects
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS projects (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    model_id TEXT,
                    dataset_id TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)

            # Models
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS models (
                    id TEXT PRIMARY KEY,
                    project_id TEXT,
                    name TEXT NOT NULL,
                    framework TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    architecture TEXT,
                    input_shape TEXT,
                    output_shape TEXT,
                    parameter_count INTEGER,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
            """)

            # Datasets
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS datasets (
                    id TEXT PRIMARY KEY,
                    project_id TEXT,
                    name TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    current_version TEXT NOT NULL,
                    format TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
            """)

            # Dataset Versions
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS dataset_versions (
                    id TEXT PRIMARY KEY,
                    dataset_id TEXT NOT NULL,
                    version TEXT NOT NULL,
                    parent_version TEXT,
                    file_path TEXT NOT NULL,
                    operation TEXT,
                    parameters TEXT,
                    row_count INTEGER,
                    column_count INTEGER,
                    created_at TEXT NOT NULL
                )
            """)

            # Evaluations
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS evaluations (
                    id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    model_id TEXT NOT NULL,
                    dataset_version TEXT NOT NULL,
                    eval_type TEXT NOT NULL,
                    metrics TEXT NOT NULL,
                    quality_metrics TEXT,
                    robustness_metrics TEXT,
                    warnings TEXT,
                    created_at TEXT NOT NULL
                )
            """)

            # Experiments
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS experiments (
                    id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    model_id TEXT NOT NULL,
                    input_dataset_version TEXT NOT NULL,
                    output_dataset_version TEXT NOT NULL,
                    operation TEXT NOT NULL,
                    parameters TEXT,
                    metrics_before TEXT,
                    metrics_after TEXT,
                    comparison TEXT,
                    agent_reasoning TEXT,
                    created_at TEXT NOT NULL
                )
            """)

            # Chat Messages
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS chat_messages (
                    id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    sender TEXT NOT NULL,
                    message TEXT NOT NULL,
                    metadata TEXT,
                    created_at TEXT NOT NULL
                )
            """)

            # Reports
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS reports (
                    id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    summary TEXT,
                    content_markdown TEXT NOT NULL,
                    content_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
            """)

            conn.commit()

            # Idempotent schema migrations for hardened model support
            cursor.execute("PRAGMA table_info(models)")
            existing_cols = {col[1] for col in cursor.fetchall()}
            if "is_hardened" not in existing_cols:
                cursor.execute("ALTER TABLE models ADD COLUMN is_hardened BOOLEAN DEFAULT 0")
            if "parent_model_id" not in existing_cols:
                cursor.execute("ALTER TABLE models ADD COLUMN parent_model_id TEXT DEFAULT NULL")
            conn.commit()

    # ------------------------------------------------------------------
    # Model / dataset registration & lookup methods
    # ------------------------------------------------------------------

    def get_model(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a model record by ID, including hardening migration columns."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT
                    id, project_id, name, framework, file_path,
                    architecture, input_shape, output_shape,
                    parameter_count, status, is_hardened, parent_model_id, created_at
                FROM models
                WHERE id = ?
                """,
                (model_id,),
            )
            row = cursor.fetchone()

        if not row:
            return None

        return dict(row)

    def list_models(self) -> List[Dict[str, Any]]:
        """Retrieve all models stored in SQLite global memory including hardening metadata."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT
                    id, project_id, name, framework, file_path,
                    architecture, input_shape, output_shape,
                    parameter_count, status, is_hardened, parent_model_id, created_at
                FROM models
                ORDER BY created_at DESC
                """
            )
            rows = cursor.fetchall()
        return [dict(row) for row in rows]

    def get_all_models(self) -> List[Dict[str, Any]]:
        """Alias for list_models."""
        return self.list_models()

    def get_dataset(self, dataset_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a dataset record by ID."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT
                    id, project_id, name, file_path,
                    current_version, format, created_at
                FROM datasets
                WHERE id = ?
                """,
                (dataset_id,),
            )
            row = cursor.fetchone()

        if not row:
            return None

        return dict(row)

    def list_datasets(self) -> List[Dict[str, Any]]:
        """Retrieve all datasets stored in SQLite global memory."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT
                    id, project_id, name, file_path,
                    current_version, format, created_at
                FROM datasets
                ORDER BY created_at DESC
                """
            )
            rows = cursor.fetchall()
        return [dict(row) for row in rows]

    def get_all_datasets(self) -> List[Dict[str, Any]]:
        """Alias for list_datasets."""
        return self.list_datasets()

    def register_model_if_not_exists(self, model_id: str, project_id: str, name: str, file_path: str, framework: str = "sklearn") -> None:
        """Guarantees model is persisted in global memory without overwriting previous records."""
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO models (id, project_id, name, framework, file_path, status, created_at)
                VALUES (?, ?, ?, ?, ?, 'ready', ?)
                """,
                (model_id, project_id, name, framework, file_path, datetime.utcnow().isoformat())
            )
            conn.commit()

    def register_dataset_if_not_exists(self, dataset_id: str, project_id: str, name: str, file_path: str, version: str = "v001") -> None:
        """Guarantees dataset is persisted in global memory without overwriting previous records."""
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO datasets (id, project_id, name, file_path, current_version, format, created_at)
                VALUES (?, ?, ?, ?, ?, 'csv', ?)
                """,
                (dataset_id, project_id, name, file_path, version, datetime.utcnow().isoformat())
            )
            conn.commit()

    def get_all_experiments_for_project(self, project_id: str) -> List[Dict[str, Any]]:
        """Retrieve all experiment history for a specific project from SQLite."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT
                    id, project_id, model_id, input_dataset_version,
                    output_dataset_version, operation, parameters,
                    metrics_before, metrics_after, comparison, created_at
                FROM experiments
                WHERE project_id = ?
                ORDER BY created_at DESC
                """,
                (project_id,),
            )
            rows = cursor.fetchall()
        return [dict(row) for row in rows]

    # ------------------------------------------------------------------
    # Chat message methods
    # ------------------------------------------------------------------

    def insert_chat_message(
        self,
        project_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Store a chat message and return its generated ID."""
        message_id = f"msg_{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}"
        created_at = datetime.utcnow().isoformat()
        metadata_json = json.dumps(metadata or {}, default=str)

        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO chat_messages
                (id, project_id, sender, message, metadata, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (message_id, project_id, role, content, metadata_json, created_at),
            )
            conn.commit()

        return message_id

    def get_chat_messages(
        self,
        project_id: str,
    ) -> List[Dict[str, Any]]:
        """Retrieve chat history for a project in chronological order."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT id, project_id, sender, message, metadata, created_at
                FROM chat_messages
                WHERE project_id = ?
                ORDER BY created_at ASC
                """,
                (project_id,),
            )
            rows = cursor.fetchall()

        messages = []
        for row in rows:
            metadata = {}
            if row["metadata"]:
                try:
                    metadata = json.loads(row["metadata"])
                except (json.JSONDecodeError, TypeError):
                    metadata = {}

            messages.append({
                "id": row["id"],
                "project_id": row["project_id"],
                "sender": row["sender"],
                "message": row["message"],
                "metadata": metadata,
                "created_at": row["created_at"],
                # Canonical chat names used by the frontend/API, retained
                # alongside legacy sender/message names for compatibility.
                "role": row["sender"],
                "content": row["message"],
            })

        return messages


db = Database()
