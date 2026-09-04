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

    # ------------------------------------------------------------------
    # Model / dataset lookup methods
    # ------------------------------------------------------------------

    def get_model(self, model_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a model record by ID.
        """

        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT
                    id,
                    project_id,
                    name,
                    framework,
                    file_path,
                    architecture,
                    input_shape,
                    output_shape,
                    parameter_count,
                    status,
                    created_at
                FROM models
                WHERE id = ?
                """,
                (model_id,),
            )

            row = cursor.fetchone()

        if not row:
            return None

        return {
            "id": row["id"],
            "project_id": row["project_id"],
            "name": row["name"],
            "framework": row["framework"],
            "file_path": row["file_path"],
            "architecture": row["architecture"],
            "input_shape": row["input_shape"],
            "output_shape": row["output_shape"],
            "parameter_count": row["parameter_count"],
            "status": row["status"],
            "created_at": row["created_at"],
        }

    def get_dataset(self, dataset_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a dataset record by ID.
        """

        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT
                    id,
                    project_id,
                    name,
                    file_path,
                    current_version,
                    format,
                    created_at
                FROM datasets
                WHERE id = ?
                """,
                (dataset_id,),
            )

            row = cursor.fetchone()

        if not row:
            return None

        return {
            "id": row["id"],
            "project_id": row["project_id"],
            "name": row["name"],
            "file_path": row["file_path"],
            "current_version": row["current_version"],
            "format": row["format"],
            "created_at": row["created_at"],
        }

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
        """
        Store a chat message and return its generated ID.

        Python/API terminology:
            role    -> stored in sender
            content -> stored in message
        """

        message_id = (
            f"msg_{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}"
        )
        created_at = datetime.utcnow().isoformat()

        metadata_json = json.dumps(
            metadata or {},
            default=str,
        )

        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO chat_messages
                (
                    id,
                    project_id,
                    sender,
                    message,
                    metadata,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    message_id,
                    project_id,
                    role,
                    content,
                    metadata_json,
                    created_at,
                ),
            )

            conn.commit()

        return message_id

    def get_chat_messages(
        self,
        project_id: str,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve chat history for a project in chronological order.
        """

        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT
                    id,
                    project_id,
                    sender,
                    message,
                    metadata,
                    created_at
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
            })

        return messages


db = Database()