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

db = Database()

