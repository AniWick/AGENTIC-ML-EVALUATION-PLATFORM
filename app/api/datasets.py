from fastapi import APIRouter, UploadFile, File, HTTPException

import shutil
import tempfile

from pathlib import Path
from datetime import datetime

from app.services.project_service import project_service
from app.datasets.versioning import dataset_version_manager
from app.core.database import db
from app.datasets.loader import dataset_loader

from app.transformations.cleaner import data_cleaner
from app.transformations.outliers import outlier_handler
from app.transformations.balancer import data_balancer
from app.transformations.scaler import data_scaler

from app.datasets.validator import compatibility_validator


router = APIRouter(tags=["datasets"])


@router.get("/datasets")
def list_datasets():

    with db._get_connection() as conn:
        return [
            dict(row)
            for row in conn.execute(
                """
                SELECT
                    id,
                    name,
                    current_version,
                    format,
                    project_id,
                    created_at
                FROM datasets
                ORDER BY created_at DESC
                """
            )
        ]


@router.post("/projects/{project_id}/datasets/{dataset_id}/select")
def select_existing_dataset(
    project_id: str,
    dataset_id: str
):

    if not project_service.get_project(project_id):
        raise HTTPException(
            status_code=404,
            detail="Project not found"
        )

    with db._get_connection() as conn:

        if not conn.execute(
            "SELECT id FROM datasets WHERE id = ?",
            (dataset_id,)
        ).fetchone():

            raise HTTPException(
                status_code=404,
                detail="Dataset not found"
            )

        conn.execute(
            """
            UPDATE projects
            SET dataset_id = ?, updated_at = ?
            WHERE id = ?
            """,
            (
                dataset_id,
                datetime.utcnow().isoformat(),
                project_id
            )
        )

        conn.commit()

    return {
        "status": "success",
        "selected_dataset_id": dataset_id
    }


@router.post("/projects/{project_id}/datasets")
def upload_project_dataset(
    project_id: str,
    file: UploadFile = File(...)
):

    proj = project_service.get_project(project_id)

    if not proj:
        raise HTTPException(
            status_code=404,
            detail="Project not found"
        )

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=Path(file.filename).suffix
    ) as tmp:

        shutil.copyfileobj(
            file.file,
            tmp
        )

        tmp_path = tmp.name

    try:

        res = project_service.upload_dataset(
            project_id,
            tmp_path,
            file.filename
        )

        return {
            "status": "success",
            "dataset": res
        }

    finally:

        if Path(tmp_path).exists():
            Path(tmp_path).unlink()


@router.get("/datasets/{dataset_id}/versions")
def get_dataset_versions(
    dataset_id: str
):

    return dataset_version_manager.list_versions(
        dataset_id
    )


@router.post("/datasets/{dataset_id}/transform/{operation}")
def transform_dataset(
    dataset_id: str,
    operation: str
):
    """Apply a safe, versioned dataset operation."""

    with db._get_connection() as conn:

        row = conn.execute(
            """
            SELECT current_version
            FROM datasets
            WHERE id = ?
            """,
            (dataset_id,)
        ).fetchone()

    if not row:
        raise HTTPException(
            status_code=404,
            detail="Dataset not found"
        )

    current_version = row["current_version"]

    dataset_path = (
        dataset_version_manager.get_version_path(
            dataset_id,
            current_version
        )
    )

    if not dataset_path:
        raise HTTPException(
            status_code=404,
            detail="Current dataset file not found"
        )

    df = dataset_loader.load(
        dataset_path
    )

    target = (
        df.columns[-1]
        if len(df.columns)
        else None
    )

    if operation == "clean":

        transformed, meta = (
            data_cleaner.handle_missing_values(
                df,
                target_column=target
            )
        )

    elif operation == "deduplicate":

        transformed, meta = (
            data_cleaner.remove_duplicates(df)
        )

    elif operation == "prune":

        transformed, meta = (
            outlier_handler.handle_outliers(
                df,
                "remove",
                target
            )
        )

    elif operation == "clip_outliers":

        transformed, meta = (
            outlier_handler.handle_outliers(
                df,
                "clip",
                target
            )
        )

    elif operation == "balance":

        transformed, meta = (
            data_balancer.balance_data(
                df,
                "smote",
                target
            )
        )

    elif operation == "scale":

        transformed, meta = (
            data_scaler.scale_data(
                df,
                "standard",
                target
            )
        )

    else:

        raise HTTPException(
            status_code=400,
            detail="Unknown dataset operation"
        )

    version = (
        dataset_version_manager.create_new_version(
            dataset_id,
            transformed,
            current_version,
            operation,
            meta
        )
    )

    return {
        "status": "success",
        "version": version.model_dump()
    }


@router.get("/projects/{project_id}/compatibility")
def check_project_compatibility(
    project_id: str
):

    project = project_service.get_project(
        project_id
    )

    if (
        not project
        or not project.model_id
        or not project.dataset_id
    ):
        raise HTTPException(
            status_code=400,
            detail="Select both a model and dataset first"
        )

    with db._get_connection() as conn:

        model = conn.execute(
            """
            SELECT file_path
            FROM models
            WHERE id = ?
            """,
            (project.model_id,)
        ).fetchone()

        dataset = conn.execute(
            """
            SELECT current_version
            FROM datasets
            WHERE id = ?
            """,
            (project.dataset_id,)
        ).fetchone()

    if not model:
        raise HTTPException(
            status_code=404,
            detail="Selected model not found"
        )

    if not dataset:
        raise HTTPException(
            status_code=404,
            detail="Selected dataset not found"
        )

    dataset_path = (
        dataset_version_manager.get_version_path(
            project.dataset_id,
            dataset["current_version"]
        )
    )

    if not dataset_path:
        raise HTTPException(
            status_code=404,
            detail="Dataset file not found"
        )

    return compatibility_validator.validate_compatibility(
        model["file_path"],
        dataset_path,
        project.model_id,
        project.dataset_id,
        dataset["current_version"]
    ).model_dump()


@router.get("/models/{model_id}/compatible-datasets")
def list_compatible_datasets(
    model_id: str
):

    # ---------------------------------------------------------
    # Get selected model
    # ---------------------------------------------------------

    with db._get_connection() as conn:

        model = conn.execute(
            """
            SELECT id, name, file_path
            FROM models
            WHERE id = ?
            """,
            (model_id,)
        ).fetchone()

        if not model:
            raise HTTPException(
                status_code=404,
                detail="Model not found"
            )

        datasets = conn.execute(
            """
            SELECT id, name, current_version
            FROM datasets
            ORDER BY created_at DESC
            """
        ).fetchall()

    compatible = []

    # ---------------------------------------------------------
    # Check every dataset independently.
    #
    # A broken/stale dataset must not crash the entire endpoint.
    # ---------------------------------------------------------

    for dataset in datasets:

        try:

            dataset_path = (
                dataset_version_manager.get_version_path(
                    dataset["id"],
                    dataset["current_version"]
                )
            )

            # Dataset doesn't physically exist anymore.
            if not dataset_path:
                continue

            result = (
                compatibility_validator.validate_compatibility(
                    model["file_path"],
                    dataset_path,
                    model["id"],
                    dataset["id"],
                    dataset["current_version"]
                )
            )

            if result.is_compatible:

                compatible.append(
                    {
                        "id": dataset["id"],
                        "name": dataset["name"],
                        "current_version": dataset["current_version"],
                        "warnings": result.warnings
                    }
                )

        except (FileNotFoundError, OSError) as e:

            # Do not allow one stale dataset to produce
            # a 500 for the whole compatibility endpoint.
            print(
                f"[Compatibility] Skipping dataset "
                f"{dataset['id']}: {e}"
            )

            continue

        except Exception as e:

            # Protect the endpoint from unexpected compatibility
            # failures for an individual dataset.
            print(
                f"[Compatibility] Error checking dataset "
                f"{dataset['id']}: {e}"
            )

            continue

    return {
        "model_id": model_id,
        "compatible_datasets": compatible
    }