from typing import Dict, Any, Callable, List, Optional
import json
from datetime import datetime

from app.models.inspector import model_inspector
from app.datasets.inspector import dataset_inspector
from app.datasets.validator import compatibility_validator
from app.datasets.versioning import dataset_version_manager
from app.datasets.loader import dataset_loader

from app.evaluation.performance import performance_evaluator
from app.evaluation.data_quality import data_quality_evaluator
from app.evaluation.robustness import robustness_evaluator
from app.evaluation.comparison import evaluation_comparator

from app.transformations.cleaner import data_cleaner
from app.transformations.balancer import data_balancer
from app.transformations.outliers import outlier_handler
from app.transformations.scaler import data_scaler
from app.transformations.noise import data_corruption

from app.memory.chroma_memory import project_memory
from app.core.database import db


class ToolRegistry:
    """Central registry defining all available tools."""

    def __init__(self):
        self.tools: Dict[str, Dict[str, Any]] = {}
        self._register_default_tools()

    def register(
        self,
        name: str,
        description: str,
        parameters_schema: Dict[str, Any],
        handler: Callable,
    ):
        self.tools[name] = {
            "name": name,
            "description": description,
            "parameters": parameters_schema,
            "handler": handler,
        }

    def get_tool_definitions(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": tool["name"],
                "description": tool["description"],
                "parameters": tool["parameters"],
            }
            for tool in self.tools.values()
        ]

    def execute_tool(
        self,
        name: str,
        params: Dict[str, Any],
    ) -> Dict[str, Any]:

        if name not in self.tools:
            return {
                "status": "error",
                "message": (
                    f"Tool '{name}' not found. "
                    f"Available tools: {list(self.tools.keys())}"
                ),
            }

        try:
            handler = self.tools[name]["handler"]

            # Always pass a dictionary to the handler.
            params = params or {}

            result = handler(**params)

            return {
                "status": "success",
                "result": result,
            }

        except Exception as exc:
            return {
                "status": "error",
                "message": (
                    f"Error executing tool '{name}': {str(exc)}"
                ),
            }

    def _register_default_tools(self):

        # =========================================================
        # 1. INSPECT MODEL
        # =========================================================

        self.register(
            name="inspect_model",
            description=(
                "Inspects model architecture, framework, input shape, "
                "parameter count, and output classes."
            ),
            parameters_schema={
                "type": "object",
                "properties": {
                    "model_path": {
                        "type": "string",
                        "description": "Absolute path to model file",
                    },
                    "model_id": {
                        "type": "string",
                        "default": "model_001",
                    },
                },
                "required": ["model_path"],
            },
            handler=lambda model_path, model_id="model_001":
                model_inspector.inspect(
                    model_path,
                    model_id,
                ),
        )

        # =========================================================
        # 2. INSPECT DATASET
        # =========================================================

        self.register(
            name="inspect_dataset",
            description=(
                "Inspects dataset rows, columns, data types, "
                "missing values, duplicates, class distribution, "
                "and outliers."
            ),
            parameters_schema={
                "type": "object",
                "properties": {
                    "dataset_path": {
                        "type": "string",
                        "description": "Path to dataset file",
                    },
                    "dataset_id": {
                        "type": "string",
                        "default": "dataset_001",
                    },
                    "version": {
                        "type": "string",
                        "default": "v001",
                    },
                    "target_col": {
                        "type": "string",
                        "description": "Optional target column name",
                    },
                },
                "required": ["dataset_path"],
            },
            handler=lambda
                dataset_path,
                dataset_id="dataset_001",
                version="v001",
                target_col=None:
                    dataset_inspector.inspect(
                        dataset_path,
                        dataset_id,
                        version,
                        target_col,
                    ),
        )

        # =========================================================
        # 3. VALIDATE MODEL / DATASET
        # =========================================================

        self.register(
            name="validate_model_dataset",
            description=(
                "Checks compatibility between the model and dataset "
                "(input shapes, feature count, classes)."
            ),
            parameters_schema={
                "type": "object",
                "properties": {
                    "model_path": {
                        "type": "string",
                    },
                    "dataset_path": {
                        "type": "string",
                    },
                    "model_id": {
                        "type": "string",
                        "default": "model_001",
                    },
                    "dataset_id": {
                        "type": "string",
                        "default": "dataset_001",
                    },
                    "version": {
                        "type": "string",
                        "default": "v001",
                    },
                },
                "required": [
                    "model_path",
                    "dataset_path",
                ],
            },
            handler=lambda
                model_path,
                dataset_path,
                model_id="model_001",
                dataset_id="dataset_001",
                version="v001":
                    compatibility_validator
                    .validate_compatibility(
                        model_path,
                        dataset_path,
                        model_id,
                        dataset_id,
                        version,
                    )
                    .model_dump(),
        )

        # =========================================================
        # 4. EVALUATE PERFORMANCE
        # =========================================================

        self.register(
            name="evaluate_performance",
            description=(
                "Evaluates objective ML metrics including accuracy, "
                "macro F1, minority F1, precision, recall, "
                "confusion matrix, and RMSE."
            ),
            parameters_schema={
                "type": "object",
                "properties": {
                    "model_path": {
                        "type": "string",
                    },
                    "dataset_path": {
                        "type": "string",
                    },
                    "target_column": {
                        "type": "string",
                        "description": "Target column name",
                    },
                    "task_type": {
                        "type": "string",
                        "default": "classification",
                        "enum": [
                            "classification",
                            "regression",
                        ],
                    },
                },
                "required": [
                    "model_path",
                    "dataset_path",
                ],
            },
            handler=lambda
                model_path,
                dataset_path,
                target_column=None,
                task_type="classification":
                    performance_evaluator
                    .evaluate(
                        model_path,
                        dataset_path,
                        target_column,
                        task_type,
                    )
                    .model_dump(),
        )

        # =========================================================
        # 5. DATA QUALITY
        # =========================================================

        self.register(
            name="evaluate_data_quality",
            description=(
                "Audits dataset for missingness, class imbalance "
                "ratio, duplicates, outliers, and constant columns."
            ),
            parameters_schema={
                "type": "object",
                "properties": {
                    "dataset_path": {
                        "type": "string",
                    },
                    "target_column": {
                        "type": "string",
                    },
                },
                "required": ["dataset_path"],
            },
            handler=lambda
                dataset_path,
                target_column=None:
                    data_quality_evaluator
                    .evaluate(
                        dataset_path,
                        target_column,
                    )
                    .model_dump(),
        )

        # =========================================================
        # 6. ROBUSTNESS
        # =========================================================

        self.register(
            name="evaluate_robustness",
            description=(
                "Evaluates model stability and performance "
                "degradation under Gaussian noise perturbation."
            ),
            parameters_schema={
                "type": "object",
                "properties": {
                    "model_path": {
                        "type": "string",
                    },
                    "dataset_path": {
                        "type": "string",
                    },
                    "target_column": {
                        "type": "string",
                    },
                    "noise_level": {
                        "type": "number",
                        "default": 0.10,
                    },
                },
                "required": [
                    "model_path",
                    "dataset_path",
                ],
            },
            handler=lambda
                model_path,
                dataset_path,
                target_column=None,
                noise_level=0.10:
                    robustness_evaluator
                    .evaluate_noise_robustness(
                        model_path,
                        dataset_path,
                        target_column,
                        noise_level,
                    )
                    .model_dump(),
        )

        # =========================================================
        # 7. HANDLE MISSING VALUES
        # =========================================================

        def _handle_missing(
            dataset_id: str,
            current_version: str,
            strategy: str = "median",
            target_column: Optional[str] = None,
        ):

            dataset_path = (
                dataset_version_manager
                .get_version_path(
                    dataset_id,
                    current_version,
                )
            )

            df = dataset_loader.load(dataset_path)

            df_clean, metadata = (
                data_cleaner.handle_missing_values(
                    df,
                    strategy,
                    target_column,
                )
            )

            new_version = (
                dataset_version_manager
                .create_new_version(
                    dataset_id=dataset_id,
                    df_transformed=df_clean,
                    parent_version=current_version,
                    operation="handle_missing_values",
                    parameters=metadata,
                )
            )

            return new_version.model_dump()

        self.register(
            name="handle_missing_values",
            description=(
                "Imputes missing values and creates a new "
                "immutable dataset version."
            ),
            parameters_schema={
                "type": "object",
                "properties": {
                    "dataset_id": {
                        "type": "string",
                    },
                    "current_version": {
                        "type": "string",
                    },
                    "strategy": {
                        "type": "string",
                        "enum": [
                            "median",
                            "mean",
                            "mode",
                        ],
                        "default": "median",
                    },
                    "target_column": {
                        "type": "string",
                    },
                },
                "required": [
                    "dataset_id",
                    "current_version",
                ],
            },
            handler=_handle_missing,
        )

        # =========================================================
        # 8. BALANCE DATA
        # =========================================================

        def _balance(
            dataset_id: str,
            current_version: str,
            method: str = "smote",
            target_column: Optional[str] = None,
        ):

            dataset_path = (
                dataset_version_manager
                .get_version_path(
                    dataset_id,
                    current_version,
                )
            )

            df = dataset_loader.load(dataset_path)

            df_balanced, metadata = (
                data_balancer.balance_data(
                    df,
                    method,
                    target_column,
                )
            )

            new_version = (
                dataset_version_manager
                .create_new_version(
                    dataset_id=dataset_id,
                    df_transformed=df_balanced,
                    parent_version=current_version,
                    operation="balance_data",
                    parameters=metadata,
                )
            )

            return new_version.model_dump()

        self.register(
            name="balance_data",
            description=(
                "Balances dataset class distribution using "
                "SMOTE, oversampling, or undersampling and "
                "produces a new version."
            ),
            parameters_schema={
                "type": "object",
                "properties": {
                    "dataset_id": {
                        "type": "string",
                    },
                    "current_version": {
                        "type": "string",
                    },
                    "method": {
                        "type": "string",
                        "enum": [
                            "smote",
                            "oversample",
                            "undersample",
                        ],
                        "default": "smote",
                    },
                    "target_column": {
                        "type": "string",
                    },
                },
                "required": [
                    "dataset_id",
                    "current_version",
                ],
            },
            handler=_balance,
        )

        # =========================================================
        # 9. HANDLE OUTLIERS
        # =========================================================

        def _outliers(
            dataset_id: str,
            current_version: str,
            method: str = "clip",
            target_column: Optional[str] = None,
        ):

            dataset_path = (
                dataset_version_manager
                .get_version_path(
                    dataset_id,
                    current_version,
                )
            )

            df = dataset_loader.load(dataset_path)

            df_out, metadata = (
                outlier_handler.handle_outliers(
                    df,
                    method,
                    target_column,
                )
            )

            new_version = (
                dataset_version_manager
                .create_new_version(
                    dataset_id=dataset_id,
                    df_transformed=df_out,
                    parent_version=current_version,
                    operation="handle_outliers",
                    parameters=metadata,
                )
            )

            return new_version.model_dump()

        self.register(
            name="handle_outliers",
            description=(
                "Handles numerical outliers via clipping "
                "or removal and saves a new dataset version."
            ),
            parameters_schema={
                "type": "object",
                "properties": {
                    "dataset_id": {
                        "type": "string",
                    },
                    "current_version": {
                        "type": "string",
                    },
                    "method": {
                        "type": "string",
                        "enum": [
                            "clip",
                            "remove",
                        ],
                        "default": "clip",
                    },
                    "target_column": {
                        "type": "string",
                    },
                },
                "required": [
                    "dataset_id",
                    "current_version",
                ],
            },
            handler=_outliers,
        )

        # =========================================================
        # 10. SCALE DATA
        # =========================================================

        def _scale(
            dataset_id: str,
            current_version: str,
            method: str = "standard",
            target_column: Optional[str] = None,
        ):

            dataset_path = (
                dataset_version_manager
                .get_version_path(
                    dataset_id,
                    current_version,
                )
            )

            df = dataset_loader.load(dataset_path)

            df_scaled, metadata = (
                data_scaler.scale_data(
                    df,
                    method,
                    target_column,
                )
            )

            new_version = (
                dataset_version_manager
                .create_new_version(
                    dataset_id=dataset_id,
                    df_transformed=df_scaled,
                    parent_version=current_version,
                    operation="scale_data",
                    parameters=metadata,
                )
            )

            return new_version.model_dump()

        self.register(
            name="scale_data",
            description=(
                "Scales numerical features using Standard "
                "or MinMax scaling and saves a new version."
            ),
            parameters_schema={
                "type": "object",
                "properties": {
                    "dataset_id": {
                        "type": "string",
                    },
                    "current_version": {
                        "type": "string",
                    },
                    "method": {
                        "type": "string",
                        "enum": [
                            "standard",
                            "minmax",
                        ],
                        "default": "standard",
                    },
                    "target_column": {
                        "type": "string",
                    },
                },
                "required": [
                    "dataset_id",
                    "current_version",
                ],
            },
            handler=_scale,
        )

        # =========================================================
        # 11. GAUSSIAN NOISE
        # =========================================================

        def _noise(
            dataset_id: str,
            current_version: str,
            noise_level: float = 0.10,
            target_column: Optional[str] = None,
        ):

            dataset_path = (
                dataset_version_manager
                .get_version_path(
                    dataset_id,
                    current_version,
                )
            )

            df = dataset_loader.load(dataset_path)

            df_noise, metadata = (
                data_corruption.add_gaussian_noise(
                    df,
                    noise_level,
                    target_column,
                )
            )

            new_version = (
                dataset_version_manager
                .create_new_version(
                    dataset_id=dataset_id,
                    df_transformed=df_noise,
                    parent_version=current_version,
                    operation="add_gaussian_noise",
                    parameters=metadata,
                )
            )

            return new_version.model_dump()

        self.register(
            name="add_gaussian_noise",
            description=(
                "Injects Gaussian noise into dataset features "
                "for robustness benchmarking."
            ),
            parameters_schema={
                "type": "object",
                "properties": {
                    "dataset_id": {
                        "type": "string",
                    },
                    "current_version": {
                        "type": "string",
                    },
                    "noise_level": {
                        "type": "number",
                        "default": 0.10,
                    },
                    "target_column": {
                        "type": "string",
                    },
                },
                "required": [
                    "dataset_id",
                    "current_version",
                ],
            },
            handler=_noise,
        )

        # =========================================================
        # 12. COMPARE RESULTS
        # =========================================================

        self.register(
            name="compare_results",
            description=(
                "Calculates objective before-vs-after "
                "comparison between baseline and experiment metrics."
            ),
            parameters_schema={
                "type": "object",
                "properties": {
                    "baseline_version": {
                        "type": "string",
                    },
                    "experiment_version": {
                        "type": "string",
                    },
                    "baseline_metrics": {
                        "type": "object",
                    },
                    "experiment_metrics": {
                        "type": "object",
                    },
                },
                "required": [
                    "baseline_version",
                    "experiment_version",
                    "baseline_metrics",
                    "experiment_metrics",
                ],
            },
            handler=lambda
                baseline_version,
                experiment_version,
                baseline_metrics,
                experiment_metrics:
                    evaluation_comparator
                    .compare(
                        baseline_version,
                        experiment_version,
                        baseline_metrics,
                        experiment_metrics,
                    )
                    .model_dump(),
        )

        # =========================================================
        # 13. SEARCH PROJECT HISTORY
        # =========================================================

        self.register(
            name="search_project_history",
            description=(
                "Queries ChromaDB project memory for "
                "previous experiments and observations."
            ),
            parameters_schema={
                "type": "object",
                "properties": {
                    "project_id": {
                        "type": "string",
                    },
                    "query": {
                        "type": "string",
                    },
                },
                "required": [
                    "project_id",
                    "query",
                ],
            },
            handler=lambda project_id, query:
                project_memory.query_history(
                    project_id,
                    query,
                ),
        )

        # =========================================================
        # 14. SAVE EXPERIMENT
        # =========================================================

        def _save_exp(
            project_id: str,
            model_id: str,
            operation: str,
            parameters: Optional[Dict[str, Any]] = None,
            input_dataset_version: str = "",
            output_dataset_version: str = "",
            metrics_before: Optional[Dict[str, Any]] = None,
            metrics_after: Optional[Dict[str, Any]] = None,
            agent_reasoning: str = "",
        ):

            parameters = parameters or {}
            metrics_before = metrics_before or {}
            metrics_after = metrics_after or {}

            if not input_dataset_version:
                raise ValueError(
                    "input_dataset_version is required"
                )

            if not output_dataset_version:
                raise ValueError(
                    "output_dataset_version is required"
                )

            # -----------------------------------------------------
            # Deterministic comparison
            # -----------------------------------------------------

            comparison = (
                evaluation_comparator
                .compare(
                    input_dataset_version,
                    output_dataset_version,
                    metrics_before,
                    metrics_after,
                )
                .model_dump()
            )

            # -----------------------------------------------------
            # Unique experiment ID
            # -----------------------------------------------------

            exp_id = (
                f"exp_"
                f"{datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')}"
            )

            # -----------------------------------------------------
            # SQLite persistence
            # -----------------------------------------------------

            with db._get_connection() as conn:

                conn.execute(
                    """
                    INSERT INTO experiments (
                        id,
                        project_id,
                        model_id,
                        input_dataset_version,
                        output_dataset_version,
                        operation,
                        parameters,
                        metrics_before,
                        metrics_after,
                        comparison,
                        agent_reasoning,
                        created_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        exp_id,
                        project_id,
                        model_id,
                        input_dataset_version,
                        output_dataset_version,
                        operation,
                        json.dumps(parameters),
                        json.dumps(metrics_before),
                        json.dumps(metrics_after),
                        json.dumps(comparison),
                        agent_reasoning,
                        datetime.utcnow().isoformat(),
                    ),
                )

                conn.commit()

            # -----------------------------------------------------
            # ChromaDB persistence
            # -----------------------------------------------------

            summary_doc = (
                f"Experiment {exp_id} on {project_id}: "
                f"Operation '{operation}' on "
                f"{input_dataset_version} -> "
                f"{output_dataset_version}.\n"
                f"Reasoning: {agent_reasoning}\n"
                f"Changes: "
                f"{', '.join(comparison.get('summary_of_changes', []))}"
            )

            project_memory.store_experiment(
                project_id=project_id,
                experiment_id=exp_id,
                summary_text=summary_doc,
                metadata={
                    "operation": operation,
                    "is_improved": comparison.get(
                        "is_improved",
                        False,
                    ),
                },
            )

            return {
                "experiment_id": exp_id,
                "project_id": project_id,
                "model_id": model_id,
                "operation": operation,
                "input_dataset_version": input_dataset_version,
                "output_dataset_version": output_dataset_version,
                "comparison": comparison,
                "agent_reasoning": agent_reasoning,
            }

        self.register(
            name="save_experiment",
            description=(
                "Persists experiment record, metric deltas, "
                "comparison, and agent reasoning to SQLite "
                "and ChromaDB."
            ),
            parameters_schema={
                "type": "object",
                "properties": {
                    "project_id": {
                        "type": "string",
                    },
                    "model_id": {
                        "type": "string",
                    },
                    "operation": {
                        "type": "string",
                    },
                    "parameters": {
                        "type": "object",
                    },
                    "input_dataset_version": {
                        "type": "string",
                    },
                    "output_dataset_version": {
                        "type": "string",
                    },
                    "metrics_before": {
                        "type": "object",
                    },
                    "metrics_after": {
                        "type": "object",
                    },
                    "agent_reasoning": {
                        "type": "string",
                    },
                },
                "required": [
                    "project_id",
                    "model_id",
                    "operation",
                    "input_dataset_version",
                    "output_dataset_version",
                    "metrics_before",
                    "metrics_after",
                ],
            },
            handler=_save_exp,
        )

        # =========================================================
        # 15. GENERATE REPORT
        # =========================================================
        #
        # IMPORTANT:
        # This tool does NOT generate the report itself.
        #
        # agent.py calls report_generator.generate_and_save(state)
        # after this tool succeeds.
        #
        # Keeping this tool lightweight prevents the report from
        # being generated twice.
        # =========================================================

        def _generate_report(
            project_id: str,
            title: str = "ML Evaluation Report",
        ):

            if not project_id:
                raise ValueError(
                    "project_id is required"
                )

            return {
                "status": "report_ready",
                "project_id": project_id,
                "title": title,
            }

        self.register(
            name="generate_report",
            description=(
                "Authorizes the final report generation step. "
                "The AgentRunner generates the actual report "
                "from the complete AgentState."
            ),
            parameters_schema={
                "type": "object",
                "properties": {
                    "project_id": {
                        "type": "string",
                    },
                    "title": {
                        "type": "string",
                        "default": "ML Evaluation Report",
                    },
                },
                "required": [
                    "project_id",
                ],
            },
            handler=_generate_report,
        )


tool_registry = ToolRegistry()
