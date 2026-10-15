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

try:
    from app.models.trainer import adversarial_retrainer
except ImportError:
    adversarial_retrainer = None


def _build_security_combinations(
    steps=None,
    attack_sequence=None,
    defense_sequence=None,
    fallback_attack=None,
    fallback_defense=None,
    max_steps: int = 5,
) -> List[Dict[str, Any]]:
    """Normalize the multi-step security arguments into ordered combinations.

    Returns an empty list when the caller asked for the single-pair path, so
    the existing behaviour of ``attack_defended_model`` is untouched.

    Zips attacks and defenses in the order named, cycling the shorter list, and
    caps the assembled list at ``max_steps``. N is an upper bound: combinations
    are never invented to reach it.
    """
    explicit: List[Dict[str, Any]] = []
    for step in steps or []:
        if isinstance(step, dict):
            explicit.append({
                "attack_type": step.get("attack_type") or step.get("attack_name"),
                "defense_type": step.get("defense_type") or step.get("defense_name"),
            })
        elif isinstance(step, str):
            explicit.append({"attack_type": step, "defense_type": None})

    if explicit:
        return explicit[:max_steps]

    attacks = [a for a in (attack_sequence or []) if a]
    defenses = [d for d in (defense_sequence or []) if d]
    if not attacks and not defenses:
        return []

    combinations: List[Dict[str, Any]] = []
    for index in range(min(max(len(attacks), len(defenses)), max_steps)):
        combinations.append({
            "attack_type": attacks[index % len(attacks)] if attacks else None,
            "defense_type": defenses[index % len(defenses)] if defenses else None,
        })
    return combinations


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
            params = params or {}
            result = handler(**params)

            if isinstance(result, dict) and str(result.get("status", "")).lower() in {
                "error",
                "failed",
            }:
                nested_message = result.get("error") or result.get("message") or "Unknown tool error"
                return {
                    "status": "error",
                    "message": f"Error executing tool '{name}': {nested_message}",
                    "result": result,
                }

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

        def _inspect_model_handler(model_path: str, model_id: str = "model_001"):
            info = model_inspector.inspect(model_path, model_id)
            if hasattr(info, "model_dump"):
                info_dict = info.model_dump()
            elif isinstance(info, dict):
                info_dict = info
            else:
                info_dict = dict(info)

            try:
                with db._get_connection() as conn:
                    conn.execute(
                        """
                        UPDATE models 
                        SET architecture = ?, output_shape = ?, parameter_count = ?
                        WHERE id = ?
                        """,
                        (
                            str(info_dict.get("model_class", "")),
                            str(info_dict.get("output_shape", "")),
                            info_dict.get("parameter_count", 0),
                            model_id
                        )
                    )
                    conn.commit()
            except Exception:
                pass

            return info_dict

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
            handler=_inspect_model_handler,
        )

        # =========================================================
        # 2. INSPECT DATASET
        # =========================================================

        def _inspect_dataset_handler(
            dataset_path: str,
            dataset_id: str = "dataset_001",
            version: str = "v001",
            target_column: Optional[str] = None,
            **kwargs
        ):
            info = dataset_inspector.inspect(
                dataset_path,
                dataset_id,
                version,
                target_col=target_column,
            )
            if hasattr(info, "model_dump"):
                info_dict = info.model_dump()
            elif isinstance(info, dict):
                info_dict = info
            else:
                info_dict = dict(info)

            try:
                row_c = info_dict.get("row_count", 0)
                col_c = info_dict.get("column_count", 0)
                with db._get_connection() as conn:
                    conn.execute(
                        """
                        UPDATE dataset_versions 
                        SET row_count = ?, column_count = ?
                        WHERE dataset_id = ? AND version = ?
                        """,
                        (row_c, col_c, dataset_id, version)
                    )
                    conn.commit()
            except Exception:
                pass

            return info_dict

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
                    "target_column": {
                        "type": "string",
                        "description": "Optional target column name",
                    },
                },
                "required": ["dataset_path"],
            },
            handler=_inspect_dataset_handler,
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
                    "model_path": {"type": "string"},
                    "dataset_path": {"type": "string"},
                    "model_id": {"type": "string", "default": "model_001"},
                    "dataset_id": {"type": "string", "default": "dataset_001"},
                    "version": {"type": "string", "default": "v001"},
                    "target_column": {
                        "type": "string",
                        "description": "Optional target column name",
                    },
                },
                "required": ["model_path", "dataset_path"],
            },
            handler=lambda
                model_path,
                dataset_path,
                model_id="model_001",
                dataset_id="dataset_001",
                version="v001",
                target_column=None:
                    compatibility_validator
                    .validate_compatibility(
                        model_path=model_path,
                        dataset_path=dataset_path,
                        model_id=model_id,
                        dataset_id=dataset_id,
                        version=version,
                        target_column=target_column,
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
                    "model_path": {"type": "string"},
                    "dataset_path": {"type": "string"},
                    "model_id": {"type": "string"},
                    "dataset_id": {"type": "string"},
                    "version": {"type": "string", "default": "v001"},
                    "target_column": {"type": "string"},
                    "task_type": {
                        "type": "string",
                        "default": "classification",
                        "enum": ["classification", "regression"],
                    },
                },
                "required": ["model_path", "dataset_path"],
            },
            handler=lambda
                model_path,
                dataset_path,
                target_column=None,
                task_type="classification",
                model_id=None,
                dataset_id=None,
                version=None:
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
                    "dataset_path": {"type": "string"},
                    "dataset_id": {"type": "string"},
                    "version": {"type": "string"},
                    "target_column": {"type": "string"},
                },
                "required": ["dataset_path"],
            },
            handler=lambda
                dataset_path,
                dataset_id=None,
                version=None,
                target_column=None:
                    data_quality_evaluator
                    .evaluate(dataset_path, target_column)
                    .model_dump(),
        )

        # =========================================================
        # 6. ROBUSTNESS (Gaussian Noise Benchmark)
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
                    "model_path": {"type": "string"},
                    "dataset_path": {"type": "string"},
                    "model_id": {"type": "string"},
                    "dataset_id": {"type": "string"},
                    "version": {"type": "string", "default": "v001"},
                    "target_column": {"type": "string"},
                    "noise_level": {"type": "number", "default": 0.10},
                },
                "required": ["model_path", "dataset_path"],
            },
            handler=lambda
                model_path,
                dataset_path,
                target_column=None,
                noise_level=0.10,
                model_id=None,
                dataset_id=None,
                version=None:
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
        # 6b. BOUNDARY ATTACK (Black-box Evasion)
        # =========================================================

        self.register(
            name="boundary_attack",
            description="Executes an adversarial Boundary Attack on the model and dataset (black-box).",
            parameters_schema={
                "type": "object",
                "properties": {
                    "model_path": {"type": "string"},
                    "dataset_path": {"type": "string"},
                    "model_id": {"type": "string"},
                    "dataset_id": {"type": "string"},
                    "version": {"type": "string", "default": "v001"},
                    "target_column": {"type": "string"},
                },
                "required": ["model_path", "dataset_path"],
            },
            handler=lambda
                model_path=None,
                dataset_path=None,
                model=None,
                dataset=None,
                target_column=None,
                **kwargs:
                    robustness_evaluator
                    .run_boundary_attack(
                        model=model,
                        dataset=dataset,
                        model_path=model_path,
                        dataset_path=dataset_path,
                        target_column=target_column,
                        **kwargs
                    ),
        )

        # =========================================================
        # 6c. HOPSKIPJUMP ATTACK (Black-box Evasion)
        # =========================================================

        self.register(
            name="hopskipjump",
            description="Executes an adversarial HopSkipJump attack on the model and dataset (black-box).",
            parameters_schema={
                "type": "object",
                "properties": {
                    "model_path": {"type": "string"},
                    "dataset_path": {"type": "string"},
                    "model_id": {"type": "string"},
                    "dataset_id": {"type": "string"},
                    "version": {"type": "string", "default": "v001"},
                    "target_column": {"type": "string"},
                },
                "required": ["model_path", "dataset_path"],
            },
            handler=lambda
                model_path=None,
                dataset_path=None,
                model=None,
                dataset=None,
                target_column=None,
                **kwargs:
                    robustness_evaluator
                    .run_hopskipjump_attack(
                        model=model,
                        dataset=dataset,
                        model_path=model_path,
                        dataset_path=dataset_path,
                        target_column=target_column,
                        **kwargs
                    ),
        )

        # =========================================================
        # 6d. ZOO ATTACK (Black-box Evasion)
        # =========================================================

        def _zoo_handler(model_path=None, dataset_path=None, target_column=None, nb_parallel=1, **kwargs):
            kwargs["nb_parallel"] = nb_parallel
            return robustness_evaluator.run_evasion_attack(
                "zoo",
                model_path=model_path,
                dataset_path=dataset_path,
                target_column=target_column,
                **kwargs
            )

        self.register(
            name="zoo",
            description="Runs ART ZOO Zeroth Order Optimization evasion attack (black-box).",
            parameters_schema={
                "type": "object",
                "properties": {
                    "model_path": {"type": "string"},
                    "dataset_path": {"type": "string"},
                    "target_column": {"type": "string"},
                    "nb_parallel": {
                        "type": "integer",
                        "default": 1,
                        "description": "Number of coordinate updates per step (clamped to 1 for small tabular datasets)",
                    },
                },
                "required": ["model_path", "dataset_path"],
            },
            handler=_zoo_handler,
        )

        # =========================================================
        # 6e. FGSM ATTACK (Gradient-based, PyTorch)
        # =========================================================

        self.register(
            name="fgsm",
            description="Runs ART Fast Gradient Sign Method (FGSM) evasion attack (PyTorch models).",
            parameters_schema={
                "type": "object",
                "properties": {
                    "model_path": {"type": "string"},
                    "dataset_path": {"type": "string"},
                    "target_column": {"type": "string"},
                    "eps": {"type": "number", "default": 0.1},
                    "batch_size": {"type": "integer", "default": 32},
                },
                "required": ["model_path"],
            },
            handler=lambda model_path=None, dataset_path=None, target_column=None, **kwargs:
                robustness_evaluator.run_evasion_attack(
                    "fgsm", model_path=model_path, dataset_path=dataset_path,
                    target_column=target_column, **kwargs
                ),
        )

        # =========================================================
        # 6f. PGD ATTACK (Gradient-based, PyTorch)
        # =========================================================

        self.register(
            name="pgd",
            description="Runs ART Projected Gradient Descent (PGD) evasion attack (PyTorch models).",
            parameters_schema={
                "type": "object",
                "properties": {
                    "model_path": {"type": "string"},
                    "dataset_path": {"type": "string"},
                    "target_column": {"type": "string"},
                    "eps": {"type": "number", "default": 0.1},
                    "eps_step": {"type": "number", "default": 0.01},
                    "max_iter": {"type": "integer", "default": 10},
                    "batch_size": {"type": "integer", "default": 32},
                },
                "required": ["model_path"],
            },
            handler=lambda model_path=None, dataset_path=None, target_column=None, **kwargs:
                robustness_evaluator.run_evasion_attack(
                    "pgd", model_path=model_path, dataset_path=dataset_path,
                    target_column=target_column, **kwargs
                ),
        )

        # =========================================================
        # 6g. C&W ATTACK (Gradient-based, PyTorch)
        # =========================================================

        self.register(
            name="cw",
            description="Runs ART Carlini & Wagner (C&W L2) evasion attack (PyTorch models).",
            parameters_schema={
                "type": "object",
                "properties": {
                    "model_path": {"type": "string"},
                    "dataset_path": {"type": "string"},
                    "target_column": {"type": "string"},
                },
                "required": ["model_path", "dataset_path"],
            },
            handler=lambda model_path=None, dataset_path=None, target_column=None, **kwargs:
                robustness_evaluator.run_evasion_attack(
                    "cw", model_path=model_path, dataset_path=dataset_path,
                    target_column=target_column, **kwargs
                ),
        )

        # =========================================================
        # 6h. GAUSSIAN AUGMENTATION (Defense)
        # =========================================================

        self.register(
            name="gaussian_augmentation",
            description="Applies ART Gaussian Augmentation preprocessor defense and evaluates performance.",
            parameters_schema={
                "type": "object",
                "properties": {
                    "model_path": {"type": "string"},
                    "dataset_path": {"type": "string"},
                    "target_column": {"type": "string"},
                    "sigma": {"type": "number", "default": 1.0},
                },
                "required": ["model_path", "dataset_path"],
            },
            handler=lambda model_path=None, dataset_path=None, target_column=None, **kwargs:
                robustness_evaluator.gaussian_augmentation_defense(
                    model_path=model_path, dataset_path=dataset_path,
                    target_column=target_column, **kwargs
                ),
        )

        # =========================================================
        # 6i. SPATIAL SMOOTHING (Defense)
        # =========================================================

        self.register(
            name="spatial_smoothing",
            description="Applies ART Spatial Smoothing preprocessor defense and evaluates performance.",
            parameters_schema={
                "type": "object",
                "properties": {
                    "model_path": {"type": "string"},
                    "dataset_path": {"type": "string"},
                    "target_column": {"type": "string"},
                    "window_size": {"type": "integer", "default": 3},
                },
                "required": ["model_path", "dataset_path"],
            },
            handler=lambda model_path=None, dataset_path=None, target_column=None, **kwargs:
                robustness_evaluator.run_defense(
                    "spatial_smoothing", model_path=model_path, dataset_path=dataset_path,
                    target_column=target_column, **kwargs
                ),
        )

        # =========================================================
        # 6j. FEATURE SQUEEZING (Defense)
        # =========================================================

        self.register(
            name="feature_squeezing",
            description="Applies ART Feature Squeezing preprocessor defense and evaluates performance.",
            parameters_schema={
                "type": "object",
                "properties": {
                    "model_path": {"type": "string"},
                    "dataset_path": {"type": "string"},
                    "target_column": {"type": "string"},
                    "bit_depth": {"type": "integer", "default": 4},
                },
                "required": ["model_path", "dataset_path"],
            },
            handler=lambda model_path=None, dataset_path=None, target_column=None, **kwargs:
                robustness_evaluator.run_defense(
                    "feature_squeezing", model_path=model_path, dataset_path=dataset_path,
                    target_column=target_column, **kwargs
                ),
        )

        # =========================================================
        # 6k. CLIP VALUES (Defense - Preprocessor)
        # =========================================================

        self.register(
            name="clip_values",
            description="Applies ART Clip Values preprocessor defense and evaluates performance.",
            parameters_schema={
                "type": "object",
                "properties": {
                    "model_path": {"type": "string"},
                    "dataset_path": {"type": "string"},
                    "target_column": {"type": "string"},
                },
                "required": ["model_path", "dataset_path"],
            },
            handler=lambda model_path=None, dataset_path=None, target_column=None, **kwargs:
                robustness_evaluator.run_defense(
                    "clip_values", model_path=model_path, dataset_path=dataset_path,
                    target_column=target_column, **kwargs
                ),
        )

        # =========================================================
        # 6l. REVERSE SIGMOID (Defense - Postprocessor)
        # =========================================================

        self.register(
            name="reverse_sigmoid",
            description="Applies ART Reverse Sigmoid postprocessor defense and evaluates performance.",
            parameters_schema={
                "type": "object",
                "properties": {
                    "model_path": {"type": "string"},
                    "dataset_path": {"type": "string"},
                    "target_column": {"type": "string"},
                    "beta": {"type": "number", "default": 1.0},
                    "gamma": {"type": "number", "default": 0.1},
                },
                "required": ["model_path", "dataset_path"],
            },
            handler=lambda model_path=None, dataset_path=None, target_column=None, **kwargs:
                robustness_evaluator.run_defense(
                    "reverse_sigmoid", model_path=model_path, dataset_path=dataset_path,
                    target_column=target_column, **kwargs
                ),
        )

        # =========================================================
        # 6m. CLASS LABELS (Defense - Postprocessor)
        # =========================================================

        self.register(
            name="class_labels",
            description="Applies ART Class Labels postprocessor defense (argmax output masking) and evaluates performance.",
            parameters_schema={
                "type": "object",
                "properties": {
                    "model_path": {"type": "string"},
                    "dataset_path": {"type": "string"},
                    "target_column": {"type": "string"},
                },
                "required": ["model_path", "dataset_path"],
            },
            handler=lambda model_path=None, dataset_path=None, target_column=None, **kwargs:
                robustness_evaluator.run_defense(
                    "class_labels", model_path=model_path, dataset_path=dataset_path,
                    target_column=target_column, **kwargs
                ),
        )

        # =========================================================
        # 6n. HIGH CONFIDENCE MASKING (Defense - Postprocessor)
        # =========================================================

        self.register(
            name="high_confidence_masking",
            description="Applies ART High Confidence Masking postprocessor defense and evaluates performance.",
            parameters_schema={
                "type": "object",
                "properties": {
                    "model_path": {"type": "string"},
                    "dataset_path": {"type": "string"},
                    "target_column": {"type": "string"},
                    "cutoff": {"type": "number", "default": 0.25},
                },
                "required": ["model_path", "dataset_path"],
            },
            handler=lambda model_path=None, dataset_path=None, target_column=None, **kwargs:
                robustness_evaluator.run_defense(
                    "high_confidence_masking", model_path=model_path, dataset_path=dataset_path,
                    target_column=target_column, **kwargs
                ),
        )

        # =========================================================
        # 6o. ATTACK DEFENDED MODEL
        # =========================================================

        def _attack_defended_model_handler(
            model_path=None,
            dataset_path=None,
            target_column=None,
            attack_name="fgsm",
            defense_name="feature_squeezing",
            steps=None,
            attack_sequence=None,
            defense_sequence=None,
            max_steps=5,
            requested_steps=None,
            **kwargs
        ):
            # Sanitize parameters: FGSM is single-step and does not support iterations
            if attack_name and attack_name.lower() == "fgsm":
                kwargs.pop("max_iter", None)
                kwargs.pop("eps_step", None)

            combinations = _build_security_combinations(
                steps=steps,
                attack_sequence=attack_sequence,
                defense_sequence=defense_sequence,
                fallback_attack=attack_name,
                fallback_defense=defense_name,
                max_steps=max_steps,
            )

            if combinations:
                return robustness_evaluator.run_attack_defense_sequence(
                    combinations=combinations,
                    model_path=model_path,
                    dataset_path=dataset_path,
                    target_column=target_column,
                    max_steps=max_steps,
                    requested_steps=requested_steps,
                    **kwargs
                )

            return robustness_evaluator.run_attack_on_defended_model(
                attack_name=attack_name,
                defense_name=defense_name,
                model_path=model_path,
                dataset_path=dataset_path,
                target_column=target_column,
                **kwargs
            )

        self.register(
            name="attack_defended_model",
            description=(
                "Executes an evasion attack against an in-memory defended "
                "classifier wrapper. Pass 'steps' with up to 5 attack/defense "
                "combinations to compare several of them against the same model "
                "in one call."
            ),
            parameters_schema={
                "type": "object",
                "properties": {
                    "model_path": {"type": "string"},
                    "dataset_path": {"type": "string"},
                    "target_column": {"type": "string"},
                    "attack_name": {"type": "string", "default": "fgsm"},
                    "defense_name": {"type": "string", "default": "feature_squeezing"},
                    "steps": {
                        "type": "array",
                        "description": (
                            "Optional list of attack/defense combinations to "
                            "compare, e.g. [{'attack_type': 'fgsm', "
                            "'defense_type': 'feature_squeezing'}]. Capped at 5."
                        ),
                        "items": {
                            "type": "object",
                            "properties": {
                                "attack_type": {"type": "string"},
                                "defense_type": {"type": "string"},
                            },
                        },
                    },
                    "attack_sequence": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of attacks to compare.",
                    },
                    "defense_sequence": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of defenses to compare.",
                    },
                    "max_steps": {"type": "integer", "default": 5, "maximum": 5},
                },
                "required": ["model_path", "dataset_path"],
            },
            handler=_attack_defended_model_handler,
        )
        # =========================================================
        # 6p. HARDEN MODEL
        # =========================================================

        self.register(
            name="harden_model",
            description="Retrains the model adversarially and saves a new hardened checkpoint asset.",
            parameters_schema={
                "type": "object",
                "properties": {
                    "model_path": {"type": "string"},
                    "dataset_path": {"type": "string"},
                    "target_column": {"type": "string"},
                    "method": {"type": "string", "default": "pgd"},
                },
                "required": ["model_path", "dataset_path"],
            },
            handler=lambda model_path=None, dataset_path=None, target_column=None, method="pgd", **kwargs:
                (
                    adversarial_retrainer.harden_and_save(
                        model_path=model_path,
                        dataset_path=dataset_path,
                        target_column=target_column,
                        method=method,
                        **kwargs
                    )
                    if adversarial_retrainer
                    else {"status": "error", "message": "AdversarialRetrainer is not loaded"}
                ),
        )

        # =========================================================
        # 6q. CUMULATIVE HARDENING LOOP
        # =========================================================

        def _harden_defense_loop_handler(
            model_path=None,
            dataset_path=None,
            target_column=None,
            attack_name="fgsm",
            defense_name="feature_squeezing",
            rounds=None,
            attack_sequence=None,
            defense_sequence=None,
            retrain_method="pgd",
            max_rounds=5,
            requested_rounds=None,
            **kwargs
        ):
            combinations = _build_security_combinations(
                steps=rounds or kwargs.pop("steps", None),
                attack_sequence=attack_sequence,
                defense_sequence=defense_sequence,
                fallback_attack=attack_name,
                fallback_defense=defense_name,
                max_steps=max_rounds,
            )
            if not combinations:
                # Default to a 2-round loop on the single named pair so the
                # workflow always shows round-over-round movement.
                combinations = [
                    {"attack_type": attack_name, "defense_type": defense_name}
                    for _ in range(min(2, max_rounds))
                ]

            if not adversarial_retrainer:
                return {"status": "error", "message": "AdversarialRetrainer is not loaded"}

            return robustness_evaluator.run_hardening_loop(
                combinations=combinations,
                model_path=model_path,
                dataset_path=dataset_path,
                target_column=target_column,
                retrain_method=retrain_method,
                max_rounds=max_rounds,
                requested_rounds=requested_rounds,
                **kwargs
            )

        self.register(
            name="harden_defense_loop",
            description=(
                "Runs up to 5 cumulative hardening rounds: each round attacks "
                "the current model, adversarially retrains it, and feeds the "
                "retrained model into the next round. Registers only the final "
                "hardened model."
            ),
            parameters_schema={
                "type": "object",
                "properties": {
                    "model_path": {"type": "string"},
                    "dataset_path": {"type": "string"},
                    "target_column": {"type": "string"},
                    "attack_name": {"type": "string", "default": "fgsm"},
                    "defense_name": {"type": "string", "default": "feature_squeezing"},
                    "retrain_method": {"type": "string", "default": "pgd"},
                    "rounds": {
                        "type": "array",
                        "description": "Optional list of per-round combinations. Capped at 5.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "attack_type": {"type": "string"},
                                "defense_type": {"type": "string"},
                            },
                        },
                    },
                    "attack_sequence": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "defense_sequence": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "max_rounds": {"type": "integer", "default": 5, "maximum": 5},
                },
                "required": ["model_path", "dataset_path"],
            },
            handler=_harden_defense_loop_handler,
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
            dataset_path = dataset_version_manager.get_version_path(
                dataset_id, current_version
            )
            df = dataset_loader.load(dataset_path)
            df_clean, metadata = data_cleaner.handle_missing_values(
                df, strategy, target_column
            )
            new_version = dataset_version_manager.create_new_version(
                dataset_id=dataset_id,
                df_transformed=df_clean,
                parent_version=current_version,
                operation="handle_missing_values",
                parameters=metadata,
            )
            return new_version.model_dump()

        self.register(
            name="handle_missing_values",
            description="Imputes missing values and creates a new immutable dataset version.",
            parameters_schema={
                "type": "object",
                "properties": {
                    "dataset_id": {"type": "string"},
                    "current_version": {"type": "string"},
                    "strategy": {
                        "type": "string",
                        "enum": ["median", "mean", "mode"],
                        "default": "median",
                    },
                    "target_column": {"type": "string"},
                },
                "required": ["dataset_id", "current_version"],
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
            dataset_path = dataset_version_manager.get_version_path(
                dataset_id, current_version
            )
            df = dataset_loader.load(dataset_path)
            df_balanced, metadata = data_balancer.balance_data(
                df, method, target_column
            )
            new_version = dataset_version_manager.create_new_version(
                dataset_id=dataset_id,
                df_transformed=df_balanced,
                parent_version=current_version,
                operation="balance_data",
                parameters=metadata,
            )
            return new_version.model_dump()

        self.register(
            name="balance_data",
            description="Balances dataset class distribution using SMOTE or sampling.",
            parameters_schema={
                "type": "object",
                "properties": {
                    "dataset_id": {"type": "string"},
                    "current_version": {"type": "string"},
                    "method": {
                        "type": "string",
                        "enum": ["smote", "oversample", "undersample"],
                        "default": "smote",
                    },
                    "target_column": {"type": "string"},
                },
                "required": ["dataset_id", "current_version"],
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
            dataset_path = dataset_version_manager.get_version_path(
                dataset_id, current_version
            )
            df = dataset_loader.load(dataset_path)
            df_out, metadata = outlier_handler.handle_outliers(
                df, method, target_column
            )
            new_version = dataset_version_manager.create_new_version(
                dataset_id=dataset_id,
                df_transformed=df_out,
                parent_version=current_version,
                operation="handle_outliers",
                parameters=metadata,
            )
            return new_version.model_dump()

        self.register(
            name="handle_outliers",
            description="Handles numerical outliers via clipping or removal.",
            parameters_schema={
                "type": "object",
                "properties": {
                    "dataset_id": {"type": "string"},
                    "current_version": {"type": "string"},
                    "method": {
                        "type": "string",
                        "enum": ["clip", "remove"],
                        "default": "clip",
                    },
                    "target_column": {"type": "string"},
                },
                "required": ["dataset_id", "current_version"],
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
            dataset_path = dataset_version_manager.get_version_path(
                dataset_id, current_version
            )
            df = dataset_loader.load(dataset_path)
            df_scaled, metadata = data_scaler.scale_data(
                df, method, target_column
            )
            new_version = dataset_version_manager.create_new_version(
                dataset_id=dataset_id,
                df_transformed=df_scaled,
                parent_version=current_version,
                operation="scale_data",
                parameters=metadata,
            )
            return new_version.model_dump()

        self.register(
            name="scale_data",
            description="Scales numerical features using Standard or MinMax scaling.",
            parameters_schema={
                "type": "object",
                "properties": {
                    "dataset_id": {"type": "string"},
                    "current_version": {"type": "string"},
                    "method": {
                        "type": "string",
                        "enum": ["standard", "minmax"],
                        "default": "standard",
                    },
                    "target_column": {"type": "string"},
                },
                "required": ["dataset_id", "current_version"],
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
            dataset_path = dataset_version_manager.get_version_path(
                dataset_id, current_version
            )
            df = dataset_loader.load(dataset_path)
            df_noise, metadata = data_corruption.add_gaussian_noise(
                df, noise_level, target_column
            )
            new_version = dataset_version_manager.create_new_version(
                dataset_id=dataset_id,
                df_transformed=df_noise,
                parent_version=current_version,
                operation="add_gaussian_noise",
                parameters=metadata,
            )
            return new_version.model_dump()

        self.register(
            name="add_gaussian_noise",
            description="Injects Gaussian noise for robustness benchmarking.",
            parameters_schema={
                "type": "object",
                "properties": {
                    "dataset_id": {"type": "string"},
                    "current_version": {"type": "string"},
                    "noise_level": {"type": "number", "default": 0.10},
                    "target_column": {"type": "string"},
                },
                "required": ["dataset_id", "current_version"],
            },
            handler=_noise,
        )

        # =========================================================
        # 11b. INJECT MISSING VALUES
        # =========================================================

        def _inject_missing(
            dataset_id: str,
            current_version: str,
            missing_ratio: float = 0.10,
            target_column: Optional[str] = None,
        ):
            dataset_path = dataset_version_manager.get_version_path(
                dataset_id, current_version
            )
            df = dataset_loader.load(dataset_path)
            df_corrupt, metadata = data_corruption.inject_missing_values(
                df, missing_ratio, target_column
            )
            new_version = dataset_version_manager.create_new_version(
                dataset_id=dataset_id,
                df_transformed=df_corrupt,
                parent_version=current_version,
                operation="inject_missing_values",
                parameters=metadata,
            )
            return new_version.model_dump()

        self.register(
            name="inject_missing_values",
            description="Injects missing values into the dataset for stress testing.",
            parameters_schema={
                "type": "object",
                "properties": {
                    "dataset_id": {"type": "string"},
                    "current_version": {"type": "string"},
                    "missing_ratio": {"type": "number", "default": 0.10},
                    "target_column": {"type": "string"},
                },
                "required": ["dataset_id", "current_version"],
            },
            handler=_inject_missing,
        )

        # =========================================================
        # 11c. INJECT LABEL NOISE
        # =========================================================

        def _inject_label_noise(
            dataset_id: str,
            current_version: str,
            noise_ratio: float = 0.10,
            target_column: Optional[str] = None,
        ):
            dataset_path = dataset_version_manager.get_version_path(
                dataset_id, current_version
            )
            df = dataset_loader.load(dataset_path)
            df_corrupt, metadata = data_corruption.inject_label_noise(
                df, noise_ratio, target_column
            )
            new_version = dataset_version_manager.create_new_version(
                dataset_id=dataset_id,
                df_transformed=df_corrupt,
                parent_version=current_version,
                operation="inject_label_noise",
                parameters=metadata,
            )
            return new_version.model_dump()

        self.register(
            name="inject_label_noise",
            description="Randomly flips target-column labels for stress testing.",
            parameters_schema={
                "type": "object",
                "properties": {
                    "dataset_id": {"type": "string"},
                    "current_version": {"type": "string"},
                    "noise_ratio": {"type": "number", "default": 0.10},
                    "target_column": {"type": "string"},
                },
                "required": ["dataset_id", "current_version"],
            },
            handler=_inject_label_noise,
        )

        # =========================================================
        # 12. COMPARE RESULTS
        # =========================================================

        self.register(
            name="compare_results",
            description="Calculates objective before-vs-after metric comparisons.",
            parameters_schema={
                "type": "object",
                "properties": {
                    "baseline_version": {"type": "string"},
                    "experiment_version": {"type": "string"},
                    "baseline_metrics": {"type": "object"},
                    "experiment_metrics": {"type": "object"},
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
            description="Queries ChromaDB project memory for previous experiments.",
            parameters_schema={
                "type": "object",
                "properties": {
                    "project_id": {"type": "string"},
                    "query": {"type": "string"},
                },
                "required": ["project_id", "query"],
            },
            handler=lambda project_id, query:
                project_memory.query_history(project_id, query),
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
            series: Optional[Dict[str, Any]] = None,
        ):
            parameters = parameters or {}
            metrics_before = metrics_before or {}
            metrics_after = metrics_after or {}

            if not input_dataset_version:
                raise ValueError("input_dataset_version is required")

            if not output_dataset_version:
                raise ValueError("output_dataset_version is required")

            comparison = evaluation_comparator.compare(
                input_dataset_version,
                output_dataset_version,
                metrics_before,
                metrics_after,
            ).model_dump()

            # Multi-step runs carry one entry per step/round. It rides on the
            # comparison blob, which consumers already read by named key, so the
            # dashboard can chart every step instead of only the last one.
            if series and series.get("items"):
                comparison["series"] = series

            exp_id = f"exp_{datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')}"

            with db._get_connection() as conn:
                conn.execute(
                    """
                    INSERT INTO experiments (
                        id, project_id, model_id, input_dataset_version,
                        output_dataset_version, operation, parameters,
                        metrics_before, metrics_after, comparison,
                        agent_reasoning, created_at
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

            model_name = parameters.get("model_name") or model_id
            dataset_name = parameters.get("dataset_name") or input_dataset_version

            formatted_summary = evaluation_comparator.format_summary_line(
                model_name=model_name,
                dataset_name=dataset_name,
                operation=operation,
                metrics=metrics_after if metrics_after else metrics_before
            )

            summary_doc = (
                f"Experiment {exp_id} on {project_id}:\n"
                f"{formatted_summary}"
            )

            project_memory.store_experiment(
                project_id=project_id,
                experiment_id=exp_id,
                summary_text=summary_doc,
                metadata={
                    "operation": operation,
                    "input_dataset_version": input_dataset_version,
                    "output_dataset_version": output_dataset_version,
                    "metrics_before": metrics_before,
                    "metrics_after": metrics_after,
                    "is_improved": comparison.get("is_improved", False),
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
            description="Persists experiment record and comparison to SQLite and ChromaDB.",
            parameters_schema={
                "type": "object",
                "properties": {
                    "project_id": {"type": "string"},
                    "model_id": {"type": "string"},
                    "operation": {"type": "string"},
                    "parameters": {"type": "object"},
                    "input_dataset_version": {"type": "string"},
                    "output_dataset_version": {"type": "string"},
                    "metrics_before": {"type": "object"},
                    "metrics_after": {"type": "object"},
                    "agent_reasoning": {"type": "string"},
                    "series": {"type": "object"},
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

        def _generate_report(
            project_id: str,
            title: str = "ML Evaluation Report",
        ):
            if not project_id:
                raise ValueError("project_id is required")

            return {
                "status": "report_ready",
                "project_id": project_id,
                "title": title,
            }

        self.register(
            name="generate_report",
            description="Authorizes final report generation from complete AgentState.",
            parameters_schema={
                "type": "object",
                "properties": {
                    "project_id": {"type": "string"},
                    "title": {
                        "type": "string",
                        "default": "ML Evaluation Report",
                    },
                },
                "required": ["project_id"],
            },
            handler=_generate_report,
        )


tool_registry = ToolRegistry()