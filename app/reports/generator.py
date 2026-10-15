import json
import logging
import os
import tempfile
import uuid
from typing import Dict, Any
from datetime import datetime

from app.core.storage import storage_manager
from app.core.database import db
from app.datasets.loader import dataset_loader


class ReportGenerator:
    """Generates a single comprehensive Markdown evaluation report."""

    @staticmethod
    def generate_and_save(state: Any) -> Dict[str, Any]:
        def percent(value: Any) -> str:
            try:
                return f"{min(100.0, max(0.0, float(value) * 100)):.2f}%"
            except (TypeError, ValueError):
                return "N/A"

        def clamp_pct(value: Any) -> float:
            """Percentage-point values must stay inside 0-100 for the report."""
            try:
                return min(100.0, max(0.0, float(value)))
            except (TypeError, ValueError):
                return 0.0

        def feature_list(features: list) -> str:
            if not features:
                return "Not available"
            groups = [features[i:i + 6] for i in range(0, len(features), 6)]
            return "\n".join("  " + ", ".join(f"`{feature}`" for feature in group) for group in groups)
        project_id = state.project_id
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
        report_id = f"report_{project_id}_{timestamp}_{uuid.uuid4().hex[:8]}"

        # ---------------------------------------------------------
        # 0. Hydrate Model & Dataset Metadata from Database
        # ---------------------------------------------------------
        model_name = "Unknown Model"
        model_framework = "sklearn"
        model_architecture = "Classification"
        target_classes = "0, 1, 2"
        # Shapes and parameter counts are read from the persisted asset, which
        # is authoritative; the inspector's view is a fallback.
        model_input_shape: Any = "N/A"
        model_output_shape: Any = None
        model_param_count: Any = "N/A"

        dataset_name = "Unknown Dataset"
        row_count = "N/A"
        column_count = "N/A"
        target_column = getattr(state, "target_column", None) or "target"
        model_metadata = getattr(state, "model_metadata", None) or {}
        dataset_metadata = getattr(state, "dataset_metadata", None) or {}
        feature_names = []
        numerical_features = []
        categorical_features = []
        class_distribution = {}
        sample_rows = []

        # Fetch model details
        if getattr(state, "model_id", None):
            try:
                m_info = db.get_model(state.model_id)
                if m_info:
                    model_name = m_info.get("name") or state.model_id
                    model_framework = m_info.get("framework") or model_framework
                    model_architecture = m_info.get("architecture") or model_architecture
                    if m_info.get("output_shape"):
                        target_classes = str(m_info.get("output_shape"))
                    model_output_shape = m_info.get("output_shape") or model_output_shape
                    model_input_shape = m_info.get("input_shape") or model_input_shape
                    if m_info.get("parameter_count") is not None:
                        model_param_count = m_info.get("parameter_count")
            except Exception:
                pass

        if isinstance(model_metadata, dict):
            model_name = model_metadata.get("model_name") or model_name
            model_architecture = (model_metadata.get("architecture") or model_metadata.get("model_class")
                                  or model_metadata.get("model_type") or model_architecture)
            target_classes = model_metadata.get("classes") or model_metadata.get("output_shape") or target_classes
            model_output_shape = model_metadata.get("output_shape") or model_output_shape
            model_input_shape = model_metadata.get("input_shape") or model_input_shape
            if model_metadata.get("parameter_count") is not None:
                model_param_count = model_metadata.get("parameter_count")

        # Fetch dataset details
        if getattr(state, "dataset_id", None):
            try:
                d_info = db.get_dataset(state.dataset_id)
                if d_info:
                    dataset_name = d_info.get("name") or state.dataset_id

                current_ver = getattr(state, "current_dataset_version", "v001")
                with db._get_connection() as conn:
                    row = conn.execute(
                        """
                        SELECT row_count, column_count, operation, parameters 
                        FROM dataset_versions 
                        WHERE dataset_id = ? AND version = ?
                        ORDER BY created_at DESC LIMIT 1
                        """,
                        (state.dataset_id, current_ver)
                    ).fetchone()
                    if row:
                        row_count = row["row_count"] if row["row_count"] is not None else row_count
                        column_count = row["column_count"] if row["column_count"] is not None else column_count
            except Exception:
                pass

        if isinstance(dataset_metadata, dict):
            dataset_name = dataset_metadata.get("dataset_name") or dataset_name
            row_count = dataset_metadata.get("row_count") or dataset_metadata.get("rows") or row_count
            column_count = dataset_metadata.get("column_count") or dataset_metadata.get("columns") or column_count
            target_column = dataset_metadata.get("target_column") or target_column
            feature_names = [
            str(x) for x in (
                dataset_metadata.get("feature_names")
                or dataset_metadata.get("column_names")
                or []
            )
            if str(x) != str(target_column)
        ]
            numerical_features = [str(x) for x in dataset_metadata.get("numerical_features", [])]
            categorical_features = [str(x) for x in dataset_metadata.get("categorical_features", [])]
            class_distribution = dataset_metadata.get("class_distribution") or {}

        # A bounded, JSON-safe dataset preview makes the report auditable.
        if getattr(state, "dataset_path", None):
            try:
                preview = dataset_loader.load(state.dataset_path).head(3)
                sample_rows = json.loads(preview.to_json(orient="records", date_format="iso"))
            except Exception:
                sample_rows = []

        # Fallback to attack/defense metadata if database state is unhydrated
        attack_results = getattr(state, "attack_results", None) or {}
        defense_results = getattr(state, "defense_results", None) or {}
        active_ds_info = (
            attack_results.get("dataset_info")
            or defense_results.get("dataset_info")
            or (getattr(state, "defended_attack_results", None) or {}).get("dataset_info")
            or {}
        )

        if (row_count == "N/A" or row_count == 0) and active_ds_info.get("total_rows"):
            row_count = active_ds_info.get("total_rows")
        if (column_count == "N/A" or column_count == 0) and active_ds_info.get("total_features"):
            column_count = active_ds_info.get("total_features")
        # Total *columns* includes the target; total *features* does not.
        # Report the two separately instead of substituting one for the other.
        total_features = active_ds_info.get("total_features")
        if not total_features and isinstance(column_count, int):
            total_features = max(0, column_count - (1 if target_column else 0))

        # ---------------------------------------------------------
        # 1. Build structured context for LLM Synthesis
        # ---------------------------------------------------------
        formatted_transformations = []
        if getattr(state, "transformations_applied", None):
            for t in state.transformations_applied:
                if isinstance(t, dict):
                    action_name = t.get("action") or t.get("operation") or "Transformation"
                    in_v = t.get("input_version") or "v001"
                    out_v = t.get("output_version") or "v002"
                    params = t.get("parameters") or {}
                    formatted_transformations.append({
                        "operation": action_name,
                        "version_transition": f"{in_v} -> {out_v}",
                        "applied_parameters": params,
                        "explanation": f"Applied '{action_name}' to adjust data distribution/quality."
                    })

        context_for_llm = json.dumps(
            {
                "project_id": project_id,
                "user_objective": getattr(state, "user_goal", "Evaluate model and dataset"),
                "asset_overview": {
                    "model_name": model_name,
                    "model_id": getattr(state, "model_id", "N/A"),
                    "framework": model_framework,
                    "architecture": model_architecture,
                    "output_classes": target_classes,
                    "input_shape": model_input_shape,
                    "output_shape": model_output_shape or target_classes,
                    "parameter_count": model_param_count,
                    "dataset_name": dataset_name,
                    "dataset_id": getattr(state, "dataset_id", "N/A"),
                    "dataset_version": getattr(state, "current_dataset_version", "v001"),
                    "row_count": row_count,
                    "column_count": column_count,
                    "target_column": target_column,
                },
                "baseline_performance": state.baseline_metrics,
                "data_quality_audit": state.data_quality,
                "robustness_benchmarks": state.robustness_results,
                "adversarial_attack_results": attack_results,
                "defense_results": defense_results,
                "defended_attack_results": getattr(state, "defended_attack_results", None) or {},
                "hardened_model_info": getattr(state, "hardened_model_info", None) or {},
                "transformations_executed": formatted_transformations,
                "experiments_history": state.experiments,
                "final_performance": state.latest_metrics,
            },
            indent=2,
            default=str
        )

        # Recorded state is authoritative: the report is rendered entirely from
        # measured state, so an LLM can never invent or replace a metric.
        # `if True:` keeps the original block indentation of the report body.
        if True:
            dim_desc = f"{row_count} rows, {column_count} total columns"
            if isinstance(total_features, int):
                dim_desc += f" ({total_features} features + 1 target)"

            md_lines = [
                "# ML Model & Dataset Evaluation Report",
                (
                    f"**Project:** `{project_id}` | "
                    f"**Generated:** `{datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}`"
                ),
                f"**User Objective:** *{getattr(state, 'user_goal', 'Evaluate model and dataset')}*",
                "",
                "---",
                "## 1. Executive Summary & Asset Overview",
                f"- **Model Name:** `{model_name}` ({model_framework})",
                f"- **Model Architecture:** `{model_architecture}` | **Target Classes:** `{target_classes}`",
                f"- **Dataset Name:** `{dataset_name}` (Version `{getattr(state, 'current_dataset_version', 'v001')}`)",
                f"- **Dimensions:** `{dim_desc}` | **Target Column:** `{target_column}`",
                f"- **Input Shape:** `{model_input_shape}` | **Output Shape:** `{model_output_shape or target_classes}`",
                f"- **Parameter Count:** `{model_param_count}`",
                "",
                "### Dataset Schema",
                f"- **Feature Count:** `{len(feature_names) or 'N/A'}`",
                f"- **Class Distribution:** `{json.dumps(class_distribution, sort_keys=True) if class_distribution else 'Not available'}`",
                "",
                "### Feature Inventory",
                f"- **Input features ({len(feature_names)}):**\n{feature_list(feature_names)}",
                (
                    f"- **Numerical features ({len(numerical_features)}):** all input features are numerical."
                    if numerical_features == feature_names
                    else f"- **Numerical features ({len(numerical_features)}):**\n{feature_list(numerical_features)}"
                ),
                f"- **Categorical features ({len(categorical_features)}):** {feature_list(categorical_features) if categorical_features else 'None'}",
                "",
                "## 2. Dataset Sample",
            ]

            if sample_rows:
                preview_columns = feature_names[:6] + ([target_column] if target_column else [])
                md_lines.extend(["", "### Representative Rows"])
                for index, row in enumerate(sample_rows, start=1):
                    values = "; ".join(f"{column}={row.get(column, 'N/A')}" for column in preview_columns)
                    md_lines.append(f"- **Row {index}:** {values}")
                if len(feature_names) > 6:
                    md_lines.append(f"\n*Preview shows 6 of {len(feature_names)} input features; the full feature inventory is above.*")

            md_lines.extend(["", "## 3. Model Baseline Performance"])

            if getattr(state, "baseline_metrics", None):
                b = state.baseline_metrics
                md_lines.extend([
                    f"- **Accuracy:** {percent(b.get('accuracy'))}",
                    f"- **Precision (macro):** {percent(b.get('precision_macro'))}",
                    f"- **Recall (macro):** {percent(b.get('recall_macro'))}",
                    f"- **F1 score (macro):** {percent(b.get('f1_macro'))}",
                    f"- **F1 score (minority class):** {percent(b.get('f1_minority'))}",
                ])
            else:
                md_lines.append("*No baseline performance metrics recorded.*")

            md_lines.extend([
                "",
                "## 4. Data Quality Audit",
            ])

            if getattr(state, "data_quality", None):
                dq = state.data_quality
                imbalance_text = f"Yes (Ratio: {round(dq.get('imbalance_ratio', 0), 2)}:1)" if dq.get("class_imbalance_detected") else "No"
                md_lines.extend([
                    f"- **Missing Values:** `{dq.get('missing_values_count', 0)}` cells",
                    f"- **Duplicate Rows:** `{dq.get('duplicate_rows_count', 0)}` rows",
                    f"- **Class Imbalance Detected:** `{imbalance_text}`",
                    f"- **Outliers Count:** `{dq.get('outliers_count', 0)}` values",
                    f"- **Constant Columns:** `{dq.get('constant_columns', [])}`",
                ])
            else:
                md_lines.append("*No data quality metrics recorded.*")

            # -------------------------------------------------------
            # 4. Gaussian noise robustness
            # -------------------------------------------------------

            md_lines.extend([
                "",
                "## 5. Robustness — Gaussian Noise Perturbation",
            ])

            if getattr(state, "robustness_results", None):
                rob = state.robustness_results
                clean_acc = float(rob.get("original_score", 1.0)) * 100
                noisy_acc = float(rob.get("perturbed_score", 1.0)) * 100
                drop = clamp_pct(rob.get("drop_percentage", 0.0))

                md_lines.extend([
                    f"- **Perturbation Type:** `{rob.get('perturbation_type', 'gaussian_noise')}`",
                    f"- **Clean Score:** `{clean_acc:.2f}%`",
                    f"- **Noisy Score:** `{noisy_acc:.2f}%`",
                    f"- **Performance Drop:** `{drop:.2f}%`",
                ])
            else:
                md_lines.append("*No Gaussian noise robustness benchmarks recorded.*")

            # -------------------------------------------------------
            # 4b. Adversarial attack results
            # -------------------------------------------------------

            md_lines.extend([
                "",
                "## 6. Adversarial Attack Analysis",
            ])

            if getattr(state, "attack_results", None) and getattr(state, "attack_results", {}).get("status") != "error":
                attack_type_raw = attack_results.get("attack_type") or "unknown"
                attack_type = attack_type_raw.replace("_", " ").title()
                clean_acc = min(100.0, max(0.0, float(attack_results.get("clean_accuracy", 0.0)) * 100))
                adv_acc = min(100.0, max(0.0, float(attack_results.get("adversarial_accuracy", 0.0)) * 100))
                drop = clamp_pct(attack_results.get("drop_percentage", 0.0))

                clean_prec = float(attack_results.get("clean_precision", 0.0)) * 100 if "clean_precision" in attack_results else None
                clean_rec = float(attack_results.get("clean_recall", 0.0)) * 100 if "clean_recall" in attack_results else None
                clean_f1 = float(attack_results.get("clean_f1", 0.0)) * 100 if "clean_f1" in attack_results else None

                adv_prec = float(attack_results.get("adversarial_precision", 0.0)) * 100 if "adversarial_precision" in attack_results else None
                adv_rec = float(attack_results.get("adversarial_recall", 0.0)) * 100 if "adversarial_recall" in attack_results else None
                adv_f1 = float(attack_results.get("adversarial_f1", 0.0)) * 100 if "adversarial_f1" in attack_results else None

                known_hparams = {
                    k: v for k, v in attack_results.items()
                    if k in {
                        "eps", "eps_step", "max_iter", "batch_size",
                        "targeted", "init_size", "max_eval", "nb_parallel",
                    }
                }

                md_lines.extend([
                    f"- **Attack Method:** `{attack_type}`",
                    f"- **Clean Baseline Performance:**",
                    f"  - **Accuracy:** `{clean_acc:.2f}%`",
                ])
                if clean_prec is not None:
                    md_lines.extend([
                        f"  - **Precision (Macro):** `{clean_prec:.2f}%`",
                        f"  - **Recall (Macro):** `{clean_rec:.2f}%`",
                        f"  - **F1 Score (Macro):** `{clean_f1:.2f}%`",
                    ])

                md_lines.extend([
                    f"- **Post-Attack Performance:**",
                    f"  - **Adversarial Accuracy:** `{adv_acc:.2f}%` (Drop: `{drop:.2f}%`)",
                ])
                if adv_prec is not None:
                    md_lines.extend([
                        f"  - **Adversarial Precision:** `{adv_prec:.2f}%` (Drop: `{clamp_pct(clean_prec - adv_prec):.2f}%`)",
                        f"  - **Adversarial Recall:** `{adv_rec:.2f}%` (Drop: `{clamp_pct(clean_rec - adv_rec):.2f}%`)",
                        f"  - **Adversarial F1 Score:** `{adv_f1:.2f}%` (Drop: `{clamp_pct(clean_f1 - adv_f1):.2f}%`)",
                    ])

                if known_hparams:
                    hparam_str = ", ".join(f"{k}={v}" for k, v in known_hparams.items())
                    md_lines.append(f"- **Hyperparameters Used:** `{hparam_str}`")
                else:
                    md_lines.append(
                        "- **Hyperparameters Used:** *not recorded for this run*"
                    )

                ds_info = attack_results.get("dataset_info") or {}
                model_info = attack_results.get("model_info") or {}
                sample_size = attack_results.get("sample_size_used")

                if ds_info:
                    md_lines.append(
                        f"- **Dataset Evaluated:** `{ds_info.get('total_rows', 'N/A')}` rows, "
                        f"`{ds_info.get('total_features', 'N/A')}` features, "
                        f"`{ds_info.get('num_classes', 'N/A')}` classes "
                        f"(target column: `{ds_info.get('target_column', 'N/A')}`)"
                    )
                if sample_size is not None:
                    md_lines.append(f"- **Sample Size Used for Attack:** `{sample_size}` rows")
                if model_info.get("layers"):
                    md_lines.append(f"- **Model Architecture:** `{' -> '.join(model_info['layers'])}`")
                if model_info.get("total_params") is not None:
                    md_lines.append(f"- **Model Parameters:** `{model_info['total_params']}`")
                if model_info.get("input_features") is not None:
                    md_lines.append(f"- **Model Input Features:** `{model_info['input_features']}`")
                if model_info.get("num_classes") is not None:
                    md_lines.append(f"- **Model Output Classes:** `{model_info['num_classes']}`")

                md_lines.append("")
                md_lines.append("**Why This Result Occurred:**")

                gradient_based = attack_type_raw in {"fgsm", "pgd", "cw"}

                if drop <= 5.0:
                    causes = []
                    if known_hparams.get("eps") is not None:
                        causes.append(
                            f"the perturbation budget used (eps={known_hparams['eps']}) may be small "
                            f"relative to this dataset's feature scale, limiting how far inputs could move"
                        )
                    else:
                        causes.append(
                            "the perturbation budget for this run was modest relative to feature scales"
                        )
                    if isinstance(total_features, int) and 0 < total_features <= 15:
                        causes.append(
                            f"this dataset has a low-dimensional feature space "
                            f"({total_features} features), leaving less room for subtle perturbations to cross decision boundaries"
                        )
                    if gradient_based:
                        causes.append(
                            "the model may exhibit local gradient masking -- cross-checking with boundary or decision-based attacks is recommended"
                        )

                    md_lines.append(
                        "This attack produced little to no measurable accuracy drop. Likely contributing "
                        "factors: " + "; ".join(causes) + "."
                    )

                    md_lines.extend([
                        "",
                        "**Optimization Recommendations:**",
                        f"- Re-run {attack_type} with a larger perturbation budget or more iterations.",
                    ])
                    if gradient_based:
                        md_lines.append(
                            "- Cross-check with a decision-based attack (Boundary Attack, HopSkipJump, or ZOO) to rule out gradient masking."
                        )
                else:
                    causes = []
                    eps_val = known_hparams.get('eps', 0.1)
                    causes.append(
                        f"Gradient-Steered Perturbations: The {attack_type} attack calculated the sign of loss gradients "
                        f"w.r.t. input features (epsilon={eps_val}), directly steering feature values in directions of maximum loss increase"
                    )
                    n_features = total_features if isinstance(total_features, int) else None
                    if n_features:
                        causes.append(
                            f"Perturbation Reach: The budget (eps={eps_val}) was applied across "
                            f"all {n_features} input feature dimensions, so a perturbation "
                            f"spread over many features can cross the decision boundary even "
                            f"when each individual shift is small"
                        )
                    causes.append(
                        f"Decision Boundary Proximity: Samples located near the classification margin were pushed "
                        f"across the threshold, reducing model accuracy by {drop:.2f}%"
                    )

                    md_lines.append(
                        f"This attack successfully reduced accuracy by {drop:.2f}% (from {clean_acc:.2f}% clean to {adv_acc:.2f}% adversarial). "
                        f"Key contributing factors:\n" + "\n".join(f"- {c}." for c in causes)
                    )
                    md_lines.extend([
                        "",
                        "**Optimization Recommendations:**",
                        "- Apply preprocessor defenses (Gaussian Augmentation, Feature Squeezing, Spatial Smoothing) to disrupt input perturbations.",
                        "- Apply postprocessor defenses (Reverse Sigmoid, High-Confidence Masking) to adjust model confidence outputs.",
                        "- Perform adversarial fine-tuning on attack samples to strengthen model robustness along vulnerable gradient directions.",
                    ])

            elif getattr(state, "attack_results", None) and getattr(state, "attack_results", {}).get("status") == "error":
                md_lines.append(
                    f"*The requested attack ({attack_results.get('attack_type', 'unknown')}) "
                    f"failed to complete: {attack_results.get('error', 'unknown error')}.*"
                )
            else:
                md_lines.append("*No adversarial attack was run for this project.*")

            # -------------------------------------------------------
            # 4c. Defense results
            # -------------------------------------------------------

            md_lines.extend([
                "",
                "## 7. Defense Analysis",
            ])

            if getattr(state, "defense_results", None) and getattr(state, "defense_results", {}).get("status") != "error":
                defense_type = defense_results.get("defense_type", "gaussian_augmentation")
                clean_acc = min(100.0, max(0.0, float(defense_results.get("clean_accuracy", 0.0)) * 100))
                defended_acc = min(100.0, max(0.0, float(defense_results.get("defended_accuracy", 0.0)) * 100))
                acc_change = defended_acc - clean_acc

                clean_prec = float(defense_results.get("clean_precision", 0.0)) * 100 if "clean_precision" in defense_results else None
                clean_rec = float(defense_results.get("clean_recall", 0.0)) * 100 if "clean_recall" in defense_results else None
                clean_f1 = float(defense_results.get("clean_f1", 0.0)) * 100 if "clean_f1" in defense_results else None

                defended_prec = float(defense_results.get("defended_precision", 0.0)) * 100 if "defended_precision" in defense_results else None
                defended_rec = float(defense_results.get("defended_recall", 0.0)) * 100 if "defended_recall" in defense_results else None
                defended_f1 = float(defense_results.get("defended_f1", 0.0)) * 100 if "defended_f1" in defense_results else None

                known_hparams = {
                    k: v for k, v in defense_results.items()
                    if k in {"sigma", "window_size", "bit_depth", "clip_values", "beta", "gamma", "cutoff"}
                }

                md_lines.extend([
                    f"- **Defense Type:** `{defense_type}`",
                    f"- **Clean Baseline Performance:**",
                    f"  - **Accuracy:** `{clean_acc:.2f}%`",
                ])
                if clean_prec is not None:
                    md_lines.extend([
                        f"  - **Precision (Macro):** `{clean_prec:.2f}%`",
                        f"  - **Recall (Macro):** `{clean_rec:.2f}%`",
                        f"  - **F1 Score (Macro):** `{clean_f1:.2f}%`",
                    ])

                md_lines.extend([
                    f"- **Defended Performance:**",
                    f"  - **Defended Accuracy:** `{defended_acc:.2f}%` (Change: `{(defended_acc - clean_acc):.2f}%`)",
                ])
                if defended_prec is not None:
                    md_lines.extend([
                        f"  - **Defended Precision:** `{defended_prec:.2f}%` (Change: `{(defended_prec - clean_prec):.2f}%`)",
                        f"  - **Defended Recall:** `{defended_rec:.2f}%` (Change: `{(defended_rec - clean_rec):.2f}%`)",
                        f"  - **Defended F1 Score:** `{defended_f1:.2f}%` (Change: `{(defended_f1 - clean_f1):.2f}%`)",
                    ])

                if known_hparams:
                    hparam_str = ", ".join(f"{k}={v}" for k, v in known_hparams.items())
                    md_lines.append(f"- **Hyperparameters Used:** `{hparam_str}`")
                else:
                    md_lines.append(
                        "- **Hyperparameters Used:** *not recorded for this run*"
                    )

                ds_info = defense_results.get("dataset_info") or {}
                model_info = defense_results.get("model_info") or {}
                sample_size = defense_results.get("sample_size_used")

                if ds_info:
                    md_lines.append(
                        f"- **Dataset Evaluated:** `{ds_info.get('total_rows', 'N/A')}` rows, "
                        f"`{ds_info.get('total_features', 'N/A')}` features, "
                        f"`{ds_info.get('num_classes', 'N/A')}` classes "
                        f"(target column: `{ds_info.get('target_column', 'N/A')}`)"
                    )
                if sample_size is not None:
                    md_lines.append(f"- **Sample Size Used for Defense:** `{sample_size}` rows")
                if model_info.get("layers"):
                    md_lines.append(f"- **Model Architecture:** `{' -> '.join(model_info['layers'])}`")
                if model_info.get("total_params") is not None:
                    md_lines.append(f"- **Model Parameters:** `{model_info['total_params']}`")
                if model_info.get("input_features") is not None:
                    md_lines.append(f"- **Model Input Features:** `{model_info['input_features']}`")
                if model_info.get("num_classes") is not None:
                    md_lines.append(f"- **Model Output Classes:** `{model_info['num_classes']}`")

                md_lines.append("")
                md_lines.append("**Why This Result Occurred:**")

                causes = []
                defense_label = defense_type.replace("_", " ").title()

                if abs(acc_change) < 1.0:
                    causes.append(
                        f"Preserved Decision Margin: The transformation applied by {defense_label} was benign enough "
                        f"not to push clean feature vectors across the model's decision boundaries"
                    )
                    causes.append(
                        f"Clean Evaluation Baseline: Evaluated on unperturbed inputs (clean {clean_acc:.2f}% vs defended {defended_acc:.2f}%), "
                        f"confirming low utility loss / impact on standard model inference"
                    )
                    causes.append(
                        f"Untested Adversarial Recovery: To measure the defense's true robustness benefit, "
                        f"it should be evaluated directly against adversarial samples generated by an attack (FGSM/PGD)"
                    )

                    md_lines.append(
                        f"Clean accuracy remained stable after applying {defense_label} "
                        f"({clean_acc:.2f}% clean vs {defended_acc:.2f}% defended). Key contributing factors:\n" +
                        "\n".join(f"- {c}." for c in causes)
                    )
                    md_lines.extend([
                        "",
                        "**Optimization Recommendations:**",
                        f"- Execute an evasion attack (FGSM, PGD, or C&W) first, then apply {defense_label} against adversarial samples to measure recovery rate.",
                        f"- Tune defense hyperparameters (e.g. sigma, bit depth, cutoff) to optimize the balance between clean accuracy and adversarial protection.",
                        "- Combine preprocessor defenses with postprocessor masking for multi-layer defense-in-depth.",
                    ])
                elif acc_change < 0:
                    causes.append(
                        f"Feature Distortion: The transformation applied by {defense_label} significantly modified input feature distributions, "
                        f"causing clean inputs to cross decision boundaries"
                    )
                    causes.append(
                        f"Clean Accuracy Overhead: Performance dropped by {abs(acc_change):.2f}% (from {clean_acc:.2f}% clean to {defended_acc:.2f}% defended), "
                        f"representing a real utility trade-off during standard inference"
                    )

                    md_lines.append(
                        f"Accuracy dropped after applying {defense_label} "
                        f"(clean {clean_acc:.2f}% vs defended {defended_acc:.2f}%). Key contributing factors:\n" +
                        "\n".join(f"- {c}." for c in causes)
                    )
                    md_lines.extend([
                        "",
                        "**Optimization Recommendations:**",
                        f"- Select a milder parameter configuration for {defense_label} (e.g., smaller noise sigma or higher bit depth) to preserve clean accuracy.",
                        "- Test alternative defense mechanisms (such as Reverse Sigmoid or High-Confidence Masking) with lower clean performance impact.",
                        "- Re-evaluate model tolerance to ensure defense overhead remains within acceptable deployment limits.",
                    ])
                else:
                    causes.append(
                        f"Noise Reduction / Regularization Effect: The transformation applied by {defense_label} smoothed out subtle feature noise, "
                        f"helping the model generalize better on evaluation samples"
                    )

                    md_lines.append(
                        f"Accuracy improved by {acc_change:.2f}% after applying {defense_label} "
                        f"(from {clean_acc:.2f}% clean to {defended_acc:.2f}% defended). Key contributing factors:\n" +
                        "\n".join(f"- {c}." for c in causes)
                    )
                    md_lines.extend([
                        "",
                        "**Optimization Recommendations:**",
                        f"- Test {defense_label} against adversarial attack samples to verify if accuracy improvements persist under active gradient pressure.",
                        "- Document this defense configuration as a primary candidate for deployment pipeline integration.",
                    ])

            elif getattr(state, "defense_results", None) and getattr(state, "defense_results", {}).get("status") == "error":
                md_lines.append(
                    f"*The requested defense failed to complete: "
                    f"{defense_results.get('error', 'unknown error')}.*"
                )
            else:
                md_lines.append("*No defense was applied for this project.*")

            defended_attack = getattr(state, "defended_attack_results", None) or {}
            if defended_attack and defended_attack.get("status") != "error":
                md_lines.extend([
                    "",
                    "## 8. Defended Attack Results",
                    f"- **Attack:** `{defended_attack.get('attack_type', 'N/A')}`",
                    f"- **Defense:** `{defended_attack.get('defense_type', 'N/A')}`",
                    "",
                    "",
                    "### 8.1 Accuracy",
                    f"- **Clean Baseline:** {percent(defended_attack.get('clean_accuracy'))}",
                    f"- **Defended (no attack):** {percent(defended_attack.get('defended_accuracy'))}",
                    f"- **Adversarial under Defense:** {percent(defended_attack.get('adversarial_accuracy'))}",
                    f"- **Accuracy drop:** {clamp_pct(defended_attack.get('drop_percentage')):.2f}%",
                    "",
                    "### 8.2 Precision, Recall and F1 (macro)",
                    f"- **Clean Baseline:** precision {percent(defended_attack.get('clean_precision'))}, recall {percent(defended_attack.get('clean_recall'))}, F1 {percent(defended_attack.get('clean_f1'))}",
                    f"- **Defended (no attack):** precision {percent(defended_attack.get('defended_precision'))}, recall {percent(defended_attack.get('defended_recall'))}, F1 {percent(defended_attack.get('defended_f1'))}",
                    f"- **Adversarial under Defense:** precision {percent(defended_attack.get('adversarial_precision'))}, recall {percent(defended_attack.get('adversarial_recall'))}, F1 {percent(defended_attack.get('adversarial_f1'))}",
                    "",
                    f"- **Adversarial samples:** `{defended_attack.get('adversarial_sample_count', 0)}`",
                ])

                att_hp = {
                    k: v for k, v in (defended_attack.get("attack_parameters") or {}).items()
                }
                if att_hp:
                    md_lines.append(
                        "- **Attack hyperparameters:** `"
                        + ", ".join(f"{k}={v}" for k, v in att_hp.items())
                        + "`"
                    )
                def_hp = defended_attack.get("defense_parameters") or {}
                if def_hp:
                    md_lines.append(
                        "- **Defense hyperparameters:** `"
                        + ", ".join(f"{k}={v}" for k, v in def_hp.items())
                        + "`"
                    )

                # A gradient attack through a non-differentiable pre-processor
                # measures masking, not robustness. Never present it as a win.
                masking = defended_attack.get("gradient_masking") or {}
                if masking.get("suspected"):
                    md_lines.append("")
                    md_lines.append("**Gradient Masking Assessment:**")
                    if masking.get("conclusion"):
                        md_lines.append(f"- {masking['conclusion']}")
                    if masking.get("checked"):
                        md_lines.append(
                            f"- Decision-based cross-check accuracy: `{percent(masking.get('cross_check_accuracy'))}` "
                            f"over `{masking.get('cross_check_sample_count', 'N/A')}` samples"
                        )
                    for note in masking.get("notes", []):
                        md_lines.append(f"- {note}")

            # 8.3 sits inside the defended-attack block so a sweep reuses the
            # same section rather than renumbering sections 9-11.
            sequence = getattr(state, "attack_defense_sequence_results", None) or {}
            if sequence.get("steps"):
                md_lines.extend([
                    "",
                    "### 8.3 Attack/Defense Comparison",
                    "",
                    "Each row is an **independent measurement against the same "
                    "model**. The defense is an input preprocessor and is never "
                    "written back, so these combinations do not stack: defense 2 "
                    "is not applied on top of defense 1.",
                    "",
                    f"- **Combinations run:** `{sequence.get('step_count')}` of at most "
                    f"`{sequence.get('max_steps')}`",
                    f"- **Requested:** `{sequence.get('requested_steps')}`",
                    f"- **Framework:** `{sequence.get('framework', 'N/A')}`",
                    f"- **Adversarial samples per step:** `{sequence.get('sample_size_used', 'N/A')}`",
                    "",
                    "| Step | Attack | Defense | Clean Acc | Defended Acc | Adv Acc | Drop % | Adv P | Adv R | Adv F1 | Masking |",
                    "|---|---|---|---|---|---|---|---|---|---|---|",
                ])
                for step in sequence.get("steps", []):
                    masking = step.get("gradient_masking") or {}
                    masking_label = (
                        "suspected" if masking.get("suspected")
                        else "no" if masking else "n/a"
                    )
                    md_lines.append(
                        "| {idx} | `{atk}` | `{dfn}` | {clean} | {defa} | {adv} | "
                        "{drop} | {p} | {r} | {f1} | {mask} |".format(
                            idx=step.get("step_index"),
                            atk=step.get("attack_type") or "none",
                            dfn=step.get("defense_type") or "none",
                            clean=percent(step.get("clean_accuracy")),
                            defa=percent(step.get("defended_accuracy")),
                            adv=percent(step.get("adversarial_accuracy")),
                            drop=f"{clamp_pct(step.get('drop_percentage')):.2f}%",
                            p=percent(step.get("adversarial_precision")),
                            r=percent(step.get("adversarial_recall")),
                            f1=percent(step.get("adversarial_f1")),
                            mask=masking_label,
                        )
                    )

                ranking = sequence.get("ranking") or {}
                strongest = ranking.get("strongest_attack") or {}
                best = ranking.get("best_defense") or {}
                if strongest.get("attack_type"):
                    md_lines.append("")
                    md_lines.append(
                        f"- **Strongest attack:** `{strongest['attack_type']}` at "
                        f"{percent(strongest.get('mean_adversarial_accuracy'))} mean adversarial accuracy"
                    )
                if best.get("defense_type"):
                    md_lines.append(
                        f"- **Best defense:** `{best['defense_type']}` at "
                        f"{percent(best.get('mean_defended_accuracy'))} mean defended accuracy"
                    )

                masked = [
                    s for s in sequence.get("steps", [])
                    if (s.get("gradient_masking") or {}).get("suspected")
                ]
                if masked:
                    labels = [str(s.get("step_index")) for s in masked]
                    md_lines.append("")
                    md_lines.append(
                        "**Gradient Masking Assessment:** "
                        + ("step " if len(labels) == 1 else "steps ")
                        + ", ".join(labels)
                        + (" shows" if len(labels) == 1 else " show")
                        + " apparent robustness that a decision-based "
                        "cross-check does not support. Do not read "
                        + ("that row" if len(labels) == 1 else "those rows")
                        + " as genuine defences."
                    )

            loop = getattr(state, "hardening_loop_results", None) or {}
            if loop.get("rounds"):
                md_lines.extend([
                    "",
                    "## 9. Adversarial Hardening Loop",
                    f"- **Rounds run:** `{loop.get('round_count')}` of at most "
                    f"`{loop.get('max_rounds')}`",
                    f"- **Requested:** `{loop.get('requested_rounds')}`",
                    f"- **Retraining Method:** `{loop.get('retrain_method', 'N/A')}`",
                    f"- **Framework:** `{loop.get('framework', 'N/A')}`",
                    "",
                    "Each round attacks the model produced by the previous round's "
                    "adversarial retraining, so the values accumulate rather than "
                    "repeating.",
                    "",
                    "| Round | Attack | Defense | Clean Acc | Adv Acc | Drop % | Adv P | Adv R | Adv F1 | Masking |",
                    "|---|---|---|---|---|---|---|---|---|---|",
                ])
                for entry in loop.get("rounds", []):
                    masking = entry.get("gradient_masking") or {}
                    masking_label = (
                        "suspected" if masking.get("suspected")
                        else "no" if masking else "n/a"
                    )
                    md_lines.append(
                        "| {idx} | `{atk}` | `{dfn}` | {clean} | {adv} | {drop} | "
                        "{p} | {r} | {f1} | {mask} |".format(
                            idx=entry.get("round_index"),
                            atk=entry.get("attack_type") or "none",
                            dfn=entry.get("defense_type") or "none",
                            clean=percent(entry.get("clean_accuracy")),
                            adv=percent(entry.get("adversarial_accuracy")),
                            drop=f"{clamp_pct(entry.get('drop_percentage')):.2f}%",
                            p=percent(entry.get("adversarial_precision")),
                            r=percent(entry.get("adversarial_recall")),
                            f1=percent(entry.get("adversarial_f1")),
                            mask=masking_label,
                        )
                    )

                improvement = loop.get("improvement") or {}
                first = improvement.get("first_adversarial_accuracy")
                final = improvement.get("final_adversarial_accuracy")
                if first is not None and final is not None:
                    md_lines.extend([
                        "",
                        f"- **Adversarial accuracy, first round:** {percent(first)}",
                        f"- **Adversarial accuracy, final round:** {percent(final)}",
                    ])
                    if improvement.get("comparable", True):
                        md_lines.append(
                            f"- **Change across the loop:** "
                            f"{(clamp_pct(final * 100) - clamp_pct(first * 100)):+.2f} points"
                        )
                    else:
                        md_lines.extend([
                            "- **Mixed-threat run:** each round measured a different "
                            "attack, so the first and final numbers are not "
                            "comparable and no single delta is reported. Use the "
                            "per-attack changes below, which compare only rounds "
                            "that used the same attack.",
                        ])

                per_attack = loop.get("per_attack_change") or {}
                if per_attack and len(per_attack) > 1:
                    md_lines.extend([
                        "",
                        "| Attack | Rounds | First Adv Acc | Last Adv Acc | Change |",
                        "|---|---|---|---|---|",
                    ])
                    for name, stats in per_attack.items():
                        md_lines.append(
                            "| `{atk}` | {n} | {f} | {l} | {d:+.2f} pts |".format(
                                atk=name,
                                n=stats.get("rounds"),
                                f=percent(stats.get("first_adversarial_accuracy")),
                                l=percent(stats.get("last_adversarial_accuracy")),
                                d=(
                                    clamp_pct(stats.get("last_adversarial_accuracy", 0) * 100)
                                    - clamp_pct(stats.get("first_adversarial_accuracy", 0) * 100)
                                ),
                            )
                        )

                if first is not None and final is not None:
                    md_lines.extend([
                        "",
                        "Read the clean-accuracy column alongside the adversarial "
                        "columns: robustness gained here is normally paid for with "
                        "some clean accuracy.",
                    ])

                if loop.get("final_model_id"):
                    md_lines.extend([
                        "",
                        f"- **Final Hardened Model ID:** `{loop.get('final_model_id')}`",
                        f"- **Final Asset Path:** `{loop.get('final_model_path', 'N/A')}`",
                        "- **Intermediate rounds** were retrained but not registered "
                        "as separate assets.",
                    ])

            hardened = getattr(state, "hardened_model_info", None) or {}
            if hardened and not loop.get("rounds"):
                training = hardened.get("training_summary") or {}
                md_lines.extend([
                    "",
                    "## 9. Adversarial Hardening",
                    f"- **Hardened Model ID:** `{hardened.get('model_id', 'N/A')}`",
                    f"- **Hardened Model Name:** `{hardened.get('model_name', 'N/A')}`",
                    f"- **Framework:** `{hardened.get('framework', 'N/A')}`",
                    f"- **Retraining Method:** `{training.get('method', 'N/A')}`",
                    f"- **Adversarial Samples Used:** `{training.get('samples_generated', 'N/A')}`",
                    f"- **Export Status:** `{training.get('export_status', 'N/A')}`",
                    f"- **New Asset Path:** `{hardened.get('file_path', 'N/A')}`",
                ])
                harden_hp = {
                    k: v for k, v in training.items()
                    if k in {"eps", "eps_step", "max_iter", "batch_size", "sample_size", "epochs", "lr"}
                }
                if harden_hp:
                    md_lines.append(
                        "- **Retraining hyperparameters:** `"
                        + ", ".join(f"{k}={v}" for k, v in harden_hp.items())
                        + "`"
                    )

            md_lines.extend([
                "",
                "## 10. Experiment Summary & Transformation Impact",
            ])

            if state.experiments:
                for exp in state.experiments:
                    comp = exp.get("comparison", {})
                    outcome = "Improved Performance" if comp.get("is_improved") else "Neutral / Degraded"
                    md_lines.extend([
                        f"### Experiment: `{exp.get('operation', 'Transformation')}`",
                        f"- **Dataset Version:** `{exp.get('input_dataset_version')} -> {exp.get('output_dataset_version')}`",
                        f"- **Outcome:** `{outcome}`",
                        "- **Summary of Changes:**",
                    ])
                    for change in comp.get("summary_of_changes", []):
                        md_lines.append(f"  - {change}")
                    if exp.get("agent_reasoning"):
                        md_lines.append(f"- **Agent Reasoning:** *{exp.get('agent_reasoning')}*")
            else:
                md_lines.append("*No data transformations were required.*")

            md_lines.extend([
                "",
                "## 11. Diagnostic Synthesis & Recommendations",
                "The model was evaluated against the provided dataset. Baseline performance, data quality, and robustness metrics were calculated deterministically."
            ])

            final_markdown = "\n".join(md_lines)

        # ---------------------------------------------------------
        # Save Markdown report
        # ---------------------------------------------------------

        report_file = (
            storage_manager.reports_dir
            / f"{report_id}.md"
        )

        storage_manager.reports_dir.mkdir(parents=True, exist_ok=True)
        fd, temp_name = tempfile.mkstemp(prefix=f".{report_id}_", suffix=".tmp", dir=storage_manager.reports_dir)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(final_markdown)

        # ---------------------------------------------------------
        # Save report metadata/content to SQLite
        # ---------------------------------------------------------

            with db._get_connection() as conn:
                conn.execute(
                """
                INSERT INTO reports
                (
                    id,
                    project_id,
                    title,
                    summary,
                    content_markdown,
                    content_json,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    report_id,
                    project_id,
                    f"Report for {dataset_name}",
                    f"Evaluation Report for model '{model_name}' and dataset '{dataset_name}'",
                    final_markdown,
                    context_for_llm,
                    datetime.utcnow().isoformat(),
                ),
            )

                conn.commit()
            os.replace(temp_name, report_file)
        except Exception as exc:
            # Best-effort cleanup only. A failure here must not replace the
            # real cause of the report failure.
            logger = logging.getLogger(__name__)
            for cleanup_path in (temp_name, str(report_file)):
                try:
                    if cleanup_path and os.path.exists(cleanup_path):
                        os.unlink(cleanup_path)
                except Exception as cleanup_exc:
                    logger.warning("Report cleanup failed for %s: %s", cleanup_path, cleanup_exc)
            try:
                with db._get_connection() as conn:
                    conn.execute("DELETE FROM reports WHERE id = ?", (report_id,))
                    conn.commit()
            except Exception as cleanup_exc:
                logger.warning("Report row cleanup failed for %s: %s", report_id, cleanup_exc)
            raise exc

        return {
            "report_id": report_id,
            "file_path": str(report_file),
            "markdown": final_markdown,
        }


report_generator = ReportGenerator()
