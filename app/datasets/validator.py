from typing import Dict, Any, List, Tuple, Optional
from app.models.inspector import model_inspector
from app.datasets.inspector import dataset_inspector
from app.schemas.dataset import ValidationResult

class CompatibilityValidator:
    """Validates compatibility between an uploaded model and dataset."""

    @staticmethod
    def validate_compatibility(
        model_path: str,
        dataset_path: str,
        model_id: str = "model_001",
        dataset_id: str = "dataset_001",
        dataset_version: str = "v001"
    ) -> ValidationResult:
        model_info = model_inspector.inspect(model_path, model_id)
        ds_info = dataset_inspector.inspect(dataset_path, dataset_id, dataset_version)

        issues: List[str] = []
        warnings: List[str] = []
        checks: Dict[str, bool] = {
            "input_shape": True,
            "feature_types": True,
            "target_compatible": True
        }

        # 1. Feature count check
        target_col = ds_info.get("target_column")
        feature_cols = [c for c in ds_info["column_names"] if c != target_col]
        dataset_feature_count = len(feature_cols)

        expected_shape = model_info.get("input_shape")
        if expected_shape and len(expected_shape) > 0:
            expected_features = expected_shape[0]
            if expected_features is not None and expected_features != dataset_feature_count:
                checks["input_shape"] = False
                issues.append(
                    f"Feature count mismatch: Dataset has {dataset_feature_count} input features, "
                    f"but model expects {expected_features} features."
                )

        # 2. Feature names check if model recorded them
        model_feature_names = model_info.get("feature_names")
        if model_feature_names:
            missing_in_dataset = set(model_feature_names) - set(feature_cols)
            if missing_in_dataset:
                checks["input_shape"] = False
                issues.append(f"Missing expected features in dataset: {list(missing_in_dataset)}")

        # 3. Categorical features check
        categorical_features = ds_info.get("categorical_features", [])
        if categorical_features:
            warnings.append(
                f"Dataset contains {len(categorical_features)} categorical columns {categorical_features}. "
                f"Ensure proper encoding before feeding into numerical models."
            )

        # 4. Missing values warning
        if ds_info.get("missing_values_count", 0) > 0:
            warnings.append(
                f"Dataset contains {ds_info['missing_values_count']} missing values which may cause inference failure."
            )

        # 5. Class count compatibility for classification
        if model_info.get("model_type") == "classification":
            class_dist = ds_info.get("class_distribution")
            output_classes = model_info.get("output_classes")
            if class_dist and output_classes:
                distinct_dataset_classes = len(class_dist)
                if distinct_dataset_classes > output_classes:
                    checks["target_compatible"] = False
                    issues.append(
                        f"Dataset has {distinct_dataset_classes} target classes, but model only outputs {output_classes} classes."
                    )

        is_compatible = len(issues) == 0

        return ValidationResult(
            is_compatible=is_compatible,
            model_id=model_id,
            dataset_id=dataset_id,
            dataset_version=dataset_version,
            checks=checks,
            issues=issues,
            warnings=warnings
        )

compatibility_validator = CompatibilityValidator()

