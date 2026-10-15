from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field

from app.datasets.loader import dataset_loader
from app.models.inspector import model_inspector


class ValidationResult(BaseModel):
    is_compatible: bool
    checks: Dict[str, bool] = Field(default_factory=dict)
    issues: List[str] = Field(default_factory=list)


class CompatibilityValidator:
    """Validates input/output shape and feature compatibility between model and dataset."""

    def validate_compatibility(
        self,
        model_path: str,
        dataset_path: str,
        model_id: str = "model_001",
        dataset_id: str = "dataset_001",
        version: str = "v001",
        target_column: Optional[str] = None,
    ) -> ValidationResult:
        issues: List[str] = []
        checks: Dict[str, bool] = {}

        try:
            df = dataset_loader.load(dataset_path)

            # Drop target column if specified or named "target"
            if target_column and target_column in df.columns:
                feature_df = df.drop(columns=[target_column])
            elif "target" in df.columns:
                feature_df = df.drop(columns=["target"])
            else:
                feature_df = df

            dataset_feature_count = feature_df.shape[1]

            model_info = model_inspector.inspect(model_path, model_id)
            if hasattr(model_info, "model_dump"):
                info_dict = model_info.model_dump()
            elif isinstance(model_info, dict):
                info_dict = model_info
            else:
                info_dict = dict(model_info)

            input_shape = info_dict.get("input_shape", [])
            expected_features = input_shape[0] if input_shape else None

            if expected_features is not None and expected_features != dataset_feature_count:
                checks["input_shape"] = False
                issues.append(
                    f"Model expects {expected_features} input features, "
                    f"but dataset contains {dataset_feature_count} feature columns."
                )
            else:
                checks["input_shape"] = True

            is_compatible = all(checks.values()) if checks else True

            return ValidationResult(
                is_compatible=is_compatible,
                checks=checks,
                issues=issues,
            )

        except Exception as exc:
            return ValidationResult(
                is_compatible=False,
                checks={"input_shape": False},
                issues=[f"Compatibility validation error: {str(exc)}"],
            )


compatibility_validator = CompatibilityValidator()