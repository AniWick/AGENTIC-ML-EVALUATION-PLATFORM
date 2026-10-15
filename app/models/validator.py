import os
from typing import Tuple, List, Dict, Any
from app.models.loader import model_loader

class ModelValidator:
    """Validates model file existence, loadability, and framework support."""

    @staticmethod
    def validate(file_path: str) -> Tuple[bool, List[str]]:
        issues = []
        if not os.path.exists(file_path):
            return False, [f"Model file does not exist at {file_path}"]

        try:
            model, adapter, framework = model_loader.load_model(file_path)
            valid, adapter_issues = adapter.validate(model)
            if not valid:
                issues.extend(adapter_issues)
        except Exception as e:
            issues.append(f"Failed to load model: {str(e)}")

        return len(issues) == 0, issues

model_validator = ModelValidator()

