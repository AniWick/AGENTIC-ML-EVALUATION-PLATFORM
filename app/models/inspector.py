from typing import Dict, Any
from app.models.loader import model_loader

class ModelInspector:
    """Inspects model architecture, input/output dimensions, feature requirements, and parameters."""

    @staticmethod
    def inspect(file_path: str, model_id: str = "model_001") -> Dict[str, Any]:
        model, adapter, framework = model_loader.load_model(file_path)
        info = adapter.inspect(model)
        info["model_id"] = model_id
        info["file_path"] = file_path
        return info

model_inspector = ModelInspector()

