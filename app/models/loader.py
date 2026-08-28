from pathlib import Path
from typing import Any, Tuple, Optional
from app.models.adapters import BaseModelAdapter, SklearnAdapter, PyTorchAdapter, ONNXAdapter

class ModelLoader:
    """Detects format and loads model using corresponding adapter."""

    ADAPTER_MAP = {
        ".pkl": SklearnAdapter,
        ".joblib": SklearnAdapter,
        ".sav": SklearnAdapter,
        ".pt": PyTorchAdapter,
        ".pth": PyTorchAdapter,
        ".onnx": ONNXAdapter
    }

    @classmethod
    def get_adapter_for_path(cls, file_path: str) -> Tuple[BaseModelAdapter, str]:
        ext = Path(file_path).suffix.lower()
        if ext in [".pkl", ".joblib", ".sav"]:
            return SklearnAdapter(), "sklearn"
        elif ext in [".pt", ".pth"]:
            return PyTorchAdapter(), "pytorch"
        elif ext in [".onnx"]:
            return ONNXAdapter(), "onnx"
        else:
            # Default try sklearn
            return SklearnAdapter(), "sklearn"

    @classmethod
    def load_model(cls, file_path: str) -> Tuple[Any, BaseModelAdapter, str]:
        adapter, framework = cls.get_adapter_for_path(file_path)
        model = adapter.load(file_path)
        return model, adapter, framework

model_loader = ModelLoader()

