from pathlib import Path
from typing import Any, Tuple

from app.models.adapters import (
    BaseModelAdapter,
    SklearnAdapter,
    PyTorchAdapter,
    ONNXAdapter
)
from app.core.storage import storage_manager


class ModelLoader:
    """Detects model format and loads model using the corresponding adapter."""

    ADAPTER_MAP = {
        ".pkl": SklearnAdapter,
        ".joblib": SklearnAdapter,
        ".sav": SklearnAdapter,
        ".pt": PyTorchAdapter,
        ".pth": PyTorchAdapter,
        ".onnx": ONNXAdapter
    }

    @classmethod
    def get_adapter_for_path(
        cls,
        file_path: str
    ) -> Tuple[BaseModelAdapter, str]:

        ext = Path(file_path).suffix.lower()

        if ext in [".pkl", ".joblib", ".sav"]:
            return SklearnAdapter(), "sklearn"

        elif ext in [".pt", ".pth"]:
            return PyTorchAdapter(), "pytorch"

        elif ext == ".onnx":
            return ONNXAdapter(), "onnx"

        else:
            # Default to sklearn
            return SklearnAdapter(), "sklearn"

    @classmethod
    def load_model(
        cls,
        file_path: str
    ) -> Tuple[Any, BaseModelAdapter, str]:

        # Resolve old/stored paths against the current storage location.
        resolved_path = storage_manager.resolve_path(file_path)

        adapter, framework = cls.get_adapter_for_path(
            str(resolved_path)
        )

        model = adapter.load(str(resolved_path))

        return model, adapter, framework


model_loader = ModelLoader()