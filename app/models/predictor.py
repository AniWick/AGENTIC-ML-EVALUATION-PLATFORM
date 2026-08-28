import numpy as np
from typing import Any, Optional, Tuple
from app.models.loader import model_loader

class ModelPredictor:
    """Handles running inferences and probability scores on datasets."""

    @staticmethod
    def predict(file_path: str, X: np.ndarray) -> np.ndarray:
        model, adapter, framework = model_loader.load_model(file_path)
        return adapter.predict(model, X)

    @staticmethod
    def predict_proba(file_path: str, X: np.ndarray) -> Optional[np.ndarray]:
        model, adapter, framework = model_loader.load_model(file_path)
        return adapter.predict_proba(model, X)

model_predictor = ModelPredictor()

