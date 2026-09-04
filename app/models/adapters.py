from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
import os
import joblib
import pickle
import numpy as np
import pandas as pd


class BaseModelAdapter(ABC):
    """Base interface for model framework adapters."""

    @abstractmethod
    def load(self, file_path: str) -> Any:
        pass

    @abstractmethod
    def inspect(self, model: Any) -> Dict[str, Any]:
        pass

    @abstractmethod
    def validate(self, model: Any) -> Tuple[bool, List[str]]:
        pass

    @abstractmethod
    def predict(self, model: Any, X: Any) -> np.ndarray:
        pass

    @abstractmethod
    def predict_proba(self, model: Any, X: Any) -> Optional[np.ndarray]:
        pass


class SklearnAdapter(BaseModelAdapter):
    """Adapter for Scikit-Learn pickled/joblib models."""

    def load(self, file_path: str) -> Any:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Model file not found: {file_path}")

        try:
            return joblib.load(file_path)
        except Exception:
            with open(file_path, "rb") as f:
                return pickle.load(f)

    def inspect(self, model: Any) -> Dict[str, Any]:
        info: Dict[str, Any] = {
            "framework": "sklearn",
            "model_class": model.__class__.__name__,
            "model_module": model.__class__.__module__,
            "status": "loaded"
        }

        # Detect classification vs regression
        if hasattr(model, "classes_"):
            info["model_type"] = "classification"
            info["output_classes"] = len(model.classes_)
            info["target_classes"] = [
                str(c) for c in model.classes_.tolist()
            ]

        elif (
            hasattr(model, "_estimator_type")
            and model._estimator_type == "classifier"
        ):
            info["model_type"] = "classification"

        elif (
            hasattr(model, "_estimator_type")
            and model._estimator_type == "regressor"
        ):
            info["model_type"] = "regression"

        else:
            info["model_type"] = (
                "classification"
                if "Classifier" in model.__class__.__name__
                else "regression"
            )

        # Detect expected input features
        if hasattr(model, "n_features_in_"):
            info["input_shape"] = [model.n_features_in_]

        if hasattr(model, "feature_names_in_"):
            info["feature_names"] = list(model.feature_names_in_)

        # Parameter count estimation
        param_count = 0

        if hasattr(model, "coef_"):
            param_count += getattr(model.coef_, "size", 0)

        if hasattr(model, "intercept_"):
            param_count += getattr(model.intercept_, "size", 0)

        info["parameter_count"] = (
            param_count if param_count > 0 else None
        )

        return info

    def validate(self, model: Any) -> Tuple[bool, List[str]]:
        issues = []

        if not hasattr(model, "predict"):
            issues.append(
                "Model object does not have a 'predict' method."
            )

        return len(issues) == 0, issues

    @staticmethod
    def _prepare_input(model: Any, X: Any) -> Any:
        """
        Prepare prediction input according to how the sklearn model
        was originally fitted.

        Models fitted with a pandas DataFrame have feature_names_in_.
        Models fitted with NumPy arrays generally do not.

        This prevents sklearn feature-name warnings while preserving
        the original feature order.
        """

        # Model was fitted with feature names.
        if hasattr(model, "feature_names_in_"):
            expected_features = list(model.feature_names_in_)

            if isinstance(X, pd.DataFrame):
                # Make sure the required columns exist.
                missing = [
                    col for col in expected_features
                    if col not in X.columns
                ]

                if missing:
                    raise ValueError(
                        f"Input dataset is missing model features: {missing}"
                    )

                # Explicitly enforce the model's training order.
                return X[expected_features]

            # If a NumPy array is supplied to a model that expects
            # feature names, create a DataFrame using those names.
            X_array = np.asarray(X)

            if X_array.ndim == 1:
                X_array = X_array.reshape(1, -1)

            if X_array.shape[1] != len(expected_features):
                raise ValueError(
                    f"Model expects {len(expected_features)} features, "
                    f"but received {X_array.shape[1]}."
                )

            return pd.DataFrame(
                X_array,
                columns=expected_features
            )

        # Model was fitted without feature names.
        # Therefore send a NumPy array to avoid sklearn's
        # "X has feature names" warning.
        if isinstance(X, pd.DataFrame):
            return X.to_numpy()

        return np.asarray(X)

    def predict(self, model: Any, X: Any) -> np.ndarray:
        X_prepared = self._prepare_input(model, X)
        return np.asarray(model.predict(X_prepared))

    def predict_proba(
        self,
        model: Any,
        X: Any
    ) -> Optional[np.ndarray]:

        if not hasattr(model, "predict_proba"):
            return None

        try:
            X_prepared = self._prepare_input(model, X)
            return np.asarray(model.predict_proba(X_prepared))
        except Exception:
            return None


class PyTorchAdapter(BaseModelAdapter):
    """Adapter for PyTorch saved models or TorchScript models."""

    def load(self, file_path: str) -> Any:
        import torch

        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"Model file not found: {file_path}"
            )

        try:
            # Try loading scripted model first
            model = torch.jit.load(
                file_path,
                map_location="cpu"
            )
            model.eval()
            return model

        except Exception:
            # Fallback to torch.load
            model = torch.load(
                file_path,
                map_location="cpu",
                weights_only=False
            )

            if hasattr(model, "eval"):
                model.eval()

            return model

    def inspect(self, model: Any) -> Dict[str, Any]:
        import torch

        info: Dict[str, Any] = {
            "framework": "pytorch",
            "model_class": model.__class__.__name__,
            "status": "loaded"
        }

        param_count = 0

        if isinstance(model, torch.nn.Module):
            param_count = sum(
                p.numel()
                for p in model.parameters()
            )

            info["parameter_count"] = param_count

        info["model_type"] = "classification"

        return info

    def validate(self, model: Any) -> Tuple[bool, List[str]]:
        issues = []

        if not callable(model) and not hasattr(model, "forward"):
            issues.append(
                "PyTorch model is not callable."
            )

        return len(issues) == 0, issues

    def predict(self, model: Any, X: np.ndarray) -> np.ndarray:
        import torch

        # Convert DataFrame to NumPy for neural network input.
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()

        tensor_x = torch.tensor(
            X,
            dtype=torch.float32
        )

        with torch.no_grad():
            outputs = model(tensor_x)

            if isinstance(outputs, torch.Tensor):

                if (
                    outputs.dim() > 1
                    and outputs.shape[1] > 1
                ):
                    return (
                        torch.argmax(outputs, dim=1)
                        .cpu()
                        .numpy()
                    )

                elif (
                    outputs.dim() > 1
                    and outputs.shape[1] == 1
                ):
                    preds = (
                        (outputs > 0.5)
                        .long()
                        .squeeze()
                        .cpu()
                        .numpy()
                    )

                    return preds

                return (
                    outputs.squeeze()
                    .cpu()
                    .numpy()
                )

            return np.array(outputs)

    def predict_proba(
        self,
        model: Any,
        X: np.ndarray
    ) -> Optional[np.ndarray]:

        import torch

        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()

        tensor_x = torch.tensor(
            X,
            dtype=torch.float32
        )

        with torch.no_grad():
            outputs = model(tensor_x)

            if (
                isinstance(outputs, torch.Tensor)
                and outputs.dim() > 1
                and outputs.shape[1] > 1
            ):
                probs = (
                    torch.softmax(outputs, dim=1)
                    .cpu()
                    .numpy()
                )

                return probs

        return None


class ONNXAdapter(BaseModelAdapter):
    """Adapter for ONNX Runtime models."""

    def load(self, file_path: str) -> Any:
        import onnxruntime as ort

        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"Model file not found: {file_path}"
            )

        return ort.InferenceSession(
            file_path,
            providers=["CPUExecutionProvider"]
        )

    def inspect(self, model: Any) -> Dict[str, Any]:
        inputs = model.get_inputs()
        outputs = model.get_outputs()

        info = {
            "framework": "onnx",
            "status": "loaded",
            "model_type": "classification",
            "input_names": [i.name for i in inputs],
            "input_shape": [i.shape for i in inputs],
            "output_names": [o.name for o in outputs],
            "output_shape": [o.shape for o in outputs],
        }

        return info

    def validate(self, model: Any) -> Tuple[bool, List[str]]:
        issues = []

        if not hasattr(model, "run"):
            issues.append(
                "ONNX session missing run method."
            )

        return len(issues) == 0, issues

    def predict(self, model: Any, X: np.ndarray) -> np.ndarray:

        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()

        input_name = model.get_inputs()[0].name

        tensor_x = X.astype(np.float32)

        outputs = model.run(
            None,
            {input_name: tensor_x}
        )

        res = outputs[0]

        if res.ndim > 1 and res.shape[1] > 1:
            return np.argmax(res, axis=1)

        return res.flatten()

    def predict_proba(
        self,
        model: Any,
        X: np.ndarray
    ) -> Optional[np.ndarray]:

        try:
            if isinstance(X, pd.DataFrame):
                X = X.to_numpy()

            input_name = model.get_inputs()[0].name

            tensor_x = X.astype(np.float32)

            outputs = model.run(
                None,
                {input_name: tensor_x}
            )

            res = outputs[0]

            if res.ndim > 1 and res.shape[1] > 1:
                # Apply softmax
                exp_res = np.exp(
                    res - np.max(
                        res,
                        axis=1,
                        keepdims=True
                    )
                )

                return (
                    exp_res
                    / np.sum(
                        exp_res,
                        axis=1,
                        keepdims=True
                    )
                )

        except Exception:
            pass

        return None