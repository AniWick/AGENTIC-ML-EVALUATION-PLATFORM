import numpy as np
from typing import Dict, Any, Optional, Tuple, List
import warnings

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, mean_squared_error

from art.estimators.classification import SklearnClassifier, PyTorchClassifier, BlackBoxClassifier
from art.attacks.evasion import (
    BoundaryAttack,
    HopSkipJump,
    ZooAttack,
    FastGradientMethod,
    ProjectedGradientDescent,
    CarliniL2Method,
)
from art.defences.preprocessor import GaussianAugmentation, SpatialSmoothing, FeatureSqueezing
try:
    from art.defences.preprocessor import ClipValues
except ImportError:
    from art.defences.preprocessor.preprocessor import Preprocessor

    class ClipValues(Preprocessor):
        """Preprocessor defense that clips input feature values to specified min/max bounds."""
        def __init__(self, clip_values: Tuple[float, float], apply_fit: bool = False, apply_predict: bool = True):
            super().__init__(is_fitted=True, apply_fit=apply_fit, apply_predict=apply_predict)
            self.clip_values = clip_values

        def __call__(self, x: np.ndarray, y: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
            x_clipped = np.clip(x, self.clip_values[0], self.clip_values[1])
            return x_clipped, y

        def estimate_gradient(self, x: np.ndarray, grad: np.ndarray) -> np.ndarray:
            return grad

        def fit(self, x: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> None:
            pass

from art.defences.postprocessor import ReverseSigmoid, ClassLabels
try:
    from art.defences.postprocessor import HighConfidenceMasking
except ImportError:
    from art.defences.postprocessor import HighConfidence as HighConfidenceMasking

from app.models.loader import model_loader
from app.models.predictor import model_predictor
from app.datasets.loader import dataset_loader
from app.schemas.evaluation import RobustnessMetrics


def _stratified_subsample(X: np.ndarray, y: np.ndarray, n: int = 30, seed: int = 42):
    """Seeded, class-balanced sample instead of the first n rows."""
    if len(X) <= n:
        return X, y
    rng = np.random.RandomState(seed)
    classes, counts = np.unique(y, return_counts=True)
    per_class = max(1, n // len(classes))
    idx = []
    for c in classes:
        c_idx = np.where(y == c)[0]
        take = min(per_class, len(c_idx))
        idx.extend(rng.choice(c_idx, size=take, replace=False))
    idx = np.array(idx)
    if len(idx) < n:
        remaining = np.setdiff1d(np.arange(len(X)), idx)
        extra = rng.choice(remaining, size=min(n - len(idx), len(remaining)), replace=False)
        idx = np.concatenate([idx, extra])
    rng.shuffle(idx)
    return X[idx], y[idx]


def _describe_model(loaded_model: Any, framework: str) -> Dict[str, Any]:
    """Best-effort, non-fabricated architecture summary for the report."""
    info: Dict[str, Any] = {"model_class": type(loaded_model).__name__, "framework": framework}
    try:
        if framework == "sklearn":
            if hasattr(loaded_model, "n_features_in_"):
                info["input_features"] = int(loaded_model.n_features_in_)
            if hasattr(loaded_model, "classes_"):
                info["num_classes"] = int(len(loaded_model.classes_))
        elif framework == "pytorch":
            layers = []
            for m in loaded_model.modules():
                if hasattr(m, "in_features") and hasattr(m, "out_features"):
                    layers.append(f"{type(m).__name__}({m.in_features}->{m.out_features})")
            info["layers"] = layers
            info["total_params"] = sum(p.numel() for p in loaded_model.parameters()) if hasattr(loaded_model, "parameters") else None
        elif framework == "onnx":
            inputs = loaded_model.get_inputs()
            outputs = loaded_model.get_outputs()
            info["input_shape"] = [i.shape for i in inputs]
            info["output_shape"] = [o.shape for o in outputs]
    except Exception:
        pass
    return info


def _map_predictions(preds: np.ndarray, loaded_model: Any, y_all: np.ndarray) -> np.ndarray:
    """Maps predicted probability/logit/index arrays into actual class labels matching target space."""
    if preds.ndim > 1:
        idx = np.argmax(preds, axis=1)
    else:
        idx = preds.astype(int)

    if hasattr(loaded_model, "classes_"):
        classes = getattr(loaded_model, "classes_")
        if len(classes) > 0 and np.max(idx) < len(classes):
            return classes[idx]

    unique_y = np.unique(y_all)
    if len(unique_y) > 0 and np.max(idx) < len(unique_y):
        if not np.issubdtype(y_all.dtype, np.integer) or not np.array_equal(unique_y, np.arange(len(unique_y))):
            return unique_y[idx]

    return idx


# Single source of truth for attack capability, shared with the planner's
# pre-flight check in app/agent/security_aliases.py.
from app.agent.security_aliases import ATTACKS_BY_FRAMEWORK  # noqa: E402

EVASION_ATTACKS_BY_FRAMEWORK: Dict[str, List[str]] = {
    framework: list(attacks)
    for framework, attacks in ATTACKS_BY_FRAMEWORK.items()
}

ATTACK_DISPLAY_NAMES: Dict[str, str] = {
    "boundary_attack": "Boundary Attack",
    "hopskipjump": "HopSkipJump Attack",
    "zoo": "ZOO Attack",
    "fgsm": "Fast Gradient Method (FGSM)",
    "pgd": "Projected Gradient Descent (PGD)",
    "cw": "Carlini & Wagner (C&W)",
}

DEFENSE_REGISTRY: List[str] = [
    "gaussian_augmentation",
    "spatial_smoothing",
    "feature_squeezing",
    "clip_values",
    "reverse_sigmoid",
    "class_labels",
    "high_confidence_masking",
]

DEFENSE_DISPLAY_NAMES: Dict[str, str] = {
    "gaussian_augmentation": "Gaussian Augmentation (Preprocessor)",
    "spatial_smoothing": "Spatial Smoothing (Preprocessor)",
    "feature_squeezing": "Feature Squeezing (Preprocessor)",
    "clip_values": "Clip Values (Preprocessor)",
    "reverse_sigmoid": "Reverse Sigmoid (Postprocessor)",
    "class_labels": "Class Labels (Postprocessor)",
    "high_confidence_masking": "High Confidence Masking (Postprocessor)",
}


def list_compatible_evasion_attacks(model_path: str) -> List[Dict[str, str]]:
    _, framework = model_loader.get_adapter_for_path(model_path)
    names = EVASION_ATTACKS_BY_FRAMEWORK.get(framework, [])
    return [
        {"name": n, "label": ATTACK_DISPLAY_NAMES.get(n, n), "framework": framework}
        for n in names
    ]


def _infer_pytorch_shape_and_classes(model: Any, X: np.ndarray) -> Tuple[Tuple[int, ...], int]:
    input_shape = (X.shape[1],) if X.ndim == 2 else tuple(X.shape[1:])
    last_linear = None
    for module in model.modules():
        if hasattr(module, "out_features") and hasattr(module, "in_features"):
            last_linear = module

    if last_linear is None:
        raise ValueError(
            "Could not determine the number of output classes for this PyTorch model automatically."
        )

    return input_shape, last_linear.out_features


def _infer_onnx_shape_and_classes(session: Any) -> Tuple[Tuple[int, ...], int]:
    try:
        input_info = session.get_inputs()[0]
        output_info = session.get_outputs()[0]

        input_shape = tuple(d for d in input_info.shape if isinstance(d, int))
        int_output_dims = [d for d in output_info.shape if isinstance(d, int)]
        nb_classes = int_output_dims[-1] if int_output_dims else None

        if not input_shape or not nb_classes:
            raise ValueError("Model I/O shapes are not fully specified.")

        return input_shape, int(nb_classes)
    except Exception as exc:
        raise ValueError(f"Could not determine ONNX model input/output shape: {exc}")


def _onnx_predict_fn(session: Any, adapter: Any):
    def predict_fn(x: np.ndarray) -> np.ndarray:
        probs = adapter.predict_proba(session, x)
        if probs is not None:
            return probs

        preds = adapter.predict(session, x)
        nb_classes = int(np.max(preds)) + 1
        one_hot = np.zeros((len(preds), nb_classes), dtype=np.float32)
        one_hot[np.arange(len(preds)), preds.astype(int)] = 1.0
        return one_hot

    return predict_fn


def _build_art_classifier(
    loaded_model: Any,
    adapter: Any,
    framework: str,
    X: np.ndarray,
    clip_values: Tuple[float, float],
    defense_names: Optional[List[str]] = None,
    defense_params: Optional[Dict[str, Any]] = None,
) -> Any:
    pre_defenses, post_defenses = [], []
    if defense_names:
        params = defense_params or {}
        for d_name in defense_names:
            d_obj, _ = _build_defence(d_name, clip_values, params)
            if d_name in {"reverse_sigmoid", "class_labels", "high_confidence_masking"}:
                post_defenses.append(d_obj)
            else:
                pre_defenses.append(d_obj)

    pre_arg = pre_defenses if pre_defenses else None
    post_arg = post_defenses if post_defenses else None

    if framework == "sklearn":
        return SklearnClassifier(
            model=loaded_model,
            clip_values=clip_values,
            preprocessing_defences=pre_arg,
            postprocessing_defences=post_arg,
        )

    if framework == "pytorch":
        import torch
        import torch.nn as nn

        class _FloatInputWrapper(nn.Module):
            def __init__(self, inner_model):
                super().__init__()
                self.inner_model = inner_model

            def forward(self, x):
                if isinstance(x, torch.Tensor) and x.dtype != torch.float32:
                    x = x.to(torch.float32)
                return self.inner_model(x)

        wrapped_model = _FloatInputWrapper(loaded_model)
        input_shape, nb_classes = _infer_pytorch_shape_and_classes(loaded_model, X)
        return PyTorchClassifier(
            model=wrapped_model,
            loss=nn.CrossEntropyLoss(),
            input_shape=input_shape,
            nb_classes=nb_classes,
            clip_values=clip_values,
            preprocessing_defences=pre_arg,
            postprocessing_defences=post_arg,
        )

    if framework == "onnx":
        input_shape, nb_classes = _infer_onnx_shape_and_classes(loaded_model)
        return BlackBoxClassifier(
            predict_fn=_onnx_predict_fn(loaded_model, adapter),
            input_shape=input_shape,
            nb_classes=nb_classes,
            clip_values=clip_values,
            preprocessing_defences=pre_arg,
            postprocessing_defences=post_arg,
        )

    raise ValueError(f"Unsupported framework for adversarial attacks: {framework}.")


def _build_attack(attack_name: str, classifier: Any, params: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
    if attack_name == "boundary_attack":
        actual_params = {
            "batch_size": int(params.get("batch_size", 32)),
            "targeted": bool(params.get("targeted", False)),
            "max_iter": int(params.get("max_iter", 10)),
            "init_size": int(params.get("init_size", 20)),
        }
        return BoundaryAttack(estimator=classifier, verbose=False, **actual_params), actual_params

    if attack_name == "hopskipjump":
        actual_params = {
            "batch_size": int(params.get("batch_size", 32)),
            "targeted": bool(params.get("targeted", False)),
            "max_iter": int(params.get("max_iter", 10)),
            "max_eval": int(params.get("max_eval", 200)),
        }
        return HopSkipJump(classifier=classifier, verbose=False, **actual_params), actual_params

    if attack_name == "zoo":
        actual_params = {
            "batch_size": int(params.get("batch_size", 1)),
            "max_iter": int(params.get("max_iter", 10)),
            "targeted": bool(params.get("targeted", False)),
            "nb_parallel": int(params.get("nb_parallel", 1)),
        }
        return ZooAttack(classifier=classifier, verbose=False, **actual_params), actual_params

    if attack_name == "fgsm":
        actual_params = {
            "eps": float(params.get("eps", 0.1)),
            "targeted": bool(params.get("targeted", False)),
        }
        return FastGradientMethod(estimator=classifier, **actual_params), actual_params

    if attack_name == "pgd":
        actual_params = {
            "eps": float(params.get("eps", 0.1)),
            "eps_step": float(params.get("eps_step", 0.01)),
            "max_iter": int(params.get("max_iter", 10)),
            "targeted": bool(params.get("targeted", False)),
        }
        return ProjectedGradientDescent(estimator=classifier, verbose=False, **actual_params), actual_params

    if attack_name == "cw":
        actual_params = {
            "max_iter": int(params.get("max_iter", 10)),
            "targeted": bool(params.get("targeted", False)),
        }
        return CarliniL2Method(classifier=classifier, verbose=False, **actual_params), actual_params

    raise ValueError(f"Unknown evasion attack: {attack_name}")


def _build_defence(
    defense_name: str,
    clip_values: Tuple[float, float],
    params: Dict[str, Any],
) -> Tuple[Any, Dict[str, Any]]:
    if defense_name == "gaussian_augmentation":
        used_params = {"sigma": float(params.get("sigma", 1.0))}
        return GaussianAugmentation(
            sigma=used_params["sigma"],
            augmentation=False,
            clip_values=clip_values,
        ), used_params

    if defense_name == "spatial_smoothing":
        used_params = {"window_size": int(params.get("window_size", 3))}
        return SpatialSmoothing(
            window_size=used_params["window_size"],
            clip_values=clip_values,
        ), used_params

    if defense_name == "feature_squeezing":
        used_params = {"bit_depth": int(params.get("bit_depth", 4))}
        return FeatureSqueezing(
            bit_depth=used_params["bit_depth"],
            clip_values=clip_values,
        ), used_params

    if defense_name == "clip_values":
        used_params = {"clip_values": clip_values}
        return ClipValues(
            clip_values=clip_values,
        ), used_params

    if defense_name == "reverse_sigmoid":
        used_params = {
            "beta": float(params.get("beta", 1.0)),
            "gamma": float(params.get("gamma", 0.1)),
        }
        return ReverseSigmoid(**used_params), used_params

    if defense_name == "class_labels":
        return ClassLabels(), {}

    if defense_name == "high_confidence_masking":
        used_params = {"cutoff": float(params.get("cutoff", 0.25))}
        return HighConfidenceMasking(**used_params), used_params

    raise ValueError(f"Unknown defense: {defense_name}")


#: Gradient attacks need a usable gradient. These defences are applied as a
#: non-differentiable pre-processing step, so the gradient an attacker sees
#: does not match the gradient the model actually consumes.
NON_DIFFERENTIABLE_DEFENSES = {
    "gaussian_augmentation",
    "spatial_smoothing",
    "feature_squeezing",
    "clip_values",
}

GRADIENT_BASED_ATTACKS = {"fgsm", "pgd", "cw"}


def _gradient_masking_cross_check(
    attack_name: str,
    defense_name: str,
    framework: str,
    defended_classifier: Any,
    subset_X: np.ndarray,
    subset_y: np.ndarray,
    loaded_model: Any,
    y: np.ndarray,
    adversarial_accuracy: float,
    params: Dict[str, Any],
) -> Dict[str, Any]:
    """Flag gradient masking and confirm it with a decision-based attack.

    A gradient attack that reports near-perfect accuracy through a
    non-differentiable pre-processor is the classic signature of gradient
    masking: the attack converged to a useless point rather than the model
    being robust. Running a decision-based attack settles the question.
    """
    if attack_name not in GRADIENT_BASED_ATTACKS or defense_name not in NON_DIFFERENTIABLE_DEFENSES:
        return {"suspected": False, "checked": False, "notes": []}

    notes = [
        f"{ATTACK_DISPLAY_NAMES.get(attack_name, attack_name)} is gradient-based and "
        f"{defense_name.replace('_', ' ').title()} is a non-differentiable "
        f"pre-processing step, so the attack's gradient does not reflect the "
        f"model's true decision boundary. Treating this accuracy as a genuine "
        f"robustness gain would be misleading."
    ]

    cross_check: Dict[str, Any] = {
        "suspected": True,
        "checked": False,
        "gradient_attack_accuracy": float(adversarial_accuracy),
        "notes": notes,
    }

    if "boundary_attack" not in EVASION_ATTACKS_BY_FRAMEWORK.get(framework, []) or not subset_X.size:
        cross_check["conclusion"] = (
            "No decision-based cross-check was available for this framework, so the "
            "accuracy above cannot be read as evidence of robustness."
        )
        return cross_check

    try:
        # A small, cheap budget is enough to separate "masked" from "robust".
        check_X, check_y = _stratified_subsample(
            subset_X, subset_y, n=min(10, len(subset_X)), seed=7
        )
        check_attack, _ = _build_attack(
            "boundary_attack",
            defended_classifier,
            {"max_iter": 5},
        )
        x_adv = check_attack.generate(x=check_X)
        if isinstance(x_adv, np.ndarray):
            x_adv = x_adv.astype(np.float32)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            check_preds = defended_classifier.predict(x_adv)
        check_accuracy = float(
            accuracy_score(check_y, _map_predictions(check_preds, loaded_model, y))
        )

        cross_check.update(
            {
                "checked": True,
                "cross_check_attack": "boundary_attack",
                "cross_check_accuracy": check_accuracy,
                "cross_check_sample_count": int(len(check_X)),
            }
        )
        # Masking is confirmed when a decision-based attack is markedly more
        # effective, i.e. the gradient attack was measuring an artefact.
        if check_accuracy <= adversarial_accuracy - 0.05:
            cross_check["confirmed"] = True
            cross_check["conclusion"] = (
                f"Gradient masking confirmed: the decision-based cross-check reduced "
                f"accuracy to {check_accuracy:.1%} versus {adversarial_accuracy:.1%} for "
                f"{ATTACK_DISPLAY_NAMES.get(attack_name, attack_name)}. The model is not as "
                f"robust as the gradient attack suggests."
            )
        else:
            cross_check["confirmed"] = False
            cross_check["conclusion"] = (
                f"Gradient masking not confirmed: a decision-based cross-check reached "
                f"{check_accuracy:.1%}, consistent with the gradient attack's "
                f"{adversarial_accuracy:.1%}."
            )
    except Exception as exc:
        cross_check["conclusion"] = (
            f"Decision-based cross-check could not run: {exc}"
        )

    return cross_check


class RobustnessEvaluator:
    """Measures model sensitivity to Gaussian perturbations and ART evasion attacks/defenses."""

    @staticmethod
    def evaluate_noise_robustness(
        model_path: str,
        dataset_path: str,
        target_column: Optional[str] = None,
        noise_level: float = 0.10,
        task_type: str = "classification"
    ) -> RobustnessMetrics:
        df = dataset_loader.load(dataset_path)
        if not target_column:
            target_column = df.columns[-1]

        y_true = df[target_column].values
        feature_cols = [c for c in df.columns if c != target_column]
        X = df[feature_cols].select_dtypes(include=[np.number]).fillna(0).astype(np.float32)

        preds_clean = model_predictor.predict(model_path, X)
        if task_type == "classification":
            clean_metrics = {
                "accuracy": float(accuracy_score(y_true, preds_clean)),
                "precision_macro": float(precision_score(y_true, preds_clean, average="macro", zero_division=0)),
                "recall_macro": float(recall_score(y_true, preds_clean, average="macro", zero_division=0)),
                "f1_macro": float(f1_score(y_true, preds_clean, average="macro", zero_division=0)),
            }
            orig_score = clean_metrics["accuracy"]
        else:
            orig_score = float(-mean_squared_error(y_true, preds_clean))

        stds = X.std(axis=0).values
        stds[stds == 0] = 1.0
        noise = np.random.normal(0, stds * noise_level, size=X.shape)
        X_perturbed = X + noise

        preds_noisy = model_predictor.predict(model_path, X_perturbed)
        if task_type == "classification":
            perturbed_score = float(accuracy_score(y_true, preds_noisy))
            noisy_metrics = {
                "accuracy": perturbed_score,
                "precision_macro": float(precision_score(y_true, preds_noisy, average="macro", zero_division=0)),
                "recall_macro": float(recall_score(y_true, preds_noisy, average="macro", zero_division=0)),
                "f1_macro": float(f1_score(y_true, preds_noisy, average="macro", zero_division=0)),
            }
            score_drop = orig_score - perturbed_score
            drop_percentage = float((score_drop / orig_score) * 100) if orig_score > 0 else 0.0
        else:
            perturbed_score = float(-mean_squared_error(y_true, preds_noisy))
            score_drop = orig_score - perturbed_score
            drop_percentage = float(abs(score_drop))
            clean_metrics = {"score": orig_score}
            noisy_metrics = {"score": perturbed_score}

        is_sensitive = drop_percentage > 10.0

        return RobustnessMetrics(
            perturbation_type=f"gaussian_noise_{int(noise_level*100)}%",
            noise_level=noise_level,
            original_score=orig_score,
            perturbed_score=perturbed_score,
            score_drop=score_drop,
            drop_percentage=drop_percentage,
            is_sensitive=is_sensitive,
            clean_metrics=clean_metrics,
            noisy_metrics=noisy_metrics,
        )

    @staticmethod
    def _resolve_inputs(model_path: str, dataset_path: str, target_column: Optional[str] = None):
        loaded_model, adapter, framework = model_loader.load_model(model_path)
        df = dataset_loader.load(dataset_path)

        if not target_column:
            target_column = df.columns[-1]

        y = df[target_column].values
        feature_cols = [c for c in df.columns if c != target_column]
        X = df[feature_cols].select_dtypes(include=[np.number]).fillna(0).astype(np.float32).values

        if len(X) == 0:
            raise ValueError(
                "No usable numeric feature data was found for this "
                "model/dataset combination. Check that the dataset "
                "path is correct and contains numeric feature columns."
            )

        if framework in {"pytorch", "onnx"} or float(np.max(np.abs(X))) > 10.0:
            from sklearn.preprocessing import StandardScaler
            X = StandardScaler().fit_transform(X).astype(np.float32)

        clip_min = float(np.min(X)) - 2.0
        clip_max = float(np.max(X)) + 2.0

        dataset_info = {
            "total_rows": int(len(X)),
            "total_features": int(X.shape[1]),
            "target_column": target_column,
            "num_classes": int(len(np.unique(y))),
            "feature_names": feature_cols,
        }

        return loaded_model, adapter, framework, X, y, (clip_min, clip_max), dataset_info

    @staticmethod
    def run_evasion_attack(
        attack_name: str,
        model_path: Optional[str] = None,
        dataset_path: Optional[str] = None,
        target_column: Optional[str] = None,
        **params: Any
    ) -> Dict[str, Any]:
        try:
            if attack_name not in ATTACK_DISPLAY_NAMES:
                raise ValueError(f"Unknown evasion attack: {attack_name}")

            if not model_path or not dataset_path:
                raise ValueError("model_path and dataset_path are required.")

            loaded_model, adapter, framework, X, y, clip_values, dataset_info = RobustnessEvaluator._resolve_inputs(
                model_path, dataset_path, target_column
            )

            allowed = EVASION_ATTACKS_BY_FRAMEWORK.get(framework, [])
            if attack_name not in allowed:
                allowed_labels = ", ".join(ATTACK_DISPLAY_NAMES.get(a, a) for a in allowed) or "none"
                raise ValueError(
                    f"'{ATTACK_DISPLAY_NAMES.get(attack_name, attack_name)}' is not "
                    f"compatible with a {framework} model. Compatible attacks: {allowed_labels}."
                )

            classifier = _build_art_classifier(loaded_model, adapter, framework, X, clip_values)
            attack, used_params = _build_attack(attack_name, classifier, params)

            subset_X, subset_y = _stratified_subsample(X, y, n=30, seed=42)
            x_adv = attack.generate(x=subset_X)
            if isinstance(x_adv, np.ndarray):
                x_adv = x_adv.astype(np.float32)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                preds_clean = classifier.predict(subset_X)
                preds_adv = classifier.predict(x_adv)

            clean_labels = _map_predictions(preds_clean, loaded_model, y)
            adv_labels = _map_predictions(preds_adv, loaded_model, y)

            clean_acc = float(accuracy_score(subset_y, clean_labels))
            adv_acc = float(accuracy_score(subset_y, adv_labels))

            clean_precision = float(precision_score(subset_y, clean_labels, average="macro", zero_division=0))
            clean_recall = float(recall_score(subset_y, clean_labels, average="macro", zero_division=0))
            clean_f1 = float(f1_score(subset_y, clean_labels, average="macro", zero_division=0))

            adv_precision = float(precision_score(subset_y, adv_labels, average="macro", zero_division=0))
            adv_recall = float(recall_score(subset_y, adv_labels, average="macro", zero_division=0))
            adv_f1 = float(f1_score(subset_y, adv_labels, average="macro", zero_division=0))
            accuracy_drop = clean_acc - adv_acc
            drop_percentage = float((accuracy_drop / clean_acc) * 100) if clean_acc > 0 else 0.0

            return {
                "attack_type": attack_name,
                "framework": framework,
                "clean_accuracy": clean_acc,
                "adversarial_accuracy": adv_acc,
                "accuracy_drop": accuracy_drop,
                "drop_percentage": drop_percentage,
                "adversarial_sample_count": int(len(subset_X)),
                "attack_parameters": dict(used_params),
                "dataset_info": dataset_info,
                "sample_size_used": int(len(subset_X)),
                "model_info": _describe_model(loaded_model, framework),
                "clean_precision": clean_precision,
                "clean_recall": clean_recall,
                "clean_f1": clean_f1,
                "adversarial_precision": adv_precision,
                "adversarial_recall": adv_recall,
                "adversarial_f1": adv_f1,
                **used_params,
            }
        except Exception as exc:
            return {
                "attack_type": attack_name,
                "status": "error",
                "error": str(exc),
            }

    @staticmethod
    def run_attack_on_defended_model(
        attack_name: str,
        defense_name: str,
        model_path: Optional[str] = None,
        dataset_path: Optional[str] = None,
        target_column: Optional[str] = None,
        **params: Any
    ) -> Dict[str, Any]:
        try:
            if attack_name not in ATTACK_DISPLAY_NAMES:
                raise ValueError(f"Unknown evasion attack: {attack_name}")
            if defense_name not in DEFENSE_REGISTRY:
                raise ValueError(f"Unknown defense: {defense_name}")
            if not model_path or not dataset_path:
                raise ValueError("model_path and dataset_path are required.")

            loaded_model, adapter, framework, X, y, clip_values, dataset_info = RobustnessEvaluator._resolve_inputs(
                model_path, dataset_path, target_column
            )

            if defense_name == "spatial_smoothing" and (X.ndim < 3 or X.shape[1] < 3):
                raise ValueError(
                    "Spatial Smoothing defense requires image or video data with spatial dimensions. "
                    "It cannot be applied to 1D/2D tabular feature matrices."
                )

            allowed = EVASION_ATTACKS_BY_FRAMEWORK.get(framework, [])
            if attack_name not in allowed:
                allowed_labels = ", ".join(ATTACK_DISPLAY_NAMES.get(a, a) for a in allowed) or "none"
                raise ValueError(
                    f"'{ATTACK_DISPLAY_NAMES.get(attack_name, attack_name)}' is not "
                    f"compatible with a {framework} model. Compatible attacks: {allowed_labels}."
                )

            subset_X, subset_y = _stratified_subsample(X, y, n=30, seed=42)

            # Stage 1: Clean Baseline Classifier
            clean_classifier = _build_art_classifier(loaded_model, adapter, framework, X, clip_values)

            # Stage 2: Defended In-Memory Classifier
            _, defense_parameters = _build_defence(
                defense_name, clip_values, params
            )
            defended_classifier = _build_art_classifier(
                loaded_model, adapter, framework, X, clip_values, defense_names=[defense_name], defense_params=params
            )

            # Stage 3: Post-Attack on Defended Model
            attack, used_params = _build_attack(attack_name, defended_classifier, params)
            x_adv = attack.generate(x=subset_X)
            if isinstance(x_adv, np.ndarray):
                x_adv = x_adv.astype(np.float32)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                preds_clean = clean_classifier.predict(subset_X)
                preds_defended = defended_classifier.predict(subset_X)
                preds_post_attack = defended_classifier.predict(x_adv)

            clean_labels = _map_predictions(preds_clean, loaded_model, y)
            defended_labels = _map_predictions(preds_defended, loaded_model, y)
            post_attack_labels = _map_predictions(preds_post_attack, loaded_model, y)

            clean_accuracy = float(accuracy_score(subset_y, clean_labels))
            defended_accuracy = float(accuracy_score(subset_y, defended_labels))
            adversarial_accuracy = float(accuracy_score(subset_y, post_attack_labels))
            accuracy_drop = defended_accuracy - adversarial_accuracy
            drop_percentage = (
                float((accuracy_drop / defended_accuracy) * 100)
                if defended_accuracy > 0
                else 0.0
            )
            # A negative "drop" is an improvement, not a drop; never report
            # a negative or out-of-range percentage.
            drop_percentage = float(min(100.0, max(0.0, drop_percentage)))

            gradient_masking = _gradient_masking_cross_check(
                attack_name=attack_name,
                defense_name=defense_name,
                framework=framework,
                defended_classifier=defended_classifier,
                subset_X=subset_X,
                subset_y=subset_y,
                loaded_model=loaded_model,
                y=y,
                adversarial_accuracy=adversarial_accuracy,
                params=params,
            )

            result = {
                "attack_type": attack_name,
                "defense_type": defense_name,
                "framework": framework,
                "clean_accuracy": clean_accuracy,
                "clean_precision": float(precision_score(subset_y, clean_labels, average="macro", zero_division=0)),
                "clean_recall": float(recall_score(subset_y, clean_labels, average="macro", zero_division=0)),
                "clean_f1": float(f1_score(subset_y, clean_labels, average="macro", zero_division=0)),
                "defended_accuracy": defended_accuracy,
                "defended_precision": float(precision_score(subset_y, defended_labels, average="macro", zero_division=0)),
                "defended_recall": float(recall_score(subset_y, defended_labels, average="macro", zero_division=0)),
                "defended_f1": float(f1_score(subset_y, defended_labels, average="macro", zero_division=0)),
                "adversarial_accuracy": adversarial_accuracy,
                "adversarial_defended_accuracy": adversarial_accuracy,
                "adversarial_precision": float(precision_score(subset_y, post_attack_labels, average="macro", zero_division=0)),
                "adversarial_recall": float(recall_score(subset_y, post_attack_labels, average="macro", zero_division=0)),
                "adversarial_f1": float(f1_score(subset_y, post_attack_labels, average="macro", zero_division=0)),
                "accuracy_drop": accuracy_drop,
                "drop_percentage": drop_percentage,
                "adversarial_sample_count": int(len(subset_X)),
                "attack_parameters": dict(used_params),
                "defense_parameters": dict(defense_parameters),
                "dataset_info": dataset_info,
                "sample_size_used": int(len(subset_X)),
                "model_info": _describe_model(loaded_model, framework),
                "gradient_masking": gradient_masking,
                **used_params,
            }
            if gradient_masking.get("suspected"):
                result["warnings"] = list(gradient_masking.get("notes", []))
            return result
        except Exception as exc:
            return {
                "attack_type": attack_name,
                "defense_type": defense_name,
                "status": "error",
                "error": str(exc),
            }

    @staticmethod
    def run_attack_defense_sequence(
        combinations: List[Dict[str, Any]],
        model_path: Optional[str] = None,
        dataset_path: Optional[str] = None,
        target_column: Optional[str] = None,
        max_steps: int = 5,
        requested_steps: Optional[int] = None,
        **params: Any
    ) -> Dict[str, Any]:
        """Score N independent attack/defense combinations against one model.

        Every combination is measured against the *same unmodified* model. An
        ART preprocessor defense changes input behaviour only and is never
        written to disk, so defenses do not stack: each row is an independent
        measurement, not a chain.

        An unexpected runtime failure aborts the whole run rather than being
        recorded as a failed row, so the real cause is never buried inside a
        multi-step summary.
        """
        from app.agent import security_aliases as _aliases

        if not combinations:
            raise ValueError(
                "No attack or defense was named. Name at least one attack "
                "or defense to compare."
            )
        if len(combinations) > max_steps:
            raise ValueError(
                f"{len(combinations)} combinations were requested but at most "
                f"{max_steps} can be run. Please choose {max_steps} or fewer."
            )

        # Resolve the framework once, then refuse the whole request up front if
        # any step is impossible -- before spending time on the earlier steps.
        _, _, framework, X, _, _, dataset_info = RobustnessEvaluator._resolve_inputs(
            model_path, dataset_path, target_column
        )

        for combo in combinations:
            attack_name = combo.get("attack_type") or combo.get("attack_name")
            defense_name = combo.get("defense_type") or combo.get("defense_name")
            if attack_name:
                _aliases.assert_attack_supported(attack_name, framework)
            if defense_name == "spatial_smoothing" and (X.ndim < 3 or X.shape[1] < 3):
                raise ValueError(
                    "Spatial Smoothing defense requires image or video data with "
                    "spatial dimensions. It cannot be applied to 1D/2D tabular "
                    "feature matrices."
                )

        steps: List[Dict[str, Any]] = []
        for index, combo in enumerate(combinations, start=1):
            attack_name = combo.get("attack_type") or combo.get("attack_name")
            defense_name = combo.get("defense_type") or combo.get("defense_name")
            step_params = {**params, **(combo.get("params") or {})}

            if attack_name and defense_name:
                outcome = RobustnessEvaluator.run_attack_on_defended_model(
                    attack_name=attack_name,
                    defense_name=defense_name,
                    model_path=model_path,
                    dataset_path=dataset_path,
                    target_column=target_column,
                    **step_params,
                )
            elif attack_name:
                outcome = RobustnessEvaluator.run_evasion_attack(
                    attack_name=attack_name,
                    model_path=model_path,
                    dataset_path=dataset_path,
                    target_column=target_column,
                    **step_params,
                )
                outcome.setdefault("defense_type", None)
            else:
                outcome = RobustnessEvaluator.run_defense(
                    defense_name=defense_name,
                    model_path=model_path,
                    dataset_path=dataset_path,
                    target_column=target_column,
                    **step_params,
                )
                outcome.setdefault("attack_type", None)

            if outcome.get("status") == "error":
                raise ValueError(outcome.get("error") or "The step could not be run.")

            outcome["step_index"] = index
            outcome.setdefault("attack_type", attack_name)
            outcome.setdefault("defense_type", defense_name)
            outcome["status"] = "success"
            steps.append(outcome)

        def _mean(values: List[float]) -> float:
            return float(sum(values) / len(values)) if values else 0.0

        by_attack: Dict[str, List[float]] = {}
        by_defense: Dict[str, List[float]] = {}
        for step in steps:
            adv = float(step.get("adversarial_accuracy") or 0.0)
            defended = float(step.get("defended_accuracy") or adv)
            if step.get("attack_type"):
                by_attack.setdefault(step["attack_type"], []).append(adv)
            if step.get("defense_type"):
                by_defense.setdefault(step["defense_type"], []).append(defended)

        ranking = {
            "strongest_attack": None,
            "best_defense": None,
        }
        if by_attack:
            name = min(by_attack, key=lambda k: _mean(by_attack[k]))
            ranking["strongest_attack"] = {
                "attack_type": name,
                "mean_adversarial_accuracy": _mean(by_attack[name]),
            }
        if by_defense:
            name = max(by_defense, key=lambda k: _mean(by_defense[k]))
            ranking["best_defense"] = {
                "defense_type": name,
                "mean_defended_accuracy": _mean(by_defense[name]),
            }

        sample = steps[0]
        return {
            "status": "success",
            "mode": "sweep",
            "framework": framework,
            "dataset_info": dataset_info,
            "sample_size_used": sample.get("sample_size_used"),
            "requested_steps": requested_steps if requested_steps is not None else len(combinations),
            "step_count": len(steps),
            "max_steps": max_steps,
            "steps": steps,
            "ranking": ranking,
            "note": (
                "Each row is an independent measurement against the same model. "
                "The defense is an input preprocessor and is not written back, so "
                "these combinations do not stack."
            ),
        }

    @staticmethod
    def run_hardening_loop(
        combinations: List[Dict[str, Any]],
        model_path: Optional[str] = None,
        dataset_path: Optional[str] = None,
        target_column: Optional[str] = None,
        retrain_method: str = "pgd",
        max_rounds: int = 5,
        requested_rounds: Optional[int] = None,
        project_id: Optional[str] = None,
        model_id: Optional[str] = None,
        **params: Any
    ) -> Dict[str, Any]:
        """Attack, adversarially retrain, and feed the new model to the next round.

        Each round genuinely accumulates: the model scored in round N+1 is the
        one produced by round N's retraining. Intermediate rounds retrain with
        ``persist=False`` so only the final model becomes a registered asset.

        The attack measured and the retraining method are separate concepts; both
        are validated up front so an impossible loop is refused before any
        expensive work starts.
        """
        from app.agent import security_aliases as _aliases
        from app.models.trainer import adversarial_retrainer as _retrainer

        if not combinations:
            raise ValueError(
                "No attack was named. Name at least one attack to harden against."
            )
        if len(combinations) > max_rounds:
            raise ValueError(
                f"{len(combinations)} hardening rounds were requested but at most "
                f"{max_rounds} can be run. Please choose {max_rounds} or fewer."
            )

        _, _, framework, _, _, _, dataset_info = RobustnessEvaluator._resolve_inputs(
            model_path, dataset_path, target_column
        )

        _aliases.assert_hardening_method_supported(retrain_method, framework)
        for combo in combinations:
            if combo.get("attack_type"):
                _aliases.assert_attack_supported(combo["attack_type"], framework)

        current_path = model_path
        original_id = model_id
        rounds: List[Dict[str, Any]] = []

        for index, combo in enumerate(combinations, start=1):
            attack_name = combo.get("attack_type") or combo.get("attack_name")
            defense_name = combo.get("defense_type") or combo.get("defense_name")

            # Score the model produced by the previous round.
            if attack_name and defense_name:
                metrics = RobustnessEvaluator.run_attack_on_defended_model(
                    attack_name=attack_name,
                    defense_name=defense_name,
                    model_path=current_path,
                    dataset_path=dataset_path,
                    target_column=target_column,
                    **params,
                )
            elif attack_name:
                metrics = RobustnessEvaluator.run_evasion_attack(
                    attack_name=attack_name,
                    model_path=current_path,
                    dataset_path=dataset_path,
                    target_column=target_column,
                    **params,
                )
                metrics.setdefault("defense_type", None)
            else:
                metrics = RobustnessEvaluator.run_defense(
                    defense_name=defense_name,
                    model_path=current_path,
                    dataset_path=dataset_path,
                    target_column=target_column,
                    **params,
                )
                metrics.setdefault("attack_type", None)

            if metrics.get("status") == "error":
                raise ValueError(
                    metrics.get("error") or "The hardening round could not be run."
                )

            # Adversarial retraining writes the next round's model.
            retrained = _retrainer.harden_and_save(
                model_path=current_path,
                dataset_path=dataset_path,
                target_column=target_column,
                method=retrain_method,
                project_id=project_id,
                model_id=original_id,
                persist=index == len(combinations),
            )

            entry = {
                "round_index": index,
                "is_final": index == len(combinations),
                "attack_type": attack_name,
                "defense_type": defense_name,
                "retrain_method": retrain_method,
                "clean_accuracy": metrics.get("clean_accuracy"),
                "adversarial_accuracy": metrics.get("adversarial_accuracy"),
                "adversarial_precision": metrics.get("adversarial_precision"),
                "adversarial_recall": metrics.get("adversarial_recall"),
                "adversarial_f1": metrics.get("adversarial_f1"),
                "drop_percentage": metrics.get("drop_percentage"),
                "gradient_masking": metrics.get("gradient_masking"),
                "retrain_summary": retrained.get("training_summary"),
                "model_id": retrained.get("model_id"),
                "file_path": retrained.get("file_path"),
            }
            rounds.append(entry)

            if index < len(combinations):
                current_path = retrained["file_path"]

        first = rounds[0].get("adversarial_accuracy") or 0.0
        final = rounds[-1].get("adversarial_accuracy") or 0.0

        # A first-to-final delta only measures hardening when every round
        # measured the same attack. With a mixed-threat curriculum the two ends
        # are different attacks, so the delta is reported per attack instead of
        # being presented as one model's improvement.
        measured_attacks = {r.get("attack_type") for r in rounds}
        mixed_threat = len(measured_attacks) > 1
        per_attack_change: Dict[str, Any] = {}
        for name in sorted(a for a in measured_attacks if a):
            seen = [
                r.get("adversarial_accuracy") or 0.0
                for r in rounds
                if r.get("attack_type") == name
            ]
            per_attack_change[name] = {
                "rounds": len(seen),
                "first_adversarial_accuracy": seen[0],
                "last_adversarial_accuracy": seen[-1],
                "delta": float(seen[-1] - seen[0]),
            }

        return {
            "status": "success",
            "mode": "hardening_loop",
            "framework": framework,
            "dataset_info": dataset_info,
            "sample_size_used": rounds[0].get("sample_size_used"),
            "retrain_method": retrain_method,
            "requested_rounds": (
                requested_rounds if requested_rounds is not None else len(combinations)
            ),
            "round_count": len(rounds),
            "max_rounds": max_rounds,
            "rounds": rounds,
            "mixed_threat": mixed_threat,
            "attacks_measured": sorted(a for a in measured_attacks if a),
            "per_attack_change": per_attack_change,
            "final_model_id": rounds[-1].get("model_id"),
            "final_model_path": rounds[-1].get("file_path"),
            "improvement": {
                "first_adversarial_accuracy": first,
                "final_adversarial_accuracy": final,
                "delta": float(final - first),
                "comparable": not mixed_threat,
            },
        }

    @staticmethod
    def run_boundary_attack(model_path=None, dataset_path=None, target_column=None, **kwargs: Any) -> Dict[str, Any]:
        return RobustnessEvaluator.run_evasion_attack(
            "boundary_attack", model_path=model_path, dataset_path=dataset_path,
            target_column=target_column, **kwargs
        )

    @staticmethod
    def run_hopskipjump_attack(model_path=None, dataset_path=None, target_column=None, **kwargs: Any) -> Dict[str, Any]:
        return RobustnessEvaluator.run_evasion_attack(
            "hopskipjump", model_path=model_path, dataset_path=dataset_path,
            target_column=target_column, **kwargs
        )

    @staticmethod
    def run_defense(
        defense_name: str,
        model_path: Optional[str] = None,
        dataset_path: Optional[str] = None,
        target_column: Optional[str] = None,
        **params: Any
    ) -> Dict[str, Any]:
        try:
            if defense_name not in DEFENSE_REGISTRY:
                raise ValueError(f"Unknown defense: {defense_name}")

            if not model_path or not dataset_path:
                raise ValueError("model_path and dataset_path are required.")

            loaded_model, adapter, framework, X, y, clip_values, dataset_info = RobustnessEvaluator._resolve_inputs(
                model_path, dataset_path, target_column
            )

            if defense_name == "spatial_smoothing" and (X.ndim < 3 or X.shape[1] < 3):
                raise ValueError(
                    "Spatial Smoothing defense requires image or video data with spatial dimensions. "
                    "It cannot be applied to 1D/2D tabular feature matrices."
                )

            classifier = _build_art_classifier(loaded_model, adapter, framework, X, clip_values)
            defence, used_params = _build_defence(defense_name, clip_values, params)

            subset_X, subset_y = _stratified_subsample(X, y, n=30, seed=42)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                preds_clean = classifier.predict(subset_X)

                if defense_name in {"reverse_sigmoid", "class_labels", "high_confidence_masking"}:
                    preds_defended = defence(preds_clean)
                else:
                    x_defended, _ = defence(subset_X, subset_y)
                    if isinstance(x_defended, np.ndarray):
                        x_defended = x_defended.astype(np.float32)
                    preds_defended = classifier.predict(x_defended)

            clean_labels = _map_predictions(preds_clean, loaded_model, y)
            defended_labels = _map_predictions(preds_defended, loaded_model, y)

            clean_acc = float(accuracy_score(subset_y, clean_labels))
            defended_acc = float(accuracy_score(subset_y, defended_labels))

            clean_precision = float(precision_score(subset_y, clean_labels, average="macro", zero_division=0))
            clean_recall = float(recall_score(subset_y, clean_labels, average="macro", zero_division=0))
            clean_f1 = float(f1_score(subset_y, clean_labels, average="macro", zero_division=0))

            defended_precision = float(precision_score(subset_y, defended_labels, average="macro", zero_division=0))
            defended_recall = float(recall_score(subset_y, defended_labels, average="macro", zero_division=0))
            defended_f1 = float(f1_score(subset_y, defended_labels, average="macro", zero_division=0))

            return {
                "defense_type": defense_name,
                "framework": framework,
                "clean_accuracy": clean_acc,
                "defended_accuracy": defended_acc,
                "accuracy_change": defended_acc - clean_acc,
                "dataset_info": dataset_info,
                "sample_size_used": int(len(subset_X)),
                "model_info": _describe_model(loaded_model, framework),
                "clean_precision": clean_precision,
                "clean_recall": clean_recall,
                "clean_f1": clean_f1,
                "defended_precision": defended_precision,
                "defended_recall": defended_recall,
                "defended_f1": defended_f1,
                **used_params,
            }
        except Exception as exc:
            return {
                "defense_type": defense_name,
                "status": "error",
                "error": str(exc),
            }

    @staticmethod
    def gaussian_augmentation_defense(model_path=None, dataset_path=None, target_column=None, **kwargs: Any) -> Dict[str, Any]:
        return RobustnessEvaluator.run_defense(
            "gaussian_augmentation", model_path=model_path, dataset_path=dataset_path,
            target_column=target_column, **kwargs
        )

    @staticmethod
    def clip_values_defense(model_path=None, dataset_path=None, target_column=None, **kwargs: Any) -> Dict[str, Any]:
        return RobustnessEvaluator.run_defense(
            "clip_values", model_path=model_path, dataset_path=dataset_path,
            target_column=target_column, **kwargs
        )

    @staticmethod
    def reverse_sigmoid_defense(model_path=None, dataset_path=None, target_column=None, **kwargs: Any) -> Dict[str, Any]:
        return RobustnessEvaluator.run_defense(
            "reverse_sigmoid", model_path=model_path, dataset_path=dataset_path,
            target_column=target_column, **kwargs
        )

    @staticmethod
    def class_labels_defense(model_path=None, dataset_path=None, target_column=None, **kwargs: Any) -> Dict[str, Any]:
        return RobustnessEvaluator.run_defense(
            "class_labels", model_path=model_path, dataset_path=dataset_path,
            target_column=target_column, **kwargs
        )

    @staticmethod
    def high_confidence_masking_defense(model_path=None, dataset_path=None, target_column=None, **kwargs: Any) -> Dict[str, Any]:
        return RobustnessEvaluator.run_defense(
            "high_confidence_masking", model_path=model_path, dataset_path=dataset_path,
            target_column=target_column, **kwargs
        )


robustness_evaluator = RobustnessEvaluator()

