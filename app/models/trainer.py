# =====================================================================
# Updated trainer_3.py
# =====================================================================
import os
import re
import logging
import tempfile
import joblib
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

from app.core.storage import storage_manager
from app.services.project_service import project_service
from app.models.loader import model_loader
from app.evaluation.robustness import (
    RobustnessEvaluator,
    _stratified_subsample,
    _build_art_classifier,
    _build_attack,
)

logger = logging.getLogger(__name__)

# "hardened_20260930_082121_149781_" - the marker prepend()ed on every hardening.
_HARDENED_PREFIX = re.compile(r"^hardened_\d{8}_\d{6}_\d{6}_")

# Windows caps a path near 260 characters, and a hardening loop rebuilds the
# name from the previous one. Without a ceiling the name grows ~32 characters
# per round and every export fallback eventually fails with a path error that
# surfaces as "Could not export hardened PyTorch model". This leaves ample room
# for the storage root, the model directory and the extension.
_MAX_STEM_CHARS = 60


def _bounded_hardened_stem(stem: str) -> tuple:
    """Split a model stem into a reusable base name and its hardening depth.

    Hardening an already-hardened model used to nest the marker, so the fifth
    round of a loop started from a hardened asset produced a name several times
    longer than the original. Stripping every existing marker and re-applying
    exactly one keeps the name a fixed length no matter how deep the chain gets,
    while the depth is still reported so a re-hardened model stays
    distinguishable in listings.
    """
    depth = 0
    base = stem
    while True:
        stripped = _HARDENED_PREFIX.sub("", base, count=1)
        if stripped == base:
            break
        base = stripped
        depth += 1
    return base[:_MAX_STEM_CHARS], depth


class AdversarialRetrainer:
    """Multi-framework adversarial retrainer engine for model hardening."""

    @staticmethod
    def _resolve_method(method: Optional[str], supported: tuple) -> str:
        """Return the requested method, refusing anything unsupported.

        Substituting a different attack would silently produce a hardened model
        trained against something the user never asked for.
        """
        requested = (method or "").strip().lower()
        if not requested:
            return supported[0]
        if requested not in supported:
            raise ValueError(
                f"Adversarial retraining method '{requested}' is not supported here. "
                f"Supported methods: {', '.join(supported)}."
            )
        return requested

    def harden_pytorch(
        self,
        model_path: str,
        dataset_path: str,
        target_column: Optional[str] = None,
        method: str = "pgd",
        output_path: Optional[str] = None,
        **kwargs: Any
    ) -> Dict[str, Any]:
        import torch
        import torch.nn as nn
        from torch.utils.data import TensorDataset, DataLoader

        loaded_model, adapter, framework, X, y, clip_values, dataset_info = RobustnessEvaluator._resolve_inputs(
            model_path, dataset_path, target_column
        )

        sample_size = int(kwargs.get("sample_size", 30))
        sub_X, sub_y = _stratified_subsample(X, y, n=sample_size, seed=42)

        attack_name = self._resolve_method(method, ("fgsm", "pgd"))
        classifier = _build_art_classifier(loaded_model, adapter, framework, X, clip_values)
        attack_params = {"max_iter": 5, "eps": float(kwargs.get("eps", 0.1))}
        attack, used_params = _build_attack(attack_name, classifier, attack_params)

        x_adv = attack.generate(x=sub_X).astype(np.float32)

        X_train = np.vstack([sub_X, x_adv]).astype(np.float32)
        y_train = np.concatenate([sub_y, sub_y])

        unique_classes = np.unique(y)
        class_to_idx = {c: i for i, c in enumerate(unique_classes)}
        y_train_idx = np.array([class_to_idx.get(val, 0) for val in y_train], dtype=np.int64)

        dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train_idx))
        loader = DataLoader(dataset, batch_size=int(kwargs.get("batch_size", 16)), shuffle=True)

        epochs = int(kwargs.get("epochs", 3))
        criterion = nn.CrossEntropyLoss()

        if hasattr(loaded_model, "parameters"):
            try:
                loaded_model.train()
                optimizer = torch.optim.Adam(loaded_model.parameters(), lr=float(kwargs.get("lr", 1e-3)))
                for _ in range(epochs):
                    for bx, by in loader:
                        optimizer.zero_grad()
                        out = loaded_model(bx)
                        loss = criterion(out, by)
                        loss.backward()
                        optimizer.step()
                loaded_model.eval()
            except Exception as exc:
                raise RuntimeError("PyTorch adversarial retraining failed") from exc

        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            saved = False
            try:
                example_input = torch.from_numpy(sub_X[:1])
                scripted = torch.jit.trace(loaded_model, example_input)
                torch.jit.save(scripted, str(output_path))
                saved = True
            except Exception:
                pass

            if not saved:
                try:
                    scripted = torch.jit.script(loaded_model)
                    torch.jit.save(scripted, str(output_path))
                    saved = True
                except Exception:
                    pass

            if not saved:
                try:
                    torch.save(loaded_model, str(output_path))
                except Exception as exc:
                    raise RuntimeError("Could not export hardened PyTorch model") from exc

        return {
            "framework": "pytorch",
            "method": attack_name,
            "samples_generated": len(x_adv),
            "epochs": epochs,
            "dataset_info": dataset_info,
            **used_params,
        }

    def harden_sklearn(
        self,
        model_path: str,
        dataset_path: str,
        target_column: Optional[str] = None,
        method: str = "hopskipjump",
        output_path: Optional[str] = None,
        **kwargs: Any
    ) -> Dict[str, Any]:
        from sklearn.base import clone

        loaded_model, adapter, framework, X, y, clip_values, dataset_info = RobustnessEvaluator._resolve_inputs(
            model_path, dataset_path, target_column
        )

        sample_size = int(kwargs.get("sample_size", 30))
        sub_X, sub_y = _stratified_subsample(X, y, n=sample_size, seed=42)

        attack_name = self._resolve_method(method, ("boundary_attack", "hopskipjump"))
        classifier = _build_art_classifier(loaded_model, adapter, framework, X, clip_values)
        attack_params = {"max_iter": 5}
        attack, used_params = _build_attack(attack_name, classifier, attack_params)

        x_adv = attack.generate(x=sub_X).astype(np.float32)

        X_hardened = np.vstack([X, x_adv])
        y_hardened = np.concatenate([y, sub_y])

        try:
            retrained_model = clone(loaded_model)
        except Exception:
            retrained_model = loaded_model

        retrained_model.fit(X_hardened, y_hardened)

        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            joblib.dump(retrained_model, str(output_path))

        return {
            "framework": "sklearn",
            "method": attack_name,
            "samples_generated": len(x_adv),
            "dataset_info": dataset_info,
            **used_params,
        }

    def harden_onnx(
        self,
        model_path: str,
        dataset_path: str,
        target_column: Optional[str] = None,
        method: str = "hopskipjump",
        output_path: Optional[str] = None,
        **kwargs: Any
    ) -> Dict[str, Any]:
        from sklearn.ensemble import RandomForestClassifier

        loaded_model, adapter, framework, X, y, clip_values, dataset_info = RobustnessEvaluator._resolve_inputs(
            model_path, dataset_path, target_column
        )

        sample_size = int(kwargs.get("sample_size", 30))
        sub_X, sub_y = _stratified_subsample(X, y, n=sample_size, seed=42)

        attack_name = self._resolve_method(method, ("boundary_attack", "hopskipjump"))
        classifier = _build_art_classifier(loaded_model, adapter, framework, X, clip_values)
        attack_params = {"max_iter": 5}
        attack, used_params = _build_attack(attack_name, classifier, attack_params)

        x_adv = attack.generate(x=sub_X).astype(np.float32)

        X_hardened = np.vstack([X, x_adv])
        y_hardened = np.concatenate([y, sub_y])

        estimator = RandomForestClassifier(n_estimators=50, random_state=42)
        estimator.fit(X_hardened, y_hardened)

        export_status = "not_exported"
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            try:
                from skl2onnx import convert_sklearn
                from skl2onnx.common.data_types import FloatTensorType
                initial_type = [("float_input", FloatTensorType([None, X_hardened.shape[1]]))]
                onx = convert_sklearn(estimator, initial_types=initial_type)
                with open(str(output_path), "wb") as f:
                    f.write(onx.SerializeToString())
            except Exception as exc:
                # Never register a copy of the original as "hardened".
                if os.path.exists(output_path):
                    os.remove(output_path)
                raise RuntimeError("Could not export hardened ONNX model") from exc
            export_status = "success"

        return {
            "framework": "onnx",
            "method": attack_name,
            "export_status": export_status,
            "samples_generated": len(x_adv),
            "dataset_info": dataset_info,
            **used_params,
        }

    def harden_and_save(
        self,
        model_path: Optional[str] = None,
        dataset_path: Optional[str] = None,
        target_column: Optional[str] = None,
        method: str = "pgd",
        project_id: Optional[str] = None,
        model_id: Optional[str] = None,
        persist: bool = True,
        **kwargs: Any
    ) -> Dict[str, Any]:
        if not model_path:
            raise ValueError("model_path is required for model hardening")
        if not dataset_path:
            raise ValueError("dataset_path is required for model hardening")

        resolved_model_path = str(storage_manager.resolve_path(model_path))
        resolved_dataset_path = str(storage_manager.resolve_path(dataset_path))

        _, _, framework = model_loader.load_model(resolved_model_path)

        proj_id = project_id or "default"
        stem, harden_depth = _bounded_hardened_stem(Path(resolved_model_path).stem)
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
        hardened_model_id = f"model_hardened_{timestamp}"

        if persist:
            model_dir = storage_manager.get_model_dir(hardened_model_id)
        else:
            # Intermediate loop rounds must not create a stray storage/models
            # directory per round; only the final model is a real asset.
            scratch = Path(tempfile.gettempdir()) / "adversarial_hardening_loop"
            scratch.mkdir(parents=True, exist_ok=True)
            model_dir = scratch / hardened_model_id
            model_dir.mkdir(parents=True, exist_ok=True)

        ext_map = {"pytorch": ".pt", "sklearn": ".joblib", "onnx": ".onnx"}
        ext = ext_map.get(framework, ".joblib")
        filename = f"hardened_{timestamp}_{stem}{ext}"
        target_output_path = model_dir / filename
        # Belt and braces: the stem is already bounded, so this only trips on a
        # pathologically deep storage root. Failing here with a clear message
        # beats a cryptic export error from deep inside torch.
        if len(str(target_output_path)) >= 240:
            raise RuntimeError(
                "The hardened model path would be too long for this system "
                f"({len(str(target_output_path))} characters). Move the storage "
                "directory to a shorter path and try again."
            )
        if framework == "pytorch":
            result = self.harden_pytorch(
                resolved_model_path, resolved_dataset_path, target_column, method=method, output_path=str(target_output_path), **kwargs
            )
        elif framework == "onnx":
            result = self.harden_onnx(
                resolved_model_path, resolved_dataset_path, target_column, method=method, output_path=str(target_output_path), **kwargs
            )
        else:
            result = self.harden_sklearn(
                resolved_model_path, resolved_dataset_path, target_column, method=method, output_path=str(target_output_path), **kwargs
            )

        if not target_output_path.is_file() or target_output_path.stat().st_size == 0:
            raise RuntimeError("Hardened model artifact was not created")
        try:
            model_loader.load_model(str(target_output_path))
        except Exception as exc:
            raise RuntimeError("Hardened model artifact could not be loaded") from exc

        # Intermediate rounds of a hardening loop retrain with persist=False so
        # only the final model becomes a registered asset.
        if not persist:
            return {
                "status": "success",
                "operation": "harden_model",
                "model_id": None,
                "model_name": None,
                "file_path": str(target_output_path),
                "framework": framework,
                "persisted": False,
                "training_summary": result,
            }

        orig_id = model_id or stem
        # The bounded stem means a re-hardened model would otherwise carry the
        # same name as the first one, so the depth keeps them distinguishable.
        depth_suffix = f" r{harden_depth}" if harden_depth else ""
        asset_info = project_service.register_hardened_model_asset(
            project_id=proj_id,
            file_path=str(target_output_path),
            original_model_id=orig_id,
            framework=framework,
            model_name=f"{stem} [Hardened{depth_suffix}]"
        )

        return {
            "status": "success",
            "operation": "harden_model",
            "model_id": asset_info["model_id"],
            "model_name": asset_info["name"],
            "file_path": asset_info["file_path"],
            "framework": framework,
            "training_summary": result,
        }


adversarial_retrainer = AdversarialRetrainer()
