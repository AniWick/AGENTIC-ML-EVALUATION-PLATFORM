from typing import Dict, Any, List
from app.schemas.evaluation import PerformanceMetrics, ComparisonResult

class EvaluationComparator:
    """Compares baseline vs experiment performance programmatically."""

    @staticmethod
    def format_summary_line(
        model_name: str,
        dataset_name: str,
        operation: str,
        metrics: Dict[str, Any]
    ) -> str:
        """Formats experiment metrics into a single structured line."""
        acc = metrics.get("accuracy", metrics.get("adversarial_accuracy", metrics.get("defended_accuracy", 0.0)))
        rec = metrics.get("recall_macro", metrics.get("recall", metrics.get("adversarial_recall", metrics.get("defended_recall", 0.0))))
        f1 = metrics.get("f1_macro", metrics.get("f1", metrics.get("adversarial_f1", metrics.get("defended_f1", 0.0))))
        prec = metrics.get("precision_macro", metrics.get("precision", metrics.get("adversarial_precision", metrics.get("defended_precision", 0.0))))

        return (
            f"model: {model_name}, "
            f"dataset: {dataset_name}, "
            f"operation: {operation}, "
            f"accuracy: {acc:.4f}, "
            f"recall: {rec:.4f}, "
            f"f1 score: {f1:.4f}, "
            f"precision: {prec:.4f}"
        )

    @staticmethod
    def compare(
        baseline_version: str,
        experiment_version: str,
        baseline_metrics: Dict[str, Any],
        experiment_metrics: Dict[str, Any]
    ) -> ComparisonResult:
        deltas: Dict[str, float] = {}
        summary: List[str] = []

        keys_to_compare = ["accuracy", "f1_macro", "f1_minority", "precision_macro", "recall_macro", "r2_score", "mae", "rmse"]

        for k in keys_to_compare:
            b_val = baseline_metrics.get(k)
            e_val = experiment_metrics.get(k)
            if b_val is not None and e_val is not None:
                delta = float(e_val - b_val)
                deltas[k] = round(delta, 4)
                direction = "increased" if delta > 0 else ("decreased" if delta < 0 else "remained unchanged")
                pct = abs(delta) * 100
                summary.append(f"{k.replace('_', ' ').title()}: {b_val:.3f} -> {e_val:.3f} ({direction} by {pct:.1f}%)")

        f1_delta = deltas.get("f1_macro", 0.0)
        min_f1_delta = deltas.get("f1_minority", 0.0)
        acc_delta = deltas.get("accuracy", 0.0)

        is_improved = (f1_delta > 0.01 or min_f1_delta > 0.03 or (acc_delta > 0.01 and f1_delta >= -0.01))

        return ComparisonResult(
            baseline_version=baseline_version,
            experiment_version=experiment_version,
            baseline_metrics=baseline_metrics,
            experiment_metrics=experiment_metrics,
            metric_deltas=deltas,
            summary_of_changes=summary,
            is_improved=is_improved
        )

evaluation_comparator = EvaluationComparator()