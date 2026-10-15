import json
from typing import Dict, Any, Optional

from app.agent.state import AgentState
from app.agent.tools import tool_registry
from app.llm.qwen_client import qwen_client
from app.llm.prompts import AGENT_SYSTEM_PROMPT
from app.datasets.versioning import dataset_version_manager
from app.agent.security_aliases import (
    IncompatibleSecurityRequest,
    assert_attack_supported,
    assert_hardening_method_supported,
    extract_attack,
    extract_defense,
    extract_hardening_method,
    extract_attacks,
    extract_defenses,
    extract_explicit_step_count,
)
import re


# Metric fields carried per step/round into the persisted experiment series.
# The chart reads these names, so they double as the on-disk contract.
_SERIES_METRIC_FIELDS = (
    "clean_accuracy",
    "clean_precision",
    "clean_recall",
    "clean_f1",
    "defended_accuracy",
    "defended_precision",
    "defended_recall",
    "defended_f1",
    "adversarial_accuracy",
    "adversarial_precision",
    "adversarial_recall",
    "adversarial_f1",
    "drop_percentage",
)


def build_series(entries: list, kind: str) -> dict:
    """Per-step or per-round metrics, ready to persist and chart.

    Values are copied straight from the evaluator output rather than recomputed,
    so the chart cannot disagree with the report. ``kind`` is ``"sweep"`` or
    ``"loop"`` and the index field is named accordingly, since rounds are
    cumulative and steps are not.
    """
    index_key = "step_index" if kind == "sweep" else "round_index"
    series = []
    for position, entry in enumerate(entries or [], start=1):
        row = {index_key: entry.get(index_key) or position}
        for field in ("attack_type", "defense_type", "retrain_method", "model_id"):
            if entry.get(field) is not None:
                row[field] = entry.get(field)
        for field in _SERIES_METRIC_FIELDS:
            if entry.get(field) is not None:
                row[field] = entry.get(field)
        series.append(row)

    payload: Dict[str, Any] = {
        "kind": kind,
        "count": len(series),
        "items": series,
    }
    if kind == "loop":
        attacks = [r.get("attack_type") for r in series if r.get("attack_type")]
        payload["mixed_threat"] = len(set(attacks)) > 1
        payload["attacks_measured"] = sorted(set(attacks))
    return payload


class Decision:
    def __init__(
        self,
        action: str,
        parameters: Dict[str, Any],
        reason: str,
        status: str = "continue",
    ):
        self.action = action
        self.parameters = parameters
        self.reason = reason
        self.status = status

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "action": self.action if self.action else None,
            "parameters": self.parameters,
            "reason": self.reason,
        }


class AgentPlanner:
    """
    Workflow-aware state-gated planner[cite: 9].

    Qwen recommends the next action, but Python is authoritative[cite: 9].
    """

    # ------------------------------------------------------------------
    # Main planner
    # ------------------------------------------------------------------
    @staticmethod
    def _extract_numeric_params(text: str, keys: list) -> Dict[str, Any]:
        text = (text or "").lower()
        found = {}
        for key in keys:
            if key == "eps":
                m = re.search(r"(?:eps|epsilon)\s*[:=]?\s*([0-9]*\.?[0-9]+)", text)
            else:
                m = re.search(rf"{key}\s*[:=]?\s*([0-9]*\.?[0-9]+)", text)
            if m:
                val = float(m.group(1))
                found[key] = int(val) if key in {"max_iter", "batch_size", "init_size", "max_eval", "window_size", "bit_depth"} else val
        if "targeted" in text:
            found["targeted"] = True
        return found

    @staticmethod
    def _resolve_original_dataset_path(state: AgentState) -> str:
        """Attacks/defenses must always evaluate the raw v001 dataset,
        never a cleaned/scaled/balanced downstream version -- otherwise
        results reflect transformed data, not the model's real input[cite: 9]."""
        try:
            v001_path = dataset_version_manager.get_version_path(state.dataset_id, "v001")
            if v001_path:
                return v001_path
        except Exception:
            pass
        return state.dataset_path

    @staticmethod
    def decide_next_step(state: AgentState) -> Decision:

        # Python is always authoritative about completion[cite: 9].
        if (
            state.workflow_complete()
            and state.requested_workflow
            not in {
                "full_evaluation",
                "improve_dataset",
                "attack",
                "evasion",
                "defend",
                "defense",
                # These workflows need to continue past workflow_complete()[cite: 9]
                "defended_attack",
                "attack_defended_model",
                "harden_model",
                "harden",
                "hardening_loop",
                "harden_defense_loop",
            }
        ):
            return Decision(
                action="",
                parameters={},
                reason=f"The requested workflow '{state.requested_workflow}' is complete.",
                status="complete",
            )

        # =================================================================
        # Attack / defend trapdoor
        # Only intercept the requested ART tool, and only if it is not done[cite: 9].
        # =================================================================
        user_goal = (state.user_goal or "").lower().strip()
        workflow = (state.requested_workflow or "").lower().strip()

        # Compound security operations have strict legal ordering and
        # metrics dependencies. They must not be delegated to an LLM[cite: 9].
        if workflow in {
            "defended_attack",
            "attack_defended_model",
            "harden_model",
            "harden",
            "hardening_loop",
            "harden_defense_loop",
        }:
            return AgentPlanner._deterministic_fallback_plan(state)

        intercept_tool = None

        if workflow in {
            "attack",
            "evasion",
            "adversarial_attack",
            "boundary_attack",
            "hopskipjump",
            "zoo",
            "fgsm",
            "pgd",
            "cw",
        } and not state.adversarial_evaluated:
            if workflow in {"boundary_attack", "hopskipjump", "zoo", "fgsm", "pgd", "cw"}:
                intercept_tool = workflow
            elif "hop" in user_goal:
                intercept_tool = "hopskipjump"
            elif "zoo" in user_goal:
                intercept_tool = "zoo"
            elif "fgsm" in user_goal or "fast gradient" in user_goal:
                intercept_tool = "fgsm"
            elif "pgd" in user_goal or "projected gradient" in user_goal:
                intercept_tool = "pgd"
            elif "cw" in user_goal or "carlini" in user_goal:
                intercept_tool = "cw"
            else:
                intercept_tool = "boundary_attack"

        elif workflow in {
            "defend",
            "defense",
            "preprocessor_defense",
            "gaussian_augmentation",
            "spatial_smoothing",
            "feature_squeezing",
            "clip_values",
            "reverse_sigmoid",
            "class_labels",
            "high_confidence_masking",
        } and not state.defended:
            if workflow in {
                "gaussian_augmentation", "spatial_smoothing", "feature_squeezing",
                "clip_values", "reverse_sigmoid", "class_labels", "high_confidence_masking",
            }:
                intercept_tool = workflow
            elif "spatial" in user_goal:
                intercept_tool = "spatial_smoothing"
            elif "squeez" in user_goal:
                intercept_tool = "feature_squeezing"
            elif "clip" in user_goal:
                intercept_tool = "clip_values"
            elif "sigmoid" in user_goal:
                intercept_tool = "reverse_sigmoid"
            elif "labels" in user_goal:
                intercept_tool = "class_labels"
            elif "confidence" in user_goal or "masking" in user_goal:
                intercept_tool = "high_confidence_masking"
            else:
                intercept_tool = "gaussian_augmentation"

        if intercept_tool and intercept_tool in tool_registry.tools:

            if not state.model_inspected and state.model_path:
                return Decision(
                    "inspect_model",
                    {"model_path": state.model_path, "model_id": state.model_id},
                    f"Inspecting model prior to executing {intercept_tool}."
                )
            if not state.dataset_inspected and state.dataset_path:
                return Decision(
                    "inspect_dataset",
                    {
                        "dataset_path": state.dataset_path,
                        "dataset_id": state.dataset_id,
                        "version": state.current_dataset_version,
                        "target_column": state.target_column,
                    },
                    f"Inspecting dataset prior to executing {intercept_tool}."
                )
            if not state.compatibility_validated and state.model_path and state.dataset_path:
                return Decision(
                    "validate_model_dataset",
                    {
                        "model_path": state.model_path,
                        "dataset_path": state.dataset_path,
                        "model_id": state.model_id,
                        "dataset_id": state.dataset_id,
                        "version": state.current_dataset_version,
                    },
                    "Validating compatibility before tool execution."
                )

            params = AgentPlanner._validate_and_fill_params(intercept_tool, {}, state)
            return Decision(
                action=intercept_tool,
                parameters=params,
                reason=f"Executing requested workflow tool: {intercept_tool}.",
                status="continue"
            )
        # =================================================================

        context_str = state.get_summary_context()

        tool_defs = json.dumps(
            tool_registry.get_tool_definitions(),
            indent=2,
            default=str,
        )

        prompt = f"""
CURRENT AGENT STATE
===================

{context_str}

USER OBJECTIVE
==============

{state.user_goal}

REQUESTED WORKFLOW
==================

{state.requested_workflow}

WORKFLOW OBJECTIVE
==================

{state.workflow_objective}

AVAILABLE TOOLS
===============

{tool_defs}

IMPORTANT PLANNING RULES
========================

1. Follow requested_workflow exactly.
2. NEVER expand a narrow workflow into full_evaluation.
3. Follow workflow boundary specifications.
4. Python state is authoritative.
5. Never select an action that is illegal for the current state.
6. Never repeat a successfully completed action unnecessarily.
7. Numerical metrics must come only from deterministic Python tools.

If the requested workflow is complete, return:
{{
    "status": "complete",
    "action": null,
    "parameters": {{}},
    "reason": "The requested workflow is complete."
}}

Otherwise return:
{{
    "status": "continue",
    "action": "<tool_name>",
    "parameters": {{}},
    "reason": "<short explanation>"
}}

Respond with VALID JSON ONLY.
"""

        try:
            llm_decision = qwen_client.generate_json(
                prompt=prompt,
                system=AGENT_SYSTEM_PROMPT,
            )
        except Exception:
            llm_decision = None

        if isinstance(llm_decision, dict):
            llm_status = llm_decision.get("status", "continue")
            action = llm_decision.get("action")

            if llm_status == "complete":
                if (
                    state.workflow_complete()
                    and state.requested_workflow not in {
                        "full_evaluation",
                        "improve_dataset",
                        "attack",
                        "evasion",
                        "defend",
                        "defense",
                    }
                ):
                    return Decision(
                        action="",
                        parameters={},
                        reason=llm_decision.get("reason", "The requested workflow is complete."),
                        status="complete",
                    )
                return AgentPlanner._deterministic_fallback_plan(state)

            if (
                action
                and action in tool_registry.tools
                and AgentPlanner._is_action_legal(action, state)
            ):
                params = AgentPlanner._validate_and_fill_params(
                    action,
                    llm_decision.get("parameters", {}),
                    state,
                )
                return Decision(
                    action=action,
                    parameters=params,
                    reason=llm_decision.get("reason", "Qwen selected the next legal workflow step."),
                    status="continue",
                )

        return AgentPlanner._deterministic_fallback_plan(state)

    # ------------------------------------------------------------------
    # Workflow legality
    # ------------------------------------------------------------------

    @staticmethod
    def _is_action_legal(action: str, state: AgentState) -> bool:
        workflow = (state.requested_workflow or "").lower()

        # ATTACK WORKFLOW
        if workflow in {
            "attack",
            "evasion",
            "adversarial_attack",
            "boundary_attack",
            "hopskipjump",
            "zoo",
            "fgsm",
            "pgd",
            "cw",
        }:
            if action in {"boundary_attack", "hopskipjump", "zoo", "fgsm", "pgd", "cw"}:
                return state.compatibility_validated and not state.adversarial_evaluated
            if action == "save_experiment":
                return state.adversarial_evaluated and not state.experiments
            if action == "generate_report":
                return state.adversarial_evaluated
            if action == "inspect_model":
                return not state.model_inspected
            if action == "inspect_dataset":
                return state.model_inspected and not state.dataset_inspected
            if action == "validate_model_dataset":
                return state.model_inspected and state.dataset_inspected and not state.compatibility_validated
            return False

        # DEFEND WORKFLOW
        if workflow in {
            "defend",
            "defense",
            "preprocessor_defense",
            "gaussian_augmentation",
            "spatial_smoothing",
            "feature_squeezing",
            "clip_values",
            "reverse_sigmoid",
            "class_labels",
            "high_confidence_masking",
        }:
            if action in {
                "gaussian_augmentation",
                "spatial_smoothing",
                "feature_squeezing",
                "clip_values",
                "reverse_sigmoid",
                "class_labels",
                "high_confidence_masking",
            }:
                return state.compatibility_validated and not state.defended
            if action == "save_experiment":
                return state.defended and not state.experiments
            if action == "generate_report":
                return state.defended
            if action == "inspect_model":
                return not state.model_inspected
            if action == "inspect_dataset":
                return state.model_inspected and not state.dataset_inspected
            if action == "validate_model_dataset":
                return state.model_inspected and state.dataset_inspected and not state.compatibility_validated
            return False

        # DEFENDED ATTACK WORKFLOW
        if workflow in {"defended_attack", "attack_defended_model"}:
            if action == "inspect_model":
                return not state.model_inspected
            if action == "inspect_dataset":
                return state.model_inspected and not state.dataset_inspected
            if action == "validate_model_dataset":
                return (
                    state.model_inspected
                    and state.dataset_inspected
                    and not state.compatibility_validated
                )
            if action == "attack_defended_model":
                return (
                    AgentPlanner._compatibility_ok(state)
                    and not state.defended_attack_evaluated
                )
            if action == "save_experiment":
                return state.defended_attack_evaluated and not state.experiments
            if action == "generate_report":
                return (
                    state.defended_attack_evaluated
                    and bool(state.experiments)
                    and AgentPlanner._report_requested(state)
                    and not AgentPlanner._action_was_taken(state, action)
                )
            return False

        # MODEL HARDENING WORKFLOW
        if workflow in {"harden_model", "harden"}:
            if action == "inspect_model":
                return not state.model_inspected
            if action == "inspect_dataset":
                return state.model_inspected and not state.dataset_inspected
            if action == "validate_model_dataset":
                return (
                    state.model_inspected
                    and state.dataset_inspected
                    and not state.compatibility_validated
                )
            if action == "harden_model":
                return AgentPlanner._compatibility_ok(state) and not state.model_hardened
            if action == "save_experiment":
                return state.model_hardened and not state.experiments
            if action == "generate_report":
                return (
                    state.model_hardened
                    and bool(state.experiments)
                    and AgentPlanner._report_requested(state)
                    and not AgentPlanner._action_was_taken(state, action)
                )
            return False

        # INSPECTION
        if workflow == "inspection":
            if action == "inspect_model":
                return not state.model_inspected
            if action == "inspect_dataset":
                return state.model_inspected and not state.dataset_inspected
            if action == "validate_model_dataset":
                return state.model_inspected and state.dataset_inspected and not state.compatibility_validated
            return False

        # BASELINE METRICS
        if workflow == "baseline_metrics":
            if action == "inspect_model":
                return not state.model_inspected
            if action == "inspect_dataset":
                return state.model_inspected and not state.dataset_inspected
            if action == "validate_model_dataset":
                return state.model_inspected and state.dataset_inspected and not state.compatibility_validated
            if action == "evaluate_performance":
                return state.compatibility_validated and not state.baseline_evaluated
            return False

        # TARGETED TRANSFORMATION
        if workflow == "targeted_transform":
            target_action = AgentPlanner._get_target_action(state)
            if action == "inspect_model":
                return not state.model_inspected
            if action == "inspect_dataset":
                return state.model_inspected and not state.dataset_inspected
            if action == "validate_model_dataset":
                return state.model_inspected and state.dataset_inspected and not state.compatibility_validated
            if action in {
                "handle_missing_values",
                "balance_data",
                "handle_outliers",
                "scale_data",
                "add_gaussian_noise",
                "inject_missing_values",
                "inject_label_noise",
            }:
                if not state.compatibility_validated or not target_action or action != target_action:
                    return False
                return action not in AgentPlanner._applied_operation_names(state)
            return False

        # IMPROVE DATASET
        if workflow == "improve_dataset":
            if action == "inspect_model":
                return not state.model_inspected
            if action == "inspect_dataset":
                return state.model_inspected and not state.dataset_inspected
            if action == "validate_model_dataset":
                return state.model_inspected and state.dataset_inspected and not state.compatibility_validated
            if action == "evaluate_performance":
                return state.compatibility_validated and (
                    not state.baseline_evaluated or state.evaluated_dataset_version != state.current_dataset_version
                )
            if action == "evaluate_data_quality":
                return state.baseline_evaluated and not state.data_quality_evaluated
            if action == "evaluate_robustness":
                return state.data_quality_evaluated and not state.robustness_evaluated
            if action in {"handle_missing_values", "balance_data", "handle_outliers", "scale_data"}:
                return AgentPlanner._transformation_is_justified(action, state)
            if action == "compare_results":
                return (
                    bool(state.transformations_applied)
                    and state.latest_metrics is not None
                    and state.evaluated_dataset_version == state.current_dataset_version
                    and not AgentPlanner._action_was_taken(state, "compare_results")
                )
            if action == "save_experiment":
                return (
                    bool(state.transformations_applied)
                    and state.latest_metrics is not None
                    and state.evaluated_dataset_version == state.current_dataset_version
                    and AgentPlanner._action_was_taken(state, "compare_results")
                    and not state.experiments
                )
            return False

        # FULL EVALUATION
        if workflow == "full_evaluation":
            if action == "inspect_model":
                return not state.model_inspected
            if action == "inspect_dataset":
                return state.model_inspected and not state.dataset_inspected
            if action == "validate_model_dataset":
                return state.model_inspected and state.dataset_inspected and not state.compatibility_validated
            if action == "evaluate_performance":
                return state.compatibility_validated and (
                    not state.baseline_evaluated or state.evaluated_dataset_version != state.current_dataset_version
                )
            if action == "evaluate_data_quality":
                return state.baseline_evaluated and not state.data_quality_evaluated
            if action == "evaluate_robustness":
                return state.data_quality_evaluated and not state.robustness_evaluated
            if action in {"handle_missing_values", "balance_data", "handle_outliers", "scale_data"}:
                return AgentPlanner._transformation_is_justified(action, state)
            if action == "compare_results":
                return (
                    bool(state.transformations_applied)
                    and state.latest_metrics is not None
                    and state.evaluated_dataset_version == state.current_dataset_version
                    and not AgentPlanner._action_was_taken(state, "compare_results")
                )
            if action == "save_experiment":
                return (
                    bool(state.transformations_applied)
                    and state.latest_metrics is not None
                    and state.evaluated_dataset_version == state.current_dataset_version
                    and AgentPlanner._action_was_taken(state, "compare_results")
                    and not state.experiments
                )
            if action == "generate_report":
                return (
                    state.baseline_evaluated
                    and state.data_quality_evaluated
                    and state.robustness_evaluated
                    and state.evaluated_dataset_version == state.current_dataset_version
                    and (not state.transformations_applied or bool(state.experiments))
                )
            return False

        # REPORT
        if workflow == "report":
            if action == "inspect_model":
                return not state.model_inspected
            if action == "inspect_dataset":
                return state.model_inspected and not state.dataset_inspected
            if action == "validate_model_dataset":
                return state.model_inspected and state.dataset_inspected and not state.compatibility_validated
            if action == "evaluate_performance":
                return state.compatibility_validated and not state.baseline_evaluated
            if action == "evaluate_data_quality":
                return state.baseline_evaluated and not state.data_quality_evaluated
            if action == "evaluate_robustness":
                return state.data_quality_evaluated and not state.robustness_evaluated
            if action == "generate_report":
                return (
                    state.baseline_evaluated
                    and state.data_quality_evaluated
                    and state.robustness_evaluated
                    and state.evaluated_dataset_version == state.current_dataset_version
                )
            return False

        return False

    # ------------------------------------------------------------------
    # Transformation justification & Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _transformation_is_justified(action: str, state: AgentState) -> bool:
        if not state.compatibility_validated or not state.data_quality_evaluated:
            return False
        if action in AgentPlanner._applied_operation_names(state):
            return False

        dq = state.data_quality or {}
        if not isinstance(dq, dict):
            return False

        if action == "handle_missing_values":
            return (
                dq.get("missing_values_count", 0) > 0
                or dq.get("missing_values", 0) > 0
                or bool(dq.get("missing_values_detected", False))
            )
        if action == "balance_data":
            return bool(
                dq.get("class_imbalance_detected", False)
                or dq.get("imbalance_detected", False)
                or dq.get("is_imbalanced", False)
            )
        if action == "handle_outliers":
            return (
                dq.get("outliers_count", 0) > 0
                or dq.get("outlier_count", 0) > 0
                or bool(dq.get("outliers_detected", False))
            )
        if action == "scale_data":
            return AgentPlanner._scaling_is_needed(state)

        return False

    @staticmethod
    def _get_target_action(state: AgentState) -> Optional[str]:
        objective = (state.workflow_objective or state.user_goal or "").lower()
        mapping = {
            "handle_missing_values": ["handle_missing_values", "missing values", "missing data", "clean missing"],
            "balance_data": ["balance_data", "class imbalance", "balance the dataset", "balance dataset", "smote"],
            "handle_outliers": ["handle_outliers", "outliers", "remove outliers"],
            "scale_data": ["scale_data", "scale dataset", "normalize data", "feature scaling"],
            "add_gaussian_noise": ["add_gaussian_noise", "gaussian noise", "add noise"],
            "inject_missing_values": ["inject_missing_values", "inject missing values"],
            "inject_label_noise": ["inject_label_noise", "label noise", "noisy labels"],
        }
        for action, phrases in mapping.items():
            if any(phrase in objective for phrase in phrases):
                return action
        return None

    @staticmethod
    def _applied_operation_names(state: AgentState) -> set:
        names = set()
        for item in state.transformations_applied:
            if isinstance(item, str):
                names.add(item)
            elif isinstance(item, dict):
                op = item.get("operation") or item.get("action")
                if op:
                    names.add(op)
        return names

    @staticmethod
    def _report_requested(state: AgentState) -> bool:
        text = f"{state.user_goal or ''} {state.workflow_objective or ''}".lower()
        return any(kw in text for kw in ["report", "summary document", "write up", "document this"])

    @staticmethod
    def _extract_attack_and_defense(text: str) -> tuple:
        """Return canonical attack and defense IDs from a compound prompt."""
        return (extract_attack(text), extract_defense(text))

    @staticmethod
    def _extract_hardening_method(text: str) -> str:
        return extract_hardening_method(text)

    @staticmethod
    def _build_security_combinations(
        text: str, default_cap: int = 5
    ) -> tuple:
        """Ordered attack/defense combinations plus the requested step count.

        N is an upper bound: combinations are never duplicated or invented to
        reach it, and never dropped below what the user explicitly named. A
        single combination returns ``([], 0)`` so the original single-pair path
        runs unchanged, which is what keeps today's prompts behaving identically.
        """
        attacks = extract_attacks(text)
        defenses = extract_defenses(text)
        if not attacks and not defenses:
            explicit = extract_explicit_step_count(text, default_cap)
            if explicit is not None and explicit >= 2:
                # The user asked for several combinations but named none, so
                # there is nothing to build them from. Guessing a default N
                # times over would produce N identical rows.
                raise IncompatibleSecurityRequest(
                    "Several attack/defense combinations were requested but no "
                    "attack or defense was named, so there is nothing to "
                    "compare. Please name at least one attack and one defense, "
                    "for example: compare fgsm, pgd and cw with "
                    "feature_squeezing."
                )
            return [], 0

        named = max(len(attacks), len(defenses))
        explicit = extract_explicit_step_count(text, default_cap)
        # Unstated N means "two", but never fewer than the combinations the
        # user actually enumerated.
        requested = explicit if explicit is not None else max(2, named)
        requested = min(requested, default_cap)

        count = min(requested, named, default_cap)
        if count <= 1:
            return [], 0

        combinations = [
            {
                "attack_type": attacks[i % len(attacks)] if attacks else None,
                "defense_type": defenses[i % len(defenses)] if defenses else None,
            }
            for i in range(count)
        ]
        return combinations, requested

    @staticmethod
    def _build_loop_rounds(text: str, default_cap: int = 5) -> tuple:
        """Rounds for the cumulative hardening loop.

        With several attacks named, each becomes one round in the order written
        and an explicit N cycles the list if it exceeds what was named — a
        mixed-threat curriculum. With one attack named, every round repeats it,
        which is the pure-accumulation case; repeating an attack across rounds
        is the point of a loop, so the "never duplicate" ceiling that applies to
        the sweep does not apply here.

        The defense stays a controlled variable unless several are named.
        Unstated N means two rounds.
        """
        attacks = extract_attacks(text)
        defenses = extract_defenses(text)
        if not attacks and not defenses:
            default_attack, default_defense = AgentPlanner._extract_attack_and_defense(text)
            attacks, defenses = [default_attack], [default_defense]
        if not attacks:
            # A loop needs a threat to harden against; fall back to the default
            # rather than producing rounds with no attack.
            attacks = [extract_attack(text)]

        explicit = extract_explicit_step_count(text, default_cap)
        requested = explicit if explicit is not None else max(2, len(attacks))
        requested = max(1, min(requested, default_cap))

        # Several attacks: one round each, cycling when N exceeds the list.
        # One attack: repeat it for every requested round.
        count = requested if len(attacks) <= 1 else requested
        count = max(1, min(count, default_cap))

        rounds = [
            {
                "attack_type": attacks[i % len(attacks)],
                "defense_type": defenses[i % len(defenses)] if defenses else None,
            }
            for i in range(count)
        ]
        return rounds, count

    @staticmethod
    def _compatibility_ok(state: AgentState) -> bool:
        """Security actions require a *passed* compatibility check, not merely
        a check that ran. An incompatible model must never be attacked."""
        return bool(state.compatibility_validated) and getattr(
            state, "compatibility_result", None
        ) is True

    @staticmethod
    def _action_was_taken(state: AgentState, action: str) -> bool:
        for item in state.actions_taken:
            if isinstance(item, dict):
                if item.get("action") != action:
                    continue
                if item.get("status") == "failed":
                    continue
                return True
            elif isinstance(item, str) and item == action:
                return True
        return False

    @staticmethod
    def _scaling_is_needed(state: AgentState) -> bool:
        metadata = state.model_metadata or {}
        if not isinstance(metadata, dict):
            return False
        if metadata.get("scaling_required") is True:
            return True
        preprocessing = metadata.get("preprocessing")
        if isinstance(preprocessing, str):
            text = preprocessing.lower()
            return any(k in text for k in ["standardscaler", "minmaxscaler", "normalization", "scaling required"])
        return False

    # ------------------------------------------------------------------
    # Parameter validation / defaults
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_and_fill_params(action: str, params: Dict[str, Any], state: AgentState) -> Dict[str, Any]:
        p = dict(params or {})

        # Extract prompt hyperparameter overrides
        if action in {
            "boundary_attack", "hopskipjump", "zoo", "fgsm", "pgd", "cw",
            "gaussian_augmentation", "spatial_smoothing", "feature_squeezing",
            "clip_values", "reverse_sigmoid", "class_labels", "high_confidence_masking",
            "attack_defended_model", "harden_model", "harden_defense_loop",
        }:
            extracted = AgentPlanner._extract_numeric_params(
                state.user_goal,
                ["eps", "eps_step", "max_iter", "batch_size", "init_size", "max_eval", "sigma", "window_size", "bit_depth", "cutoff", "beta", "gamma"]
            )
            for k, v in extracted.items():
                p.setdefault(k, v)

        if action in {
            "inspect_model",
            "validate_model_dataset",
            "evaluate_performance",
            "evaluate_robustness",
            "boundary_attack",
            "hopskipjump",
            "zoo",
            "fgsm",
            "pgd",
            "cw",
            "gaussian_augmentation",
            "spatial_smoothing",
            "feature_squeezing",
            "clip_values",
            "reverse_sigmoid",
            "class_labels",
            "high_confidence_masking",
            "attack_defended_model",
            "harden_model",
            "harden_defense_loop",
        }:
            p["model_path"] = state.model_path
            p["model_id"] = state.model_id

        if action in {
            "boundary_attack", "hopskipjump", "zoo", "fgsm", "pgd", "cw",
            "gaussian_augmentation", "spatial_smoothing", "feature_squeezing",
            "clip_values", "reverse_sigmoid", "class_labels", "high_confidence_masking",
            "attack_defended_model", "harden_model", "harden_defense_loop",
        }:
            p["dataset_path"] = AgentPlanner._resolve_original_dataset_path(state)
            p["dataset_id"] = state.dataset_id
            p["version"] = state.current_dataset_version
            if state.target_column:
                p["target_column"] = state.target_column

        elif action in {
            "inspect_dataset",
            "validate_model_dataset",
            "evaluate_performance",
            "evaluate_data_quality",
            "evaluate_robustness",
        }:
            p["dataset_path"] = state.dataset_path
            p["dataset_id"] = state.dataset_id
            p["version"] = state.current_dataset_version
            if state.target_column:
                p["target_column"] = state.target_column

        if action == "attack_defended_model":
            attack_name, defense_name = AgentPlanner._extract_attack_and_defense(
                state.user_goal
            )
            p["attack_name"] = attack_name
            p["defense_name"] = defense_name

        if action == "harden_model":
            p["project_id"] = state.project_id
            p["method"] = AgentPlanner._extract_hardening_method(state.user_goal)

        if action == "harden_defense_loop":
            attack_name, defense_name = AgentPlanner._extract_attack_and_defense(
                state.user_goal
            )
            rounds, requested = AgentPlanner._build_loop_rounds(state.user_goal)
            p["project_id"] = state.project_id
            p["attack_name"] = attack_name
            p["defense_name"] = defense_name
            p["retrain_method"] = AgentPlanner._extract_hardening_method(state.user_goal)
            p["rounds"] = rounds
            p["max_rounds"] = 5
            p["requested_rounds"] = requested or len(rounds)

        if action == "save_experiment":
            p.setdefault("project_id", state.project_id)
            p.setdefault("model_id", state.model_id)
            p.setdefault("operation", "dataset_transformation")
            if not p.get("input_dataset_version"):
                p["input_dataset_version"] = AgentPlanner._first_transformation_input_version(state) or "v001"
            if not p.get("output_dataset_version"):
                p["output_dataset_version"] = state.current_dataset_version
            if not p.get("metrics_before"):
                p["metrics_before"] = state.baseline_metrics or {}
            if not p.get("metrics_after"):
                p["metrics_after"] = state.latest_metrics or {}
            if not p.get("parameters"):
                p["parameters"] = {}
            if not p.get("agent_reasoning"):
                p["agent_reasoning"] = "Operation recorded automatically."

        if action == "compare_results":
            # The tool takes positional-style keyword arguments; the registry
            # will not invent them, so supply every required one explicitly.
            baseline_version = (
                AgentPlanner._first_transformation_input_version(state)
                or "v001"
            )
            p["baseline_version"] = baseline_version
            p["experiment_version"] = state.current_dataset_version or "v001"
            p["baseline_metrics"] = state.baseline_metrics or {}
            p["experiment_metrics"] = state.latest_metrics or {}

        if action == "generate_report":
            p.setdefault("project_id", state.project_id)
            p.setdefault("title", f"Evaluation Report - {state.project_id}")

        return p

    @staticmethod
    def _first_transformation_input_version(state: AgentState) -> Optional[str]:
        for item in state.transformations_applied:
            if isinstance(item, dict) and item.get("input_version"):
                return item["input_version"]
        return None

    # ------------------------------------------------------------------
    # Deterministic fallback planner
    # ------------------------------------------------------------------

    @staticmethod
    def _deterministic_fallback_plan(state: AgentState) -> Decision:
        workflow = (state.requested_workflow or "").lower()

        # ATTACK WORKFLOW
        if workflow in {"attack", "evasion", "adversarial_attack", "boundary_attack", "hopskipjump", "zoo", "fgsm", "pgd", "cw"}:
            if not state.model_inspected:
                return Decision("inspect_model", {"model_path": state.model_path, "model_id": state.model_id}, "Inspect model prior to attack.")
            if not state.dataset_inspected:
                return Decision("inspect_dataset", {"dataset_path": state.dataset_path, "dataset_id": state.dataset_id, "version": state.current_dataset_version, "target_column": state.target_column}, "Inspect dataset prior to attack.")
            if not state.compatibility_validated:
                return Decision("validate_model_dataset", {"model_path": state.model_path, "dataset_path": state.dataset_path, "model_id": state.model_id, "dataset_id": state.dataset_id, "version": state.current_dataset_version}, "Validate compatibility.")

            _goal = (state.user_goal or "").lower()
            if "hop" in _goal:
                attack_choice = "hopskipjump"
            elif "zoo" in _goal:
                attack_choice = "zoo"
            elif "fgsm" in _goal:
                attack_choice = "fgsm"
            elif "pgd" in _goal:
                attack_choice = "pgd"
            elif "cw" in _goal:
                attack_choice = "cw"
            else:
                attack_choice = "boundary_attack"

            if not state.adversarial_evaluated:
                extracted = AgentPlanner._extract_numeric_params(
                    state.user_goal, ["eps", "eps_step", "max_iter", "batch_size", "init_size", "max_eval"]
                )
                return Decision(
                    attack_choice,
                    {
                        "model_path": state.model_path,
                        "dataset_path": AgentPlanner._resolve_original_dataset_path(state),
                        "target_column": state.target_column,
                        **extracted,
                    },
                    f"Execute adversarial attack: {attack_choice}.",
                )

            if not state.experiments:
                ar = state.attack_results or {}
                metrics_before = {"accuracy": state.clean_accuracy} if state.clean_accuracy is not None else {}
                metrics_after = {"accuracy": state.adversarial_accuracy} if state.adversarial_accuracy is not None else {}
                if ar.get("clean_precision") is not None:
                    metrics_before["precision"] = ar.get("clean_precision")
                    metrics_before["recall"] = ar.get("clean_recall")
                    metrics_before["f1"] = ar.get("clean_f1")
                if ar.get("adversarial_precision") is not None:
                    metrics_after["precision"] = ar.get("adversarial_precision")
                    metrics_after["recall"] = ar.get("adversarial_recall")
                    metrics_after["f1"] = ar.get("adversarial_f1")

                acc = state.adversarial_accuracy if state.adversarial_accuracy is not None else 0.0
                rec = ar.get("adversarial_recall", acc)
                f1 = ar.get("adversarial_f1", acc)
                prec = ar.get("adversarial_precision", acc)

                reasoning_str = (
                    f"model: {state.model_id}, "
                    f"dataset: {state.current_dataset_version or 'v001'}, "
                    f"operation: {state.attack_type or attack_choice}, "
                    f"accuracy: {acc:.4f}, "
                    f"recall: {rec:.4f}, "
                    f"f1 score: {f1:.4f}, "
                    f"precision: {prec:.4f}"
                )

                return Decision(
                    "save_experiment",
                    {
                        "project_id": state.project_id,
                        "model_id": state.model_id,
                        "operation": state.attack_type or attack_choice,
                        "input_dataset_version": state.current_dataset_version or "v001",
                        "output_dataset_version": state.current_dataset_version or "v001",
                        "metrics_before": metrics_before,
                        "metrics_after": metrics_after,
                        "agent_reasoning": reasoning_str,
                    },
                    "Persist the adversarial attack outcome as a project experiment.",
                )

            if AgentPlanner._report_requested(state) and not AgentPlanner._action_was_taken(state, "generate_report"):
                return Decision("generate_report", {"project_id": state.project_id, "title": f"Adversarial Attack Report - {state.project_id}"}, "Generate attack report.")

            return Decision("", {}, "Adversarial attack workflow completed.", status="complete")

        # DEFEND WORKFLOW
        if workflow in {
            "defend",
            "defense",
            "preprocessor_defense",
            "gaussian_augmentation",
            "spatial_smoothing",
            "feature_squeezing",
            "clip_values",
            "reverse_sigmoid",
            "class_labels",
            "high_confidence_masking",
        }:
            if not state.model_inspected:
                return Decision("inspect_model", {"model_path": state.model_path, "model_id": state.model_id}, "Inspect model prior to defense.")
            if not state.dataset_inspected:
                return Decision("inspect_dataset", {"dataset_path": state.dataset_path, "dataset_id": state.dataset_id, "version": state.current_dataset_version, "target_column": state.target_column}, "Inspect dataset prior to defense.")
            if not state.compatibility_validated:
                return Decision("validate_model_dataset", {"model_path": state.model_path, "dataset_path": state.dataset_path, "model_id": state.model_id, "dataset_id": state.dataset_id, "version": state.current_dataset_version}, "Validate compatibility.")

            _goal_def = (state.user_goal or "").lower()
            defense_choice = (
                "spatial_smoothing" if "spatial" in _goal_def
                else "feature_squeezing" if "squeez" in _goal_def
                else "clip_values" if "clip" in _goal_def
                else "reverse_sigmoid" if "sigmoid" in _goal_def
                else "class_labels" if "labels" in _goal_def
                else "high_confidence_masking" if "confidence" in _goal_def or "masking" in _goal_def
                else "gaussian_augmentation"
            )

            if not state.defended:
                extracted = AgentPlanner._extract_numeric_params(
                    state.user_goal, ["sigma", "window_size", "bit_depth"]
                )
                return Decision(
                    defense_choice,
                    {
                        "model_path": state.model_path,
                        "dataset_path": AgentPlanner._resolve_original_dataset_path(state),
                        "target_column": state.target_column,
                        **extracted,
                    },
                    f"Apply ART defense: {defense_choice}.",
                )

            if not state.experiments:
                dr = state.defense_results or {}
                clean_acc = dr.get("clean_accuracy", state.clean_accuracy)
                defended_acc = dr.get("defended_accuracy")

                metrics_before = {"accuracy": clean_acc} if clean_acc is not None else {}
                metrics_after = {"accuracy": defended_acc} if defended_acc is not None else {}

                if dr.get("clean_precision") is not None:
                    metrics_before["precision"] = dr.get("clean_precision")
                    metrics_before["recall"] = dr.get("clean_recall")
                    metrics_before["f1"] = dr.get("clean_f1")
                if dr.get("defended_precision") is not None:
                    metrics_after["precision"] = dr.get("defended_precision")
                    metrics_after["recall"] = dr.get("defended_recall")
                    metrics_after["f1"] = dr.get("defended_f1")

                op_name = dr.get("defense_type") or defense_choice
                acc = defended_acc if defended_acc is not None else 0.0
                rec = dr.get("defended_recall", acc)
                f1 = dr.get("defended_f1", acc)
                prec = dr.get("defended_precision", acc)

                reasoning_str = (
                    f"model: {state.model_id}, "
                    f"dataset: {state.current_dataset_version or 'v001'}, "
                    f"operation: {op_name}, "
                    f"accuracy: {acc:.4f}, "
                    f"recall: {rec:.4f}, "
                    f"f1 score: {f1:.4f}, "
                    f"precision: {prec:.4f}"
                )

                return Decision(
                    "save_experiment",
                    {
                        "project_id": state.project_id,
                        "model_id": state.model_id,
                        "operation": op_name,
                        "input_dataset_version": state.current_dataset_version or "v001",
                        "output_dataset_version": state.current_dataset_version or "v001",
                        "metrics_before": metrics_before,
                        "metrics_after": metrics_after,
                        "agent_reasoning": reasoning_str,
                    },
                    "Persist the defense outcome as a project experiment.",
                )

            if AgentPlanner._report_requested(state) and not AgentPlanner._action_was_taken(state, "generate_report"):
                return Decision("generate_report", {"project_id": state.project_id, "title": f"Defense Report - {state.project_id}"}, "Generate defense report.")

            return Decision("", {}, "Defense workflow completed.", status="complete")

        # DEFENDED ATTACK WORKFLOW
        if workflow in {"defended_attack", "attack_defended_model"}:
            if not state.model_inspected:
                return Decision("inspect_model", {"model_path": state.model_path, "model_id": state.model_id}, "Inspect model.")
            if not state.dataset_inspected:
                return Decision("inspect_dataset", {"dataset_path": state.dataset_path, "dataset_id": state.dataset_id, "version": state.current_dataset_version, "target_column": state.target_column}, "Inspect dataset.")
            if not state.compatibility_validated:
                return Decision("validate_model_dataset", {"model_path": state.model_path, "dataset_path": state.dataset_path, "model_id": state.model_id, "dataset_id": state.dataset_id, "version": state.current_dataset_version}, "Validate compatibility.")

            if not state.defended_attack_evaluated and not state.attack_defense_sequence_evaluated:
                attack_name, defense_name = AgentPlanner._extract_attack_and_defense(
                    state.user_goal
                )
                combinations, requested = AgentPlanner._build_security_combinations(
                    state.user_goal
                )
                if not AgentPlanner._compatibility_ok(state):
                    raise IncompatibleSecurityRequest(
                        "The model and dataset are not compatible, so no adversarial "
                        "attack can be run against this project. Please use a "
                        "compatible model and dataset pair."
                    )
                # Pre-flight: refuse an attack this model cannot execute rather
                # than planning a guaranteed runtime failure. Every combination
                # is checked up front so step 3 cannot fail after steps 1 and 2.
                framework = (state.model_metadata or {}).get("framework")
                if combinations:
                    for combo in combinations:
                        if combo.get("attack_type"):
                            assert_attack_supported(combo["attack_type"], framework)
                else:
                    assert_attack_supported(attack_name, framework)
                params = {
                    "model_path": state.model_path,
                    "model_id": state.model_id,
                    "dataset_path": AgentPlanner._resolve_original_dataset_path(state),
                    "dataset_id": state.dataset_id,
                    "version": "v001",
                    "target_column": state.target_column,
                    "attack_name": attack_name,
                    "defense_name": defense_name,
                }
                if combinations:
                    params["steps"] = combinations
                    params["max_steps"] = 5
                    params["requested_steps"] = requested
                # Honor explicit hyperparameters from the prompt (eps, max_iter,
                # batch_size, ...) which the LLM-bypass path would otherwise drop.
                params = AgentPlanner._validate_and_fill_params(
                    "attack_defended_model", params, state
                )
                if combinations:
                    return Decision(
                        "attack_defended_model",
                        params,
                        f"Compare {len(combinations)} attack/defense combinations "
                        f"(up to {requested} requested) against the same model.",
                    )
                return Decision(
                    "attack_defended_model",
                    params,
                    f"Execute {attack_name} against the model wrapped with {defense_name}.",
                )

            if not state.experiments:
                sequence = state.attack_defense_sequence_results or {}
                if sequence.get("steps"):
                    result = sequence["steps"][-1]
                    attack_name = result.get("attack_type") or "fgsm"
                    defense_name = result.get("defense_type") or "feature_squeezing"
                    operation = f"attack_defense_sequence_{sequence.get('step_count')}"
                else:
                    result = state.defended_attack_results or {}
                    attack_name = result.get("attack_type") or "fgsm"
                    defense_name = result.get("defense_type") or "feature_squeezing"
                    operation = f"{attack_name}_on_{defense_name}"
                metrics_before = {
                    key: result.get(source_key)
                    for key, source_key in (
                        ("accuracy", "defended_accuracy"),
                        ("precision", "defended_precision"),
                        ("recall", "defended_recall"),
                        ("f1", "defended_f1"),
                    )
                    if result.get(source_key) is not None
                }
                metrics_after = {
                    key: result.get(source_key)
                    for key, source_key in (
                        ("accuracy", "adversarial_accuracy"),
                        ("precision", "adversarial_precision"),
                        ("recall", "adversarial_recall"),
                        ("f1", "adversarial_f1"),
                    )
                    if result.get(source_key) is not None
                }
                reasoning = (
                    f"model={state.model_id}; dataset={state.current_dataset_version or 'v001'}; "
                    f"attack={attack_name}; defense={defense_name}; "
                    f"defended_accuracy={metrics_before.get('accuracy')}; "
                    f"adversarial_accuracy={metrics_after.get('accuracy')}."
                )
                if sequence.get("steps"):
                    per_step = "; ".join(
                        f"{s.get('attack_type')} vs {s.get('defense_type')}: "
                        f"adv_acc={s.get('adversarial_accuracy')}, "
                        f"adv_f1={s.get('adversarial_f1')}"
                        for s in sequence["steps"]
                    )
                    reasoning = (
                        f"model={state.model_id}; combinations={sequence.get('step_count')}"
                        f"/{sequence.get('max_steps')} requested={sequence.get('requested_steps')}; "
                        f"{per_step}"
                    )
                # Per-step metrics are persisted alongside the last-step summary
                # so the dashboard can chart every step, not just the final one.
                series = (
                    build_series(sequence.get("steps"), "sweep")
                    if sequence.get("steps")
                    else None
                )

                return Decision(
                    "save_experiment",
                    {
                        "project_id": state.project_id,
                        "model_id": state.model_id,
                        "operation": operation,
                        "parameters": {
                            "attack": result.get("attack_parameters", {}),
                            "defense": result.get("defense_parameters", {}),
                            **(
                                {
                                    "step_count": sequence.get("step_count"),
                                    "requested_steps": sequence.get("requested_steps"),
                                    "max_steps": sequence.get("max_steps"),
                                }
                                if sequence.get("steps")
                                else {}
                            ),
                        },
                        "series": series,
                        "input_dataset_version": "v001",
                        "output_dataset_version": "v001",
                        "metrics_before": metrics_before,
                        "metrics_after": metrics_after,
                        "agent_reasoning": reasoning,
                    },
                    "Persist the defended attack outcome before any requested report.",
                )

            if (
                AgentPlanner._report_requested(state)
                and not AgentPlanner._action_was_taken(state, "generate_report")
            ):
                return Decision(
                    "generate_report",
                    {
                        "project_id": state.project_id,
                        "title": f"Defended Attack Report - {state.project_id}",
                    },
                    "Generate the report after experiment persistence.",
                )

            return Decision("", {}, "Defended attack workflow completed.", status="complete")

        # MODEL HARDENING WORKFLOW
        if workflow in {"harden_model", "harden"}:
            if not state.model_inspected:
                return Decision("inspect_model", {"model_path": state.model_path, "model_id": state.model_id}, "Inspect model.")
            if not state.dataset_inspected:
                return Decision("inspect_dataset", {"dataset_path": state.dataset_path, "dataset_id": state.dataset_id, "version": state.current_dataset_version, "target_column": state.target_column}, "Inspect dataset.")
            if not state.compatibility_validated:
                return Decision("validate_model_dataset", {"model_path": state.model_path, "dataset_path": state.dataset_path, "model_id": state.model_id, "dataset_id": state.dataset_id, "version": state.current_dataset_version}, "Validate compatibility.")

            if not state.model_hardened:
                method = AgentPlanner._extract_hardening_method(state.user_goal)
                if not AgentPlanner._compatibility_ok(state):
                    raise IncompatibleSecurityRequest(
                        "The model and dataset are not compatible, so adversarial "
                        "retraining cannot run against this project. Please use a "
                        "compatible model and dataset pair."
                    )
                assert_hardening_method_supported(
                    method,
                    (state.model_metadata or {}).get("framework"),
                )
                params = {
                    "model_path": state.model_path,
                    "model_id": state.model_id,
                    "dataset_path": AgentPlanner._resolve_original_dataset_path(state),
                    "dataset_id": state.dataset_id,
                    "version": "v001",
                    "target_column": state.target_column,
                    "method": method,
                    "project_id": state.project_id,
                }
                params = AgentPlanner._validate_and_fill_params(
                    "harden_model", params, state
                )
                return Decision(
                    "harden_model",
                    params,
                    "Retrain and export a hardened model asset.",
                )

            if not state.experiments:
                hardened_info = state.hardened_model_info or {}
                training_summary = hardened_info.get("training_summary", {})
                method_used = training_summary.get(
                    "method",
                    AgentPlanner._extract_hardening_method(state.user_goal),
                )
                new_model_id = hardened_info.get("model_id") or state.model_id
                reasoning = (
                    f"hardened_model={new_model_id}; "
                    f"model_path={hardened_info.get('file_path') or state.model_path}; "
                    f"dataset={state.current_dataset_version or 'v001'}; "
                    f"method={method_used}; samples_generated="
                    f"{training_summary.get('samples_generated', 0)}."
                )
                return Decision(
                    "save_experiment",
                    {
                        "project_id": state.project_id,
                        "model_id": new_model_id,
                        "operation": f"harden_model_{method_used}",
                        "parameters": {"hardening": training_summary},
                        "input_dataset_version": "v001",
                        "output_dataset_version": "v001",
                        "metrics_before": {},
                        "metrics_after": {},
                        "agent_reasoning": reasoning,
                    },
                    "Persist model hardening before any requested report.",
                )

            if (
                AgentPlanner._report_requested(state)
                and not AgentPlanner._action_was_taken(state, "generate_report")
            ):
                return Decision(
                    "generate_report",
                    {
                        "project_id": state.project_id,
                        "title": f"Model Hardening Report - {state.project_id}",
                    },
                    "Generate the hardening report after experiment persistence.",
                )

            return Decision("", {}, "Model hardening workflow completed.", status="complete")

        # CUMULATIVE HARDENING LOOP WORKFLOW
        if workflow in {"hardening_loop", "harden_defense_loop"}:
            if not state.model_inspected:
                return Decision("inspect_model", {"model_path": state.model_path, "model_id": state.model_id}, "Inspect model.")
            if not state.dataset_inspected:
                return Decision("inspect_dataset", {"dataset_path": state.dataset_path, "dataset_id": state.dataset_id, "version": state.current_dataset_version, "target_column": state.target_column}, "Inspect dataset.")
            if not state.compatibility_validated:
                return Decision("validate_model_dataset", {"model_path": state.model_path, "dataset_path": state.dataset_path, "model_id": state.model_id, "dataset_id": state.dataset_id, "version": state.current_dataset_version}, "Validate compatibility.")

            if not state.hardening_loop_evaluated:
                attack_name, defense_name = AgentPlanner._extract_attack_and_defense(
                    state.user_goal
                )
                method = AgentPlanner._extract_hardening_method(state.user_goal)
                rounds, requested = AgentPlanner._build_loop_rounds(
                    state.user_goal
                )
                if not AgentPlanner._compatibility_ok(state):
                    raise IncompatibleSecurityRequest(
                        "The model and dataset are not compatible, so adversarial "
                        "retraining cannot run against this project. Please use a "
                        "compatible model and dataset pair."
                    )
                framework = (state.model_metadata or {}).get("framework")
                assert_hardening_method_supported(method, framework)
                for combo in rounds:
                    if combo.get("attack_type"):
                        assert_attack_supported(combo["attack_type"], framework)
                params = {
                    "model_path": state.model_path,
                    "model_id": state.model_id,
                    "dataset_path": AgentPlanner._resolve_original_dataset_path(state),
                    "dataset_id": state.dataset_id,
                    "version": "v001",
                    "target_column": state.target_column,
                    "attack_name": attack_name,
                    "defense_name": defense_name,
                    "retrain_method": method,
                    "rounds": rounds,
                    "max_rounds": 5,
                    "requested_rounds": requested or len(rounds),
                    "project_id": state.project_id,
                }
                params = AgentPlanner._validate_and_fill_params(
                    "harden_defense_loop", params, state
                )
                return Decision(
                    "harden_defense_loop",
                    params,
                    f"Run {len(rounds)} cumulative hardening rounds using {method} retraining.",
                )

            if not state.experiments:
                loop = state.hardening_loop_results or {}
                round_list = loop.get("rounds") or []
                first = round_list[0] if round_list else {}
                last = round_list[-1] if round_list else {}
                metrics_before = {
                    "accuracy": first.get("adversarial_accuracy"),
                    "precision": first.get("adversarial_precision"),
                    "recall": first.get("adversarial_recall"),
                    "f1": first.get("adversarial_f1"),
                }
                metrics_after = {
                    "accuracy": last.get("adversarial_accuracy"),
                    "precision": last.get("adversarial_precision"),
                    "recall": last.get("adversarial_recall"),
                    "f1": last.get("adversarial_f1"),
                }
                per_round = "; ".join(
                    f"round{r.get('round_index')}: adv_acc={r.get('adversarial_accuracy')}, "
                    f"adv_f1={r.get('adversarial_f1')}"
                    for r in round_list
                )
                # Every round keeps the model it was measured on, so a chart or
                # report reader can tell which artefact produced which number.
                series = build_series(round_list, "loop") if round_list else None
                reasoning = (
                    f"model={loop.get('final_model_id')}; "
                    f"rounds={loop.get('round_count')}/{loop.get('max_rounds')} "
                    f"requested={loop.get('requested_rounds')}; "
                    f"retrain_method={loop.get('retrain_method')}; {per_round}"
                )
                return Decision(
                    "save_experiment",
                    {
                        "project_id": state.project_id,
                        "model_id": loop.get("final_model_id") or state.model_id,
                        "operation": f"hardening_loop_{loop.get('round_count')}_rounds",
                        "parameters": {
                            "hardening": {
                                "retrain_method": loop.get("retrain_method"),
                                "round_count": loop.get("round_count"),
                                "max_rounds": loop.get("max_rounds"),
                                "requested_rounds": loop.get("requested_rounds"),
                            }
                        },
                        "series": series,
                        "input_dataset_version": "v001",
                        "output_dataset_version": "v001",
                        "metrics_before": metrics_before,
                        "metrics_after": metrics_after,
                        "agent_reasoning": reasoning,
                    },
                    "Persist the hardening loop outcome before any requested report.",
                )

            if (
                AgentPlanner._report_requested(state)
                and not AgentPlanner._action_was_taken(state, "generate_report")
            ):
                return Decision(
                    "generate_report",
                    {
                        "project_id": state.project_id,
                        "title": f"Hardening Loop Report - {state.project_id}",
                    },
                    "Generate the hardening loop report after experiment persistence.",
                )

            return Decision("", {}, "Hardening loop workflow completed.", status="complete")

        # INSPECTION
        if workflow == "inspection":
            if not state.model_inspected:
                return Decision("inspect_model", {"model_path": state.model_path, "model_id": state.model_id}, "Inspect the model.")
            if not state.dataset_inspected:
                return Decision("inspect_dataset", {"dataset_path": state.dataset_path, "dataset_id": state.dataset_id, "version": state.current_dataset_version, "target_column": state.target_column}, "Inspect dataset.")
            if not state.compatibility_validated:
                return Decision("validate_model_dataset", {"model_path": state.model_path, "dataset_path": state.dataset_path, "model_id": state.model_id, "dataset_id": state.dataset_id, "version": state.current_dataset_version}, "Validate compatibility.")
            return Decision("", {}, "Inspection workflow is complete.", status="complete")

        # BASELINE METRICS
        if workflow == "baseline_metrics":
            if not state.model_inspected:
                return Decision("inspect_model", {"model_path": state.model_path, "model_id": state.model_id}, "Inspect model.")
            if not state.dataset_inspected:
                return Decision("inspect_dataset", {"dataset_path": state.dataset_path, "dataset_id": state.dataset_id, "version": state.current_dataset_version, "target_column": state.target_column}, "Inspect dataset.")
            if not state.compatibility_validated:
                return Decision("validate_model_dataset", {"model_path": state.model_path, "dataset_path": state.dataset_path, "model_id": state.model_id, "dataset_id": state.dataset_id, "version": state.current_dataset_version}, "Validate compatibility.")
            if not state.baseline_evaluated:
                return Decision("evaluate_performance", {"model_path": state.model_path, "dataset_path": state.dataset_path, "target_column": state.target_column, "task_type": state.task_type}, "Calculate baseline performance.")
            return Decision("", {}, "Baseline metrics workflow is complete.", status="complete")

        # TARGETED TRANSFORMATION
        if workflow == "targeted_transform":
            target_action = AgentPlanner._get_target_action(state)
            if not target_action:
                return Decision("", {}, "No specific transformation was identified.", status="complete")
            if not state.model_inspected:
                return Decision("inspect_model", {"model_path": state.model_path, "model_id": state.model_id}, "Inspect model.")
            if not state.dataset_inspected:
                return Decision("inspect_dataset", {"dataset_path": state.dataset_path, "dataset_id": state.dataset_id, "version": state.current_dataset_version, "target_column": state.target_column}, "Inspect dataset.")
            if not state.compatibility_validated:
                return Decision("validate_model_dataset", {"model_path": state.model_path, "dataset_path": state.dataset_path, "model_id": state.model_id, "dataset_id": state.dataset_id, "version": state.current_dataset_version}, "Validate compatibility.")
            if target_action in AgentPlanner._applied_operation_names(state):
                return Decision("", {}, "The requested transformation has already been applied.", status="complete")

            params = {"dataset_id": state.dataset_id, "current_version": state.current_dataset_version}
            if state.target_column:
                params["target_column"] = state.target_column
            if target_action == "handle_missing_values":
                params["strategy"] = "median"
            elif target_action == "balance_data":
                params["method"] = "smote"
            elif target_action == "handle_outliers":
                params["method"] = "clip"
            elif target_action == "scale_data":
                params["method"] = "standard"

            return Decision(target_action, params, f"Apply transformation: {target_action}.")

        # IMPROVE DATASET
        if workflow == "improve_dataset":
            if not state.model_inspected:
                return Decision("inspect_model", {"model_path": state.model_path, "model_id": state.model_id}, "Inspect model.")
            if not state.dataset_inspected:
                return Decision("inspect_dataset", {"dataset_path": state.dataset_path, "dataset_id": state.dataset_id, "version": state.current_dataset_version, "target_column": state.target_column}, "Inspect dataset.")
            if not state.compatibility_validated:
                return Decision("validate_model_dataset", {"model_path": state.model_path, "dataset_path": state.dataset_path, "model_id": state.model_id, "dataset_id": state.dataset_id, "version": state.current_dataset_version}, "Validate compatibility.")
            if not state.baseline_evaluated:
                return Decision("evaluate_performance", {"model_path": state.model_path, "dataset_path": state.dataset_path, "target_column": state.target_column, "task_type": state.task_type}, "Obtain baseline performance.")
            if not state.data_quality_evaluated:
                return Decision("evaluate_data_quality", {"dataset_path": state.dataset_path, "target_column": state.target_column}, "Evaluate data quality.")
            if not state.robustness_evaluated:
                return Decision("evaluate_robustness", {"model_path": state.model_path, "dataset_path": state.dataset_path, "target_column": state.target_column, "noise_level": 0.10}, "Evaluate robustness.")

            for action in ["handle_missing_values", "balance_data", "handle_outliers", "scale_data"]:
                if AgentPlanner._transformation_is_justified(action, state):
                    params = {"dataset_id": state.dataset_id, "current_version": state.current_dataset_version}
                    if state.target_column:
                        params["target_column"] = state.target_column
                    return Decision(action, params, f"Apply {action}.")

            if state.transformations_applied and state.evaluated_dataset_version != state.current_dataset_version:
                return Decision("evaluate_performance", {"model_path": state.model_path, "dataset_path": state.dataset_path, "target_column": state.target_column, "task_type": state.task_type}, "Re-evaluate performance.")
            if state.transformations_applied and state.latest_metrics is not None and not AgentPlanner._action_was_taken(state, "compare_results"):
                return Decision(
                    "compare_results",
                    AgentPlanner._validate_and_fill_params("compare_results", {}, state),
                    "Compare evaluation results.",
                )
            if state.transformations_applied and state.latest_metrics is not None and AgentPlanner._action_was_taken(state, "compare_results") and not state.experiments:
                return Decision(
                    "save_experiment",
                    {
                        "project_id": state.project_id,
                        "model_id": state.model_id,
                        "operation": "dataset_transformation",
                        "input_dataset_version": AgentPlanner._first_transformation_input_version(state) or "v001",
                        "output_dataset_version": state.current_dataset_version,
                        "metrics_before": state.baseline_metrics or {},
                        "metrics_after": state.latest_metrics or {},
                    },
                    "Persist experiment.",
                )

            return Decision("", {}, "Dataset improvement workflow is complete.", status="complete")

        # FULL EVALUATION
        if workflow == "full_evaluation":
            if not state.model_inspected:
                return Decision("inspect_model", {"model_path": state.model_path, "model_id": state.model_id}, "Inspect model.")
            if not state.dataset_inspected:
                return Decision("inspect_dataset", {"dataset_path": state.dataset_path, "dataset_id": state.dataset_id, "version": state.current_dataset_version, "target_column": state.target_column}, "Inspect dataset.")
            if not state.compatibility_validated:
                return Decision("validate_model_dataset", {"model_path": state.model_path, "dataset_path": state.dataset_path, "model_id": state.model_id, "dataset_id": state.dataset_id, "version": state.current_dataset_version}, "Validate compatibility.")
            if not state.baseline_evaluated:
                return Decision("evaluate_performance", {"model_path": state.model_path, "dataset_path": state.dataset_path, "target_column": state.target_column, "task_type": state.task_type}, "Calculate baseline performance.")
            if not state.data_quality_evaluated:
                return Decision("evaluate_data_quality", {"dataset_path": state.dataset_path, "target_column": state.target_column}, "Evaluate data quality.")
            if not state.robustness_evaluated:
                return Decision("evaluate_robustness", {"model_path": state.model_path, "dataset_path": state.dataset_path, "target_column": state.target_column, "noise_level": 0.10}, "Evaluate robustness.")

            for action in ["handle_missing_values", "balance_data", "handle_outliers", "scale_data"]:
                if AgentPlanner._transformation_is_justified(action, state):
                    params = {"dataset_id": state.dataset_id, "current_version": state.current_dataset_version}
                    if state.target_column:
                        params["target_column"] = state.target_column
                    return Decision(action, params, f"Apply {action}.")

            if state.transformations_applied and state.evaluated_dataset_version != state.current_dataset_version:
                return Decision("evaluate_performance", {"model_path": state.model_path, "dataset_path": state.dataset_path, "target_column": state.target_column, "task_type": state.task_type}, "Re-evaluate performance.")
            if state.transformations_applied and state.latest_metrics is not None and not AgentPlanner._action_was_taken(state, "compare_results"):
                return Decision(
                    "compare_results",
                    AgentPlanner._validate_and_fill_params("compare_results", {}, state),
                    "Compare results.",
                )
            if state.transformations_applied and state.latest_metrics is not None and AgentPlanner._action_was_taken(state, "compare_results") and not state.experiments:
                return Decision(
                    "save_experiment",
                    {
                        "project_id": state.project_id,
                        "model_id": state.model_id,
                        "operation": "dataset_transformation",
                        "input_dataset_version": AgentPlanner._first_transformation_input_version(state) or "v001",
                        "output_dataset_version": state.current_dataset_version,
                        "metrics_before": state.baseline_metrics or {},
                        "metrics_after": state.latest_metrics or {},
                    },
                    "Persist experiment.",
                )

            if state.baseline_evaluated and state.data_quality_evaluated and state.robustness_evaluated and state.evaluated_dataset_version == state.current_dataset_version and (not state.transformations_applied or bool(state.experiments)):
                return Decision("generate_report", {"project_id": state.project_id, "title": f"Evaluation Report - {state.project_id}"}, "Generate report.")

            return Decision("", {}, "Full evaluation workflow is complete.", status="complete")

        # REPORT
        if workflow == "report":
            if not state.model_inspected:
                return Decision("inspect_model", {"model_path": state.model_path, "model_id": state.model_id}, "Inspect model.")
            if not state.dataset_inspected:
                return Decision("inspect_dataset", {"dataset_path": state.dataset_path, "dataset_id": state.dataset_id, "version": state.current_dataset_version, "target_column": state.target_column}, "Inspect dataset.")
            if not state.compatibility_validated:
                return Decision("validate_model_dataset", {"model_path": state.model_path, "dataset_path": state.dataset_path, "model_id": state.model_id, "dataset_id": state.dataset_id, "version": state.current_dataset_version}, "Validate compatibility.")
            if not state.baseline_evaluated:
                return Decision("evaluate_performance", {"model_path": state.model_path, "dataset_path": state.dataset_path, "target_column": state.target_column, "task_type": state.task_type}, "Obtain baseline metrics.")
            if not state.data_quality_evaluated:
                return Decision("evaluate_data_quality", {"dataset_path": state.dataset_path, "target_column": state.target_column}, "Obtain data quality info.")
            if not state.robustness_evaluated:
                return Decision("evaluate_robustness", {"model_path": state.model_path, "dataset_path": state.dataset_path, "target_column": state.target_column, "noise_level": 0.10}, "Obtain robustness info.")

            return Decision("generate_report", {"project_id": state.project_id, "title": f"Evaluation Report - {state.project_id}"}, "Generate report.")

        return Decision("", {}, f"Unknown workflow: {workflow}", status="complete")


agent_planner = AgentPlanner()