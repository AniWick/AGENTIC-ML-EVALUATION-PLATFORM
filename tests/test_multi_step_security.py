"""Multi-step security workflows: comparative sweep and hardening loop."""
import json

import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier

from app.agent import security_aliases as aliases
from app.agent.planner import AgentPlanner, build_series
from app.evaluation.robustness import robustness_evaluator


# ----------------------------------------------------------------------
# Extraction + N parsing
# ----------------------------------------------------------------------

def test_plural_extraction_preserves_prompt_order():
    text = "compare fgsm, pgd and cw with feature_squeezing"
    assert aliases.extract_attacks(text) == ["fgsm", "pgd", "cw"]
    assert aliases.extract_defenses(text) == ["feature_squeezing"]


def test_plural_extraction_deduplicates_alias_spellings():
    text = "compare feature-squeezing, feature squeezing and feature_squeezing"
    assert aliases.extract_defenses(text) == ["feature_squeezing"]


def test_step_count_ignores_tuning_parameters():
    assert aliases.extract_step_count("eps=0.1 max_iter=10") == 2
    assert aliases.extract_step_count("max_iter=10 defense") == 2


def test_step_count_reads_explicit_forms_and_caps_at_five():
    assert aliases.extract_step_count("n=4") == 4
    assert aliases.extract_step_count("run 3 attacks") == 3
    assert aliases.extract_step_count("9 rounds") == 5
    assert aliases.extract_step_count("n=99") == 5


def test_hardening_loop_detection():
    assert aliases.is_hardening_loop_request("harden the model over 3 rounds")
    assert aliases.is_hardening_loop_request("iteratively harden my model")
    assert not aliases.is_hardening_loop_request("harden my model")


# ----------------------------------------------------------------------
# Round sequencing + retraining-method disambiguation
# ----------------------------------------------------------------------

def test_single_attack_repeats_for_every_requested_round():
    rounds, requested = AgentPlanner._build_loop_rounds(
        "harden my model over 3 rounds using pgd"
    )
    assert requested == 3
    assert [r["attack_type"] for r in rounds] == ["pgd", "pgd", "pgd"]


def test_each_named_attack_becomes_one_round_in_order():
    rounds, requested = AgentPlanner._build_loop_rounds(
        "harden my model over 5 rounds against fgsm, pgd, hopskipjump and zoo "
        "with feature_squeezing using pgd"
    )
    assert requested == 5
    assert [r["attack_type"] for r in rounds] == [
        "fgsm", "pgd", "hopskipjump", "zoo", "fgsm",
    ]
    # The defense stays a controlled variable across rounds.
    assert {r["defense_type"] for r in rounds} == {"feature_squeezing"}


def test_requested_rounds_cycles_when_more_rounds_than_attacks():
    rounds, _ = AgentPlanner._build_loop_rounds(
        "harden over 4 rounds with boundary attack and hopskipjump using hopskipjump"
    )
    assert [r["attack_type"] for r in rounds] == [
        "boundary_attack", "hopskipjump", "boundary_attack", "hopskipjump",
    ]


def test_loop_rounds_never_exceed_the_cap():
    rounds, requested = AgentPlanner._build_loop_rounds(
        "harden my model over 99 rounds against fgsm, pgd, cw, zoo, hopskipjump"
    )
    assert requested == 5
    assert len(rounds) <= 5


def test_loop_falls_back_to_a_default_attack_when_only_a_defense_is_named():
    rounds, _ = AgentPlanner._build_loop_rounds("harden my model with feature_squeezing")
    assert rounds and all(r["attack_type"] for r in rounds)


def test_retraining_method_follows_the_preposition_not_the_first_word():
    # FGSM is the threat; PGD is only named after "using".
    assert aliases.extract_hardening_method(
        "harden over 3 rounds against fgsm using pgd"
    ) == "pgd"
    assert aliases.extract_hardening_method(
        "harden over 4 rounds with boundary attack and hopskipjump using hopskipjump"
    ) == "hopskipjump"
    # No preposition: the only alias mentioned wins.
    assert aliases.extract_hardening_method("harden my model with fgsm") == "fgsm"


def test_hopskipjump_alias_matches_the_run_together_spelling():
    assert "hopskipjump" in aliases.extract_attacks("compare hopskipjump and zoo")
    assert aliases.extract_hardening_method("harden using hopskipjump") == "hopskipjump"


# ----------------------------------------------------------------------
# Combination assembly
# ----------------------------------------------------------------------

def test_single_pair_without_n_keeps_the_single_path():
    combos, requested = AgentPlanner._build_security_combinations(
        "run fgsm attack with feature_squeezing defense"
    )
    assert combos == []
    assert requested == 0


def test_named_combinations_are_zipped_in_order():
    combos, requested = AgentPlanner._build_security_combinations(
        "compare fgsm, pgd and cw with feature_squeezing"
    )
    assert requested == 3
    assert [c["attack_type"] for c in combos] == ["fgsm", "pgd", "cw"]
    assert all(c["defense_type"] == "feature_squeezing" for c in combos)


def test_n_is_an_upper_bound_and_is_never_padded():
    combos, requested = AgentPlanner._build_security_combinations(
        "run 5 combinations of fgsm and pgd with feature_squeezing"
    )
    assert requested == 5
    assert len(combos) == 2


def test_shorter_list_is_never_padded_with_duplicates():
    # Two attacks, one defense, three requested: only the two distinct
    # combinations run, because repeating one to reach N is forbidden.
    combos, requested = AgentPlanner._build_security_combinations(
        "run 3 rounds with feature_squeezing, fgsm and pgd"
    )
    assert requested == 3
    assert [c["attack_type"] for c in combos] == ["fgsm", "pgd"]
    assert {c["defense_type"] for c in combos} == {"feature_squeezing"}


def test_cap_of_five_is_never_exceeded():
    combos, _ = AgentPlanner._build_security_combinations(
        "run 5 rounds with fgsm, pgd, cw, zoo, hopskipjump, boundary attack"
    )
    assert len(combos) <= 5


# ----------------------------------------------------------------------
# Real model + dataset for the runner tests
# ----------------------------------------------------------------------

@pytest.fixture(scope="module")
def sklearn_assets(tmp_path_factory):
    import joblib

    root = tmp_path_factory.mktemp("security")
    rng = np.random.RandomState(0)
    X = rng.normal(size=(80, 4))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    frame = pd.DataFrame(X, columns=["f0", "f1", "f2", "f3"])
    frame["label"] = y

    data_path = root / "data.csv"
    frame.to_csv(data_path, index=False)

    model = RandomForestClassifier(n_estimators=5, random_state=0).fit(X, y)
    model_path = root / "model.joblib"
    joblib.dump(model, model_path)

    return str(model_path), str(data_path)


def test_sweep_steps_match_a_direct_single_pair_call(sklearn_assets):
    model_path, data_path = sklearn_assets
    combos = [{"attack_type": "hopskipjump", "defense_type": "feature_squeezing"}]
    seq = robustness_evaluator.run_attack_defense_sequence(
        combos, model_path=model_path, dataset_path=data_path, target_column="label"
    )
    assert seq["mode"] == "sweep"
    assert seq["step_count"] == 1

    direct = robustness_evaluator.run_attack_on_defended_model(
        "hopskipjump", "feature_squeezing",
        model_path=model_path, dataset_path=data_path, target_column="label",
    )
    for key in (
        "clean_accuracy", "defended_accuracy", "adversarial_accuracy",
        "adversarial_precision", "adversarial_recall", "adversarial_f1",
        "drop_percentage",
    ):
        assert seq["steps"][0][key] == direct[key], key


def test_sweep_tags_every_step(sklearn_assets):
    model_path, data_path = sklearn_assets
    combos = [
        {"attack_type": "hopskipjump", "defense_type": "feature_squeezing"},
        {"attack_type": "zoo", "defense_type": "feature_squeezing"},
    ]
    seq = robustness_evaluator.run_attack_defense_sequence(
        combos, model_path=model_path, dataset_path=data_path, target_column="label"
    )
    assert [s["step_index"] for s in seq["steps"]] == [1, 2]
    assert [s["attack_type"] for s in seq["steps"]] == ["hopskipjump", "zoo"]
    assert all(s["status"] == "success" for s in seq["steps"])
    assert seq["requested_steps"] == 2


def test_save_experiment_persists_the_series_onto_the_comparison_blob():
    """End-to-end through the real save tool against the test database.

    The dashboard reads per-step metrics off comparison.series, so this is the
    only assertion that actually proves the chart will have data to draw.
    """
    from app.agent.tools import tool_registry
    from app.core.database import db

    series = build_series(
        [
            {"step_index": 1, "attack_type": "zoo", "adversarial_accuracy": 0.5},
            {"step_index": 2, "attack_type": "cw", "adversarial_accuracy": 0.6},
        ],
        "sweep",
    )

    tool_registry.execute_tool(
        "save_experiment",
        {
            "project_id": "proj_series_test",
            "model_id": "model_series_test",
            "operation": "attack_defense_sequence_2",
            "parameters": {"step_count": 2},
            "input_dataset_version": "v001",
            "output_dataset_version": "v001",
            "metrics_before": {"accuracy": 1.0},
            "metrics_after": {"accuracy": 0.6},
            "series": series,
        },
    )

    with db._get_connection() as conn:
        row = conn.execute(
            "SELECT comparison FROM experiments WHERE project_id = ?",
            ("proj_series_test",),
        ).fetchone()
    assert row is not None, "save_experiment did not persist a row"

    stored = json.loads(row["comparison"])
    assert stored["series"]["count"] == 2
    assert [i["attack_type"] for i in stored["series"]["items"]] == ["zoo", "cw"]
    # The summary metrics the report and existing charts rely on are unaffected.
    assert stored["experiment_metrics"]["accuracy"] == 0.6

    with db._get_connection() as conn:
        conn.execute("DELETE FROM experiments WHERE project_id = ?", ("proj_series_test",))


def test_save_experiment_without_a_series_still_works():
    """Single-pair rows must keep the original comparison shape exactly."""
    from app.agent.tools import tool_registry
    from app.core.database import db

    tool_registry.execute_tool(
        "save_experiment",
        {
            "project_id": "proj_noseries_test",
            "model_id": "model_noseries_test",
            "operation": "fgsm_on_feature_squeezing",
            "parameters": {},
            "input_dataset_version": "v001",
            "output_dataset_version": "v001",
            "metrics_before": {"accuracy": 1.0},
            "metrics_after": {"accuracy": 0.5},
        },
    )

    with db._get_connection() as conn:
        row = conn.execute(
            "SELECT comparison FROM experiments WHERE project_id = ?",
            ("proj_noseries_test",),
        ).fetchone()
    assert row is not None
    stored = json.loads(row["comparison"])
    assert "series" not in stored
    assert stored["experiment_metrics"]["accuracy"] == 0.5

    with db._get_connection() as conn:
        conn.execute("DELETE FROM experiments WHERE project_id = ?", ("proj_noseries_test",))


# ----------------------------------------------------------------------
# Persisted series: what the dashboard charts
# ----------------------------------------------------------------------

def test_series_carries_one_entry_per_step(sklearn_assets):
    model_path, data_path = sklearn_assets
    seq = robustness_evaluator.run_attack_defense_sequence(
        [
            {"attack_type": "hopskipjump", "defense_type": "feature_squeezing"},
            {"attack_type": "zoo", "defense_type": "feature_squeezing"},
        ],
        model_path=model_path, dataset_path=data_path, target_column="label",
    )
    series = build_series(seq["steps"], "sweep")

    assert series["kind"] == "sweep"
    assert series["count"] == seq["step_count"] == 2
    assert [i["step_index"] for i in series["items"]] == [1, 2]
    assert [i["attack_type"] for i in series["items"]] == ["hopskipjump", "zoo"]
    for item in series["items"]:
        assert item["adversarial_accuracy"] is not None


def test_loop_series_carries_a_model_per_round(sklearn_assets):
    model_path, data_path = sklearn_assets
    loop = robustness_evaluator.run_hardening_loop(
        [{"attack_type": "hopskipjump", "defense_type": None}],
        model_path=model_path, dataset_path=data_path,
        target_column="label", retrain_method="hopskipjump",
    )
    series = build_series(loop["rounds"], "loop")

    assert series["kind"] == "loop"
    assert series["count"] == loop["round_count"] == 1
    # Each round records the artefact it was measured on, so a chart reader can
    # tell which model produced which number.
    assert series["items"][0]["model_id"] is not None
    assert series["items"][0]["round_index"] == 1


def test_loop_series_flags_a_mixed_threat_run():
    mixed = build_series(
        [
            {"round_index": 1, "attack_type": "fgsm", "adversarial_accuracy": 0.9},
            {"round_index": 2, "attack_type": "cw", "adversarial_accuracy": 0.8},
        ],
        "loop",
    )
    assert mixed["mixed_threat"] is True
    assert mixed["attacks_measured"] == ["cw", "fgsm"]

    single = build_series(
        [
            {"round_index": 1, "attack_type": "fgsm", "adversarial_accuracy": 0.9},
            {"round_index": 2, "attack_type": "fgsm", "adversarial_accuracy": 0.95},
        ],
        "loop",
    )
    assert single["mixed_threat"] is False


def test_series_survives_the_comparison_json_round_trip():
    """The chart reads this off the comparison blob, so it must persist."""
    series = build_series(
        [{"round_index": 1, "attack_type": "cw", "adversarial_accuracy": 0.5}],
        "loop",
    )
    comparison = {
        "baseline_metrics": {"accuracy": 1.0},
        "experiment_metrics": {"accuracy": 0.5},
        "metric_deltas": {"accuracy": -0.5},
        "series": series,
    }
    restored = json.loads(json.dumps(comparison))

    assert restored["series"]["count"] == 1
    assert restored["series"]["items"][0]["attack_type"] == "cw"
    # The pre-existing keys the report and old charts read are untouched.
    assert restored["baseline_metrics"] == {"accuracy": 1.0}
    assert restored["metric_deltas"] == {"accuracy": -0.5}


def test_series_omits_absent_metrics_rather_than_reporting_zero():
    series = build_series(
        [{"step_index": 1, "attack_type": "zoo", "clean_accuracy": 0.8}],
        "sweep",
    )
    item = series["items"][0]
    assert item["clean_accuracy"] == 0.8
    # A missing value must stay missing; writing 0.0 would read as a real
    # measurement of zero and show up as a bar dropping to nothing.
    assert "adversarial_f1" not in item


def test_sweep_refuses_more_than_five_combinations(sklearn_assets):
    model_path, data_path = sklearn_assets
    combos = [
        {"attack_type": "hopskipjump", "defense_type": "feature_squeezing"}
    ] * 6
    with pytest.raises(ValueError, match="at most 5"):
        robustness_evaluator.run_attack_defense_sequence(
            combos, model_path=model_path, dataset_path=data_path,
            target_column="label",
        )


def test_sweep_refuses_an_unsupported_attack_up_front(sklearn_assets):
    model_path, data_path = sklearn_assets
    combos = [
        {"attack_type": "hopskipjump", "defense_type": "feature_squeezing"},
        {"attack_type": "fgsm", "defense_type": "feature_squeezing"},
    ]
    with pytest.raises(aliases.IncompatibleSecurityRequest):
        robustness_evaluator.run_attack_defense_sequence(
            combos, model_path=model_path, dataset_path=data_path,
            target_column="label",
        )


def test_sweep_refuses_an_empty_combination_list(sklearn_assets):
    model_path, data_path = sklearn_assets
    with pytest.raises(ValueError, match="No attack or defense"):
        robustness_evaluator.run_attack_defense_sequence(
            [], model_path=model_path, dataset_path=data_path, target_column="label"
        )


def test_sweep_refuses_spatial_smoothing_on_tabular_data(sklearn_assets):
    model_path, data_path = sklearn_assets
    combos = [
        {"attack_type": "hopskipjump", "defense_type": "spatial_smoothing"}
    ]
    with pytest.raises(ValueError, match="Spatial Smoothing"):
        robustness_evaluator.run_attack_defense_sequence(
            combos, model_path=model_path, dataset_path=data_path,
            target_column="label",
        )


# ----------------------------------------------------------------------
# Hardening loop legality
# ----------------------------------------------------------------------

def test_loop_refuses_more_than_five_rounds(sklearn_assets):
    model_path, data_path = sklearn_assets
    combos = [{"attack_type": "hopskipjump", "defense_type": None}] * 6
    with pytest.raises(ValueError, match="at most 5"):
        robustness_evaluator.run_hardening_loop(
            combos, model_path=model_path, dataset_path=data_path,
            target_column="label",
        )


def test_loop_refuses_an_unsupported_retrain_method(sklearn_assets):
    model_path, data_path = sklearn_assets
    combos = [{"attack_type": "hopskipjump", "defense_type": None}]
    with pytest.raises(aliases.IncompatibleSecurityRequest):
        robustness_evaluator.run_hardening_loop(
            combos, model_path=model_path, dataset_path=data_path,
            target_column="label", retrain_method="fgsm",
        )


# ----------------------------------------------------------------------
# Hardened model naming must not grow without bound
# ----------------------------------------------------------------------

def test_hardened_stem_strips_existing_markers_and_reports_depth():
    from app.models.trainer import _bounded_hardened_stem

    assert _bounded_hardened_stem("wine_model") == ("wine_model", 0)
    assert _bounded_hardened_stem(
        "hardened_20260930_082121_149781_wine_model"
    ) == ("wine_model", 1)
    # A depth-3 chain is the state that produced the too-long path.
    assert _bounded_hardened_stem(
        "hardened_20260930_082129_530139_hardened_20260930_082128_956155_"
        "hardened_20260930_082121_149781_pytorch_tabular_model"
    ) == ("pytorch_tabular_model", 3)


def test_repeated_hardening_keeps_the_filename_a_fixed_length(sklearn_assets):
    """The path-length bug: each round used to nest another timestamp prefix.

    Windows rejects paths near 260 characters, so a loop that starts from an
    already-hardened model eventually failed to export at all.
    """
    from app.models.trainer import adversarial_retrainer

    model_path, data_path = sklearn_assets
    current = model_path
    lengths = []
    for _ in range(4):
        out = adversarial_retrainer.harden_and_save(
            model_path=current, dataset_path=data_path,
            target_column="label", method="hopskipjump", persist=False,
        )
        assert out["persisted"] is False
        current = out["file_path"]
        lengths.append(len(Path(current).name))

    assert len(set(lengths)) == 1, f"filename length grew across rounds: {lengths}"
    assert max(lengths) < 120


def test_loop_refuses_an_empty_round_list(sklearn_assets):
    model_path, data_path = sklearn_assets
    with pytest.raises(ValueError, match="No attack"):
        robustness_evaluator.run_hardening_loop(
            [], model_path=model_path, dataset_path=data_path,
            target_column="label",
        )


def test_loop_refuses_a_mixed_threat_run_with_an_unsupported_attack(sklearn_assets):
    model_path, data_path = sklearn_assets
    combos = [
        {"attack_type": "hopskipjump", "defense_type": None},
        {"attack_type": "fgsm", "defense_type": None},
    ]
    with pytest.raises(aliases.IncompatibleSecurityRequest):
        robustness_evaluator.run_hardening_loop(
            combos, model_path=model_path, dataset_path=data_path,
            target_column="label", retrain_method="hopskipjump",
        )
