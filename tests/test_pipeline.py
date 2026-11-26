"""Suíte expandida de testes automatizados (dados, anti-leakage, modelagem, negócio e plots)."""

from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from src.config import (
    CONTINUOUS_COLS,
    FEATURE_ENGINEERED_COL,
    MAX_VALID_AGE,
    MAX_VALID_EMP_LENGTH,
    TARGET_COL,
)
from src.data_prep import clean_data, eda_summary, engineer_features, load_dataset
from src.modeling import (
    build_final_evaluation,
    compare_balancing_methods,
    evaluate_knn_grid,
    evaluate_thresholds,
    evaluate_tree_grid,
    extract_feature_importance,
    select_best_model,
)
from src.plots import (
    plot_confusion_matrices,
    plot_correlation_heatmap,
    plot_eda_distributions,
    plot_feature_importance,
    plot_outliers_comparison,
    plot_overfitting_curves,
    plot_roc_and_threshold,
    plot_target_imbalance,
)
from src.preprocessing import encode_features, make_sampler, split_balance_and_scale


@pytest.fixture(scope="module")
def raw_df() -> pd.DataFrame:
    return load_dataset()


@pytest.fixture(scope="module")
def clean_df_and_report(raw_df: pd.DataFrame):
    return clean_data(raw_df)


@pytest.fixture(scope="module")
def encoded_df(clean_df_and_report) -> pd.DataFrame:
    df_clean, _ = clean_df_and_report
    return encode_features(engineer_features(df_clean))


@pytest.fixture(scope="module")
def sample_bundle(encoded_df: pd.DataFrame) -> dict:
    sample = encoded_df.sample(n=4500, random_state=42)
    return split_balance_and_scale(sample, balancing_method="rus")


# --- FASE 1: CARGA E EDA ---


def test_load_dataset_and_eda_summary_contract(raw_df: pd.DataFrame):
    assert raw_df.shape == (32581, 12)
    assert TARGET_COL in raw_df.columns
    eda = eda_summary(raw_df)
    assert eda["shape"] == (32581, 12)
    assert eda["duplicates"] == 165
    assert eda["null_counts"]["person_emp_length"] > 0
    assert eda["null_counts"]["loan_int_rate"] > 0
    assert set(eda["target_counts"].index) == {0, 1}
    assert eda["target_pct"][0] > 70.0


# --- FASE 2: LIMPEZA E OUTLIERS ---


def test_clean_data_removes_duplicates_anomalies_and_nulls(raw_df: pd.DataFrame, clean_df_and_report):
    df_clean, report = clean_df_and_report
    assert report["duplicates_removed"] == 165
    assert report["invalid_age_removed"] == 5
    assert report["invalid_emp_removed"] == 2
    assert df_clean.duplicated().sum() == 0
    assert df_clean.isna().sum().sum() == 0
    assert (df_clean["person_age"] <= MAX_VALID_AGE).all()
    assert (df_clean["person_emp_length"] <= MAX_VALID_EMP_LENGTH).all()
    assert df_clean["person_income"].max() <= report["income_cap_p99"] + 1e-6


def test_clean_data_does_not_mutate_input_dataframe(raw_df: pd.DataFrame):
    snapshot = raw_df.head(50).copy(deep=True)
    subset = raw_df.head(50).copy(deep=True)
    clean_data(subset)
    pd.testing.assert_frame_equal(subset, snapshot)


# --- FASE 3: FEATURE ENGINEERING ---


def test_engineer_features_formula_and_non_mutation(clean_df_and_report):
    df_clean, _ = clean_df_and_report
    before_cols = list(df_clean.columns)
    df_feat = engineer_features(df_clean)

    assert list(df_clean.columns) == before_cols
    assert FEATURE_ENGINEERED_COL in df_feat.columns
    expected = (df_clean["loan_amnt"] / df_clean["person_income"]) * 100.0
    assert np.allclose(df_feat[FEATURE_ENGINEERED_COL], expected)
    assert np.isfinite(df_feat[FEATURE_ENGINEERED_COL]).all()
    assert (df_feat[FEATURE_ENGINEERED_COL] > 0).all()


@pytest.mark.parametrize(
    "col,bad_value",
    [
        ("person_income", 0),
        ("person_income", -5000),
        ("person_income", np.nan),
        ("loan_amnt", np.nan),
    ],
)
def test_engineer_features_rejects_invalid_inputs(clean_df_and_report, col: str, bad_value):
    df_clean, _ = clean_df_and_report
    corrupted = df_clean.head(5).copy()
    corrupted.loc[corrupted.index[0], col] = bad_value
    with pytest.raises(ValueError):
        engineer_features(corrupted)


# --- FASE 4: ENCODING, SPLIT, BALANCEAMENTO E ESCALONAMENTO ---


def test_encode_features_types_and_invalid_category_guards(clean_df_and_report):
    df_clean, _ = clean_df_and_report
    df_feat = engineer_features(df_clean)
    df_enc = encode_features(df_feat)

    assert df_enc.select_dtypes(exclude="number").empty
    assert df_enc.isna().sum().sum() == 0
    assert set(df_enc["cb_person_default_on_file"].unique()) == {0, 1}
    assert set(df_enc["loan_grade"].unique()) == set(range(7))

    bad_grade = df_feat.head(5).copy()
    bad_grade.loc[bad_grade.index[0], "loan_grade"] = "Z"
    with pytest.raises(ValueError):
        encode_features(bad_grade)

    bad_default = df_feat.head(5).copy()
    bad_default.loc[bad_default.index[0], "cb_person_default_on_file"] = "UNKNOWN"
    with pytest.raises(ValueError):
        encode_features(bad_default)


def test_make_sampler_rejects_unknown_method():
    with pytest.raises(ValueError):
        make_sampler("invalid_method")


@pytest.mark.parametrize("method", ["rus", "smote"])
def test_split_balance_and_scale_strict_anti_leakage(encoded_df: pd.DataFrame, method: str):
    subset = encoded_df.sample(n=3000, random_state=42)
    bundle = split_balance_and_scale(subset, balancing_method=method)

    # Zero interseção de índices entre treino e teste
    train_idx = set(bundle["X_train_raw"].index)
    test_idx = set(bundle["X_test"].index)
    assert train_idx.isdisjoint(test_idx)

    # Split estratificado 80/20
    assert abs(len(bundle["X_test"]) / len(subset) - 0.20) < 0.005
    assert abs(bundle["y_train_raw"].mean() - bundle["y_test"].mean()) < 0.005

    # Balanceamento 50/50 apenas no treino; teste mantém desbalanceamento original
    bal_counts = bundle["y_train_bal"].value_counts()
    assert bal_counts[0] == bal_counts[1]
    assert bundle["y_test"].mean() < 0.30

    # StandardScaler exclusivamente nas contínuas do KNN (fit no treino balanceado)
    knn_means = bundle["X_train_knn"][CONTINUOUS_COLS].mean().abs()
    knn_stds = bundle["X_train_knn"][CONTINUOUS_COLS].std(ddof=0)
    assert (knn_means < 1e-6).all()
    assert ((knn_stds - 1.0).abs() < 1e-6).all()

    # Colunas binárias/categóricas permanecem intactas no KNN
    assert set(bundle["X_train_knn"]["cb_person_default_on_file"].unique()).issubset({0, 1})

    # Árvore treinada rigorosamente SEM escalonamento
    assert bundle["X_train_tree"]["person_income"].max() > 1000.0
    assert bundle["X_train_tree"].equals(bundle["X_train_bal"])
    assert bundle["X_test_tree"].equals(bundle["X_test"])
    assert TARGET_COL not in bundle["X_train_knn"].columns


# --- FASE 5: CV, MODELAGEM E DETECÇÃO DE OVERFITTING ---


def test_compare_balancing_methods_cv_no_leakage(sample_bundle: dict):
    cv_df = compare_balancing_methods(
        sample_bundle["X_train_raw"],
        sample_bundle["y_train_raw"],
        n_splits=3,
        knn_k=7,
        tree_depth=5,
    )
    assert set(cv_df["codigo"]) == {"rus", "smote"}
    assert (cv_df["tree_cv_f1_mean"] > 0.5).all()
    assert (cv_df["knn_cv_f1_mean"] > 0.4).all()
    assert (cv_df["tree_cv_f1_std"] >= 0.0).all()


def test_modeling_grids_detect_overfitting_and_select_generalizing_model(sample_bundle: dict):
    knn_df, knn_models = evaluate_knn_grid(
        sample_bundle["X_train_knn"],
        sample_bundle["y_train_bal"],
        sample_bundle["X_test_knn"],
        sample_bundle["y_test"],
        k_values=[3, 5, 7, 9],
    )
    tree_df, tree_models = evaluate_tree_grid(
        sample_bundle["X_train_tree"],
        sample_bundle["y_train_bal"],
        sample_bundle["X_test_tree"],
        sample_bundle["y_test"],
        max_depths=[3, 5, 7, None],
    )

    assert len(knn_df) == 4
    assert len(tree_df) == 4
    unlimited_row = tree_df[tree_df["param_label"] == "Ilimitada"].iloc[0]
    depth5_row = tree_df[tree_df["param_label"] == "5"].iloc[0]
    assert unlimited_row["train_f1"] > 0.99
    assert unlimited_row["gap_f1"] > depth5_row["gap_f1"]

    best_tree, best_tree_row = select_best_model(tree_df, tree_models, key_col="param_label")
    assert best_tree_row["param_label"] != "Ilimitada"
    assert hasattr(best_tree, "predict_proba")

    best_knn, _ = select_best_model(knn_df, knn_models, key_col="param_value")
    assert hasattr(best_knn, "predict_proba")


# --- FASE 6: AVALIAÇÃO FINAL, FEATURE IMPORTANCE E THRESHOLD ---


def test_final_evaluation_and_feature_importance(sample_bundle: dict):
    tree_df, tree_models = evaluate_tree_grid(
        sample_bundle["X_train_tree"],
        sample_bundle["y_train_bal"],
        sample_bundle["X_test_tree"],
        sample_bundle["y_test"],
        max_depths=[5, 7],
    )
    best_tree, row = select_best_model(tree_df, tree_models, key_col="param_label")
    ev = build_final_evaluation(
        best_tree, sample_bundle["X_test_tree"], sample_bundle["y_test"], f"Tree {row['param_label']}"
    )

    assert ev["confusion_matrix"].shape == (2, 2)
    assert ev["tn"] + ev["fp"] + ev["fn"] + ev["tp"] == len(sample_bundle["y_test"])
    assert 0.5 < ev["roc_auc"] <= 1.0
    assert "Inadimplente (1)" in ev["classification_report_text"]

    fi_df = extract_feature_importance(best_tree, sample_bundle["feature_names"])
    assert abs(fi_df["importance"].sum() - 1.0) < 1e-6
    assert fi_df["importance"].is_monotonic_decreasing
    assert FEATURE_ENGINEERED_COL in set(fi_df["feature"])


def test_threshold_analysis_monotonicity_and_financial_formula(sample_bundle: dict):
    _, tree_models = evaluate_tree_grid(
        sample_bundle["X_train_tree"],
        sample_bundle["y_train_bal"],
        sample_bundle["X_test_tree"],
        sample_bundle["y_test"],
        max_depths=[7],
    )
    model = tree_models["7"]
    thr_df = evaluate_thresholds(
        model,
        sample_bundle["X_test_tree"],
        sample_bundle["y_test"],
        ticket_medio=10000.0,
        taxa_juros_media=0.10,
        lgd=0.60,
        thresholds=[0.20, 0.50, 0.80],
    )

    # Aumentar threshold reduz (ou mantém) Recall e FPs, e aumenta (ou mantém) FNs
    assert thr_df["recall"].is_monotonic_decreasing
    assert thr_df["fp"].is_monotonic_decreasing
    assert thr_df["fn"].is_monotonic_increasing

    # Verificação exata da fórmula financeira
    for _, r in thr_df.iterrows():
        expected_loss = r["fp"] * (10000.0 * 0.10) + r["fn"] * (10000.0 * 0.60)
        assert abs(r["total_loss_usd"] - expected_loss) < 1e-4


# --- GRÁFICOS (SMOKE TEST COMPLETO) ---


def test_all_plots_render_and_save_non_empty_png(raw_df: pd.DataFrame, clean_df_and_report, sample_bundle: dict, tmp_path: Path):
    df_clean, _ = clean_df_and_report
    knn_df, knn_models = evaluate_knn_grid(
        sample_bundle["X_train_knn"],
        sample_bundle["y_train_bal"],
        sample_bundle["X_test_knn"],
        sample_bundle["y_test"],
        k_values=[3, 7],
    )
    tree_df, tree_models = evaluate_tree_grid(
        sample_bundle["X_train_tree"],
        sample_bundle["y_train_bal"],
        sample_bundle["X_test_tree"],
        sample_bundle["y_test"],
        max_depths=[5, None],
    )
    knn_eval = build_final_evaluation(knn_models[7], sample_bundle["X_test_knn"], sample_bundle["y_test"], "KNN (K=7)")
    tree_eval = build_final_evaluation(tree_models["5"], sample_bundle["X_test_tree"], sample_bundle["y_test"], "Árvore (d=5)")
    fi_df = extract_feature_importance(tree_models["5"], sample_bundle["feature_names"])
    thr_df = evaluate_thresholds(tree_models["5"], sample_bundle["X_test_tree"], sample_bundle["y_test"], 9500.0, 0.11)

    outputs = [
        (plot_eda_distributions(raw_df, tmp_path / "p1.png"), tmp_path / "p1.png"),
        (plot_target_imbalance(raw_df, tmp_path / "p2.png"), tmp_path / "p2.png"),
        (plot_correlation_heatmap(raw_df, tmp_path / "p3.png"), tmp_path / "p3.png"),
        (plot_outliers_comparison(raw_df, df_clean, tmp_path / "p4.png"), tmp_path / "p4.png"),
        (plot_overfitting_curves(knn_df, tree_df, tmp_path / "p5.png"), tmp_path / "p5.png"),
        (plot_confusion_matrices(knn_eval, tree_eval, tmp_path / "p6.png"), tmp_path / "p6.png"),
        (plot_feature_importance(fi_df, save_path=tmp_path / "p7.png"), tmp_path / "p7.png"),
        (plot_roc_and_threshold(knn_eval, tree_eval, thr_df, tmp_path / "p8.png"), tmp_path / "p8.png"),
    ]
    for fig, path in outputs:
        assert path.exists()
        assert path.stat().st_size > 5000
        plt.close(fig)
