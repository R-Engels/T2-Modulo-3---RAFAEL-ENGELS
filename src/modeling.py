"""Fases 5 e 6: otimização KNN/Árvore, CV anti-leakage, feature importance, threshold e avaliação."""

from typing import Any
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from src.config import (
    CONTINUOUS_COLS,
    CV_FOLDS,
    DEFAULT_LGD,
    KNN_K_VALUES,
    RANDOM_STATE,
    THRESHOLD_GRID,
    TREE_MAX_DEPTHS,
)
from src.preprocessing import make_sampler, scale_continuous_for_knn


def _compute_metrics(model: Any, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
    """Calcula métricas padronizadas de classificação binária."""
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]
    return {
        "accuracy": float(accuracy_score(y, y_pred)),
        "precision": float(precision_score(y, y_pred, zero_division=0)),
        "recall": float(recall_score(y, y_pred, zero_division=0)),
        "f1": float(f1_score(y, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, y_prob)),
    }


def compare_balancing_methods(
    X_train_raw: pd.DataFrame,
    y_train_raw: pd.Series,
    n_splits: int = CV_FOLDS,
    knn_k: int = 9,
    tree_depth: int = 7,
    continuous_cols: list[str] | None = None,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Compara RandomUnderSampler vs SMOTE via Stratified K-Fold CV sem data leakage."""
    cont_cols = list(continuous_cols or CONTINUOUS_COLS)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    X_reset = X_train_raw.reset_index(drop=True)
    y_reset = y_train_raw.reset_index(drop=True)

    rows: list[dict[str, Any]] = []
    for method, label in [("rus", "RandomUnderSampler"), ("smote", "SMOTE")]:
        tree_f1s, tree_aucs, knn_f1s, knn_aucs = [], [], [], []
        for tr_idx, va_idx in skf.split(X_reset, y_reset):
            Xtr, Xva = X_reset.iloc[tr_idx], X_reset.iloc[va_idx]
            ytr, yva = y_reset.iloc[tr_idx], y_reset.iloc[va_idx]

            sampler = make_sampler(method, random_state=random_state)
            Xb, yb = sampler.fit_resample(Xtr, ytr)

            # Árvore sem escalonamento
            mt = DecisionTreeClassifier(max_depth=tree_depth, random_state=random_state)
            mt.fit(Xb, yb)
            tree_f1s.append(f1_score(yva, mt.predict(Xva), zero_division=0))
            tree_aucs.append(roc_auc_score(yva, mt.predict_proba(Xva)[:, 1]))

            # KNN com StandardScaler ajustado apenas no fold de treino
            Xbk, Xvk, _ = scale_continuous_for_knn(Xb, Xva, continuous_cols=cont_cols)
            mk = KNeighborsClassifier(n_neighbors=knn_k, n_jobs=-1)
            mk.fit(Xbk, yb)
            knn_f1s.append(f1_score(yva, mk.predict(Xvk), zero_division=0))
            knn_aucs.append(roc_auc_score(yva, mk.predict_proba(Xvk)[:, 1]))

        rows.append(
            {
                "metodo": label,
                "codigo": method,
                "tree_cv_f1_mean": float(np.mean(tree_f1s)),
                "tree_cv_f1_std": float(np.std(tree_f1s)),
                "tree_cv_auc_mean": float(np.mean(tree_aucs)),
                "knn_cv_f1_mean": float(np.mean(knn_f1s)),
                "knn_cv_f1_std": float(np.std(knn_f1s)),
                "knn_cv_auc_mean": float(np.mean(knn_aucs)),
            }
        )

    return pd.DataFrame(rows)


def evaluate_knn_grid(
    X_train_knn: pd.DataFrame,
    y_train: pd.Series,
    X_test_knn: pd.DataFrame,
    y_test: pd.Series,
    k_values: list[int] | None = None,
) -> tuple[pd.DataFrame, dict[int, KNeighborsClassifier]]:
    """Treina e compara múltiplos valores de K no KNN (dados escalonados) em treino e teste."""
    ks = list(k_values or KNN_K_VALUES)
    rows: list[dict[str, Any]] = []
    models: dict[int, KNeighborsClassifier] = {}

    for k in ks:
        model = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
        model.fit(X_train_knn, y_train)
        models[k] = model

        tr = _compute_metrics(model, X_train_knn, y_train)
        te = _compute_metrics(model, X_test_knn, y_test)
        rows.append(
            {
                "param_value": k,
                "param_label": f"K={k}",
                "train_accuracy": tr["accuracy"],
                "test_accuracy": te["accuracy"],
                "train_precision": tr["precision"],
                "test_precision": te["precision"],
                "train_recall": tr["recall"],
                "test_recall": te["recall"],
                "train_f1": tr["f1"],
                "test_f1": te["f1"],
                "train_roc_auc": tr["roc_auc"],
                "test_roc_auc": te["roc_auc"],
                "gap_accuracy": tr["accuracy"] - te["accuracy"],
                "gap_f1": tr["f1"] - te["f1"],
            }
        )

    return pd.DataFrame(rows), models


def evaluate_tree_grid(
    X_train_tree: pd.DataFrame,
    y_train: pd.Series,
    X_test_tree: pd.DataFrame,
    y_test: pd.Series,
    max_depths: list[int | None] | None = None,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, dict[str, DecisionTreeClassifier]]:
    """Treina e compara múltiplos max_depth na Árvore de Decisão (sem escalonamento)."""
    depths = list(TREE_MAX_DEPTHS if max_depths is None else max_depths)
    rows: list[dict[str, Any]] = []
    models: dict[str, DecisionTreeClassifier] = {}

    for depth in depths:
        label = "Ilimitada" if depth is None else str(depth)
        model = DecisionTreeClassifier(max_depth=depth, random_state=random_state)
        model.fit(X_train_tree, y_train)
        models[label] = model

        tr = _compute_metrics(model, X_train_tree, y_train)
        te = _compute_metrics(model, X_test_tree, y_test)
        rows.append(
            {
                "param_value": depth,
                "param_label": label,
                "actual_depth": int(model.get_depth()),
                "n_leaves": int(model.get_n_leaves()),
                "train_accuracy": tr["accuracy"],
                "test_accuracy": te["accuracy"],
                "train_precision": tr["precision"],
                "test_precision": te["precision"],
                "train_recall": tr["recall"],
                "test_recall": te["recall"],
                "train_f1": tr["f1"],
                "test_f1": te["f1"],
                "train_roc_auc": tr["roc_auc"],
                "test_roc_auc": te["roc_auc"],
                "gap_accuracy": tr["accuracy"] - te["accuracy"],
                "gap_f1": tr["f1"] - te["f1"],
            }
        )

    return pd.DataFrame(rows), models


def select_best_model(
    results_df: pd.DataFrame,
    models_dict: dict[Any, Any],
    metric: str = "test_f1",
    max_gap: float = 0.16,
    key_col: str = "param_label",
) -> tuple[Any, pd.Series]:
    """Seleciona melhor modelo no teste controlando o gap de overfitting (train_f1 - test_f1)."""
    candidates = results_df[results_df["gap_f1"] <= max_gap]
    pool = candidates if not candidates.empty else results_df
    best_idx = pool[metric].idxmax()
    best_row = results_df.loc[best_idx]

    key = best_row[key_col]
    if key not in models_dict:
        key = best_row["param_value"]
    return models_dict[key], best_row


def extract_feature_importance(
    tree_model: DecisionTreeClassifier,
    feature_names: list[str],
) -> pd.DataFrame:
    """Retorna tabela ordenada da importância das variáveis da Árvore de Decisão."""
    df_fi = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": tree_model.feature_importances_,
        }
    ).sort_values("importance", ascending=False, ignore_index=True)
    df_fi["importance_pct"] = (df_fi["importance"] * 100.0).round(2)
    return df_fi


def evaluate_thresholds(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    ticket_medio: float,
    taxa_juros_media: float,
    lgd: float = DEFAULT_LGD,
    thresholds: list[float] | None = None,
) -> pd.DataFrame:
    """Avalia sensibilidade de métricas e perda financeira estimada ao variar o ponto de corte."""
    thrs = list(thresholds or THRESHOLD_GRID)
    y_prob = model.predict_proba(X_test)[:, 1]
    custo_fn = ticket_medio * lgd
    custo_fp = ticket_medio * taxa_juros_media

    rows: list[dict[str, Any]] = []
    for thr in thrs:
        y_pred = (y_prob >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        loss_fp = float(fp * custo_fp)
        loss_fn = float(fn * custo_fn)
        rows.append(
            {
                "threshold": float(thr),
                "accuracy": float(accuracy_score(y_test, y_pred)),
                "precision": float(precision_score(y_test, y_pred, zero_division=0)),
                "recall": float(recall_score(y_test, y_pred, zero_division=0)),
                "f1": float(f1_score(y_test, y_pred, zero_division=0)),
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
                "loss_fp_usd": loss_fp,
                "loss_fn_usd": loss_fn,
                "total_loss_usd": loss_fp + loss_fn,
            }
        )
    return pd.DataFrame(rows)


def build_final_evaluation(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str,
) -> dict[str, Any]:
    """Gera Classification Report, Matriz de Confusão, curva ROC e taxas de erro FP/FN."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    roc_fpr, roc_tpr, _ = roc_curve(y_test, y_prob)

    report_text = classification_report(
        y_test,
        y_pred,
        target_names=["Bom Pagador (0)", "Inadimplente (1)"],
        digits=4,
    )
    report_dict = classification_report(y_test, y_pred, output_dict=True)

    return {
        "model_name": model_name,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "confusion_matrix": cm,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "fpr": float(fp / (fp + tn)),
        "fnr": float(fn / (fn + tp)),
        "roc_fpr": roc_fpr,
        "roc_tpr": roc_tpr,
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
        "classification_report_text": report_text,
        "classification_report_dict": report_dict,
    }
