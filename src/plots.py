"""Gráficos padronizados para EDA, outliers, overfitting, importância, ROC e ponto de corte."""

from pathlib import Path
from typing import Any
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.config import FEATURE_ENGINEERED_COL, TARGET_COL

sns.set_theme(style="whitegrid", palette="deep")


def _save_fig(fig: plt.Figure, save_path: Path | None) -> plt.Figure:
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_eda_distributions(
    df: pd.DataFrame, save_path: Path | None = None
) -> plt.Figure:
    """Plota histogramas das principais variáveis numéricas para diagnóstico de simetria."""
    cols = [
        ("person_age", "Idade do Solicitante"),
        ("person_income", "Renda Anual (USD)"),
        ("person_emp_length", "Tempo de Emprego (anos)"),
        ("loan_amnt", "Valor do Empréstimo (USD)"),
        ("loan_int_rate", "Taxa de Juros (%)"),
        ("loan_percent_income", "Proporção Empréstimo/Renda"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, (col, title) in zip(axes.flat, cols):
        series = df[col].dropna()
        sns.histplot(series, bins=35, kde=True, ax=ax, color="#2b6cb0")
        ax.set_title(f"{title}\n(skew = {series.skew():.2f})", fontsize=10, fontweight="bold")
        ax.set_xlabel(col)
        ax.set_ylabel("Frequência")
    fig.suptitle("Fase 1 (EDA) - Distribuição das Variáveis Numéricas (Base Bruta)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return _save_fig(fig, save_path)


def plot_target_imbalance(
    df: pd.DataFrame, save_path: Path | None = None
) -> plt.Figure:
    """Plota gráfico de barras evidenciando o desbalanceamento da variável alvo loan_status."""
    counts = df[TARGET_COL].value_counts().sort_index()
    total = counts.sum()
    labels = ["0 - Em Dia (Bom Pagador)", "1 - Inadimplente (Calote)"]
    colors = ["#2f855a", "#c53030"]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(labels, counts.values, color=colors, width=0.55)
    for bar, count in zip(bars, counts.values):
        pct = count / total * 100
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            count + total * 0.015,
            f"{count:,}\n({pct:.2f}%)",
            ha="center",
            va="bottom",
            fontweight="bold",
        )
    ax.set_ylim(0, counts.max() * 1.18)
    ax.set_ylabel("Quantidade de Clientes")
    ax.set_title("Fase 1 (EDA) - Desbalanceamento da Variável Alvo (loan_status)", fontweight="bold")
    fig.tight_layout()
    return _save_fig(fig, save_path)


def plot_correlation_heatmap(
    df: pd.DataFrame, save_path: Path | None = None
) -> plt.Figure:
    """Plota mapa de calor de correlação de Pearson entre variáveis numéricas e alvo."""
    corr = df.select_dtypes(include="number").corr(method="pearson")
    fig, ax = plt.subplots(figsize=(10, 7.5))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title("Fase 1 (EDA) - Mapa de Calor de Correlação de Pearson", fontweight="bold")
    fig.tight_layout()
    return _save_fig(fig, save_path)


def plot_outliers_comparison(
    df_raw: pd.DataFrame,
    df_clean: pd.DataFrame,
    save_path: Path | None = None,
) -> plt.Figure:
    """Compara boxplots antes e depois do tratamento de outliers nas colunas críticas."""
    cols = [
        ("person_age", "Idade"),
        ("person_emp_length", "Tempo de Emprego"),
        ("person_income", "Renda Anual"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 7))
    for i, (col, title) in enumerate(cols):
        sns.boxplot(x=df_raw[col].dropna(), ax=axes[0, i], color="#e53e3e")
        axes[0, i].set_title(f"{title} - Antes (Bruto)", fontweight="bold")
        sns.boxplot(x=df_clean[col].dropna(), ax=axes[1, i], color="#3182ce")
        axes[1, i].set_title(f"{title} - Depois (Tratado)", fontweight="bold")
    fig.suptitle("Fase 2 (Data Prep) - Diagnóstico e Tratamento de Outliers via Boxplots", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return _save_fig(fig, save_path)


def plot_overfitting_curves(
    knn_df: pd.DataFrame,
    tree_df: pd.DataFrame,
    save_path: Path | None = None,
) -> plt.Figure:
    """Plota curvas Treino vs Teste (F1-Score e Acurácia) para diagnosticar overfitting."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))

    ax = axes[0]
    x_knn = knn_df["param_label"].tolist()
    ax.plot(x_knn, knn_df["train_f1"], marker="o", linewidth=2, label="Treino F1", color="#2b6cb0")
    ax.plot(x_knn, knn_df["test_f1"], marker="s", linewidth=2, label="Teste F1", color="#dd6b20")
    ax.plot(x_knn, knn_df["train_accuracy"], marker="^", linestyle="--", alpha=0.7, label="Treino Acurácia", color="#63b3ed")
    ax.plot(x_knn, knn_df["test_accuracy"], marker="v", linestyle="--", alpha=0.7, label="Teste Acurácia", color="#f6ad55")
    ax.set_title("Otimização KNN - Treino vs Teste por K", fontweight="bold")
    ax.set_xlabel("Número de Vizinhos (n_neighbors)")
    ax.set_ylabel("Score")
    ax.set_ylim(0.55, 1.05)
    ax.legend(loc="center right")

    ax = axes[1]
    x_tree = tree_df["param_label"].tolist()
    ax.plot(x_tree, tree_df["train_f1"], marker="o", linewidth=2, label="Treino F1", color="#2f855a")
    ax.plot(x_tree, tree_df["test_f1"], marker="s", linewidth=2, label="Teste F1", color="#c53030")
    ax.plot(x_tree, tree_df["train_accuracy"], marker="^", linestyle="--", alpha=0.7, label="Treino Acurácia", color="#68d391")
    ax.plot(x_tree, tree_df["test_accuracy"], marker="v", linestyle="--", alpha=0.7, label="Teste Acurácia", color="#fc8181")
    ax.set_title("Otimização Árvore de Decisão - Treino vs Teste por max_depth", fontweight="bold")
    ax.set_xlabel("Profundidade Máxima (max_depth)")
    ax.set_ylabel("Score")
    ax.set_ylim(0.55, 1.05)
    ax.legend(loc="lower left")

    fig.suptitle("Fase 5 (Modelagem) - Diagnóstico de Overfitting (Treino vs Teste)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return _save_fig(fig, save_path)


def plot_confusion_matrices(
    knn_eval: dict[str, Any],
    tree_eval: dict[str, Any],
    save_path: Path | None = None,
) -> plt.Figure:
    """Plota lado a lado as Matrizes de Confusão dos melhores modelos (KNN vs Árvore)."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    tick_labels = ["Bom Pagador (0)", "Inadimplente (1)"]

    for ax, ev, cmap in zip(axes, [knn_eval, tree_eval], ["Blues", "Greens"]):
        cm = ev["confusion_matrix"]
        total = cm.sum()
        cell_tags = [["VN", "FP\n(Bom negado)"], ["FN\n(Calote aprovado)", "VP"]]
        annot = np.empty_like(cm, dtype=object)
        for r in range(2):
            for c in range(2):
                annot[r, c] = f"{cell_tags[r][c]}\n{cm[r, c]:,}\n({cm[r, c] / total * 100:.1f}%)"

        sns.heatmap(
            cm,
            annot=annot,
            fmt="",
            cmap=cmap,
            cbar=False,
            xticklabels=tick_labels,
            yticklabels=tick_labels,
            ax=ax,
            linewidths=0.8,
        )
        ax.set_title(
            f"{ev['model_name']}\nF1={ev['f1']:.4f} | Recall={ev['recall']:.4f} | Prec={ev['precision']:.4f}",
            fontweight="bold",
        )
        ax.set_xlabel("Predição do Modelo")
        ax.set_ylabel("Valor Real")

    fig.suptitle("Fase 6 (Avaliação) - Matrizes de Confusão Comparativas (Teste)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return _save_fig(fig, save_path)


def plot_feature_importance(
    fi_df: pd.DataFrame,
    top_n: int = 10,
    save_path: Path | None = None,
) -> plt.Figure:
    """Plota as variáveis mais influentes da Árvore vencedora destacando comprometimento_renda."""
    top = fi_df.head(top_n).iloc[::-1]
    colors = [
        "#dd6b20" if feat == FEATURE_ENGINEERED_COL else "#2f855a"
        for feat in top["feature"]
    ]
    fig, ax = plt.subplots(figsize=(9, 5.2))
    bars = ax.barh(top["feature"], top["importance_pct"], color=colors)
    for bar, pct in zip(bars, top["importance_pct"]):
        ax.text(pct + 0.5, bar.get_y() + bar.get_height() / 2, f"{pct:.1f}%", va="center", fontsize=9, fontweight="bold")
    ax.set_xlim(0, top["importance_pct"].max() * 1.18)
    ax.set_xlabel("Importância Relativa (%)")
    ax.set_title("Explicabilidade - Top 10 Variáveis da Árvore de Decisão (Laranja = Feature Criada)", fontweight="bold")
    fig.tight_layout()
    return _save_fig(fig, save_path)


def plot_roc_and_threshold(
    knn_eval: dict[str, Any],
    tree_eval: dict[str, Any],
    thr_df: pd.DataFrame,
    save_path: Path | None = None,
) -> plt.Figure:
    """Plota Curvas ROC comparativas e análise financeira do ponto de corte (threshold)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))

    ax = axes[0]
    ax.plot(knn_eval["roc_fpr"], knn_eval["roc_tpr"], linewidth=2, color="#2b6cb0",
            label=f"{knn_eval['model_name']} (AUC = {knn_eval['roc_auc']:.4f})")
    ax.plot(tree_eval["roc_fpr"], tree_eval["roc_tpr"], linewidth=2, color="#2f855a",
            label=f"{tree_eval['model_name']} (AUC = {tree_eval['roc_auc']:.4f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Aleatório (AUC = 0.50)")
    ax.set_xlabel("Taxa de Falsos Positivos (FPR)")
    ax.set_ylabel("Taxa de Verdadeiros Positivos (Recall)")
    ax.set_title("Curva ROC - Poder Discriminatório (Teste)", fontweight="bold")
    ax.legend(loc="lower right")

    ax1 = axes[1]
    ax1.plot(thr_df["threshold"], thr_df["total_loss_usd"] / 1e6, marker="o", linewidth=2.2,
             color="#c53030", label="Perda Estimada (USD milhões)")
    best_thr_row = thr_df.loc[thr_df["total_loss_usd"].idxmin()]
    ax1.axvline(best_thr_row["threshold"], color="#c53030", linestyle=":", alpha=0.8)
    ax1.set_xlabel("Ponto de Corte de Probabilidade (Threshold)")
    ax1.set_ylabel("Perda Financeira Estimada (USD milhões)", color="#c53030")
    ax1.tick_params(axis="y", labelcolor="#c53030")

    ax2 = ax1.twinx()
    ax2.plot(thr_df["threshold"], thr_df["f1"], marker="s", linestyle="--", color="#2f855a", label="F1-Score")
    ax2.plot(thr_df["threshold"], thr_df["recall"], marker="^", linestyle="--", color="#2b6cb0", label="Recall (Calotes)")
    ax2.set_ylabel("F1-Score / Recall", color="#2f855a")
    ax2.set_ylim(0.40, 1.02)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=8.5)
    ax1.set_title(
        f"Análise de Ponto de Corte - Mínima Perda em thr={best_thr_row['threshold']:.2f}",
        fontweight="bold",
    )

    fig.suptitle("Fase 6 (Negócios) - Curva ROC e Impacto Financeiro do Ponto de Corte", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return _save_fig(fig, save_path)
