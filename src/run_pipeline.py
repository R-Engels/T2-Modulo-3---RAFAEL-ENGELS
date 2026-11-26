"""Ponto de entrada CLI para executar o pipeline completo de Risco de Crédito."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import DEFAULT_LGD, FIGURES_DIR
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
from src.preprocessing import encode_features, split_balance_and_scale


def run_all() -> dict:
    """Executa todas as 6 fases do pipeline, salva figuras e retorna artefatos."""
    df_raw = load_dataset()
    eda = eda_summary(df_raw)

    df_clean, clean_report = clean_data(df_raw)
    df_feat = engineer_features(df_clean)
    df_enc = encode_features(df_feat)
    bundle = split_balance_and_scale(df_enc)

    cv_balancing_df = compare_balancing_methods(
        bundle["X_train_raw"],
        bundle["y_train_raw"],
    )

    knn_df, knn_models = evaluate_knn_grid(
        bundle["X_train_knn"],
        bundle["y_train_bal"],
        bundle["X_test_knn"],
        bundle["y_test"],
    )
    tree_df, tree_models = evaluate_tree_grid(
        bundle["X_train_tree"],
        bundle["y_train_bal"],
        bundle["X_test_tree"],
        bundle["y_test"],
    )

    best_knn_model, best_knn_row = select_best_model(knn_df, knn_models, key_col="param_value")
    best_tree_model, best_tree_row = select_best_model(tree_df, tree_models, key_col="param_label")

    knn_eval = build_final_evaluation(
        best_knn_model,
        bundle["X_test_knn"],
        bundle["y_test"],
        f"KNN ({best_knn_row['param_label']})",
    )
    tree_eval = build_final_evaluation(
        best_tree_model,
        bundle["X_test_tree"],
        bundle["y_test"],
        f"Árvore de Decisão (max_depth={best_tree_row['param_label']})",
    )

    fi_df = extract_feature_importance(best_tree_model, bundle["feature_names"])
    ticket_medio = float(df_clean["loan_amnt"].mean())
    taxa_juros_media = float(df_clean["loan_int_rate"].mean() / 100.0)
    thr_df = evaluate_thresholds(
        best_tree_model,
        bundle["X_test_tree"],
        bundle["y_test"],
        ticket_medio=ticket_medio,
        taxa_juros_media=taxa_juros_media,
        lgd=DEFAULT_LGD,
    )

    for fig in [
        plot_eda_distributions(df_raw, FIGURES_DIR / "01_eda_distribuicoes.png"),
        plot_target_imbalance(df_raw, FIGURES_DIR / "02_eda_desbalanceamento_alvo.png"),
        plot_correlation_heatmap(df_raw, FIGURES_DIR / "03_eda_correlacao_pearson.png"),
        plot_outliers_comparison(df_raw, df_clean, FIGURES_DIR / "04_outliers_boxplots.png"),
        plot_overfitting_curves(knn_df, tree_df, FIGURES_DIR / "05_overfitting_knn_vs_arvore.png"),
        plot_confusion_matrices(knn_eval, tree_eval, FIGURES_DIR / "06_matrizes_confusao_comparacao.png"),
        plot_feature_importance(fi_df, save_path=FIGURES_DIR / "07_feature_importance_arvore.png"),
        plot_roc_and_threshold(knn_eval, tree_eval, thr_df, FIGURES_DIR / "08_roc_e_ponto_de_corte.png"),
    ]:
        plt.close(fig)

    return {
        "eda": eda,
        "clean_report": clean_report,
        "bundle": bundle,
        "cv_balancing": cv_balancing_df,
        "knn_results": knn_df,
        "tree_results": tree_df,
        "best_knn_row": best_knn_row,
        "best_tree_row": best_tree_row,
        "knn_eval": knn_eval,
        "tree_eval": tree_eval,
        "feature_importance": fi_df,
        "thresholds": thr_df,
        "ticket_medio": ticket_medio,
        "taxa_juros_media": taxa_juros_media,
    }


def main() -> None:
    res = run_all()
    print("=== FASE 2: RELATÓRIO DE LIMPEZA ===")
    for k, v in res["clean_report"].items():
        print(f"  {k}: {v}")

    print("\n=== FASE 4: BALANCEAMENTO E CV 5-FOLD (ANTI-LEAKAGE) ===")
    b = res["bundle"]
    print(f"  Método selecionado: {b['balancing_method'].upper()}")
    print("  Treino bruto:", b["X_train_raw"].shape, b["y_train_raw"].value_counts().to_dict())
    print("  Treino balanceado:", b["X_train_bal"].shape, b["y_train_bal"].value_counts().to_dict())
    print("  Teste (intacto):", b["X_test"].shape, b["y_test"].value_counts().to_dict())
    print(res["cv_balancing"].to_string(index=False))

    print("\n=== FASE 5: RESULTADOS KNN ===")
    cols = ["param_label", "train_accuracy", "test_accuracy", "train_f1", "test_f1", "test_recall", "test_precision", "test_roc_auc", "gap_f1"]
    print(res["knn_results"][cols].to_string(index=False))

    print("\n=== FASE 5: RESULTADOS ÁRVORE DE DECISÃO ===")
    tcols = ["param_label", "actual_depth", "n_leaves", "train_accuracy", "test_accuracy", "train_f1", "test_f1", "test_recall", "test_precision", "test_roc_auc", "gap_f1"]
    print(res["tree_results"][tcols].to_string(index=False))

    print(f"\n=== FASE 6: MELHOR KNN ({res['knn_eval']['model_name']}) ===")
    print(res["knn_eval"]["classification_report_text"])
    print("  CM:", res["knn_eval"]["confusion_matrix"].tolist(), f"AUC={res['knn_eval']['roc_auc']:.4f}, FPR={res['knn_eval']['fpr']:.4f}, FNR={res['knn_eval']['fnr']:.4f}")

    print(f"\n=== FASE 6: MELHOR ÁRVORE ({res['tree_eval']['model_name']}) ===")
    print(res["tree_eval"]["classification_report_text"])
    print("  CM:", res["tree_eval"]["confusion_matrix"].tolist(), f"AUC={res['tree_eval']['roc_auc']:.4f}, FPR={res['tree_eval']['fpr']:.4f}, FNR={res['tree_eval']['fnr']:.4f}")

    print("\n=== TOP 8 FEATURE IMPORTANCE (ÁRVORE) ===")
    print(res["feature_importance"].head(8).to_string(index=False))

    print("\n=== ANÁLISE DE PONTO DE CORTE (THRESHOLD) ===")
    print(res["thresholds"][["threshold", "accuracy", "precision", "recall", "f1", "fp", "fn", "total_loss_usd"]].to_string(index=False))


if __name__ == "__main__":
    main()
