"""
Módulo de Evaluación Científica y Generación de Figuras de Alta Resolución (300 DPI)
Proyecto: Predicción Temprana de la Deserción Estudiantil Universitaria mediante Redes Neuronales Profundas Regularizadas
Autor: William David Ortiz Mamani (UPeU - Ingeniería de Sistemas)
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def generate_all_figures():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    proc_dir = os.path.join(base_dir, "data", "processed")
    tables_dir = os.path.join(base_dir, "results", "tables")
    figures_dir = os.path.join(base_dir, "results", "figures")
    os.makedirs(figures_dir, exist_ok=True)

    with open(os.path.join(proc_dir, "metadata.json"), "r", encoding="utf-8") as f:
        metadata = json.load(f)

    with open(os.path.join(tables_dir, "metricas_detalladas.json"), "r", encoding="utf-8") as f:
        results = json.load(f)

    X_raw = np.load(os.path.join(proc_dir, "X_raw_full.npy"))
    y_raw = np.load(os.path.join(proc_dir, "y_raw_full.npy"))
    feature_cols = metadata["feature_columns"]
    feature_labels = metadata["feature_labels_es"]

    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.size"] = 10
    plt.rcParams["axes.edgecolor"] = "#333333"
    plt.rcParams["axes.linewidth"] = 0.8

    # =========================================================================
    # FIGURA 1: Exploración del Dataset (Distribución y Correlaciones con Deserción)
    # =========================================================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.2), gridspec_kw={"width_ratios": [1.0, 1.45]})

    # Panel A: Distribución Estratificada por Partición
    splits = ["Entrenamiento\n(70% | n=4,200)", "Validación\n(15% | n=900)", "Prueba\n(15% | n=900)"]
    n_totals = [metadata["train_samples"], metadata["val_samples"], metadata["test_samples"]]
    pos_rates = [metadata["positive_rate_train"], metadata["positive_rate_val"], metadata["positive_rate_test"]]
    pos_counts = [int(round(n * r)) for n, r in zip(n_totals, pos_rates)]
    neg_counts = [n - p for n, p in zip(n_totals, pos_counts)]

    x_pos = np.arange(len(splits))
    w = 0.36
    b1 = ax1.bar(x_pos - w / 2, neg_counts, width=w, color="#1f77b4", edgecolor="black", alpha=0.88, label="Retención / Sin Riesgo (0)")
    b2 = ax1.bar(x_pos + w / 2, pos_counts, width=w, color="#d62728", edgecolor="black", alpha=0.88, label="Riesgo de Deserción (1)")

    for bar, total in zip(b1, n_totals):
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, h + 45, f"{h}\n({h/total*100:.1f}%)", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    for bar, total in zip(b2, n_totals):
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, h + 45, f"{h}\n({h/total*100:.1f}%)", ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="#a50f15")

    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(splits, fontsize=9.5)
    ax1.set_ylabel("Número de Estudiantes", fontweight="bold")
    ax1.set_title("(a) Partición Estratificada del Dataset (N = 6,000)", fontweight="bold", fontsize=11)
    ax1.set_ylim(0, max(neg_counts) * 1.22)
    ax1.grid(axis="y", linestyle="--", alpha=0.4)
    ax1.legend(loc="upper right", frameon=True, fontsize=8.5)

    # Panel B: Correlación de Pearson con Riesgo de Deserción
    corrs = []
    for i, col in enumerate(feature_cols):
        c = np.corrcoef(X_raw[:, i], y_raw)[0, 1] if np.std(X_raw[:, i]) > 0 else 0.0
        corrs.append((feature_labels[col], float(c)))

    corrs_sorted = sorted(corrs, key=lambda x: abs(x[1]), reverse=True)[:12][::-1]
    labels_c = [item[0] for item in corrs_sorted]
    vals_c = [item[1] for item in corrs_sorted]
    colors_c = ["#d62728" if v > 0 else "#1f77b4" for v in vals_c]

    y_pos = np.arange(len(labels_c))
    bars_c = ax2.barh(y_pos, vals_c, color=colors_c, edgecolor="black", height=0.68, alpha=0.88)
    ax2.axvline(0, color="black", linewidth=1.0)
    for bar, v in zip(bars_c, vals_c):
        offset = 0.018 if v >= 0 else -0.018
        ha = "left" if v >= 0 else "right"
        ax2.text(v + offset, bar.get_y() + bar.get_height() / 2, f"{v:+.3f}", va="center", ha=ha, fontsize=8.5, fontweight="bold")

    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(labels_c, fontsize=9)
    ax2.set_xlim(-0.72, 0.45)
    ax2.set_xlabel("Coeficiente de Correlación de Pearson (r) con Deserción", fontweight="bold")
    ax2.set_title("(b) Principales Factores Asociados a la Deserción Estudiantil", fontweight="bold", fontsize=11)
    ax2.grid(axis="x", linestyle="--", alpha=0.4)

    plt.tight_layout()
    fig1_path = os.path.join(figures_dir, "fig1_exploracion_dataset.png")
    plt.savefig(fig1_path, dpi=300, bbox_inches="tight")
    plt.close()

    # =========================================================================
    # FIGURA 2: Curvas de Aprendizaje y Mitigación del Overfitting (4 Modelos)
    # =========================================================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.2))

    styles = {
        "M1_Baseline": ("#d62728", "--", "M1: Baseline (Sin Regularizar)"),
        "M2_L2": ("#ff7f0e", "-.", "M2: Regularización L2 (0.003)"),
        "M3_Dropout": ("#9467bd", ":", "M3: Regularización Dropout (0.35)"),
        "M4_Propuesto": ("#2ca02c", "-", "M4: Propuesta (BN + L2 + Dropout + ES)"),
    }

    for code, (color, ls, lbl) in styles.items():
        val_loss = results[code]["history"]["val_loss"]
        ep = np.arange(1, len(val_loss) + 1)
        lw = 2.4 if code == "M4_Propuesto" else 1.7
        ax1.plot(ep, val_loss, color=color, linestyle=ls, linewidth=lw, label=lbl)

    # Marcar punto óptimo de EarlyStopping en M4
    m4_best_ep = results["M4_Propuesto"]["best_val_epoch"]
    m4_min_loss = results["M4_Propuesto"]["min_val_loss"]
    ax1.scatter([m4_best_ep], [m4_min_loss], color="#2ca02c", s=80, zorder=5, edgecolor="black")
    ax1.annotate(
        f"Óptimo EarlyStopping\n(Época {m4_best_ep}, Loss={m4_min_loss:.4f})",
        xy=(m4_best_ep, m4_min_loss),
        xytext=(m4_best_ep + 6, m4_min_loss - 0.04),
        arrowprops=dict(arrowstyle="->", color="#1b611b", lw=1.2),
        fontsize=8.5,
        fontweight="bold",
        color="#1b611b",
    )

    ax1.set_title("(a) Evolución de la Pérdida de Validación (Mitigación de Overfitting)", fontweight="bold", fontsize=10.5)
    ax1.set_xlabel("Épocas de Entrenamiento", fontweight="bold")
    ax1.set_ylabel("Pérdida de Validación (Binary Crossentropy)", fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax1.legend(loc="upper left", fontsize=8.5, frameon=True)

    # Panel B: Contraste Directo Train vs Val Loss (M1 Baseline vs M4 Propuesto)
    ep_m1 = np.arange(1, len(results["M1_Baseline"]["history"]["loss"]) + 1)
    ep_m4 = np.arange(1, len(results["M4_Propuesto"]["history"]["loss"]) + 1)

    ax2.plot(ep_m1, results["M1_Baseline"]["history"]["loss"], color="#d62728", linestyle="--", alpha=0.7, linewidth=1.6, label="M1 Baseline - Train Loss")
    ax2.plot(ep_m1, results["M1_Baseline"]["history"]["val_loss"], color="#d62728", linestyle="-", linewidth=2.0, label="M1 Baseline - Validation Loss (Overfitting)")
    ax2.plot(ep_m4, results["M4_Propuesto"]["history"]["loss"], color="#2ca02c", linestyle="--", alpha=0.8, linewidth=1.6, label="M4 Propuesto - Train Loss")
    ax2.plot(ep_m4, results["M4_Propuesto"]["history"]["val_loss"], color="#2ca02c", linestyle="-", linewidth=2.3, label="M4 Propuesto - Validation Loss (Estable)")

    ax2.set_title("(b) Divergencia Train vs. Validation: Modelo Base vs. Propuesto", fontweight="bold", fontsize=10.5)
    ax2.set_xlabel("Épocas de Entrenamiento", fontweight="bold")
    ax2.set_ylabel("Binary Crossentropy Loss", fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.4)
    ax2.legend(loc="upper left", fontsize=8.5, frameon=True)

    plt.tight_layout()
    fig2_path = os.path.join(figures_dir, "fig2_curvas_aprendizaje_overfitting.png")
    plt.savefig(fig2_path, dpi=300, bbox_inches="tight")
    plt.close()

    # =========================================================================
    # FIGURA 3: Matriz de Confusión (Modelo Propuesto) y Curvas ROC-AUC Comparativas
    # =========================================================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.2, 5.2))

    m4_metrics = results["M4_Propuesto"]["test_metrics"]
    cm = np.array([
        [m4_metrics["tn"], m4_metrics["fp"]],
        [m4_metrics["fn"], m4_metrics["tp"]],
    ])
    cm_pct = cm / cm.sum(axis=1, keepdims=True) * 100

    im = ax1.imshow(cm, cmap="Blues", alpha=0.88)
    classes = ["Continúa / Sin Riesgo (0)", "Deserción / En Riesgo (1)"]
    ax1.set_xticks([0, 1])
    ax1.set_yticks([0, 1])
    ax1.set_xticklabels(classes, fontsize=9, fontweight="bold")
    ax1.set_yticklabels(classes, fontsize=9, fontweight="bold")
    ax1.set_xlabel("Predicción de la Red Neuronal", fontweight="bold")
    ax1.set_ylabel("Estado Real del Estudiante", fontweight="bold")
    ax1.set_title("(a) Matriz de Confusión — Modelo Propuesto (Test n=900)", fontweight="bold", fontsize=10.5)

    labels_cm = [["Verdaderos Negativos (TN)", "Falsos Positivos (FP)"], ["Falsos Negativos (FN)", "Verdaderos Positivos (TP)"]]
    for i in range(2):
        for j in range(2):
            text_color = "white" if cm[i, j] > cm.max() / 2 else "#111111"
            ax1.text(
                j,
                i,
                f"{labels_cm[i][j]}\n\n{cm[i, j]} alumnos\n({cm_pct[i, j]:.1f}%)",
                ha="center",
                va="center",
                color=text_color,
                fontsize=9.5,
                fontweight="bold",
            )

    # Panel B: Curva ROC Comparativa
    for code, (color, ls, lbl) in styles.items():
        m = results[code]["test_metrics"]
        fpr = m["fpr"]
        tpr = m["tpr"]
        auc_val = m["roc_auc"]
        lw = 2.5 if code == "M4_Propuesto" else 1.7
        ax2.plot(fpr, tpr, color=color, linestyle=ls, linewidth=lw, label=f"{lbl} (AUC = {auc_val:.4f})")

    ax2.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1.1, label="Clasificador Aleatorio (AUC = 0.5000)")
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.03])
    ax2.set_xlabel("Tasa de Falsos Positivos (1 - Especificidad)", fontweight="bold")
    ax2.set_ylabel("Tasa de Verdaderos Positivos (Sensibilidad / Recall)", fontweight="bold")
    ax2.set_title("(b) Curvas ROC Comparativas en el Conjunto de Prueba", fontweight="bold", fontsize=10.5)
    ax2.grid(True, linestyle="--", alpha=0.4)
    ax2.legend(loc="lower right", fontsize=8.3, frameon=True)

    plt.tight_layout()
    fig3_path = os.path.join(figures_dir, "fig3_matriz_confusion_roc.png")
    plt.savefig(fig3_path, dpi=300, bbox_inches="tight")
    plt.close()

    # =========================================================================
    # FIGURA 4: Importancia de Variables por Permutación en la Red Neuronal
    # =========================================================================
    importances = results.get("feature_importances", [0.0] * len(feature_cols))
    imp_pairs = sorted(
        [(feature_labels[col], imp * 100) for col, imp in zip(feature_cols, importances)],
        key=lambda x: x[1],
        reverse=True,
    )[:12][::-1]

    fig, ax = plt.subplots(figsize=(10.5, 5.0))
    y_idx = np.arange(len(imp_pairs))
    imp_names = [p[0] for p in imp_pairs]
    imp_vals = [p[1] for p in imp_pairs]

    bars = ax.barh(y_idx, imp_vals, color="#2b5c8f", edgecolor="black", height=0.68, alpha=0.9)
    for bar, v in zip(bars, imp_vals):
        ax.text(v + max(imp_vals) * 0.015, bar.get_y() + bar.get_height() / 2, f"+{v:.2f}% caída en Acc", va="center", fontsize=8.5, fontweight="bold")

    ax.set_yticks(y_idx)
    ax.set_yticklabels(imp_names, fontsize=9.5)
    ax.set_xlim(0, max(imp_vals) * 1.22 if max(imp_vals) > 0 else 5.0)
    ax.set_xlabel("Disminución en la Exactitud de Prueba al Permutar la Variable (%)", fontweight="bold")
    ax.set_title("Importancia Relativa de las Variables Predictoras en la Red Neuronal Propuesta (M4)", fontweight="bold", fontsize=11)
    ax.grid(axis="x", linestyle="--", alpha=0.4)

    plt.tight_layout()
    fig4_path = os.path.join(figures_dir, "fig4_importancia_variables.png")
    plt.savefig(fig4_path, dpi=300, bbox_inches="tight")
    plt.close()

    print("4 figuras científicas generadas exitosamente a 300 DPI en:", figures_dir)


if __name__ == "__main__":
    generate_all_figures()
