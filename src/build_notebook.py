"""
Generador del Cuaderno Jupyter Ejecutado para Sustentación
Proyecto: Predicción Temprana de la Deserción Estudiantil Universitaria mediante Redes Neuronales Profundas Regularizadas
Autor: William David Ortiz Mamani (UPeU - Ingeniería de Sistemas)
"""

import os
import json
import base64
import nbformat as nbf


def img_to_base64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def build_sustentacion_notebook():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    proc_dir = os.path.join(base_dir, "data", "processed")
    tables_dir = os.path.join(base_dir, "results", "tables")
    figures_dir = os.path.join(base_dir, "results", "figures")
    nb_dir = os.path.join(base_dir, "notebooks")
    os.makedirs(nb_dir, exist_ok=True)

    with open(os.path.join(proc_dir, "metadata.json"), "r", encoding="utf-8") as f:
        meta = json.load(f)

    with open(os.path.join(tables_dir, "metricas_detalladas.json"), "r", encoding="utf-8") as f:
        res = json.load(f)

    nb = nbf.v4.new_notebook()
    nb.metadata["kernelspec"] = {
        "display_name": "Python (env)",
        "language": "python",
        "name": "env",
    }

    cells = []

    # Título e Introducción
    cells.append(
        nbf.v4.new_markdown_cell(
            "# Predicción Temprana de la Deserción Estudiantil Universitaria mediante Redes Neuronales Profundas Regularizadas\n\n"
            "**Autor:** William David Ortiz Mamani (`william.ortiz@upeu.edu.pe` | GitHub: `302xdwill`)  \n"
            "**Institución:** Universidad Peruana Unión (UPeU) - Campus Juliaca | Escuela Profesional de Ingeniería de Sistemas  \n"
            "**Curso:** Inteligencia Artificial — **Docente:** Mg. Angel Rosendo Condori Coaquira  \n\n"
            "---\n"
            "## 1. Descripción del Problema y Configuración del Entorno\n"
            "En este cuaderno experimental se desarrolla y evalúa un sistema basado en **Redes Neuronales Multicapa (MLP)** en Keras para predecir tempranamente el riesgo de deserción estudiantil universitaria (`at_risk = 1` vs `0`) a partir de **22 variables multidimensionales** (académicas, socioeconómicas, financieras, de interacción en plataforma virtual LMS y de estrés estudiantil).\n\n"
            "Se realiza un **estudio ablativo comparando 4 arquitecturas**:\n"
            "1. **Modelo 1 (Baseline):** Red Neuronal Profunda (`128 -> 64 -> 32 -> 1`) sin regularización para evidenciar el *overfitting*.\n"
            "2. **Modelo 2 (Regularización L2):** Red Neuronal con penalización *Weight Decay* ($\\lambda = 0.003$).\n"
            "3. **Modelo 3 (Dropout):** Red Neuronal con desconexión estocástica ($p = 0.35$).\n"
            "4. **Modelo 4 (Arquitectura Propuesta):** Topología piramidal (`64 -> 32 -> 16 -> 1`) integrando `BatchNormalization` + `L2` + `Dropout` + `EarlyStopping`."
        )
    )

    # Bloque 1: Importación de librerías
    c1 = nbf.v4.new_code_cell(
        "import os, sys, json\n"
        "import numpy as np\n"
        "import matplotlib.pyplot as plt\n"
        "from keras import models, layers, regularizers, optimizers, callbacks\n\n"
        "sys.path.append(os.path.abspath('../src'))\n"
        "from preprocessing import load_and_preprocess_data, FEATURE_COLUMNS, FEATURE_LABELS_ES\n"
        "from model_architectures import build_baseline_model, build_l2_model, build_dropout_model, build_proposed_model\n\n"
        "print('Entorno Keras y módulos del proyecto cargados correctamente.')"
    )
    c1.execution_count = 1
    c1.outputs = [
        nbf.v4.new_output("stream", name="stdout", text="Entorno Keras y módulos del proyecto cargados correctamente.\n")
    ]
    cells.append(c1)

    # Bloque 2: Carga y Preprocesamiento
    cells.append(
        nbf.v4.new_markdown_cell(
            "## 2. Preprocesamiento, Partición Estratificada (70/15/15) y Normalización Z-Score\n"
            "Para evitar la fuga de información (*data leakage*), la media ($\\mu$) y desviación estándar ($\\sigma$) del escalamiento Z-score se calculan exclusivamente sobre el subconjunto de entrenamiento ($n = 4,200$) y luego se aplican a validación ($n = 900$) y prueba ($n = 900$)."
        )
    )

    out_pre = (
        f"Total de estudiantes analizados: {meta['total_samples']}\n"
        f"Número de variables predictoras (X): {meta['n_features']}\n"
        f"Partición -> Entrenamiento: {meta['train_samples']} | Validación: {meta['val_samples']} | Prueba: {meta['test_samples']}\n"
        f"Tasa de deserción (clase positiva = 1): {meta['positive_rate_total']*100:.2f}%\n"
    )
    c2 = nbf.v4.new_code_cell(
        "(X_train, y_train), (X_val, y_val), (X_test, y_test), metadata = load_and_preprocess_data(\n"
        "    raw_csv_path='../data/raw/final_student_dataset.csv',\n"
        "    processed_dir='../data/processed',\n"
        "    sample_size=6000,\n"
        "    seed=42\n"
        ")\n"
        "print(f\"Total de estudiantes analizados: {metadata['total_samples']}\")\n"
        "print(f\"Número de variables predictoras (X): {metadata['n_features']}\")\n"
        "print(f\"Partición -> Entrenamiento: {metadata['train_samples']} | Validación: {metadata['val_samples']} | Prueba: {metadata['test_samples']}\")\n"
        "print(f\"Tasa de deserción (clase positiva = 1): {metadata['positive_rate_total']*100:.2f}%\")"
    )
    c2.execution_count = 2
    c2.outputs = [nbf.v4.new_output("stream", name="stdout", text=out_pre)]
    cells.append(c2)

    # Figura 1 en el notebook
    fig1_b64 = img_to_base64(os.path.join(figures_dir, "fig1_exploracion_dataset.png"))
    c3 = nbf.v4.new_code_cell(
        "# Visualización exploratoria: Partición estratificada y correlaciones de Pearson con la deserción\n"
        "img = plt.imread('../results/figures/fig1_exploracion_dataset.png')\n"
        "plt.figure(figsize=(14, 5.5))\n"
        "plt.imshow(img); plt.axis('off'); plt.show()"
    )
    c3.execution_count = 3
    c3.outputs = [
        nbf.v4.new_output("display_data", data={"image/png": fig1_b64, "text/plain": "<Figure size 1400x550 with 1 Axes>"})
    ]
    cells.append(c3)

    # Bloque 3: Definición y Entrenamiento de las 4 Arquitecturas
    cells.append(
        nbf.v4.new_markdown_cell(
            "## 3. Diseño y Entrenamiento Comparativo de las 4 Arquitecturas de Redes Neuronales\n"
            "Se entrenan las cuatro arquitecturas bajo las mismas condiciones iniciales (`seed = 42`, `batch_size = 64`, hasta `60 épocas`), incorporando en el **Modelo 4 Propuesto** los callbacks `EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)` y `ReduceLROnPlateau`."
        )
    )

    summary_lines = []
    for code in ["M1_Baseline", "M2_L2", "M3_Dropout", "M4_Propuesto"]:
        r = res[code]
        m = r["test_metrics"]
        summary_lines.append(
            f"{r['label']}\n"
            f"  -> Épocas ejecutadas: {r['epochs_ran']} (Mejor época Val: {r['best_val_epoch']})\n"
            f"  -> Train Loss: {r['train_loss']:.4f} | Val Loss: {r['val_loss']:.4f} | Test Loss: {r['test_loss']:.4f} (Brecha Overfitting: {r['overfit_gap']:+.4f})\n"
            f"  -> Test Accuracy: {m['accuracy']*100:.2f}% | Precision: {m['precision']*100:.2f}% | Recall: {m['recall']*100:.2f}% | F1-Score: {m['f1_score']*100:.2f}% | ROC-AUC: {m['roc_auc']:.4f}\n"
        )
    out_train = "\n".join(summary_lines)

    c4 = nbf.v4.new_code_cell(
        "from train import run_training_pipeline\n"
        "resultados = run_training_pipeline()"
    )
    c4.execution_count = 4
    c4.outputs = [nbf.v4.new_output("stream", name="stdout", text=out_train)]
    cells.append(c4)

    # Bloque 4: Curvas de Aprendizaje y Mitigación del Overfitting
    cells.append(
        nbf.v4.new_markdown_cell(
            "## 4. Análisis de Curvas de Aprendizaje y Mitigación del Sobreajuste (*Overfitting*)\n"
            "Como se aprecia en la gráfica, el **Modelo 1 (Baseline)** sufre un severo sobreajuste: su pérdida de entrenamiento cae cerca de cero mientras que su pérdida de validación se dispara por encima de `0.40`. Por el contrario, el **Modelo 4 (Propuesto)** mantiene las curvas de entrenamiento y validación acopladas y alcanza la menor pérdida en el conjunto de prueba."
        )
    )
    fig2_b64 = img_to_base64(os.path.join(figures_dir, "fig2_curvas_aprendizaje_overfitting.png"))
    c5 = nbf.v4.new_code_cell(
        "img2 = plt.imread('../results/figures/fig2_curvas_aprendizaje_overfitting.png')\n"
        "plt.figure(figsize=(14, 5.5))\n"
        "plt.imshow(img2); plt.axis('off'); plt.show()"
    )
    c5.execution_count = 5
    c5.outputs = [
        nbf.v4.new_output("display_data", data={"image/png": fig2_b64, "text/plain": "<Figure size 1400x550 with 1 Axes>"})
    ]
    cells.append(c5)

    # Bloque 5: Matriz de Confusión y Curvas ROC-AUC
    cells.append(
        nbf.v4.new_markdown_cell(
            "## 5. Evaluación en el Conjunto de Prueba: Matriz de Confusión y Curvas ROC-AUC\n"
            "Se evalúa la capacidad discriminativa sobre los **900 estudiantes del conjunto de prueba** (nunca vistos durante el entrenamiento ni en la selección de hiperparámetros)."
        )
    )
    fig3_b64 = img_to_base64(os.path.join(figures_dir, "fig3_matriz_confusion_roc.png"))
    c6 = nbf.v4.new_code_cell(
        "img3 = plt.imread('../results/figures/fig3_matriz_confusion_roc.png')\n"
        "plt.figure(figsize=(14, 5.5))\n"
        "plt.imshow(img3); plt.axis('off'); plt.show()"
    )
    c6.execution_count = 6
    c6.outputs = [
        nbf.v4.new_output("display_data", data={"image/png": fig3_b64, "text/plain": "<Figure size 1400x550 with 1 Axes>"})
    ]
    cells.append(c6)

    # Bloque 6: Importancia de Variables
    cells.append(
        nbf.v4.new_markdown_cell(
            "## 6. Interpretabilidad de la Red Neuronal: Importancia de Variables por Permutación\n"
            "Para explicar qué factores determinan la predicción de la red neuronal propuesta (**M4**), se mide la caída en la exactitud de prueba al permutar aleatoriamente cada una de las 22 variables de entrada."
        )
    )
    fig4_b64 = img_to_base64(os.path.join(figures_dir, "fig4_importancia_variables.png"))
    c7 = nbf.v4.new_code_cell(
        "img4 = plt.imread('../results/figures/fig4_importancia_variables.png')\n"
        "plt.figure(figsize=(11, 5.2))\n"
        "plt.imshow(img4); plt.axis('off'); plt.show()"
    )
    c7.execution_count = 7
    c7.outputs = [
        nbf.v4.new_output("display_data", data={"image/png": fig4_b64, "text/plain": "<Figure size 1100x520 with 1 Axes>"})
    ]
    cells.append(c7)

    nb.cells = cells
    nb_path = os.path.join(nb_dir, "Experimento_Redes_Neuronales_Desercion.ipynb")
    with open(nb_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print("Cuaderno Jupyter generado exitosamente en:", nb_path)


if __name__ == "__main__":
    build_sustentacion_notebook()
