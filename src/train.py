"""
Módulo de Entrenamiento Experimental y Control de Sobreajuste
Proyecto: Predicción Temprana de la Deserción Estudiantil Universitaria mediante Redes Neuronales Profundas Regularizadas
Autor: William David Ortiz Mamani (UPeU - Ingeniería de Sistemas)
"""

import os
import json
import csv
import numpy as np
import tensorflow as tf
from keras import callbacks

from preprocessing import load_and_preprocess_data
from model_architectures import (
    build_baseline_model,
    build_l2_model,
    build_dropout_model,
    build_proposed_model,
)


def compute_binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5):
    """Calcula Exactitud, Precisión, Sensibilidad (Recall), Especificidad, F1-Score y ROC-AUC en NumPy puro."""
    y_true = y_true.astype(int).flatten()
    y_prob = y_prob.flatten()
    y_pred = (y_prob >= threshold).astype(int)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    accuracy = (tp + tn) / max(len(y_true), 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    # Cálculo exacto de ROC-AUC por integración trapezoidal
    desc_idx = np.argsort(-y_prob)
    y_true_sorted = y_true[desc_idx]
    y_prob_sorted = y_prob[desc_idx]

    distinct_indices = np.where(np.diff(y_prob_sorted))[0]
    threshold_idxs = np.r_[distinct_indices, y_true_sorted.size - 1]

    tps = np.cumsum(y_true_sorted)[threshold_idxs]
    fps = 1 + threshold_idxs - tps

    tps = np.r_[0, tps]
    fps = np.r_[0, fps]

    tpr = tps / max(tps[-1], 1)
    fpr = fps / max(fps[-1], 1)
    roc_auc = float(np.trapezoid(tpr, fpr))

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "f1_score": float(f1),
        "roc_auc": float(roc_auc),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
    }


def run_training_pipeline():
    tf.random.set_seed(42)
    np.random.seed(42)

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_csv = os.path.join(base_dir, "data", "raw", "final_student_dataset.csv")
    proc_dir = os.path.join(base_dir, "data", "processed")
    models_dir = os.path.join(base_dir, "models")
    tables_dir = os.path.join(base_dir, "results", "tables")

    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)

    (X_train, y_train), (X_val, y_val), (X_test, y_test), metadata = load_and_preprocess_data(
        raw_csv, proc_dir, sample_size=6000, seed=42
    )

    input_dim = X_train.shape[1]
    epochs = 60
    batch_size = 64

    model_specs = [
        ("Modelo 1: Baseline (Sin Regularizar)", "M1_Baseline", build_baseline_model(input_dim), False),
        ("Modelo 2: Regularización L2 (Weight Decay)", "M2_L2", build_l2_model(input_dim), False),
        ("Modelo 3: Regularización Dropout", "M3_Dropout", build_dropout_model(input_dim), False),
        ("Modelo 4: Propuesta Optimizada (BN + L2 + Dropout + EarlyStopping)", "M4_Propuesto", build_proposed_model(input_dim), True),
    ]

    all_results = {}
    csv_rows = []

    for label, code, model, use_es in model_specs:
        print(f"\n==========================================")
        print(f"Entrenando {label}...")
        print(f"==========================================")

        cb_list = []
        if use_es:
            cb_list = [
                callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=10,
                    restore_best_weights=True,
                    verbose=1,
                ),
                callbacks.ReduceLROnPlateau(
                    monitor="val_loss",
                    factor=0.5,
                    patience=5,
                    min_lr=1e-5,
                    verbose=1,
                ),
            ]

        history = model.fit(
            X_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            callbacks=cb_list,
            verbose=0,
        )

        train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
        val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
        test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)

        y_prob_test = model.predict(X_test, verbose=0).flatten()
        metrics = compute_binary_metrics(y_test, y_prob_test)

        epochs_ran = len(history.history["loss"])
        best_val_epoch = int(np.argmin(history.history["val_loss"]) + 1)
        min_val_loss = float(np.min(history.history["val_loss"]))
        final_train_loss = float(history.history["loss"][-1])
        final_val_loss = float(history.history["val_loss"][-1])
        overfit_gap = float(final_val_loss - final_train_loss)

        all_results[code] = {
            "label": label,
            "epochs_ran": epochs_ran,
            "best_val_epoch": best_val_epoch,
            "train_loss": float(train_loss),
            "train_acc": float(train_acc),
            "val_loss": float(val_loss),
            "val_acc": float(val_acc),
            "test_loss": float(test_loss),
            "test_acc": float(test_acc),
            "min_val_loss": min_val_loss,
            "overfit_gap": overfit_gap,
            "history": {
                "loss": [float(x) for x in history.history["loss"]],
                "val_loss": [float(x) for x in history.history["val_loss"]],
                "accuracy": [float(x) for x in history.history["accuracy"]],
                "val_accuracy": [float(x) for x in history.history["val_accuracy"]],
            },
            "test_metrics": metrics,
            "y_prob_test": y_prob_test.tolist(),
        }

        csv_rows.append({
            "Modelo": label,
            "Epocas_Ejecutadas": epochs_ran,
            "Mejor_Epoca_Val": best_val_epoch,
            "Train_Loss": round(float(train_loss), 4),
            "Val_Loss": round(float(val_loss), 4),
            "Test_Loss": round(float(test_loss), 4),
            "Gap_Overfitting": round(overfit_gap, 4),
            "Test_Accuracy_%": round(metrics["accuracy"] * 100, 2),
            "Precision_%": round(metrics["precision"] * 100, 2),
            "Recall_Sensibilidad_%": round(metrics["recall"] * 100, 2),
            "Especificidad_%": round(metrics["specificity"] * 100, 2),
            "F1_Score_%": round(metrics["f1_score"] * 100, 2),
            "ROC_AUC": round(metrics["roc_auc"], 4),
        })

        print(
            f"-> {code}: Test Loss={test_loss:.4f} | Test Acc={metrics['accuracy']*100:.2f}% | "
            f"F1={metrics['f1_score']*100:.2f}% | AUC={metrics['roc_auc']:.4f} | Gap={overfit_gap:+.4f}"
        )

        if code == "M4_Propuesto":
            model_path = os.path.join(models_dir, "mejor_modelo_desercion.keras")
            model.save(model_path)
            print(f"Modelo propuesto guardado en: {model_path}")

            # Calcular importancia de variables por permutación sobre el conjunto de prueba
            base_acc = metrics["accuracy"]
            importances = []
            np.random.seed(42)
            for j in range(input_dim):
                X_temp = X_test.copy()
                np.random.shuffle(X_temp[:, j])
                prob_perm = model.predict(X_temp, verbose=0).flatten()
                acc_perm = np.mean(((prob_perm >= 0.5).astype(int)) == y_test.astype(int))
                importances.append(float(max(base_acc - acc_perm, 0.0)))
            all_results["feature_importances"] = importances

    # Guardar resultados en JSON y CSV
    with open(os.path.join(tables_dir, "metricas_detalladas.json"), "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=4, ensure_ascii=False)

    csv_path = os.path.join(tables_dir, "comparativa_modelos.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"\nTablas comparativas guardadas en: {csv_path}")
    return all_results


if __name__ == "__main__":
    run_training_pipeline()
