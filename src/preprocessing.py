"""
Módulo de Preprocesamiento e Ingeniería de Características
Proyecto: Predicción Temprana de la Deserción Estudiantil Universitaria mediante Redes Neuronales Profundas Regularizadas
Autor: William David Ortiz Mamani (UPeU - Ingeniería de Sistemas)
"""

import os
import csv
import json
import numpy as np

# las 22 variables predictoras multidimensionales con varianza real y sin fuga de datos (data leakage)
FEATURE_COLUMNS = [
    # 1. Demografía y Nivel Socioeconómico
    "age",
    "gender_encoded",
    "international_student",
    "first_generation",
    "socioeconomic_deprivation_index",
    # 2. Estado Financiero Universitario
    "scholarship_holder",
    "tuition_up_to_date",
    # 3. Historial Académico Previo
    "admission_grade",
    "prior_qualification_grade",
    # 4. Rendimiento Académico Actual
    "gpa_score_avg",
    "gpa_score_trend",
    "pass_rate",
    "first_assessment_score",
    "study_hours_per_week",
    "study_load",
    # 5. Compromiso Conductual en Plataforma Virtual (LMS / VLE)
    "attendance_rate",
    "lms_engagement_score",
    "lms_active_days_ratio",
    "longest_inactivity_gap_days",
    "missed_submissions_count",
    # 6. Factores Psicológicos y Fisiológicos (Estrés y Sueño)
    "stress_level_scale",
    "sleep_hours_per_day",
]

FEATURE_LABELS_ES = {
    "age": "Edad del Estudiante",
    "gender_encoded": "Género (Codificado)",
    "international_student": "Estudiante Internacional",
    "first_generation": "Primera Generación Universitaria",
    "socioeconomic_deprivation_index": "Índice de Privación Socioeconómica",
    "scholarship_holder": "Beneficiario de Beca",
    "tuition_up_to_date": "Pensiones al Día",
    "admission_grade": "Nota de Admisión",
    "prior_qualification_grade": "Calificación Previa",
    "gpa_score_avg": "Promedio Ponderado (GPA)",
    "gpa_score_trend": "Tendencia del Promedio",
    "pass_rate": "Tasa de Cursos Aprobados",
    "first_assessment_score": "Nota de Primera Evaluación",
    "study_hours_per_week": "Horas de Estudio / Semana",
    "study_load": "Carga Académica",
    "attendance_rate": "Tasa de Asistencia (%)",
    "lms_engagement_score": "Interacción en Plataforma Virtual (LMS)",
    "lms_active_days_ratio": "Ratio de Días Activos en LMS",
    "longest_inactivity_gap_days": "Días Máximos de Inactividad LMS",
    "missed_submissions_count": "Tareas No Entregadas",
    "stress_level_scale": "Nivel de Estrés Académico (1-10)",
    "sleep_hours_per_day": "Horas de Sueño por Día",
}


def load_and_preprocess_data(
    raw_csv_path: str,
    processed_dir: str,
    sample_size: int = 6000,
    seed: int = 42,
):
    """
    Extrae la cohorte multimodal 'dropout_risk_2026' desde final_student_dataset.csv,
    realiza una partición estratificada (70% Entrenamiento, 15% Validación, 15% Prueba)
    y aplica normalización Z-score ajustada únicamente sobre el conjunto de entrenamiento.
    """
    np.random.seed(seed)
    X_list = []
    y_list = []

    with open(raw_csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["source_dataset"] == "dropout_risk_2026":
                features = [float(row[col]) for col in FEATURE_COLUMNS]
                target = int(row["at_risk"])
                X_list.append(features)
                y_list.append(target)
                if len(X_list) >= sample_size:
                    break

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)

    # Partición estratificada: 70% Train, 15% Val, 15% Test
    idx_pos = np.where(y == 1)[0]
    idx_neg = np.where(y == 0)[0]
    np.random.shuffle(idx_pos)
    np.random.shuffle(idx_neg)

    def split_indices(indices):
        n = len(indices)
        n_train = int(0.70 * n)
        n_val = int(0.15 * n)
        return (
            indices[:n_train],
            indices[n_train : n_train + n_val],
            indices[n_train + n_val :],
        )

    pos_tr, pos_va, pos_te = split_indices(idx_pos)
    neg_tr, neg_va, neg_te = split_indices(idx_neg)

    train_idx = np.concatenate([pos_tr, neg_tr])
    val_idx = np.concatenate([pos_va, neg_va])
    test_idx = np.concatenate([pos_te, neg_te])

    np.random.shuffle(train_idx)
    np.random.shuffle(val_idx)
    np.random.shuffle(test_idx)

    X_train_raw, y_train = X[train_idx], y[train_idx]
    X_val_raw, y_val = X[val_idx], y[val_idx]
    X_test_raw, y_test = X[test_idx], y[test_idx]

    # Estandarización Z-Score usando exclusivamente estadísticos de entrenamiento
    mean = X_train_raw.mean(axis=0)
    std = X_train_raw.std(axis=0)
    std[std == 0] = 1.0

    X_train = (X_train_raw - mean) / std
    X_val = (X_val_raw - mean) / std
    X_test = (X_test_raw - mean) / std

    os.makedirs(processed_dir, exist_ok=True)
    np.save(os.path.join(processed_dir, "X_train.npy"), X_train)
    np.save(os.path.join(processed_dir, "y_train.npy"), y_train)
    np.save(os.path.join(processed_dir, "X_val.npy"), X_val)
    np.save(os.path.join(processed_dir, "y_val.npy"), y_val)
    np.save(os.path.join(processed_dir, "X_test.npy"), X_test)
    np.save(os.path.join(processed_dir, "y_test.npy"), y_test)
    np.save(os.path.join(processed_dir, "X_raw_full.npy"), X)
    np.save(os.path.join(processed_dir, "y_raw_full.npy"), y)

    metadata = {
        "total_samples": int(len(y)),
        "n_features": len(FEATURE_COLUMNS),
        "feature_columns": FEATURE_COLUMNS,
        "feature_labels_es": FEATURE_LABELS_ES,
        "train_samples": int(len(y_train)),
        "val_samples": int(len(y_val)),
        "test_samples": int(len(y_test)),
        "positive_rate_total": float(y.mean()),
        "positive_rate_train": float(y_train.mean()),
        "positive_rate_val": float(y_val.mean()),
        "positive_rate_test": float(y_test.mean()),
        "scaler_mean": mean.tolist(),
        "scaler_std": std.tolist(),
    }

    with open(os.path.join(processed_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False)

    return (X_train, y_train), (X_val, y_val), (X_test, y_test), metadata


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_path = os.path.join(base_dir, "data", "raw", "final_student_dataset.csv")
    proc_dir = os.path.join(base_dir, "data", "processed")
    _, _, _, meta = load_and_preprocess_data(raw_path, proc_dir)
    print(f"Preprocesamiento completado: {meta['total_samples']} registros, {meta['n_features']} variables.")
