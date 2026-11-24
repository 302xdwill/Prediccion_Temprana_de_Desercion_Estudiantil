"""
Sistema Web Local con FastAPI + Keras Deep Neural Network
Proyecto: Predicción Temprana de la Deserción Estudiantil Universitaria mediante Redes Neuronales Profundas Regularizadas
Autor: William David Ortiz Mamani (UPeU - Campus Juliaca | Ingeniería de Sistemas)
Docente: Mg. Angel Rosendo Condori Coaquira
"""

import os
import json
import numpy as np
import keras
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "mejor_modelo_desercion.keras")
META_PATH = os.path.join(BASE_DIR, "data", "processed", "metadata.json")
METRICS_PATH = os.path.join(BASE_DIR, "results", "tables", "metricas_detalladas.json")
FIGURES_DIR = os.path.join(BASE_DIR, "results", "figures")
PDF_PATH = os.path.join(BASE_DIR, "manuscript", "entregas_pdf", "Articulo_Redes_Neuronales_William_Ortiz.pdf")
PRESENTATION_PATH = os.path.join(BASE_DIR, "presentacion", "Presentacion_Exposicion_William_Ortiz.html")
TEMPLATE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates", "index.html")

# Cargar metadatos del escalador Z-score y métricas experimentales
with open(META_PATH, "r", encoding="utf-8") as f:
    METADATA = json.load(f)

with open(METRICS_PATH, "r", encoding="utf-8") as f:
    METRICS_DATA = json.load(f)

FEATURE_COLUMNS = METADATA["feature_columns"]
SCALER_MEAN = np.array(METADATA["scaler_mean"], dtype=np.float32)
SCALER_STD = np.array(METADATA["scaler_std"], dtype=np.float32)

# Cargar la Red Neuronal Propuesta entrenada (M4: 64 -> 32 -> 16 -> 1 con BN + L2 + Dropout + EarlyStopping)
print(f"Cargando Red Neuronal Profunda desde: {MODEL_PATH}")
MODEL = keras.models.load_model(MODEL_PATH)
print("Red Neuronal cargada exitosamente y lista para inferencia en tiempo real.")

app = FastAPI(
    title="SIA-UPeU: Sistema Inteligente de Predicción Temprana de Deserción Estudiantil",
    description="Aplicación Web con FastAPI y Redes Neuronales Profundas Regularizadas (Keras/TensorFlow)",
    version="1.0.0",
)

app.mount("/figures", StaticFiles(directory=FIGURES_DIR), name="figures")


class StudentInput(BaseModel):
    student_name: str = Field(default="Estudiante Evaluado", description="Nombre o código del estudiante")
    # 1. Demografía y Nivel Socioeconómico
    age: float = Field(default=20, ge=15, le=65)
    gender_encoded: int = Field(default=1, ge=0, le=2)
    international_student: int = Field(default=0, ge=0, le=1)
    first_generation: int = Field(default=0, ge=0, le=1)
    socioeconomic_deprivation_index: float = Field(default=4.5, ge=1.0, le=10.0)
    # 2. Estado Financiero
    scholarship_holder: int = Field(default=0, ge=0, le=1)
    tuition_up_to_date: int = Field(default=1, ge=0, le=1)
    # 3. Historial Académico Previo
    admission_grade: float = Field(default=75.0, ge=0.0, le=100.0)
    prior_qualification_grade: float = Field(default=76.0, ge=0.0, le=100.0)
    # 4. Rendimiento Académico Universitario
    gpa_score_avg: float = Field(default=72.0, ge=0.0, le=100.0)
    gpa_score_trend: float = Field(default=1.5, ge=-60.0, le=60.0)
    pass_rate: float = Field(default=0.85, ge=0.0, le=1.0)
    first_assessment_score: float = Field(default=74.0, ge=0.0, le=100.0)
    study_hours_per_week: float = Field(default=20.0, ge=1.0, le=50.0)
    study_load: float = Field(default=18.0, ge=3.0, le=30.0)
    # 5. Compromiso Conductual en Plataforma Virtual (LMS / PatmOS)
    attendance_rate: float = Field(default=88.0, ge=0.0, le=100.0)
    lms_engagement_score: float = Field(default=68.0, ge=0.0, le=100.0)
    lms_active_days_ratio: float = Field(default=0.72, ge=0.0, le=1.0)
    longest_inactivity_gap_days: int = Field(default=4, ge=0, le=30)
    missed_submissions_count: int = Field(default=1, ge=0, le=20)
    # 6. Factores Psicológicos y Fisiológicos
    stress_level_scale: float = Field(default=4.5, ge=1.0, le=10.0)
    sleep_hours_per_day: float = Field(default=7.0, ge=3.0, le=12.0)


@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    with open(TEMPLATE_PATH, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.get("/pdf")
async def download_pdf():
    return FileResponse(
        PDF_PATH,
        media_type="application/pdf",
        filename="Articulo_Redes_Neuronales_William_Ortiz.pdf",
    )


@app.get("/presentacion", response_class=HTMLResponse)
async def serve_presentation():
    with open(PRESENTATION_PATH, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.get("/api/presets")
async def get_student_presets():
    """Casos reales representativos del conjunto de prueba para demostración en 1 clic durante la sustentación."""
    return {
        "presets": [
            {
                "id": "caso_estable",
                "title": "Caso 1: Estudiante Destacado / Retención Segura",
                "badge": "Bajo Riesgo",
                "description": "Alumno con buen promedio ponderado (84.5), pensiones al día, alta participación en aula virtual PatmOS y estrés controlado.",
                "data": {
                    "student_name": "Ana María Quispe (Ciclo III - Ing. Sistemas)",
                    "age": 19,
                    "gender_encoded": 0,
                    "international_student": 0,
                    "first_generation": 0,
                    "socioeconomic_deprivation_index": 3.2,
                    "scholarship_holder": 1,
                    "tuition_up_to_date": 1,
                    "admission_grade": 85.0,
                    "prior_qualification_grade": 84.0,
                    "gpa_score_avg": 84.5,
                    "gpa_score_trend": 4.2,
                    "pass_rate": 0.95,
                    "first_assessment_score": 86.0,
                    "study_hours_per_week": 24.0,
                    "study_load": 20.0,
                    "attendance_rate": 94.0,
                    "lms_engagement_score": 78.0,
                    "lms_active_days_ratio": 0.85,
                    "longest_inactivity_gap_days": 4,
                    "missed_submissions_count": 0,
                    "stress_level_scale": 3.5,
                    "sleep_hours_per_day": 7.5,
                },
            },
            {
                "id": "caso_critico",
                "title": "Caso 2: Alerta Crítica de Deserción (Académico + LMS)",
                "badge": "Riesgo Crítico",
                "description": "Alumno con caída severa en primera evaluación (42.0), 8 tareas sin entregar en plataforma virtual, 12 días de inactividad y deuda de pensión.",
                "data": {
                    "student_name": "Carlos Alberto Mamani (Ciclo II - Ing. Sistemas)",
                    "age": 22,
                    "gender_encoded": 1,
                    "international_student": 0,
                    "first_generation": 1,
                    "socioeconomic_deprivation_index": 7.8,
                    "scholarship_holder": 0,
                    "tuition_up_to_date": 0,
                    "admission_grade": 58.0,
                    "prior_qualification_grade": 60.0,
                    "gpa_score_avg": 44.0,
                    "gpa_score_trend": -18.5,
                    "pass_rate": 0.45,
                    "first_assessment_score": 42.0,
                    "study_hours_per_week": 15.0,
                    "study_load": 22.0,
                    "attendance_rate": 52.0,
                    "lms_engagement_score": 24.0,
                    "lms_active_days_ratio": 0.25,
                    "longest_inactivity_gap_days": 12,
                    "missed_submissions_count": 8,
                    "stress_level_scale": 8.8,
                    "sleep_hours_per_day": 4.8,
                },
            },
            {
                "id": "caso_moderado",
                "title": "Caso 3: Riesgo Moderado / Alerta Preventiva (Estrés + Finanzas)",
                "badge": "Riesgo Moderado",
                "description": "Estudiante de primera generación con notas regulares (65.5), pero con atraso en pensiones, 3 tareas no entregadas y alto estrés académico.",
                "data": {
                    "student_name": "Luis Fernando Condori (Ciclo IV - Ing. Sistemas)",
                    "age": 21,
                    "gender_encoded": 1,
                    "international_student": 0,
                    "first_generation": 1,
                    "socioeconomic_deprivation_index": 6.2,
                    "scholarship_holder": 0,
                    "tuition_up_to_date": 0,
                    "admission_grade": 70.0,
                    "prior_qualification_grade": 72.0,
                    "gpa_score_avg": 65.5,
                    "gpa_score_trend": -4.0,
                    "pass_rate": 0.74,
                    "first_assessment_score": 63.0,
                    "study_hours_per_week": 19.0,
                    "study_load": 20.0,
                    "attendance_rate": 76.0,
                    "lms_engagement_score": 52.0,
                    "lms_active_days_ratio": 0.55,
                    "longest_inactivity_gap_days": 7,
                    "missed_submissions_count": 3,
                    "stress_level_scale": 7.5,
                    "sleep_hours_per_day": 5.5,
                },
            },
            {
                "id": "caso_abandono_virtual",
                "title": "Caso 4: Desvinculación Temprana en Plataforma Virtual",
                "badge": "Riesgo Alto",
                "description": "Ingresó con buena nota de admisión (76.0), pero reprobó el primer parcial (48.0) y dejó de conectarse 14 días seguidos al LMS.",
                "data": {
                    "student_name": "Valeria Huanca Flores (Ciclo I - Ing. Sistemas)",
                    "age": 18,
                    "gender_encoded": 0,
                    "international_student": 0,
                    "first_generation": 1,
                    "socioeconomic_deprivation_index": 6.8,
                    "scholarship_holder": 0,
                    "tuition_up_to_date": 1,
                    "admission_grade": 76.0,
                    "prior_qualification_grade": 75.0,
                    "gpa_score_avg": 49.5,
                    "gpa_score_trend": -15.0,
                    "pass_rate": 0.52,
                    "first_assessment_score": 48.0,
                    "study_hours_per_week": 16.0,
                    "study_load": 21.0,
                    "attendance_rate": 58.0,
                    "lms_engagement_score": 28.0,
                    "lms_active_days_ratio": 0.30,
                    "longest_inactivity_gap_days": 14,
                    "missed_submissions_count": 6,
                    "stress_level_scale": 8.2,
                    "sleep_hours_per_day": 5.0,
                },
            },
        ]
    }


@app.get("/api/metrics")
async def get_experimental_metrics():
    """Devuelve las métricas reales del estudio ablativo de las 4 redes neuronales para el Dashboard."""
    models_summary = []
    for code in ["M1_Baseline", "M2_L2", "M3_Dropout", "M4_Propuesto"]:
        r = METRICS_DATA[code]
        tm = r["test_metrics"]
        models_summary.append({
            "code": code,
            "label": r["label"],
            "epochs_ran": r["epochs_ran"],
            "best_val_epoch": r["best_val_epoch"],
            "train_loss": round(r["train_loss"], 4),
            "val_loss": round(r["val_loss"], 4),
            "test_loss": round(r["test_loss"], 4),
            "overfit_gap": round(r["overfit_gap"], 4),
            "accuracy": round(tm["accuracy"] * 100, 2),
            "precision": round(tm["precision"] * 100, 2),
            "recall": round(tm["recall"] * 100, 2),
            "specificity": round(tm["specificity"] * 100, 2),
            "f1_score": round(tm["f1_score"] * 100, 2),
            "roc_auc": round(tm["roc_auc"], 4),
            "tp": tm["tp"],
            "tn": tm["tn"],
            "fp": tm["fp"],
            "fn": tm["fn"],
        })
    return {
        "dataset_info": {
            "total_samples": METADATA["total_samples"],
            "n_features": METADATA["n_features"],
            "train_samples": METADATA["train_samples"],
            "val_samples": METADATA["val_samples"],
            "test_samples": METADATA["test_samples"],
            "positive_rate": round(METADATA["positive_rate_total"] * 100, 2),
        },
        "models": models_summary,
    }


@app.post("/api/predict")
async def predict_dropout_risk(payload: StudentInput):
    """
    Ejecuta la inferencia en vivo usando la Red Neuronal Propuesta (M4) en Keras:
    1. Construye el vector de 22 características en el orden exacto de entrenamiento.
    2. Aplica la normalización Z-score (z = (x - mean_train) / std_train).
    3. Propaga el tensor por las capas Dense(64)+BN+Dropout -> Dense(32)+BN+Dropout -> Dense(16)+Dropout -> Sigmoid(1).
    4. Genera el diagnóstico explicable por dimensión y recomendaciones de tutoría.
    """
    input_dict = payload.model_dump()
    raw_vector = np.array([float(input_dict[col]) for col in FEATURE_COLUMNS], dtype=np.float32)

    # Normalización Z-score exacta
    z_vector = (raw_vector - SCALER_MEAN) / SCALER_STD
    z_batch = np.expand_dims(z_vector, axis=0)

    # Inferencia en la Red Neuronal Keras
    prob = float(MODEL.predict(z_batch, verbose=0)[0][0])
    pred_class = 1 if prob >= 0.50 else 0

    if prob >= 0.65:
        risk_level = "CRÍTICO"
        risk_color = "#ef4444"
        action_priority = "Intervención Inmediata (Prioridad 1 — 24 a 48 horas)"
    elif prob >= 0.38:
        risk_level = "MODERADO"
        risk_color = "#f59e0b"
        action_priority = "Seguimiento Preventivo (Prioridad 2 — Semanal)"
    else:
        risk_level = "BAJO / ESTABLE"
        risk_color = "#10b981"
        action_priority = "Monitoreo Regular (Trayectoria Exitosa)"

    # Cálculo de índices de vulnerabilidad por dimensión (0 a 100%) para el gráfico de barras del alumno
    acad_risk = max(0.0, min(100.0, (100.0 - payload.gpa_score_avg) * 0.45 + (100.0 - payload.first_assessment_score) * 0.35 + (1.0 - payload.pass_rate) * 100.0 * 0.20))
    lms_risk = max(0.0, min(100.0, (100.0 - payload.attendance_rate) * 0.30 + (100.0 - payload.lms_engagement_score) * 0.30 + (payload.missed_submissions_count / 10.0) * 100.0 * 0.25 + (payload.longest_inactivity_gap_days / 14.0) * 100.0 * 0.15))
    fin_risk = max(0.0, min(100.0, (0.0 if payload.tuition_up_to_date == 1 else 60.0) + (0.0 if payload.scholarship_holder == 1 else 20.0) + payload.socioeconomic_deprivation_index * 2.0))
    psych_risk = max(0.0, min(100.0, (payload.stress_level_scale / 10.0) * 65.0 + max(0.0, (7.5 - payload.sleep_hours_per_day) / 4.5) * 35.0))
    socio_risk = max(0.0, min(100.0, (payload.socioeconomic_deprivation_index / 10.0) * 75.0 + (25.0 if payload.first_generation == 1 else 0.0)))

    # Detección de factores de alerta específicos y recomendaciones personalizadas
    alerts = []
    recommendations = []

    if payload.gpa_score_avg < 60 or payload.first_assessment_score < 60:
        alerts.append(f"Rendimiento académico crítico: GPA ({payload.gpa_score_avg:.1f}) y 1ra Evaluación ({payload.first_assessment_score:.1f}) por debajo del umbral aprobatorio.")
        recommendations.append("Asignar tutoría académica par y nivelación intensiva antes del segundo examen parcial.")

    if payload.missed_submissions_count >= 3 or payload.longest_inactivity_gap_days >= 7:
        alerts.append(f"Desvinculación en Plataforma Virtual (LMS): {payload.missed_submissions_count} tareas no entregadas y {payload.longest_inactivity_gap_days} días de inactividad continua.")
        recommendations.append("Activar alerta en coordinación de carrera para contacto telefónico inmediato por ausentismo en plataforma PatmOS.")

    if payload.attendance_rate < 75:
        alerts.append(f"Baja asistencia presencial/sincrónica ({payload.attendance_rate:.1f}%).")
        recommendations.append("Citar al estudiante a entrevista con el tutor de ciclo para evaluar motivos de inasistencia.")

    if payload.tuition_up_to_date == 0 or payload.socioeconomic_deprivation_index >= 6.5:
        alerts.append(f"Vulnerabilidad financiera: Pensiones con retraso y privación socioeconómica de {payload.socioeconomic_deprivation_index:.1f}/10.")
        recommendations.append("Derivar a Bienestar Universitario para evaluación de refinanciamiento, bolsa de trabajo o beca socioeconómica.")

    if payload.stress_level_scale >= 7.0 or payload.sleep_hours_per_day <= 5.5:
        alerts.append(f"Sobrecarga psicosocial: Estrés académico alto ({payload.stress_level_scale:.1f}/10) con solo {payload.sleep_hours_per_day:.1f} h de sueño/día.")
        recommendations.append("Programar sesión de acompañamiento con el servicio de Psicopedagogía Universitaria.")

    if not alerts:
        alerts.append("Sin alertas críticas: El estudiante mantiene indicadores académicos, financieros y conductuales saludables.")
        recommendations.append("Mantener seguimiento ordinario e incentivar su postulación a grupos de investigación o ayudantías de cátedra.")

    return {
        "student_name": payload.student_name,
        "probability": round(prob, 4),
        "probability_pct": round(prob * 100, 2),
        "prediction": pred_class,
        "prediction_label": "EN RIESGO DE DESERCIÓN (1)" if pred_class == 1 else "RETENCIÓN REGULAR / SIN RIESGO (0)",
        "risk_level": risk_level,
        "risk_color": risk_color,
        "action_priority": action_priority,
        "dimension_scores": {
            "Rendimiento Académico": round(acad_risk, 1),
            "Inactividad en Aula Virtual (LMS)": round(lms_risk, 1),
            "Vulnerabilidad Financiera": round(fin_risk, 1),
            "Estrés y Fatiga (Salud Mental)": round(psych_risk, 1),
            "Riesgo Socioeconómico": round(socio_risk, 1),
        },
        "alerts": alerts,
        "recommendations": recommendations,
        "z_score_sample": {
            "gpa_score_avg_z": round(float(z_vector[FEATURE_COLUMNS.index("gpa_score_avg")]), 3),
            "first_assessment_z": round(float(z_vector[FEATURE_COLUMNS.index("first_assessment_score")]), 3),
            "missed_submissions_z": round(float(z_vector[FEATURE_COLUMNS.index("missed_submissions_count")]), 3),
            "lms_engagement_z": round(float(z_vector[FEATURE_COLUMNS.index("lms_engagement_score")]), 3),
        },
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
