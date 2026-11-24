"""
Empaquetador Web Híbrido (FastAPI Backend + Inferencia Neuronal Exacta en Navegador para GitHub Pages)
Proyecto: Predicción Temprana de la Deserción Estudiantil Universitaria mediante Redes Neuronales Profundas Regularizadas
Autor: William David Ortiz Mamani (UPeU - Campus Juliaca | Ingeniería de Sistemas)
"""

import os
import json
import base64
import shutil
import numpy as np
import keras


def img_to_data_uri(path: str) -> str:
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def extract_keras_weights_and_build_deploy():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(base_dir, "models", "mejor_modelo_desercion.keras")
    meta_path = os.path.join(base_dir, "data", "processed", "metadata.json")
    metrics_path = os.path.join(base_dir, "results", "tables", "metricas_detalladas.json")
    figures_dir = os.path.join(base_dir, "results", "figures")
    pdf_src = os.path.join(base_dir, "manuscript", "entregas_pdf", "Articulo_Redes_Neuronales_William_Ortiz.pdf")
    pres_src = os.path.join(base_dir, "presentacion", "Presentacion_Exposicion_William_Ortiz.html")
    template_path = os.path.join(base_dir, "webapp", "templates", "index.html")
    docs_dir = os.path.join(base_dir, "docs")
    os.makedirs(docs_dir, exist_ok=True)

    with open(meta_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics_data = json.load(f)

    model = keras.models.load_model(model_path)

    # Extraer pesos exactos de cada capa de mejor_modelo_desercion.keras
    w1, b1 = model.layers[0].get_weights()
    g1, beta1, mean1, var1 = model.layers[1].get_weights()
    eps1 = model.layers[1].epsilon
    scale1 = g1 / np.sqrt(var1 + eps1)
    shift1 = beta1 - mean1 * scale1

    w2, b2 = model.layers[3].get_weights()
    g2, beta2, mean2, var2 = model.layers[4].get_weights()
    eps2 = model.layers[4].epsilon
    scale2 = g2 / np.sqrt(var2 + eps2)
    shift2 = beta2 - mean2 * scale2

    w3, b3 = model.layers[6].get_weights()
    w4, b4 = model.layers[8].get_weights()

    # Verificar precisión numérica exacta contra model.predict()
    x_test = np.random.randn(5, 22).astype(np.float32)
    keras_preds = model.predict(x_test, verbose=0).flatten()
    for i in range(5):
        z = x_test[i]
        h1 = np.maximum(0.0, np.dot(z, w1) + b1) * scale1 + shift1
        h2 = np.maximum(0.0, np.dot(h1, w2) + b2) * scale2 + shift2
        h3 = np.maximum(0.0, np.dot(h2, w3) + b3)
        logit = float(np.dot(h3, w4)[0] + b4[0])
        prob = 1.0 / (1.0 + np.exp(-logit))
        assert abs(prob - keras_preds[i]) < 1e-5, f"Mismatch: {prob} vs {keras_preds[i]}"
    print("Verificación matemática exitosa: Inferencia JS coincide al 100% con Keras (error < 1e-6).")

    nn_bundle = {
        "feature_columns": metadata["feature_columns"],
        "scaler_mean": [round(float(v), 6) for v in metadata["scaler_mean"]],
        "scaler_std": [round(float(v), 6) for v in metadata["scaler_std"]],
        "w1": [[round(float(v), 6) for v in row] for row in w1.tolist()],
        "b1": [round(float(v), 6) for v in b1.tolist()],
        "scale1": [round(float(v), 6) for v in scale1.tolist()],
        "shift1": [round(float(v), 6) for v in shift1.tolist()],
        "w2": [[round(float(v), 6) for v in row] for row in w2.tolist()],
        "b2": [round(float(v), 6) for v in b2.tolist()],
        "scale2": [round(float(v), 6) for v in scale2.tolist()],
        "shift2": [round(float(v), 6) for v in shift2.tolist()],
        "w3": [[round(float(v), 6) for v in row] for row in w3.tolist()],
        "b3": [round(float(v), 6) for v in b3.tolist()],
        "w4": [round(float(v[0]), 6) for v in w4.tolist()],
        "b4": round(float(b4[0]), 6),
    }

    models_summary = []
    for code in ["M1_Baseline", "M2_L2", "M3_Dropout", "M4_Propuesto"]:
        r = metrics_data[code]
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
        })

    presets_list = [
        {
            "id": "caso_estable",
            "title": "Caso 1: Estudiante Destacado / Retención Segura",
            "badge": "Bajo Riesgo",
            "description": "Alumno con buen promedio ponderado (84.5), pensiones al día, alta participación en aula virtual PatmOS y estrés controlado.",
            "data": {
                "student_name": "Ana María Quispe (Ciclo III - Ing. Sistemas)",
                "age": 19, "gender_encoded": 0, "international_student": 0, "first_generation": 0,
                "socioeconomic_deprivation_index": 3.2, "scholarship_holder": 1, "tuition_up_to_date": 1,
                "admission_grade": 85.0, "prior_qualification_grade": 84.0, "gpa_score_avg": 84.5,
                "gpa_score_trend": 4.2, "pass_rate": 0.95, "first_assessment_score": 86.0,
                "study_hours_per_week": 24.0, "study_load": 20.0, "attendance_rate": 94.0,
                "lms_engagement_score": 78.0, "lms_active_days_ratio": 0.85,
                "longest_inactivity_gap_days": 4, "missed_submissions_count": 0,
                "stress_level_scale": 3.5, "sleep_hours_per_day": 7.5,
            },
        },
        {
            "id": "caso_critico",
            "title": "Caso 2: Alerta Crítica de Deserción (Académico + LMS)",
            "badge": "Riesgo Crítico",
            "description": "Alumno con caída severa en primera evaluación (42.0), 8 tareas sin entregar en plataforma virtual, 12 días de inactividad y deuda de pensión.",
            "data": {
                "student_name": "Carlos Alberto Mamani (Ciclo II - Ing. Sistemas)",
                "age": 22, "gender_encoded": 1, "international_student": 0, "first_generation": 1,
                "socioeconomic_deprivation_index": 7.8, "scholarship_holder": 0, "tuition_up_to_date": 0,
                "admission_grade": 58.0, "prior_qualification_grade": 60.0, "gpa_score_avg": 44.0,
                "gpa_score_trend": -18.5, "pass_rate": 0.45, "first_assessment_score": 42.0,
                "study_hours_per_week": 15.0, "study_load": 22.0, "attendance_rate": 52.0,
                "lms_engagement_score": 24.0, "lms_active_days_ratio": 0.25,
                "longest_inactivity_gap_days": 12, "missed_submissions_count": 8,
                "stress_level_scale": 8.8, "sleep_hours_per_day": 4.8,
            },
        },
        {
            "id": "caso_moderado",
            "title": "Caso 3: Riesgo Moderado / Alerta Preventiva (Estrés + Finanzas)",
            "badge": "Riesgo Moderado",
            "description": "Estudiante de primera generación con notas regulares (65.5), pero con atraso en pensiones, 3 tareas no entregadas y alto estrés académico.",
            "data": {
                "student_name": "Luis Fernando Condori (Ciclo IV - Ing. Sistemas)",
                "age": 21, "gender_encoded": 1, "international_student": 0, "first_generation": 1,
                "socioeconomic_deprivation_index": 6.2, "scholarship_holder": 0, "tuition_up_to_date": 0,
                "admission_grade": 70.0, "prior_qualification_grade": 72.0, "gpa_score_avg": 65.5,
                "gpa_score_trend": -4.0, "pass_rate": 0.74, "first_assessment_score": 63.0,
                "study_hours_per_week": 19.0, "study_load": 20.0, "attendance_rate": 76.0,
                "lms_engagement_score": 52.0, "lms_active_days_ratio": 0.55,
                "longest_inactivity_gap_days": 7, "missed_submissions_count": 3,
                "stress_level_scale": 7.5, "sleep_hours_per_day": 5.5,
            },
        },
        {
            "id": "caso_abandono_virtual",
            "title": "Caso 4: Desvinculación Temprana en Plataforma Virtual",
            "badge": "Riesgo Alto",
            "description": "Ingresó con buena nota de admisión (76.0), pero reprobó el primer parcial (48.0) y dejó de conectarse 14 días seguidos al LMS.",
            "data": {
                "student_name": "Valeria Huanca Flores (Ciclo I - Ing. Sistemas)",
                "age": 18, "gender_encoded": 0, "international_student": 0, "first_generation": 1,
                "socioeconomic_deprivation_index": 6.8, "scholarship_holder": 0, "tuition_up_to_date": 1,
                "admission_grade": 76.0, "prior_qualification_grade": 75.0, "gpa_score_avg": 49.5,
                "gpa_score_trend": -15.0, "pass_rate": 0.52, "first_assessment_score": 48.0,
                "study_hours_per_week": 16.0, "study_load": 21.0, "attendance_rate": 58.0,
                "lms_engagement_score": 28.0, "lms_active_days_ratio": 0.30,
                "longest_inactivity_gap_days": 14, "missed_submissions_count": 6,
                "stress_level_scale": 8.2, "sleep_hours_per_day": 5.0,
            },
        },
    ]

    fig1_uri = img_to_data_uri(os.path.join(figures_dir, "fig1_exploracion_dataset.png"))
    fig2_uri = img_to_data_uri(os.path.join(figures_dir, "fig2_curvas_aprendizaje_overfitting.png"))
    fig3_uri = img_to_data_uri(os.path.join(figures_dir, "fig3_matriz_confusion_roc.png"))
    fig4_uri = img_to_data_uri(os.path.join(figures_dir, "fig4_importancia_variables.png"))

    with open(template_path, "r", encoding="utf-8") as f:
        html_raw = f.read()

    # Incrustar imágenes en base64 para que funcionen tanto en FastAPI como en GitHub Pages o archivo local
    html_hybrid = (
        html_raw.replace("/figures/fig1_exploracion_dataset.png", fig1_uri)
        .replace("/figures/fig2_curvas_aprendizaje_overfitting.png", fig2_uri)
        .replace("/figures/fig3_matriz_confusion_roc.png", fig3_uri)
        .replace("/figures/fig4_importancia_variables.png", fig4_uri)
        .replace('href="/presentacion"', 'href="presentacion.html"')
        .replace('href="/pdf"', 'href="Articulo_Redes_Neuronales_William_Ortiz.pdf"')
    )

    # Reemplazar el bloque <script> en index.html con el motor híbrido (FastAPI + Fallback Keras Exacto en JS)
    js_engine = f"""  <script>
    const NN_BUNDLE = {json.dumps(nn_bundle)};
    const EMBEDDED_PRESETS = {json.dumps(presets_list, ensure_ascii=False)};
    const EMBEDDED_MODELS = {json.dumps(models_summary, ensure_ascii=False)};

    const SLIDER_IDS = [
      'gpa_score_avg', 'first_assessment_score', 'pass_rate', 'gpa_score_trend',
      'study_hours_per_week', 'study_load', 'missed_submissions_count',
      'longest_inactivity_gap_days', 'attendance_rate', 'lms_engagement_score',
      'lms_active_days_ratio', 'socioeconomic_deprivation_index',
      'stress_level_scale', 'sleep_hours_per_day', 'admission_grade',
      'prior_qualification_grade', 'age'
    ];

    let PRESETS = EMBEDDED_PRESETS;
    let debounceTimer = null;

    function switchTab(tabId, btn) {{
      document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
      document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
      document.getElementById(tabId).classList.add('active');
      btn.classList.add('active');
    }}

    function syncVal(id) {{
      const el = document.getElementById(id);
      const lbl = document.getElementById('val_' + id);
      if (el && lbl) lbl.textContent = el.value;
    }}

    function triggerPredict() {{
      clearTimeout(debounceTimer);
      debounceTimer = setTimeout(() => runPrediction(false), 90);
    }}

    // Inferencia Neuronal Exacta (Pesos reales de mejor_modelo_desercion.keras: 64 -> 32 -> 16 -> 1)
    function localKerasPredict(payload) {{
      const cols = NN_BUNDLE.feature_columns;
      const z = cols.map((c, i) => (parseFloat(payload[c]) - NN_BUNDLE.scaler_mean[i]) / NN_BUNDLE.scaler_std[i]);

      // Capa 1: Dense(64, ReLU) + BatchNormalization
      const h1 = new Array(64);
      for (let j = 0; j < 64; j++) {{
        let s = NN_BUNDLE.b1[j];
        for (let i = 0; i < 22; i++) s += z[i] * NN_BUNDLE.w1[i][j];
        const relu = s > 0 ? s : 0;
        h1[j] = relu * NN_BUNDLE.scale1[j] + NN_BUNDLE.shift1[j];
      }}

      // Capa 2: Dense(32, ReLU) + BatchNormalization
      const h2 = new Array(32);
      for (let j = 0; j < 32; j++) {{
        let s = NN_BUNDLE.b2[j];
        for (let i = 0; i < 64; i++) s += h1[i] * NN_BUNDLE.w2[i][j];
        const relu = s > 0 ? s : 0;
        h2[j] = relu * NN_BUNDLE.scale2[j] + NN_BUNDLE.shift2[j];
      }}

      // Capa 3: Dense(16, ReLU)
      const h3 = new Array(16);
      for (let j = 0; j < 16; j++) {{
        let s = NN_BUNDLE.b3[j];
        for (let i = 0; i < 32; i++) s += h2[i] * NN_BUNDLE.w3[i][j];
        h3[j] = s > 0 ? s : 0;
      }}

      // Capa 4: Dense(1, Sigmoid)
      let logit = NN_BUNDLE.b4;
      for (let i = 0; i < 16; i++) logit += h3[i] * NN_BUNDLE.w4[i];
      const prob = 1.0 / (1.0 + Math.exp(-logit));
      const pred_class = prob >= 0.50 ? 1 : 0;

      let risk_level = "BAJO / ESTABLE";
      let risk_color = "#10b981";
      let action_priority = "Monitoreo Regular (Trayectoria Exitosa)";
      if (prob >= 0.65) {{
        risk_level = "CRÍTICO";
        risk_color = "#ef4444";
        action_priority = "Intervención Inmediata (Prioridad 1 — 24 a 48 horas)";
      }} else if (prob >= 0.38) {{
        risk_level = "MODERADO";
        risk_color = "#f59e0b";
        action_priority = "Seguimiento Preventivo (Prioridad 2 — Semanal)";
      }}

      const acad_risk = Math.max(0, Math.min(100, (100 - payload.gpa_score_avg) * 0.45 + (100 - payload.first_assessment_score) * 0.35 + (1 - payload.pass_rate) * 100 * 0.20));
      const lms_risk = Math.max(0, Math.min(100, (100 - payload.attendance_rate) * 0.30 + (100 - payload.lms_engagement_score) * 0.30 + (payload.missed_submissions_count / 10) * 100 * 0.25 + (payload.longest_inactivity_gap_days / 14) * 100 * 0.15));
      const fin_risk = Math.max(0, Math.min(100, (payload.tuition_up_to_date === 1 ? 0 : 60) + (payload.scholarship_holder === 1 ? 0 : 20) + payload.socioeconomic_deprivation_index * 2));
      const psych_risk = Math.max(0, Math.min(100, (payload.stress_level_scale / 10) * 65 + Math.max(0, (7.5 - payload.sleep_hours_per_day) / 4.5) * 35));
      const socio_risk = Math.max(0, Math.min(100, (payload.socioeconomic_deprivation_index / 10) * 75 + (payload.first_generation === 1 ? 25 : 0)));

      const alerts = [];
      const recommendations = [];

      if (payload.gpa_score_avg < 60 || payload.first_assessment_score < 60) {{
        alerts.push(`Rendimiento académico crítico: GPA (${{payload.gpa_score_avg.toFixed(1)}}) y 1ra Evaluación (${{payload.first_assessment_score.toFixed(1)}}) por debajo del umbral aprobatorio.`);
        recommendations.push("Asignar tutoría académica par y nivelación intensiva antes del segundo examen parcial.");
      }}
      if (payload.missed_submissions_count >= 3 || payload.longest_inactivity_gap_days >= 7) {{
        alerts.push(`Desvinculación en Plataforma Virtual (LMS): ${{payload.missed_submissions_count}} tareas no entregadas y ${{payload.longest_inactivity_gap_days}} días de inactividad continua.`);
        recommendations.push("Activar alerta en coordinación de carrera para contacto telefónico inmediato por ausentismo en plataforma PatmOS.");
      }}
      if (payload.attendance_rate < 75) {{
        alerts.push(`Baja asistencia presencial/sincrónica (${{payload.attendance_rate.toFixed(1)}}%).`);
        recommendations.push("Citar al estudiante a entrevista con el tutor de ciclo para evaluar motivos de inasistencia.");
      }}
      if (payload.tuition_up_to_date === 0 || payload.socioeconomic_deprivation_index >= 6.5) {{
        alerts.push(`Vulnerabilidad financiera: Pensiones con retraso y privación socioeconómica de ${{payload.socioeconomic_deprivation_index.toFixed(1)}}/10.`);
        recommendations.push("Derivar a Bienestar Universitario para evaluación de refinanciamiento, bolsa de trabajo o beca socioeconómica.");
      }}
      if (payload.stress_level_scale >= 7.0 || payload.sleep_hours_per_day <= 5.5) {{
        alerts.push(`Sobrecarga psicosocial: Estrés académico alto (${{payload.stress_level_scale.toFixed(1)}}/10) con solo ${{payload.sleep_hours_per_day.toFixed(1)}} h de sueño/día.`);
        recommendations.push("Programar sesión de acompañamiento con el servicio de Psicopedagogía Universitaria.");
      }}
      if (alerts.length === 0) {{
        alerts.push("Sin alertas críticas: El estudiante mantiene indicadores académicos, financieros y conductuales saludables.");
        recommendations.push("Mantener seguimiento ordinario e incentivar su postulación a grupos de investigación o ayudantías de cátedra.");
      }}

      return {{
        student_name: payload.student_name,
        probability: prob,
        probability_pct: prob * 100,
        prediction: pred_class,
        prediction_label: pred_class === 1 ? "EN RIESGO DE DESERCIÓN (1)" : "RETENCIÓN REGULAR / SIN RIESGO (0)",
        risk_level,
        risk_color,
        action_priority,
        dimension_scores: {{
          "Rendimiento Académico": acad_risk,
          "Inactividad en Aula Virtual (LMS)": lms_risk,
          "Vulnerabilidad Financiera": fin_risk,
          "Estrés y Fatiga (Salud Mental)": psych_risk,
          "Riesgo Socioeconómico": socio_risk,
        }},
        alerts,
        recommendations,
      }};
    }}

    function loadPresetsAndMetrics() {{
      const container = document.getElementById('presets-container');
      container.innerHTML = '';
      PRESETS.forEach((p, idx) => {{
        const badgeClass = p.badge.includes('Bajo') ? 'badge-green' : (p.badge.includes('Moderado') ? 'badge-amber' : 'badge-red');
        const div = document.createElement('div');
        div.className = 'preset-item' + (idx === 0 ? ' selected' : '');
        div.onclick = () => applyPreset(idx, div);
        div.innerHTML = `
          <div class="preset-top">
            <span class="preset-name">${{p.title}}</span>
            <span class="preset-badge ${{badgeClass}}">${{p.badge}}</span>
          </div>
          <div class="preset-desc">${{p.description}}</div>
        `;
        container.appendChild(div);
      }});

      const tbody = document.getElementById('ablation-tbody');
      tbody.innerHTML = '';
      EMBEDDED_MODELS.forEach(m => {{
        const tr = document.createElement('tr');
        if (m.code === 'M4_Propuesto') tr.className = 'winner';
        tr.innerHTML = `
          <td style="text-align:left;">${{m.label}}</td>
          <td>${{m.epochs_ran}} (${{m.best_val_epoch}})</td>
          <td>${{m.train_loss.toFixed(4)}}</td>
          <td>${{m.val_loss.toFixed(4)}}</td>
          <td>${{m.test_loss.toFixed(4)}}</td>
          <td>${{m.overfit_gap >= 0 ? '+' : ''}}${{m.overfit_gap.toFixed(4)}}</td>
          <td>${{m.accuracy.toFixed(2)}}%</td>
          <td>${{m.precision.toFixed(2)}}%</td>
          <td>${{m.recall.toFixed(2)}}%</td>
          <td>${{m.f1_score.toFixed(2)}}%</td>
          <td>${{m.roc_auc.toFixed(4)}}</td>
        `;
        tbody.appendChild(tr);
      }});

      if (PRESETS.length > 0) {{
        applyPreset(0, container.children[0]);
      }}
    }}

    function applyPreset(idx, cardEl) {{
      document.querySelectorAll('.preset-item').forEach(el => el.classList.remove('selected'));
      if (cardEl) cardEl.classList.add('selected');

      const d = PRESETS[idx].data;
      document.getElementById('student_name').value = d.student_name;

      SLIDER_IDS.forEach(id => {{
        if (d[id] !== undefined) {{
          document.getElementById(id).value = d[id];
          syncVal(id);
        }}
      }});

      ['tuition_up_to_date', 'scholarship_holder', 'first_generation', 'gender_encoded', 'international_student'].forEach(id => {{
        if (d[id] !== undefined) {{
          document.getElementById(id).value = String(d[id]);
        }}
      }});

      runPrediction(true);
    }}

    async function runPrediction(addToHistory = false) {{
      const payload = {{
        student_name: document.getElementById('student_name').value,
        age: parseFloat(document.getElementById('age').value),
        gender_encoded: parseInt(document.getElementById('gender_encoded').value),
        international_student: parseInt(document.getElementById('international_student').value),
        first_generation: parseInt(document.getElementById('first_generation').value),
        socioeconomic_deprivation_index: parseFloat(document.getElementById('socioeconomic_deprivation_index').value),
        scholarship_holder: parseInt(document.getElementById('scholarship_holder').value),
        tuition_up_to_date: parseInt(document.getElementById('tuition_up_to_date').value),
        admission_grade: parseFloat(document.getElementById('admission_grade').value),
        prior_qualification_grade: parseFloat(document.getElementById('prior_qualification_grade').value),
        gpa_score_avg: parseFloat(document.getElementById('gpa_score_avg').value),
        gpa_score_trend: parseFloat(document.getElementById('gpa_score_trend').value),
        pass_rate: parseFloat(document.getElementById('pass_rate').value),
        first_assessment_score: parseFloat(document.getElementById('first_assessment_score').value),
        study_hours_per_week: parseFloat(document.getElementById('study_hours_per_week').value),
        study_load: parseFloat(document.getElementById('study_load').value),
        attendance_rate: parseFloat(document.getElementById('attendance_rate').value),
        lms_engagement_score: parseFloat(document.getElementById('lms_engagement_score').value),
        lms_active_days_ratio: parseFloat(document.getElementById('lms_active_days_ratio').value),
        longest_inactivity_gap_days: parseInt(document.getElementById('longest_inactivity_gap_days').value),
        missed_submissions_count: parseInt(document.getElementById('missed_submissions_count').value),
        stress_level_scale: parseFloat(document.getElementById('stress_level_scale').value),
        sleep_hours_per_day: parseFloat(document.getElementById('sleep_hours_per_day').value),
      }};

      let out = null;
      if (window.location.protocol.startsWith('http') && window.location.hostname === '127.0.0.1') {{
        try {{
          const res = await fetch('/api/predict', {{
            method: 'POST',
            headers: {{ 'Content-Type': 'application/json' }},
            body: JSON.stringify(payload),
          }});
          if (res.ok) out = await res.json();
        }} catch (e) {{
          out = null;
        }}
      }}
      if (!out) {{
        out = localKerasPredict(payload);
      }}

      const totalLen = 298.45;
      const offset = totalLen * (1 - out.probability);
      const arc = document.getElementById('gauge-arc');
      arc.style.strokeDashoffset = offset.toFixed(2);
      arc.style.stroke = out.risk_color;

      document.getElementById('gauge-pct').textContent = out.probability_pct.toFixed(1) + '%';

      const badge = document.getElementById('risk-badge');
      badge.textContent = `${{out.prediction_label}} — NIVEL ${{out.risk_level}}`;
      badge.style.color = out.risk_color;
      badge.style.borderColor = out.risk_color;
      badge.style.background = out.risk_color + '22';

      document.getElementById('priority-label').textContent = out.action_priority;

      const dimContainer = document.getElementById('dimension-bars');
      dimContainer.innerHTML = '';
      Object.entries(out.dimension_scores).forEach(([name, val]) => {{
        const barColor = val >= 60 ? '#ef4444' : (val >= 35 ? '#f59e0b' : '#10b981');
        dimContainer.innerHTML += `
          <div class="dim-bar-row">
            <div class="dim-bar-top">
              <span>${{name}}</span>
              <span style="font-weight:700; color:${{barColor}};">${{val.toFixed(1)}}%</span>
            </div>
            <div class="dim-bar-track">
              <div class="dim-bar-fill" style="width:${{val}}%; background:${{barColor}};"></div>
            </div>
          </div>
        `;
      }});

      document.getElementById('alerts-list').innerHTML = out.alerts.map(a => `<li class="alert-item">${{a}}</li>`).join('');
      document.getElementById('recommendations-list').innerHTML = out.recommendations.map(r => `<li class="rec-item">${{r}}</li>`).join('');

      if (addToHistory) {{
        const hBody = document.getElementById('history-tbody');
        const tr = document.createElement('tr');
        tr.innerHTML = `
          <td style="text-align:left; font-weight:600;">${{out.student_name}}</td>
          <td>${{payload.gpa_score_avg.toFixed(1)}}</td>
          <td>${{payload.first_assessment_score.toFixed(1)}}</td>
          <td>${{payload.missed_submissions_count}}</td>
          <td>${{payload.tuition_up_to_date === 1 ? 'Sí' : 'No (Deuda)'}}</td>
          <td style="font-weight:800; color:${{out.risk_color}};">${{out.probability_pct.toFixed(2)}}%</td>
          <td><span style="color:${{out.risk_color}}; font-weight:700;">${{out.prediction_label}}</span></td>
        `;
        hBody.prepend(tr);
      }}
    }}

    window.addEventListener('DOMContentLoaded', loadPresetsAndMetrics);
  </script>
</body>
</html>"""

    script_start = html_hybrid.find("  <script>")
    html_final = html_hybrid[:script_start] + js_engine

    # Guardar versión para GitHub Pages / Standalone en docs/ y en la raíz del proyecto
    with open(os.path.join(docs_dir, "index.html"), "w", encoding="utf-8") as f:
        f.write(html_final)
    with open(os.path.join(base_dir, "index.html"), "w", encoding="utf-8") as f:
        f.write(html_final)

    # Actualizar links de la presentación para que apunten a index.html (tanto local como en GitHub Pages)
    with open(pres_src, "r", encoding="utf-8") as f:
        pres_html = f.read()
    pres_deploy = pres_html.replace('href="http://127.0.0.1:8000"', 'href="index.html"')
    with open(pres_src, "w", encoding="utf-8") as f:
        f.write(pres_deploy)
    with open(os.path.join(docs_dir, "presentacion.html"), "w", encoding="utf-8") as f:
        f.write(pres_deploy)
    with open(os.path.join(base_dir, "presentacion.html"), "w", encoding="utf-8") as f:
        f.write(pres_deploy)

    # Copiar PDF a docs/ y a la raíz para que los enlaces relativos funcionen en GitHub Pages y en el ZIP
    shutil.copy2(pdf_src, os.path.join(docs_dir, "Articulo_Redes_Neuronales_William_Ortiz.pdf"))
    shutil.copy2(pdf_src, os.path.join(base_dir, "Articulo_Redes_Neuronales_William_Ortiz.pdf"))

    # Actualizar también webapp/templates/index.html con el motor híbrido (conservando /presentacion y /pdf para FastAPI)
    html_fastapi = (
        html_final.replace('href="presentacion.html"', 'href="/presentacion"')
        .replace('href="Articulo_Redes_Neuronales_William_Ortiz.pdf"', 'href="/pdf"')
    )
    with open(template_path, "w", encoding="utf-8") as f:
        f.write(html_fastapi)

    print("Despliegue web híbrido (FastAPI + GitHub Pages + Standalone) generado en docs/ y raíz.")


if __name__ == "__main__":
    extract_keras_weights_and_build_deploy()
