"""
Generador de Presentación Interactiva en HTML5 para Exposición y Sustentación
Proyecto: Predicción Temprana de la Deserción Estudiantil Universitaria mediante Redes Neuronales Profundas Regularizadas
Autor: William David Ortiz Mamani (UPeU - Campus Juliaca | Ingeniería de Sistemas)
"""

import os
import json
import base64


def img_to_data_uri(path: str) -> str:
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def build_html_presentation():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    figures_dir = os.path.join(base_dir, "results", "figures")
    pres_dir = os.path.join(base_dir, "presentacion")
    os.makedirs(pres_dir, exist_ok=True)

    fig1_uri = img_to_data_uri(os.path.join(figures_dir, "fig1_exploracion_dataset.png"))
    fig2_uri = img_to_data_uri(os.path.join(figures_dir, "fig2_curvas_aprendizaje_overfitting.png"))
    fig3_uri = img_to_data_uri(os.path.join(figures_dir, "fig3_matriz_confusion_roc.png"))
    fig4_uri = img_to_data_uri(os.path.join(figures_dir, "fig4_importancia_variables.png"))

    html = """<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Presentación de Exposición | William David Ortiz Mamani - UPeU</title>
  <style>
    :root {{
      --bg-deep: #070b14;
      --bg-slide: #0f172a;
      --bg-card: #162038;
      --border: #283a5e;
      --text-white: #f8fafc;
      --text-muted: #94a3b8;
      --blue: #38bdf8;
      --indigo: #6366f1;
      --green: #10b981;
      --amber: #f59e0b;
      --red: #ef4444;
    }}

    * {{
      box-sizing: border-box;
      margin: 0;
      padding: 0;
    }}

    body {{
      font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
      background: radial-gradient(circle at 80% 10%, #1e3a8a 0%, var(--bg-deep) 55%);
      color: var(--text-white);
      height: 100vh;
      overflow: hidden;
      display: flex;
      flex-direction: column;
    }}

    /* Top Control Bar */
    .top-bar {{
      height: 54px;
      background: rgba(15, 23, 42, 0.92);
      border-bottom: 1px solid var(--border);
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 0 24px;
      z-index: 50;
    }}

    .top-brand {{
      display: flex;
      align-items: center;
      gap: 12px;
      font-size: 13.5px;
      font-weight: 700;
    }}

    .upeu-badge {{
      background: linear-gradient(135deg, #2563eb, #06b6d4);
      color: white;
      padding: 4px 11px;
      border-radius: 7px;
      font-weight: 800;
      font-size: 12px;
    }}

    .top-controls {{
      display: flex;
      align-items: center;
      gap: 10px;
    }}

    .ctrl-btn {{
      background: var(--bg-card);
      color: var(--text-white);
      border: 1px solid var(--border);
      padding: 6px 13px;
      border-radius: 8px;
      font-size: 12.5px;
      font-weight: 600;
      cursor: pointer;
      text-decoration: none;
      transition: all 0.2s;
    }}

    .ctrl-btn:hover {{
      border-color: var(--blue);
      background: #1e293b;
    }}

    .ctrl-btn.primary {{
      background: linear-gradient(135deg, #10b981, #059669);
      border-color: transparent;
    }}

    /* Progress Bar */
    .progress-track {{
      height: 4px;
      background: #1e293b;
      width: 100%;
    }}

    .progress-fill {{
      height: 100%;
      background: linear-gradient(90deg, #38bdf8, #10b981);
      width: 10%;
      transition: width 0.35s ease;
    }}

    /* Slides Viewport */
    .deck-viewport {{
      flex: 1;
      position: relative;
      display: flex;
      align-items: center;
      justify-content: center;
      padding: 20px 36px;
      overflow: hidden;
    }}

    .slide {{
      display: none;
      width: 100%;
      max-width: 1360px;
      height: 100%;
      max-height: 760px;
      background: rgba(15, 23, 42, 0.88);
      border: 1px solid var(--border);
      border-radius: 20px;
      padding: 34px 44px;
      box-shadow: 0 20px 50px rgba(0, 0, 0, 0.6);
      flex-direction: column;
      justify-content: space-between;
      animation: fadeSlide 0.32s ease;
      overflow-y: auto;
    }}

    .slide.active {{
      display: flex;
    }}

    @keyframes fadeSlide {{
      from {{ opacity: 0; transform: translateY(10px) scale(0.99); }}
      to {{ opacity: 1; transform: translateY(0) scale(1); }}
    }}

    .slide-header {{
      border-bottom: 1px solid var(--border);
      padding-bottom: 12px;
      margin-bottom: 18px;
      display: flex;
      justify-content: space-between;
      align-items: flex-end;
    }}

    .section-tag {{
      font-size: 12px;
      font-weight: 800;
      color: var(--blue);
      text-transform: uppercase;
      letter-spacing: 1px;
      margin-bottom: 3px;
    }}

    .slide-title {{
      font-size: 26px;
      font-weight: 800;
      color: #ffffff;
    }}

    .slide-num-tag {{
      font-size: 13px;
      color: var(--text-muted);
      font-family: monospace;
    }}

    .slide-body {{
      flex: 1;
      display: flex;
      flex-direction: column;
      justify-content: center;
      gap: 18px;
    }}

    /* Grids & Cards inside slides */
    .grid-2 {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 22px;
      align-items: center;
    }}

    .grid-3 {{
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 16px;
    }}

    .grid-4 {{
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 16px;
    }}

    .box-card {{
      background: var(--bg-card);
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 18px 20px;
    }}

    .box-card h3 {{
      font-size: 16px;
      color: var(--blue);
      margin-bottom: 8px;
    }}

    .box-card p, .box-card li {{
      font-size: 14px;
      color: #e2e8f0;
      line-height: 1.5;
    }}

    .box-card ul {{
      padding-left: 18px;
      display: flex;
      flex-direction: column;
      gap: 6px;
    }}

    .kpi-big {{
      font-size: 36px;
      font-weight: 900;
      color: var(--green);
      line-height: 1.1;
      margin: 6px 0;
    }}

    .img-frame {{
      background: #ffffff;
      padding: 8px;
      border-radius: 12px;
      text-align: center;
      box-shadow: 0 8px 25px rgba(0,0,0,0.4);
    }}

    .img-frame img {{
      width: 100%;
      max-height: 410px;
      object-fit: contain;
      border-radius: 8px;
    }}

    /* Table inside slide */
    table.slide-table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 14px;
      background: var(--bg-card);
      border-radius: 12px;
      overflow: hidden;
    }}

    table.slide-table th {{
      background: #1e293b;
      color: var(--blue);
      padding: 12px 10px;
      text-align: center;
      border-bottom: 2px solid var(--border);
    }}

    table.slide-table td {{
      padding: 12px 10px;
      text-align: center;
      border-bottom: 1px solid var(--border);
    }}

    table.slide-table tr.winner-row {{
      background: rgba(16, 185, 129, 0.18);
      font-weight: 800;
      color: #6ee7b7;
    }}

    /* Speaker Notes Box (Guion para William) */
    .speaker-notes {{
      margin-top: 14px;
      background: rgba(245, 158, 11, 0.12);
      border: 1px solid rgba(245, 158, 11, 0.4);
      border-left: 4px solid var(--amber);
      border-radius: 10px;
      padding: 10px 16px;
      font-size: 13px;
      color: #fde68a;
      display: block;
    }}

    .speaker-notes.hidden {{
      display: none;
    }}

    /* Bottom Navigation Bar */
    .bottom-bar {{
      height: 52px;
      background: rgba(15, 23, 42, 0.92);
      border-top: 1px solid var(--border);
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 0 28px;
    }}

    .dots-nav {{
      display: flex;
      gap: 8px;
    }}

    .dot {{
      width: 10px;
      height: 10px;
      border-radius: 50%;
      background: #334155;
      cursor: pointer;
      transition: all 0.2s;
    }}

    .dot.active {{
      background: var(--blue);
      width: 26px;
      border-radius: 99px;
    }}
  </style>
</head>
<body>

  <!-- Barra Superior -->
  <header class="top-bar">
    <div class="top-brand">
      <span class="upeu-badge">UPeU • ING. SISTEMAS</span>
      <span>Sustentación de Artículo Científico — William David Ortiz Mamani</span>
    </div>
    <div class="top-controls">
      <button class="ctrl-btn" onclick="toggleNotes()">💡 Ocultar/Mostrar Guion de Exposición (Tecla N)</button>
      <button class="ctrl-btn" onclick="toggleFullScreen()">⛶ Pantalla Completa (Tecla F)</button>
      <a href="http://127.0.0.1:8000" target="_blank" class="ctrl-btn primary">⚡ Ir al Sistema Web en Vivo (FastAPI)</a>
    </div>
  </header>

  <div class="progress-track">
    <div class="progress-fill" id="progress-bar"></div>
  </div>

  <!-- Contenedor de Diapositivas -->
  <main class="deck-viewport">

    <!-- ================================================================= -->
    <!-- DIAPOSITIVA 1: PORTADA                                            -->
    <!-- ================================================================= -->
    <section class="slide active">
      <div class="slide-body" style="text-align:center; align-items:center;">
        <div style="background:rgba(56,189,248,0.12); border:1px solid rgba(56,189,248,0.4); color:var(--blue); padding:6px 18px; border-radius:99px; font-size:13px; font-weight:700; letter-spacing:0.8px;">
          UNIVERSIDAD PERUANA UNIÓN (UPeU) — CAMPUS JULIACA • FACULTAD DE INGENIERÍA Y ARQUITECTURA
        </div>

        <h1 style="font-size:36px; font-weight:900; max-width:1050px; line-height:1.25; color:#ffffff;">
          Predicción Temprana de la Deserción Estudiantil Universitaria mediante Redes Neuronales Profundas con Regularización L2, Dropout y Batch Normalization
        </h1>

        <p style="font-size:17px; font-style:italic; color:var(--text-muted); max-width:920px;">
          Early Prediction of University Student Dropout Using Deep Neural Networks with L2 Regularization, Dropout, and Batch Normalization
        </p>

        <div class="grid-4" style="width:100%; max-width:1080px; margin-top:10px;">
          <div class="box-card">
            <div style="font-size:12px; color:var(--text-muted);">MUESTRA MULTIMODAL</div>
            <div class="kpi-big" style="color:var(--blue);">6,000</div>
            <div style="font-size:12.5px;">Estudiantes Universitarios (22 variables)</div>
          </div>
          <div class="box-card">
            <div style="font-size:12px; color:var(--text-muted);">REDUCCIÓN DE TEST LOSS</div>
            <div class="kpi-big">-78.3%</div>
            <div style="font-size:12.5px;">De 2.0283 (Baseline) a 0.4408 (M4)</div>
          </div>
          <div class="box-card">
            <div style="font-size:12px; color:var(--text-muted);">EXACTITUD (TEST ACC)</div>
            <div class="kpi-big">81.26%</div>
            <div style="font-size:12.5px;">En conjunto de prueba ciego (n=900)</div>
          </div>
          <div class="box-card">
            <div style="font-size:12px; color:var(--text-muted);">PODER DISCRIMINATIVO</div>
            <div class="kpi-big" style="color:#a855f7;">0.8752</div>
            <div style="font-size:12.5px;">Área Bajo la Curva ROC (ROC-AUC)</div>
          </div>
        </div>

        <div style="margin-top:10px; font-size:15px; line-height:1.6;">
          <div><b>Autor:</b> William David Ortiz Mamani (<code>william.ortiz@upeu.edu.pe</code> | GitHub: <code>302xdwill</code>)</div>
          <div><b>Curso:</b> Inteligencia Artificial &nbsp;•&nbsp; <b>Docente:</b> Mg. Angel Rosendo Condori Coaquira</div>
        </div>
      </div>

      <div class="speaker-notes">
        🎙️ <b>Guion para William (Diapositiva 1):</b> "Buenos días/tardes ingeniero y compañeros. Hoy presentaré mi artículo científico titulado 'Predicción Temprana de la Deserción Estudiantil Universitaria mediante Redes Neuronales Profundas con Regularización L2, Dropout y Batch Normalization', junto con su implementación práctica en un sistema web con FastAPI."
      </div>
    </section>

    <!-- ================================================================= -->
    <!-- DIAPOSITIVA 2: INTRODUCCIÓN Y PROBLEMÁTICA                        -->
    <!-- ================================================================= -->
    <section class="slide">
      <div class="slide-header">
        <div>
          <div class="section-tag">I. Introducción y Justificación</div>
          <div class="slide-title">¿Qué Problema Real y Técnico Resuelve nuestro Artículo?</div>
        </div>
        <span class="slide-num-tag">Diapositiva 02 / 10</span>
      </div>

      <div class="slide-body">
        <div class="grid-3">
          <div class="box-card" style="border-top:4px solid var(--red);">
            <h3>1. Problema Socioeducativo</h3>
            <ul>
              <li>Entre el <b>20% y 35%</b> de los universitarios abandonan su carrera en los primeros ciclos.</li>
              <li>Las universidades suelen detectar el abandono de manera <b>reactiva y tardía</b> (cuando el alumno ya no se matricula).</li>
              <li>Se requiere un sistema de <b>alerta temprana</b> desde las primeras semanas del semestre para que Tutoría Universitaria intervenga a tiempo.</li>
            </ul>
          </div>

          <div class="box-card" style="border-top:4px solid var(--amber);">
            <h3>2. El Reto Técnico en Redes Neuronales</h3>
            <ul>
              <li>La deserción depende de interacciones <b>no lineales</b> entre notas, finanzas, plataforma virtual (LMS) y estrés.</li>
              <li>Sin embargo, al entrenar una Red Neuronal Profunda sobre datos tabulares humanos, ocurre un severo <b>Sobreajuste (Overfitting)</b>: la red memoriza el ruido del entrenamiento y falla ante nuevos alumnos.</li>
            </ul>
          </div>

          <div class="box-card" style="border-top:4px solid var(--green);">
            <h3>3. Nuestro Objetivo Científico</h3>
            <ul>
              <li>Diseñar, entrenar y validar un <b>Estudio Ablativo de 4 Arquitecturas de Redes Neuronales en Keras</b>.</li>
              <li>Demostrar cómo la sinergia de <b>BatchNormalization + L2 + Dropout + EarlyStopping</b> erradica el overfitting.</li>
              <li>Desplegar el modelo ganador en un <b>Sistema Web Local con FastAPI (SIA-UPeU)</b>.</li>
            </ul>
          </div>
        </div>
      </div>

      <div class="speaker-notes">
        🎙️ <b>Guion para William (Diapositiva 2):</b> "El artículo aborda dos problemas: primero, la necesidad real en universidades como la UPeU de detectar qué alumno está en riesgo de desertar antes de que termine el semestre; y segundo, el problema técnico del Overfitting cuando aplicamos redes neuronales profundas sobre datos tabulares educativos."
      </div>
    </section>

    <!-- ================================================================= -->
    <!-- DIAPOSITIVA 3: DATASET MULTIMODAL Y 22 VARIABLES                  -->
    <!-- ================================================================= -->
    <section class="slide">
      <div class="slide-header">
        <div>
          <div class="section-tag">II. Materiales y Métodos (Metodología)</div>
          <div class="slide-title">Dataset Multimodal: N = 6,000 Estudiantes y 22 Variables (6 Dimensiones)</div>
        </div>
        <span class="slide-num-tag">Diapositiva 03 / 10</span>
      </div>

      <div class="slide-body">
        <div class="grid-3">
          <div class="box-card">
            <h3>1. Rendimiento Académico (6 vars)</h3>
            <ul>
              <li><code>gpa_score_avg</code>: Promedio ponderado (0-100)</li>
              <li><code>first_assessment_score</code>: Nota 1ra evaluación</li>
              <li><code>pass_rate</code>: Tasa de cursos aprobados</li>
              <li><code>gpa_score_trend</code>: Tendencia del promedio</li>
              <li><code>study_hours_per_week</code> y <code>study_load</code></li>
            </ul>
          </div>

          <div class="box-card">
            <h3>2. Plataforma Virtual LMS / PatmOS (5 vars)</h3>
            <ul>
              <li><code>missed_submissions_count</code>: Tareas no entregadas</li>
              <li><code>longest_inactivity_gap_days</code>: Días sin entrar al LMS</li>
              <li><code>attendance_rate</code>: Tasa de asistencia (%)</li>
              <li><code>lms_engagement_score</code>: Interacción digital</li>
              <li><code>lms_active_days_ratio</code>: Ratio de días activos</li>
            </ul>
          </div>

          <div class="box-card">
            <h3>3. Situación Financiera (2 vars)</h3>
            <ul>
              <li><code>tuition_up_to_date</code>: Pago de pensiones al día (1 = Sí, 0 = Deuda)</li>
              <li><code>scholarship_holder</code>: Beneficiario de beca académica o social</li>
            </ul>
          </div>

          <div class="box-card">
            <h3>4. Demografía y Socioeconomía (5 vars)</h3>
            <ul>
              <li><code>socioeconomic_deprivation_index</code>: Índice de privación económica (1 a 10)</li>
              <li><code>first_generation</code>: 1ra generación universitaria</li>
              <li><code>age</code>, <code>gender_encoded</code>, <code>international_student</code></li>
            </ul>
          </div>

          <div class="box-card">
            <h3>5. Historial Académico Previo (2 vars)</h3>
            <ul>
              <li><code>admission_grade</code>: Puntaje en examen de admisión</li>
              <li><code>prior_qualification_grade</code>: Promedio de estudios secundarios previos</li>
            </ul>
          </div>

          <div class="box-card">
            <h3>6. Factores Psicológicos y Sueño (2 vars)</h3>
            <ul>
              <li><code>stress_level_scale</code>: Nivel de estrés académico percibido (escala 1 a 10)</li>
              <li><code>sleep_hours_per_day</code>: Horas de sueño diario</li>
              <li style="color:var(--green); font-weight:700;">Variable Objetivo Y (<code>at_risk</code>): 1 = Deserción (32.2%), 0 = Continúa (67.8%)</li>
            </ul>
          </div>
        </div>
      </div>

      <div class="speaker-notes">
        🎙️ <b>Guion para William (Diapositiva 3):</b> "Trabajamos con 6,000 registros estudiantiles sin valores nulos y 22 variables predictoras divididas en 6 dimensiones: desde la nota de la primera evaluación y tareas no entregadas en el aula virtual LMS, hasta el estado de pago de pensiones y el nivel de estrés académico."
      </div>
    </section>

    <!-- ================================================================= -->
    <!-- DIAPOSITIVA 4: PREPROCESAMIENTO Y FIGURA 1                        -->
    <!-- ================================================================= -->
    <section class="slide">
      <div class="slide-header">
        <div>
          <div class="section-tag">II. Materiales y Métodos — Preprocesamiento</div>
          <div class="slide-title">Partición Estratificada (70/15/15), Escalado Z-Score y Correlaciones</div>
        </div>
        <span class="slide-num-tag">Diapositiva 04 / 10</span>
      </div>

      <div class="slide-body">
        <div class="img-frame">
          <img src="{fig1_uri}" alt="Figura 1" />
        </div>
        <div class="grid-2">
          <div class="box-card" style="padding:12px 16px;">
            <h3>Partición Estratificada sin Fuga de Datos (Data Leakage)</h3>
            <p>Se dividió la muestra en <b>Entrenamiento (70%, n=4,200)</b>, <b>Validación (15%, n=900)</b> y <b>Prueba ciega (15%, n=900)</b>. La normalización <b>Z-Score</b> (<code>z = (x - &mu;<sub>train</sub>) / &sigma;<sub>train</sub></code>) se calculó exclusivamente con la media y desviación estándar del conjunto de entrenamiento.</p>
          </div>
          <div class="box-card" style="padding:12px 16px;">
            <h3>Interpretación de Correlaciones de Pearson (Figura 1b)</h3>
            <p>Factores protectores (azul): <b>Promedio GPA (-0.561)</b>, <b>1ra Evaluación (-0.533)</b> y <b>Asistencia (-0.254)</b>. Factores de riesgo (rojo): <b>Tareas No Entregadas (+0.291)</b>, <b>Días de Inactividad LMS (+0.209)</b> y <b>Privación Socioeconómica (+0.185)</b>.</p>
          </div>
        </div>
      </div>

      <div class="speaker-notes">
        🎙️ <b>Guion para William (Diapositiva 4):</b> "En la Figura 1 vemos nuestra partición estratificada 70/15/15 manteniendo el 32.2% de casos positivos en cada bloque. Aplicamos estandarización Z-Score ajustada solo en entrenamiento para evitar fuga de datos hacia validación y prueba."
      </div>
    </section>

    <!-- ================================================================= -->
    <!-- DIAPOSITIVA 5: LAS 4 ARQUITECTURAS DE REDES NEURONALES            -->
    <!-- ================================================================= -->
    <section class="slide">
      <div class="slide-header">
        <div>
          <div class="section-tag">II. Materiales y Métodos — Diseño Experimental</div>
          <div class="slide-title">Estudio Ablativo: 4 Arquitecturas de Redes Neuronales en Keras</div>
        </div>
        <span class="slide-num-tag">Diapositiva 05 / 10</span>
      </div>

      <div class="slide-body">
        <div class="grid-4">
          <div class="box-card" style="border-top:4px solid var(--red);">
            <h3>Modelo 1: Baseline</h3>
            <p style="font-family:monospace; color:#fca5a5; margin-bottom:8px;">128 &rarr; 64 &rarr; 32 &rarr; 1</p>
            <ul>
              <li>Activación oculta: <code>ReLU</code></li>
              <li>Salida: <code>Sigmoid (1)</code></li>
              <li><b>Sin regularización</b></li>
              <li>Objetivo: Demostrar cómo ocurre el <i>Overfitting</i> en una red de alta capacidad.</li>
            </ul>
          </div>

          <div class="box-card" style="border-top:4px solid var(--amber);">
            <h3>Modelo 2: Regularización L2</h3>
            <p style="font-family:monospace; color:#fcd34d; margin-bottom:8px;">128 &rarr; 64 &rarr; 32 &rarr; 1</p>
            <ul>
              <li>Añade penalización <b>Weight Decay</b>: <code>regularizers.l2(0.003)</code></li>
              <li>Suma <code>&lambda; &Sigma; w<sub>i</sub><sup>2</sup></code> a la función de costo <i>Binary Crossentropy</i>.</li>
              <li>Evita pesos sinápticos extremos.</li>
            </ul>
          </div>

          <div class="box-card" style="border-top:4px solid #a855f7;">
            <h3>Modelo 3: Dropout</h3>
            <p style="font-family:monospace; color:#d8b4fe; margin-bottom:8px;">128 &rarr; 64 &rarr; 32 &rarr; 1</p>
            <ul>
              <li>Intercala capas <code>Dropout(0.35)</code> y <code>Dropout(0.20)</code>.</li>
              <li>Apaga neuronas al azar en cada <i>mini-batch</i>.</li>
              <li>Impide la co-adaptación entre características.</li>
            </ul>
          </div>

          <div class="box-card" style="border-top:4px solid var(--green); background:rgba(16,185,129,0.08);">
            <h3 style="color:var(--green);">Modelo 4: Propuesta (Ganador)</h3>
            <p style="font-family:monospace; color:#6ee7b7; margin-bottom:8px;">64 &rarr; 32 &rarr; 16 &rarr; 1</p>
            <ul>
              <li><b>Topología Piramidal</b> compresiva.</li>
              <li>Integra <code>BatchNormalization</code> + <code>L2(0.002)</code> + <code>Dropout(0.35&rarr;0.25&rarr;0.15)</code>.</li>
              <li>Callbacks: <code>EarlyStopping</code> (paciencia=10) + <code>ReduceLROnPlateau</code>.</li>
            </ul>
          </div>
        </div>
      </div>

      <div class="speaker-notes">
        🎙️ <b>Guion para William (Diapositiva 5):</b> "Para darle rigor científico al artículo, no entrenamos una sola red, sino que comparamos 4 arquitecturas en Keras: el Modelo 1 sin regularizar, el Modelo 2 con penalización L2, el Modelo 3 con Dropout y nuestra Arquitectura Propuesta M4 con estructura piramidal 64-32-16, BatchNormalization, L2, Dropout y EarlyStopping."
      </div>
    </section>

    <!-- ================================================================= -->
    <!-- DIAPOSITIVA 6: CURVAS DE APRENDIZAJE Y MITIGACIÓN DE OVERFITTING  -->
    <!-- ================================================================= -->
    <section class="slide">
      <div class="slide-header">
        <div>
          <div class="section-tag">III. Resultados y Discusión — Dinámica de Convergencia</div>
          <div class="slide-title">Demostración Experimental: Mitigación del Sobreajuste (Overfitting)</div>
        </div>
        <span class="slide-num-tag">Diapositiva 06 / 10</span>
      </div>

      <div class="slide-body">
        <div class="img-frame">
          <img src="{fig2_uri}" alt="Figura 2" />
        </div>
        <div class="grid-2">
          <div class="box-card" style="padding:12px 16px; border-left:4px solid var(--red);">
            <h3>¿Qué pasó con el Modelo 1 (Línea Roja)?</h3>
            <p>Su error de entrenamiento cayó casi a cero (<b>0.0003</b>), pero su pérdida de validación se disparó hasta <b>1.9622</b> (brecha de <b>+1.9619</b>). Es el ejemplo clásico de una red neuronal que memoriza pero no generaliza.</p>
          </div>
          <div class="box-card" style="padding:12px 16px; border-left:4px solid var(--green);">
            <h3>¿Cómo lo resolvió nuestra Propuesta M4 (Línea Verde)?</h3>
            <p>Las curvas de entrenamiento y validación se mantienen pegadas (brecha de solo <b>+0.0165</b>). <code>EarlyStopping</code> detuvo el entrenamiento en la época 39 y restauró los pesos óptimos de la <b>época 29</b> (Loss = <b>0.3925</b>).</p>
          </div>
        </div>
      </div>

      <div class="speaker-notes">
        🎙️ <b>Guion para William (Diapositiva 6):</b> "Esta es la gráfica central de nuestro artículo. La línea roja muestra cómo el Modelo 1 sin regularizar sufre un overfitting severo, subiendo su pérdida a casi 2.0. En cambio, la línea verde de nuestro Modelo 4 mantiene las curvas de entrenamiento y validación acopladas, y EarlyStopping restaura automáticamente los mejores pesos en la época 29."
      </div>
    </section>

    <!-- ================================================================= -->
    <!-- DIAPOSITIVA 7: TABLA COMPARATIVA DE MÉTRICAS EN TEST              -->
    <!-- ================================================================= -->
    <section class="slide">
      <div class="slide-header">
        <div>
          <div class="section-tag">III. Resultados y Discusión — Evaluación Cuantitativa</div>
          <div class="slide-title">Tabla I: Comparativa en el Conjunto de Prueba Independiente (n = 900)</div>
        </div>
        <span class="slide-num-tag">Diapositiva 07 / 10</span>
      </div>

      <div class="slide-body">
        <table class="slide-table">
          <thead>
            <tr>
              <th style="text-align:left;">Arquitectura de Red Neuronal</th>
              <th>Épocas (Óptima)</th>
              <th>Train Loss</th>
              <th>Val Loss</th>
              <th>Test Loss</th>
              <th>Brecha Overfit</th>
              <th>Accuracy (%)</th>
              <th>Precisión (%)</th>
              <th>Recall (%)</th>
              <th>F1-Score (%)</th>
              <th>ROC-AUC</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td style="text-align:left;">M1: Baseline (Sin Regularizar)</td>
              <td>60 (1)</td>
              <td>0.0003</td>
              <td>1.9622</td>
              <td style="color:#f87171; font-weight:700;">2.0283</td>
              <td style="color:#f87171;">+1.9619</td>
              <td>78.49%</td>
              <td>67.90%</td>
              <td>63.23%</td>
              <td>65.48%</td>
              <td>0.8226</td>
            </tr>
            <tr>
              <td style="text-align:left;">M2: Regularización L2 (&lambda; = 0.003)</td>
              <td>60 (14)</td>
              <td>0.3503</td>
              <td>0.4411</td>
              <td>0.5160</td>
              <td>+0.0786</td>
              <td>79.82%</td>
              <td>74.44%</td>
              <td>57.04%</td>
              <td>64.59%</td>
              <td>0.8522</td>
            </tr>
            <tr>
              <td style="text-align:left;">M3: Regularización Dropout (p = 0.35)</td>
              <td>60 (4)</td>
              <td>0.2327</td>
              <td>0.4121</td>
              <td>0.5010</td>
              <td>+0.1108</td>
              <td>78.82%</td>
              <td>72.52%</td>
              <td>55.33%</td>
              <td>62.77%</td>
              <td>0.8426</td>
            </tr>
            <tr class="winner-row">
              <td style="text-align:left;">🏆 M4: Propuesta (BN + L2 + Dropout + EarlyStopping)</td>
              <td>39 (29)</td>
              <td>0.3719</td>
              <td>0.3925</td>
              <td>0.4408 (-78.3%)</td>
              <td>+0.0165</td>
              <td>81.26%</td>
              <td>75.00%</td>
              <td>62.89%</td>
              <td>68.41%</td>
              <td>0.8752</td>
            </tr>
          </tbody>
        </table>

        <div class="grid-3">
          <div class="box-card">
            <h3>Menor Error Logarítmico</h3>
            <p>El Modelo 4 redujo el <b>Test Loss</b> de <code>2.0283</code> a <code>0.4408</code> (una reducción del <b>78.3%</b>), lo que garantiza probabilidades bien calibradas.</p>
          </div>
          <div class="box-card">
            <h3>Mayor Exactitud y F1-Score</h3>
            <p>Alcanza la mayor exactitud general (<b>81.26%</b>), la mayor precisión (<b>75.00%</b>) y el mejor equilibrio armónico F1-Score (<b>68.41%</b>).</p>
          </div>
          <div class="box-card">
            <h3>Máximo Poder Discriminativo</h3>
            <p>Logra un <b>ROC-AUC de 0.8752</b>, superando por más de 5 puntos porcentuales al modelo sin regularizar (0.8226).</p>
          </div>
        </div>
      </div>

      <div class="speaker-notes">
        🎙️ <b>Guion para William (Diapositiva 7):</b> "En la Tabla I vemos los resultados sobre los 900 alumnos del conjunto de prueba. Nuestra arquitectura propuesta M4 gana en todas las métricas: reduce la pérdida de prueba en un 78.3%, alcanza 81.26% de exactitud, 68.41% de F1-Score y un ROC-AUC de 0.8752."
      </div>
    </section>

    <!-- ================================================================= -->
    <!-- DIAPOSITIVA 8: MATRIZ DE CONFUSIÓN Y CURVA ROC                    -->
    <!-- ================================================================= -->
    <section class="slide">
      <div class="slide-header">
        <div>
          <div class="section-tag">III. Resultados y Discusión — Desempeño Clasificatorio</div>
          <div class="slide-title">Matriz de Confusión y Curvas ROC-AUC en Conjunto de Prueba (n = 900)</div>
        </div>
        <span class="slide-num-tag">Diapositiva 08 / 10</span>
      </div>

      <div class="slide-body">
        <div class="img-frame">
          <img src="{fig3_uri}" alt="Figura 3" />
        </div>
        <div class="grid-2">
          <div class="box-card" style="padding:12px 16px;">
            <h3>Análisis de la Matriz de Confusión (Figura 3a)</h3>
            <p>De los <b>611 estudiantes sin riesgo</b>, la red clasifica correctamente a <b>550 (90.0% de Especificidad)</b>, evitando falsas alarmas. De los <b>291 estudiantes en riesgo real de deserción</b>, detecta tempranamente a <b>183 alumnos (62.9% de Sensibilidad)</b>.</p>
          </div>
          <div class="box-card" style="padding:12px 16px;">
            <h3>Superioridad en la Curva ROC (Figura 3b)</h3>
            <p>La curva verde correspondiente al <b>Modelo 4 Propuesto (AUC = 0.8752)</b> domina por encima de las curvas de Dropout (0.8426), L2 (0.8522) y Baseline (0.8226) en todos los umbrales de decisión.</p>
          </div>
        </div>
      </div>

      <div class="speaker-notes">
        🎙️ <b>Guion para William (Diapositiva 8):</b> "En la Matriz de Confusión sobre los 900 estudiantes de prueba, nuestra red logra un 90% de especificidad identificando a 550 alumnos estables y detecta a 183 alumnos en riesgo real de abandono, con una curva ROC superior en todo el espectro."
      </div>
    </section>

    <!-- ================================================================= -->
    <!-- DIAPOSITIVA 9: IMPORTANCIA DE VARIABLES (EXPLICABILIDAD)          -->
    <!-- ================================================================= -->
    <section class="slide">
      <div class="slide-header">
        <div>
          <div class="section-tag">III. Resultados y Discusión — Explicabilidad de la IA</div>
          <div class="slide-title">¿Qué Factores Determinan la Predicción de nuestra Red Neuronal?</div>
        </div>
        <span class="slide-num-tag">Diapositiva 09 / 10</span>
      </div>

      <div class="slide-body">
        <div class="img-frame">
          <img src="{fig4_uri}" alt="Figura 4" />
        </div>
        <div class="box-card" style="padding:14px 18px;">
          <h3>Hallazgo Clave para la Gestión Universitaria (Importancia por Permutación)</h3>
          <p>Al permutar aleatoriamente cada variable en el conjunto de prueba, comprobamos que la Red Neuronal toma decisiones combinando: <b>1° Promedio Ponderado GPA (+9.87%)</b>, <b>2° Tareas No Entregadas en Plataforma Virtual (+1.22%)</b>, <b>3° Índice de Privación Socioeconómica (+1.11%)</b>, <b>4° Interacción en Plataforma Virtual LMS (+1.00%)</b>, <b>5° Primera Generación Universitaria y Pago de Pensiones (+0.89%)</b> y <b>Nota de la 1ra Evaluación Parcial (+0.67%)</b>.</p>
        </div>
      </div>

      <div class="speaker-notes">
        🎙️ <b>Guion para William (Diapositiva 9):</b> "Para que la red neuronal no sea una caja negra, medimos la importancia de cada variable por permutación. Descubrimos que además del promedio GPA, las tareas no entregadas en la plataforma virtual LMS, el nivel socioeconómico y el pago de pensiones son los detonantes principales que la red utiliza."
      </div>
    </section>

    <!-- ================================================================= -->
    <!-- DIAPOSITIVA 10: CONCLUSIONES Y DEMOSTRACIÓN EN VIVO               -->
    <!-- ================================================================= -->
    <section class="slide">
      <div class="slide-header">
        <div>
          <div class="section-tag">IV. Conclusiones y Producto Tecnológico</div>
          <div class="slide-title">Conclusiones Científicas y Demostración del Sistema Web (SIA-UPeU)</div>
        </div>
        <span class="slide-num-tag">Diapositiva 10 / 10</span>
      </div>

      <div class="slide-body">
        <div class="grid-3">
          <div class="box-card" style="border-top:4px solid var(--blue);">
            <h3>Conclusión 1: Mitigación del Overfitting</h3>
            <p>Se demostró que una red neuronal profunda sin regularizar colapsa por sobreajuste (<code>Test Loss = 2.0283</code>). La arquitectura piramidal propuesta (<code>64&rarr;32&rarr;16&rarr;1</code>) con <b>BatchNormalization, L2, Dropout y EarlyStopping</b> redujo la pérdida en un <b>78.3%</b> (<code>0.4408</code>).</p>
          </div>

          <div class="box-card" style="border-top:4px solid var(--green);">
            <h3>Conclusión 2: Alta Precisión Temprana</h3>
            <p>El modelo propuesto alcanzó <b>81.26% de Exactitud</b>, <b>75.00% de Precisión</b> y un <b>ROC-AUC de 0.8752</b> sobre datos no vistos, demostrando que es posible predecir la deserción desde el primer examen parcial.</p>
          </div>

          <div class="box-card" style="border-top:4px solid #a855f7;">
            <h3>Conclusión 3: Aplicación Práctica en FastAPI</h3>
            <p>El modelo entrenado (<code>mejor_modelo_desercion.keras</code>) fue integrado en el sistema web <b>SIA-UPeU</b> con FastAPI, permitiendo evaluar alumnos en tiempo real y emitir planes de acción para Tutoría Universitaria.</p>
          </div>
        </div>

        <div class="box-card" style="text-align:center; background:linear-gradient(135deg, rgba(37,99,235,0.22), rgba(16,185,129,0.22)); border:1px solid var(--blue); padding:26px;">
          <h2 style="font-size:24px; margin-bottom:10px;">🚀 Pasemos a la Demostración en Vivo del Sistema Web y Código</h2>
          <p style="font-size:15px; color:#cbd5e1; margin-bottom:18px;">
            Inferencia neuronal en tiempo real con FastAPI + Keras sobre los 4 perfiles estudiantiles de prueba
          </p>
          <div style="display:flex; justify-content:center; gap:16px; flex-wrap:wrap;">
            <a href="http://127.0.0.1:8000" target="_blank" class="ctrl-btn primary" style="padding:12px 26px; font-size:15px;">
              ⚡ Abrir Simulador Web en Vivo (http://127.0.0.1:8000)
            </a>
          </div>
        </div>
      </div>

      <div class="speaker-notes">
        🎙️ <b>Guion para William (Diapositiva 10):</b> "Como conclusión, demostramos que la combinación de BatchNormalization, L2, Dropout y EarlyStopping reduce el error de prueba en 78.3% y alcanza un ROC-AUC de 0.8752. A continuación, pasaré a mostrar el funcionamiento en vivo de nuestro sistema web desarrollado en Python con FastAPI."
      </div>
    </section>

  </main>

  <!-- Barra Inferior de Navegación -->
  <footer class="bottom-bar">
    <button class="ctrl-btn" onclick="prevSlide()">&larr; Anterior</button>
    <div class="dots-nav" id="dots-container"></div>
    <button class="ctrl-btn primary" onclick="nextSlide()">Siguiente Diapositiva &rarr;</button>
  </footer>

    <script>
    const slides = document.querySelectorAll('.slide');
    const dotsContainer = document.getElementById('dots-container');
    const progressBar = document.getElementById('progress-bar');
    let currentSlide = 0;
    let showNotes = true;

    function initDeck() {
      dotsContainer.innerHTML = '';
      slides.forEach((_, idx) => {
        const dot = document.createElement('div');
        dot.className = 'dot' + (idx === 0 ? ' active' : '');
        dot.onclick = () => goToSlide(idx);
        dotsContainer.appendChild(dot);
      });
      updateUI();
    }

    function goToSlide(idx) {
      slides[currentSlide].classList.remove('active');
      currentSlide = (idx + slides.length) % slides.length;
      slides[currentSlide].classList.add('active');
      updateUI();
    }

    function nextSlide() {
      if (currentSlide < slides.length - 1) goToSlide(currentSlide + 1);
    }

    function prevSlide() {
      if (currentSlide > 0) goToSlide(currentSlide - 1);
    }

    function updateUI() {
      const dots = document.querySelectorAll('.dot');
      dots.forEach((d, i) => d.classList.toggle('active', i === currentSlide));
      const pct = ((currentSlide + 1) / slides.length) * 100;
      progressBar.style.width = pct + '%';
    }

    function toggleNotes() {
      showNotes = !showNotes;
      document.querySelectorAll('.speaker-notes').forEach(n => {
        n.classList.toggle('hidden', !showNotes);
      });
    }

    function toggleFullScreen() {
      if (!document.fullscreenElement) {
        document.documentElement.requestFullscreen();
      } else {
        document.exitFullscreen();
      }
    }

    window.addEventListener('keydown', (e) => {
      if (e.key === 'ArrowRight' || e.key === ' ') {
        e.preventDefault();
        nextSlide();
      } else if (e.key === 'ArrowLeft') {
        e.preventDefault();
        prevSlide();
      } else if (e.key.toLowerCase() === 'n') {
        toggleNotes();
      } else if (e.key.toLowerCase() === 'f') {
        toggleFullScreen();
      }
    });

    window.addEventListener('DOMContentLoaded', initDeck);
  </script>
</body>
</html>
"""
    html = (
        html.replace("{fig1_uri}", fig1_uri)
        .replace("{fig2_uri}", fig2_uri)
        .replace("{fig3_uri}", fig3_uri)
        .replace("{fig4_uri}", fig4_uri)
        .replace("{{", "{")
        .replace("}}", "}")
    )

    out_path = os.path.join(pres_dir, "Presentacion_Exposicion_William_Ortiz.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print("Presentación HTML interactiva generada exitosamente en:", out_path)


if __name__ == "__main__":
    build_html_presentation()
