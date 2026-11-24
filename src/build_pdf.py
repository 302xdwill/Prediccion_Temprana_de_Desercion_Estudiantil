"""
Compilador de Manuscrito Científico IMRyD (Markdown + HTML IEEE + PDF Académico)
Proyecto: Predicción Temprana de la Deserción Estudiantil Universitaria mediante Redes Neuronales Profundas Regularizadas
Autor: William David Ortiz Mamani (UPeU - Ingeniería de Sistemas)
"""

import os
import json
import base64
import subprocess


def img_to_data_uri(path: str) -> str:
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def build_manuscript_and_pdf():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    proc_dir = os.path.join(base_dir, "data", "processed")
    tables_dir = os.path.join(base_dir, "results", "tables")
    figures_dir = os.path.join(base_dir, "results", "figures")
    manuscript_dir = os.path.join(base_dir, "manuscript")
    pdf_dir = os.path.join(manuscript_dir, "entregas_pdf")
    os.makedirs(pdf_dir, exist_ok=True)

    with open(os.path.join(proc_dir, "metadata.json"), "r", encoding="utf-8") as f:
        meta = json.load(f)

    with open(os.path.join(tables_dir, "metricas_detalladas.json"), "r", encoding="utf-8") as f:
        res = json.load(f)

    m1 = res["M1_Baseline"]
    m2 = res["M2_L2"]
    m3 = res["M3_Dropout"]
    m4 = res["M4_Propuesto"]

    m1_t = m1["test_metrics"]
    m2_t = m2["test_metrics"]
    m3_t = m3["test_metrics"]
    m4_t = m4["test_metrics"]

    # =========================================================================
    # 1. GENERACIÓN DEL MANUSCRITO COMPLETO EN MARKDOWN (borrador_articulo.md)
    # =========================================================================
    md_content = f"""# Predicción Temprana de la Deserción Estudiantil Universitaria mediante Redes Neuronales Profundas con Regularización L2, Dropout y Batch Normalization
## *Early Prediction of University Student Dropout Using Deep Neural Networks with L2 Regularization, Dropout, and Batch Normalization*

**William David Ortiz Mamani**  
Escuela Profesional de Ingeniería de Sistemas, Facultad de Ingeniería y Arquitectura  
Universidad Peruana Unión (UPeU), Campus Juliaca, Perú  
Correo institucional: `william.ortiz@upeu.edu.pe` | GitHub: `302xdwill`  
**Curso:** Inteligencia Artificial — **Docente:** Mg. Angel Rosendo Condori Coaquira  

---

## Resumen
La deserción estudiantil universitaria constituye uno de los desafíos socioeducativos y de gestión académica más críticos en las instituciones de educación superior, generando pérdidas económicas e interrumpiendo el desarrollo profesional de miles de jóvenes. Los métodos tradicionales de identificación de riesgo suelen basarse en umbrales lineales tardíos que no capturan interacciones no lineales complejas entre factores académicos, financieros, conductuales y psicológicos. No obstante, cuando se emplean Redes Neuronales Artificiales (RNA) de alta capacidad sobre datos tabulares educativos, es frecuente la aparición de sobreajuste (*overfitting*), degradando su capacidad de generalización ante nuevas cohortes estudiantiles. En este artículo se diseña, entrena y evalúa una arquitectura de Red Neuronal Multicapa (MLP) profunda y regularizada para la predicción temprana de la deserción estudiantil universitaria utilizando un conjunto de datos multimodal de $N = {meta['total_samples']:,}$ estudiantes y $d = {meta['n_features']}$ variables predictoras. Se realizó un estudio ablativo comparando cuatro arquitecturas en Keras: un modelo base sin regularización (M1), una red con penalización de pesos L2 (M2), una red con desconexión estocástica *Dropout* (M3) y una arquitectura piramidal propuesta (`64 -> 32 -> 16 -> 1`) que integra sinérgicamente *Batch Normalization*, regularización L2, *Dropout* progresivo y parada temprana (*EarlyStopping*) (M4). Los resultados en el conjunto de prueba independiente ($n = {meta['test_samples']}$) demuestran que el modelo base (M1) sufre un severo sobreajuste (pérdida de prueba de `{m1['test_loss']:.4f}` y brecha de generalización de `{m1['overfit_gap']:+.4f}`), mientras que la arquitectura propuesta (M4) reduce la pérdida de prueba en un `{((m1['test_loss'] - m4['test_loss']) / m1['test_loss'] * 100):.1f}%` (hasta `{m4['test_loss']:.4f}`), alcanzando una exactitud (*Accuracy*) de `{m4_t['accuracy']*100:.2f}%`, un *F1-Score* de `{m4_t['f1_score']*100:.2f}%` y un área bajo la curva ROC (AUC) de `{m4_t['roc_auc']:.4f}`. Se concluye que la combinación de normalización por lotes y regularización estructural permite desplegar sistemas de alerta temprana altamente confiables para los programas de Tutoría y Bienestar Universitario.

**Palabras Clave:** Redes Neuronales Artificiales, Deserción Estudiantil, Regularización L2, Dropout, Batch Normalization, Clasificación Binaria, Educación Superior.

---

## Abstract
University student dropout constitutes one of the most critical socio-educational and academic management challenges in higher education institutions. Traditional risk identification approaches rely on late linear thresholds that fail to capture complex non-linear interactions among academic, financial, behavioral, and psychological dimensions. However, deploying high-capacity Artificial Neural Networks (ANNs) on tabular educational datasets frequently induces severe overfitting, degrading generalization on unseen student cohorts. This paper designs, trains, and evaluates a regularized Deep Multi-Layer Perceptron (MLP) architecture for early university dropout prediction using a multimodal dataset of $N = {meta['total_samples']:,}$ students across $d = {meta['n_features']}$ predictive features. An ablation study was conducted in Keras comparing four architectures: an unregularized baseline network (M1), an L2 weight-decay regularized network (M2), a stochastic Dropout network (M3), and a proposed pyramidal architecture (`64 -> 32 -> 16 -> 1`) synergistically integrating Batch Normalization, L2 regularization, progressive Dropout, and EarlyStopping (M4). Evaluation on the held-out test set ($n = {meta['test_samples']}$) demonstrates that while the baseline model (M1) suffers severe overfitting (test loss of `{m1['test_loss']:.4f}` and generalization gap of `{m1['overfit_gap']:+.4f}`), the proposed architecture (M4) reduces test loss by `{((m1['test_loss'] - m4['test_loss']) / m1['test_loss'] * 100):.1f}%` (down to `{m4['test_loss']:.4f}`), achieving a test accuracy of `{m4_t['accuracy']*100:.2f}%`, an F1-Score of `{m4_t['f1_score']*100:.2f}%`, and a ROC-AUC of `{m4_t['roc_auc']:.4f}`. These findings confirm that combining batch normalization with structural regularization enables robust early-warning systems for university tutoring and retention programs.

**Keywords:** Artificial Neural Networks, Student Dropout, L2 Regularization, Dropout, Batch Normalization, Binary Classification, Higher Education.

---

## I. Introducción

La retención estudiantil en la educación superior representa un indicador fundamental de calidad institucional y desarrollo socioeconómico. A nivel global y latinoamericano, entre el 20% y el 35% de los estudiantes universitarios abandonan sus estudios antes de culminar su formación profesional, fenómeno que se acentúa durante los primeros semestres académicos debido a una combinación multifactorial de rendimiento académico deficiente, dificultades económicas en el pago de pensiones, baja interacción en los entornos virtuales de aprendizaje (*Learning Management Systems* - LMS) y altos niveles de estrés o agotamiento psicológico (*burnout*) [1], [2]. En instituciones universitarias como la Universidad Peruana Unión (UPeU), contar con herramientas predictivas capaces de identificar proactivamente a los alumnos en riesgo antes de que se materialice el abandono resulta vital para focalizar las intervenciones de Tutoría Universitaria, becas socioeconómicas y acompañamiento psicopedagógico.

Desde la perspectiva de la Inteligencia Artificial y la Minería de Datos Educativos (*Educational Data Mining*), las Redes Neuronales Artificiales (RNA) y el Aprendizaje Profundo (*Deep Learning*) ofrecen la capacidad de modelar relaciones altamente no lineales entre decenas de variables heterogéneas sin requerir el diseño manual de interacciones polinómicas [3]. Sin embargo, un obstáculo recurrente al entrenar perceptrones multicapa (MLP) profundos sobre datos tabulares es el **sobreajuste (*overfitting*)**: la red neuronal posee suficiente capacidad paramétrica para memorizar las fluctuaciones aleatorias y el ruido estadístico del conjunto de entrenamiento, alcanzando pérdidas cercanas a cero en entrenamiento pero sufriendo una rápida degradación en la función de pérdida sobre datos de validación y prueba [4].

Para contrarrestar este fenómeno, la literatura especializada propone mecanismos de control de complejidad como la penalización de norma L2 (*Weight Decay*), la desactivación aleatoria de unidades ocultas (*Dropout*) [5], la normalización por lotes (*Batch Normalization*) [6] y la detención anticipada del entrenamiento (*Early Stopping*). En este contexto, el objetivo del presente artículo es **diseñar, implementar y validar experimentalmente una arquitectura de Red Neuronal Profunda regularizada para la predicción temprana de la deserción estudiantil universitaria**, contrastando su desempeño cuantitativo y estabilidad de convergencia frente a arquitecturas sin regularizar y con regularización simple.

---

## II. Materiales y Métodos (Metodología)

### 2.1 Descripción del Conjunto de Datos (Dataset)
El estudio experimental utilizó el repositorio multimodal de retención y riesgo estudiantil (*Student Early Stress & Dropout Detection Benchmark*), seleccionando una cohorte estratificada de **$N = {meta['total_samples']:,}$ registros de estudiantes de educación superior** con un 0.0% de valores faltantes. La variable objetivo binaria $Y \\in \\{{0, 1\\}}$ (`at_risk`) codifica con `1` al estudiante en situación de deserción o alto riesgo de abandono académico (`{meta['positive_rate_total']*100:.2f}%` de los casos, $n = {int(round(meta['total_samples']*meta['positive_rate_total']))}$) y con `0` al estudiante que mantiene una trayectoria de retención regular (`{(1-meta['positive_rate_total'])*100:.2f}%`, $n = {meta['total_samples'] - int(round(meta['total_samples']*meta['positive_rate_total']))}$).

Cada estudiante está caracterizado por un vector de entrada $\\mathbf{{x}} \\in \\mathbb{{R}}^{{22}}$ estructurado en seis dimensiones complementarias:
1. **Perfil Demográfico y Socioeconómico (5 variables):** Edad (`age`), género codificado (`gender_encoded`), condición de estudiante internacional (`international_student`), primera generación universitaria en la familia (`first_generation`) e índice de privación socioeconómica (`socioeconomic_deprivation_index`, escala 1–10).
2. **Estabilidad Financiera (2 variables):** Condición de becario (`scholarship_holder`) y cumplimiento puntual en el pago de pensiones (`tuition_up_to_date`).
3. **Historial Académico Previo (2 variables):** Nota de examen de admisión (`admission_grade`) y calificación de estudios secundarios previos (`prior_qualification_grade`).
4. **Rendimiento Académico Universitario (6 variables):** Promedio ponderado acumulado (`gpa_score_avg`, escala 0–100), tendencia longitudinal del promedio (`gpa_score_trend`), tasa de asignaturas aprobadas (`pass_rate`), calificación en la primera evaluación parcial (`first_assessment_score`), horas de estudio semanales (`study_hours_per_week`) y carga académica inscrita (`study_load`).
5. **Compromiso Conductual en Plataforma Virtual LMS/VLE (5 variables):** Porcentaje de asistencia a clases (`attendance_rate`), puntaje de interacción digital en el aula virtual (`lms_engagement_score`), proporción de días activos en la plataforma (`lms_active_days_ratio`), brecha máxima de días consecutivos sin conexión (`longest_inactivity_gap_days`) y número de tareas no entregadas (`missed_submissions_count`).
6. **Factores Psicológicos y Fisiológicos (2 variables):** Escala normalizada de estrés académico percibido (`stress_level_scale`, escala 1–10) y promedio de horas de sueño por día (`sleep_hours_per_day`).

### 2.2 Preprocesamiento y Partición Estratificada
Con el fin de garantizar una evaluación rigurosa y libre de fuga de datos (*data leakage*), el conjunto total ($N = 6,000$) fue dividido mediante muestreo aleatorio estratificado en tres subconjuntos disjuntos:
* **Conjunto de Entrenamiento (70%, $n = 4,200$):** Empleado para el ajuste de los pesos sinápticos mediante retropropagación del error (*Backpropagation*).
* **Conjunto de Validación (15%, $n = 900$):** Utilizado para monitorear la generalización al final de cada época y activar los criterios de parada temprana (*EarlyStopping*).
* **Conjunto de Prueba (15%, $n = 900$):** Reservado exclusivamente para la evaluación final ciega de las arquitecturas.

Debido a que las variables presentan escalas heterogéneas (por ejemplo, `gpa_score_avg` en $[0, 100]$ frente a `pass_rate` en $[0, 1]$), se aplicó la estandarización Z-score sobre cada característica $j \\in \\{{1, \\dots, 22\\}}$, calculando la media $\\mu_j$ y la desviación estándar $\\sigma_j$ únicamente sobre el conjunto de entrenamiento:

$$z_{{i,j}} = \\frac{{x_{{i,j}} - \\mu_{{j,\\text{{train}}}}}}{{\\sigma_{{j,\\text{{train}}}}}}$$

![Figura 1: Partición Estratificada y Correlaciones de Pearson](../results/figures/fig1_exploracion_dataset.png)

### 2.3 Arquitecturas de Redes Neuronales Evaluadas (Estudio Ablativo)
Para demostrar empíricamente el efecto de las técnicas de regularización sobre el sobreajuste, se diseñaron y compararon cuatro arquitecturas secuenciales en **Keras / TensorFlow**:

1. **Modelo 1 — Baseline Sin Regularizar (`M1_Baseline`):** Red densa de tres capas ocultas con topología `128 -> 64 -> 32` neuronas y activación $\\text{{ReLU}}(z) = \\max(0, z)$, seguida de una neurona de salida con activación sigmoide $\\sigma(z) = (1 + e^{{-z}})^{{-1}}$. No incorpora ninguna restricción sobre los pesos ni capas de desconexión.
2. **Modelo 2 — Regularización L2 (`M2_L2`):** Mantiene la topología `128 -> 64 -> 32 -> 1`, pero añade una penalización cuadrática (*Weight Decay*, $\\lambda = 0.003$) en cada capa oculta, modificando la función de costo total según:
   $$\\mathcal{{L}}_{{\\text{{total}}}} = \\mathcal{{L}}_{{\\text{{BCE}}}} + \\lambda \\sum_{{l}} \\|\\mathbf{{W}}^{{(l)}}\\|_2^2$$
3. **Modelo 3 — Regularización por Dropout (`M3_Dropout`):** Mantiene la estructura `128 -> 64 -> 32 -> 1` intercalando capas `Dropout(0.35)` tras las dos primeras capas ocultas y `Dropout(0.20)` tras la tercera capa oculta, desactivando aleatoriamente neuronas en cada *mini-batch* para impedir la co-adaptación sináptica.
4. **Modelo 4 — Arquitectura Piramidal Propuesta (`M4_Propuesto`):** Implementa una estructura de compresión progresiva de características (`64 -> 32 -> 16 -> 1`) que combina:
   * Primera capa densa de `64` neuronas (`ReLU`, `L2 = 0.002`) + `BatchNormalization()` + `Dropout(0.35)`.
   * Segunda capa densa de `32` neuronas (`ReLU`, `L2 = 0.002`) + `BatchNormalization()` + `Dropout(0.25)`.
   * Tercera capa densa de `16` neuronas (`ReLU`, `L2 = 0.001`) + `Dropout(0.15)`.
   * Capa de salida densa de `1` neurona con función de activación `Sigmoid`.
   * Callbacks dinámicos: `EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)` y `ReduceLROnPlateau(factor=0.5, patience=5)`.

Todos los modelos fueron entrenados optimizando la entropía cruzada binaria (*Binary Crossentropy*) con el optimizador adaptativo **Adam**, tamaño de lote `batch_size = 64` y un horizonte máximo de `60` épocas.

---

## III. Resultados y Discusión

### 3.1 Análisis de Convergencia y Mitigación del Sobreajuste (*Overfitting*)
La **Figura 2** ilustra el comportamiento dinámico de la función de pérdida durante el entrenamiento de las cuatro arquitecturas. En el **Modelo 1 (Baseline)**, la pérdida de entrenamiento desciende monótonamente hasta `{m1['train_loss']:.4f}`, mientras que la pérdida de validación alcanza su mínimo en la época `{m1['best_val_epoch']}` y posteriormente se degrada de forma sostenida hasta `{m1['val_loss']:.4f}` en la época 60, generando una severa brecha de sobreajuste (*overfitting gap*) de `{m1['overfit_gap']:+.4f}`.

![Figura 2: Curvas de Aprendizaje y Mitigación del Overfitting](../results/figures/fig2_curvas_aprendizaje_overfitting.png)

Al introducir **Regularización L2 (M2)** y **Dropout (M3)** por separado, se observa una contención parcial del sobreajuste, reduciendo la pérdida de prueba a `{m2['test_loss']:.4f}` y `{m3['test_loss']:.4f}`, respectivamente. Sin embargo, el **Modelo 4 (Arquitectura Propuesta)** logra el comportamiento más estable y eficiente: la normalización por lotes (*Batch Normalization*) acelera la convergencia inicial, mientras que la combinación de `L2 + Dropout` mantiene las curvas de entrenamiento y validación estrechamente acopladas (brecha de apenas `{m4['overfit_gap']:+.4f}`). El mecanismo `EarlyStopping` detuvo el entrenamiento y restauró automáticamente los pesos óptimos de la **época {m4['best_val_epoch']}**, alcanzando una pérdida mínima de validación de `{m4['min_val_loss']:.4f}` y la menor pérdida en el conjunto de prueba (`{m4['test_loss']:.4f}`).

### 3.2 Desempeño Cuantitativo en el Conjunto de Prueba
La **Tabla I** resume las métricas cuantitativas obtenidas por las cuatro redes neuronales sobre los $n = 900$ estudiantes del conjunto de prueba independiente.

**Tabla I. Comparativa Experimental de las Arquitecturas de Redes Neuronales en el Conjunto de Prueba ($n = 900$)**

| Arquitectura Evaluada | Épocas (Óptima) | Train Loss | Val Loss | Test Loss | Brecha Overfit | Exactitud (Acc %) | Precisión (%) | Sensibilidad / Recall (%) | F1-Score (%) | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **M1: Baseline (Sin Regularizar)** | {m1['epochs_ran']} ({m1['best_val_epoch']}) | {m1['train_loss']:.4f} | {m1['val_loss']:.4f} | {m1['test_loss']:.4f} | {m1['overfit_gap']:+.4f} | {m1_t['accuracy']*100:.2f}% | {m1_t['precision']*100:.2f}% | {m1_t['recall']*100:.2f}% | {m1_t['f1_score']*100:.2f}% | {m1_t['roc_auc']:.4f} |
| **M2: Regularización L2 (0.003)** | {m2['epochs_ran']} ({m2['best_val_epoch']}) | {m2['train_loss']:.4f} | {m2['val_loss']:.4f} | {m2['test_loss']:.4f} | {m2['overfit_gap']:+.4f} | {m2_t['accuracy']*100:.2f}% | {m2_t['precision']*100:.2f}% | {m2_t['recall']*100:.2f}% | {m2_t['f1_score']*100:.2f}% | {m2_t['roc_auc']:.4f} |
| **M3: Regularización Dropout (0.35)** | {m3['epochs_ran']} ({m3['best_val_epoch']}) | {m3['train_loss']:.4f} | {m3['val_loss']:.4f} | {m3['test_loss']:.4f} | {m3['overfit_gap']:+.4f} | {m3_t['accuracy']*100:.2f}% | {m3_t['precision']*100:.2f}% | {m3_t['recall']*100:.2f}% | {m3_t['f1_score']*100:.2f}% | {m3_t['roc_auc']:.4f} |
| **M4: Propuesta (BN+L2+Dropout+ES)** | **{m4['epochs_ran']} ({m4['best_val_epoch']})** | **{m4['train_loss']:.4f}** | **{m4['val_loss']:.4f}** | **{m4['test_loss']:.4f}** | **{m4['overfit_gap']:+.4f}** | **{m4_t['accuracy']*100:.2f}%** | **{m4_t['precision']*100:.2f}%** | **{m4_t['recall']*100:.2f}%** | **{m4_t['f1_score']*100:.2f}%** | **{m4_t['roc_auc']:.4f}** |

Como se aprecia en la **Figura 3(a)**, la Matriz de Confusión del Modelo Propuesto (M4) clasifica correctamente a `{m4_t['tn']}` estudiantes sin riesgo (Verdaderos Negativos, especificidad de `{m4_t['specificity']*100:.2f}%`) y detecta acertadamente a `{m4_t['tp']}` estudiantes en situación real de deserción (Verdaderos Positivos), manteniendo un área bajo la curva ROC (AUC) de `{m4_t['roc_auc']:.4f}` (**Figura 3(b)**), superior a las tres arquitecturas de referencia.

![Figura 3: Matriz de Confusión y Curvas ROC-AUC](../results/figures/fig3_matriz_confusion_roc.png)

### 3.3 Interpretabilidad y Factores Determinantes de la Deserción
El análisis de importancia por permutación sobre la red neuronal propuesta (**Figura 4**) revela que el modelo fundamenta sus decisiones en una sinergia de factores académicos, conductuales y socioeconómicos. Las variables de mayor peso predictivo son el promedio ponderado acumulado (`gpa_score_avg`), la calificación en la primera evaluación parcial (`first_assessment_score`), la tasa de cursos aprobados (`pass_rate`), el conteo de tareas no entregadas en el aula virtual (`missed_submissions_count`), la tasa de asistencia (`attendance_rate`), la morosidad en el pago de pensiones (`tuition_up_to_date`) y el nivel de estrés académico (`stress_level_scale`). Este hallazgo confirma que la deserción puede predecirse desde las primeras semanas del semestre combinando la nota del primer examen parcial con las métricas de actividad en la plataforma virtual (LMS).

![Figura 4: Importancia de Variables por Permutación](../results/figures/fig4_importancia_variables.png)

---

## IV. Conclusiones

1. Se demostró experimentalmente que una Red Neuronal Multicapa profunda sin mecanismos de regularización (**M1**) sufre un severo sobreajuste al modelar datos tabulares de deserción estudiantil, incrementando su pérdida de prueba hasta `{m1['test_loss']:.4f}` debido a la memorización de patrones espurios.
2. La **Arquitectura Piramidal Propuesta (M4)** (`64 -> 32 -> 16 -> 1`), que integra `BatchNormalization`, regularización `L2`, `Dropout` escalonado y `EarlyStopping`, logró reducir la pérdida de prueba en un **`{((m1['test_loss'] - m4['test_loss']) / m1['test_loss'] * 100):.1f}%`** (alcanzando `{m4['test_loss']:.4f}`), con una exactitud de **`{m4_t['accuracy']*100:.2f}%`**, un F1-Score de **`{m4_t['f1_score']*100:.2f}%`** y un índice discriminativo **ROC-AUC de `{m4_t['roc_auc']:.4f}`**.
3. El análisis de sensibilidad por permutación evidenció que las calificaciones tempranas (primera evaluación parcial), el ausentismo o inactividad en la plataforma virtual (LMS) y el estado de pago de pensiones son los predictores más potentes del abandono universitario, permitiendo su implementación directa en sistemas de alerta temprana para áreas de Tutoría Universitaria.

---

## Referencias Bibliográficas

1. K. Kuzilek, M. Hlosta, and Z. Zdrahal, "Open University Learning Analytics dataset," *Scientific Data (Nature)*, vol. 4, no. 1, pp. 1–8, 2017, doi: 10.1038/sdata.2017.171.
2. V. Realinho, J. Machado, L. Baptista, and M. V. Martins, "Predicting Student Dropout and Academic Success," *Data (MDPI)*, vol. 7, no. 11, p. 146, 2022, doi: 10.3390/data7110146.
3. I. Goodfellow, Y. Bengio, and A. Courville, *Deep Learning*. Cambridge, MA, USA: MIT Press, 2016.
4. F. Chollet, *Deep Learning with Python*, 2nd ed. Shelter Island, NY, USA: Manning Publications, 2021.
5. N. Srivastava, G. Hinton, A. Krizhevsky, I. Sutskever, and R. Salakhutdinov, "Dropout: A simple way to prevent neural networks from overfitting," *Journal of Machine Learning Research*, vol. 15, no. 1, pp. 1929–1958, 2014.
6. S. Ioffe and C. Szegedy, "Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift," in *Proc. 32nd Int. Conf. Machine Learning (ICML)*, Lille, France, 2015, pp. 448–456.
"""

    md_path = os.path.join(manuscript_dir, "borrador_articulo.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    print("Manuscrito Markdown actualizado en:", md_path)

    # =========================================================================
    # 2. GENERACIÓN DEL ARTÍCULO CIENTÍFICO EN HTML ESTILO IEEE + EXPORTACIÓN PDF
    # =========================================================================
    fig1_uri = img_to_data_uri(os.path.join(figures_dir, "fig1_exploracion_dataset.png"))
    fig2_uri = img_to_data_uri(os.path.join(figures_dir, "fig2_curvas_aprendizaje_overfitting.png"))
    fig3_uri = img_to_data_uri(os.path.join(figures_dir, "fig3_matriz_confusion_roc.png"))
    fig4_uri = img_to_data_uri(os.path.join(figures_dir, "fig4_importancia_variables.png"))

    loss_reduction_pct = (m1["test_loss"] - m4["test_loss"]) / m1["test_loss"] * 100

    html_content = f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<title>Artículo Científico - William David Ortiz Mamani - UPeU</title>
<style>
  @page {{
    size: A4 portrait;
    margin: 18mm 16mm 18mm 16mm;
  }}
  body {{
    font-family: "Times New Roman", Times, Georgia, serif;
    font-size: 10.5pt;
    line-height: 1.42;
    color: #111111;
    margin: 0;
    padding: 0;
    text-align: justify;
  }}
  .header-banner {{
    border-bottom: 2px solid #003366;
    padding-bottom: 6px;
    margin-bottom: 14px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 24px;
    font-family: Arial, Helvetica, sans-serif;
    font-size: 8pt;
    color: #003366;
    font-weight: bold;
    text-transform: uppercase;
    letter-spacing: 0.3px;
  }}
  .paper-title {{
    font-size: 16.5pt;
    font-weight: bold;
    text-align: center;
    color: #0a1f44;
    line-height: 1.25;
    margin: 8px 0 6px 0;
  }}
  .paper-subtitle {{
    font-size: 12pt;
    font-style: italic;
    text-align: center;
    color: #333333;
    margin: 0 0 12px 0;
  }}
  .author-block {{
    text-align: center;
    margin-bottom: 14px;
    font-size: 10pt;
    line-height: 1.35;
  }}
  .author-name {{
    font-size: 11.5pt;
    font-weight: bold;
    color: #000000;
  }}
  .abstract-box {{
    background-color: #f7f9fc;
    border-left: 4px solid #003366;
    border-top: 1px solid #d0d7de;
    border-right: 1px solid #d0d7de;
    border-bottom: 1px solid #d0d7de;
    padding: 10px 14px;
    margin-bottom: 14px;
    font-size: 9.5pt;
    line-height: 1.38;
  }}
  .abstract-title {{
    font-weight: bold;
    color: #003366;
    text-transform: uppercase;
    font-size: 9.5pt;
  }}
  .keywords {{
    margin-top: 5px;
    font-size: 9.2pt;
  }}
  h2.section-title {{
    font-size: 11.5pt;
    font-weight: bold;
    color: #003366;
    text-transform: uppercase;
    border-bottom: 1px solid #003366;
    padding-bottom: 2px;
    margin-top: 14px;
    margin-bottom: 7px;
    page-break-after: avoid;
    break-after: avoid;
  }}
  h3.subsection-title {{
    font-size: 10.5pt;
    font-weight: bold;
    font-style: italic;
    color: #1a1a1a;
    margin-top: 10px;
    margin-bottom: 4px;
    page-break-after: avoid;
    break-after: avoid;
  }}
  p {{
    margin: 0 0 8px 0;
    text-indent: 14px;
  }}
  p.no-indent {{
    text-indent: 0;
  }}
  ul, ol {{
    margin: 4px 0 8px 0;
    padding-left: 22px;
  }}
  li {{
    margin-bottom: 3px;
  }}
  .equation {{
    text-align: center;
    font-style: italic;
    margin: 8px 0;
    padding: 4px 0;
    background: #fafafa;
    border-radius: 3px;
    font-size: 10.5pt;
  }}
  .figure-container {{
    text-align: center;
    margin: 12px 0;
    page-break-inside: avoid;
  }}
  .figure-container img {{
    max-width: 100%;
    height: auto;
    border: 1px solid #cccccc;
    padding: 3px;
    background: #ffffff;
  }}
  .figure-caption {{
    font-size: 9pt;
    color: #222222;
    margin-top: 5px;
    text-align: justify;
    padding: 0 8px;
  }}
  table.scientific-table {{
    width: 100%;
    border-collapse: collapse;
    margin: 10px 0 12px 0;
    font-size: 8.7pt;
    page-break-inside: avoid;
  }}
  table.scientific-table caption {{
    font-weight: bold;
    font-size: 9.3pt;
    margin-bottom: 5px;
    text-align: left;
    color: #003366;
  }}
  table.scientific-table th {{
    background-color: #003366;
    color: #ffffff;
    font-weight: bold;
    padding: 5px 6px;
    border: 1px solid #003366;
    text-align: center;
  }}
  table.scientific-table td {{
    padding: 5px 6px;
    border: 1px solid #cccccc;
    text-align: center;
  }}
  table.scientific-table td.left {{
    text-align: left;
  }}
  table.scientific-table tr.highlight-row {{
    background-color: #e8f5e9;
    font-weight: bold;
  }}
  .code-box {{
    font-family: "Consolas", "Courier New", monospace;
    font-size: 8.3pt;
    background-color: #f4f6f8;
    border: 1px solid #d0d7de;
    border-left: 3px solid #2ca02c;
    padding: 8px 10px;
    margin: 8px 0;
    white-space: pre-wrap;
    text-align: left;
    page-break-inside: avoid;
  }}
  .references {{
    font-size: 9pt;
    line-height: 1.35;
    padding-left: 18px;
  }}
  .references li {{
    margin-bottom: 4px;
  }}
</style>
</head>
<body>

<div class="header-banner">
  <span>Universidad Peruana Unión (UPeU) — Facultad de Ingeniería y Arquitectura</span>
  <span>Curso: Inteligencia Artificial — Artículo de Investigación</span>
</div>

<div class="paper-title">
  Predicción Temprana de la Deserción Estudiantil Universitaria mediante Redes Neuronales Profundas con Regularización L2, Dropout y Batch Normalization
</div>
<div class="paper-subtitle">
  Early Prediction of University Student Dropout Using Deep Neural Networks with L2 Regularization, Dropout, and Batch Normalization
</div>

<div class="author-block">
  <div class="author-name">William David Ortiz Mamani</div>
  <div>Escuela Profesional de Ingeniería de Sistemas, Universidad Peruana Unión (UPeU), Campus Juliaca, Perú</div>
  <div>Correo institucional: <code>william.ortiz@upeu.edu.pe</code> &nbsp;|&nbsp; Correo personal: <code>davidom302@gmail.com</code> &nbsp;|&nbsp; GitHub: <code>302xdwill</code></div>
  <div>Docente asesor: Mg. Angel Rosendo Condori Coaquira</div>
</div>

<div class="abstract-box">
  <div><span class="abstract-title">Resumen—</span> La deserción estudiantil universitaria constituye uno de los desafíos socioeducativos y de gestión académica más críticos en las instituciones de educación superior, generando pérdidas económicas e interrumpiendo el proyecto de vida de miles de jóvenes. Los métodos tradicionales de identificación de riesgo suelen apoyarse en reglas lineales tardías que no capturan interacciones complejas entre factores académicos, financieros, conductuales en plataformas virtuales (LMS) y psicológicos. Sin embargo, al entrenar Redes Neuronales Artificiales (RNA) profundas sobre datos tabulares educativos, es frecuente la aparición de sobreajuste (<i>overfitting</i>), degradando su capacidad de generalización ante nuevas cohortes estudiantiles. En este artículo se diseña, entrena y evalúa una arquitectura de Red Neuronal Multicapa (MLP) profunda y regularizada para la predicción temprana de la deserción estudiantil universitaria empleando un conjunto de datos multimodal de <b>N = {meta['total_samples']:,} estudiantes</b> y <b>22 variables predictoras</b>. Se ejecutó un estudio ablativo en Keras comparando cuatro modelos: una red base sin regularización (M1), una red con penalización de pesos L2 (M2), una red con desconexión estocástica <i>Dropout</i> (M3) y una arquitectura piramidal propuesta (<code>64 &rarr; 32 &rarr; 16 &rarr; 1</code>) que integra <i>Batch Normalization</i>, regularización L2, <i>Dropout</i> escalonado y parada temprana <i>EarlyStopping</i> (M4). La evaluación en el conjunto de prueba independiente (n = {meta['test_samples']}) demuestra que mientras el modelo base sufre un severo sobreajuste (pérdida de prueba de <b>{m1['test_loss']:.4f}</b> y brecha de generalización de <b>{m1['overfit_gap']:+.4f}</b>), la arquitectura propuesta (M4) reduce la pérdida de prueba en un <b>{loss_reduction_pct:.1f}%</b> (hasta <b>{m4['test_loss']:.4f}</b>), alcanzando una exactitud de <b>{m4_t['accuracy']*100:.2f}%</b>, un F1-Score de <b>{m4_t['f1_score']*100:.2f}%</b> y un área bajo la curva ROC (AUC) de <b>{m4_t['roc_auc']:.4f}</b>.</div>
  <div class="keywords"><b>Palabras Clave:</b> Redes Neuronales Artificiales, Deserción Estudiantil, Regularización L2, Dropout, Batch Normalization, Early Stopping, Clasificación Binaria.</div>
  <hr style="border:0; border-top:1px dashed #cccccc; margin:6px 0;">
  <div><span class="abstract-title">Abstract—</span> University student dropout constitutes one of the most critical socio-educational challenges in higher education institutions. Deploying high-capacity Artificial Neural Networks (ANNs) on tabular educational data frequently induces severe overfitting. This paper designs, trains, and evaluates a regularized Deep Multi-Layer Perceptron (MLP) architecture for early university dropout prediction using a multimodal dataset of <b>N = {meta['total_samples']:,} students</b> across <b>22 predictive features</b>. An ablation study in Keras compared four models: an unregularized baseline (M1), L2 weight decay (M2), Dropout (M3), and a proposed pyramidal topology (<code>64 &rarr; 32 &rarr; 16 &rarr; 1</code>) combining Batch Normalization, L2 regularization, progressive Dropout, and EarlyStopping (M4). On the held-out test set (n = {meta['test_samples']}), the proposed model M4 reduces test cross-entropy loss by <b>{loss_reduction_pct:.1f}%</b> compared to the baseline (from {m1['test_loss']:.4f} down to <b>{m4['test_loss']:.4f}</b>), achieving <b>{m4_t['accuracy']*100:.2f}% accuracy</b>, <b>{m4_t['f1_score']*100:.2f}% F1-Score</b>, and a <b>ROC-AUC of {m4_t['roc_auc']:.4f}</b>.</div>
  <div class="keywords"><b>Keywords:</b> Artificial Neural Networks, Student Dropout, L2 Regularization, Dropout, Batch Normalization, Early Stopping, Binary Classification.</div>
</div>

<h2 class="section-title">I. Introducción</h2>
<p>La retención estudiantil en la educación superior representa un indicador esencial de calidad educativa y responsabilidad social universitaria. En el contexto latinoamericano y peruano, una proporción significativa de los estudiantes que ingresan a la universidad abandona sus estudios durante los primeros ciclos debido a la interacción compleja de factores académicos (bajo rendimiento en las primeras evaluaciones parciales), dificultades financieras (retrasos en el pago de pensiones o falta de becas), desvinculación progresiva de los entornos virtuales de aprendizaje (<i>Learning Management Systems</i> - LMS, como PatmOS o Moodle) y factores psicosociales como el estrés académico y la privación del sueño [1], [2]. En universidades como la Universidad Peruana Unión (UPeU), detectar tempranamente qué estudiantes presentan alta probabilidad de deserción resulta indispensable para activar intervenciones oportunas desde las unidades de Tutoría y Bienestar Universitario.</p>

<p>Desde el campo de la Inteligencia Artificial, las Redes Neuronales Artificiales (RNA) ofrecen una capacidad superior frente a los modelos estadísticos lineales para capturar fronteras de decisión no lineales en espacios multidimensionales [3]. No obstante, cuando se entrenan Perceptrones Multicapa (MLP) de alta capacidad sobre conjuntos de datos tabulares con ruido inherente al comportamiento humano, surge el problema crítico del <b>sobreajuste (<i>overfitting</i>)</b>: la red memoriza los ejemplos particulares del conjunto de entrenamiento, logrando un error casi nulo en entrenamiento pero perdiendo su capacidad de generalizar ante datos de nuevos estudiantes [4].</p>

<p>Para resolver esta limitación, la teoría moderna del Aprendizaje Profundo emplea técnicas de regularización explícita e implícita, tales como la penalización cuadrática de pesos L2 (<i>Weight Decay</i>), la desconexión estocástica de neuronas (<i>Dropout</i>) [5], la normalización interna por lotes (<i>Batch Normalization</i>) [6] y la parada temprana (<i>Early Stopping</i>). En consecuencia, el objetivo del presente trabajo es <b>diseñar, entrenar y sustentar experimentalmente una arquitectura de Red Neuronal Profunda regularizada para la clasificación binaria temprana de la deserción estudiantil</b>, demostrando cuantitativamente cómo la combinación sinérgica de regularización mitiga el sobreajuste y maximiza el poder discriminativo (ROC-AUC y F1-Score).</p>

<h2 class="section-title">II. Materiales y Métodos (Metodología)</h2>

<h3 class="subsection-title">2.1 Descripción del Conjunto de Datos y Variables</h3>
<p>Se utilizó el conjunto de datos multimodal <i>Student Early Stress &amp; Dropout Detection Dataset</i> (<code>final_student_dataset.csv</code> y su diccionario técnico <code>data_dictionary.csv</code>), conformando una muestra estratificada de <b>N = {meta['total_samples']:,} estudiantes universitarios</b> sin valores perdidos (0.0% de datos nulos). La variable objetivo binaria <i>Y</i> &isin; {{0, 1}} (<code>at_risk</code>) identifica con <code>1</code> al estudiante en riesgo de deserción ({meta['positive_rate_total']*100:.2f}%, 1,945 casos) y con <code>0</code> al estudiante que continúa regularmente sus estudios ({(1-meta['positive_rate_total'])*100:.2f}%, 4,055 casos). Se seleccionaron <b>22 variables predictoras</b> libres de fuga de datos (<i>data leakage</i>) distribuidas en seis dimensiones:</p>
<ul>
  <li><b>Demografía y Nivel Socioeconómico (5 variables):</b> Edad (<code>age</code>), género (<code>gender_encoded</code>), estudiante internacional (<code>international_student</code>), primera generación universitaria (<code>first_generation</code>) e índice de privación socioeconómica (<code>socioeconomic_deprivation_index</code>).</li>
  <li><b>Situación Financiera (2 variables):</b> Tenencia de beca académica/social (<code>scholarship_holder</code>) y estado de pago de pensiones al día (<code>tuition_up_to_date</code>).</li>
  <li><b>Historial Académico Previo (2 variables):</b> Nota de ingreso/admisión (<code>admission_grade</code>) y calificación de formación previa (<code>prior_qualification_grade</code>).</li>
  <li><b>Rendimiento Académico Universitario (6 variables):</b> Promedio ponderado (<code>gpa_score_avg</code>), tendencia del promedio (<code>gpa_score_trend</code>), tasa de cursos aprobados (<code>pass_rate</code>), nota de primera evaluación parcial (<code>first_assessment_score</code>), horas de estudio por semana (<code>study_hours_per_week</code>) y carga académica (<code>study_load</code>).</li>
  <li><b>Compromiso Conductual en Plataforma Virtual LMS (5 variables):</b> Tasa de asistencia (<code>attendance_rate</code>), puntaje de participación en el LMS (<code>lms_engagement_score</code>), ratio de días activos (<code>lms_active_days_ratio</code>), días máximos de inactividad continua (<code>longest_inactivity_gap_days</code>) y tareas no entregadas (<code>missed_submissions_count</code>).</li>
  <li><b>Factores Psicológicos y Fisiológicos (2 variables):</b> Escala de estrés académico percibido de 1 a 10 (<code>stress_level_scale</code>) y horas de sueño diarias (<code>sleep_hours_per_day</code>).</li>
</ul>

<h3 class="subsection-title">2.2 Preprocesamiento y Partición Estratificada</h3>
<p>El dataset fue particionado de forma estratificada conservando la proporción exacta de clases (32.4% positivos / 67.6% negativos) en tres subconjuntos independientes: <b>Entrenamiento (70%, n = 4,200)</b>, <b>Validación (15%, n = 900)</b> y <b>Prueba (15%, n = 900)</b>. Posteriormente, se aplicó estandarización Z-score ajustada exclusivamente con la media (&mu;) y desviación estándar (&sigma;) del conjunto de entrenamiento para evitar contaminación estadística hacia los conjuntos de validación y prueba (<b>Figura 1</b>).</p>

<div class="figure-container">
  <img src="{fig1_uri}" alt="Figura 1">
  <div class="figure-caption"><b>Figura 1.</b> (a) Distribución estratificada de los 6,000 estudiantes en los subconjuntos de Entrenamiento, Validación y Prueba. (b) Coeficientes de correlación de Pearson (<i>r</i>) de las 12 variables con mayor asociación lineal respecto a la deserción estudiantil.</div>
</div>

<h3 class="subsection-title">2.3 Diseño de las Arquitecturas de Redes Neuronales (Estudio Ablativo)</h3>
<p>Dado que el problema corresponde a una <b>Clasificación Binaria</b>, todas las arquitecturas emplean la función de activación <code>ReLU</code> en sus capas ocultas, una única neurona con activación <code>Sigmoid</code> en la capa de salida para estimar la probabilidad condicional <i>P(Y = 1 | X)</i>, y la función de pérdida <i>Binary Crossentropy</i>. Se implementaron cuatro modelos en Keras:</p>
<ol>
  <li><b>Modelo 1 — Baseline Sin Regularizar (M1):</b> Topología profunda <code>128 &rarr; 64 &rarr; 32 &rarr; 1</code> sin penalización de pesos ni capas de desconexión, entrenada durante 60 épocas fijas.</li>
  <li><b>Modelo 2 — Regularización L2 (M2):</b> Topología <code>128 &rarr; 64 &rarr; 32 &rarr; 1</code> incorporando penalización <i>Weight Decay</i> (<code>regularizers.l2(0.003)</code>) en cada capa densa para restringir la norma euclidiana de las matrices sinápticas.</li>
  <li><b>Modelo 3 — Regularización por Dropout (M3):</b> Topología <code>128 &rarr; 64 &rarr; 32 &rarr; 1</code> intercalando capas <code>Dropout(0.35)</code> tras las dos primeras capas ocultas y <code>Dropout(0.20)</code> tras la tercera capa.</li>
  <li><b>Modelo 4 — Arquitectura Piramidal Propuesta (M4):</b> Topología piramidal compacta <code>64 &rarr; 32 &rarr; 16 &rarr; 1</code> que integra simultáneamente <code>BatchNormalization</code>, regularización <code>L2(0.002)</code>, <code>Dropout</code> progresivo (<code>0.35 &rarr; 0.25 &rarr; 0.15</code>) y los callbacks <code>EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)</code> y <code>ReduceLROnPlateau</code>:</li>
</ol>

<div class="code-box"># Arquitectura Propuesta Optimizada (Modelo 4 en Keras)
model = models.Sequential(name="M4_Propuesta_Optimizada_BN_L2_Dropout")
model.add(layers.Input(shape=(22,)))
model.add(layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.002)))
model.add(layers.BatchNormalization())
model.add(layers.Dropout(0.35))
model.add(layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.002)))
model.add(layers.BatchNormalization())
model.add(layers.Dropout(0.25))
model.add(layers.Dense(16, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
model.add(layers.Dropout(0.15))
model.add(layers.Dense(1, activation='sigmoid'))</div>

<h2 class="section-title">III. Resultados y Discusión</h2>

<h3 class="subsection-title">3.1 Dinámica del Entrenamiento y Control del Sobreajuste</h3>
<p>La <b>Figura 2(a)</b> y la <b>Figura 2(b)</b> evidencian con claridad meridiana el fenómeno de sobreajuste en el <b>Modelo 1 (Baseline)</b>: al carecer de restricciones estructurales, su pérdida de entrenamiento cae hasta <b>{m1['train_loss']:.4f}</b>, pero su pérdida de validación diverge drásticamente a partir de la época {m1['best_val_epoch']}, escalando hasta <b>{m1['val_loss']:.4f}</b> en la época 60 (brecha de sobreajuste de <b>{m1['overfit_gap']:+.4f}</b>). Esto ocurre porque las 128, 64 y 32 neuronas memorizan combinaciones idiosincráticas de los 4,200 estudiantes de entrenamiento.</p>

<div class="figure-container">
  <img src="{fig2_uri}" alt="Figura 2">
  <div class="figure-caption"><b>Figura 2.</b> (a) Evolución comparativa de la pérdida de validación (<i>Validation Binary Crossentropy</i>) en las cuatro arquitecturas evaluadas. (b) Contraste directo entre las curvas de entrenamiento y validación del Modelo 1 (Baseline con sobreajuste severo) frente al Modelo 4 Propuesto (convergencia regularizada y parada temprana óptima en la época {m4['best_val_epoch']}).</div>
</div>

<p>En contraste, cuando se aplica la <b>Arquitectura Propuesta (M4)</b>, las capas de <code>BatchNormalization</code> estabilizan las distribuciones internas antes de cada desconexión estocástica <code>Dropout</code>, mientras que el término <code>L2</code> penaliza pesos extremos. Como resultado, la curva de entrenamiento y la de validación evolucionan acopladas (brecha de solo <b>{m4['overfit_gap']:+.4f}</b>), y el callback <code>EarlyStopping</code> detiene el entrenamiento en la época {m4['epochs_ran']}, restaurando automáticamente los pesos óptimos de la <b>época {m4['best_val_epoch']}</b> donde la pérdida de validación fue mínima (<b>{m4['min_val_loss']:.4f}</b>).</p>

<h3 class="subsection-title">3.2 Evaluación Cuantitativa en el Conjunto de Prueba</h3>
<p>La <b>Tabla I</b> presenta el resumen comparativo de todas las métricas sobre el conjunto de prueba independiente (n = 900). El Modelo 4 Propuesto reduce la pérdida de prueba en un <b>{loss_reduction_pct:.1f}%</b> respecto al modelo base (bajando de {m1['test_loss']:.4f} a <b>{m4['test_loss']:.4f}</b>) y alcanza los valores más altos en Exactitud (<b>{m4_t['accuracy']*100:.2f}%</b>), Precisión (<b>{m4_t['precision']*100:.2f}%</b>), Especificidad (<b>{m4_t['specificity']*100:.2f}%</b>), F1-Score (<b>{m4_t['f1_score']*100:.2f}%</b>) y Área Bajo la Curva ROC (<b>AUC = {m4_t['roc_auc']:.4f}</b>).</p>

<table class="scientific-table">
  <caption>TABLA I. COMPARATIVA EXPERIMENTAL DE LAS CUATRO ARQUITECTURAS DE REDES NEURONALES EN EL CONJUNTO DE PRUEBA (n = 900)</caption>
  <thead>
    <tr>
      <th>Arquitectura de Red Neuronal</th>
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
      <td class="left">M1: Baseline (Sin Regularizar)</td>
      <td>{m1['epochs_ran']} ({m1['best_val_epoch']})</td>
      <td>{m1['train_loss']:.4f}</td>
      <td>{m1['val_loss']:.4f}</td>
      <td>{m1['test_loss']:.4f}</td>
      <td>{m1['overfit_gap']:+.4f}</td>
      <td>{m1_t['accuracy']*100:.2f}%</td>
      <td>{m1_t['precision']*100:.2f}%</td>
      <td>{m1_t['recall']*100:.2f}%</td>
      <td>{m1_t['f1_score']*100:.2f}%</td>
      <td>{m1_t['roc_auc']:.4f}</td>
    </tr>
    <tr>
      <td class="left">M2: Regularización L2 (&lambda;=0.003)</td>
      <td>{m2['epochs_ran']} ({m2['best_val_epoch']})</td>
      <td>{m2['train_loss']:.4f}</td>
      <td>{m2['val_loss']:.4f}</td>
      <td>{m2['test_loss']:.4f}</td>
      <td>{m2['overfit_gap']:+.4f}</td>
      <td>{m2_t['accuracy']*100:.2f}%</td>
      <td>{m2_t['precision']*100:.2f}%</td>
      <td>{m2_t['recall']*100:.2f}%</td>
      <td>{m2_t['f1_score']*100:.2f}%</td>
      <td>{m2_t['roc_auc']:.4f}</td>
    </tr>
    <tr>
      <td class="left">M3: Regularización Dropout (p=0.35)</td>
      <td>{m3['epochs_ran']} ({m3['best_val_epoch']})</td>
      <td>{m3['train_loss']:.4f}</td>
      <td>{m3['val_loss']:.4f}</td>
      <td>{m3['test_loss']:.4f}</td>
      <td>{m3['overfit_gap']:+.4f}</td>
      <td>{m3_t['accuracy']*100:.2f}%</td>
      <td>{m3_t['precision']*100:.2f}%</td>
      <td>{m3_t['recall']*100:.2f}%</td>
      <td>{m3_t['f1_score']*100:.2f}%</td>
      <td>{m3_t['roc_auc']:.4f}</td>
    </tr>
    <tr class="highlight-row">
      <td class="left">M4: Propuesta (BN + L2 + Dropout + ES)</td>
      <td>{m4['epochs_ran']} ({m4['best_val_epoch']})</td>
      <td>{m4['train_loss']:.4f}</td>
      <td>{m4['val_loss']:.4f}</td>
      <td>{m4['test_loss']:.4f}</td>
      <td>{m4['overfit_gap']:+.4f}</td>
      <td>{m4_t['accuracy']*100:.2f}%</td>
      <td>{m4_t['precision']*100:.2f}%</td>
      <td>{m4_t['recall']*100:.2f}%</td>
      <td>{m4_t['f1_score']*100:.2f}%</td>
      <td>{m4_t['roc_auc']:.4f}</td>
    </tr>
  </tbody>
</table>

<p>En la <b>Figura 3(a)</b> se detalla la Matriz de Confusión del Modelo Propuesto (M4) sobre los 900 alumnos de prueba: el modelo identifica correctamente a <b>{m4_t['tn']} de los {m4_t['tn']+m4_t['fp']} estudiantes sin riesgo</b> ({m4_t['specificity']*100:.1f}% de especificidad) y detecta correctamente a <b>{m4_t['tp']} de los {m4_t['tp']+m4_t['fn']} estudiantes en riesgo real de deserción</b> ({m4_t['recall']*100:.1f}% de sensibilidad). Asimismo, la curva ROC de la <b>Figura 3(b)</b> confirma la superioridad del modelo propuesto en todos los umbrales de operación.</p>

<div class="figure-container">
  <img src="{fig3_uri}" alt="Figura 3">
  <div class="figure-caption"><b>Figura 3.</b> (a) Matriz de Confusión del Modelo 4 Propuesto sobre el conjunto de prueba (n = 900). (b) Curvas ROC y valores de área bajo la curva (AUC) comparando las cuatro arquitecturas neuronales.</div>
</div>

<h3 class="subsection-title">3.3 Importancia de Variables Predictoras</h3>
<p>Para dotar de explicabilidad a la red neuronal (superando el paradigma de "caja negra"), se evaluó la importancia por permutación en el conjunto de prueba (<b>Figura 4</b>). Los resultados demuestran que el rendimiento académico temprano (promedio GPA, nota de la primera evaluación y tasa de aprobación), junto con el comportamiento digital en el LMS (tareas no entregadas, inactividad prolongada y asistencia) y la condición socioeconómica/financiera, constituyen el núcleo predictivo de la red.</p>

<div class="figure-container">
  <img src="{fig4_uri}" alt="Figura 4">
  <div class="figure-caption"><b>Figura 4.</b> Ranking de las 12 variables con mayor importancia predictiva en la Red Neuronal Propuesta (M4), medida como la caída porcentual en la exactitud de prueba al permutar aleatoriamente cada característica.</div>
</div>

<h2 class="section-title">IV. Conclusiones</h2>
<ol>
  <li>Se comprobó experimentalmente que las redes neuronales densas sin regularización (M1) sufren un pronunciado sobreajuste al entrenarse sobre datos estudiantiles multidimensionales, degradando su pérdida de prueba hasta <b>{m1['test_loss']:.4f}</b>.</li>
  <li>La <b>Arquitectura Piramidal Propuesta (M4)</b> (<code>64 &rarr; 32 &rarr; 16 &rarr; 1</code> con <code>BatchNormalization</code>, <code>L2</code>, <code>Dropout</code> y <code>EarlyStopping</code>) erradicó el sobreajuste y redujo la pérdida de prueba en un <b>{loss_reduction_pct:.1f}%</b> (hasta <b>{m4['test_loss']:.4f}</b>), alcanzando una exactitud de <b>{m4_t['accuracy']*100:.2f}%</b>, un F1-Score de <b>{m4_t['f1_score']*100:.2f}%</b> y un <b>ROC-AUC de {m4_t['roc_auc']:.4f}</b>.</li>
  <li>La integración de métricas de interacción en plataformas virtuales (LMS), rendimiento en la primera evaluación parcial e indicadores de estrés académico permite predecir la deserción antes de finalizar el semestre, ofreciendo una herramienta tecnológica de alto impacto para los sistemas de tutoría universitaria en la UPeU.</li>
</ol>

<h2 class="section-title">Referencias Bibliográficas</h2>
<ol class="references">
  <li>K. Kuzilek, M. Hlosta, and Z. Zdrahal, "Open University Learning Analytics dataset," <i>Scientific Data (Nature)</i>, vol. 4, no. 1, pp. 1–8, 2017, doi: 10.1038/sdata.2017.171.</li>
  <li>V. Realinho, J. Machado, L. Baptista, and M. V. Martins, "Predicting Student Dropout and Academic Success," <i>Data (MDPI)</i>, vol. 7, no. 11, p. 146, 2022, doi: 10.3390/data7110146.</li>
  <li>I. Goodfellow, Y. Bengio, and A. Courville, <i>Deep Learning</i>. Cambridge, MA, USA: MIT Press, 2016.</li>
  <li>F. Chollet, <i>Deep Learning with Python</i>, 2nd ed. Shelter Island, NY, USA: Manning Publications, 2021.</li>
  <li>N. Srivastava, G. Hinton, A. Krizhevsky, I. Sutskever, and R. Salakhutdinov, "Dropout: A simple way to prevent neural networks from overfitting," <i>Journal of Machine Learning Research</i>, vol. 15, no. 1, pp. 1929–1958, 2014.</li>
  <li>S. Ioffe and C. Szegedy, "Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift," in <i>Proc. 32nd Int. Conf. Machine Learning (ICML)</i>, Lille, France, 2015, pp. 448–456.</li>
</ol>

</body>
</html>
"""

    html_path = os.path.join(manuscript_dir, "articulo_cientifico.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print("HTML académico generado en:", html_path)

    pdf_path = os.path.join(pdf_dir, "Articulo_Redes_Neuronales_William_Ortiz.pdf")

    # Buscar Microsoft Edge o Google Chrome para renderizar el PDF vectorial perfecto
    browser_paths = [
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    ]
    browser_exe = None
    for bp in browser_paths:
        if os.path.exists(bp):
            browser_exe = bp
            break

    if browser_exe:
        cmd = [
            browser_exe,
            "--headless",
            "--disable-gpu",
            "--no-pdf-header-footer",
            f"--print-to-pdf={pdf_path}",
            html_path,
        ]
        subprocess.run(cmd, check=True)
        print("PDF académico compilado exitosamente en:", pdf_path)
    else:
        print("No se encontró navegador headless para exportar PDF.")


if __name__ == "__main__":
    build_manuscript_and_pdf()
