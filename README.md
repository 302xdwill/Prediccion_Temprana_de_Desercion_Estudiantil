# Prediccion_Temprana_de_Desercion_Estudiantil

## Predicción Temprana de la Deserción Estudiantil Universitaria mediante Redes Neuronales Profundas con Regularización L2, Dropout y Batch Normalization
### *Early Prediction of University Student Dropout Using Deep Neural Networks with L2 Regularization, Dropout, and Batch Normalization*

**Autor:** William David Ortiz Mamani  
**Institución:** Universidad Peruana Unión (UPeU) — Campus Juliaca, Escuela Profesional de Ingeniería de Sistemas  
**Correo Institucional:** `william.ortiz@upeu.edu.pe` | **Correo Personal:** `davidom302@gmail.com` | **GitHub:** [`@302xdwill`](https://github.com/302xdwill)  
**Curso:** Inteligencia Artificial — **Docente:** Mg. Angel Rosendo Condori Coaquira  

---

## 🌐 Enlaces de Despliegue en Vivo (GitHub Pages — Sin necesidad de instalar nada)

Para abrir el sistema directamente desde la computadora del docente o cualquier navegador web:

* 🚀 **1. Sistema Web Interactivo en Vivo (`SIA-UPeU` con Red Neuronal M4):**  
  👉 **[https://302xdwill.github.io/Prediccion_Temprana_de_Desercion_Estudiantil/](https://302xdwill.github.io/Prediccion_Temprana_de_Desercion_Estudiantil/)**
* 🎤 **2. Presentación Interactiva de Diapositivas (10 Slides para Exposición):**  
  👉 **[https://302xdwill.github.io/Prediccion_Temprana_de_Desercion_Estudiantil/presentacion.html](https://302xdwill.github.io/Prediccion_Temprana_de_Desercion_Estudiantil/presentacion.html)**
* 📄 **3. Artículo Científico Completo en Formato PDF (IMRyD):**  
  👉 **[https://302xdwill.github.io/Prediccion_Temprana_de_Desercion_Estudiantil/Articulo_Redes_Neuronales_William_Ortiz.pdf](https://302xdwill.github.io/Prediccion_Temprana_de_Desercion_Estudiantil/Articulo_Redes_Neuronales_William_Ortiz.pdf)**

---

## 📊 Resumen de Resultados Experimentales (Estudio Ablativo de 4 Redes Neuronales)

Evaluación sobre el conjunto de prueba independiente ($n = 900$ estudiantes de una cohorte estratificada de $N = 6,000$ alumnos y $22$ variables predictoras):

| Arquitectura de Red Neuronal | Épocas (Óptima) | Train Loss | Val Loss | Test Loss | Brecha Overfit | Accuracy (%) | Precisión (%) | Recall (%) | F1-Score (%) | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **M1: Baseline (Sin Regularizar)** | 60 (1) | 0.0003 | 1.9622 | 2.0283 | +1.9619 | 78.49% | 67.90% | 63.23% | 65.48% | 0.8226 |
| **M2: Regularización L2 ($\lambda=0.003$)** | 60 (14) | 0.3503 | 0.4411 | 0.5160 | +0.0786 | 79.82% | 74.44% | 57.04% | 64.59% | 0.8522 |
| **M3: Regularización Dropout ($p=0.35$)** | 60 (4) | 0.2327 | 0.4121 | 0.5010 | +0.1108 | 78.82% | 72.52% | 55.33% | 62.77% | 0.8426 |
| **🏆 M4: Propuesta (BN + L2 + Dropout + ES)** | **39 (29)** | **0.3719** | **0.3925** | **0.4408 (-78.3%)** | **+0.0165** | **81.26%** | **75.00%** | **62.89%** | **68.41%** | **0.8752** |

---

## 📁 Estructura del Repositorio

```text
Prediccion_Temprana_de_Desercion_Estudiantil/
├── index.html                                          # Sistema Web SIA-UPeU (Híbrido FastAPI + Inferencia Keras JS)
├── presentacion.html                                   # Presentación de 10 diapositivas para la exposición
├── Articulo_Redes_Neuronales_William_Ortiz.pdf         # Artículo Científico final en PDF
├── docs/                                               # Directorio de despliegue automático en GitHub Pages
├── webapp/
│   ├── app.py                                          # Servidor Backend en Python con FastAPI + Keras
│   └── templates/index.html                            # Plantilla web ejecutiva
├── src/
│   ├── preprocessing.py                                # Partición estratificada (70/15/15) y Z-score
│   ├── model_architectures.py                          # Definición de las 4 arquitecturas en Keras
│   ├── train.py                                        # Entrenamiento comparativo y EarlyStopping
│   └── evaluate.py                                     # Generación de figuras científicas a 300 DPI
├── notebooks/
│   └── Experimento_Redes_Neuronales_Desercion.ipynb    # Cuaderno Jupyter ejecutado paso a paso
├── models/
│   └── mejor_modelo_desercion.keras                    # Pesos entrenados de la Red Neuronal M4
├── results/
│   ├── figures/                                        # 4 figuras científicas (300 DPI)
│   └── tables/                                         # Tablas CSV y JSON de métricas
└── manuscript/
    ├── borrador_articulo.md                            # Manuscrito completo en Markdown
    └── entregas_pdf/Articulo_Redes_Neuronales_William_Ortiz.pdf
```

---

## 💻 Ejecución Local con FastAPI (Opcional)

```bash
pip install -r requirements.txt
python webapp/app.py
```
Luego abrir `http://127.0.0.1:8000` (o ejecutar `iniciar_sistema.bat` en Windows).
