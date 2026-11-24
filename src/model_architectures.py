"""
Módulo de Arquitecturas de Redes Neuronales Profundas (Estudio Ablativo)
Proyecto: Predicción Temprana de la Deserción Estudiantil Universitaria mediante Redes Neuronales Profundas Regularizadas
Autor: William David Ortiz Mamani (UPeU - Ingeniería de Sistemas)
"""

from keras import models, layers, regularizers, optimizers


def build_baseline_model(input_dim: int):
    """
    Modelo 1 (Baseline): Red Neuronal Multicapa de alta capacidad sin regularización.
    Diseñada para evidenciar el fenómeno de sobreajuste (Overfitting) cuando no se
    restringe la magnitud de los pesos ni la co-adaptación neuronal.
    """
    model = models.Sequential(name="M1_Baseline_Sin_Regularizar")
    model.add(layers.Input(shape=(input_dim,)))
    model.add(layers.Dense(128, activation="relu"))
    model.add(layers.Dense(64, activation="relu"))
    model.add(layers.Dense(32, activation="relu"))
    model.add(layers.Dense(1, activation="sigmoid"))

    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.003),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_l2_model(input_dim: int, l2_lambda: float = 0.003):
    """
    Modelo 2 (Regularización L2 - Weight Decay):
    Añade una penalización proporcional al cuadrado de la magnitud de los pesos (L2)
    en la función de costo para suavizar la frontera de decisión.
    """
    model = models.Sequential(name="M2_Regularizacion_L2")
    model.add(layers.Input(shape=(input_dim,)))
    model.add(
        layers.Dense(
            128,
            activation="relu",
            kernel_regularizer=regularizers.l2(l2_lambda),
        )
    )
    model.add(
        layers.Dense(
            64,
            activation="relu",
            kernel_regularizer=regularizers.l2(l2_lambda),
        )
    )
    model.add(
        layers.Dense(
            32,
            activation="relu",
            kernel_regularizer=regularizers.l2(l2_lambda),
        )
    )
    model.add(layers.Dense(1, activation="sigmoid"))

    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.002),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_dropout_model(input_dim: int, dropout_rate: float = 0.35):
    """
    Modelo 3 (Regularización por Dropout):
    Introduce apagado estocástico de neuronas durante el entrenamiento para romper
    la co-adaptación entre variables académicas y conductuales.
    """
    model = models.Sequential(name="M3_Regularizacion_Dropout")
    model.add(layers.Input(shape=(input_dim,)))
    model.add(layers.Dense(128, activation="relu"))
    model.add(layers.Dropout(dropout_rate))
    model.add(layers.Dense(64, activation="relu"))
    model.add(layers.Dropout(dropout_rate))
    model.add(layers.Dense(32, activation="relu"))
    model.add(layers.Dropout(0.20))
    model.add(layers.Dense(1, activation="sigmoid"))

    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.002),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_proposed_model(input_dim: int):
    """
    Modelo 4 (Arquitectura Propuesta Optimizada):
    Topología piramidal decreciente (64 -> 32 -> 16 -> 1) que integra sinérgicamente:
    - Regularización L2 (Weight Decay)
    - Batch Normalization (estabilización del desplazamiento interno de covariables)
    - Dropout progresivo (0.35 -> 0.25 -> 0.15)
    - Optimización Adam con tasa de aprendizaje calibrada
    """
    model = models.Sequential(name="M4_Propuesta_Optimizada_BN_L2_Dropout")
    model.add(layers.Input(shape=(input_dim,)))

    # Bloque Oculto 1
    model.add(
        layers.Dense(
            64,
            activation="relu",
            kernel_regularizer=regularizers.l2(0.002),
        )
    )
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.35))

    # Bloque Oculto 2
    model.add(
        layers.Dense(
            32,
            activation="relu",
            kernel_regularizer=regularizers.l2(0.002),
        )
    )
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.25))

    # Bloque Oculto 3
    model.add(
        layers.Dense(
            16,
            activation="relu",
            kernel_regularizer=regularizers.l2(0.001),
        )
    )
    model.add(layers.Dropout(0.15))

    # Capa de Salida Binaria
    model.add(layers.Dense(1, activation="sigmoid"))

    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.0015),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model
