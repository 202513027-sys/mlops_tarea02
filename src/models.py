"""
models.py
=========
Diccionario central de modelos del AutoML. Contiene:
  - Los 3 modelos base que trabajamos en clase (XGBoost, LightGBM, CatBoost):
    debemos agregar más modelos, pero se mantienen
    en el mismo diccionario para que el sistema siga siendo un AutoML completo.
  - 7 modelos adicionales (pidieron mínimo 5), eligiendo uno o mas de cada
    categoria de la lista sugerida, para que la comparacion final sea justa
    entre paradigmas distintos.

Si alguien quiere agrerar otro: agrega una entrada a MODEL_REGISTRY con el
mismo formato. No hay que tocar ningun otro archivo del proyecto salvo el config 
"""

from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier


MODEL_REGISTRY = {
    # =========================================================
    # MODELOS BASE 
    # =========================================================
    "XGBoost": {
        "categoria": "base_arboles_ensamble",
        "es_base": True,
        "clase": XGBClassifier,
        "necesita_normalizacion": False,
        "configs": [
            {"n_estimators": 200, "max_depth": 5, "learning_rate": 0.05,
             "eval_metric": "logloss", "random_state": 42, "n_jobs": -1},
            {"n_estimators": 400, "max_depth": 3, "learning_rate": 0.1,
             "eval_metric": "logloss", "random_state": 42, "n_jobs": -1},
        ],
    },
    "LightGBM": {
        "categoria": "base_arboles_ensamble",
        "es_base": True,
        "clase": LGBMClassifier,
        "necesita_normalizacion": False,
        "configs": [
            {"n_estimators": 200, "max_depth": 5, "learning_rate": 0.05,
             "random_state": 42, "n_jobs": -1, "verbose": -1},
            {"n_estimators": 400, "num_leaves": 63, "learning_rate": 0.05,
             "random_state": 42, "n_jobs": -1, "verbose": -1},
        ],
    },
    "CatBoost": {
        "categoria": "base_arboles_ensamble",
        "es_base": True,
        "clase": CatBoostClassifier,
        "necesita_normalizacion": False,
        "configs": [
            {"iterations": 200, "depth": 5, "learning_rate": 0.05,
             "random_state": 42, "verbose": False, "allow_writing_files":False},
            {"iterations": 400, "depth": 6, "learning_rate": 0.03,
             "random_state": 42, "verbose": False, "allow_writing_files":False},
        ],
    },

    # =========================================================
    # MODELOS ADICIONALES (tenemos 7 con distintos hiperarametros)
    # =========================================================

    # ---- Lineales ----
    "Regresion_Logistica": {
        "categoria": "lineal",
        "es_base": False,
        "clase": LogisticRegression,
        "necesita_normalizacion": True,
        "configs": [
            {"C": 1.0, "max_iter": 1000, "solver": "lbfgs"},
            {"C": 0.1, "max_iter": 1000, "solver": "lbfgs"},
        ],
    },
    "LDA": {
        "categoria": "lineal",
        "es_base": False,
        "clase": LinearDiscriminantAnalysis,
        "necesita_normalizacion": True,
        "configs": [
            {"solver": "svd"},
            {"solver": "lsqr", "shrinkage": "auto"},
        ],
    },

    # ---- Arboles y ensambles ----
    "Random_Forest": {
        "categoria": "arboles_ensamble",
        "es_base": False,
        "clase": RandomForestClassifier,
        "necesita_normalizacion": False,
        "configs": [
            {"n_estimators": 200, "max_depth": 8, "random_state": 42, "n_jobs": -1},
            {"n_estimators": 400, "max_depth": None, "random_state": 42, "n_jobs": -1},
        ],
    },
    "AdaBoost": {
        "categoria": "arboles_ensamble",
        "es_base": False,
        "clase": AdaBoostClassifier,
        "necesita_normalizacion": False,
        "configs": [
            {"n_estimators": 100, "learning_rate": 1.0, "random_state": 42},
            {"n_estimators": 300, "learning_rate": 0.5, "random_state": 42},
        ],
    },

    # ---- Basado en instancias ----
    # KNN mide DISTANCIAS entre clientes: si una variable esta en soles (miles)
    # y otra es una edad (decenas), la de mayor escala domina la distancia.
    # Por eso este modelo SI necesita normalizacion.
    "KNN_Ponderado": {
        "categoria": "instancias",
        "es_base": False,
        "clase": KNeighborsClassifier,
        "necesita_normalizacion": True,
        "configs": [
            {"n_neighbors": 15, "weights": "distance", "n_jobs": -1},
            {"n_neighbors": 30, "weights": "distance", "n_jobs": -1},
        ],
    },

    # ---- Probabilistico ----
    "Naive_Bayes_Gaussiano": {
        "categoria": "probabilistico",
        "es_base": False,
        "clase": GaussianNB,
        "necesita_normalizacion": False,
        "configs": [
            {"var_smoothing": 1e-9},
            {"var_smoothing": 1e-6},
        ],
    },

    # ---- Redes neuronales ----
    # MLP entrena por gradiente: si las variables no estan en escalas
    # parecidas, el entrenamiento se vuelve inestable o no converge.
    "MLP": {
        "categoria": "redes_neuronales",
        "es_base": False,
        "clase": MLPClassifier,
        "necesita_normalizacion": True,
        "configs": [
            {"hidden_layer_sizes": (64, 32), "max_iter": 500, "random_state": 42,
             "early_stopping": True},
            {"hidden_layer_sizes": (128,), "max_iter": 500, "random_state": 42,
             "early_stopping": True},
        ],
    },
}


def listar_experimentos(random_state: int = 42) -> list:
    """Genera la lista plana de exp a partir de MODEL_REGISTRY.

    Cada modelo aparece una vez por cada configuracion de hiperparametros
    definida. Si la config del modelo acepta 'random_state' y no lo trae
    explicito, se le inyecta la semilla del config.yaml para reproducibilidad.

    Parametros
    ----------
    random_state : int
        Semilla aleatoria global (desde config.yaml -> split.random_state).

    Retorna
    -------
    list de dicts, uno por experimento (modelo x configuracion).
    """
    experimentos = []
    for nombre, info in MODEL_REGISTRY.items():
        for i, hp in enumerate(info["configs"], start=1):
            hp_final = dict(hp)
            experimentos.append({
                "nombre_experimento": f"{nombre}_cfg{i}",
                "nombre_modelo": nombre,
                "categoria": info["categoria"],
                "es_base": info["es_base"],
                "clase": info["clase"],
                "necesita_normalizacion": info["necesita_normalizacion"],
                "hiperparametros": hp_final,
            })
    return experimentos


# =========================================================
# P2 -- Espacios de busqueda para HPO (src/hpo.py)
# =========================================================
# Al menos 3 modelos deben pasar por busqueda automatica de hiperparametros
# antes del entrenamiento final (ver src/hpo.py). Elegimos uno de arboles
# base, uno de ensambles adicionales y uno lineal, para mostrar que la
# busqueda funciona igual sin importar la familia del modelo.
#
# Los resultados de esta busqueda REEMPLAZAN los 2 configs manuales de
# esos 3 modelos por una unica config optimizada (ver hpo.py y training.py).
HPO_SEARCH_SPACES = {
    "XGBoost": {
        "n_estimators": [100, 200, 300, 400, 500],
        "max_depth": [3, 4, 5, 6, 7],
        "learning_rate": [0.01, 0.03, 0.05, 0.08, 0.1],
        "subsample": [0.7, 0.8, 0.9, 1.0],
    },
    "Random_Forest": {
        "n_estimators": [100, 200, 300, 400],
        "max_depth": [4, 6, 8, 10, None],
        "min_samples_leaf": [1, 2, 5, 10],
    },
    "LightGBM": {
        "n_estimators": [100, 200, 300, 400],
        "num_leaves": [15, 31, 63],
        "learning_rate": [0.01, 0.03, 0.05, 0.1],
        "subsample": [0.7, 0.85, 1.0],
    },
    "Regresion_Logistica": {
        "C": [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0],
    },
}
