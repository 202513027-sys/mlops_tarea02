"""
hpo.py
======
Optimizacion de Hiperparametros (HPO). Busca, para cada modelo marcado en
models.HPO_SEARCH_SPACES, la mejor combinacion de hiperparametros ANTES del
entrenamiento definitivo (training.py usa los resultados de aqui en vez de
las configs manuales de esos modelos).

Estrategia elegida: RandomizedSearchCV de scikit-learn (permitida por el
enunciado junto con Optuna/hyperopt/skopt). Se prefirio sobre Optuna porque
funciona igual de bien para espacios de busqueda pequenos como los nuestros,
sin agregar una dependencia nueva al proyecto, y es compatible out-of-the-box
con cualquier estimador de la API de scikit-learn (incluye XGBoost/LightGBM/
CatBoost via su wrapper sklearn).

Prohibido (y no usado aqui): TPOT, H2O AutoML, auto-sklearn, PyCaret y
equivalentes -- esos abstraen tambien la seleccion de modelos y el pipeline
completo, no solo la busqueda de hiperparametros.
"""

import time

from sklearn.model_selection import RandomizedSearchCV

from src.models import HPO_SEARCH_SPACES, MODEL_REGISTRY


def buscar_hiperparametros(data: dict, config: dict) -> dict:
    """Ejecuta RandomizedSearchCV para cada modelo en HPO_SEARCH_SPACES.

    Parametros
    ----------
    data : dict
        Salida de preprocessing.preprocesar() (usa X_train/X_train_norm segun
        si el modelo necesita normalizacion).
    config : dict
        Config completo; se lee config['hpo'] (n_trials, cv, random_state).

    Retorna
    -------
    dict {nombre_modelo: {"strategy", "n_trials", "best_params",
    "best_cv_score", "time_s"}}, uno por cada modelo optimizado.
    """
    hpo_cfg = config.get("hpo", {})
    n_trials = hpo_cfg.get("n_trials", 20)
    cv = hpo_cfg.get("cv", 3)
    random_state = config["split"]["random_state"]
    modelos_a_optimizar = hpo_cfg.get("models", list(HPO_SEARCH_SPACES.keys()))

    resultados = {}

    for nombre_modelo in modelos_a_optimizar:
        if nombre_modelo not in HPO_SEARCH_SPACES or nombre_modelo not in MODEL_REGISTRY:
            print(f"[hpo] Aviso: '{nombre_modelo}' no tiene espacio de busqueda definido, se omite.")
            continue

        info_modelo = MODEL_REGISTRY[nombre_modelo]
        clase = info_modelo["clase"]
        espacio = HPO_SEARCH_SPACES[nombre_modelo]
        usa_norm = info_modelo["necesita_normalizacion"]

        X_train = data["X_train_norm"] if usa_norm else data["X_train"]
        y_train = data["y_train"]

        print(f"[hpo] Buscando hiperparametros para {nombre_modelo} "
              f"({n_trials} intentos, cv={cv})...")

        # Instancia base: fija los parametros que NO son parte de la busqueda
        # (ej. random_state, eval_metric, n_jobs) para que RandomizedSearchCV
        # solo explore el espacio definido en HPO_SEARCH_SPACES.
        kwargs_fijos = {}
        if "random_state" in clase().get_params():
            kwargs_fijos["random_state"] = random_state
        if nombre_modelo == "LightGBM":
            kwargs_fijos["verbose"] = -1  # silencia los avisos de LightGBM
        estimador_base = clase(**kwargs_fijos)

        buscador = RandomizedSearchCV(
            estimator=estimador_base,
            param_distributions=espacio,
            n_iter=n_trials,
            scoring="roc_auc",
            cv=cv,
            random_state=random_state,
            n_jobs=1,  # el paralelismo del proyecto vive en parallelism.py, no aqui
            refit=False,
        )

        inicio = time.time()
        buscador.fit(X_train, y_train)
        tiempo = time.time() - inicio

        resultados[nombre_modelo] = {
            "strategy": "random_search",
            "n_trials": n_trials,
            "best_params": buscador.best_params_,
            "best_cv_score": round(float(buscador.best_score_), 4),
            "time_s": round(tiempo, 2),
        }
        print(f"[hpo]   -> mejor AUC (cv): {buscador.best_score_:.4f} "
              f"| params: {buscador.best_params_} | {tiempo:.1f}s")

    return resultados
