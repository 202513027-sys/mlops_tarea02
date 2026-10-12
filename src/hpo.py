"""
hpo.py
======
Optimizacion de Hiperparametros (HPO). Busca, para cada modelo de
config.yaml -> hpo.models, la mejor combinacion de hiperparametros ANTES del
entrenamiento definitivo (training.py usa estos resultados en vez de las
configs manuales de esos modelos, igual que en la P2).

Cambio del parcial (decision D5)
--------------------------------
En la P2 usabamos RandomizedSearchCV (busqueda al azar). En el parcial se
pide una libreria de HPO de mercado, asi que la estrategia por defecto pasa
a ser **Optuna** con su muestreador TPE (bayesiano): en vez de probar
combinaciones al azar, aprende de los intentos anteriores y prueba primero
las zonas del espacio que vienen dando mejor AUC.

  hpo.libreria: "optuna"         -> Optuna TPE (por defecto, parcial)
  hpo.libreria: "random_search"  -> RandomizedSearchCV (la busqueda de la P2,
                                    se conserva como linea base de comparacion)

Para que la comparacion con la P2 sea justa se usa el MISMO espacio de
busqueda (models.HPO_SEARCH_SPACES): solo cambia la forma de recorrerlo.

Presupuesto de busqueda (configurable): n_trials por modelo, cv folds,
timeout por modelo y max_filas (la busqueda usa una muestra estratificada
del train para no tardar horas; el modelo final si se entrena con todo el
train disponible).
"""

import time

import numpy as np
import optuna
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_score

from src.models import HPO_SEARCH_SPACES, MODEL_REGISTRY


def _muestra_estratificada(X, y, max_filas: int, semilla: int):
    """Toma como maximo max_filas filas manteniendo la proporcion de 0 y 1."""
    if not max_filas or len(X) <= max_filas:
        return X, y
    from sklearn.model_selection import train_test_split
    X_m, _, y_m, _ = train_test_split(X, y, train_size=max_filas, stratify=y, random_state=semilla)
    return X_m, y_m


def _parametros_fijos(nombre_modelo: str, random_state: int) -> dict:
    """Parametros que NO se buscan: se toman de la primera config manual del
    modelo (models.py), asi el HPO evalua exactamente el mismo modelo que
    despues se entrena."""
    base = dict(MODEL_REGISTRY[nombre_modelo]["configs"][0])
    for clave in HPO_SEARCH_SPACES[nombre_modelo]:
        base.pop(clave, None)
    if "random_state" in base:
        base["random_state"] = random_state
    return base


def _buscar_con_optuna(nombre, clase, fijos, espacio, X, y, n_trials, cv, semilla, timeout_s):
    """Busqueda con Optuna (TPE). Retorna (best_params, best_score, trials)."""
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    folds = StratifiedKFold(n_splits=cv, shuffle=True, random_state=semilla)

    def objetivo(trial):
        # Mismo espacio discreto de la P2, recorrido de forma "inteligente"
        params = {k: trial.suggest_categorical(k, v) for k, v in espacio.items()}
        modelo = clase(**fijos, **params)
        return cross_val_score(modelo, X, y, cv=folds, scoring="roc_auc", n_jobs=1).mean()

    estudio = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=semilla),
        study_name=f"hpo_{nombre}",
    )
    estudio.optimize(objetivo, n_trials=n_trials, timeout=timeout_s)

    trials = [
        {"trial": t.number, "auc_cv": round(t.value, 4) if t.value is not None else None,
         "params": t.params, "duracion_s": round(t.duration.total_seconds(), 2) if t.duration else None}
        for t in estudio.trials
    ]
    return estudio.best_params, float(estudio.best_value), trials


def _buscar_con_random_search(clase, fijos, espacio, X, y, n_trials, cv, semilla):
    """Busqueda de la P2 (RandomizedSearchCV), conservada como linea base."""
    buscador = RandomizedSearchCV(
        estimator=clase(**fijos), param_distributions=espacio, n_iter=n_trials,
        scoring="roc_auc", cv=cv, random_state=semilla, n_jobs=1, refit=False,
    )
    buscador.fit(X, y)
    res = buscador.cv_results_
    trials = [
        {"trial": i, "auc_cv": round(float(res["mean_test_score"][i]), 4),
         "params": res["params"][i], "duracion_s": round(float(res["mean_fit_time"][i] * cv), 2)}
        for i in range(len(res["params"]))
    ]
    return buscador.best_params_, float(buscador.best_score_), trials


def buscar_hiperparametros(data: dict, config: dict) -> dict:
    """Ejecuta la busqueda de hiperparametros para cada modelo de hpo.models.

    Parametros
    ----------
    data : dict
        Datos preprocesados (usa X_train o X_train_norm segun el modelo).
    config : dict
        Config completo; se lee config['hpo'] y la semilla de config['split'].

    Retorna
    -------
    dict {nombre_modelo: {"libreria", "n_trials", "best_params",
    "best_cv_score", "time_s", "n_filas_busqueda", "trials"}}.
    """
    hpo_cfg = config.get("hpo", {})
    libreria = hpo_cfg.get("libreria", "optuna")
    n_trials = hpo_cfg.get("n_trials", 15)
    cv = hpo_cfg.get("cv", 3)
    timeout_s = hpo_cfg.get("timeout_s_por_modelo")
    max_filas = hpo_cfg.get("max_filas")
    semilla = config["split"]["random_state"]

    resultados = {}
    for nombre_modelo in hpo_cfg.get("models", []):
        if nombre_modelo not in HPO_SEARCH_SPACES or nombre_modelo not in MODEL_REGISTRY:
            print(f"[hpo] Aviso: '{nombre_modelo}' no tiene espacio de busqueda definido, se omite.")
            continue

        info = MODEL_REGISTRY[nombre_modelo]
        clase, espacio = info["clase"], HPO_SEARCH_SPACES[nombre_modelo]
        X = data["X_train_norm"] if info["necesita_normalizacion"] else data["X_train"]
        X, y = _muestra_estratificada(X, data["y_train"], max_filas, semilla)
        fijos = _parametros_fijos(nombre_modelo, semilla)

        print(f"[hpo] {nombre_modelo}: {libreria}, {n_trials} trials, cv={cv}, {len(X)} filas...")
        inicio = time.time()
        if libreria == "optuna":
            best_params, best_score, trials = _buscar_con_optuna(
                nombre_modelo, clase, fijos, espacio, X, y, n_trials, cv, semilla, timeout_s)
        else:
            best_params, best_score, trials = _buscar_con_random_search(
                clase, fijos, espacio, X, y, n_trials, cv, semilla)
        tiempo = time.time() - inicio

        resultados[nombre_modelo] = {
            "libreria": libreria,
            "strategy": "optuna_tpe" if libreria == "optuna" else "random_search",
            "n_trials": len(trials),
            "best_params": {k: (v.item() if isinstance(v, np.generic) else v) for k, v in best_params.items()},
            "best_cv_score": round(best_score, 4),
            "time_s": round(tiempo, 2),
            "cv": cv,
            "n_filas_busqueda": int(len(X)),
            "trials": trials,
        }
        print(f"[hpo]   -> mejor AUC (cv): {best_score:.4f} | params: {best_params} | {tiempo:.1f}s")

    return resultados
