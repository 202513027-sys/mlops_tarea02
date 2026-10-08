"""
evaluation.py
=============
Calcula las metricas de cada modelo entrenado (AUC train/test, decay, tiempo)
y contiene GetBestModel(): la funcion que recorre todos los modelos
entrenados y selecciona el ganaador.

Criterio  base:
    de los modelos con decaimiento train->test < max_decay_pct,
    se elige el de mayor AUC en test.
    decay = (AUC_train - AUC_test) / AUC_train * 100

Criterio innovador:
    Ademas del filtro de decay, se calcula un score compuesto que tambien
    considera el AUC de validacion temporal y el tiempo de entrenamiento.
    Justificacion: el AUC de test viene de un split aleatorio
    dentro de los mismos meses de entrenamiento (optimista, el modelo "ya
    conoce" ese periodo). El AUC de validacion temporal es la metrica que
    realmente se parece a como funcionara el modelo en produccion, sobre
    datos de un mes que nunca vio. Como el modelo se reentrena mensualmente
    sobre decenas de miles de clientes, el tiempo de entrenamiento tambien
    es un costo operativo real, no solo un detalle tecnico -- de ahi la
    pequena penalizacion por tiempo.
"""

from sklearn.metrics import roc_auc_score


def calcular_auc(modelo, X, y) -> float:
    """Calcula el AUC-ROC de un modelo ya entrenado sobre un conjunto (X, y)."""
    proba = modelo.predict_proba(X)[:, 1]
    return roc_auc_score(y, proba)


def calcular_decay(auc_train: float, auc_test: float) -> float:
    """% de caida de desempeno de train a test.

    Un valor positivo alto indica que el modelo memorizo el train
    (overfitting) y generaliza peor de lo esperado.
    """
    if auc_train == 0:
        return 0.0
    return (auc_train - auc_test) / auc_train * 100


def GetBestModel(resultados: list, max_decay_pct: float = 10.0,
                  usar_criterio_innovador: bool = True) -> dict:
    """Selecciona el modelo campeon entre todos los experimentos entrenados.

    Parametros
    ----------
    resultados : list
        Lista de dicts {"modelo": ..., "metadata": ...}, uno por experimento,
        producida por training.py.
    max_decay_pct : float
        Decaimiento maximo permitido train->test (%), desde config.yaml.
    usar_criterio_innovador : bool
        Si True, ademas del filtro de decay, ordena los candidatos con un
        score compuesto (AUC test + AUC val + tiempo). Si False, aplica
        EXACTAMENTE el criterio minimo del curso (mayor AUC test).

    Retorna
    -------
    dict con "ganador_base", "ganador_innovador", "ganador_final" y el
    ranking completo ordenado por AUC test.
    """
    candidatos = [r for r in resultados
                  if r["metadata"]["performance"]["decay_pct"] < max_decay_pct]

    if not candidatos:
        # Ningun modelo cumple el umbral: en vez de dejar el pipeline sin
        # campeon, se avisa y se eligen los de menor decay como respaldo.
        print(f"[GetBestModel] Ningun modelo tiene decay < {max_decay_pct}%. "
              f"Se usan los de menor decaimiento como respaldo.")
        candidatos = sorted(
            resultados, key=lambda r: r["metadata"]["performance"]["decay_pct"]
        )[:3]

    # --- Criterio BASE: mayor AUC test entre los candidatos ---
    ganador_base = max(candidatos, key=lambda r: r["metadata"]["performance"]["auc_test"])

    # --- Criterio INNOVADOR ---
    tiempos = [r["metadata"]["performance"]["training_time_sec"] for r in candidatos]
    tiempo_max = max(tiempos) if tiempos and max(tiempos) > 0 else 1.0

    def score_innovador(r):
        p = r["metadata"]["performance"]
        tiempo_norm = p["training_time_sec"] / tiempo_max
        return (0.6 * p["auc_test"]) + (0.4 * p["auc_val"]) - (0.05 * tiempo_norm)

    for r in candidatos:
        r["metadata"]["performance"]["score_innovador"] = round(score_innovador(r), 4)

    ganador_innovador = max(candidatos, key=score_innovador)
    ganador_final = ganador_innovador if usar_criterio_innovador else ganador_base

    ranking = sorted(
        resultados, key=lambda r: r["metadata"]["performance"]["auc_test"], reverse=True
    )

    return {
        "ganador_base": ganador_base,
        "ganador_innovador": ganador_innovador,
        "ganador_final": ganador_final,
        "ranking_completo": ranking,
        "criterio_usado": "innovador" if usar_criterio_innovador else "base_curso",
        "max_decay_pct_aplicado": max_decay_pct,
    }


def revalidar_con_backtesting(seleccion: dict, pre_data: dict, config: dict) -> dict:
    """Extension P2 de GetBestModel(): revalida al campeon de la P1 exigiendo
    ademas estabilidad en backtesting walk-forward.

    Prueba los candidatos de la P1 (los que ya pasaron el filtro de decay,
    ordenados del mejor al peor por el criterio de la P1) en orden, corriendo
    backtesting.ejecutar_backtesting_para_modelo() SOLO hasta encontrar el
    primero que no muestre drift severo en TODAS sus olas. Esto es barato
    (no hay que reentrenar nada) y evita correr backtesting de los 20
    experimentos cuando el primer candidato ya es estable.

    Si el candidato original (ganador_final de GetBestModel) no pasa, se
    registra el motivo del cambio para que quede trazado en el JSON.

    Parametros
    ----------
    seleccion : dict
        Resultado de GetBestModel().
    pre_data : dict
        Salida de preprocessing.preprocesar() (trae df_futuro_raw, encoders, etc).
    config : dict
        Config completo (se lee config['backtesting']).

    Retorna
    -------
    dict con "campeon_final" (el resultado {modelo, metadata} elegido, ya con
    su metadata["backtesting"] completo), "motivo_cambio" (None si el
    candidato original de la P1 ya era estable) y "candidatos_probados".
    """
    from src import backtesting  # import local para evitar ciclos de import

    max_decay_pct = seleccion["max_decay_pct_aplicado"]
    candidatos_ordenados = [
        r for r in seleccion["ranking_completo"]
        if r["metadata"]["performance"]["decay_pct"] < max_decay_pct
    ]
    if not candidatos_ordenados:
        candidatos_ordenados = [seleccion["ganador_final"]]

    # El candidato de la P1 (ganador_final) siempre se prueba primero,
    # independientemente de su posicion en el ranking por AUC test.
    ganador_p1 = seleccion["ganador_final"]
    orden_prueba = [ganador_p1] + [r for r in candidatos_ordenados if r is not ganador_p1]

    candidatos_probados = []
    campeon_final = None
    motivo_cambio = None

    for candidato in orden_prueba:
        auc_train = candidato["metadata"]["performance"]["auc_train"]
        resultado_bt = backtesting.ejecutar_backtesting_para_modelo(
            candidato["modelo"],
            candidato["necesita_normalizacion"],
            auc_train,
            pre_data,
            config,
        )
        candidato["metadata"]["backtesting"] = resultado_bt
        candidatos_probados.append(candidato["metadata"]["ml_name"])

        if not resultado_bt["drift_severo_en_todas_las_olas"]:
            campeon_final = candidato
            if candidato is not ganador_p1:
                motivo_cambio = (
                    f"El candidato original '{ganador_p1['metadata']['ml_name']}' "
                    f"mostro drift severo en TODAS sus olas de backtesting "
                    f"(decay > {resultado_bt['drift_alert_threshold']}% en cada mes evaluado). "
                    f"Se selecciono '{candidato['metadata']['ml_name']}' por ser el siguiente "
                    f"candidato mas estable temporalmente."
                )
            break

    if campeon_final is None:
        # Ningun candidato fue estable: se avisa y se usa el original de la
        # P1 igual, para no dejar el pipeline sin campeon, documentando el riesgo.
        campeon_final = ganador_p1
        motivo_cambio = (
            "Ningun candidato evaluado fue estable en backtesting (drift severo en "
            "todas las olas para todos). Se mantiene el campeon original de la P1; "
            "se recomienda reentrenar con datos mas recientes antes de produccion."
        )

    campeon_final["metadata"]["motivo_cambio_backtesting"] = motivo_cambio

    return {
        "campeon_final": campeon_final,
        "motivo_cambio": motivo_cambio,
        "candidatos_probados": candidatos_probados,
    }
