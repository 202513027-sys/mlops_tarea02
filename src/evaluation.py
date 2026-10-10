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
        """Score compuesto: 0.6*AUC test + 0.4*AUC val - 0.05*tiempo normalizado."""
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
    """Extension P2 de GetBestModel(): elige al campeon final por estabilidad
    temporal, tal como pide el enunciado.

    Reglas (enunciado, seccion GetBestModel):
      1. Solo compiten los candidatos que ya cumplen el criterio de la P1
         (decay train->test < max_decay_pct).
      2. A TODOS esos candidatos se les corre el backtesting walk-forward
         (no hay que reentrenar nada: solo predecir sobre los meses futuros).
      3. Se descartan los que muestran drift severo (TODAS sus olas superan
         drift_alert_threshold).
      4. Entre los que quedan, el campeon es el de MENOR decay promedio en
         las olas de backtesting (desempate: mayor AUC test).
      5. Si el campeon difiere del ganador de la P1 se registra la razon en
         metadata["motivo_cambio_backtesting"] (con numeros concretos).

    Importante: aplicado al pie de la letra, este criterio puede elegir un
    modelo con MENOR AUC test que el ganador de la P1 (por ejemplo uno mas
    simple, como Regresion Logistica) si es mas estable en el tiempo. Es una
    consecuencia directa del criterio del enunciado.

    Parametros
    ----------
    seleccion : dict
        Resultado de GetBestModel().
    pre_data : dict
        Salida de preprocessing.preprocesar() (trae df_futuro_raw, encoders, etc).
    config : dict
        Config completo (se lee config['backtesting']; opcional
        'max_candidatos' para limitar cuantos candidatos se evaluan).

    Retorna
    -------
    dict con "campeon_final" ({modelo, metadata}), "motivo_cambio" (None si
    el campeon sigue siendo el ganador de la P1), "candidatos_probados" y
    "tabla_candidatos" (resumen por candidato, para consola/JSON).
    """
    from src import backtesting  # import local para evitar ciclos de import

    max_decay_pct = seleccion["max_decay_pct_aplicado"]
    ganador_p1 = seleccion["ganador_final"]

    candidatos = [
        r for r in seleccion["ranking_completo"]
        if r["metadata"]["performance"]["decay_pct"] < max_decay_pct
    ]
    if ganador_p1 not in candidatos:  # caso respaldo (ningun modelo paso el filtro)
        candidatos.insert(0, ganador_p1)

    max_cand = config.get("backtesting", {}).get("max_candidatos")
    if max_cand and len(candidatos) > max_cand:
        # Opcional (ahorra tiempo con datasets enormes): los mejores por AUC test
        # (el ranking ya viene ordenado), siempre incluyendo al ganador de la P1.
        recortados = candidatos[:max_cand]
        if ganador_p1 not in recortados:
            recortados[-1] = ganador_p1
        candidatos = recortados

    # Los meses futuros se transforman UNA sola vez y se reutilizan en todos
    # los candidatos (antes se repetia por modelo).
    olas_cache = backtesting.preparar_olas_futuras(pre_data)

    tabla = []
    for cand in candidatos:
        bt = backtesting.ejecutar_backtesting_para_modelo(
            cand["modelo"], cand["necesita_normalizacion"],
            cand["metadata"]["performance"]["auc_train"],
            pre_data, config, olas_cache=olas_cache,
        )
        cand["metadata"]["backtesting"] = bt
        tabla.append({
            "modelo": cand["metadata"]["ml_name"],
            "auc_test": cand["metadata"]["performance"]["auc_test"],
            "avg_auc_backtest": bt["avg_auc_backtest"],
            "avg_decay_backtest": bt["avg_decay_backtest"],
            "drift_severo_en_todas_las_olas": bt["drift_severo_en_todas_las_olas"],
        })

    def _clave(r):
        """Orden: menor decay promedio en backtesting; desempate por mayor AUC test."""
        d = r["metadata"]["backtesting"]["avg_decay_backtest"]
        return (float("inf") if d is None else d, -r["metadata"]["performance"]["auc_test"])

    # Guarda opcional (apagada por defecto): el decay se mide contra el AUC de
    # train, asi que un modelo mediocre/subajustado puede salir "muy estable".
    # Con backtesting.min_avg_auc_backtest se exige ademas un AUC minimo.
    min_auc_bt = config.get("backtesting", {}).get("min_avg_auc_backtest")

    def _es_estable(c):
        """True si no tiene drift severo en todas las olas y supera el AUC minimo opcional."""
        bt = c["metadata"]["backtesting"]
        if bt["drift_severo_en_todas_las_olas"]:
            return False
        return not (min_auc_bt and (bt["avg_auc_backtest"] or 0) < min_auc_bt)

    estables = [c for c in candidatos if _es_estable(c)]

    motivo_cambio = None
    if not estables:
        campeon = ganador_p1
        motivo_cambio = (
            "Ningun candidato fue estable en backtesting (todos mostraron drift severo en "
            "todas las olas). Se mantiene el ganador de la P1; se recomienda reentrenar "
            "con datos mas recientes antes de produccion."
        )
    else:
        campeon = min(estables, key=_clave)
        if campeon is not ganador_p1:
            m_c, m_p = campeon["metadata"], ganador_p1["metadata"]
            bt_c, bt_p = m_c["backtesting"], m_p["backtesting"]
            if bt_p["drift_severo_en_todas_las_olas"]:
                causa = (f"'{m_p['ml_name']}' (ganador de la P1) mostro drift severo en TODAS sus "
                         f"olas (decay > {bt_p['drift_alert_threshold']}% en cada mes)")
            else:
                causa = (f"'{m_p['ml_name']}' (ganador de la P1) tuvo decay promedio en backtesting "
                         f"de {bt_p['avg_decay_backtest']}%")
            motivo_cambio = (
                f"{causa}, mientras que '{m_c['ml_name']}' tuvo {bt_c['avg_decay_backtest']}% "
                f"(el menor entre {len(estables)} candidato(s) estable(s)). "
                f"AUC test: {m_p['performance']['auc_test']} (P1) vs "
                f"{m_c['performance']['auc_test']} (nuevo campeon)."
            )

    campeon["metadata"]["motivo_cambio_backtesting"] = motivo_cambio
    campeon["metadata"]["ganador_p1_previo"] = ganador_p1["metadata"]["ml_name"]

    return {
        "campeon_final": campeon,
        "motivo_cambio": motivo_cambio,
        "candidatos_probados": [t["modelo"] for t in tabla],
        "tabla_candidatos": tabla,
    }
