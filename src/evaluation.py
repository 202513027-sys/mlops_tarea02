"""
evaluation.py
=============
Metricas de cada modelo y seleccion del campeon con GetBestModel().

Metricas tecnicas (igual que P1/P2)
-----------------------------------
- AUC-ROC en train, test y validacion temporal.
- decay % = (AUC_train - AUC_test) / AUC_train * 100   (decay RELATIVO, D1)

Metrica primaria de NEGOCIO (nuevo en el parcial, decision D4)
-------------------------------------------------------------
Lift en el top 10%: si el banco contacta solo al 10% de clientes con mayor
probabilidad segun el modelo, cuantas veces mas ventas consigue que si
contactara a un 10% elegido al azar.

    lift = tasa_de_venta_en_el_top10% / tasa_de_venta_promedio

  lift = 1   -> el modelo no ayuda (igual que elegir al azar)
  lift = 4   -> el top 10% compra 4 veces mas que el promedio

Tambien se calculan (como referencia, NO deciden): KS, PR-AUC y el % de
ventas capturadas en el top 10%. El porcentaje "top" sale de
config.yaml -> metrica_negocio.top_pct.

GetBestModel() rediseñado (decision D8) -- ver docstring de la funcion.
"""

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve


# ---------------------------------------------------------------------------
# Metricas tecnicas
# ---------------------------------------------------------------------------
def calcular_auc(modelo, X, y) -> float:
    """Calcula el AUC-ROC de un modelo ya entrenado sobre un conjunto (X, y)."""
    proba = modelo.predict_proba(X)[:, 1]
    return roc_auc_score(y, proba)


def calcular_decay(auc_train: float, auc_test: float) -> float:
    """% de caida de desempeno de train a test (decay relativo, D1).

    Un valor positivo alto indica que el modelo memorizo el train
    (overfitting) y generaliza peor de lo esperado.
    """
    if auc_train == 0:
        return 0.0
    return (auc_train - auc_test) / auc_train * 100


# ---------------------------------------------------------------------------
# Metricas de negocio
# ---------------------------------------------------------------------------
def metricas_negocio(y_real, probas, top_pct: float = 10.0) -> dict:
    """Calcula lift, tasa de respuesta y captura en el top_pct%, KS y PR-AUC.

    Parametros
    ----------
    y_real : array de 0/1 (1 = el cliente compro el producto)
    probas : array con la probabilidad predicha por el modelo
    top_pct : % de clientes que se contactarian (los de mayor probabilidad)
    """
    y = np.asarray(y_real).astype(int)
    p = np.asarray(probas, dtype=float)
    n = len(y)
    k = max(1, int(np.ceil(n * top_pct / 100)))

    # orden de mayor a menor probabilidad ("estable" = reproducible si hay empates)
    orden = np.argsort(-p, kind="stable")
    top = y[orden[:k]]

    tasa_base = y.mean() if n else 0.0
    tasa_top = top.mean() if k else 0.0
    ventas_totales = y.sum()

    fpr, tpr, _ = roc_curve(y, p) if len(np.unique(y)) > 1 else ([0], [0], None)

    return {
        "top_pct": top_pct,
        "lift": round(float(tasa_top / tasa_base), 4) if tasa_base > 0 else None,
        "tasa_respuesta_top": round(float(tasa_top), 4),
        "tasa_respuesta_base": round(float(tasa_base), 4),
        "captura_ventas_top": round(float(top.sum() / ventas_totales), 4) if ventas_totales else None,
        "ks": round(float(np.max(np.asarray(tpr) - np.asarray(fpr))), 4),
        "pr_auc": round(float(average_precision_score(y, p)), 4) if ventas_totales else None,
    }


# ---------------------------------------------------------------------------
# GetBestModel (rediseñado en el parcial, D8)
# ---------------------------------------------------------------------------
def _meta(r):
    """Acepta tanto {"modelo","metadata"} (formato P2) como la metadata sola."""
    return r["metadata"] if "metadata" in r else r


def _lift_negocio(meta: dict):
    """Lift promedio en las olas de backtesting (meses futuros). Si no hubo
    backtesting, se usa el lift en test como respaldo."""
    bt = meta.get("backtesting") or {}
    if bt.get("avg_lift_backtest") is not None:
        return bt["avg_lift_backtest"]
    return meta["negocio_test"]["lift"]


def GetBestModel(resultados: list, config: dict) -> dict:
    """Elige el modelo campeon. La metrica de NEGOCIO decide; los otros
    factores funcionan como filtros o como desempate.

    Paso 1 - Filtro de sobreajuste (igual que P1):
        decay train->test < selection.max_decay_pct (10%).
    Paso 2 - Filtro de estabilidad en el tiempo (igual idea que P2):
        decay PROMEDIO en las olas de backtesting < selection.max_decay_backtest_pct (5%)
        y sin drift severo en todas las olas.
    Paso 3 - Metrica de negocio:
        entre los que pasan, gana el de mayor LIFT promedio en backtesting
        (meses futuros que el modelo nunca vio).
    Paso 4 - Costo operativo como desempate:
        si varios modelos estan a menos de selection.tolerancia_lift_pct (2%)
        del mejor lift, se elige el que entrena mas rapido (mismo valor de
        negocio, menor costo de computo).

    Que pasa si ningun modelo es satisfactorio:
        - Nadie pasa el paso 2 -> estado "CON_ALERTA": se elige el mejor lift
          entre los que pasan el paso 1, pero se marca "no promover" (no se
          convierte en champion en el Model Registry).
        - Nadie pasa el paso 1 -> estado "SIN_MODELO_SATISFACTORIO": se toma
          el de mejor lift entre los 3 con menor decay, solo como referencia.

    Parametros
    ----------
    resultados : list
        Metadata de cada modelo (con "performance", "negocio_test" y
        "backtesting"), o dicts {"modelo","metadata"} como en la P2.
    config : dict
        Config completo (se lee config["selection"]).

    Retorna
    -------
    dict con "campeon" (metadata), "estado", "motivo", "promover",
    "criterio" (parametros usados), "descartados" y "ranking".
    """
    sel = config["selection"]
    max_decay = sel["max_decay_pct"]
    max_decay_bt = sel["max_decay_backtest_pct"]
    tolerancia = sel["tolerancia_lift_pct"]

    metas = [_meta(r) for r in resultados]
    descartados = {}

    # Paso 1: sobreajuste
    paso1 = []
    for m in metas:
        d = m["performance"]["decay_pct"]
        if d < max_decay:
            paso1.append(m)
        else:
            descartados[m["ml_name"]] = f"sobreajuste: decay train->test {d}% >= {max_decay}%"

    # Paso 2: estabilidad en backtesting
    paso2 = []
    for m in paso1:
        bt = m.get("backtesting") or {}
        d_bt = bt.get("avg_decay_backtest")
        if d_bt is None:
            descartados[m["ml_name"]] = "sin resultados de backtesting"
        elif bt.get("drift_severo_en_todas_las_olas"):
            descartados[m["ml_name"]] = "drift severo en todas las olas de backtesting"
        elif d_bt >= max_decay_bt:
            descartados[m["ml_name"]] = f"inestable: decay promedio en backtesting {d_bt}% >= {max_decay_bt}%"
        else:
            paso2.append(m)

    if paso2:
        estado, pool, promover = "OK", paso2, True
    elif paso1:
        estado, pool, promover = "CON_ALERTA", paso1, False
    else:
        estado, promover = "SIN_MODELO_SATISFACTORIO", False
        pool = sorted(metas, key=lambda m: m["performance"]["decay_pct"])[:3]

    # Paso 3 y 4: mejor lift; desempate por menor tiempo de entrenamiento
    mejor_lift = max(_lift_negocio(m) for m in pool)
    cercanos = [m for m in pool if _lift_negocio(m) >= mejor_lift * (1 - tolerancia / 100)]
    campeon = min(cercanos, key=lambda m: m["performance"]["training_time_sec"])

    mejor_por_lift = max(pool, key=_lift_negocio)
    motivo = (f"{campeon['ml_name']}: lift promedio en backtesting {_lift_negocio(campeon)} "
              f"(mejor del grupo: {mejor_lift}); {len(pool)} modelo(s) compitieron en la etapa final.")
    if campeon is not mejor_por_lift:
        motivo += (f" Gana por costo: esta a menos de {tolerancia}% del lift de "
                   f"{mejor_por_lift['ml_name']} y entrena en "
                   f"{campeon['performance']['training_time_sec']}s vs "
                   f"{mejor_por_lift['performance']['training_time_sec']}s.")
    if estado == "CON_ALERTA":
        motivo += " ALERTA: ningun modelo fue estable en backtesting; no se recomienda promoverlo."
    if estado == "SIN_MODELO_SATISFACTORIO":
        motivo += " ALERTA: todos los modelos sobreajustan; revisar datos/hiperparametros."

    ranking = sorted(metas, key=_lift_negocio, reverse=True)
    return {
        "campeon": campeon,
        "estado": estado,
        "promover": promover,
        "motivo": motivo,
        "criterio": {
            "metrica_primaria": f"lift_top{config['metrica_negocio']['top_pct']}%_backtesting",
            "max_decay_pct": max_decay,
            "max_decay_backtest_pct": max_decay_bt,
            "tolerancia_lift_pct": tolerancia,
            "desempate": "menor training_time_sec",
            "tipo_decay": "relativo (%)",
        },
        "candidatos_finales": [m["ml_name"] for m in pool],
        "descartados": descartados,
        "ranking": [m["ml_name"] for m in ranking],
    }
