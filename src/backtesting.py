"""
backtesting.py
==============
Evaluacion walk-forward (expanding window): el modelo campeon se entrena UNA
sola vez sobre todos los meses hasta el cutoff (eso ya lo hizo training.py),
y aqui se evalua, mes por mes, contra particiones futuras que NUNCA
participaron en el entrenamiento ni en el ajuste de encoders/scaler.

Ventana elegida: "expanding" (un solo entrenamiento, evaluado hacia
adelante en el tiempo) en vez de "rolling" (reentrenar en cada ola con una
ventana movil) -- la justificacion de negocio esta en el README. Como
alternativa de innovacion para el punto extra, se calcula e interpreta el
"concept drift" de cada ola mediante el PSI (Population Stability Index)
entre la distribucion de probabilidades predichas en train y en cada mes
futuro.

El backtesting se corre para TODOS los candidatos que pasan el filtro de decay
de la P1. En el parcial se evalua a TODOS los modelos entrenados y, ademas
del AUC/decay/PSI de la P2, cada ola calcula la metrica de negocio (lift en
el top %, KS y PR-AUC) que usa GetBestModel() para elegir al campeon. Es barato por diseño: no
hay que reentrenar nada; los meses futuros se transforman una sola vez
(preparar_olas_futuras) y se reutilizan en todos los modelos.
"""

import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.evaluation import metricas_negocio
from src.preprocessing import (
    eliminar_columnas,
    imputar_nulos_categoricos,
    imputar_nulos_numericos,
    transform_label_encoding_seguro,
)


def calcular_psi(probas_referencia: np.ndarray, probas_nuevas: np.ndarray, buckets: int = 10) -> float:
    """Population Stability Index entre dos distribuciones de probabilidades.

    Mide cuanto cambio la distribucion de scores del modelo entre el periodo
    de entrenamiento (referencia) y un mes nuevo. Es la metrica estandar de
    concept/data drift en la industria de riesgo crediticio.

    Interpretacion (regla de la industria):
      PSI < 0.10            -> sin cambio significativo
      0.10 <= PSI < 0.25     -> cambio moderado, vigilar
      PSI >= 0.25            -> cambio significativo, el modelo puede estar
                                 perdiendo validez para esta poblacion
    """
    cortes = np.quantile(probas_referencia, np.linspace(0, 1, buckets + 1))
    cortes[0], cortes[-1] = -np.inf, np.inf
    cortes = np.unique(cortes)
    if len(cortes) < 3:
        return 0.0

    ref_hist, _ = np.histogram(probas_referencia, bins=cortes)
    new_hist, _ = np.histogram(probas_nuevas, bins=cortes)

    ref_pct = np.clip(ref_hist / max(len(probas_referencia), 1), 1e-6, None)
    new_pct = np.clip(new_hist / max(len(probas_nuevas), 1), 1e-6, None)

    psi = np.sum((new_pct - ref_pct) * np.log(new_pct / ref_pct))
    return float(psi)


def interpretar_psi(psi: float) -> str:
    """Traduce el valor de PSI a una etiqueta legible para el README/consola."""
    if psi < 0.10:
        return "sin cambio significativo"
    if psi < 0.25:
        return "cambio moderado (vigilar)"
    return "cambio significativo (posible concept drift)"


def transformar_mes_futuro(df_mes_raw: pd.DataFrame, pre_data: dict) -> tuple:
    """Aplica a un mes futuro (crudo, nunca visto) EXACTAMENTE la misma
    limpieza que preprocessing.preprocesar() aplico al entrenamiento, pero
    reutilizando los encoders/scaler YA AJUSTADOS -- nunca se re-entrena
    nada con datos de backtesting, eso invalidaria la prueba.

    Retorna
    -------
    (X_mes, X_mes_norm, y_mes, n_obs)
    """
    df = df_mes_raw.copy()
    df = eliminar_columnas(df, pre_data["cols_excluir"] + pre_data["cols_alto_nulo"])

    cols_num = df.select_dtypes(include=[np.number]).columns.tolist()
    df = imputar_nulos_numericos(df, cols_num, valor=0.0)

    cols_cat = [c for c in pre_data["cols_categoricas_codificadas"] if c in df.columns]
    df = imputar_nulos_categoricos(df, cols_cat, valor=pre_data["valor_imputacion_categorica"])

    for col, encoder in pre_data["encoders"].items():
        if col in df.columns:
            df[col] = transform_label_encoding_seguro(df[col], encoder)

    target_col = pre_data["target_col"]
    y_mes = df[target_col] if target_col in df.columns else None

    X_mes = df[pre_data["feature_cols"]]
    X_mes_norm = pd.DataFrame(
        pre_data["scaler"].transform(X_mes), columns=X_mes.columns, index=X_mes.index
    )

    return X_mes, X_mes_norm, y_mes, len(df)


def preparar_olas_futuras(pre_data: dict) -> list:
    """Transforma UNA sola vez cada mes futuro (limpieza + encoders + scaler ya
    ajustados) para reutilizarlo en el backtesting de todos los candidatos.

    Retorna
    -------
    list de dicts {"mes", "etiqueta", "X", "X_norm", "y", "n_obs"}, en orden
    cronologico.
    """
    col_mes = pre_data["col_mes"]
    df_futuro_raw = pre_data["df_futuro_raw"]
    meses_futuros = pre_data["meses_ordenados"][pre_data["cutoff_aplicado"]:]
    olas = []
    for mes in meses_futuros:
        df_mes = df_futuro_raw[df_futuro_raw[col_mes] == mes]
        if df_mes.empty:
            continue
        etiqueta = df_mes["partition"].iloc[0] if "partition" in df_mes.columns else str(mes)
        X, X_norm, y, n_obs = transformar_mes_futuro(df_mes, pre_data)
        olas.append({"mes": mes, "etiqueta": etiqueta, "X": X, "X_norm": X_norm,
                     "y": y, "n_obs": n_obs})
    return olas


def ejecutar_backtesting_para_modelo(modelo, necesita_normalizacion: bool,
                                      auc_train: float, pre_data: dict,
                                      config: dict, olas_cache: list = None) -> dict:
    """Corre las olas walk-forward para UN modelo ya entrenado.

    olas_cache : salida de preparar_olas_futuras(); si es None se calcula aqui.

    Retorna el dict con el formato exacto pedido (train_cutoff_partition,
    waves, avg_auc_backtest, drift_alert), mas el PSI por ola como
    extension de concept drift.
    """
    bt_cfg = config.get("backtesting", {})
    top_pct = config.get("metrica_negocio", {}).get("top_pct", 10)
    min_waves = bt_cfg.get("min_waves", 4)
    drift_threshold = bt_cfg.get("drift_alert_threshold", 5.0)

    cutoff = pre_data["cutoff_aplicado"]
    if olas_cache is None:
        olas_cache = preparar_olas_futuras(pre_data)

    if len(olas_cache) < min_waves:
        print(f"[backtesting] AVISO: solo hay {len(olas_cache)} particion(es) futuras "
              f"disponibles (se pidieron minimo {min_waves}). Esto es normal mientras no "
              f"se hayan descargado todas las particiones del dataset -- ver README.")

    # Distribucion de referencia para el PSI: probabilidades predichas sobre
    # el propio train (como luce el modelo cuando "esta sano").
    X_train_eval = pre_data["X_train_norm"] if necesita_normalizacion else pre_data["X_train"]
    probas_train = modelo.predict_proba(X_train_eval)[:, 1]

    inicio_bt = time.time()
    waves = []
    for ola in olas_cache:
        mes, etiqueta, y_mes, n_obs = ola["mes"], ola["etiqueta"], ola["y"], ola["n_obs"]
        X_eval = ola["X_norm"] if necesita_normalizacion else ola["X"]
        probas_mes = modelo.predict_proba(X_eval)[:, 1]

        if y_mes is not None and y_mes.nunique() > 1:
            auc_mes = roc_auc_score(y_mes, probas_mes)
        else:
            auc_mes = None

        decay_mes = ((auc_train - auc_mes) / auc_train * 100) if auc_mes is not None else None
        psi = calcular_psi(probas_train, probas_mes)
        # Parcial: metrica de negocio (lift en el top %) tambien por cada ola
        negocio = metricas_negocio(y_mes, probas_mes, top_pct) if auc_mes is not None else {}

        waves.append({
            "partition": etiqueta,
            "p_fecinformacion": int(mes),
            "auc": round(auc_mes, 4) if auc_mes is not None else None,
            "decay_pct": round(decay_mes, 2) if decay_mes is not None else None,
            "n_obs": int(n_obs),
            "psi": round(psi, 4),
            "psi_interpretacion": interpretar_psi(psi),
            "lift": negocio.get("lift"),
            "ks": negocio.get("ks"),
            "pr_auc": negocio.get("pr_auc"),
        })

    aucs_validas = [w["auc"] for w in waves if w["auc"] is not None]
    decays_validos = [w["decay_pct"] for w in waves if w["decay_pct"] is not None]

    avg_auc_backtest = round(float(np.mean(aucs_validas)), 4) if aucs_validas else None
    avg_decay_backtest = round(float(np.mean(decays_validos)), 2) if decays_validos else None
    lifts_validos = [w["lift"] for w in waves if w["lift"] is not None]
    avg_lift_backtest = round(float(np.mean(lifts_validos)), 4) if lifts_validos else None

    # Alerta de degradacion: al menos una ola cae mas de drift_threshold puntos
    # porcentuales por debajo del AUC de entrenamiento.
    drift_alert = any(w["decay_pct"] is not None and w["decay_pct"] > drift_threshold for w in waves)
    drift_severo_en_todas = len(waves) > 0 and all(
        w["decay_pct"] is not None and w["decay_pct"] > drift_threshold for w in waves
    )

    return {
        "evaluado": True,
        "train_cutoff_partition": cutoff,
        "waves": waves,
        "avg_auc_backtest": avg_auc_backtest,
        "avg_decay_backtest": avg_decay_backtest,
        "avg_lift_backtest": avg_lift_backtest,
        "top_pct_negocio": top_pct,
        "drift_alert": drift_alert,
        "drift_alert_threshold": drift_threshold,
        "drift_severo_en_todas_las_olas": drift_severo_en_todas,
        "tiempo_backtesting_s": round(time.time() - inicio_bt, 2),
    }
