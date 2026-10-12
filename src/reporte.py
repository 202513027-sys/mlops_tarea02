"""
reporte.py  (NUEVO en el parcial)
==================================
Arma el resumen final de la corrida (tarea generar_reporte del DAG):

  - results_summary.json : todo lo importante de la corrida en un archivo.
  - tablas_readme.md     : las tablas del README (modelos, backtesting, logs
                           de computo, HPO) ya escritas en Markdown, para
                           copiarlas y pegarlas sin hacerlas a mano.
  - Un resumen impreso en consola (se ve en los logs de Airflow).
"""

import os

from src import monitoring, registry


def _leer(carpeta, nombre, defecto=None):
    return registry.leer_json(os.path.join(carpeta, nombre), defecto)


def construir_resumen(ctx) -> dict:
    """Junta la informacion de todos los JSON de la corrida en un solo dict."""
    carpeta = ctx["carpeta"]
    nombres = _leer(carpeta, "experimentos.json", [])
    metas = [_leer(carpeta, f"{n}_metadata.json") for n in nombres]
    sel = _leer(carpeta, "seleccion.json", {})
    hpo = _leer(carpeta, "hpo_results.json", {})
    compute = monitoring.leer_compute_logs(carpeta)
    campeon = next((m for m in metas if m["es_campeon"]), None)

    tabla_modelos = []
    for m in sorted(metas, key=lambda x: (x["backtesting"]["avg_lift_backtest"] or 0), reverse=True):
        p, n, bt, c = m["performance"], m["negocio_test"], m["backtesting"], m["compute"]
        tabla_modelos.append({
            "modelo": m["ml_name"], "familia": m["categoria"], "hardware": m["hardware"],
            "auc_train": p["auc_train"], "auc_test": p["auc_test"], "decay_pct": p["decay_pct"],
            "lift_test": n["lift"], "ks_test": n["ks"], "pr_auc_test": n["pr_auc"],
            "bt_auc_prom": bt["avg_auc_backtest"], "bt_lift_prom": bt["avg_lift_backtest"],
            "bt_decay_prom": bt["avg_decay_backtest"], "bt_drift_alert": bt["drift_alert"],
            "tiempo_s": p["training_time_sec"], "tamano_modelo_mb": c.get("tamano_modelo_mb"),
            "es_campeon": m["es_campeon"],
            "descartado_por": sel.get("descartados", {}).get(m["ml_name"]),
        })

    return {
        "corrida_id": ctx["corrida_id"],
        "carpeta": carpeta,
        "trazabilidad": ctx["corrida"]["trazabilidad"],
        "ejecuciones_dag": ctx["corrida"]["ejecuciones"],
        "mlflow": {"experiment_id": ctx["corrida"]["mlflow_experiment_id"],
                   "parent_run_id": ctx["parent_run_id"]},
        "campeon": campeon["ml_name"] if campeon else None,
        "seleccion": sel,
        "model_registry": _leer(carpeta, "model_registry.json"),
        "top_pct": ctx["config"]["metrica_negocio"]["top_pct"],
        "metrica_primaria_negocio": f"lift en el top {ctx['config']['metrica_negocio']['top_pct']}% "
                                    f"(promedio en las olas de backtesting)",
        "tabla_modelos": tabla_modelos,
        "backtesting_campeon": campeon["backtesting"] if campeon else None,
        "hpo": {k: {kk: vv for kk, vv in v.items() if kk != "trials"} for k, v in hpo.items()},
        "preprocesamiento": _leer(carpeta, "preprocesamiento.json"),
        "parallelism": _leer(carpeta, "parallelism.json"),
        "validacion_datos": (_leer(carpeta, "validacion_datos.json") or {}).get("resumen"),
        "compute_logs": compute,
    }


def _fmt(v):
    if v is None:
        return "-"
    if isinstance(v, bool):
        return "Sí" if v else "No"
    return str(v)


def _tabla(encabezados, filas) -> str:
    lineas = ["| " + " | ".join(encabezados) + " |", "|" + "---|" * len(encabezados)]
    lineas += ["| " + " | ".join(_fmt(x) for x in fila) + " |" for fila in filas]
    return "\n".join(lineas)


def tablas_markdown(r: dict) -> str:
    """Genera las tablas del README en Markdown."""
    top = r["metrica_primaria_negocio"]
    lift = f"Lift@{r['top_pct']}%"
    partes = [f"# Tablas de la corrida `{r['corrida_id']}` ({os.path.basename(r['carpeta'])})\n",
              f"Campeón: **{r['campeon']}** — estado de la selección: **{r['seleccion'].get('estado')}**.  ",
              f"Métrica primaria de negocio: {top}.\n",
              f"> {r['seleccion'].get('motivo')}\n"]

    partes.append("## Tabla de modelos (ordenada por lift en backtesting)\n")
    partes.append(_tabla(
        ["Modelo", "Familia", "AUC train", "AUC test", "Decay %", f"{lift} test", "KS test",
         f"{lift} backtest (prom)", "Decay backtest prom %", "Tiempo (s)", "Campeón", "Descartado por"],
        [[("🏆 " if t["es_campeon"] else "") + t["modelo"], t["familia"], t["auc_train"], t["auc_test"],
          t["decay_pct"], t["lift_test"], t["ks_test"], t["bt_lift_prom"], t["bt_decay_prom"],
          t["tiempo_s"], t["es_campeon"], t["descartado_por"]] for t in r["tabla_modelos"]]))

    bt = r["backtesting_campeon"]
    if bt:
        partes.append(f"\n\n## Backtesting del campeón ({r['campeon']}, entrenado hasta la partición "
                      f"{bt['train_cutoff_partition']}, sin reentrenar)\n")
        partes.append(_tabla(
            ["Ola", "p_fecinformacion", "AUC", "Decay %", lift, "KS", "PR-AUC", "PSI", "N obs"],
            [[w["partition"], w["p_fecinformacion"], w["auc"], w["decay_pct"], w["lift"], w["ks"],
              w["pr_auc"], w["psi"], w["n_obs"]] for w in bt["waves"]]))
        partes.append(f"\n\nPromedios: AUC {bt['avg_auc_backtest']} · lift {bt['avg_lift_backtest']} · "
                      f"decay {bt['avg_decay_backtest']}% · alerta de drift: {_fmt(bt['drift_alert'])}")

    partes.append("\n\n## Backtesting de todos los modelos (promedio de las olas)\n")
    partes.append(_tabla(
        ["Modelo", "AUC prom", f"{lift} prom", "Decay prom %", "¿Alguna ola sobre el umbral?"],
        [[t["modelo"], t["bt_auc_prom"], t["bt_lift_prom"], t["bt_decay_prom"], t["bt_drift_alert"]]
         for t in r["tabla_modelos"]]))

    comp = r["compute_logs"]
    partes.append(f"\n\n## Logs de cómputo por modelo (máquina: {comp['sistema'].get('nucleos_logicos')} "
                  f"núcleos, {comp['sistema'].get('ram_total_gb')} GB RAM, "
                  f"modo: {comp['sistema'].get('modo_computo')})\n")
    partes.append(_tabla(
        ["Modelo", "Duración (s)", "CPU % prom", "CPU % máx", "RAM MB prom", "RAM MB máx",
         "GPU", "Tamaño modelo (MB)"],
        [[n, c.get("duracion_s"), c.get("cpu_pct_prom"), c.get("cpu_pct_max"), c.get("ram_mb_prom"),
          c.get("ram_mb_max"), c.get("gpu_nombre") or "solo CPU", c.get("tamano_modelo_mb")]
         for n, c in comp["modelos"].items()]))
    partes.append("\n\n*CPU %: 100% = un núcleo completo (un 350% equivale a 3,5 núcleos ocupados).*")

    partes.append("\n\n## Logs de cómputo por tarea del DAG\n")
    partes.append(_tabla(
        ["Tarea", "Duración (s)", "CPU % prom", "CPU % máx", "RAM MB prom", "RAM MB máx", "Modo"],
        [[n, c.get("duracion_s"), c.get("cpu_pct_prom"), c.get("cpu_pct_max"), c.get("ram_mb_prom"),
          c.get("ram_mb_max"), c.get("modo_computo")] for n, c in comp["tareas"].items()]))

    if r["hpo"]:
        partes.append("\n\n## Búsqueda de hiperparámetros\n")
        partes.append(_tabla(
            ["Modelo", "Librería", "Trials", "Mejor AUC (cv)", "Tiempo (s)", "Filas usadas", "Mejores parámetros"],
            [[k, v["libreria"], v["n_trials"], v["best_cv_score"], v["time_s"], v["n_filas_busqueda"],
              f"`{v['best_params']}`"] for k, v in r["hpo"].items()]))

    mr = r.get("model_registry")
    if mr:
        partes.append(f"\n\n## Model Registry\n\nVersión {mr.get('version')} → **{mr.get('accion')}**. "
                      f"{mr.get('motivo', '')}")
    return "\n".join(partes) + "\n"


def imprimir_resumen(r: dict) -> None:
    """Resumen corto en consola (aparece en el log de la tarea en Airflow)."""
    print("\n" + "=" * 100)
    print(f"  RESUMEN DE LA CORRIDA {r['corrida_id']}  ->  {r['carpeta']}")
    print("=" * 100)
    print(f"{'Modelo':<30}{'AUC test':>9}{'Decay%':>8}{'Lift test':>10}{'Lift BT':>9}{'DecayBT%':>9}{'Tiempo':>9}")
    for t in r["tabla_modelos"]:
        marca = " <- CAMPEON" if t["es_campeon"] else ""
        print(f"{t['modelo']:<30}{_fmt(t['auc_test']):>9}{_fmt(t['decay_pct']):>8}{_fmt(t['lift_test']):>10}"
              f"{_fmt(t['bt_lift_prom']):>9}{_fmt(t['bt_decay_prom']):>9}{_fmt(t['tiempo_s']):>9}{marca}")
    print("-" * 100)
    print(f"Campeon: {r['campeon']} | estado: {r['seleccion'].get('estado')}")
    print(f"Motivo: {r['seleccion'].get('motivo')}")
    if r.get("model_registry"):
        print(f"Model Registry: {r['model_registry']}")
    print(f"Trazabilidad: {r['trazabilidad']}")
    print("=" * 100 + "\n")
