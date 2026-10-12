"""
pipeline.py  (NUEVO en el parcial)
===================================
Las etapas del pipeline, UNA funcion por tarea del DAG de Airflow.

Este archivo es el "pegamento" entre Airflow y los modulos de la P1/P2:
  - dags/training_dag.py llama a ejecutar_tarea("nombre_de_la_tarea", ...)
  - main.py (ejecucion local sin Docker) llama a las mismas tareas en orden.
Asi hay un solo lugar con la logica: el DAG y main.py hacen exactamente lo mismo.

Orden de las tareas (ver ORDEN_TAREAS):
  validar_datos -> preprocesar_datos -> optimizar_hiperparametros ->
  entrenar_modelos -> backtesting -> seleccionar_campeon ->
  registrar_experimentos -> generar_reporte

Ideas clave para entender el codigo
-----------------------------------
1. CORRIDA: una corrida = (misma configuracion + mismos datos + mismo commit).
   Su identificador `corrida_id` es un hash de esas tres cosas, asi que si se
   vuelve a lanzar el DAG sin cambiar nada, se reconoce que es la MISMA
   corrida y se reutiliza su carpeta data/output/<fecha_hora>/.
2. PUNTOS DE CONTROL: cada tarea, antes de trabajar, revisa si su resultado
   ya existe en disco. Si existe, no repite el trabajo ("reutilizado").
   Esto, junto con la busqueda de runs por tags en MLflow (registry.py),
   hace que las tareas sean IDEMPOTENTES (decision D7).
3. XCOM: entre tareas solo viaja la ruta de la carpeta de la corrida y su id.
   Los datos y modelos se guardan en disco (data/intermedios/<corrida_id>/)
   o en MLflow, nunca en XCom.
"""

import glob
import hashlib
import os
from datetime import datetime

import joblib
import pandas as pd
import yaml

from src import monitoring, registry

RAIZ = os.environ.get("PROYECTO_DIR") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUTA_CONFIG_DEFECTO = os.path.join("config", "config.yaml")

ORDEN_TAREAS = [
    "validar_datos",
    "preprocesar_datos",
    "optimizar_hiperparametros",
    "entrenar_modelos",
    "backtesting",
    "seleccionar_campeon",
    "registrar_experimentos",
    "generar_reporte",
]

# Parametros que se pueden cambiar al lanzar el DAG ("Trigger DAG w/ config")
# o desde main.py, SIN tocar codigo ni config.yaml (decision D2).
OVERRIDES_PERMITIDOS = {
    "paralelo": ("parallelism", "enabled"),
    "n_jobs": ("parallelism", "n_jobs"),
    "use_gpu": ("gpu", "use_gpu"),
    "input_path": ("dataset", "input_path"),
}


# ---------------------------------------------------------------------------
# Configuracion
# ---------------------------------------------------------------------------
def cargar_config(ruta: str, overrides: dict = None) -> dict:
    """Lee config.yaml y aplica los cambios permitidos (si se pasaron)."""
    with open(ruta, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    for clave, valor in (overrides or {}).items():
        if valor is None or valor == "" or clave not in OVERRIDES_PERMITIDOS:
            continue
        seccion, sub = OVERRIDES_PERMITIDOS[clave]
        print(f"[pipeline] Cambio de configuracion al lanzar: {seccion}.{sub} = {valor}")
        config[seccion][sub] = valor
    return config


def _carpeta_intermedios(config: dict, corrida_id: str) -> str:
    """Carpeta propia de la corrida para archivos pesados (datos, modelos)."""
    ruta = os.path.join(config["output"]["dir_intermedios"], corrida_id)
    os.makedirs(os.path.join(ruta, "modelos"), exist_ok=True)
    return ruta


def _ruta(carpeta: str, nombre: str) -> str:
    return os.path.join(carpeta, nombre)


# ---------------------------------------------------------------------------
# Inicio de la corrida (lo hace la primera tarea)
# ---------------------------------------------------------------------------
def iniciar_corrida(config: dict, dag_run_id: str) -> str:
    """Calcula la trazabilidad, encuentra (o crea) la carpeta de la corrida y
    el run "padre" de MLflow. Retorna la ruta de la carpeta de la corrida."""
    print("[pipeline] Calculando hashes de datos y configuracion (trazabilidad)...")
    git = registry.info_git(RAIZ)
    datos = registry.hash_datos(config["dataset"]["input_path"])
    config_hash = registry.hash_config(config)
    base_id = f"{config_hash}|{datos['data_hash']}|{git['git_commit']}"
    corrida_id = hashlib.sha256(base_id.encode()).hexdigest()[:12]

    # ¿Ya existe una carpeta para esta misma corrida? -> se reutiliza (idempotencia)
    out = config["output"]["output_dir"]
    carpeta = None
    for ruta_json in sorted(glob.glob(os.path.join(out, "*", "corrida.json"))):
        if registry.leer_json(ruta_json, {}).get("corrida_id") == corrida_id:
            carpeta = os.path.dirname(ruta_json)
            print(f"[pipeline] Corrida {corrida_id} ya existe: se reutiliza {carpeta}")
            break

    if carpeta is None:
        carpeta = registry.crear_carpeta_corrida(out)
        corrida = {
            "corrida_id": corrida_id,
            "creada": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "trazabilidad": {
                **git,
                "data_hash": datos["data_hash"],
                "config_hash": config_hash,
                "docker_image": registry.imagen_docker(),
                "airflow_dag_run_id": dag_run_id,
            },
            "hash_por_archivo": datos["archivos"],
            "ejecuciones": [],
        }
        with open(_ruta(carpeta, "config.yaml"), "w", encoding="utf-8") as f:
            yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)
    else:
        corrida = registry.leer_json(_ruta(carpeta, "corrida.json"))

    corrida["ejecuciones"].append({
        "airflow_dag_run_id": dag_run_id,
        "inicio": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "docker_image": registry.imagen_docker(),
    })

    # Run "padre" de MLflow: uno solo por corrida (se busca por el tag corrida_id)
    client, exp_id = registry.conectar_mlflow(config)
    tags_padre = {"corrida_id": corrida_id, "tipo_run": "corrida"}
    padre = registry.buscar_run(client, exp_id, tags_padre)
    if padre is None:
        padre = client.create_run(exp_id, run_name=f"corrida_{os.path.basename(carpeta)}",
                                  tags={**tags_padre, **corrida["trazabilidad"]})
        registry.log_params(client, padre.info.run_id, _aplanar(config))
        registry.log_artefacto(client, padre.info.run_id, _ruta(carpeta, "config.yaml"))
    registry.log_tags(client, padre.info.run_id, {"n_ejecuciones_dag": len(corrida["ejecuciones"])})
    corrida["mlflow_experiment_id"] = exp_id
    corrida["mlflow_parent_run_id"] = padre.info.run_id
    registry.guardar_json(_ruta(carpeta, "corrida.json"), corrida)
    return carpeta


def _aplanar(d: dict, prefijo: str = "") -> dict:
    """{"a": {"b": 1}} -> {"a.b": 1} (para registrar la config como parametros)."""
    plano = {}
    for k, v in d.items():
        clave = f"{prefijo}{k}"
        if isinstance(v, dict):
            plano.update(_aplanar(v, clave + "."))
        else:
            plano[clave] = v
    return plano


def _contexto(carpeta: str, dag_run_id: str) -> dict:
    """Todo lo que una tarea necesita saber de su corrida."""
    config = cargar_config(_ruta(carpeta, "config.yaml"))  # config CONGELADA de la corrida
    corrida = registry.leer_json(_ruta(carpeta, "corrida.json"))
    traza = {**corrida["trazabilidad"], "airflow_dag_run_id": dag_run_id,
             "docker_image": registry.imagen_docker(), "corrida_id": corrida["corrida_id"]}
    return {
        "config": config,
        "carpeta": carpeta,
        "corrida": corrida,
        "corrida_id": corrida["corrida_id"],
        "dag_run_id": dag_run_id,
        "traza": traza,
        "intermedios": _carpeta_intermedios(config, corrida["corrida_id"]),
        "parent_run_id": corrida["mlflow_parent_run_id"],
    }


def _modelos(ctx) -> str:
    return os.path.join(ctx["intermedios"], "modelos")


def _nombres_experimentos(ctx) -> list:
    return registry.leer_json(_ruta(ctx["carpeta"], "experimentos.json"), [])


# ---------------------------------------------------------------------------
# Datos intermedios (en disco, NO por XCom)
# ---------------------------------------------------------------------------
def _ruta_datos(ctx) -> str:
    return os.path.join(ctx["intermedios"], "datos.joblib")


def cargar_datos_intermedios(ctx, normalizar: tuple = (), con_olas: bool = False) -> dict:
    """Carga los datos preprocesados de la corrida desde disco.

    normalizar : que conjuntos se devuelven tambien normalizados (con el
                 scaler guardado), ej. ("X_train",). Solo se calcula lo que
                 la tarea necesita, para ahorrar memoria.
    con_olas   : si True, tambien normaliza los meses futuros del backtesting
                 (si False, se descartan para liberar memoria).
    """
    data = joblib.load(_ruta_datos(ctx))
    sc = data["scaler"]
    for k in normalizar:
        data[f"{k}_norm"] = pd.DataFrame(sc.transform(data[k]), columns=data[k].columns,
                                         index=data[k].index)
    if con_olas:
        for ola in data["olas"]:
            ola["X_norm"] = pd.DataFrame(sc.transform(ola["X"]), columns=ola["X"].columns,
                                         index=ola["X"].index)
    else:
        data["olas"] = None
    return data


# ===========================================================================
# TAREAS
# ===========================================================================
def tarea_validar_datos(ctx) -> dict:
    """Compuerta de calidad de datos (punto extra 6.4)."""
    from src import validacion

    config = ctx["config"]
    if not config["validacion_datos"].get("habilitada", True):
        print("[validacion] Deshabilitada en config.yaml")
        return {"reutilizado": False, "validacion": "deshabilitada"}

    ruta = _ruta(ctx["carpeta"], "validacion_datos.json")
    reporte = registry.leer_json(ruta)
    reutilizado = bool(reporte and reporte["valido"])
    if reutilizado:
        print("[validacion] Los datos de esta corrida ya fueron validados: OK (se reutiliza)")
    else:
        reporte = validacion.validar_datos(config)
        registry.guardar_json(ruta, reporte)

    # Registro en MLflow (si ya existe el run de esta corrida, no se duplica)
    client, exp_id = registry.conectar_mlflow(config)
    run, ya_estaba = registry.obtener_o_crear_run(
        client, exp_id, {"corrida_id": ctx["corrida_id"], "tipo_run": "validacion_datos"},
        "validacion_datos", ctx["traza"], ctx["parent_run_id"])
    if ya_estaba:
        return {"reutilizado": True, "validacion": "OK"}
    rid = run.info.run_id
    registry.log_metrics(client, rid, {
        "n_particiones": reporte.get("resumen", {}).get("n_particiones", 0),
        "n_filas": reporte.get("resumen", {}).get("n_filas", 0),
        "pct_nulos_global": reporte.get("resumen", {}).get("pct_nulos_global", 0),
        "n_errores": len(reporte["errores"]),
        "n_advertencias": len(reporte["advertencias"]),
    })
    registry.log_tags(client, rid, {"resultado_validacion": "OK" if reporte["valido"] else "FALLA"})
    registry.log_artefacto(client, rid, ruta)
    registry.terminar_run(client, rid, "FINISHED" if reporte["valido"] else "FAILED")
    registry.log_tags(client, ctx["parent_run_id"],
                      {"validacion_datos": "OK" if reporte["valido"] else "FALLA"})

    for a in reporte["advertencias"]:
        print(f"[validacion] AVISO: {a}")
    if not reporte["valido"]:
        registry.terminar_run(client, ctx["parent_run_id"], "FAILED")
        mensaje = "DATOS INVALIDOS, el DAG se detiene:\n  - " + "\n  - ".join(reporte["errores"])
        raise validacion.DatosInvalidosError(mensaje)
    print(f"[validacion] OK: {reporte['resumen']}")
    return {"reutilizado": reutilizado, "validacion": "OK"}


def tarea_preprocesar_datos(ctx) -> dict:
    """Igual que la P2 (preprocessing.preprocesar) + guarda el resultado en disco."""
    from sklearn.model_selection import train_test_split
    from src import backtesting, preprocessing

    ruta_datos = _ruta_datos(ctx)
    if os.path.exists(ruta_datos):
        print(f"[preprocesar] Ya existe {ruta_datos}: se reutiliza")
        return {"reutilizado": True}

    config = ctx["config"]
    pre = preprocessing.preprocesar(config)

    # Opcional: limitar filas de entrenamiento (muestra estratificada, D2/D5)
    n_train_original = len(pre["X_train"])
    max_filas = config.get("entrenamiento", {}).get("max_filas_train")
    if max_filas and n_train_original > max_filas:
        X_tr, _, y_tr, _ = train_test_split(pre["X_train"], pre["y_train"], train_size=max_filas,
                                            stratify=pre["y_train"],
                                            random_state=config["split"]["random_state"])
        pre["X_train"], pre["y_train"] = X_tr, y_tr
        print(f"[preprocesar] Train reducido de {n_train_original} a {max_filas} filas (estratificado)")

    # Los meses futuros se transforman UNA vez con los encoders/scaler ya ajustados
    olas = backtesting.preparar_olas_futuras(pre)
    olas = [{k: v for k, v in o.items() if k != "X_norm"} for o in olas]

    claves = ["X_train", "X_test", "X_val", "y_train", "y_test", "y_val", "scaler", "encoders",
              "feature_cols", "cols_excluir", "cols_alto_nulo", "cols_categoricas_codificadas",
              "valor_imputacion_categorica", "mes_validacion_usado", "meses_ordenados",
              "cutoff_aplicado", "col_mes", "target_col"]
    datos = {k: pre[k] for k in claves}
    datos["olas"] = olas
    joblib.dump(datos, ruta_datos + ".tmp")
    os.replace(ruta_datos + ".tmp", ruta_datos)

    resumen = {
        "filas_train_original": int(n_train_original),
        "filas_train_usadas": int(len(datos["X_train"])),
        "filas_test": int(len(datos["X_test"])),
        "filas_validacion_temporal": int(len(datos["X_val"])),
        "n_features": len(datos["feature_cols"]),
        "columnas_descartadas_por_nulos": datos["cols_alto_nulo"],
        "cutoff_aplicado": int(datos["cutoff_aplicado"]),
        "meses_ordenados": [int(m) for m in datos["meses_ordenados"]],
        "mes_validacion_usado": int(datos["mes_validacion_usado"]),
        "olas_backtesting": [{"partition": o["etiqueta"], "mes": int(o["mes"]), "n_obs": int(o["n_obs"])}
                             for o in olas],
        "tasa_target_train": round(float(datos["y_train"].mean()), 4),
    }
    registry.guardar_json(_ruta(ctx["carpeta"], "preprocesamiento.json"), resumen)
    return {"reutilizado": False}


def tarea_optimizar_hiperparametros(ctx) -> dict:
    """HPO con Optuna (o RandomizedSearchCV como linea base) + registro en MLflow."""
    from src import hpo as hpo_mod

    config = ctx["config"]
    ruta = _ruta(ctx["carpeta"], "hpo_results.json")
    resultados = registry.leer_json(ruta)
    reutilizado = resultados is not None
    if reutilizado:
        print("[hpo] Ya existe hpo_results.json para esta corrida: se reutiliza")
    else:
        data = cargar_datos_intermedios(ctx, normalizar=("X_train",))
        resultados = hpo_mod.buscar_hiperparametros(data, config)
        registry.guardar_json(ruta, resultados)

    # Registro de cada busqueda en MLflow (un run hijo por modelo optimizado)
    client, exp_id = registry.conectar_mlflow(config)
    for modelo, info in resultados.items():
        run, ya_estaba = registry.obtener_o_crear_run(
            client, exp_id, {"corrida_id": ctx["corrida_id"], "tipo_run": "hpo", "modelo": modelo},
            f"HPO_{modelo}", ctx["traza"], ctx["parent_run_id"])
        if ya_estaba:
            continue
        rid = run.info.run_id
        registry.log_params(client, rid, {
            "libreria": info["libreria"], "estrategia": info["strategy"], "n_trials": info["n_trials"],
            "cv": info["cv"], "n_filas_busqueda": info["n_filas_busqueda"],
            **{f"best_{k}": v for k, v in info["best_params"].items()},
        })
        registry.log_metrics(client, rid, {"best_cv_auc": info["best_cv_score"],
                                           "n_trials": info["n_trials"], "tiempo_busqueda_s": info["time_s"]})
        for t in info["trials"]:  # historial: un punto por trial (se ve como grafico en MLflow)
            if t["auc_cv"] is not None:
                registry.log_metrics(client, rid, {"trial_auc_cv": t["auc_cv"]}, step=t["trial"])
        ruta_trials = os.path.join(ctx["intermedios"], f"hpo_{modelo}_trials.json")
        registry.guardar_json(ruta_trials, info)
        registry.log_artefacto(client, rid, ruta_trials)
        registry.terminar_run(client, rid)
    return {"reutilizado": reutilizado, "modelos_optimizados": list(resultados)}


def tarea_entrenar_modelos(ctx) -> dict:
    """Entrena los experimentos que falten (secuencial o paralelo, D2)."""
    from src import parallelism, training

    config = ctx["config"]
    hpo_results = registry.leer_json(_ruta(ctx["carpeta"], "hpo_results.json"), {})
    experimentos = training.construir_lista_experimentos(config, hpo_results)
    registry.guardar_json(_ruta(ctx["carpeta"], "experimentos.json"),
                          [e["nombre_experimento"] for e in experimentos])

    # Punto de control por modelo: el .json se escribe al final, si existe el modelo esta completo
    pendientes = [e for e in experimentos
                  if not os.path.exists(os.path.join(_modelos(ctx), f"{e['nombre_experimento']}.json"))]
    hechos = len(experimentos) - len(pendientes)
    if not pendientes:
        print(f"[entrenar] Los {len(experimentos)} modelos de esta corrida ya estan entrenados: se reutilizan")
        return {"reutilizado": True, "n_modelos": len(experimentos)}
    if hechos:
        print(f"[entrenar] {hechos} modelo(s) ya estaban entrenados (reintento); faltan {len(pendientes)}")

    data = cargar_datos_intermedios(ctx)
    info_par = parallelism.entrenar_y_guardar_todos(pendientes, data, _modelos(ctx), config)
    info_par["experimentos_reutilizados"] = hechos
    registry.guardar_json(_ruta(ctx["carpeta"], "parallelism.json"), info_par)
    return {"reutilizado": False, "n_modelos": len(experimentos)}


def tarea_backtesting(ctx) -> dict:
    """Evalua cada modelo, SIN reentrenarlo, en los meses posteriores al corte."""
    from src import backtesting

    nombres = _nombres_experimentos(ctx)
    pendientes = [n for n in nombres
                  if not os.path.exists(os.path.join(_modelos(ctx), f"{n}_backtesting.json"))]
    if not pendientes:
        print("[backtesting] Ya calculado para todos los modelos: se reutiliza")
        return {"reutilizado": True}

    data = cargar_datos_intermedios(ctx, normalizar=("X_train",), con_olas=True)
    for n in pendientes:
        meta = registry.leer_json(os.path.join(_modelos(ctx), f"{n}.json"))
        modelo = joblib.load(os.path.join(_modelos(ctx), f"{n}.joblib"))
        bt = backtesting.ejecutar_backtesting_para_modelo(
            modelo, meta["necesita_normalizacion"], meta["performance"]["auc_train"],
            data, ctx["config"], olas_cache=data["olas"])
        registry.guardar_json(os.path.join(_modelos(ctx), f"{n}_backtesting.json"), bt)
        print(f"[backtesting] {n}: AUC prom={bt['avg_auc_backtest']} lift prom={bt['avg_lift_backtest']} "
              f"decay prom={bt['avg_decay_backtest']}%")
    return {"reutilizado": False}


def tarea_seleccionar_campeon(ctx) -> dict:
    """GetBestModel() con la metrica de negocio (D8)."""
    from src import evaluation

    ruta_sel = _ruta(ctx["carpeta"], "seleccion.json")
    if os.path.exists(ruta_sel):
        sel = registry.leer_json(ruta_sel)
        print(f"[seleccion] Ya seleccionado: {sel['campeon']} (se reutiliza)")
        return {"reutilizado": True, "campeon": sel["campeon"]}

    metas = []
    for n in _nombres_experimentos(ctx):
        meta = registry.leer_json(os.path.join(_modelos(ctx), f"{n}.json"))
        meta["backtesting"] = registry.leer_json(os.path.join(_modelos(ctx), f"{n}_backtesting.json"))
        metas.append(meta)

    sel = evaluation.GetBestModel(metas, ctx["config"])
    campeon = sel["campeon"]["ml_name"]

    # <modelo>_metadata.json en la carpeta de la corrida (como en P1/P2)
    for m in metas:
        m["es_campeon"] = m["ml_name"] == campeon
        registry.guardar_json(_ruta(ctx["carpeta"], f"{m['ml_name']}_metadata.json"), m)

    # backtesting_results.json: formato P2 (olas del campeon) + todos los modelos
    bt_campeon = sel["campeon"]["backtesting"]
    registry.guardar_json(_ruta(ctx["carpeta"], "backtesting_results.json"), {
        "modelo": campeon, **bt_campeon,
        "todos_los_modelos": {m["ml_name"]: m["backtesting"] for m in metas},
    })
    registry.guardar_json(ruta_sel, {**sel, "campeon": campeon})
    print(f"[seleccion] Campeon: {campeon} | estado: {sel['estado']}\n[seleccion] {sel['motivo']}")
    return {"reutilizado": False, "campeon": campeon}


def _versiones_pip() -> list:
    """Requisitos del modelo para MLflow (evita que MLflow los adivine, que es lento)."""
    from importlib.metadata import version
    paquetes = ["scikit-learn", "xgboost", "lightgbm", "catboost", "pandas", "numpy"]
    return [f"{p}=={version(p)}" for p in paquetes]


def tarea_registrar_experimentos(ctx) -> dict:
    """Un run de MLflow por modelo (parametros, metricas, artefactos, tags) +
    criterio de seleccion en el run padre + Model Registry (punto extra)."""
    import mlflow.sklearn  # noqa: F401  (carga el "sabor" sklearn de MLflow)

    config, carpeta = ctx["config"], ctx["carpeta"]
    client, exp_id = registry.conectar_mlflow(config)
    sel = registry.leer_json(_ruta(carpeta, "seleccion.json"))
    pip_req = _versiones_pip()
    run_ids, nuevos = {}, 0

    for n in _nombres_experimentos(ctx):
        meta = registry.leer_json(_ruta(carpeta, f"{n}_metadata.json"))
        run, ya_estaba = registry.obtener_o_crear_run(
            client, exp_id, {"corrida_id": ctx["corrida_id"], "tipo_run": "modelo", "modelo": n},
            n, {**ctx["traza"], "familia": meta["categoria"], "hardware": meta["hardware"],
                "es_campeon": str(meta["es_campeon"])}, ctx["parent_run_id"])
        rid = run.info.run_id
        run_ids[n] = rid
        if ya_estaba:
            continue
        nuevos += 1
        perf, neg, bt, comp = meta["performance"], meta["negocio_test"], meta["backtesting"], meta["compute"]

        registry.log_params(client, rid, {
            **{f"hp_{k}": v for k, v in meta["hyperparameters"].items()},
            "modelo": n, "familia": meta["categoria"], "hardware": meta["hardware"],
            "hpo_libreria": (meta.get("hpo") or {}).get("libreria", "ninguna (config manual)"),
            "train_cutoff_partition": config["backtesting"]["train_cutoff_partition"],
            "max_filas_train": config["entrenamiento"]["max_filas_train"],
            "random_state": config["split"]["random_state"],
            "top_pct_negocio": config["metrica_negocio"]["top_pct"],
            "modo_entrenamiento": "paralelo" if config["parallelism"]["enabled"] else "secuencial",
        })
        metricas = {
            "auc_train": perf["auc_train"], "auc_test": perf["auc_test"], "auc_val": perf["auc_val"],
            "decay_pct": perf["decay_pct"], "training_time_s": perf["training_time_sec"],
            "lift_test": neg["lift"], "ks_test": neg["ks"], "pr_auc_test": neg["pr_auc"],
            "tasa_respuesta_top_test": neg["tasa_respuesta_top"],
            "captura_ventas_top_test": neg["captura_ventas_top"],
            "bt_auc_prom": bt["avg_auc_backtest"], "bt_lift_prom": bt["avg_lift_backtest"],
            "bt_decay_prom": bt["avg_decay_backtest"], "bt_drift_alert": int(bt["drift_alert"]),
        }
        for w in bt["waves"]:
            p = w["partition"]
            metricas.update({f"bt_auc_{p}": w["auc"], f"bt_lift_{p}": w["lift"],
                             f"bt_decay_{p}": w["decay_pct"], f"bt_psi_{p}": w["psi"]})
        for k in ("duracion_s", "cpu_pct_prom", "cpu_pct_max", "ram_mb_prom", "ram_mb_max",
                  "gpu_pct_prom", "gpu_pct_max", "gpu_mem_mb_max", "tamano_modelo_mb", "nucleos_usados_max"):
            metricas[f"computo_{k}"] = comp.get(k)
        registry.log_metrics(client, rid, metricas)
        registry.log_tags(client, rid, {"modo_computo": comp.get("modo_computo")})

        # Artefactos: metadata, curva ROC, backtesting, logs de computo y config
        tmp = os.path.join(ctx["intermedios"], "artefactos", n)
        os.makedirs(tmp, exist_ok=True)
        registry.guardar_json(os.path.join(tmp, "backtesting_results.json"), bt)
        registry.guardar_json(os.path.join(tmp, "compute_logs.json"), comp)
        registry.log_artefacto(client, rid, _ruta(carpeta, f"{n}_metadata.json"))
        registry.log_artefacto(client, rid, os.path.join(tmp, "backtesting_results.json"))
        registry.log_artefacto(client, rid, os.path.join(tmp, "compute_logs.json"))
        registry.log_artefacto(client, rid, os.path.join(_modelos(ctx), f"{n}_roc.png"), "graficos")
        registry.log_artefacto(client, rid, _ruta(carpeta, "config.yaml"))

        # Modelo serializado
        modelo = joblib.load(os.path.join(_modelos(ctx), f"{n}.joblib"))
        import mlflow
        with mlflow.start_run(run_id=rid):
            # cloudpickle: formato que soporta XGBoost/LightGBM/CatBoost (skops, el formato
            # por defecto de MLflow 3, solo acepta tipos de scikit-learn)
            info = mlflow.sklearn.log_model(
                modelo, name="modelo", pip_requirements=pip_req,
                serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE)
        registry.log_tags(client, rid, {"model_uri": info.model_uri})
        registry.terminar_run(client, rid)
        print(f"[registro] Run de MLflow creado para {n}")

    # Run padre: criterio de seleccion aplicado en ESTA corrida
    padre = ctx["parent_run_id"]
    campeon = sel["campeon"]
    meta_c = registry.leer_json(_ruta(carpeta, f"{campeon}_metadata.json"))
    registry.log_tags(client, padre, {
        "campeon": campeon, "estado_seleccion": sel["estado"],
        "motivo_seleccion": sel["motivo"][:4900], "campeon_run_id": run_ids.get(campeon),
    })
    registry.log_metrics(client, padre, {
        "campeon_lift_test": meta_c["negocio_test"]["lift"],
        "campeon_bt_lift_prom": meta_c["backtesting"]["avg_lift_backtest"],
        "campeon_auc_test": meta_c["performance"]["auc_test"],
        "campeon_decay_pct": meta_c["performance"]["decay_pct"],
        "n_modelos": len(run_ids),
    })
    for nombre in ("seleccion.json", "backtesting_results.json", "hpo_results.json",
                   "preprocesamiento.json", "validacion_datos.json", "parallelism.json", "corrida.json"):
        registry.log_artefacto(client, padre, _ruta(carpeta, nombre))

    # Model Registry (punto extra)
    if config["tracking"].get("model_registry", False):
        run_campeon = client.get_run(run_ids[campeon])
        decision = registry.promover_en_model_registry(
            client, config, run_campeon.data.tags["model_uri"], ctx["corrida_id"],
            {"bt_lift_prom": meta_c["backtesting"]["avg_lift_backtest"],
             "lift_test": meta_c["negocio_test"]["lift"],
             "auc_test": meta_c["performance"]["auc_test"], "modelo": campeon},
            run_ids[campeon], sel["promover"])
        if decision["accion"] == "ya_registrado":  # idempotencia: esta corrida ya tenia version
            print(f"[registro] Model Registry: la corrida ya tenia la version {decision['version']}")
        else:
            registry.guardar_json(_ruta(carpeta, "model_registry.json"), decision)
            registry.log_tags(client, padre, {"model_registry_decision": decision["accion"],
                                              "model_registry_version": decision["version"]})
            registry.log_artefacto(client, padre, _ruta(carpeta, "model_registry.json"))
            print(f"[registro] Model Registry: version {decision['version']} -> {decision['accion']}")
    return {"reutilizado": nuevos == 0, "runs_nuevos": nuevos}


def tarea_generar_reporte(ctx) -> dict:
    """Resumen final: results_summary.json + tablas en Markdown para el README."""
    from src import reporte

    resumen = reporte.construir_resumen(ctx)
    registry.guardar_json(_ruta(ctx["carpeta"], "results_summary.json"), resumen)
    ruta_md = _ruta(ctx["carpeta"], "tablas_readme.md")
    with open(ruta_md, "w", encoding="utf-8") as f:
        f.write(reporte.tablas_markdown(resumen))
    reporte.imprimir_resumen(resumen)

    client, _ = registry.conectar_mlflow(ctx["config"])
    padre = ctx["parent_run_id"]
    registry.log_artefacto(client, padre, _ruta(ctx["carpeta"], "results_summary.json"))
    registry.log_artefacto(client, padre, ruta_md)
    registry.log_tags(client, padre, {"estado_corrida": "completada"})
    registry.terminar_run(client, padre)
    return {"reutilizado": False}


TAREAS = {
    "validar_datos": tarea_validar_datos,
    "preprocesar_datos": tarea_preprocesar_datos,
    "optimizar_hiperparametros": tarea_optimizar_hiperparametros,
    "entrenar_modelos": tarea_entrenar_modelos,
    "backtesting": tarea_backtesting,
    "seleccionar_campeon": tarea_seleccionar_campeon,
    "registrar_experimentos": tarea_registrar_experimentos,
    "generar_reporte": tarea_generar_reporte,
}


# ===========================================================================
# Punto de entrada unico (lo usan el DAG y main.py)
# ===========================================================================
def ejecutar_tarea(nombre: str, dag_run_id: str, carpeta_corrida: str = None,
                   ruta_config: str = None, overrides: dict = None) -> dict:
    """Ejecuta UNA tarea del pipeline midiendo sus recursos (Seccion 9).

    - La primera tarea (validar_datos) recibe la ruta de config.yaml y crea/
      encuentra la corrida.
    - Las demas reciben la carpeta de la corrida (via XCom en Airflow).

    Retorna un dict liviano (rutas/ids) que el DAG pasa por XCom.
    """
    os.chdir(RAIZ)  # las rutas de config.yaml son relativas a la raiz del proyecto
    print(f"\n{'=' * 70}\n[pipeline] TAREA: {nombre}   (dag_run_id={dag_run_id})\n{'=' * 70}")

    config_inicial = None
    if nombre == "validar_datos":
        config_inicial = cargar_config(ruta_config or RUTA_CONFIG_DEFECTO, overrides)
        intervalo = config_inicial["monitoreo"]["intervalo_s"]
    else:
        intervalo = cargar_config(_ruta(carpeta_corrida, "config.yaml"))["monitoreo"]["intervalo_s"]

    with monitoring.MonitorRecursos(nombre, intervalo) as mon:
        if nombre == "validar_datos":
            carpeta_corrida = iniciar_corrida(config_inicial, dag_run_id)
        ctx = _contexto(carpeta_corrida, dag_run_id)
        if nombre == "entrenar_modelos" and ctx["config"]["parallelism"]["enabled"]:
            from src.parallelism import contar_workers
            par = ctx["config"]["parallelism"]
            mon.nucleos_asignados = contar_workers(par.get("n_jobs", "auto"), par.get("n_jobs_max"))
        salida = TAREAS[nombre](ctx) or {}

    _registrar_computo_tarea(ctx, nombre, mon.resultado, salida.get("reutilizado", False))
    salida.update({"carpeta_corrida": carpeta_corrida, "corrida_id": ctx["corrida_id"]})
    return {k: v for k, v in salida.items() if isinstance(v, (str, int, float, bool, list, type(None)))}


def _registrar_computo_tarea(ctx, nombre: str, medicion: dict, reutilizado: bool) -> None:
    """Guarda el log de computo de la tarea en compute_logs.json y en MLflow."""
    logs = monitoring.leer_compute_logs(ctx["carpeta"])
    if reutilizado:
        # Re-ejecucion que no hizo trabajo nuevo: se anota, pero no se pisa la medicion original
        logs["reejecuciones_omitidas"].append({"tarea": nombre, "dag_run_id": ctx["dag_run_id"],
                                               "fecha": medicion["inicio"], "duracion_s": medicion["duracion_s"]})
    else:
        logs["tareas"][nombre] = {**medicion, "dag_run_id": ctx["dag_run_id"]}

    # Logs por modelo (vienen de la metadata de cada entrenamiento)
    for n in _nombres_experimentos(ctx):
        meta = registry.leer_json(os.path.join(_modelos(ctx), f"{n}.json"))
        if meta and meta.get("compute"):
            logs["modelos"][n] = meta["compute"]
    monitoring.guardar_compute_logs(ctx["carpeta"], logs)

    try:
        client, _ = registry.conectar_mlflow(ctx["config"])
        if not reutilizado:
            registry.log_metrics(client, ctx["parent_run_id"], {
                f"tarea_{nombre}_{k}": medicion.get(k)
                for k in ("duracion_s", "cpu_pct_prom", "cpu_pct_max", "ram_mb_prom", "ram_mb_max",
                          "gpu_pct_max", "gpu_mem_mb_max")})
        registry.log_artefacto(client, ctx["parent_run_id"], monitoring.ruta_compute_logs(ctx["carpeta"]))
    except Exception as e:  # medir nunca debe tumbar el pipeline
        print(f"[pipeline] Aviso: no se pudo subir el log de computo a MLflow: {e}")
