"""
registry.py
===========
P1/P2: guarda la metadata de cada modelo como .json y crea el directorio de
la corrida con timestamp (data/output/YYYY-MM-DD_HH-MM-SS/).

Extension del parcial (Secciones 5.1, 5.2 y 5.3):
  1. TRAZABILIDAD: calcula git_commit, git_branch, data_hash, config_hash y
     docker_image (funciones *_trazabilidad / hash_*).
  2. TRACKING en MLflow: un run "padre" por corrida y, colgando de el, un run
     por modelo, uno por cada busqueda de HPO y uno para la validacion de
     datos. Cada run lleva los tags de trazabilidad.
  3. IDEMPOTENCIA (D7): antes de crear un run se busca si ya existe uno con
     los mismos tags (corrida_id + tipo_run + modelo). Si existe y termino,
     no se crea otro; asi re-ejecutar el DAG no duplica runs.
  4. MODEL REGISTRY (punto extra): registra al campeon y decide con alias
     "champion" / "challenger" si reemplaza al campeon vigente.
"""

import glob
import hashlib
import json
import os
import platform
import sys
from datetime import datetime

import yaml

import joblib
import numpy
import pandas
import sklearn
import xgboost
import lightgbm
import catboost


def crear_carpeta_corrida(output_dir: str) -> str:
    """Crea output_dir/YYYY-MM-DD_HH-MM-SS/ y retorna la ruta creada.

    Ese directorio con timestamp es la unidad de experimento registrado:
    cada corrida queda versionada por separado, sin sobrescribir corridas
    anteriores.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    carpeta = os.path.join(output_dir, timestamp)
    os.makedirs(carpeta, exist_ok=True)
    print(f"[registry] Carpeta de corrida creada: {carpeta}")
    return carpeta


def versiones_librerias() -> dict:
    """Retorna la version de cada libreria relevante, para reproducibilidad."""
    return {
        "python": sys.version.split()[0],
        "sistema_operativo": platform.platform(),
        "pandas": pandas.__version__,
        "numpy": numpy.__version__,
        "scikit-learn": sklearn.__version__,
        "xgboost": xgboost.__version__,
        "lightgbm": lightgbm.__version__,
        "catboost": catboost.__version__,
    }


def construir_metadata(nombre_modelo: str, auc_train: float, auc_test: float,
                        auc_val: float, decay_pct: float, training_time_sec: float,
                        hiperparametros: dict) -> dict:
    """Construye el dict de metadata con el formato exacto pedido:
    ml_name, performance, hyperparameters, library_versions, timestamp.
    """
    return {
        "ml_name": nombre_modelo,
        "performance": {
            "auc_train": round(auc_train, 4),
            "auc_test": round(auc_test, 4),
            "auc_val": round(auc_val, 4),
            "decay_pct": round(decay_pct, 2),
            "training_time_sec": round(training_time_sec, 2),
        },
        "hyperparameters": hiperparametros,
        "library_versions": versiones_librerias(),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


def guardar_modelo_y_metadata(carpeta_corrida: str, nombre_experimento: str,
                               modelo, metadata: dict) -> None:
    """Guarda <modelo>_model.pkl y <modelo>_metadata.json en la carpeta de la corrida."""
    ruta_pkl = os.path.join(carpeta_corrida, f"{nombre_experimento}_model.pkl")
    ruta_json = os.path.join(carpeta_corrida, f"{nombre_experimento}_metadata.json")

    joblib.dump(modelo, ruta_pkl)

    with open(ruta_json, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)


def guardar_resumen_corrida(carpeta_corrida: str, resumen: dict) -> None:
    """Guarda results_summary.json con el ranking completo y el modelo campeon.

    (En la P1 este archivo se llamaba resumen_corrida.json; la P2 pide el
    nombre results_summary.json -- se renombra aqui para mantener un solo
    lugar de verdad sobre el nombre del archivo.)
    """
    ruta = os.path.join(carpeta_corrida, "results_summary.json")
    with open(ruta, "w", encoding="utf-8") as f:
        json.dump(resumen, f, indent=2, ensure_ascii=False, default=str)
    print(f"[registry] Resumen de la corrida guardado en: {ruta}")


def guardar_backtesting_results(carpeta_corrida: str, backtesting_dict: dict) -> None:
    """Guarda backtesting_results.json (NUEVO en la P2): las olas walk-forward
    del modelo campeon final, con AUC/decay/n_obs/PSI por mes."""
    ruta = os.path.join(carpeta_corrida, "backtesting_results.json")
    with open(ruta, "w", encoding="utf-8") as f:
        json.dump(backtesting_dict, f, indent=2, ensure_ascii=False, default=str)
    print(f"[registry] Resultados de backtesting guardados en: {ruta}")


# ===========================================================================
# PARCIAL - 1. Trazabilidad (Seccion 5.2)
# ===========================================================================
def hash_archivo(ruta: str) -> str:
    """SHA-256 de un archivo, leido por bloques (sirve para archivos grandes)."""
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        for bloque in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(bloque)
    return h.hexdigest()


def hash_datos(input_path: str) -> dict:
    """Hash de cada CSV de entrada y un hash global que los combina.

    Retorna {"data_hash": <global>, "archivos": {nombre: hash}}.
    Si cambia una sola fila de una particion, cambia el data_hash.
    """
    if os.path.isdir(input_path):
        archivos = sorted(glob.glob(os.path.join(input_path, "*.csv")))
    else:
        archivos = [input_path]
    por_archivo = {os.path.basename(a): hash_archivo(a) for a in archivos}
    global_ = hashlib.sha256(json.dumps(por_archivo, sort_keys=True).encode()).hexdigest()
    return {"data_hash": global_, "archivos": por_archivo}


def hash_config(config: dict) -> str:
    """Hash de la configuracion EFECTIVA de la corrida (config.yaml + cambios
    pasados al lanzar el DAG). Se calcula sobre el YAML ordenado, asi el
    orden de las claves o los comentarios no cambian el hash."""
    texto = yaml.safe_dump(config, sort_keys=True, allow_unicode=True)
    return hashlib.sha256(texto.encode()).hexdigest()


def info_git(raiz: str = ".") -> dict:
    """Commit y rama actuales leyendo la carpeta .git (no necesita tener el
    programa git instalado dentro del contenedor). Las variables de entorno
    GIT_COMMIT / GIT_BRANCH, si existen, tienen prioridad."""
    commit, rama = os.environ.get("GIT_COMMIT"), os.environ.get("GIT_BRANCH")
    git_dir = os.path.join(raiz, ".git")
    try:
        with open(os.path.join(git_dir, "HEAD"), encoding="utf-8") as f:
            head = f.read().strip()
        if head.startswith("ref:"):
            ref = head.split(" ", 1)[1]
            rama = rama or ref.replace("refs/heads/", "")
            ruta_ref = os.path.join(git_dir, *ref.split("/"))
            if os.path.exists(ruta_ref):
                with open(ruta_ref, encoding="utf-8") as f:
                    commit = commit or f.read().strip()
            else:  # la referencia puede estar "empaquetada" en packed-refs
                with open(os.path.join(git_dir, "packed-refs"), encoding="utf-8") as f:
                    for linea in f:
                        if linea.strip().endswith(ref):
                            commit = commit or linea.split()[0]
        else:
            commit, rama = commit or head, rama or "HEAD-desacoplado"
    except OSError:
        pass
    return {"git_commit": commit or "desconocido", "git_branch": rama or "desconocida"}


def imagen_docker() -> str:
    """Nombre:tag de la imagen Docker (la define docker-compose.yml en la
    variable DOCKER_IMAGE). Fuera de Docker se indica explicitamente."""
    return os.environ.get("DOCKER_IMAGE", "sin_docker (python main.py local)")


def guardar_json(ruta: str, datos) -> None:
    """Escribe un JSON de forma atomica (temporal + renombrar)."""
    with open(ruta + ".tmp", "w", encoding="utf-8") as f:
        json.dump(datos, f, indent=2, ensure_ascii=False, default=str)
    os.replace(ruta + ".tmp", ruta)


def leer_json(ruta: str, por_defecto=None):
    """Lee un JSON; si no existe devuelve por_defecto."""
    if not os.path.exists(ruta):
        return por_defecto
    with open(ruta, encoding="utf-8") as f:
        return json.load(f)


# ===========================================================================
# PARCIAL - 2/3. Tracking en MLflow con idempotencia
# ===========================================================================
def conectar_mlflow(config: dict):
    """Configura MLflow y devuelve (client, experiment_id).

    La URI del servidor sale de la variable de entorno MLFLOW_TRACKING_URI
    (docker-compose la define como http://mlflow:5000) o, si no existe, de
    config.yaml -> tracking.tracking_uri (para correr python main.py).
    """
    import mlflow
    from mlflow.tracking import MlflowClient

    uri = os.environ.get("MLFLOW_TRACKING_URI") or config["tracking"]["tracking_uri"]
    if uri.startswith("http") and not os.environ.get("MLFLOW_TRACKING_URI") and not _servidor_responde(uri):
        # Solo pasa con "python main.py" sin haber levantado docker compose:
        # se registra en una base local para que la ejecucion local siga funcionando.
        respaldo = config["tracking"]["tracking_uri_respaldo"]
        print(f"[registry] AVISO: no hay servidor MLflow en {uri}; se usa el respaldo local {respaldo}")
        uri = respaldo
    mlflow.set_tracking_uri(uri)
    nombre_exp = config["tracking"]["experiment_name"]
    exp = mlflow.set_experiment(nombre_exp)
    return MlflowClient(), exp.experiment_id


_CACHE_SERVIDOR = {}


def _servidor_responde(uri: str) -> bool:
    """True si el servidor de MLflow responde (se revisa una sola vez)."""
    if uri not in _CACHE_SERVIDOR:
        import requests
        try:
            _CACHE_SERVIDOR[uri] = requests.get(uri.rstrip("/") + "/health", timeout=3).ok
        except Exception:
            _CACHE_SERVIDOR[uri] = False
    return _CACHE_SERVIDOR[uri]


def _filtro_tags(tags: dict) -> str:
    return " and ".join(f"tags.{k} = '{v}'" for k, v in tags.items())


def buscar_run(client, experiment_id: str, tags_busqueda: dict):
    """Busca un run activo (no borrado) que tenga EXACTAMENTE estos tags."""
    runs = client.search_runs([experiment_id], filter_string=_filtro_tags(tags_busqueda),
                              max_results=5, order_by=["attributes.start_time ASC"])
    return runs[0] if runs else None


def obtener_o_crear_run(client, experiment_id: str, tags_busqueda: dict, nombre_run: str,
                        tags_extra: dict = None, parent_run_id: str = None):
    """Mecanismo de idempotencia (D7).

    Retorna (run, ya_existia_terminado):
      - Si ya hay un run con estos tags y esta FINISHED -> se devuelve ese y
        ya_existia_terminado=True (quien llama no vuelve a registrar nada).
      - Si hay uno pero quedo a medias (la tarea fallo) -> se borra y se crea
        uno nuevo, para no dejar runs incompletos duplicados.
      - Si no hay -> se crea.
    """
    existente = buscar_run(client, experiment_id, tags_busqueda)
    if existente is not None:
        if existente.info.status == "FINISHED":
            return existente, True
        client.delete_run(existente.info.run_id)

    tags = {**tags_busqueda, **(tags_extra or {})}
    if parent_run_id:
        tags["mlflow.parentRunId"] = parent_run_id
    run = client.create_run(experiment_id, run_name=nombre_run, tags=tags)
    return run, False


def _limpiar_nombre(nombre: str) -> str:
    """MLflow solo acepta letras, numeros y _ - . espacio / en nombres de metricas."""
    return "".join(c if c.isalnum() or c in "_-./ " else "_" for c in str(nombre))


def log_params(client, run_id: str, params: dict) -> None:
    """Registra parametros (MLflow limita el largo de cada valor a 6000 caracteres)."""
    from mlflow.entities import Param
    lista = [Param(_limpiar_nombre(k), str(v)[:6000]) for k, v in params.items()]
    for i in range(0, len(lista), 100):
        client.log_batch(run_id, params=lista[i:i + 100])


def log_metrics(client, run_id: str, metricas: dict, step: int = 0) -> None:
    """Registra metricas numericas (ignora las que son None o no numericas)."""
    import time as _time
    from mlflow.entities import Metric
    ts = int(_time.time() * 1000)
    lista = [Metric(_limpiar_nombre(k), float(v), ts, step) for k, v in metricas.items()
             if isinstance(v, (int, float)) and not isinstance(v, bool) and v == v]
    for i in range(0, len(lista), 500):
        client.log_batch(run_id, metrics=lista[i:i + 500])


def log_tags(client, run_id: str, tags: dict) -> None:
    for k, v in tags.items():
        client.set_tag(run_id, k, str(v))


def log_artefacto(client, run_id: str, ruta: str, carpeta: str = None) -> None:
    """Sube un archivo como artefacto del run (si existe)."""
    if ruta and os.path.exists(ruta):
        client.log_artifact(run_id, ruta, artifact_path=carpeta)


def log_modelo_sklearn(run_id: str, modelo, nombre: str = "modelo") -> str:
    """Guarda el modelo serializado en MLflow (formato sklearn) dentro del
    run indicado. Retorna la URI del modelo (para el Model Registry)."""
    import mlflow
    import mlflow.sklearn
    with mlflow.start_run(run_id=run_id):
        info = mlflow.sklearn.log_model(modelo, name=nombre)
    return info.model_uri


def terminar_run(client, run_id: str, estado: str = "FINISHED") -> None:
    client.set_terminated(run_id, status=estado)


# ===========================================================================
# PARCIAL - 4. Model Registry: champion / challenger (Seccion 5.3, punto extra)
# ===========================================================================
def promover_en_model_registry(client, config: dict, model_uri: str, corrida_id: str,
                               metricas: dict, run_id: str, puede_promover: bool) -> dict:
    """Registra al campeon de la corrida como nueva version del modelo y
    decide su alias.

    Criterio (documentado en D8 / README):
      - Si no hay "champion" vigente -> la nueva version es champion.
      - Si hay uno, el nuevo modelo (challenger) lo reemplaza solo si su lift
        promedio en backtesting supera al del champion en al menos
        tracking.margen_promocion_pct % Y fue estable (estado OK en GetBestModel).
      - Si no lo supera, queda con alias "challenger" (para revisarlo).
    Idempotente: si esta corrida ya registro una version, no registra otra.
    """
    import mlflow

    nombre = config["tracking"]["registered_model_name"]
    margen = config["tracking"].get("margen_promocion_pct", 2.0)

    try:
        client.get_registered_model(nombre)
    except Exception:
        client.create_registered_model(
            nombre, description="Campeon del AutoML de propension de venta financiera (MCD-402).")

    # Idempotencia: ¿esta corrida ya registro una version?
    for v in client.search_model_versions(f"name = '{nombre}'"):
        if v.tags.get("corrida_id") == corrida_id:
            return {"accion": "ya_registrado", "version": v.version,
                    "alias": v.aliases if hasattr(v, "aliases") else None}

    version = mlflow.register_model(model_uri, nombre, tags={
        "corrida_id": corrida_id, "run_id": run_id,
        **{k: str(v) for k, v in metricas.items()},
    })
    nueva = version.version

    lift_nuevo = metricas.get("bt_lift_prom")
    try:
        champion = client.get_model_version_by_alias(nombre, "champion")
    except Exception:
        champion = None

    if not puede_promover:
        accion = "challenger"
        motivo = "GetBestModel marco el modelo con alerta (no estable); no se promueve."
    elif champion is None:
        accion = "champion"
        motivo = "No habia champion vigente."
    else:
        lift_champ = float(champion.tags.get("bt_lift_prom", "nan"))
        umbral = lift_champ * (1 + margen / 100)
        if lift_nuevo is not None and lift_nuevo >= umbral:
            accion = "champion"
            motivo = (f"Lift {lift_nuevo} >= {round(umbral, 4)} (champion v{champion.version} "
                      f"lift {lift_champ} + {margen}% de margen).")
        else:
            accion = "challenger"
            motivo = (f"Lift {lift_nuevo} < {round(umbral, 4)} (champion v{champion.version} "
                      f"lift {lift_champ} + {margen}% de margen); se mantiene el champion.")

    client.set_registered_model_alias(nombre, accion, nueva)
    client.set_model_version_tag(nombre, nueva, "decision", accion)
    client.set_model_version_tag(nombre, nueva, "motivo_decision", motivo)
    return {"accion": accion, "version": nueva, "motivo": motivo,
            "champion_previo": champion.version if champion else None}
