"""
training.py
============
Construye la lista de experimentos (modelo x configuracion), aplicando los
resultados de hpo.py (cuando existen) y la configuracion de GPU/CPU, entrena
cada experimento y calcula sus metricas (src/evaluation.py).

Cambios del parcial (metadata extendida):
  - Cada entrenamiento se mide con monitoring.MonitorRecursos (CPU, RAM,
    GPU, duracion) -> metadata["compute"].
  - Se calcula la metrica de negocio en test -> metadata["negocio_test"].
  - entrenar_y_guardar() guarda cada modelo en cuanto termina, en la carpeta
    de intermedios de la corrida. Si la tarea del DAG falla a la mitad y
    Airflow la reintenta, los modelos ya guardados NO se vuelven a entrenar
    (parte del mecanismo de idempotencia, D7).
  - config.yaml -> entrenamiento.excluir_experimentos permite dejar fuera
    experimentos (ver D5/D2 en docs/decisiones_diseno.md).
"""

import json
import os
import shutil
import subprocess
import time

import joblib
import pandas as pd

from src import evaluation
from src import models as models_module
from src import monitoring
from src import registry


# Parametros especificos de cada libreria para forzar entrenamiento en GPU.
# Solo estos 3 modelos (los "base" del curso) soportan GPU de fabrica.
GPU_PARAMS_POR_MODELO = {
    "XGBoost": {"device": "cuda"},
    "LightGBM": {"device": "gpu"},
    "CatBoost": {"task_type": "GPU"},
}


def detectar_gpu() -> bool:
    """Detecta si la maquina tiene una GPU NVIDIA disponible (via nvidia-smi).

    No asume ninguna libreria de GPU instalada: si 'nvidia-smi' no existe o
    falla, se asume que no hay GPU y el pipeline sigue en CPU sin errores,
    tal como pide el enunciado.
    """
    if shutil.which("nvidia-smi") is None:
        return False
    try:
        resultado = subprocess.run(
            ["nvidia-smi"], capture_output=True, timeout=5, text=True
        )
        return resultado.returncode == 0
    except Exception:
        return False


def resolver_uso_gpu(config: dict) -> bool:
    """Decide si esta corrida usa GPU, combinando lo que pide config.yaml
    con lo que la maquina realmente tiene disponible.

    config['gpu']['use_gpu'] puede ser "auto" (usa GPU si hay), true
    (fuerza intento de GPU, pero cae a CPU si no hay) o false (CPU siempre).
    """
    quiere_gpu = config.get("gpu", {}).get("use_gpu", "auto")
    hay_gpu = detectar_gpu()

    if quiere_gpu is False or str(quiere_gpu).lower() == "false":
        print("[training] GPU desactivada por configuracion (gpu.use_gpu: false). Usando CPU.")
        return False

    if hay_gpu:
        print("[training] GPU NVIDIA detectada. Los modelos XGBoost/LightGBM/CatBoost "
              "se entrenaran con device/task_type en GPU.")
        return True

    if str(quiere_gpu).lower() == "true":
        print("[training] gpu.use_gpu=true pero no se detecto GPU en esta maquina. "
              "Se continua en CPU sin error (requisito del enunciado).")
    else:
        print("[training] No se detecto GPU. Pipeline continua en CPU.")
    return False


def construir_lista_experimentos(config: dict, hpo_results: dict = None) -> list:
    """Construye la lista final de experimentos a entrenar.

    - Parte de MODEL_REGISTRY (models.py) con sus configs manuales.
    - Si hpo_results trae resultados para un modelo (ver hpo.py), sus
      2 configs manuales se REEMPLAZAN por una sola config optimizada
      (el mejor_params encontrado), y ese experimento guarda el campo
      "hpo" en su metadata.
    - Inyecta el parametro de GPU/CPU segun resolver_uso_gpu().
    """
    random_state = config["split"]["random_state"]
    experimentos = models_module.listar_experimentos(random_state=random_state)
    hpo_results = hpo_results or {}

    if hpo_results:
        experimentos = [e for e in experimentos if e["nombre_modelo"] not in hpo_results]
        for nombre_modelo, hpo_info in hpo_results.items():
            info_modelo = models_module.MODEL_REGISTRY[nombre_modelo]
            # Se parte de la primera config manual (trae kwargs fijos como
            # eval_metric/n_jobs/random_state) y se sobreescribe con los
            # mejores hiperparametros encontrados por hpo.py.
            hp_base = dict(info_modelo["configs"][0])
            hp_base.update(hpo_info["best_params"])
            experimentos.append({
                "nombre_experimento": f"{nombre_modelo}_hpo",
                "nombre_modelo": nombre_modelo,
                "categoria": info_modelo["categoria"],
                "es_base": info_modelo["es_base"],
                "clase": info_modelo["clase"],
                "necesita_normalizacion": info_modelo["necesita_normalizacion"],
                "hiperparametros": hp_base,
                "hpo": hpo_info,
            })

    # Parcial: experimentos excluidos desde config.yaml (sin tocar codigo)
    excluir = set(config.get("entrenamiento", {}).get("excluir_experimentos") or [])
    if excluir:
        experimentos = [e for e in experimentos if e["nombre_experimento"] not in excluir]
        print(f"[training] Excluidos por config.yaml: {sorted(excluir)}")

    usar_gpu = resolver_uso_gpu(config)
    for exp in experimentos:
        gpu_params = GPU_PARAMS_POR_MODELO.get(exp["nombre_modelo"])
        if usar_gpu and gpu_params:
            exp["hiperparametros"] = {**exp["hiperparametros"], **gpu_params}
            exp["hardware"] = "gpu"
        else:
            exp["hardware"] = "cpu"

    return experimentos


def _normalizados(data: dict) -> tuple:
    """Devuelve (X_train_norm, X_test_norm, X_val_norm). Si no vienen ya
    calculados, se calculan aqui con el scaler ajustado en preprocessing
    (asi a cada worker paralelo se le envia la mitad de datos)."""
    if "X_train_norm" in data:
        return data["X_train_norm"], data["X_test_norm"], data["X_val_norm"]
    scaler = data["scaler"]
    return tuple(
        pd.DataFrame(scaler.transform(data[k]), columns=data[k].columns, index=data[k].index)
        for k in ("X_train", "X_test", "X_val")
    )


def entrenar_experimento(experimento: dict, data: dict, intervalo_monitoreo_s: float = 1.0,
                         top_pct: float = 10.0) -> dict:
    """Entrena un modelo y calcula sus metricas.

    Usa las versiones normalizadas de los datos si el modelo lo requiere
    (ver 'necesita_normalizacion' en models.py), y las versiones sin
    normalizar en caso contrario (arboles y ensambles no lo necesitan).

    Retorna
    -------
    dict con el modelo entrenado y su metadata (formato ml_name/performance/...).
    """
    nombre_experimento = experimento["nombre_experimento"]
    clase = experimento["clase"]
    hp = experimento["hiperparametros"]
    usa_norm = experimento["necesita_normalizacion"]

    pid = os.getpid()
    print(f"[training] Entrenando {nombre_experimento} (proceso PID {pid}) ...")

    if usa_norm:
        X_train, X_test, X_val = _normalizados(data)
    else:
        X_train, X_test, X_val = data["X_train"], data["X_test"], data["X_val"]
    y_train, y_test, y_val = data["y_train"], data["y_test"], data["y_val"]

    hardware_usado = experimento.get("hardware", "cpu")
    nota_fallback = None

    # Parcial: el entrenamiento se mide con el monitor de recursos (Seccion 9)
    n_hilos = hp.get("n_jobs", hp.get("thread_count"))
    monitor = MonitorRecursosSeguro(nombre_experimento, intervalo_monitoreo_s, n_hilos)
    monitor.__enter__()

    inicio = time.time()
    t_inicio_epoch = inicio
    try:
        modelo = clase(**hp)
        modelo.fit(X_train, y_train)
    except Exception as error_gpu:
        # Si el modelo fallo estando en GPU (ej. LightGBM instalado sin soporte
        # GPU, memoria insuficiente), se reintenta en CPU en vez de detener todo
        # el pipeline. Si ya estaba en CPU, el error es real y se propaga.
        if hardware_usado != "gpu":
            raise
        gpu_keys = GPU_PARAMS_POR_MODELO.get(experimento["nombre_modelo"], {}).keys()
        hp = {k: v for k, v in hp.items() if k not in gpu_keys}
        print(f"[training] AVISO: {nombre_experimento} fallo en GPU "
              f"({type(error_gpu).__name__}); reintentando en CPU.")
        nota_fallback = f"GPU fallo ({type(error_gpu).__name__}); se entreno en CPU."
        hardware_usado = "cpu"
        inicio = time.time()
        modelo = clase(**hp)
        modelo.fit(X_train, y_train)
    tiempo_entrenamiento = time.time() - inicio
    monitor.__exit__(None, None, None)

    # XGBoost no lanza error si no halla GPU: pasa a CPU solo, con una advertencia.
    # Se lee el dispositivo real del modelo para no reportar "gpu" falsamente.
    if hardware_usado == "gpu" and experimento["nombre_modelo"] == "XGBoost":
        try:
            import json
            cfg = json.loads(modelo.get_booster().save_config())
            if cfg["learner"]["generic_param"].get("device", "cpu") == "cpu":
                hardware_usado = "cpu"
                nota_fallback = "XGBoost no encontro GPU utilizable; se entreno en CPU."
        except Exception:
            pass

    auc_train = evaluation.calcular_auc(modelo, X_train, y_train)
    probas_test = modelo.predict_proba(X_test)[:, 1]
    auc_test = evaluation.roc_auc_score(y_test, probas_test)
    auc_val = evaluation.calcular_auc(modelo, X_val, y_val)
    decay_pct = evaluation.calcular_decay(auc_train, auc_test)

    metadata = registry.construir_metadata(
        nombre_modelo=nombre_experimento,
        auc_train=auc_train, auc_test=auc_test, auc_val=auc_val,
        decay_pct=decay_pct, training_time_sec=tiempo_entrenamiento,
        hiperparametros={k: v for k, v in hp.items()},
    )
    metadata["categoria"] = experimento["categoria"]
    metadata["es_base"] = experimento["es_base"]
    metadata["hardware"] = hardware_usado
    # Traza para verificar el paralelismo: PID del proceso y ventana de tiempo.
    metadata["worker_pid"] = pid
    metadata["t_inicio_epoch"] = round(t_inicio_epoch, 3)
    metadata["t_fin_epoch"] = round(time.time(), 3)
    if nota_fallback:
        metadata["gpu_fallback"] = nota_fallback
    # Parcial: metrica de negocio en test y logs de computo del entrenamiento
    metadata["negocio_test"] = evaluation.metricas_negocio(y_test, probas_test, top_pct)
    metadata["compute"] = monitor.resultado
    metadata["necesita_normalizacion"] = usa_norm
    hpo_info = experimento.get("hpo")
    if hpo_info:  # la lista completa de trials va aparte (hpo_results.json), no en cada modelo
        hpo_info = {k: v for k, v in hpo_info.items() if k != "trials"}
    metadata["hpo"] = hpo_info  # None si el modelo no paso por HPO
    metadata["parallelism"] = None  # se completa en main.py tras medir speed-up
    metadata["backtesting"] = None  # se completa en main.py solo para los candidatos evaluados

    return {"modelo": modelo, "metadata": metadata, "necesita_normalizacion": usa_norm,
            "probas_test": probas_test}


class MonitorRecursosSeguro(monitoring.MonitorRecursos):
    """Igual que MonitorRecursos, pero si el monitoreo falla por algun motivo
    raro del sistema, el entrenamiento sigue (medir nunca debe tumbar el pipeline)."""

    def __exit__(self, *exc):
        try:
            return super().__exit__(*exc)
        except Exception as e:
            self.resultado = {"nombre": self.nombre, "error_monitoreo": str(e)}
            return False


def guardar_curva_roc(y_real, probas, nombre: str, ruta_png: str) -> None:
    """Guarda la curva ROC del modelo en test como imagen PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, roc_auc_score

    fpr, tpr, _ = roc_curve(y_real, probas)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, label=f"AUC test = {roc_auc_score(y_real, probas):.4f}")
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Azar")
    ax.set_xlabel("Tasa de falsos positivos")
    ax.set_ylabel("Tasa de verdaderos positivos")
    ax.set_title(f"Curva ROC - {nombre}")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(ruta_png, dpi=90)
    plt.close(fig)


def entrenar_y_guardar(experimento: dict, data: dict, carpeta_modelos: str,
                       intervalo_monitoreo_s: float = 1.0, top_pct: float = 10.0) -> dict:
    """Entrena UN experimento y guarda en carpeta_modelos:
        <nombre>.joblib (modelo), <nombre>.json (metadata), <nombre>_roc.png

    Los archivos se escriben primero con nombre temporal y luego se renombran:
    asi nunca queda un modelo "a medias" que un reintento confunda con uno
    terminado. Retorna solo la metadata (liviana).
    """
    nombre = experimento["nombre_experimento"]
    r = entrenar_experimento(experimento, data, intervalo_monitoreo_s, top_pct)

    ruta_modelo = os.path.join(carpeta_modelos, f"{nombre}.joblib")
    joblib.dump(r["modelo"], ruta_modelo + ".tmp")
    os.replace(ruta_modelo + ".tmp", ruta_modelo)

    meta = r["metadata"]
    if meta.get("compute"):
        meta["compute"]["tamano_modelo_mb"] = monitoring.tamano_archivo_mb(ruta_modelo)

    try:
        guardar_curva_roc(data["y_test"], r["probas_test"], nombre,
                          os.path.join(carpeta_modelos, f"{nombre}_roc.png"))
    except Exception as e:
        print(f"[training] Aviso: no se pudo dibujar la curva ROC de {nombre}: {e}")

    ruta_json = os.path.join(carpeta_modelos, f"{nombre}.json")
    with open(ruta_json + ".tmp", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False, default=str)
    os.replace(ruta_json + ".tmp", ruta_json)
    print(f"[training] {nombre} guardado (AUC test={meta['performance']['auc_test']}, "
          f"lift test={meta['negocio_test']['lift']}).")
    return meta


def entrenar_todos_secuencial(experimentos: list, data: dict) -> list:
    """Entrena todos los experimentos en un for-loop estandar (sin paralelismo).

    Se usa tanto para el modo secuencial normal (P1) como para medir
    T_secuencial dentro del benchmark de parallelism.py.
    """
    return [entrenar_experimento(exp, data) for exp in experimentos]


def construir_tabla_resultados(ranking: list) -> list:
    """Convierte el ranking de GetBestModel() en una lista de filas planas,
    lista para imprimir o exportar a una tabla (usada por main.py y el README).
    """
    filas = []
    for r in ranking:
        p = r["metadata"]["performance"]
        filas.append({
            "modelo": r["metadata"]["ml_name"],
            "categoria": r["metadata"]["categoria"],
            "es_base": r["metadata"]["es_base"],
            "hardware": r["metadata"]["hardware"],
            "auc_train": p["auc_train"],
            "auc_test": p["auc_test"],
            "auc_val": p["auc_val"],
            "decay_pct": p["decay_pct"],
            "tiempo_seg": p["training_time_sec"],
        })
    return filas
