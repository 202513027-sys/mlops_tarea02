"""
training.py
============
Construye la lista de experimentos (modelo x configuracion), aplicando los
resultados de hpo.py (cuando existen) y la configuracion de GPU/CPU, entrena
cada experimento y calcula sus metricas (src/evaluation.py).

A diferencia de la P1, este modulo YA NO escribe los .pkl/.json en disco por
su cuenta -- eso se centraliza en main.py (via registry.py) una vez que cada
experimento ya tiene TODOS sus campos de metadata completos (performance,
hpo, hardware, y mas tarde parallelism/backtesting), para no tener que
reescribir el mismo archivo varias veces.
"""

import shutil
import subprocess
import time

from src import evaluation
from src import models as models_module
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

    usar_gpu = resolver_uso_gpu(config)
    for exp in experimentos:
        gpu_params = GPU_PARAMS_POR_MODELO.get(exp["nombre_modelo"])
        if usar_gpu and gpu_params:
            exp["hiperparametros"] = {**exp["hiperparametros"], **gpu_params}
            exp["hardware"] = "gpu"
        else:
            exp["hardware"] = "cpu"

    return experimentos


def entrenar_experimento(experimento: dict, data: dict) -> dict:
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

    print(f"[training] Entrenando {nombre_experimento} ...")

    if usa_norm:
        X_train, X_test, X_val = data["X_train_norm"], data["X_test_norm"], data["X_val_norm"]
    else:
        X_train, X_test, X_val = data["X_train"], data["X_test"], data["X_val"]
    y_train, y_test, y_val = data["y_train"], data["y_test"], data["y_val"]

    hardware_usado = experimento.get("hardware", "cpu")
    nota_fallback = None

    inicio = time.time()
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
    auc_test = evaluation.calcular_auc(modelo, X_test, y_test)
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
    if nota_fallback:
        metadata["gpu_fallback"] = nota_fallback
    metadata["hpo"] = experimento.get("hpo")  # None si el modelo no paso por HPO
    metadata["parallelism"] = None  # se completa en main.py tras medir speed-up
    metadata["backtesting"] = None  # se completa en main.py solo para los candidatos evaluados

    return {"modelo": modelo, "metadata": metadata, "necesita_normalizacion": usa_norm}


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
