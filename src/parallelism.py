"""
parallelism.py
===============
Orquesta el entrenamiento paralelo de TODOS los modelos del diccionario y
mide el speed-up real contra el mismo trabajo corriendo en un for-loop
secuencial estandar.

Backend elegido: joblib.Parallel + delayed, con n_jobs configurable desde
config.yaml (parallelism.n_jobs: -1 usa todos los nucleos disponibles).
Se eligio joblib sobre ProcessPoolExecutor/multiprocessing.Pool porque es
el recomendado por el enunciado para trabajar con scikit-learn, y porque
serializa (pickling) los objetos de scikit-learn/XGBoost/LightGBM/CatBoost
de forma mas robusta que multiprocessing puro.

IMPORTANTE: el paralelismo esta orquestado explicitamente AQUI, a nivel de
"un proceso de Python por experimento". No se usa n_jobs interno de
scikit-learn como sustituto (de hecho, cuando corre en paralelo aqui, cada
modelo individual se fuerza a n_jobs=1 internamente para no competir por
los mismos nucleos -- ver _forzar_n_jobs_interno_a_uno).
"""

import os
import time

from joblib import Parallel, delayed

from src.training import entrenar_experimento, entrenar_todos_secuencial


# Claves de config.yaml / preprocessing que SI usa el entrenamiento. Se envia solo
# esto a cada worker: df_futuro_raw (millones de filas crudas del backtesting),
# encoders y scaler NO hacen falta para entrenar y encarecian la serializacion.
CLAVES_ENTRENAMIENTO = (
    "X_train", "X_test", "X_val",
    "X_train_norm", "X_test_norm", "X_val_norm",
    "y_train", "y_test", "y_val",
)


def solo_datos_entrenamiento(data: dict) -> dict:
    """Devuelve un dict liviano con unicamente los arrays que usa
    training.entrenar_experimento() (ver CLAVES_ENTRENAMIENTO)."""
    return {k: data[k] for k in CLAVES_ENTRENAMIENTO}


def _forzar_n_jobs_interno_a_uno(experimento: dict) -> dict:
    """Limita a UN hilo el paralelismo interno de cada modelo.

    Cubre tanto n_jobs (XGBoost, LightGBM, RandomForest, KNN) como
    thread_count (CatBoost, que NO usa n_jobs y por defecto usa todos los
    hilos). Asi el unico paralelismo es el de parallelism.py (un proceso por
    experimento) y no hay 16 procesos x 16 hilos compitiendo por los nucleos.
    Se aplica por igual a la corrida secuencial y a la paralela del benchmark.
    """
    exp = dict(experimento)
    hp = dict(exp["hiperparametros"])
    if "n_jobs" in hp:
        hp["n_jobs"] = 1
    if exp.get("nombre_modelo") == "CatBoost":
        hp["thread_count"] = 1
    exp["hiperparametros"] = hp
    return exp


def contar_workers(n_jobs: int) -> int:
    """Traduce n_jobs (puede ser -1) al numero real de workers que se usaran."""
    cpus = os.cpu_count() or 1
    if n_jobs is None or n_jobs == -1:
        return cpus
    return max(1, min(n_jobs, cpus))


def entrenar_en_paralelo(experimentos: list, data: dict, n_jobs: int) -> tuple:
    """Entrena todos los experimentos en paralelo con joblib.Parallel
    (un proceso por experimento, backend loky).

    Cada modelo corre con 1 hilo interno (ver _forzar_n_jobs_interno_a_uno) y
    cada worker recibe solo los arrays de entrenamiento.

    Retorna
    -------
    (resultados, tiempo_segundos)
    """
    experimentos_ajustados = [_forzar_n_jobs_interno_a_uno(e) for e in experimentos]
    data_liviana = solo_datos_entrenamiento(data)

    inicio = time.time()
    resultados = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(entrenar_experimento)(exp, data_liviana) for exp in experimentos_ajustados
    )
    tiempo = time.time() - inicio
    return resultados, tiempo


def resumir_concurrencia(resultados: list) -> dict:
    """Evidencia de paralelismo a partir de la traza de cada experimento:
    cuantos procesos (PID) distintos se usaron y cuantos experimentos
    estuvieron entrenando AL MISMO TIEMPO como maximo."""
    intervalos = [(r["metadata"]["t_inicio_epoch"], r["metadata"]["t_fin_epoch"]) for r in resultados]
    eventos = sorted([(i, 1) for i, _ in intervalos] + [(f, -1) for _, f in intervalos])
    activos = maximo = 0
    for _, delta in eventos:
        activos += delta
        maximo = max(maximo, activos)
    return {
        "procesos_distintos_usados": len({r["metadata"]["worker_pid"] for r in resultados}),
        "max_experimentos_simultaneos": maximo,
    }


def ejecutar_con_benchmark(experimentos: list, data: dict, config: dict) -> tuple:
    """Entrena todos los experimentos y mide el speed-up real del paralelismo.

    Corre el MISMO conjunto de experimentos dos veces, con la MISMA
    configuracion interna (1 hilo por modelo, ver _forzar_n_jobs_interno_a_uno)
    y los mismos datos: una vez en un for-loop secuencial (T_secuencial) y una
    vez con joblib.Parallel (T_paralelo). Asi la unica diferencia es el
    paralelismo ENTRE experimentos y el speed-up es comparable. Los resultados
    que usa el resto del pipeline son los de la corrida paralela.

    Opcional: config['parallelism']['medir_referencia_multinucleo'] = true mide
    ademas un secuencial "estilo P1" (modelos con todos sus hilos), solo como
    referencia informativa (t_sequential_multithread_s).

    Retorna
    -------
    (resultados, info_parallelism) con backend, n_workers, t_sequential_s,
    t_parallel_s, speed_up y la evidencia de concurrencia.
    """
    par_cfg = config.get("parallelism", {})
    n_jobs = par_cfg.get("n_jobs", -1)
    n_workers = contar_workers(n_jobs)
    data_liviana = solo_datos_entrenamiento(data)
    exp_1hilo = [_forzar_n_jobs_interno_a_uno(e) for e in experimentos]

    t_multinucleo = None
    if par_cfg.get("medir_referencia_multinucleo", False):
        print(f"\n[parallelism] Referencia estilo P1 (secuencial, cada modelo con todos sus hilos)...")
        _, t_multinucleo = _medir_secuencial(experimentos, data_liviana)

    print(f"\n[parallelism] Midiendo T_secuencial (for-loop, 1 hilo por modelo, "
          f"{len(experimentos)} experimentos)...")
    _, t_secuencial = _medir_secuencial(exp_1hilo, data_liviana)

    print(f"[parallelism] Midiendo T_paralelo (joblib.Parallel, n_jobs={n_jobs} "
          f"-> {n_workers} workers, 1 hilo por modelo)...")
    resultados, t_paralelo = entrenar_en_paralelo(experimentos, data, n_jobs)

    speed_up = (t_secuencial / t_paralelo) if t_paralelo > 0 else float("nan")

    info_parallelism = {
        "backend": "joblib",
        "n_workers": n_workers,
        "t_sequential_s": round(t_secuencial, 2),
        "t_parallel_s": round(t_paralelo, 2),
        "speed_up": round(speed_up, 4),
        "hilos_por_modelo": 1,
        **resumir_concurrencia(resultados),
    }
    if t_multinucleo is not None:
        info_parallelism["t_sequential_multithread_s"] = round(t_multinucleo, 2)

    print(f"[parallelism] T_secuencial={t_secuencial:.2f}s | T_paralelo={t_paralelo:.2f}s "
          f"| speed_up={speed_up:.2f}x")
    print(f"[parallelism] Evidencia: {info_parallelism['procesos_distintos_usados']} procesos distintos, "
          f"hasta {info_parallelism['max_experimentos_simultaneos']} experimentos entrenando a la vez.")
    if speed_up < 1.0:
        print("[parallelism] AVISO: speed_up < 1.0 -- el costo de crear procesos y copiar "
              "datos supera la ganancia. Prueba con parallelism.n_jobs mas bajo (ej. 4).")

    return resultados, info_parallelism


def _medir_secuencial(experimentos: list, data: dict) -> tuple:
    """Entrena los experimentos uno tras otro y devuelve (resultados, segundos)."""
    inicio = time.time()
    resultados = entrenar_todos_secuencial(experimentos, data)
    tiempo = time.time() - inicio
    return resultados, tiempo
