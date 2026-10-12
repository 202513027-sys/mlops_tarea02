"""
monitoring.py  (NUEVO en el parcial)
=====================================
Captura los indicadores operativos de computo que pide la Seccion 9:

  - CPU (promedio y maximo) y numero de nucleos.
  - Memoria RAM (promedio y maximo).
  - GPU y memoria de GPU si hay una NVIDIA visible; si no, se deja escrito
    explicitamente que la corrida fue "solo CPU".
  - Duracion de cada tarea del DAG y de cada entrenamiento.
  - Tamano del modelo serializado (ver tamano_archivo_mb()).

Como se usa (es un "context manager", igual que abrir un archivo con 'with'):

    with MonitorRecursos("entrenar_modelos", intervalo_s=1.0) as mon:
        ... codigo pesado ...
    print(mon.resultado)   # dict con cpu/ram/gpu/duracion

Por dentro, un hilo secundario toma una "foto" de los recursos cada
`intervalo_s` segundos (config.yaml -> monitoreo.intervalo_s) mientras el
codigo principal trabaja. Se mide el proceso actual MAS sus procesos hijos,
porque el entrenamiento en paralelo (joblib) crea procesos hijos.

Herramientas: psutil (CPU/RAM) y pynvml (GPU NVIDIA, opcional).
"""

import json
import os
import platform
import threading
import time
from datetime import datetime

import psutil


# ---------------------------------------------------------------------------
# GPU (opcional): si pynvml no esta instalado o no hay GPU, todo sigue en CPU
# ---------------------------------------------------------------------------
def _abrir_gpu():
    """Devuelve (handle, nombre) de la primera GPU NVIDIA, o (None, None) si no hay."""
    try:
        import pynvml
        pynvml.nvmlInit()
        if pynvml.nvmlDeviceGetCount() == 0:
            return None, None
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        nombre = pynvml.nvmlDeviceGetName(handle)
        if isinstance(nombre, bytes):
            nombre = nombre.decode()
        return handle, nombre
    except Exception:
        return None, None


def _leer_gpu(handle):
    """Lee (uso_gpu_pct, memoria_gpu_mb) de una GPU ya abierta."""
    import pynvml
    uso = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
    mem = pynvml.nvmlDeviceGetMemoryInfo(handle).used / 1024 ** 2
    return float(uso), float(mem)


def info_sistema() -> dict:
    """Describe la maquina donde corre el pipeline (se guarda en compute_logs.json)."""
    _, nombre_gpu = _abrir_gpu()
    return {
        "sistema_operativo": platform.platform(),
        "nucleos_logicos": os.cpu_count(),
        "ram_total_gb": round(psutil.virtual_memory().total / 1024 ** 3, 2),
        "gpu_disponible": nombre_gpu is not None,
        "gpu_nombre": nombre_gpu,
        "modo_computo": "cpu+gpu" if nombre_gpu else "solo_cpu",
        "en_docker": os.path.exists("/.dockerenv"),
    }


# ---------------------------------------------------------------------------
# Monitor de recursos
# ---------------------------------------------------------------------------
class MonitorRecursos:
    """Mide CPU, RAM, GPU y duracion de un bloque de codigo.

    Parametros
    ----------
    nombre : str
        Nombre de lo que se mide (tarea del DAG o experimento).
    intervalo_s : float
        Cada cuantos segundos se toma una muestra (frecuencia de muestreo).
    nucleos_asignados : int, opcional
        Cuantos nucleos/workers se le dieron a este bloque (ej. n_jobs).
    """

    def __init__(self, nombre: str, intervalo_s: float = 1.0, nucleos_asignados: int = None):
        self.nombre = nombre
        self.intervalo_s = max(0.1, float(intervalo_s))
        self.nucleos_asignados = nucleos_asignados
        self.resultado = None
        self._muestras = []
        self._procesos = {}          # pid -> psutil.Process (guarda el estado de cpu_percent)
        self._detener = threading.Event()
        self._hilo = None
        self._gpu_handle, self._gpu_nombre = _abrir_gpu()

    # --- toma una muestra de todos los recursos -----------------------------
    def _procesos_actuales(self):
        """Proceso actual + todos sus hijos (los workers de joblib)."""
        principal = psutil.Process(os.getpid())
        try:
            hijos = principal.children(recursive=True)
        except psutil.Error:
            hijos = []
        vivos = {}
        for p in [principal] + hijos:
            if p.pid not in self._procesos:
                self._procesos[p.pid] = p
                try:
                    p.cpu_percent(None)  # la primera llamada solo "arranca" el contador
                except psutil.Error:
                    continue
            vivos[p.pid] = self._procesos[p.pid]
        return vivos.values()

    def _muestrear(self):
        cpu_proc, ram_mb = 0.0, 0.0
        for p in self._procesos_actuales():
            try:
                cpu_proc += p.cpu_percent(None)
                ram_mb += p.memory_info().rss / 1024 ** 2
            except psutil.Error:
                continue  # el proceso hijo termino entre una linea y otra
        muestra = {
            "cpu_proceso_pct": cpu_proc,                    # 100% = 1 nucleo completo
            "cpu_sistema_pct": psutil.cpu_percent(None),   # 0-100% de toda la maquina
            "ram_mb": ram_mb,
        }
        if self._gpu_handle is not None:
            try:
                muestra["gpu_pct"], muestra["gpu_mem_mb"] = _leer_gpu(self._gpu_handle)
            except Exception:
                pass
        self._muestras.append(muestra)

    def _bucle(self):
        while not self._detener.wait(self.intervalo_s):
            self._muestrear()

    # --- context manager ------------------------------------------------------
    def __enter__(self):
        self._inicio = time.time()
        psutil.cpu_percent(None)
        self._procesos_actuales()
        self._hilo = threading.Thread(target=self._bucle, daemon=True)
        self._hilo.start()
        return self

    def __exit__(self, *exc):
        self._detener.set()
        self._hilo.join(timeout=5)
        self._muestrear()  # al menos una muestra aunque el bloque dure < intervalo_s
        self.resultado = self._resumir(time.time() - self._inicio)
        return False  # no se "traga" errores: si el bloque fallo, el error sigue

    def _resumir(self, duracion_s: float) -> dict:
        def prom(clave):
            vals = [m[clave] for m in self._muestras if clave in m]
            return round(sum(vals) / len(vals), 2) if vals else None

        def maximo(clave):
            vals = [m[clave] for m in self._muestras if clave in m]
            return round(max(vals), 2) if vals else None

        cpu_max = maximo("cpu_proceso_pct") or 0
        resumen = {
            "nombre": self.nombre,
            "inicio": datetime.fromtimestamp(self._inicio).strftime("%Y-%m-%d %H:%M:%S"),
            "duracion_s": round(duracion_s, 2),
            "cpu_pct_prom": prom("cpu_proceso_pct"),
            "cpu_pct_max": maximo("cpu_proceso_pct"),
            "cpu_sistema_pct_prom": prom("cpu_sistema_pct"),
            "nucleos_disponibles": os.cpu_count(),
            "nucleos_asignados": self.nucleos_asignados,
            # cuantos nucleos "llenos" se usaron en el pico (ej. 350% = 3.5 nucleos)
            "nucleos_usados_max": round(cpu_max / 100, 2),
            "ram_mb_prom": prom("ram_mb"),
            "ram_mb_max": maximo("ram_mb"),
            "n_muestras": len(self._muestras),
            "intervalo_muestreo_s": self.intervalo_s,
        }
        if self._gpu_nombre:
            resumen.update({
                "modo_computo": "cpu+gpu",
                "gpu_nombre": self._gpu_nombre,
                "gpu_pct_prom": prom("gpu_pct"),
                "gpu_pct_max": maximo("gpu_pct"),
                "gpu_mem_mb_max": maximo("gpu_mem_mb"),
            })
        else:
            resumen.update({"modo_computo": "solo_cpu", "gpu_nombre": None,
                            "gpu_pct_prom": None, "gpu_pct_max": None, "gpu_mem_mb_max": None})
        return resumen


def tamano_archivo_mb(ruta: str) -> float:
    """Tamano de un archivo en MB (se usa para el modelo serializado)."""
    return round(os.path.getsize(ruta) / 1024 ** 2, 3)


# ---------------------------------------------------------------------------
# compute_logs.json dentro del directorio de la corrida
# ---------------------------------------------------------------------------
def ruta_compute_logs(carpeta_corrida: str) -> str:
    return os.path.join(carpeta_corrida, "compute_logs.json")


def leer_compute_logs(carpeta_corrida: str) -> dict:
    """Lee compute_logs.json (o devuelve la estructura vacia si aun no existe)."""
    ruta = ruta_compute_logs(carpeta_corrida)
    if os.path.exists(ruta):
        with open(ruta, encoding="utf-8") as f:
            return json.load(f)
    return {"sistema": info_sistema(), "tareas": {}, "modelos": {}, "reejecuciones_omitidas": []}


def guardar_compute_logs(carpeta_corrida: str, logs: dict) -> None:
    """Escribe compute_logs.json de forma atomica (archivo temporal + renombrar)."""
    ruta = ruta_compute_logs(carpeta_corrida)
    tmp = ruta + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(logs, f, indent=2, ensure_ascii=False, default=str)
    os.replace(tmp, ruta)


def registrar_en_compute_logs(carpeta_corrida: str, seccion: str, clave: str, datos: dict) -> None:
    """Agrega/actualiza una entrada: seccion = "tareas" o "modelos"."""
    logs = leer_compute_logs(carpeta_corrida)
    logs.setdefault(seccion, {})[clave] = datos
    guardar_compute_logs(carpeta_corrida, logs)
