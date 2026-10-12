# AutoML — Propensión de Venta Financiera (Parcial: orquestación de pipelines)

**MCD-402 Machine Learning Operations · 2026-2**

Este repositorio continúa la Práctica 1 (AutoML) y la Práctica 2 (paralelismo, HPO y backtesting). En el parcial el AutoML pasa a ser un **pipeline de entrenamiento orquestado y reproducible**:

| Pieza nueva | Herramienta | Dónde está |
|---|---|---|
| Orquestación | Apache Airflow 2.10 (DAG manual, `schedule=None`) | `dags/training_dag.py` |
| Gestión de experimentos + Model Registry | MLflow 3.17 | `src/registry.py` |
| Contenerización | Docker + Docker Compose | `Dockerfile`, `docker-compose.yml` |
| Logs de cómputo (CPU, RAM, GPU, tiempos, tamaño) | psutil + pynvml | `src/monitoring.py` |
| HPO con librería de mercado | Optuna (TPE) | `src/hpo.py` |
| Métrica de negocio en `GetBestModel()` | Lift en el top 10 % | `src/evaluation.py` |
| Validación de datos (compuerta del DAG) | pandas | `src/validacion.py` |

Las decisiones de diseño (D1–D9) están justificadas en **[docs/decisiones_diseno.md](docs/decisiones_diseno.md)**.

> La versión entregada en la P2 está etiquetada como `v2.0-p2` (ver sección 12).

---

## 1. Cómo funciona (en una imagen)

```
                    docker compose up -d
 ┌──────────────────────────────────────────────────────────────────────┐
 │  airflow (http://localhost:8080)                                     │
 │                                                                      │
 │  validar_datos → preprocesar_datos → optimizar_hiperparametros →     │
 │  entrenar_modelos → backtesting → seleccionar_campeon →              │
 │  registrar_experimentos → generar_reporte                            │
 │        │                                   │                         │
 │        ▼                                   ▼                         │
 │  data/output/<fecha_hora>/  (JSON)    mlflow (http://localhost:5000) │
 │  data/intermedios/<corrida_id>/       runs, métricas, artefactos,    │
 │  (datos procesados y modelos)         modelos y Model Registry       │
 └──────────────────────────────────────────────────────────────────────┘
```

- Cada **tarea** del DAG llama a una función de `src/pipeline.py`. `python main.py` llama a **las mismas funciones** en el mismo orden, así que el DAG y la ejecución local hacen exactamente lo mismo.
- Entre tareas, por XCom, solo viaja la **ruta de la carpeta de la corrida** y su identificador. Los datos y los modelos van a disco o a MLflow.
- Una **corrida** se identifica por `corrida_id` = hash(configuración + datos + commit de git). Si se vuelve a lanzar el DAG sin cambiar nada, se reconoce la misma corrida y no se duplica nada (idempotencia, sección 8).

## 2. Requisitos

- **Docker Desktop** (en Windows, con WSL 2). Comprobar con `docker --version` y `docker compose version`.
- **RAM:** la tarea más pesada (preprocesar las 10 particiones) usa ~3 GB. Docker con WSL 2 usa por defecto hasta el 50 % de la RAM de la máquina; con 12 GB o más no hay que tocar nada (ver sección 13 si falta memoria).
- **Git**, para clonar el repositorio.
- **Python** solo si se quiere correr `python main.py` fuera de Docker (sección 6).

## 3. Cómo obtener el dataset

El dataset **no está en el repositorio**. Se descarga del Drive del curso:
<https://drive.google.com/drive/folders/1IKpzgvN9g9SxBO-dOGoWHmc_E-worDXU>

Se colocan los 10 CSV **directamente** en `data/input/`:

```
automl_venta_financiera/
└── data/
    └── input/
        ├── p1_extrac.csv
        ├── p2_extrac.csv
        ├── ...
        └── p10_extrac.csv
```

Docker monta la carpeta del proyecto dentro del contenedor (`/opt/proyecto`), así que el pipeline lee `data/input/` sin copiar nada. La primera tarea del DAG verifica que estén las 10 particiones (sección 10).

## 4. Ejecución con Docker (paso a paso)

Todos los comandos se ejecutan desde la carpeta del proyecto (`automl_venta_financiera/`).

### 4.1 Construir las imágenes y levantar los servicios

```bash
docker compose build        # la primera vez tarda ~10 min (descarga Airflow, MLflow y librerías)
docker compose up -d        # levanta postgres, mlflow, init-permisos y airflow
docker compose ps           # esperar a que mlflow y airflow digan "running" / "healthy"
```

| Servicio | Para qué | Dirección |
|---|---|---|
| `airflow` | interfaz web + scheduler | <http://localhost:8080> — usuario **admin**, contraseña **admin** |
| `mlflow` | gestor de experimentos | <http://localhost:5000> |
| `postgres` | base de datos interna de Airflow | (interno) |
| `init-permisos` | da permisos a los volúmenes y termina | (se ejecuta una vez) |

La interfaz de Airflow tarda 1–2 minutos en responder la primera vez.

### 4.2 Ejecutar el DAG

**Opción A, desde la interfaz:** abrir <http://localhost:8080>, buscar el DAG `automl_venta_financiera` y pulsar ▶ **Trigger DAG**. Para cambiar algo solo en esa ejecución, usar **Trigger DAG w/ config**, por ejemplo:

```json
{"paralelo": false}                 // entrenar en secuencial
{"n_jobs": 2}                       // 2 workers en paralelo
{"input_path": "data/muestra"}      // prueba rápida con una muestra (sección 5)
```

**Opción B, desde la terminal** (ejecuta todo el DAG y muestra los logs en pantalla):

```bash
docker compose exec airflow airflow dags test automl_venta_financiera
```

Con las 10 particiones completas el DAG tarda alrededor de 25–40 minutos, según la máquina. Al terminar:

- En MLflow (<http://localhost:5000>), experimento `automl_venta_financiera`, aparece un run **`corrida_<fecha_hora>`** con sus runs hijos: un run por modelo, uno por cada búsqueda de HPO y uno de validación de datos.
- En `data/output/<fecha_hora>/` quedan todos los JSON de la corrida (sección 7), incluido `tablas_readme.md` con las tablas de este README.

### 4.3 Apagar y volver a levantar

```bash
docker compose down         # apaga; los volúmenes (MLflow, Airflow, intermedios) se conservan
docker compose up -d        # vuelve a levantar todo, con los experimentos anteriores
docker compose down -v      # ¡cuidado! borra también los volúmenes (MLflow y Airflow desde cero)
```

### 4.4 Ver logs si algo falla

```bash
docker compose logs -f airflow
docker compose logs -f mlflow
```

## 5. Prueba rápida con una muestra (opcional)

Para comprobar que todo funciona antes de la corrida completa, se puede crear una muestra de 15 mil filas por partición en `data/muestra/` (está en `.gitignore`) y lanzar el DAG con `{"input_path": "data/muestra"}`. Tarda unos 5 minutos.

```bash
docker compose exec airflow python -c "import glob,os,pandas as pd; os.makedirs('data/muestra',exist_ok=True); [pd.read_csv(f,nrows=15000).to_csv('data/muestra/'+os.path.basename(f),index=False) for f in glob.glob('data/input/*.csv')]"
```

## 6. Ejecución local sin Docker (como en la P1/P2)

```bash
pip install -r requirements.txt
python main.py                          # usa config/config.yaml
python main.py --input data/input --config config/config.yaml
python main.py --sequential             # o --parallel, o --n-jobs 2
```

`main.py` ejecuta las mismas 8 tareas del DAG. Si `docker compose` está levantado, los runs van al mismo MLflow (<http://localhost:5000>); si no, se guardan en `mlflow_local.db` (respaldo local, `config.yaml → tracking.tracking_uri_respaldo`), que se puede ver con `mlflow ui --backend-store-uri sqlite:///mlflow_local.db`.

> Airflow no corre de forma nativa en Windows: para el DAG siempre se usa Docker.

## 7. Estructura del proyecto

```
automl_venta_financiera/
├── main.py                       ← ejecución local (mismas tareas que el DAG)
├── Dockerfile                    ← NUEVO: imagen de Airflow + librerías del AutoML
├── docker-compose.yml            ← NUEVO: postgres + mlflow + airflow
├── docker/mlflow/Dockerfile      ← NUEVO: imagen del servidor de MLflow
├── requirements.txt              ← librerías para main.py (local)
├── requirements-docker.txt       ← NUEVO: librerías dentro de Docker (versiones fijas)
├── config/
│   └── config.yaml               ← + entrenamiento, métrica de negocio, HPO, validación,
│                                   tracking (MLflow), monitoreo y airflow
├── dags/
│   └── training_dag.py           ← NUEVO: DAG de Airflow
├── src/
│   ├── pipeline.py               ← NUEVO: una función por tarea del DAG
│   ├── preprocessing.py          ← igual que P2 (+ optimización de memoria)
│   ├── validacion.py             ← NUEVO: compuerta de calidad de datos
│   ├── hpo.py                    ← Optuna (y RandomizedSearchCV de la P2 como línea base)
│   ├── models.py                 ← igual que P2
│   ├── parallelism.py            ← igual que P2 (+ modo "auto" y guardado por modelo)
│   ├── training.py               ← metadata extendida (métrica de negocio y cómputo)
│   ├── backtesting.py            ← igual que P2 (+ lift por ola)
│   ├── evaluation.py             ← GetBestModel() rediseñado + métricas de negocio
│   ├── registry.py               ← extendido: trazabilidad + MLflow + Model Registry
│   ├── monitoring.py             ← NUEVO: logs de cómputo
│   └── reporte.py                ← NUEVO: results_summary.json y tablas del README
├── scripts/
│   └── verificar_idempotencia.py ← NUEVO: evidencia de idempotencia
├── docs/
│   ├── decisiones_diseno.md      ← NUEVO: D1 a D9
│   └── img/                      ← NUEVO: capturas de Airflow y MLflow
├── notebooks/
│   ├── validacion.ipynb          ← ejecuta el pipeline paso a paso (Colab o local)
│   └── backtesting_analysis.ipynb← gráficos de AUC, lift y PSI por ola
└── data/
    ├── input/                    ← dataset (en .gitignore)
    ├── intermedios/              ← datos procesados y modelos de cada corrida (en .gitignore)
    └── output/
        └── YYYY-MM-DD_HH-MM-SS/  ← una carpeta por corrida
            ├── corrida.json              (corrida_id, trazabilidad, ejecuciones del DAG)
            ├── config.yaml               (configuración efectiva de la corrida)
            ├── validacion_datos.json
            ├── preprocesamiento.json
            ├── hpo_results.json
            ├── <modelo>_metadata.json    (uno por modelo)
            ├── backtesting_results.json
            ├── seleccion.json            (criterio y motivo de GetBestModel)
            ├── model_registry.json
            ├── compute_logs.json         ← NUEVO
            ├── results_summary.json
            └── tablas_readme.md          (tablas listas para el README)
```

## 8. Qué se registra en MLflow

Experimento **`automl_venta_financiera`**. Cada corrida tiene un **run padre** (`corrida_<fecha_hora>`) y, colgando de él:

| Run hijo | Contenido |
|---|---|
| `<modelo>` (uno por modelo) | **Parámetros:** hiperparámetros finales (`hp_*`), familia, hardware, corte, filas de train, semilla, modo de entrenamiento. **Métricas:** `auc_train`, `auc_test`, `auc_val`, `decay_pct`, `lift_test`, `ks_test`, `pr_auc_test`, backtesting por ola (`bt_auc_p7`, `bt_lift_p7`, `bt_decay_p7`, `bt_psi_p7`, …) y promedios (`bt_*_prom`), `training_time_s` y logs de cómputo (`computo_*`). **Artefactos:** modelo serializado (`modelo/`), curva ROC (`graficos/`), `backtesting_results.json`, `compute_logs.json`, `<modelo>_metadata.json` y `config.yaml`. |
| `HPO_<modelo>` | librería, trials, mejores hiperparámetros, mejor AUC de cv, tiempo y la curva `trial_auc_cv` (un punto por trial). |
| `validacion_datos` | resultado de la compuerta de datos (sección 10). |

El run padre guarda la configuración completa como parámetros, el **criterio de selección aplicado** (tags `campeon`, `estado_seleccion`, `motivo_seleccion`), las métricas de cómputo de cada tarea del DAG (`tarea_<tarea>_duracion_s`, `_cpu_pct_*`, `_ram_mb_*`) y los artefactos `results_summary.json`, `compute_logs.json`, `seleccion.json`, `tablas_readme.md`, etc.

**Tags de trazabilidad en todos los runs** (Sección 5.2):

| Tag | Qué es |
|---|---|
| `git_commit`, `git_branch` | commit y rama del código (se leen de `.git`) |
| `data_hash` | SHA-256 combinado de los 10 CSV (el hash de cada archivo está en `corrida.json`) |
| `config_hash` | SHA-256 de la configuración efectiva; el `config.yaml` completo va como artefacto |
| `docker_image` | `automl-venta-financiera:3.0-parcial` (definido en `docker-compose.yml`) |
| `airflow_dag_run_id` | ejecución del DAG que generó el run |
| `corrida_id` | identificador de la corrida (hash de los tres primeros) |

### Model Registry (punto extra)

El campeón de cada corrida se registra como nueva versión del modelo **`automl_venta_financiera`**:

- Si no hay *champion*, la nueva versión recibe el alias **`champion`**.
- Si ya hay uno, el nuevo modelo (*challenger*) lo reemplaza solo si su **lift promedio en backtesting es al menos 2 % mayor** (`tracking.margen_promocion_pct`) **y** `GetBestModel()` lo marcó como estable (estado `OK`). Si no, queda con el alias **`challenger`**.
- Cada versión guarda los tags `corrida_id`, `run_id`, `bt_lift_prom`, `decision` y `motivo_decision`; la decisión también queda en `model_registry.json` y en el run padre.

## 9. Resultados de la corrida final

<!-- RESULTADOS: se completa con data/output/<corrida>/tablas_readme.md de la corrida final -->

*(Pendiente: se completa con la corrida final ejecutada en Docker.)*

## 10. Validación de datos (punto extra)

`validar_datos` es la primera tarea y funciona como compuerta del DAG. Verifica:

1. que estén las 10 particiones esperadas (`p1`–`p10`);
2. que todos los CSV tengan el mismo esquema y las columnas obligatorias (`partition`, `p_fecinformacion`, `target`);
3. que las columnas obligatorias no tengan nulos y que el % global de nulos no supere el 30 %;
4. que cada partición sea un único mes `YYYYMMDD` válido dentro de [20220301, 20221201];
5. que el `target` solo tenga 0 y 1.

Si algo falla, el DAG se detiene **sin reintentos** (`AirflowFailException`) con un mensaje que dice qué está mal, por ejemplo:

```
DATOS INVALIDOS, el DAG se detiene:
  - Particiones esperadas [...]; faltan ['p10'], sobran []
```

El resultado queda en `validacion_datos.json` y en el run `validacion_datos` de MLflow (tag `resultado_validacion`).

## 11. Idempotencia y reintentos

- **Reintentos:** `config.yaml → airflow.retries` (2) y `airflow.retry_delay_minutes` (1).
- **Re-ejecutar no duplica nada:** misma configuración + mismos datos + mismo commit dan el mismo `corrida_id`, así que se reutiliza la carpeta de la corrida, cada tarea salta el trabajo ya hecho y en MLflow se busca el run por tags antes de crearlo. El mecanismo completo está en D7 ([docs/decisiones_diseno.md](docs/decisiones_diseno.md#d7-mecanismo-de-idempotencia)).
- **Un reintento a mitad del entrenamiento** solo entrena los modelos que faltaban (cada modelo se guarda apenas termina).

**Cómo demostrarlo:**

```bash
docker compose exec airflow airflow dags test automl_venta_financiera   # 1.ª vez: entrena todo
docker compose exec airflow airflow dags test automl_venta_financiera   # 2.ª vez: segundos, no repite nada
docker compose exec airflow python scripts/verificar_idempotencia.py
```

<!-- IDEMPOTENCIA: pegar aquí la salida de verificar_idempotencia.py -->

## 12. Versionado con Git (punto extra)

Se trabaja sobre el mismo repositorio de la P2, con una rama por funcionalidad que se integra a `main` mediante merge o pull request:

| Rama | Contenido |
|---|---|
| `feature/docker` | Dockerfile, docker-compose.yml, imagen de MLflow |
| `feature/tracking` | MLflow, trazabilidad, Model Registry |
| `feature/airflow` | DAG, pipeline por tareas, idempotencia, validación de datos |
| `feature/compute-logs` | monitoring.py y compute_logs.json |
| `feature/docs` | README, decisiones de diseño, capturas |

Tags de versión: `v2.0-p2` (entrega de la P2) y `v3.0-parcial` (esta entrega).

## 13. Problemas frecuentes

| Problema | Solución |
|---|---|
| La tarea `preprocesar_datos` muere sin error (`Killed` / código -9) | Falta RAM para Docker. Cerrar otros programas o crear `C:\Users\<usuario>\.wslconfig` con `[wsl2]` y `memory=8GB`, luego `wsl --shutdown` y reabrir Docker Desktop. También se puede bajar `parallelism.n_jobs_max`. |
| `port is already allocated` (8080 o 5000) | Otro programa usa ese puerto. Cerrarlo o cambiar el puerto de la izquierda en `docker-compose.yml` (ej. `"8081:8080"`). |
| El DAG no aparece en Airflow | Esperar 1–2 minutos y revisar `docker compose logs airflow` (errores de importación). |
| `DATOS INVALIDOS` | Revisar que los 10 CSV estén en `data/input/` con su nombre original. |
| La carpeta del proyecto está en OneDrive | Pausar la sincronización mientras corre el DAG (escribe muchos archivos). |
| Se quiere empezar de cero | `docker compose down -v` (borra MLflow y Airflow) y borrar `data/output/<corrida>` y `data/intermedios/`. |

## 14. Entrega

- Repositorio **privado** de la P2, con el colaborador **`manassesmauricio-cloud`** (GitHub → Settings → Collaborators).
- Decisiones de diseño: [docs/decisiones_diseno.md](docs/decisiones_diseno.md).
- Capturas de Airflow y MLflow: [docs/img/](docs/img/).
- Corrida completa: `data/output/<corrida final>/`.
