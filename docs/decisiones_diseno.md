# Decisiones de diseño — Parcial MCD-402 (Orquestación de pipelines de entrenamiento)

Este documento justifica las decisiones D1 a D9 del parcial. Para cada una se indica la **opción elegida**, las **alternativas descartadas**, la **justificación** y la **evidencia** (archivo, métrica o resultado de la corrida). La evidencia numérica corresponde a la corrida final documentada en el README (`data/output/<corrida>/`).

Resumen rápido:

| # | Decisión | Opción elegida |
|---|---|---|
| D1 | Decay porcentual o absoluto | **Porcentual (relativo)**, igual que P1 y P2 |
| D2 | Cómputo secuencial o paralelo | **Paralelo con `n_jobs: "auto"`** (núcleos − 1, tope 4); cae solo a secuencial; GPU automática |
| D3 | Granularidad del DAG | **Una tarea por etapa** (8 tareas), con puntos de control por modelo dentro de cada tarea |
| D4 | Métrica primaria de negocio | **Lift en el top 10 %** (decil superior) |
| D5 | Estrategia de HPO | **Optuna (TPE)**, 3 modelos, 15 trials, cv = 3, muestra de 100 mil filas |
| D6 | Gestor de experimentos | **MLflow** local en Docker, backend **SQLite** en volumen + artefactos servidos por el propio servidor |
| D7 | Idempotencia | **`corrida_id` determinista** (hash de config + datos + commit) + búsqueda de runs por tags + puntos de control |
| D8 | Lógica de `GetBestModel()` | **Filtros (sobreajuste y estabilidad) → mayor lift en backtesting → desempate por costo** |
| D9 | Periodicidad en producción | **Mensual**, al cerrar cada mes de datos (p. ej. día 5, 6:00 a. m.), más reentrenamiento anticipado si hay drift |

---

## D1. Decay porcentual o absoluto

**Elegido:** decay **relativo (%)** en los dos lugares donde se usa:

- Filtro de sobreajuste: `decay = (AUC_train − AUC_test) / AUC_train × 100` < `selection.max_decay_pct` (10 %).
- Estabilidad / alerta de drift en backtesting: `(AUC_train − AUC_ola) / AUC_train × 100`, promedio < `selection.max_decay_backtest_pct` (5 %); alerta si alguna ola supera `backtesting.drift_alert_threshold` (5 %).

**Alternativa descartada:** decay absoluto (`AUC_train − AUC_test`, en puntos de AUC).

**Justificación:**

- **Consistencia:** la P1 y la P2 ya usaban decay relativo; cambiarlo obligaría a recalibrar todos los umbrales y a re-explicar resultados anteriores.
- **Comparabilidad entre modelos:** los modelos tienen niveles de AUC muy distintos (Naive Bayes ≈ 0,73 frente a boosting ≈ 0,85). Una caída de 0,03 no pesa lo mismo en ambos; el relativo expresa la caída como proporción del desempeño propio de cada modelo.
- **Lectura de negocio:** "el modelo perdió 4 % de su capacidad" se entiende sin conocer el AUC base.

**Limitación conocida (vista en la P2):** un modelo débil puede tener decay bajo o negativo y verse "estable". Por eso el decay en el parcial **solo filtra**: no decide al campeón. Lo decide la métrica de negocio (D8).

**Evidencia:** `selection` en `config/config.yaml`; columna `Decay %` y `Descartado por` de la tabla de modelos del README (modelos excluidos por decay ≥ 10 %).

---

## D2. Cómputo secuencial o paralelo en el despliegue

**Elegido:** entrenamiento **paralelo entre modelos** con `joblib` (backend `loky`, un proceso por experimento y 1 hilo por modelo, igual que en la P2), con estas reglas **configurables sin tocar código**:

| Situación | Qué pasa |
|---|---|
| `parallelism.n_jobs: "auto"` (por defecto) | workers = núcleos − 1 (uno libre para Airflow/sistema), con tope `n_jobs_max: 4` |
| Máquina con 2 núcleos o menos | "auto" da 1 worker y el pipeline **pasa solo a modo secuencial** (cada modelo usa todos sus hilos internos) |
| Poca RAM | bajar `n_jobs_max` (cada worker copia los datos de entrenamiento, ~250 MB con 300 mil filas) |
| Con GPU NVIDIA | `gpu.use_gpu: "auto"` la detecta (XGBoost, LightGBM y CatBoost). Si un modelo falla en GPU, se reentrena en CPU (comportamiento de la P2) |
| Sin GPU (Docker por defecto) | todo en CPU; los logs de cómputo dicen explícitamente `"modo_computo": "solo_cpu"` |

**Cómo se ajusta en otra máquina, sin modificar código:**

1. Editando `config/config.yaml` → `parallelism` / `gpu`.
2. O al lanzar el DAG: **Trigger DAG w/ config** con `{"paralelo": false}`, `{"n_jobs": 2}` o `{"use_gpu": false}`.
3. O en local: `python main.py --sequential`, `--parallel`, `--n-jobs 2`.
4. GPU en Docker: descomentar el bloque `deploy.resources...` del servicio `airflow` en `docker-compose.yml`.

**Alternativas descartadas:**

- *Siempre secuencial:* desperdicia núcleos en máquinas grandes.
- *`n_jobs: -1` fijo (todos los núcleos):* en la P2 provocó un error de memoria y un speed-up < 1 en un laptop de 16 núcleos; además deja sin CPU al scheduler de Airflow.
- *Paralelizar con tareas de Airflow (una por modelo):* ver D3.

**Decisión relacionada (costo de cómputo):** `entrenamiento.max_filas_train: 300000` (muestra estratificada). Con el corte en 6 el train tiene ~1,1 millones de filas; 300 mil mantiene los tiempos parecidos a la P2 dentro de Docker. El test, la validación temporal y el backtesting se evalúan completos.

**Evidencia:** `parallelism.json` y `compute_logs.json` de la corrida (`n_workers`, `nucleos_disponibles`, `nucleos_usados_max`, `modo_computo`); tabla de logs de cómputo del README.

---

## D3. Granularidad del DAG

**Elegido:** **una tarea por etapa** (8 tareas en línea): `validar_datos → preprocesar_datos → optimizar_hiperparametros → entrenar_modelos → backtesting → seleccionar_campeon → registrar_experimentos → generar_reporte`. Cada tarea procesa todos los modelos.

**Alternativa descartada:** una tarea por modelo con *dynamic task mapping* (`.expand()`).

| Aspecto | Una tarea por etapa (elegida) | Una tarea por modelo (descartada) |
|---|---|---|
| Lectura del DAG | 8 cajas, fácil de explicar | 14 × 3 cajas mapeadas |
| Paralelismo | `joblib` dentro de `entrenar_modelos` (controlado por D2) | lo reparte el executor de Airflow |
| Memoria / disco | los datos se cargan **una vez** por tarea | cada tarea mapeada vuelve a cargar `datos.joblib` (~600 MB) |
| Reintentos | la tarea se reintenta, pero **solo reentrena los modelos que faltan** (punto de control por modelo) | se reintenta solo el modelo que falló |
| Trazabilidad | cada modelo sigue siendo **su propio run** en MLflow | igual |

**Justificación:** con 14 experimentos y un laptop, la ventaja de *task mapping* (reintentar un solo modelo) se obtiene igual con los puntos de control por modelo (`data/intermedios/<corrida_id>/modelos/<modelo>.json`), sin multiplicar la carga de datos ni la complejidad. El costo es que en la interfaz de Airflow no se ve el estado de cada modelo por separado (sí se ve en el log de la tarea y en MLflow).

**Evidencia:** `dags/training_dag.py`; captura del DAG en `docs/img/`; mensaje `"[entrenar] N modelo(s) ya estaban entrenados (reintento); faltan M"` en los logs de un reintento.

---

## D4. Métrica primaria de negocio

**Elegido:** **lift en el top 10 %** (`metrica_negocio.top_pct: 10`).

> Si el banco contacta solo al 10 % de clientes con mayor probabilidad según el modelo, ¿cuántas veces más ventas consigue que contactando a un 10 % elegido al azar?
>
> `lift = tasa de venta en el top 10 % / tasa de venta promedio`

**Justificación:**

- El problema es de **campañas**: el negocio contacta a un grupo limitado por la capacidad del call center o por el presupuesto, no a toda la base. El lift mide la calidad del modelo **justo donde se usa** (el decil superior); el AUC mide el orden en toda la población, incluidos clientes que nunca se contactarían.
- La clase positiva es rara (~3,4 % de ventas). Con este desbalance, dos modelos con AUC casi igual pueden concentrar ventas en el top de forma distinta.
- Se explica en una frase a un gerente comercial.

**Alternativas descartadas como métrica primaria (se registran igual como apoyo):**

- **KS:** habitual en banca de riesgo, pero mide la separación máxima en *algún* punto de corte, no necesariamente en el que usará la campaña.
- **PR-AUC:** buena para desbalance, pero es abstracta para negocio y resume todos los umbrales.
- **Ganancia esperada:** requiere un costo por contacto y un ingreso por venta que no tenemos; inventarlos haría que la decisión dependa de supuestos arbitrarios. Queda como mejora si el negocio entrega esos valores (el lift se convierte directamente en ganancia).

Además se registran `tasa_respuesta_top`, `captura_ventas_top` (qué % del total de ventas cae en el top 10 %), `ks` y `pr_auc`, en test y en cada ola de backtesting.

**Evidencia:** `src/evaluation.py → metricas_negocio()`; métricas `lift_test` y `bt_lift_*` en cada run de MLflow; columnas de lift en las tablas de modelos y de backtesting del README.

---

## D5. Estrategia de HPO / AutoML

**Elegido:** **Optuna** con muestreador **TPE** (`hpo.libreria: "optuna"`).

| Parámetro | Valor | Por qué |
|---|---|---|
| Modelos | XGBoost, LightGBM, Regresión logística | los mismos 3 de la P2 (dos boosting y un lineal), así la comparación es directa |
| Espacio de búsqueda | el mismo de la P2 (`models.HPO_SEARCH_SPACES`) | solo cambia **cómo** se recorre: al azar (P2) vs. bayesiano (parcial) |
| Presupuesto | 15 trials por modelo, cv = 3, tope de 900 s por modelo | suficiente para un espacio discreto pequeño |
| Datos | muestra estratificada de 100 mil filas del train | el ranking de hiperparámetros es estable con 100 mil filas; buscar sobre 1,1 millones multiplicaría el tiempo |
| Objetivo | AUC promedio en cv | métrica técnica estable para optimizar; el **negocio** decide después, en `GetBestModel()` |

**Alternativas descartadas:**

- *RandomizedSearchCV (P2):* se conserva como **línea base** (`hpo.libreria: "random_search"`), pero no aprende de los intentos anteriores.
- *AutoML completo (H2O, FLAML, AutoGluon):* decide también qué modelos usar, lo que quitaría control sobre los modelos de la P1/P2 y sobre la comparación justa entre ellos; H2O además exige Java en la imagen.

**Costo vs. ganancia:** cada búsqueda queda en MLflow (run `HPO_<modelo>`) con número de trials, mejores hiperparámetros, mejor AUC de cv, tiempo y la curva `trial_auc_cv` por trial. La tabla de HPO del README compara el tiempo invertido frente al AUC/lift obtenido por el modelo `<modelo>_hpo` contra sus configuraciones manuales.

**Otra decisión de costo:** se excluyen `KNN_Ponderado_cfg1`, `KNN_Ponderado_cfg2` y `Random_Forest_cfg2` (`entrenamiento.excluir_experimentos`). En la P2 los tres sobreajustaron (AUC train = 1,0, decay de 17 % a 29 %), nunca fueron candidatos, y estaban entre los más caros (Random_Forest_cfg2: 647 s; KNN: predicción lenta sobre cientos de miles de filas).

**Evidencia:** `hpo_results.json` de la corrida; runs `HPO_*` en MLflow; tabla de HPO del README.

---

## D6. Gestor de experimentos

**Elegido:** **MLflow 3.17**, levantado en `docker-compose.yml`:

- **Backend store:** SQLite (`/mlflow/mlflow.db`) dentro del volumen `mlflow_data`.
- **Artefactos:** el propio servidor los recibe y guarda (`--serve-artifacts --artifacts-destination /mlflow/artifacts`), en el mismo volumen. Airflow los sube por HTTP, sin compartir carpetas.
- **Cliente:** `mlflow-skinny` en la imagen de Airflow. El servidor va en **otra imagen** porque MLflow 3 necesita SQLAlchemy 2 y Airflow 2.10 usa SQLAlchemy 1.4.

**Por qué MLflow y no Neptune:** es gratuito, corre en local (también en la máquina del docente con `docker compose up`), no necesita credenciales ni cuentas, se integra con scikit-learn y XGBoost/LightGBM/CatBoost, y trae Model Registry (punto extra).

**Por qué SQLite y no PostgreSQL para MLflow:**

- **Concurrencia:** el único proceso que escribe en SQLite es el servidor de MLflow; los clientes escriben vía API REST. Además el DAG tiene `max_active_runs=1` y las tareas son secuenciales, así que nunca hay muchas escrituras simultáneas. SQLite alcanza.
- **Persistencia:** el volumen `mlflow_data` sobrevive a `docker compose down` (solo se pierde con `docker compose down -v`).
- **Cuándo cambiar a PostgreSQL:** con varios usuarios o varios servidores de MLflow escribiendo a la vez (SQLite bloquea la base completa en cada escritura).
- Airflow **sí** usa PostgreSQL (servicio `postgres`), porque su `LocalExecutor` no funciona con SQLite.

**Evidencia:** servicio `mlflow` en `docker-compose.yml`; `docker/mlflow/Dockerfile`; captura de la interfaz de MLflow en `docs/img/`.

---

## D7. Mecanismo de idempotencia

**Elegido:** un **identificador de corrida determinista** más una **búsqueda previa por tags** antes de crear cualquier run.

1. **`corrida_id` = hash(`config_hash` + `data_hash` + `git_commit`)** (12 caracteres). La misma configuración, los mismos datos y el mismo código dan siempre el mismo `corrida_id`. Si cualquiera de los tres cambia, es una corrida nueva.
2. **Carpeta de la corrida:** si ya existe `data/output/*/corrida.json` con ese `corrida_id`, se **reutiliza** esa carpeta (no se crea otra con fecha nueva). Cada ejecución del DAG se anota en `corrida.json → ejecuciones`.
3. **Archivos intermedios propios de cada corrida:** `data/intermedios/<corrida_id>/` (datos procesados y modelos). Nunca hay rutas compartidas sin identificador, así que dos corridas no se pisan.
4. **Puntos de control por tarea y por modelo:** antes de trabajar, cada tarea revisa si su resultado ya existe (`datos.joblib`, `hpo_results.json`, `modelos/<modelo>.json`, `<modelo>_backtesting.json`, `seleccion.json`). Los archivos se escriben con nombre temporal y se renombran al final (escritura atómica), así un archivo a medias nunca se confunde con uno terminado.
5. **MLflow sin duplicados:** antes de crear un run se busca uno con los tags `corrida_id` + `tipo_run` (+ `modelo`). Si existe y terminó (`FINISHED`), no se registra nada nuevo; si quedó a medias (la tarea falló), se borra y se vuelve a crear.
6. **Model Registry:** antes de registrar una versión se busca si ya hay una con el tag `corrida_id` de esta corrida.
7. **`max_active_runs=1`** en el DAG: dos ejecuciones simultáneas no pueden escribir sobre la misma corrida.

**Alternativas descartadas:**

- *Un run nuevo en cada ejecución (como el repo de referencia):* duplica runs.
- *Sobrescribir siempre la misma carpeta:* se pierden corridas anteriores con otra configuración.
- *Usar `dag_run_id` como identificador:* cambia en cada ejecución, así que no evita duplicados.

**Evidencia:** sección "Idempotencia" del README: el DAG se ejecutó dos veces y `scripts/verificar_idempotencia.py` muestra 2 ejecuciones y un solo juego de runs (14 modelos + 3 HPO + 1 validación + 1 corrida), sin duplicados. La segunda ejecución tarda segundos y `compute_logs.json → reejecuciones_omitidas` lista las tareas que no repitieron trabajo.

---

## D8. Lógica de `GetBestModel()`

**Elegido:** filtros + métrica de negocio + desempate por costo (`src/evaluation.py`):

1. **Filtro de sobreajuste:** decay train→test < 10 % (D1).
2. **Filtro de estabilidad temporal:** decay promedio en las olas de backtesting (p7–p10) < 5 % y sin drift severo en todas las olas.
3. **Métrica de negocio:** entre los que pasan, gana el de **mayor lift promedio en el top 10 % en backtesting**, es decir, en meses que el modelo nunca vio, que es como funcionará en producción.
4. **Costo operativo como desempate:** si varios modelos están a menos de 2 % del mejor lift (`tolerancia_lift_pct`), gana el que **entrena más rápido**: mismo valor de negocio, menor costo de cómputo (Sección 9).

**Qué pasa si ningún modelo es satisfactorio:**

| Caso | Estado | Comportamiento |
|---|---|---|
| Nadie pasa el filtro 2 | `CON_ALERTA` | se elige el mejor lift entre los que pasan el filtro 1, pero `promover = false`: **no** se convierte en *champion* en el Model Registry |
| Nadie pasa el filtro 1 | `SIN_MODELO_SATISFACTORIO` | se reporta el mejor lift entre los 3 de menor decay, solo como referencia, y se recomienda revisar datos e hiperparámetros |

**Cambio respecto a la P2 (y por qué):** en la P2 el campeón era el de *menor decay promedio en backtesting*, lo que eligió `Naive_Bayes_Gaussiano_cfg2` (AUC test 0,729) frente a `XGBoost_hpo` (0,851). Era estable, pero era malo. Ahora el decay **filtra** y el negocio **decide**.

**Factores considerados y no usados para decidir:** AUC-ROC (se registra como referencia técnica; el lift ya refleja el orden de los clientes donde importa); tamaño del modelo e interpretabilidad (todos los modelos pesan menos de unos MB; la interpretabilidad no es requisito del negocio en este caso).

**Trazabilidad del criterio:** el run padre de cada corrida en MLflow guarda los tags `campeon`, `estado_seleccion` y `motivo_seleccion`, y el artefacto `seleccion.json` con los parámetros del criterio, los candidatos finales y el motivo de descarte de cada modelo.

---

## D9. Periodicidad de ejecución en producción

**Elegido (documentado, no implementado; en el parcial `schedule=None`):** **mensual**, por ejemplo `schedule="0 6 5 * *"` (día 5 de cada mes, 6:00 a. m.), más un **reentrenamiento anticipado** si el monitoreo detecta drift.

**Justificación:**

- **Ritmo de los datos:** cada partición es una foto mensual (`p_fecinformacion` = primer día de cada mes). Ejecutar más seguido (diario o semanal) volvería a entrenar con exactamente los mismos datos y, gracias a la idempotencia (D7), no haría nada. Sería costo sin beneficio.
- **Ritmo del negocio:** las campañas comerciales se planifican por mes; el modelo nuevo tiene que estar listo antes de armar la lista de contactos del mes.
- **Etiquetas:** el `target` (si el cliente compró) se conoce cuando termina la ventana de respuesta de la campaña; por eso se deja un margen de algunos días al inicio del mes (el día 5).
- **Evidencia de estabilidad:** el backtesting mide cuánto se degrada el campeón entrenado hasta p6 al aplicarlo, sin reentrenar, en los 4 meses siguientes (decay por ola en el README). Con una degradación pequeña en 4 meses, reentrenar cada mes deja un margen de seguridad amplio; trimestral sería el mínimo aceptable.

**Disparador anticipado (propuesta):** si en la carga mensual el PSI del score supera 0,25 o el decay de la ola nueva supera el 5 %, se lanza el DAG fuera de calendario (con un sensor o un trigger desde el proceso de carga de datos).

**Alternativas descartadas:** diario o semanal (no hay datos nuevos); trimestral fijo (se arriesga a usar un modelo degradado hasta 3 meses sin reaccionar).
