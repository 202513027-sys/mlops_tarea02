# AutoML — Propensión de Venta Financiera (P2: Paralelismo + Backtesting Temporal)

Sistema AutoML en Python puro para predecir propensión de venta de un producto
financiero. Esta es la **extensión P2** del proyecto de la Práctica 1: mismo
criterio de selección de modelos (`GetBestModel`), más tres piezas nuevas:

1. **Entrenamiento paralelo** (CPU, orquestado explícitamente) con medición real de speed-up.
2. **Detección automática de GPU**, con fallback a CPU sin errores.
3. **Optimización de hiperparámetros (HPO)** para ≥3 modelos.
4. **Backtesting temporal walk-forward** sobre ≥4 meses futuros, con alertas de drift y revalidación del campeón.

> Si vienes de la P1: el formato se mantiene igual a propósito — `config.yaml`
> sigue siendo el único archivo que hay que tocar para correr esto, y el
> comando de ejecución es casi el mismo, solo con un flag nuevo (`--parallel`).

---

## 1. Instalación

```bash
pip install -r requirements.txt
```

No se necesita instalar nada adicional para usar GPU (ver sección 5): si la
máquina tiene una GPU NVIDIA, XGBoost/LightGBM/CatBoost ya la soportan con las
versiones que trae `requirements.txt`.

## 2. Cómo obtener el dataset

El dataset viene particionado por mes (`p1_extrac.csv`, `p2_extrac.csv`, ...,
columna `partition`). Descárgalo del Drive del curso y colócalo así:

```
automl_venta_financiera/
└── data/
    └── input/
        ├── p1_extrac.csv
        ├── p2_extrac.csv
        ├── ...
```

> **Cambio respecto a la P1:** los CSV van **directo** dentro de `data/input/`
> (ya no en `data/input/data/`). `preprocessing.py` detecta solo si
> `input_path` apunta a un archivo único o a una carpeta con varios CSV.

Para correr el proyecto completo se necesitan las 10 particiones (`p1`…`p10`). Con menos particiones el pipeline también funciona: ajusta `backtesting.train_cutoff_partition` en `config.yaml` para dejar al menos 4 particiones libres como "meses futuros" del backtesting.

## 3. Cómo ejecutar

```bash
# Modo secuencial (igual que la P1, un for-loop normal)
python main.py --input data/input --config config/config.yaml

# Modo paralelo (mide speed-up real, ver seccion 4)
python main.py --input data/input --config config/config.yaml --parallel
```

El flag `--parallel` tiene prioridad sobre `config.yaml`; si se omite, se usa
`parallelism.enabled` del config (por defecto `false`).

Cada corrida crea una carpeta con timestamp en `data/output/` con:

- `<modelo>_model.pkl` + `<modelo>_metadata.json` por cada uno de los
  experimentos entrenados (igual que la P1).
- `backtesting_results.json` — resultado del walk-forward del modelo campeón.
- `results_summary.json` — resumen completo de la corrida (ganador por cada
  criterio, HPO, paralelismo, backtesting, ranking completo).

## 3b. Resultados reales: comparación de todos los modelos

Corrida con las **10 particiones reales** (p1–p10, `train_cutoff_partition: 2`), ejecutada en un laptop Windows 11 con GPU NVIDIA GTX 1650 (4 GB), Python 3.14, 16 workers. Ordenada por AUC test.

| # | Modelo | Categoría | Hardware | AUC train | AUC test | AUC val | Decay % | Tiempo (s) |
|---|---|---|---|---|---|---|---|---|
| 1 | XGBoost_hpo 🏆 | base_arboles_ensamble | gpu | 0.8663 | 0.8518 | 0.8387 | 1.67 | 5.86 |
| 2 | CatBoost_cfg2 | base_arboles_ensamble | gpu | 0.8586 | 0.851 | 0.8377 | 0.88 | 10.04 |
| 3 | LightGBM_cfg1 | base_arboles_ensamble | cpu | 0.8886 | 0.8508 | 0.8376 | 4.25 | 15.66 |
| 4 | CatBoost_cfg1 | base_arboles_ensamble | gpu | 0.8521 | 0.8495 | 0.8363 | 0.31 | 6.23 |
| 5 | LightGBM_cfg2 | base_arboles_ensamble | cpu | 0.9758 | 0.846 | 0.8335 | 13.3 | 31.16 |
| 6 | AdaBoost_cfg1 | arboles_ensamble | cpu | 0.8458 | 0.8448 | 0.8311 | 0.12 | 98.39 |
| 7 | Random_Forest_hpo | arboles_ensamble | cpu | 0.8989 | 0.8443 | 0.8311 | 6.07 | 125.12 |
| 8 | AdaBoost_cfg2 | arboles_ensamble | cpu | 0.8446 | 0.8441 | 0.8308 | 0.06 | 244.1 |
| 9 | MLP_cfg2 | redes_neuronales | cpu | 0.8676 | 0.8382 | 0.8227 | 3.38 | 22.78 |
| 10 | MLP_cfg1 | redes_neuronales | cpu | 0.8617 | 0.8381 | 0.8252 | 2.74 | 28.83 |
| 11 | Regresion_Logistica_hpo | lineal | cpu | 0.8298 | 0.8317 | 0.8204 | -0.22 | 2.43 |
| 12 | LDA_cfg2 | lineal | cpu | 0.8258 | 0.8276 | 0.8164 | -0.22 | 2.11 |
| 13 | LDA_cfg1 | lineal | cpu | 0.8258 | 0.8275 | 0.8163 | -0.21 | 4.51 |
| 14 | Naive_Bayes_Gaussiano_cfg1 | probabilistico | cpu | 0.7457 | 0.7556 | 0.7387 | -1.33 | 0.7 |
| 15 | KNN_Ponderado_cfg2 | instancias | cpu | 1.0 | 0.7511 | 0.7483 | 24.89 | 0.19 |
| 16 | Naive_Bayes_Gaussiano_cfg2 | probabilistico | cpu | 0.714 | 0.729 | 0.7159 | -2.1 | 0.64 |
| 17 | KNN_Ponderado_cfg1 | instancias | cpu | 1.0 | 0.7067 | 0.7039 | 29.33 | 0.19 |

**Ganador (criterio `innovador`): `XGBoost_hpo`**, con AUC test 0.8518 y decay 1.67%.

Observaciones:

- Los 4 primeros (XGBoost, CatBoost_cfg2, LightGBM_cfg1, CatBoost_cfg1) difieren en AUC test por menos de 0.003; XGBoost gana por el puntaje innovador (AUC con penalización por decay), no por una ventaja grande de desempeño.
- KNN llega a AUC train = 1.0 y cae a ~0.71–0.75 en test (decay 25–29%): memoriza el entrenamiento. LightGBM_cfg2 también sobreajusta (decay 13.3%) y queda fuera del criterio innovador por superar el máximo de 10%.
- Los modelos lineales tienen decay negativo (rinden igual o algo mejor en test) pero con menos poder predictivo (AUC ≈ 0.83).
- Solo el candidato ganador pasa por backtesting (ver sección 7); el resto queda con `backtesting: null` por diseño, para no pagar el costo de evaluar 17 modelos.

## 4. Paralelismo y speed-up

**Librería elegida: `joblib.Parallel` + `delayed`** (sobre
`ProcessPoolExecutor`/`multiprocessing` puro), porque serializa objetos de
scikit-learn/XGBoost/LightGBM/CatBoost de forma más robusta, y es la opción
que recomienda el enunciado para trabajar con `scikit-learn`.

El paralelismo está orquestado **explícitamente a nivel de experimento** (un
proceso por modelo+configuración) — no se usa el `n_jobs` interno de
scikit-learn como sustituto. De hecho, cuando corre en modo paralelo, cada
modelo se fuerza a `n_jobs=1` internamente para que no compita por los mismos
núcleos contra el paralelismo externo (`src/parallelism.py:_forzar_n_jobs_interno_a_uno`).

`--parallel` corre el **mismo conjunto de experimentos dos veces** sobre el
mismo hardware: una vez en un for-loop secuencial (`T_secuencial`) y una vez
con `joblib.Parallel` (`T_paralelo`). Los resultados que se usan para el resto
del pipeline (métricas, `GetBestModel`, backtesting) son los de la corrida
paralela.

### Resultado medido (corrida real: laptop Windows 11, 16 workers, 17 experimentos incluyendo 3 con HPO)

| Métrica | Valor |
|---|---|
| Backend | `joblib` (`loky`) |
| Workers | 16 |
| T secuencial | 789.59 s (≈ 13.2 min) |
| T paralelo | 1343.12 s (≈ 22.4 min) |
| **Speed-up** | **0.5879x** |

**El paralelismo resultó MÁS LENTO que el secuencial (speed-up < 1).** Se reporta tal cual se midió. Las causas probables, por qué ocurre con esta carga y este hardware (son hipótesis técnicas razonadas, no se aislaron con experimentos separados):

- **Demasiados workers para la memoria y la carga.** Con `n_jobs: -1` se lanzaron 16 procesos; cada uno recibe una copia del dataset (cientos de miles de filas por mes, varias particiones) vía serialización (pickling). La copia y el consumo de RAM por proceso son un costo que el for-loop secuencial no tiene.
- **Una sola GPU compartida.** XGBoost y CatBoost entrenan en la GTX 1650 (4 GB); si varios procesos las usan a la vez compiten por la misma GPU y memoria de video, en vez de sumar capacidad.
- **Pocas tareas largas dominan el tiempo.** AdaBoost_cfg2 (244 s) y Random Forest (125 s) pesan más que todo el resto; en paralelo, el tiempo total no baja del de la tarea más larga, y el resto de los experimentos (varios de menos de 5 s) casi no aporta ganancia.
- **Más procesos que núcleos físicos.** 16 workers suelen corresponder a hilos lógicos (hyper-threading) de un laptop; los modelos pesados no escalan linealmente con ellos.

Con el hardware de esta entrega, el paralelismo a nivel de experimento no ayuda. Se esperaría un speed-up > 1 reduciendo `parallelism.n_jobs` (por ejemplo a 4) o en una máquina con más núcleos físicos y varias GPU. El tiempo de HPO (≈ 57 min solo para Random Forest) no entra en esta métrica porque corre antes del benchmark.

## 5. GPU: detección automática y fallback a CPU

`src/training.py:detectar_gpu()` revisa si existe el comando `nvidia-smi` y si
responde correctamente. Según `config.yaml -> gpu.use_gpu`:

| Valor | Comportamiento |
|---|---|
| `"auto"` (default) | Usa GPU si `detectar_gpu()` la encuentra; si no, CPU sin error. |
| `true` | Intenta forzar GPU; si la máquina no tiene, cae a CPU sin error (no se detiene el pipeline). |
| `false` | Siempre CPU, aunque haya GPU disponible. |

Solo 3 modelos del curso soportan GPU "de fábrica" (sin librerías extra):

```python
GPU_PARAMS_POR_MODELO = {
    "XGBoost":  {"device": "cuda"},
    "LightGBM": {"device": "gpu"},
    "CatBoost": {"task_type": "GPU"},
}
```

Cada experimento guarda en su metadata el campo `"hardware": "gpu"` o
`"cpu"` según corresponda.

**Respaldo a CPU por modelo:** si un modelo falla estando en GPU (por ejemplo
LightGBM instalado con `pip` sin soporte GPU, o falta de memoria de video),
`training.entrenar_experimento()` lo reentrena automáticamente en CPU sin
detener el pipeline, y deja en su metadata `"hardware": "cpu"` y el campo
`"gpu_fallback"` con el motivo. XGBoost no lanza error cuando no encuentra
GPU (cambia solo a CPU), por eso se lee el dispositivo real del modelo
entrenado para no reportar "gpu" falsamente. Probado simulando una GPU
detectada en una máquina sin GPU: los 3 modelos cayeron a CPU y la corrida
terminó normal.

> En el equipo de la corrida real (GTX 1650) XGBoost y CatBoost entrenaron en GPU (`"hardware": "gpu"`). Si la máquina no tiene GPU, todo corre en CPU sin errores.

## 6. Optimización de hiperparámetros (HPO)

**Estrategia elegida: `RandomizedSearchCV`** de scikit-learn (permitida por el
enunciado; se descartó Optuna/hyperopt/skopt para no agregar una dependencia
nueva, y están prohibidos TPOT/H2O/auto-sklearn/PyCaret).

Se corre para **3 modelos** (mínimo pedido), configurable en `config.yaml ->
hpo.models`:

```yaml
hpo:
  models: ["XGBoost", "Random_Forest", "Regresion_Logistica"]
  strategy: "random_search"
  n_trials: 20
  cv: 3
```

El modelo con hiperparámetros óptimos reemplaza las 2 configuraciones
manuales de la P1 para ese modelo, y queda registrado como
`<Modelo>_hpo` en la tabla comparativa, con su metadata incluyendo el campo
`"hpo"` (`best_params`, `best_cv_score`, `n_trials`, `time_s`).

### Mejores hiperparámetros encontrados (corrida real)

| Modelo | Mejor `best_params` | `best_cv_score` | Tiempo de búsqueda |
|---|---|---|---|
| XGBoost | `{'subsample': 0.9, 'n_estimators': 200, 'max_depth': 3, 'learning_rate': 0.1}` | 0.8478 | 274.34 s |
| Random_Forest | `{'n_estimators': 200, 'min_samples_leaf': 10, 'max_depth': 10}` | 0.8406 | 3422.4 s |
| Regresion_Logistica | `{'C': 10.0}` | 0.8284 | 10.43 s |

El `best_cv_score` (validación cruzada, 3 folds) de XGBoost fue 0.8478 y su AUC test final 0.8518, consistente entre sí. La búsqueda de XGBoost cayó en el borde del espacio (`max_depth=3` es el mínimo explorado), señal de que árboles aún más simples podrían rendir igual; Random Forest fue la búsqueda más costosa por mucho.

## 7. Backtesting temporal (walk-forward)

### Diseño: ventana "expanding", no "rolling"

Se eligió **un solo entrenamiento** (sobre todas las particiones hasta el
`train_cutoff_partition`) evaluado hacia adelante contra cada mes futuro, en
vez de reentrenar el modelo en cada ola con una ventana móvil ("rolling").

**Justificación de negocio:** el objetivo del backtesting es responder "si
despliego este modelo HOY, ¿qué tan rápido se degrada en los próximos meses
sin reentrenarlo?" — que es exactamente el escenario real de producción (un
modelo no se reentrena cada mes automáticamente). Una ventana rolling
respondería una pregunta distinta ("¿qué tan bien funcionaría si reentrenara
cada mes?"), que es más cara computacionalmente y no es la que un equipo de
riesgo/negocio necesita para decidir cada cuánto reentrenar.

### Partición temporal correcta

El orden temporal se determina por **`p_fecinformacion`** (fecha real,
`YYYYMMDD`), nunca por el nombre del archivo — ordenar por nombre de archivo
rompe con particiones de dos cifras (`p10` quedaría antes de `p2` en orden
lexicográfico). `preprocessing._separar_por_cutoff()` ordena los meses únicos
de `p_fecinformacion` y usa las primeras `train_cutoff_partition` para
entrenar; el resto queda 100% fuera del entrenamiento y del ajuste de
encoders/scaler, y se usa como "meses futuros" del walk-forward.

### Resultado del campeón (corrida real, `XGBoost_hpo`, AUC train = 0.8663)

| Mes (partición) | AUC | Decay vs. train | N obs | PSI | Alerta drift (>5%) |
|---|---|---|---|---|---|
| p3 | 0.8502 | 1.86% | 322,356 | 0.0051 | No |
| p4 | 0.8558 | 1.21% | 304,813 | 0.0058 | No |
| p5 | 0.8605 | 0.67% | 295,732 | 0.0092 | No |
| p6 | 0.8385 | 3.21% | 236,914 | 0.0426 | No |
| p7 | 0.8325 | 3.9% | 244,666 | 0.0248 | No |
| p8 | 0.8453 | 2.42% | 240,726 | 0.0206 | No |
| p9 | 0.85 | 1.88% | 263,753 | 0.0185 | No |
| p10 | 0.8578 | 0.98% | 317,378 | 0.0042 | No |

- **AUC promedio backtesting:** 0.8488
- **Decay promedio:** 2.02%
- **Alerta de drift:** `false`: ninguna de las 8 olas superó el umbral de 5.0% (peor ola: p7 con 3.9%), así que el campeón **no cambió** (`motivo_cambio_backtesting = null`, candidatos probados: `['XGBoost_hpo']`).
- Las olas p6 y p7 muestran la mayor caída de AUC (3.21% y 3.9%) y el mayor PSI (0.043 y 0.025), pero siguen por debajo de los umbrales de alerta.

`evaluation.revalidar_con_backtesting()` prueba primero al ganador del
criterio P1; si todas sus olas superan el umbral de drift, pasa al siguiente
candidato del ranking y repite, hasta encontrar uno estable (o agotar
candidatos). Esta lógica de cambio de campeón está implementada y lista, pero
en las corridas de prueba el ganador original siempre pasó la revalidación
(nunca hubo drift severo en *todas* las olas a la vez).

### Innovación (punto extra): PSI como medida de concept drift

Además de `auc`/`decay_pct` (lo pedido por el enunciado), cada ola calcula el
**Population Stability Index (PSI)** entre la distribución de probabilidades
predichas en entrenamiento y en el mes evaluado — la métrica estándar de la
industria de riesgo crediticio para detectar *data/concept drift*, independiente
de si se conoce el `target` real del mes futuro.

| PSI | Interpretación |
|---|---|
| < 0.10 | Sin cambio significativo |
| 0.10 – 0.25 | Cambio moderado (vigilar) |
| ≥ 0.25 | Cambio significativo (posible concept drift) |

En la corrida real, el PSI de las 8 olas estuvo entre 0.0042 y 0.0426 ("sin cambio significativo"), confirmando que la caída de AUC observada es decay natural y no un cambio brusco de la población de clientes.

Ver `notebooks/backtesting_analysis.ipynb` para el gráfico de AUC por ola
(con la línea de referencia en `AUC_train`) y el gráfico de PSI por ola —
ambos se regeneran automáticamente leyendo la última corrida de
`data/output/`.

## 8. Notebooks

- `notebooks/validacion.ipynb` — reproduce el flujo completo (`preprocesar` →
  `HPO` → entrenamiento paralelo con benchmark → `GetBestModel` →
  `revalidar_con_backtesting` → guardar) en Google Colab, sin modificar nada.
- `notebooks/backtesting_analysis.ipynb` — lee la última corrida guardada en
  `data/output/` y genera el gráfico de AUC por ola (con línea de `AUC_train`)
  y el gráfico de PSI, que es el entregable visual mínimo pedido.

## 9. Alcance y limitaciones de esta entrega

- Los resultados numéricos de este README provienen de **una corrida real con las 10 particiones disponibles** (p1–p10; las olas de backtesting p3–p10 tienen entre 237 mil y 322 mil filas cada una).
- Se usó `train_cutoff_partition: 2`: el modelo se entrena solo con p1–p2 y las 8 particiones restantes (p3–p10) se usan como olas de backtesting (el mínimo pedido es 4). Un cutoff mayor (por ejemplo 6, que deja p7–p10 como olas) entrenaría con más historia; se puede probar cambiando solo ese valor en `config.yaml`.
- El speed-up medido fue < 1 (ver sección 4); se documenta el análisis en vez de ocultarlo.
- LightGBM se instaló sin soporte GPU, así que sus experimentos quedaron en CPU con el campo `gpu_fallback` en su metadata (comportamiento esperado del respaldo).
- La carpeta `data/output/` incluida en este repositorio puede ser de una corrida de ejemplo previa; los números definitivos son los de este README y los de `results_summary.json` de la corrida real.

## 10. Entrega del repositorio

Este proyecto se entrega como repositorio **privado** de GitHub (no .zip).

> **Importante:** el profesor (`manassesmauricio-cloud`) debe ser invitado
> manualmente como colaborador del repositorio **antes de la fecha límite**.
> Esto se deja pendiente a propósito para que el equipo lo revise antes de dar
> acceso — hazlo desde GitHub → Settings → Collaborators del repo.
