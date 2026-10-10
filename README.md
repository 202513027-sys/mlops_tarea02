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

Corrida con las **10 particiones reales** (p1–p10, `train_cutoff_partition: 2`), ejecutada en un laptop Windows 11 con GPU NVIDIA GTX 1650 (4 GB), Python 3.14, `parallelism.n_jobs: 4`, `hpo.n_trials: 10`. Ordenada por AUC test. 🏅 = ganador del criterio de la P1 (AUC test + decay); 🏆 = campeón final tras el backtesting (ver sección 7).

| # | Modelo | Categoría | Hardware | AUC train | AUC test | AUC val | Decay % | Tiempo (s) |
|---|---|---|---|---|---|---|---|---|
| 1 | XGBoost_hpo 🏅 | base_arboles_ensamble | gpu | 0.8902 | 0.8511 | 0.8386 | 4.4 | 7.56 |
| 2 | LightGBM_hpo | base_arboles_ensamble | cpu* | 0.8764 | 0.8511 | 0.8383 | 2.89 | 24.9 |
| 3 | CatBoost_cfg2 | base_arboles_ensamble | gpu | 0.8586 | 0.851 | 0.8377 | 0.88 | 30.42 |
| 4 | CatBoost_cfg1 | base_arboles_ensamble | gpu | 0.8521 | 0.8495 | 0.8363 | 0.31 | 25.55 |
| 5 | AdaBoost_cfg1 | arboles_ensamble | cpu | 0.8458 | 0.8448 | 0.8311 | 0.12 | 87.47 |
| 6 | AdaBoost_cfg2 | arboles_ensamble | cpu | 0.8446 | 0.8441 | 0.8308 | 0.06 | 327.48 |
| 7 | Random_Forest_cfg1 | arboles_ensamble | cpu | 0.8707 | 0.8414 | 0.828 | 3.37 | 108.68 |
| 8 | MLP_cfg2 | redes_neuronales | cpu | 0.8676 | 0.8382 | 0.8227 | 3.38 | 28.11 |
| 9 | MLP_cfg1 | redes_neuronales | cpu | 0.8617 | 0.8381 | 0.8252 | 2.74 | 35.8 |
| 10 | Random_Forest_cfg2 | arboles_ensamble | cpu | 1.0 | 0.8323 | 0.822 | 16.77 | 647.5 |
| 11 | Regresion_Logistica_hpo | lineal | cpu | 0.8298 | 0.8317 | 0.8204 | -0.22 | 1.5 |
| 12 | LDA_cfg2 | lineal | cpu | 0.8258 | 0.8276 | 0.8164 | -0.22 | 1.94 |
| 13 | LDA_cfg1 | lineal | cpu | 0.8258 | 0.8275 | 0.8163 | -0.21 | 2.75 |
| 14 | Naive_Bayes_Gaussiano_cfg1 | probabilistico | cpu | 0.7457 | 0.7556 | 0.7387 | -1.33 | 0.5 |
| 15 | KNN_Ponderado_cfg2 | instancias | cpu | 1.0 | 0.7511 | 0.7483 | 24.89 | 0.19 |
| 16 | Naive_Bayes_Gaussiano_cfg2 🏆 | probabilistico | cpu | 0.714 | 0.729 | 0.7159 | -2.1 | 0.44 |
| 17 | KNN_Ponderado_cfg1 | instancias | cpu | 1.0 | 0.7067 | 0.7039 | 29.33 | 0.55 |

\* `LightGBM_hpo` pidió GPU, pero el LightGBM instalado con `pip` no trae soporte GPU, así que se reentrenó en CPU (campo `gpu_fallback` en su metadata). El tiempo es por experimento corriendo con 1 hilo.

**Ganador de la P1 (criterio `innovador`): `XGBoost_hpo`**, con AUC test 0.8511 y decay 4.4%. **Campeón final tras backtesting: `Naive_Bayes_Gaussiano_cfg2`** (AUC test 0.729), por el criterio de estabilidad del enunciado; se discute en la sección 7.

Observaciones:

- Los 4 primeros por AUC test (XGBoost_hpo, LightGBM_hpo, CatBoost_cfg2, CatBoost_cfg1) difieren en menos de 0.002; XGBoost gana el criterio de la P1 por el puntaje innovador, no por una ventaja grande de desempeño.
- KNN llega a AUC train = 1.0 y cae a ~0.71–0.75 en test (decay 25–29%): memoriza el entrenamiento. Random_Forest_cfg2 también sobreajusta (AUC train 1.0, decay 16.8%). Estos tres quedan fuera del filtro de decay (< 10%) y por eso su `backtesting` es `null`.
- Los modelos lineales y Naive Bayes tienen decay negativo (rinden igual o algo mejor en test que en train), pero con menos poder predictivo.
- Los otros 14 experimentos cumplen el filtro de decay de la P1 (< 10%) y pasan todos por backtesting (sección 7).

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
mismo hardware, **con la misma configuración interna** (1 hilo por modelo:
`n_jobs=1` y, para CatBoost, `thread_count=1`): una vez en un for-loop
secuencial (`T_secuencial`) y una vez con `joblib.Parallel` (`T_paralelo`). Así
la única diferencia entre ambas es el paralelismo *entre* experimentos. A cada
worker se le envían solo los arrays de entrenamiento (no `df_futuro_raw`, que
solo usa el backtesting). Los resultados que usa el resto del pipeline son los
de la corrida paralela.

**Cómo verificar que corre en paralelo:** cada experimento guarda su `worker_pid`
y su ventana de tiempo; el resumen incluye `procesos_distintos_usados` y
`max_experimentos_simultaneos`. Además, en el Administrador de tareas se ven
varios `python.exe` activos a la vez durante el entrenamiento.

### Resultado medido (corrida real: laptop Windows 11, 4 workers, 17 experimentos incluyendo 3 con HPO)

Comparación justa: la corrida secuencial y la paralela usan **la misma configuración (1 hilo por modelo)** y los mismos arrays de entrenamiento.

| Métrica | Valor |
|---|---|
| Backend | `joblib` (`loky`) |
| Workers (`n_jobs: 4`) | 4 |
| T secuencial | 1153.46 s (≈ 19.2 min) |
| T paralelo | 714.73 s (≈ 11.9 min) |
| **Speed-up** | **1.61x** |
| Procesos distintos usados | 4 |
| Máx. experimentos entrenando a la vez | 4 |

El paralelismo **sí reduce el tiempo** (de 19.2 a 11.9 min) y la traza lo confirma: 4 procesos distintos y hasta 4 experimentos simultáneos.

**Por qué no llega a 4x:** el tiempo paralelo no puede bajar del de la tarea individual más larga. `Random_Forest_cfg2` tarda 647.5 s por sí sola y `AdaBoost_cfg2` 327.5 s; el tope teórico es 1153.46 / 647.5 ≈ 1.78x, y se obtuvo 1.61x (≈ 90% de ese tope). El resto de los experimentos dura segundos y casi no aporta ganancia.

**Historia de la medición (transparencia):** una primera corrida con `n_jobs: -1` (16 workers) dio 0.59x. No era comparable: el secuencial usaba modelos multinúcleo (`n_jobs=-1`, CatBoost con todos los hilos) y el paralelo los forzaba a 1 hilo; además CatBoost (`thread_count`) no se limitaba y a cada worker se le enviaba `df_futuro_raw` (millones de filas que el entrenamiento no usa). Tras corregir eso y bajar a 4 workers, el speed-up pasó a 1.61x. Además, en una corrida intermedia apareció un error de memoria (`Unable to allocate ... MiB`); por eso se recomienda un `n_jobs` bajo en laptops, ya que cada proceso guarda su propia copia de los datos.

Para comparar contra el estilo de la P1 (modelos con todos sus hilos) se puede activar `parallelism.medir_referencia_multinucleo: true`; no se usó en esta corrida.

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
  models: ["XGBoost", "LightGBM", "Regresion_Logistica"]
  strategy: "random_search"
  n_trials: 10
  cv: 3
```

**Hardware del HPO:** la búsqueda corre siempre en **CPU** (`hpo.py` no pasa parámetros de GPU). Se prefirió así por reproducibilidad y para no competir con el entrenamiento por la GPU; además la búsqueda es secuencial (`n_jobs=1` en `RandomizedSearchCV`), ya que el paralelismo del proyecto vive en `parallelism.py`.

**Cambio de Random Forest a LightGBM:** en una primera corrida el HPO incluía Random_Forest, que tomó 3422 s (≈ 57 min) y en otros intentos horas, porque `RandomizedSearchCV` lo entrena con un solo núcleo. Se reemplazó por LightGBM (espacio de búsqueda en `models.py`), que tardó 83 s. Random Forest sigue entrenándose con sus 2 configuraciones manuales.

El modelo con hiperparámetros óptimos reemplaza las 2 configuraciones
manuales de la P1 para ese modelo, y queda registrado como
`<Modelo>_hpo` en la tabla comparativa, con su metadata incluyendo el campo
`"hpo"` (`best_params`, `best_cv_score`, `n_trials`, `time_s`).

### Mejores hiperparámetros encontrados (corrida real, 10 intentos, cv=3)

| Modelo | Mejor `best_params` | `best_cv_score` | Tiempo de búsqueda |
|---|---|---|---|
| XGBoost | `{'subsample': 1.0, 'n_estimators': 400, 'max_depth': 5, 'learning_rate': 0.03}` | 0.8472 | 116.29 s |
| LightGBM | `{'subsample': 1.0, 'num_leaves': 15, 'n_estimators': 400, 'learning_rate': 0.03}` | 0.8468 | 83.18 s |
| Regresion_Logistica | `{'C': 10.0}` | 0.8284 | 9.5 s |

El `best_cv_score` (validación cruzada, 3 folds) de XGBoost fue 0.8472 y su AUC test final 0.8511, consistente entre sí; lo mismo ocurre con LightGBM (0.8468 vs 0.8511). La regresión logística eligió el valor más alto de `C` explorado (10.0), señal de que apenas necesita regularización.

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

### Resultado del backtesting de los 14 candidatos (corrida real)

Decay promedio = promedio, sobre las 8 olas (p3–p10), de `(AUC_train − AUC_ola) / AUC_train × 100`. Un valor **negativo** significa que el modelo rindió *mejor* en los meses futuros que en su propio entrenamiento. Ningún candidato tuvo drift severo en *todas* las olas.

| Candidato | AUC test | AUC prom. backtesting | Decay prom. backtesting (%) | ¿Alguna ola > 5%? |
|---|---|---|---|---|
| XGBoost_hpo | 0.8511 | 0.8483 | 4.71 | Sí (3 olas > 5%) |
| LightGBM_hpo | 0.8511 | 0.8487 | 3.16 | No |
| CatBoost_cfg2 | 0.851 | 0.8489 | 1.13 | No |
| CatBoost_cfg1 | 0.8495 | 0.848 | 0.47 | No |
| AdaBoost_cfg1 | 0.8448 | 0.8412 | 0.55 | No |
| AdaBoost_cfg2 | 0.8441 | 0.8417 | 0.34 | No |
| Random_Forest_cfg1 | 0.8414 | 0.8392 | 3.61 | No |
| MLP_cfg2 | 0.8382 | 0.831 | 4.22 | No |
| MLP_cfg1 | 0.8381 | 0.8342 | 3.19 | No |
| Regresion_Logistica_hpo | 0.8317 | 0.8301 | -0.03 | No |
| LDA_cfg2 | 0.8276 | 0.8251 | 0.08 | No |
| LDA_cfg1 | 0.8275 | 0.8248 | 0.12 | No |
| Naive_Bayes_Gaussiano_cfg1 | 0.7556 | 0.7498 | -0.54 | No |
| Naive_Bayes_Gaussiano_cfg2 🏆 | 0.729 | 0.7348 | -2.91 | No |

**Campeón final: `Naive_Bayes_Gaussiano_cfg2`** (menor decay promedio: −2.91%). El ganador de la P1, `XGBoost_hpo`, quedó con 4.71% (3 de sus 8 olas superaron el umbral de 5%: p6 5.92%, p7 6.55% y p8 5.09%, aunque no todas). El cambio quedó registrado en el JSON del campeón:

> *'XGBoost_hpo' (ganador de la P1) tuvo decay promedio en backtesting de 4.71%, mientras que 'Naive_Bayes_Gaussiano_cfg2' tuvo -2.91% (el menor entre 14 candidato(s) estable(s)). AUC test: 0.8511 (P1) vs 0.729 (nuevo campeon).*

Olas del campeón (`Naive_Bayes_Gaussiano_cfg2`, AUC train = 0.714):

| Mes (partición) | AUC | Decay vs. train | N obs | PSI |
|---|---|---|---|---|
| p3 | 0.7283 | -2.01% | 322,356 | 0.0098 |
| p4 | 0.7426 | -4.0% | 304,813 | 0.0174 |
| p5 | 0.7586 | -6.24% | 295,732 | 0.0456 |
| p6 | 0.7225 | -1.2% | 236,914 | 0.0787 |
| p7 | 0.7207 | -0.94% | 244,666 | 0.0684 |
| p8 | 0.731 | -2.38% | 240,726 | 0.0553 |
| p9 | 0.7316 | -2.47% | 263,753 | 0.0332 |
| p10 | 0.7429 | -4.05% | 317,378 | 0.1113 (moderado) |

### Criterio de campeón (extensión de `GetBestModel`)

`evaluation.revalidar_con_backtesting()` aplica el enunciado así:

1. Compiten los candidatos que cumplen la P1 (decay train→test < 10%).
2. A **todos** se les corre el backtesting walk-forward (no hay reentrenamiento).
3. Se descartan los que muestran drift severo (todas las olas sobre el umbral).
4. El campeón es el de **menor decay promedio en backtesting** (desempate: mayor AUC test).
5. Si el campeón difiere del ganador de la P1, la razón queda en
   `motivo_cambio_backtesting` del JSON del campeón, con los números concretos.

**Consecuencia del criterio (se observó en esta corrida):** aplicado al pie de
la letra, el criterio del enunciado eligió un modelo con **mucho menor AUC test**
(0.729 frente a 0.8511 de XGBoost). No es un error del código: el enunciado pide
el *menor decay promedio*, y el decay se mide contra el AUC de entrenamiento.
Un modelo simple y poco potente como Naive Bayes tiene un AUC de entrenamiento
bajo (0.714) y los meses futuros le salen ligeramente mejor, por lo que su decay
es negativo y "gana" en estabilidad. Estable no significa bueno: ese modelo es
~12 puntos de AUC peor que los de árboles.

**Alternativa (opcional, apagada por defecto):** `backtesting.min_avg_auc_backtest`
exige un AUC promedio mínimo en backtesting para poder ser campeón. Con, por
ejemplo, `0.80`, con los mismos resultados de esta corrida el campeón sería
`Regresion_Logistica_hpo` (decay promedio −0.03%, AUC backtesting 0.8301), un
modelo estable y con desempeño razonable. Esta entrega mantiene el criterio literal
del enunciado y deja documentada la alternativa; la decisión final es del equipo.

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

En la corrida real, el PSI del campeón estuvo entre 0.0098 y 0.1113; solo la ola p10 (0.1113) superó 0.10 ("cambio moderado, vigilar") y el resto quedó en "sin cambio significativo". Para `XGBoost_hpo` el PSI máximo fue 0.0412 (p6): la caída de AUC de ese modelo no se debe a un cambio brusco de la población de clientes.

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

- Los resultados numéricos de este README provienen de **una corrida real con las 10 particiones** (p1–p10; las olas de backtesting p3–p10 tienen entre 237 mil y 322 mil filas cada una), con `hpo.n_trials: 10` y `parallelism.n_jobs: 4`.
- Se usó `train_cutoff_partition: 2`: el modelo se entrena solo con p1–p2 y las 8 particiones restantes (p3–p10) se usan como olas de backtesting (el mínimo pedido es 4). Un cutoff mayor (por ejemplo 6, que deja p7–p10 como olas) entrenaría con más historia; se puede probar cambiando solo ese valor en `config.yaml`.
- El campeón final (`Naive_Bayes_Gaussiano_cfg2`) es consecuencia literal del criterio del enunciado y tiene bastante menos AUC que el ganador de la P1 (sección 7); la guarda opcional `min_avg_auc_backtest` permite exigir un AUC mínimo.
- El speed-up de 1.61x está limitado por dos tareas largas (`Random_Forest_cfg2`, 647 s, y `AdaBoost_cfg2`, 327 s); por la memoria RAM de un laptop se usó `n_jobs: 4` (cada proceso guarda su propia copia de los datos).
- LightGBM se instaló sin soporte GPU, así que sus experimentos quedan en CPU con el campo `gpu_fallback` en su metadata (comportamiento esperado del respaldo).
- Los resultados pueden variar en el 3.er–4.º decimal entre máquinas con y sin GPU (XGBoost y CatBoost en GPU).
- La carpeta `data/output/` incluida en este repositorio puede contener corridas anteriores; los números definitivos son los de este README y los de `results_summary.json` de la última corrida (`2026-10-09_19-08-23`).

## 10. Entrega del repositorio

Este proyecto se entrega como repositorio **privado** de GitHub (no .zip).

> **Importante:** el profesor (`manassesmauricio-cloud`) debe ser invitado
> manualmente como colaborador del repositorio **antes de la fecha límite**.
> Esto se deja pendiente a propósito para que el equipo lo revise antes de dar
> acceso — hazlo desde GitHub → Settings → Collaborators del repo.
