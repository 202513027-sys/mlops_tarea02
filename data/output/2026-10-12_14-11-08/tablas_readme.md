# Tablas de la corrida `45b00315acd5` (2026-10-12_14-11-08)

Campeón: **LightGBM_hpo** — estado de la selección: **OK**.  
Métrica primaria de negocio: lift en el top 10% (promedio en las olas de backtesting).

> LightGBM_hpo: lift promedio en backtesting 4.9455 (mejor del grupo: 4.9796); 14 modelo(s) compitieron en la etapa final. Gana por costo: esta a menos de 2.0% del lift de CatBoost_cfg2 y entrena en 17.49s vs 58.32s.

## Tabla de modelos (ordenada por lift en backtesting)

| Modelo | Familia | AUC train | AUC test | Decay % | Lift@10% test | KS test | Lift@10% backtest (prom) | Decay backtest prom % | Tiempo (s) | Campeón | Descartado por |
|---|---|---|---|---|---|---|---|---|---|---|---|
| CatBoost_cfg2 | base_arboles_ensamble | 0.8588 | 0.8499 | 1.03 | 5.1147 | 0.5473 | 4.9796 | 1.35 | 58.32 | No | - |
| XGBoost_hpo | base_arboles_ensamble | 0.8654 | 0.8517 | 1.58 | 5.1424 | 0.5484 | 4.9719 | 1.96 | 17.68 | No | - |
| CatBoost_cfg1 | base_arboles_ensamble | 0.8533 | 0.8486 | 0.55 | 5.1101 | 0.5446 | 4.9653 | 0.97 | 29.16 | No | - |
| 🏆 LightGBM_hpo | base_arboles_ensamble | 0.8679 | 0.8503 | 2.02 | 5.1206 | 0.5464 | 4.9455 | 2.34 | 17.49 | Sí | - |
| AdaBoost_cfg2 | arboles_ensamble | 0.8485 | 0.8453 | 0.37 | 4.9689 | 0.539 | 4.8604 | 0.71 | 212.61 | No | - |
| Random_Forest_cfg1 | arboles_ensamble | 0.8683 | 0.8398 | 3.28 | 4.9557 | 0.528 | 4.8 | 3.79 | 120.89 | No | - |
| AdaBoost_cfg1 | arboles_ensamble | 0.8473 | 0.8435 | 0.45 | 4.901 | 0.5348 | 4.781 | 0.86 | 93.2 | No | - |
| Regresion_Logistica_hpo | lineal | 0.8313 | 0.8308 | 0.07 | 4.7796 | 0.5091 | 4.5661 | 0.58 | 1.38 | No | - |
| MLP_cfg1 | redes_neuronales | 0.8404 | 0.8334 | 0.83 | 4.7724 | 0.5191 | 4.5446 | 1.41 | 17.88 | No | - |
| MLP_cfg2 | redes_neuronales | 0.8652 | 0.8375 | 3.2 | 4.7908 | 0.5241 | 4.5309 | 3.87 | 21.38 | No | - |
| LDA_cfg2 | lineal | 0.8295 | 0.8282 | 0.15 | 4.7203 | 0.5066 | 4.4851 | 1.23 | 2.67 | No | - |
| LDA_cfg1 | lineal | 0.8294 | 0.8281 | 0.16 | 4.717 | 0.5063 | 4.4829 | 1.24 | 5.26 | No | - |
| Naive_Bayes_Gaussiano_cfg1 | probabilistico | 0.7545 | 0.7523 | 0.29 | 2.9789 | 0.4136 | 2.9993 | 0.36 | 0.5 | No | - |
| Naive_Bayes_Gaussiano_cfg2 | probabilistico | 0.7292 | 0.7295 | -0.05 | 2.7256 | 0.3887 | 2.7881 | -0.0 | 0.42 | No | - |


## Backtesting del campeón (LightGBM_hpo, entrenado hasta la partición 6, sin reentrenar)

| Ola | p_fecinformacion | AUC | Decay % | Lift@10% | KS | PR-AUC | PSI | N obs |
|---|---|---|---|---|---|---|---|---|
| p7 | 20220901 | 0.8342 | 3.88 | 4.592 | 0.5175 | 0.1518 | 0.0094 | 244666 |
| p8 | 20221001 | 0.848 | 2.3 | 4.9259 | 0.5366 | 0.2684 | 0.0073 | 240726 |
| p9 | 20221101 | 0.8503 | 2.03 | 4.9619 | 0.552 | 0.1723 | 0.0028 | 263753 |
| p10 | 20221201 | 0.858 | 1.14 | 5.3024 | 0.5621 | 0.1532 | 0.0117 | 317378 |


Promedios: AUC 0.8476 · lift 4.9455 · decay 2.34% · alerta de drift: No


## Backtesting de todos los modelos (promedio de las olas)

| Modelo | AUC prom | Lift@10% prom | Decay prom % | ¿Alguna ola sobre el umbral? |
|---|---|---|---|---|
| CatBoost_cfg2 | 0.8471 | 4.9796 | 1.35 | No |
| XGBoost_hpo | 0.8484 | 4.9719 | 1.96 | No |
| CatBoost_cfg1 | 0.845 | 4.9653 | 0.97 | No |
| LightGBM_hpo | 0.8476 | 4.9455 | 2.34 | No |
| AdaBoost_cfg2 | 0.8425 | 4.8604 | 0.71 | No |
| Random_Forest_cfg1 | 0.8354 | 4.8 | 3.79 | Sí |
| AdaBoost_cfg1 | 0.84 | 4.781 | 0.86 | No |
| Regresion_Logistica_hpo | 0.8265 | 4.5661 | 0.58 | No |
| MLP_cfg1 | 0.8285 | 4.5446 | 1.41 | No |
| MLP_cfg2 | 0.8317 | 4.5309 | 3.87 | Sí |
| LDA_cfg2 | 0.8193 | 4.4851 | 1.23 | No |
| LDA_cfg1 | 0.8192 | 4.4829 | 1.24 | No |
| Naive_Bayes_Gaussiano_cfg1 | 0.7518 | 2.9993 | 0.36 | No |
| Naive_Bayes_Gaussiano_cfg2 | 0.7292 | 2.7881 | -0.0 | No |


## Logs de cómputo por modelo (máquina: 16 núcleos, 7.67 GB RAM, modo: solo_cpu)

| Modelo | Duración (s) | CPU % prom | CPU % máx | RAM MB prom | RAM MB máx | GPU | Tamaño modelo (MB) |
|---|---|---|---|---|---|---|---|
| CatBoost_cfg1 | 29.19 | 99.51 | 100.9 | 348.12 | 368.16 | solo CPU | 0.125 |
| CatBoost_cfg2 | 58.34 | 99.93 | 105.5 | 350.67 | 375.05 | solo CPU | 0.439 |
| LDA_cfg1 | 5.29 | 163.7 | 371.2 | 1042.07 | 1251.98 | solo CPU | 0.005 |
| LDA_cfg2 | 2.7 | 92.63 | 100.1 | 1120.53 | 1372.61 | solo CPU | 0.03 |
| Random_Forest_cfg1 | 120.9 | 100.12 | 100.9 | 401.75 | 406.82 | solo CPU | 6.845 |
| AdaBoost_cfg1 | 93.22 | 100.11 | 101.8 | 528.22 | 534.7 | solo CPU | 0.064 |
| AdaBoost_cfg2 | 212.62 | 100.13 | 101.8 | 640.15 | 651.94 | solo CPU | 0.186 |
| Naive_Bayes_Gaussiano_cfg1 | 0.52 | 101.1 | 101.1 | 444.59 | 444.59 | solo CPU | 0.004 |
| Naive_Bayes_Gaussiano_cfg2 | 0.44 | 101.0 | 101.0 | 444.34 | 444.34 | solo CPU | 0.004 |
| MLP_cfg1 | 17.89 | 100.21 | 101.2 | 1211.3 | 1228.85 | solo CPU | 0.143 |
| MLP_cfg2 | 21.39 | 100.75 | 112.4 | 1264.83 | 1277.55 | solo CPU | 0.185 |
| XGBoost_hpo | 17.7 | 100.06 | 100.8 | 540.49 | 541.94 | solo CPU | 0.451 |
| LightGBM_hpo | 17.5 | 100.32 | 101.8 | 438.53 | 562.94 | solo CPU | 0.522 |
| Regresion_Logistica_hpo | 1.39 | 99.9 | 100.0 | 1242.09 | 1309.62 | solo CPU | 0.003 |


*CPU %: 100% = un núcleo completo (un 350% equivale a 3,5 núcleos ocupados).*


## Logs de cómputo por tarea del DAG

| Tarea | Duración (s) | CPU % prom | CPU % máx | RAM MB prom | RAM MB máx | Modo |
|---|---|---|---|---|---|---|
| validar_datos | 63.09 | 65.33 | 101.1 | 679.28 | 1039.6 | solo_cpu |
| preprocesar_datos | 73.97 | 86.19 | 101.0 | 1648.29 | 2917.97 | solo_cpu |
| optimizar_hiperparametros | 280.74 | 1471.14 | 1589.9 | 944.48 | 1026.37 | solo_cpu |
| entrenar_modelos | 275.91 | 271.69 | 688.6 | 2667.23 | 4003.07 | solo_cpu |
| backtesting | 120.52 | 171.82 | 1320.3 | 1713.12 | 1986.09 | solo_cpu |
| seleccionar_campeon | 0.16 | 12.7 | 12.7 | 334.94 | 334.94 | solo_cpu |
| registrar_experimentos | 50.53 | 11.94 | 95.2 | 410.95 | 424.79 | solo_cpu |
| generar_reporte | 1.44 | 66.45 | 83.9 | 375.54 | 378.59 | solo_cpu |


## Búsqueda de hiperparámetros

| Modelo | Librería | Trials | Mejor AUC (cv) | Tiempo (s) | Filas usadas | Mejores parámetros |
|---|---|---|---|---|---|---|
| XGBoost | optuna | 15 | 0.8455 | 95.4 | 100000 | `{'n_estimators': 400, 'max_depth': 3, 'learning_rate': 0.05, 'subsample': 0.8}` |
| LightGBM | optuna | 15 | 0.8428 | 150.9 | 100000 | `{'n_estimators': 300, 'num_leaves': 15, 'learning_rate': 0.03, 'subsample': 1.0}` |
| Regresion_Logistica | optuna | 15 | 0.8317 | 27.03 | 100000 | `{'C': 0.01}` |


## Model Registry

Versión 1 → **champion**. No habia champion vigente.
