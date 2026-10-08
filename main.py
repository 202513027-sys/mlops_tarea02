"""
main.py
=======
Orquestador del AutoML. NO contiene logica de negocio: solo llama, en
secuencia, a preprocessing -> hpo -> (parallelism | training secuencial) ->
evaluation.GetBestModel() -> evaluation.revalidar_con_backtesting(), y
reporta/guarda el resultado via registry.

Comando de ejecucion (igual que la P1, con un flag opcional nuevo):
    python main.py --input data/input --config config/config.yaml
    python main.py --input data/input --config config/config.yaml --parallel
"""

import argparse
import sys

import yaml

from src import evaluation
from src import parallelism
from src import preprocessing
from src import registry
from src import training
from src.hpo import buscar_hiperparametros


def parsear_argumentos():
    """Define y parsea los argumentos de linea de comandos."""
    parser = argparse.ArgumentParser(description="AutoML - Propension de venta financiera (P2)")
    parser.add_argument("--input", type=str, default=None,
                         help="Ruta al CSV de entrenamiento, o a una carpeta con varios CSV "
                              "particionados (sobreescribe config.yaml)")
    parser.add_argument("--config", type=str, default="config/config.yaml",
                         help="Ruta al archivo de configuracion YAML")
    parser.add_argument("--parallel", action="store_true", default=None,
                         help="Fuerza el modo paralelo (mide speed-up). Si se omite, se usa "
                              "config.yaml -> parallelism.enabled")
    return parser.parse_args()


def cargar_configuracion(ruta_config: str, input_override: str = None) -> dict:
    """Carga config.yaml y, si se paso --input, sobreescribe la ruta del dataset."""
    with open(ruta_config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if input_override:
        config["dataset"]["input_path"] = input_override
    return config


def main():
    args = parsear_argumentos()

    try:
        config = cargar_configuracion(args.config, args.input)
    except FileNotFoundError:
        print(f"\n[ERROR] No se encontro el archivo de configuracion: {args.config}\n")
        sys.exit(1)

    modo_paralelo = args.parallel if args.parallel is not None else \
        config.get("parallelism", {}).get("enabled", False)

    print("\n" + "#" * 90)
    print("#  AutoML P2 - Paralelismo + Backtesting Temporal - Venta de Producto Financiero")
    print(f"#  Modo: {'PARALELO (con benchmark de speed-up)' if modo_paralelo else 'SECUENCIAL (comportamiento P1)'}")
    print("#" * 90 + "\n")

    try:
        pre_data = preprocessing.preprocesar(config)
    except FileNotFoundError:
        print(f"\n[ERROR] No se encontro el dataset en: {config['dataset']['input_path']}")
        print("Coloca el/los CSV en data/input/, o indica la ruta con --input.")
        print("Ver README.md -> 'Como obtener el dataset'.\n")
        sys.exit(1)

    # --- HPO: busca hiperparametros para >=3 modelos antes del entrenamiento final ---
    hpo_results = buscar_hiperparametros(pre_data, config)

    experimentos = training.construir_lista_experimentos(config, hpo_results)

    # --- Entrenamiento: secuencial (P1) o paralelo con benchmark de speed-up (P2) ---
    if modo_paralelo:
        resultados, info_parallelism = parallelism.ejecutar_con_benchmark(experimentos, pre_data, config)
    else:
        resultados = training.entrenar_todos_secuencial(experimentos, pre_data)
        info_parallelism = {
            "backend": None, "n_workers": 1, "t_sequential_s": None,
            "t_parallel_s": None, "speed_up": None,
            "nota": "Corrida en modo secuencial (--parallel no se uso); sin benchmark de speed-up.",
        }

    for r in resultados:
        r["metadata"]["parallelism"] = info_parallelism

    # --- Seleccion: criterio P1 (AUC test + decay) ---
    seleccion = evaluation.GetBestModel(
        resultados,
        max_decay_pct=config["selection"]["max_decay_pct"],
        usar_criterio_innovador=config["selection"]["usar_criterio_innovador"],
    )

    # --- Extension P2: revalida con estabilidad en backtesting walk-forward ---
    revalidacion = evaluation.revalidar_con_backtesting(seleccion, pre_data, config)
    campeon = revalidacion["campeon_final"]

    tabla = training.construir_tabla_resultados(seleccion["ranking_completo"])

    print("\n" + "=" * 100)
    print("  TABLA COMPARATIVA DE TODOS LOS EXPERIMENTOS (ordenada por AUC test)")
    print("=" * 100)
    for fila in tabla:
        print(fila)
    print("=" * 100)

    print(f"\nMODELO GANADOR INICIAL (criterio P1, {seleccion['criterio_usado']}): "
          f"{seleccion['ganador_final']['metadata']['ml_name']}")
    print(f"MODELO CAMPEON FINAL (tras revalidar con backtesting): {campeon['metadata']['ml_name']}")
    if revalidacion["motivo_cambio"]:
        print(f"  >> CAMBIO DE CAMPEON: {revalidacion['motivo_cambio']}")
    bt = campeon["metadata"]["backtesting"]
    print(f"  AUC train: {campeon['metadata']['performance']['auc_train']}")
    print(f"  AUC promedio en backtesting: {bt['avg_auc_backtest']}")
    print(f"  Decay promedio en backtesting: {bt['avg_decay_backtest']}%")
    print(f"  Alerta de drift: {bt['drift_alert']} (umbral: {bt['drift_alert_threshold']}%)")
    if info_parallelism.get("speed_up") is not None:
        print(f"  Speed-up medido: {info_parallelism['speed_up']}x "
              f"({info_parallelism['n_workers']} workers)")

    carpeta_corrida = registry.crear_carpeta_corrida(config["output"]["output_dir"])

    for r in resultados:
        registry.guardar_modelo_y_metadata(
            carpeta_corrida, r["metadata"]["ml_name"], r["modelo"], r["metadata"]
        )

    registry.guardar_backtesting_results(carpeta_corrida, campeon["metadata"]["backtesting"])

    registry.guardar_resumen_corrida(carpeta_corrida, {
        "ganador_base": seleccion["ganador_base"]["metadata"],
        "ganador_innovador": seleccion["ganador_innovador"]["metadata"],
        "ganador_final_criterio_p1": seleccion["ganador_final"]["metadata"],
        "campeon_final_tras_backtesting": campeon["metadata"],
        "motivo_cambio_backtesting": revalidacion["motivo_cambio"],
        "candidatos_probados_en_backtesting": revalidacion["candidatos_probados"],
        "criterio_usado": seleccion["criterio_usado"],
        "parallelism": info_parallelism,
        "hpo_results": hpo_results,
        "cutoff_aplicado": pre_data["cutoff_aplicado"],
        "meses_ordenados": [int(m) for m in pre_data["meses_ordenados"]],
        "mes_validacion_usado": pre_data["mes_validacion_usado"],
        "ranking_completo": [r["metadata"] for r in seleccion["ranking_completo"]],
    })

    print(f"\nTodos los archivos (.pkl, .json, backtesting_results.json, results_summary.json) "
          f"quedaron en: {carpeta_corrida}\n")


if __name__ == "__main__":
    main()
