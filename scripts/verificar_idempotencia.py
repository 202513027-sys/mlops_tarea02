"""
verificar_idempotencia.py
=========================
Evidencia de idempotencia para el README (Seccion 6.2): cuenta cuantos runs
hay en MLflow por corrida y por tipo, y cuantas veces se ejecuto el DAG sobre
cada corrida. Si el DAG se ejecuto 2 veces con la misma configuracion y los
mismos datos, debe haber 2 ejecuciones pero UN solo juego de runs.

Uso (con docker compose levantado):
    docker compose exec airflow python scripts/verificar_idempotencia.py
"""

import glob
import json
import os
import sys
from collections import Counter

import yaml

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)
os.chdir(RAIZ)

from src import registry  # noqa: E402


def main():
    with open("config/config.yaml", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    client, exp_id = registry.conectar_mlflow(config)
    runs = client.search_runs([exp_id], max_results=5000)

    print(f"\nExperimento MLflow: {config['tracking']['experiment_name']} (id {exp_id})")
    print(f"Runs activos en total: {len(runs)}\n")

    por_corrida = {}
    for r in runs:
        cid = r.data.tags.get("corrida_id", "?")
        por_corrida.setdefault(cid, Counter())[r.data.tags.get("tipo_run", "?")] += 1

    for ruta in sorted(glob.glob(os.path.join(config["output"]["output_dir"], "*", "corrida.json"))):
        with open(ruta, encoding="utf-8") as f:
            corrida = json.load(f)
        cid = corrida["corrida_id"]
        conteo = por_corrida.get(cid, Counter())
        ejecuciones = corrida.get("ejecuciones", [])
        print(f"Corrida {cid}  ({os.path.basename(os.path.dirname(ruta))})")
        print(f"  Ejecuciones del DAG sobre esta corrida: {len(ejecuciones)}")
        for e in ejecuciones:
            print(f"    - {e['airflow_dag_run_id']}  ({e['inicio']})")
        print(f"  Runs en MLflow: {dict(conteo)}  -> total {sum(conteo.values())}")
        duplicados = [n for n, c in Counter(
            r.data.tags.get("modelo") for r in runs
            if r.data.tags.get("corrida_id") == cid and r.data.tags.get("tipo_run") == "modelo").items() if c > 1]
        print(f"  Modelos con runs duplicados: {duplicados or 'ninguno'}\n")


if __name__ == "__main__":
    main()
