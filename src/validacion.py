"""
validacion.py  (NUEVO en el parcial, punto extra 6.4)
======================================================
Compuerta de calidad de datos: es la PRIMERA tarea del DAG. Si los datos no
cumplen lo esperado, el DAG se detiene con un mensaje claro (y sin
reintentos, porque reintentar no arregla un archivo malo).

Que se revisa (todo configurable en config.yaml -> validacion_datos):
  1. Particiones: que esten las 10 esperadas (p1..p10), ni mas ni menos.
  2. Esquema: que todos los CSV tengan las mismas columnas y que esten las
     columnas obligatorias (partition, p_fecinformacion, target).
  3. Nulos: las columnas obligatorias no pueden tener nulos y el % global de
     nulos no puede pasar de max_pct_nulos_global.
  4. Fechas: cada particion corresponde a UN solo mes (p_fecinformacion
     YYYYMMDD valido), dentro del rango [fecha_min, fecha_max].
  5. Target binario: solo valores 0 y 1.
"""

import glob
import os

import pandas as pd


class DatosInvalidosError(Exception):
    """Se lanza cuando los datos no pasan la validacion."""


def validar_datos(config: dict) -> dict:
    """Revisa los CSV de entrada y devuelve un reporte.

    Retorna
    -------
    dict con "valido" (bool), "errores" (list), "advertencias" (list) y
    "detalle" por archivo. No lanza la excepcion: quien llama decide.
    """
    vcfg = config["validacion_datos"]
    ds = config["dataset"]
    input_path = ds["input_path"]
    col_mes, target = ds["col_mes"], ds["target_col"]
    requeridas = vcfg["columnas_requeridas"]
    errores, advertencias, detalle = [], [], []

    archivos = sorted(glob.glob(os.path.join(input_path, "*.csv"))) if os.path.isdir(input_path) \
        else ([input_path] if os.path.exists(input_path) else [])
    if not archivos:
        return {"valido": False, "errores": [f"No hay archivos CSV en {input_path}"],
                "advertencias": [], "detalle": []}

    columnas_ref = None
    total_celdas = total_nulos = total_filas = 0
    mes_por_particion = {}
    nulos_por_columna = {}

    for ruta in archivos:
        nombre = os.path.basename(ruta)
        print(f"[validacion] Revisando {nombre} ...")
        df = pd.read_csv(ruta, low_memory=False)
        cols = list(df.columns)

        # 2. Esquema
        if columnas_ref is None:
            columnas_ref = cols
        elif cols != columnas_ref:
            faltan = sorted(set(columnas_ref) - set(cols))
            sobran = sorted(set(cols) - set(columnas_ref))
            errores.append(f"{nombre}: esquema distinto al primer archivo (faltan {faltan}, sobran {sobran})")
        faltan_req = [c for c in requeridas if c not in cols]
        if faltan_req:
            errores.append(f"{nombre}: faltan columnas obligatorias {faltan_req}")
            continue

        # 3. Nulos
        nulos = df.isna().sum()
        for c, n in nulos.items():
            nulos_por_columna[c] = nulos_por_columna.get(c, 0) + int(n)
        total_nulos += int(nulos.sum())
        total_celdas += df.size
        total_filas += len(df)
        for c in requeridas:
            if nulos[c] > 0:
                errores.append(f"{nombre}: la columna obligatoria '{c}' tiene {int(nulos[c])} nulos")

        # 4. Fechas / particiones
        particiones = df["partition"].dropna().unique().tolist()
        meses = df[col_mes].dropna().unique().tolist()
        if len(particiones) != 1 or len(meses) != 1:
            errores.append(f"{nombre}: deberia tener 1 particion y 1 mes, tiene "
                           f"{len(particiones)} particion(es) y {len(meses)} mes(es)")
        for p in particiones:
            mes_por_particion.setdefault(str(p), set()).update(int(m) for m in meses)
        for m in meses:
            fecha = pd.to_datetime(str(int(m)), format="%Y%m%d", errors="coerce")
            if pd.isna(fecha):
                errores.append(f"{nombre}: {col_mes}={m} no es una fecha YYYYMMDD valida")
            elif not (vcfg["fecha_min"] <= int(m) <= vcfg["fecha_max"]):
                errores.append(f"{nombre}: {col_mes}={int(m)} fuera del rango "
                               f"[{vcfg['fecha_min']}, {vcfg['fecha_max']}]")

        # 5. Target binario
        valores_target = set(df[target].dropna().unique().tolist())
        if not valores_target.issubset({0, 1}):
            errores.append(f"{nombre}: target tiene valores distintos de 0/1: {sorted(valores_target)[:5]}")

        detalle.append({
            "archivo": nombre, "filas": int(len(df)), "columnas": len(cols),
            "particiones": [str(p) for p in particiones], "meses": [int(m) for m in meses],
            "pct_nulos": round(float(nulos.sum() / df.size * 100), 2),
            "tasa_target": round(float(df[target].mean()), 4),
        })
        del df

    # 1. Particiones esperadas
    esperadas = set(vcfg["particiones_esperadas"])
    encontradas = set(mes_por_particion)
    if encontradas != esperadas:
        errores.append(f"Particiones esperadas {sorted(esperadas)}; faltan "
                       f"{sorted(esperadas - encontradas)}, sobran {sorted(encontradas - esperadas)}")
    meses_unicos = sorted({m for ms in mes_por_particion.values() for m in ms})
    if len(meses_unicos) != len(mes_por_particion):
        errores.append("Hay particiones que comparten el mismo mes o un mes repartido en varias particiones")

    # 3b. Nulos global y columnas con muchos nulos (solo aviso: preprocessing las descarta)
    pct_global = round(total_nulos / total_celdas * 100, 2) if total_celdas else 0.0
    if pct_global > vcfg["max_pct_nulos_global"]:
        errores.append(f"% de nulos global {pct_global}% > maximo permitido {vcfg['max_pct_nulos_global']}%")
    umbral_desc = ds["umbral_descarte_nulos"]
    muy_nulas = {c: round(n / total_filas * 100, 1) for c, n in nulos_por_columna.items()
                 if total_filas and n / total_filas * 100 > umbral_desc}
    if muy_nulas:
        advertencias.append(f"Columnas con mas de {umbral_desc}% de nulos (preprocessing las descarta): {muy_nulas}")

    return {
        "valido": not errores,
        "errores": errores,
        "advertencias": advertencias,
        "resumen": {
            "n_archivos": len(archivos),
            "n_particiones": len(encontradas),
            "n_filas": total_filas,
            "pct_nulos_global": pct_global,
            "rango_fechas": [meses_unicos[0], meses_unicos[-1]] if meses_unicos else None,
        },
        "detalle": detalle,
    }
