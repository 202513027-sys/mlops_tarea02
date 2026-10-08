"""
preprocessing.py
=================
Carga el dataset, lo limpia, codifica variables categoricas, normaliza
las variables numericas y retorna los arrays de entrenamiento, prueba y
validacion temporal listos para entrenar.

Extension P2: el dataset tiene varias particiones mensuales (columna
'partition', p1..p14) identificadas por 'p_fecinformacion' (YYYYMMDD).
Para que el backtesting walk-forward (backtesting.py) tenga meses futuros
genuinamente no vistos por el modelo, este modulo separa el dataset en:
  - meses 1..cutoff  -> alimentan el pipeline de la P1 (train/test/val)
  - meses cutoff+1.. -> se devuelven crudos ("df_futuro_raw"), sin tocar,
    para que backtesting.py los transforme con los MISMOS encoders/scaler
    ya ajustados (nunca se re-entrenan con datos futuros).
"""

import glob
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


def cargar_datos(input_path: str) -> pd.DataFrame:
    """Carga el dataset crudo desde disco.

    Soporta dos casos:
      - input_path apunta a UN archivo CSV: lo carga directo.
      - input_path apunta a una CARPETA con varios CSV particionados
        (ej. p1.csv, p2.csv, ..., p14.csv): los carga todos y los concatena.

    A diferencia de la P1, la columna 'partition' (p1, p2, ...) ya NO se
    descarta aqui: en la P2 es la unidad de tiempo que usa el backtesting
    walk-forward (junto con 'p_fecinformacion'). Se sigue excluyendo de las
    features del modelo mas adelante, via config['dataset']['cols_excluir'].

    Parametros
    ----------
    input_path : str
        Ruta a un archivo CSV, o a una carpeta que contiene varios CSV
        con el mismo esquema de columnas.

    Retorna
    -------
    pd.DataFrame con los datos crudos, ya unidos si eran varias particiones.
    """
    if os.path.isdir(input_path):
        archivos = sorted(glob.glob(os.path.join(input_path, "*.csv")))
        if not archivos:
            raise FileNotFoundError(f"No se encontraron archivos .csv en: {input_path}")

        print(f"[preprocessing] {len(archivos)} particiones encontradas en {input_path}, uniendo...")
        partes = []
        for archivo in archivos:
            parte = pd.read_csv(archivo)
            print(f"  - {os.path.basename(archivo)}: {parte.shape[0]} filas")
            partes.append(parte)
        df = pd.concat(partes, ignore_index=True)
    else:
        df = pd.read_csv(input_path)

    print(f"[preprocessing] Dataset cargado: {df.shape[0]} filas, {df.shape[1]} columnas")
    return df


def resumen_nulos(df: pd.DataFrame, umbral: float = 0.0) -> pd.DataFrame:
    """Calcula el % de nulos por columna, ordenado de mayor a menor.

    Solo se retornan columnas con % de nulos mayor a este umbral.
    """
    pct_nan = (df.isna().sum() / len(df) * 100).sort_values(ascending=False)
    return pct_nan[pct_nan > umbral].rename("pct_nulos").to_frame()


def eliminar_columnas(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """Elimina columnas si existen en el DataFrame (identificadoras o con muchos nulos)."""
    return df.drop(columns=[c for c in cols if c in df.columns])


def imputar_nulos_numericos(df: pd.DataFrame, cols: list, valor: float = 0.0) -> pd.DataFrame:
    """Imputa nulos numericos con un valor fijo (por defecto: ausencia = 0)."""
    df = df.copy()
    for col in cols:
        if col in df.columns:
            df[col] = df[col].fillna(valor)
    return df


def imputar_nulos_categoricos(df: pd.DataFrame, cols: list, valor: str = "Otros") -> pd.DataFrame:
    """Imputa nulos categoricos con una categoria por defecto (criterio de negocio)."""
    df = df.copy()
    for col in cols:
        if col in df.columns:
            df[col] = df[col].fillna(valor)
    return df


def aplicar_label_encoding(df: pd.DataFrame, cols: list) -> tuple:
    """Aplica LabelEncoder a cada columna categorica.

    Retorna
    -------
    (df_codificado, dict de encoders por columna). Los encoders se retornan
    para poder guardarlos: en produccion hay que reutilizar el MISMO encoder,
    nunca entrenar uno nuevo con datos nuevos (incluye el uso en backtesting.py).
    """
    df = df.copy()
    encoders = {}
    for col in cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
    return df, encoders


def transform_label_encoding_seguro(serie: pd.Series, encoder: LabelEncoder) -> pd.Series:
    """Aplica un LabelEncoder YA AJUSTADO a datos nuevos (ej. un mes futuro del
    backtesting), sin volver a entrenarlo.

    LabelEncoder.transform() lanza un error si aparece una categoria que el
    encoder nunca vio en entrenamiento (ej. un codigo de region nuevo el mes
    siguiente). En un sistema real esto puede pasar, asi que las categorias
    desconocidas se mapean a un valor de reserva (len(classes_)) en vez de
    tumbar el pipeline -- es una limitacion conocida y documentada, no un bug.
    """
    clases = {c: i for i, c in enumerate(encoder.classes_)}
    desconocido = len(encoder.classes_)
    return serie.astype(str).map(lambda v: clases.get(v, desconocido))


def normalizar(X_train: pd.DataFrame, X_test: pd.DataFrame, X_val: pd.DataFrame) -> tuple:
    """Normaliza (media 0, desviacion 1) las variables numericas.

    El StandardScaler se AJUSTA solo con X_train y se aplica despues a
    test/val, para no filtrar informacion de esos conjuntos al entrenamiento.

    Retorna
    -------
    (X_train_norm, X_test_norm, X_val_norm, scaler)
    """
    scaler = StandardScaler()
    X_train_norm = pd.DataFrame(
        scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index
    )
    X_test_norm = pd.DataFrame(
        scaler.transform(X_test), columns=X_test.columns, index=X_test.index
    )
    X_val_norm = pd.DataFrame(
        scaler.transform(X_val), columns=X_val.columns, index=X_val.index
    )
    return X_train_norm, X_test_norm, X_val_norm, scaler


def _separar_por_cutoff(df_raw: pd.DataFrame, col_mes: str, cutoff_partition: int) -> tuple:
    """Divide el dataset crudo en (df_conocido, df_futuro, meses_ordenados) segun
    el cutoff de particiones del backtesting.

    meses_ordenados es la lista de valores UNICOS de col_mes, ordenados de mas
    antiguo a mas reciente -- esa posicion (1-indexada) es lo que config.yaml
    llama "particion": la posicion 1 es p1, la 2 es p2, etc., sin depender del
    nombre de archivo (evita bugs de orden lexicografico con p10, p11...).

    Parametros
    ----------
    cutoff_partition : int
        Cuantos meses (desde el mas antiguo) entran al pipeline de
        entrenamiento P1 (train/test/val). El resto queda crudo para backtesting.
    """
    meses_ordenados = sorted(df_raw[col_mes].unique())
    cutoff_partition = min(cutoff_partition, len(meses_ordenados))
    meses_conocidos = set(meses_ordenados[:cutoff_partition])

    df_conocido = df_raw[df_raw[col_mes].isin(meses_conocidos)].copy()
    df_futuro = df_raw[~df_raw[col_mes].isin(meses_conocidos)].copy()

    print(f"[preprocessing] Cutoff aplicado: {cutoff_partition} particion(es) conocidas "
          f"({df_conocido.shape[0]} filas) | {len(meses_ordenados) - cutoff_partition} "
          f"particion(es) reservadas para backtesting ({df_futuro.shape[0]} filas)")

    return df_conocido, df_futuro, meses_ordenados


def preprocesar(config: dict) -> dict:
    """Pipeline completo: carga, corte temporal por cutoff, limpieza, encoding,
    particion y normalizacion.

    Parametros
    ----------
    config : dict
        Diccionario cargado desde config/config.yaml.

    Retorna
    -------
    dict con las versiones SIN normalizar (para modelos de arboles) y
    normalizadas (para modelos lineales, de instancias y redes neuronales)
    de train/test/val, ademas de encoders, scaler, feature_cols, y los
    datos crudos reservados para el backtesting ("df_futuro_raw",
    "meses_ordenados", "cutoff_aplicado").
    """
    ds_cfg = config["dataset"]
    split_cfg = config["split"]
    bt_cfg = config.get("backtesting", {})

    df_raw_completo = cargar_datos(ds_cfg["input_path"])

    cutoff_partition = bt_cfg.get("train_cutoff_partition", 999)
    df_raw, df_futuro_raw, meses_ordenados = _separar_por_cutoff(
        df_raw_completo, ds_cfg["col_mes"], cutoff_partition
    )
    cutoff_aplicado = min(cutoff_partition, len(meses_ordenados))

    df_proc = df_raw.copy()

    # 1. Columnas con demasiados nulos -> se descartan junto con las identificadoras
    nulos_full = resumen_nulos(df_raw, umbral=ds_cfg["umbral_descarte_nulos"])
    cols_alto_nulo = nulos_full.index.tolist()
    print(f"[preprocessing] Columnas descartadas por nulos "
          f"(>{ds_cfg['umbral_descarte_nulos']}%): {len(cols_alto_nulo)}")

    cols_excluir = list(ds_cfg["cols_excluir"])
    if "partition" not in cols_excluir:
        cols_excluir.append("partition")

    df_proc = eliminar_columnas(df_proc, cols_excluir + cols_alto_nulo)

    # 2. Imputar numericos restantes con 0 (ausencia = 0, criterio de negocio)
    cols_num_nan = (
        df_proc.select_dtypes(include=[np.number])
        .columns[df_proc.select_dtypes(include=[np.number]).isna().any()]
        .tolist()
    )
    if "edad" in cols_num_nan:
        cols_num_nan.remove("edad")
        df_proc["edad"] = df_proc["edad"].fillna(df_proc["edad"].median())

    df_proc = imputar_nulos_numericos(df_proc, cols_num_nan, valor=0.0)

    # 3. Imputar categoricas restantes
    cols_categoricas = df_proc.select_dtypes(include="object").columns.tolist()
    df_proc = imputar_nulos_categoricos(
        df_proc, cols_categoricas, valor=ds_cfg["valor_imputacion_categorica"]
    )

    # 4. Encoding de categoricas (se guardan los encoders, se reusan en backtesting)
    cols_cat_final = df_proc.select_dtypes(include="object").columns.tolist()
    df_proc, encoders = aplicar_label_encoding(df_proc, cols_cat_final)

    print(f"[preprocessing] Shape final tras limpieza: {df_proc.shape}")
    print(f"[preprocessing] Nulos restantes: {df_proc.isna().sum().sum()}")

    # 5. Particion temporal dentro de los meses conocidos (train/test/val P1)
    df_proc = df_proc.copy()
    df_proc["_mes_orig"] = df_raw[ds_cfg["col_mes"]].values

    mes_validacion = ds_cfg.get("mes_validacion")
    meses_conocidos_ordenados = sorted(df_raw[ds_cfg["col_mes"]].unique())
    if mes_validacion is None or mes_validacion not in meses_conocidos_ordenados:
        # Auto: si no se especifico (o cae fuera del rango conocido tras el
        # cutoff), se usa el mes mas reciente DENTRO de lo conocido como
        # validacion temporal -- asi el cutoff no obliga a editar este valor
        # a mano cada vez que cambia la cantidad de particiones disponibles.
        mes_validacion = meses_conocidos_ordenados[-1]
        print(f"[preprocessing] mes_validacion no especificado o fuera de rango: "
              f"se usa automaticamente el mas reciente dentro del cutoff ({mes_validacion})")

    es_validacion = df_proc["_mes_orig"] == mes_validacion
    df_val = df_proc[es_validacion].drop(columns=["_mes_orig"])
    df_main = df_proc[~es_validacion].drop(columns=["_mes_orig"])

    print(f"[preprocessing] Train+Test: {df_main.shape} | Validacion temporal: {df_val.shape}")

    target_col = ds_cfg["target_col"]
    feature_cols = [c for c in df_main.columns if c != target_col]

    X_main = df_main[feature_cols]
    y_main = df_main[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X_main, y_main,
        test_size=split_cfg["test_size"],
        random_state=split_cfg["random_state"],
        stratify=y_main,
    )

    X_val = df_val[feature_cols]
    y_val = df_val[target_col]

    # 6. Normalizacion (para modelos lineales, de instancias y redes neuronales)
    X_train_norm, X_test_norm, X_val_norm, scaler = normalizar(X_train, X_test, X_val)

    return {
        "X_train": X_train, "X_test": X_test, "X_val": X_val,
        "X_train_norm": X_train_norm, "X_test_norm": X_test_norm, "X_val_norm": X_val_norm,
        "y_train": y_train, "y_test": y_test, "y_val": y_val,
        "encoders": encoders,
        "scaler": scaler,
        "feature_cols": feature_cols,
        "cols_excluir": cols_excluir,
        "cols_alto_nulo": cols_alto_nulo,
        "cols_categoricas_codificadas": cols_cat_final,
        "valor_imputacion_categorica": ds_cfg["valor_imputacion_categorica"],
        "mes_validacion_usado": mes_validacion,
        # Reservado para backtesting.py: meses futuros NUNCA vistos por el modelo.
        "df_futuro_raw": df_futuro_raw,
        "meses_ordenados": meses_ordenados,
        "cutoff_aplicado": cutoff_aplicado,
        "col_mes": ds_cfg["col_mes"],
        "target_col": target_col,
    }
