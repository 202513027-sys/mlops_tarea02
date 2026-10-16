"""
Pruebas unitarias de la metrica de negocio y de GetBestModel() (src/evaluation.py).

Como correrlas (desde la carpeta raiz del proyecto):
    pip install pytest numpy scikit-learn
    python -m pytest tests -v

No necesitan el dataset, ni Docker, ni MLflow: usan datos inventados pequenos.
"""

import numpy as np
import pytest

from src.evaluation import GetBestModel, calcular_decay, metricas_negocio


# ---------------------------------------------------------------------------
# Datos de prueba: 1000 clientes, 100 compran (tasa base = 10 %)
# ---------------------------------------------------------------------------
@pytest.fixture
def y_real():
    y = np.zeros(1000, dtype=int)
    y[:100] = 1
    return y


def test_modelo_perfecto_tiene_lift_10(y_real):
    """Si el modelo pone a TODOS los compradores arriba, el top 10 % son solo
    compradores: tasa 100 % vs 10 % promedio -> lift = 10."""
    probas = y_real.astype(float)
    m = metricas_negocio(y_real, probas, top_pct=10)
    assert m["lift"] == pytest.approx(10.0)
    assert m["captura_ventas_top"] == pytest.approx(1.0)  # captura el 100 % de las ventas
    assert m["ks"] == pytest.approx(1.0)
    assert m["pr_auc"] == pytest.approx(1.0)


def test_modelo_al_azar_tiene_lift_cercano_a_1():
    """Un modelo que da probabilidades al azar no ayuda: lift ~ 1."""
    rng = np.random.default_rng(42)
    y = rng.binomial(1, 0.1, size=200_000)
    probas = rng.random(200_000)
    m = metricas_negocio(y, probas, top_pct=10)
    assert m["lift"] == pytest.approx(1.0, abs=0.1)


def test_modelo_al_reves_tiene_lift_0(y_real):
    """Si el modelo pone a los compradores al FINAL, el top 10 % no tiene ventas."""
    probas = 1 - y_real.astype(float)
    m = metricas_negocio(y_real, probas, top_pct=10)
    assert m["lift"] == pytest.approx(0.0)


def test_top_20_por_ciento(y_real):
    """Con top 20 %, el modelo perfecto mete 100 compradores en 200 contactos:
    tasa 50 % vs 10 % -> lift = 5."""
    m = metricas_negocio(y_real, y_real.astype(float), top_pct=20)
    assert m["lift"] == pytest.approx(5.0)
    assert m["tasa_respuesta_top"] == pytest.approx(0.5)


def test_tasa_base_es_la_proporcion_de_compradores(y_real):
    m = metricas_negocio(y_real, np.linspace(0, 1, 1000), top_pct=10)
    assert m["tasa_respuesta_base"] == pytest.approx(0.10)


def test_decay_relativo():
    """Decay relativo (D1): de AUC 0.90 a 0.81 se pierde el 10 %."""
    assert calcular_decay(0.90, 0.81) == pytest.approx(10.0)
    assert calcular_decay(0.80, 0.80) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# GetBestModel(): el negocio decide, pero el sobreajuste descalifica (D8)
# ---------------------------------------------------------------------------
CONFIG = {
    "selection": {"max_decay_pct": 10.0, "max_decay_backtest_pct": 5.0, "tolerancia_lift_pct": 2.0},
    "metrica_negocio": {"top_pct": 10},
}


def _modelo(nombre, decay, lift_bt, decay_bt=1.0, tiempo=10.0):
    """Metadata minima de un modelo, con el mismo formato que usa el pipeline."""
    return {
        "ml_name": nombre,
        "performance": {"decay_pct": decay, "training_time_sec": tiempo},
        "negocio_test": {"lift": lift_bt},
        "backtesting": {"avg_decay_backtest": decay_bt, "avg_lift_backtest": lift_bt,
                        "drift_severo_en_todas_las_olas": False},
    }


def test_gana_el_mayor_lift():
    sel = GetBestModel([_modelo("A", 2, 4.0), _modelo("B", 2, 5.0)], CONFIG)
    assert sel["campeon"]["ml_name"] == "B"
    assert sel["estado"] == "OK"


def test_modelo_sobreajustado_queda_fuera_aunque_tenga_mejor_lift():
    sel = GetBestModel([_modelo("Sobreajustado", 15, 9.0), _modelo("Sano", 2, 5.0)], CONFIG)
    assert sel["campeon"]["ml_name"] == "Sano"
    assert "Sobreajustado" in sel["descartados"]


def test_empate_de_lift_gana_el_mas_rapido():
    """A menos de 2 % del mejor lift, gana el que entrena en menos tiempo."""
    lento = _modelo("Lento", 2, 5.00, tiempo=60)
    rapido = _modelo("Rapido", 2, 4.95, tiempo=15)
    sel = GetBestModel([lento, rapido], CONFIG)
    assert sel["campeon"]["ml_name"] == "Rapido"


def test_sin_modelos_estables_no_se_promueve():
    """Si ninguno es estable en backtesting, hay campeon pero con alerta y sin promover."""
    sel = GetBestModel([_modelo("A", 2, 5.0, decay_bt=8.0)], CONFIG)
    assert sel["estado"] == "CON_ALERTA"
    assert sel["promover"] is False
