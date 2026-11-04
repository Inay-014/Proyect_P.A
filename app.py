"""
================================================================================
 Inmobiliaria Hábitat · ¿Se puede ver venir a quién dejará de pagar el arriendo?
================================================================================
Aplicación web interactiva (Streamlit + Plotly) que acompaña el caso Hábitat:
EDA de 100.000 contratos de arrendamiento (2019-2025), modelo de riesgo de mora
>90 días y simulador de las tres decisiones (aprobar, codeudor, cobrar).

CÓMO EJECUTAR EN LOCAL
    pip install -r requirements.txt
    streamlit run app.py

CÓMO DESPLEGAR EN STREAMLIT COMMUNITY CLOUD
    1. Suba a un repositorio de GitHub: app.py, requirements.txt,
       .streamlit/config.toml y (opcional) inmobiliaria_arriendos.csv.
    2. En share.streamlit.io → "New app" → elija el repo y "app.py".
    3. Si el CSV no está en el repo (p. ej., por confidencialidad), la app
       genera datos SINTÉTICOS con distribuciones coherentes con el caso y lo
       indica con un aviso. Las cifras ancla del documento solo se reproducen
       con el CSV real.

METODOLOGÍA (idéntica a los notebooks del equipo)
    · Limpieza: se eliminan 300 duplicados; el ingreso faltante se reconstruye
      como canon / ratio.
    · Modelo: regresión logística con 7 variables conocidas al firmar
      (puntaje, bandera de puntaje faltante, ratio acotado en 2, pagos tardíos
      acotados en 4, es_2020, tipo de garantía, tipo de inmueble).
    · Validación temporal: entrena 2019-2024, prueba una sola vez en 2025.
    · Umbral: maximiza el beneficio neto con predicciones fuera de pliegue
      (5 pliegues) de 2019-2024 y luego se aplica a 2025.
================================================================================
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ------------------------------------------------------------------------------
# 0. CONFIGURACIÓN GENERAL Y PALETA CORPORATIVA
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="Hábitat · Riesgo de mora",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

SEMILLA = 42
NOMBRE_CSV = "inmobiliaria_arriendos.csv"

# Paleta institucional: borgoña / rojo oscuro, grises suaves y blancos
BORGONA = "#7B1E2B"
ROJO = "#A93A3F"
ROSA = "#E7C6C9"
GRIS_OSCURO = "#3B3B3B"
GRIS = "#8C8C8C"
GRIS_CLARO = "#D9D6D3"
FONDO = "#FAF8F7"
VERDE = "#4F7A5A"  # solo para "mejora" (uso mínimo)

NUMERICAS = ["score_crediticio", "score_nulo", "ratio_acot", "pagos_acot", "es_2020"]
CATEGORICAS = ["tipo_garantia", "tipo_inmueble"]
GARANTIAS = ["Sin garantía", "Depósito", "Seguro de arrendamiento", "Codeudor"]
NIVELES = ["Bajo", "Medio", "Alto"]
TRAMOS_SCORE = ([0, 450, 550, 600, 650, 700, 750, 800, 1000],
                ["≤450", "451-550", "551-600", "601-650", "651-700", "701-750", "751-800", ">800"])
TRAMOS_RATIO = ([0, .2, .3, .4, .5, .7, 1, 5],
                ["≤20 %", "20-30 %", "30-40 %", "40-50 %", "50-70 %", "70-100 %", ">100 %"])
TRAMOS_INGRESO = ([0, 1.5e6, 3e6, 5e6, 8e6, np.inf],
                  ["Hasta 1,5 M", "1,5-3 M", "3-5 M", "5-8 M", "Más de 8 M"])

# Estilos CSS: tarjetas KPI, encabezado y paneles narrativos
st.markdown(
    f"""
    <style>
      .stApp {{ background-color: {FONDO}; }}
      .block-container {{ padding-top: 1.6rem; max-width: 1280px; }}
      h1, h2, h3, h4 {{ color: {GRIS_OSCURO}; letter-spacing: -0.01em; }}
      .hero {{
        background: linear-gradient(120deg, {BORGONA} 0%, #5A1520 100%);
        color: white; border-radius: 14px; padding: 30px 34px 26px 34px; margin-bottom: 18px;
      }}
      .hero .kicker {{ font-size: 0.78rem; letter-spacing: 0.14em; text-transform: uppercase; opacity: 0.8; }}
      .hero h1 {{ color: white; font-size: 2.15rem; line-height: 1.15; margin: 6px 0 8px 0; }}
      .hero p  {{ color: #F3E3E5; font-size: 1.02rem; margin: 0; }}
      .kpi {{
        background: white; border: 1px solid {GRIS_CLARO}; border-left: 5px solid {BORGONA};
        border-radius: 12px; padding: 16px 18px; height: 100%;
      }}
      .kpi .lbl {{ font-size: 0.78rem; color: {GRIS}; text-transform: uppercase; letter-spacing: 0.06em; }}
      .kpi .val {{ font-size: 1.95rem; font-weight: 700; color: {BORGONA}; line-height: 1.2; margin-top: 4px; }}
      .kpi .sub {{ font-size: 0.86rem; color: {GRIS_OSCURO}; margin-top: 2px; }}
      .card {{
        background: white; border: 1px solid {GRIS_CLARO}; border-radius: 12px;
        padding: 18px 20px; height: 100%;
      }}
      .card h4 {{ margin: 0 0 6px 0; font-size: 1.02rem; }}
      .card p, .card li {{ font-size: 0.93rem; color: {GRIS_OSCURO}; }}
      .card.si  {{ border-top: 5px solid {BORGONA}; }}
      .card.no  {{ border-top: 5px solid {GRIS}; }}
      .num {{
        display: inline-block; background: {BORGONA}; color: white; border-radius: 50%;
        width: 26px; height: 26px; text-align: center; line-height: 26px; font-weight: 700;
        margin-right: 8px; font-size: 0.9rem;
      }}
      .tag {{ display: inline-block; background: {ROSA}; color: {BORGONA}; border-radius: 999px;
              padding: 2px 10px; font-size: 0.75rem; font-weight: 600; }}
      .nota {{ font-size: 0.8rem; color: {GRIS}; }}
      div[data-testid="stMetricValue"] {{ color: {BORGONA}; }}
      .stTabs [data-baseweb="tab-list"] {{ gap: 6px; }}
      .stTabs [data-baseweb="tab"] {{
        background: white; border: 1px solid {GRIS_CLARO}; border-radius: 8px 8px 0 0; padding: 8px 16px;
      }}
      .stTabs [aria-selected="true"] {{ border-bottom: 3px solid {BORGONA}; color: {BORGONA}; font-weight: 600; }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ------------------------------------------------------------------------------
# 1. UTILIDADES DE FORMATO (convención colombiana: coma decimal, punto de miles)
# ------------------------------------------------------------------------------
def num(x: float, d: int = 0) -> str:
    """Formatea con punto de miles y coma decimal: 2181.4 -> '2.181'."""
    s = f"{x:,.{d}f}"
    return s.replace(",", "X").replace(".", ",").replace("X", ".")


def pct(x: float, d: int = 1) -> str:
    """Formatea una proporción (0-1) como porcentaje: 0.0877 -> '8,8 %'."""
    return f"{num(x * 100, d)} %"


def mostrar(fig: go.Figure) -> None:
    """Dibuja un gráfico a lo ancho, compatible con versiones nuevas y antiguas de Streamlit."""
    try:
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    except Exception:  # versiones antiguas de Streamlit
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def tabla(data) -> None:
    """Muestra una tabla a lo ancho (compatible con versiones nuevas y antiguas de Streamlit)."""
    try:
        st.dataframe(data, hide_index=True, width="stretch")
    except Exception:  # versiones antiguas de Streamlit
        st.dataframe(data, hide_index=True, use_container_width=True)


def estilo(fig: go.Figure, titulo: str = "", alto: int = 360, yfmt: str | None = None) -> go.Figure:
    """Plantilla visual común para todos los gráficos."""
    fig.update_layout(
        title=dict(text=titulo, font=dict(size=15, color=GRIS_OSCURO), x=0, xanchor="left"),
        height=alto, margin=dict(l=10, r=10, t=50 if titulo else 20, b=10),
        plot_bgcolor="white", paper_bgcolor="white",
        font=dict(family="Inter, Segoe UI, Arial, sans-serif", color=GRIS_OSCURO, size=12),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1, title=None),
        hoverlabel=dict(bgcolor="white", font_size=12),
    )
    fig.update_xaxes(showgrid=False, linecolor=GRIS_CLARO)
    fig.update_yaxes(gridcolor="#EFECEA", zeroline=False)
    if yfmt:
        fig.update_yaxes(ticksuffix=yfmt)
    return fig


def kpi(etiqueta: str, valor: str, sub: str = "") -> str:
    return f'<div class="kpi"><div class="lbl">{etiqueta}</div><div class="val">{valor}</div><div class="sub">{sub}</div></div>'


# ------------------------------------------------------------------------------
# 2. DATOS: CSV real o, si no existe, datos sintéticos coherentes con el caso
# ------------------------------------------------------------------------------
def generar_sinteticos(n: int = 100_000, seed: int = SEMILLA) -> pd.DataFrame:
    """
    Genera contratos sintéticos que imitan las distribuciones y relaciones del caso:
    mora ≈ 8,8 %; el puntaje y el ratio canon/ingreso concentran el riesgo; ciudad,
    estrato y canal NO cambian el riesgo. Solo se usa si falta el CSV real.
    """
    rng = np.random.default_rng(seed)
    elegir = lambda opciones, p: rng.choice(opciones, size=n, p=p)  # noqa: E731

    ciudad = elegir(["Bogotá", "Medellín", "Cali", "Barranquilla", "Bucaramanga", "Cartagena"],
                    [.379, .221, .148, .102, .081, .069])
    estrato = elegir([1, 2, 3, 4, 5, 6], [.05, .18, .319, .25, .141, .06])
    canal = elegir(["Portal web", "Referido", "Vitrina", "Redes sociales"], [.402, .25, .198, .15])
    inmueble = elegir(["Apartamento", "Casa", "Local comercial", "Oficina", "Bodega"], [.498, .201, .152, .099, .05])
    garantia = elegir(GARANTIAS, [.08, .121, .201, .598])
    meses = elegir([6, 12, 18, 24], [.101, .649, .05, .2])
    anio = rng.integers(2019, 2026, n)

    score = np.clip(rng.normal(660, 104, n), 300, 950).round()
    ratio = np.clip(np.exp(rng.normal(np.log(0.31), 0.28, n)), 0.138, None)
    # Cola de subdeclaración del ingreso: ≈0,6 % con canon > ingreso
    cola = rng.random(n) < 0.0056
    ratio[cola] = rng.uniform(1.0, 2.5, cola.sum())
    canon = np.clip(np.exp(rng.normal(np.log(980_000), 0.5, n)), 300_000, None).round(-4)
    ingreso = (canon / ratio).round(-4)
    pagos = np.minimum(rng.poisson(0.5, n), 6)

    # Mecanismo de riesgo (odds ratios del documento: ratio +0,1 → ×1,73; score +50 → ×0,735 ...)
    ef_gar = {"Sin garantía": 0.0, "Depósito": np.log(.53), "Seguro de arrendamiento": np.log(.52), "Codeudor": np.log(.32)}
    ef_inm = {"Apartamento": 0.0, "Casa": 0.05, "Oficina": 0.0, "Local comercial": np.log(1.63), "Bodega": np.log(1.74)}
    logit = (np.log(1.73) * np.minimum(ratio, 2) / 0.1 + np.log(0.735) * (score - 650) / 50
             + np.log(1.53) * np.minimum(pagos, 4) + 0.10 * (anio == 2020)
             + np.vectorize(ef_gar.get)(garantia) + np.vectorize(ef_inm.get)(inmueble))
    # Intercepto calibrado por bisección para una mora global ≈ 8,77 %
    lo, hi = -30.0, 10.0
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if (1 / (1 + np.exp(-(logit + mid)))).mean() < 0.0877 else (lo, mid)
    mora = (rng.random(n) < 1 / (1 + np.exp(-(logit + lo)))).astype(int)

    df = pd.DataFrame({
        "id_contrato": np.arange(1, n + 1), "anio_inicio": anio, "ciudad": ciudad, "estrato": estrato,
        "tipo_inmueble": inmueble, "canon_mensual": canon, "ingreso_arrendatario": ingreso,
        "ratio_canon_ingreso": ratio.round(3), "tiene_codeudor": (garantia == "Codeudor").astype(int),
        "tipo_garantia": garantia, "score_crediticio": score, "meses_contrato": meses,
        "pagos_tardios_previos": pagos, "canal_captacion": canal, "mora_90dias": mora,
    })
    # Faltantes como en el original (≈5 % en ingreso y puntaje)
    df.loc[rng.random(n) < 0.05, "score_crediticio"] = np.nan
    df.loc[rng.random(n) < 0.05, "ingreso_arrendatario"] = np.nan
    return df


@st.cache_data(show_spinner="Cargando y limpiando los contratos…")
def cargar_datos() -> tuple[pd.DataFrame, bool]:
    """Devuelve (datos limpios, es_real). Busca el CSV junto a app.py o en ./data/."""
    base = Path(__file__).parent
    candidatos = [base / NOMBRE_CSV, base / "data" / NOMBRE_CSV, Path(os.getenv("HABITAT_CSV", "_no_existe_"))]
    ruta = next((p for p in candidatos if p.exists()), None)
    if ruta is not None:
        df, es_real = pd.read_csv(ruta), True
    else:
        df, es_real = generar_sinteticos(), False

    # --- Limpieza (EDA, pasos 1-2) ---
    df = df.drop_duplicates().reset_index(drop=True)  # 300 filas duplicadas exactas
    df["ingreso_arrendatario"] = df["ingreso_arrendatario"].fillna(df["canon_mensual"] / df["ratio_canon_ingreso"])

    # --- Variables derivadas del modelo (no usan información de otros contratos: sin fuga) ---
    df["score_nulo"] = df["score_crediticio"].isna().astype(int)
    df["ratio_acot"] = df["ratio_canon_ingreso"].clip(upper=2)
    df["pagos_acot"] = df["pagos_tardios_previos"].clip(upper=4)
    df["es_2020"] = (df["anio_inicio"] == 2020).astype(int)

    # --- Tramos para el EDA ---
    df["tramo_score"] = pd.cut(df["score_crediticio"], TRAMOS_SCORE[0], labels=TRAMOS_SCORE[1])
    df["tramo_ratio"] = pd.cut(df["ratio_canon_ingreso"], TRAMOS_RATIO[0], labels=TRAMOS_RATIO[1])
    df["tramo_ingreso"] = pd.cut(df["ingreso_arrendatario"], TRAMOS_INGRESO[0], labels=TRAMOS_INGRESO[1])
    df["pagos_tardios_txt"] = df["pagos_tardios_previos"].clip(upper=3).map({0: "0", 1: "1", 2: "2", 3: "3 o más"})
    return df, es_real


# ------------------------------------------------------------------------------
# 3. MODELO (se entrena una sola vez por sesión del servidor gracias al caché)
# ------------------------------------------------------------------------------
def tuberia(categoricas: list[str] = CATEGORICAS) -> Pipeline:
    """Imputa el puntaje con la mediana de entrenamiento, escala y codifica categorías."""
    prep = ColumnTransformer([
        ("num", Pipeline([("rellenar", SimpleImputer(strategy="median")), ("escalar", StandardScaler())]), NUMERICAS),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categoricas),
    ])
    return Pipeline([("preparacion", prep), ("modelo", LogisticRegression(max_iter=3000))])


def _diseno_interaccion(g: np.ndarray, n: np.ndarray, lg: np.ndarray) -> pd.DataFrame:
    """Matriz para la logística con interacción garantía × nivel de riesgo."""
    D = {"logit_riesgo_base": lg}
    for gg in GARANTIAS[1:]:
        for nn in NIVELES:
            D[f"{gg}|{nn}"] = ((g == gg) & (n == nn)).astype(float)
    for nn in NIVELES[1:]:
        D[f"nivel_{nn}"] = (n == nn).astype(float)
    return pd.DataFrame(D)


@st.cache_resource(show_spinner="Entrenando el modelo de riesgo (solo la primera vez)…")
def entrenar(df: pd.DataFrame) -> dict:
    """Entrena con 2019-2024, evalúa en 2025 y prepara todo lo que usa el simulador."""
    tr, te = df[df["anio_inicio"] <= 2024], df[df["anio_inicio"] == 2025]
    y_tr, y_te = tr["mora_90dias"].values, te["mora_90dias"].values
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEMILLA)

    # (a) Modelo final y predicciones fuera de pliegue para elegir el umbral
    modelo = tuberia()
    prob_oof = cross_val_predict(modelo, tr, y_tr, cv=cv, method="predict_proba")[:, 1]
    modelo.fit(tr, y_tr)
    p_te = modelo.predict_proba(te)[:, 1]

    # (b) Riesgo base SIN garantía → niveles bajo (50 %), medio (30 %), alto (20 %)
    modelo_base = tuberia(["tipo_inmueble"])
    riesgo_base_oof = cross_val_predict(modelo_base, tr, y_tr, cv=cv, method="predict_proba")[:, 1]
    corte_50, corte_80 = np.quantile(riesgo_base_oof, [0.5, 0.8])
    modelo_base.fit(tr, y_tr)
    a_nivel = lambda r: np.where(r < corte_50, "Bajo", np.where(r < corte_80, "Medio", "Alto"))  # noqa: E731
    nivel_tr = a_nivel(riesgo_base_oof)
    nivel_te = a_nivel(modelo_base.predict_proba(te)[:, 1])

    # (c) Efecto de cada garantía por nivel (efecto marginal promedio, logística con interacción)
    lg = np.log(riesgo_base_oof / (1 - riesgo_base_oof))
    g_tr = tr["tipo_garantia"].values
    m_int = LogisticRegression(C=1e6, max_iter=2000).fit(_diseno_interaccion(g_tr, nivel_tr, lg), y_tr)
    filas = []
    for nn in NIVELES:
        msk = nivel_tr == nn
        for gg in GARANTIAS:
            X = _diseno_interaccion(np.full(msk.sum(), gg, dtype=object), nivel_tr[msk], lg[msk])
            filas.append({"nivel": nn, "garantia": gg, "mora": m_int.predict_proba(X)[:, 1].mean()})
    efectos = pd.DataFrame(filas)

    return {
        "modelo": modelo, "te": te.reset_index(drop=True), "p_te": p_te, "y_te": y_te,
        "prob_oof": prob_oof, "y_tr": y_tr,
        "canon_tr": tr["canon_mensual"].values, "meses_tr": tr["meses_contrato"].values,
        "canon_te": te["canon_mensual"].values, "meses_te": te["meses_contrato"].values,
        "nivel_te": nivel_te, "efectos": efectos, "cortes": (corte_50, corte_80), "modelo_base": modelo_base,
        "roc_auc": roc_auc_score(y_te, p_te), "pr_auc": average_precision_score(y_te, p_te),
        "n_tr": len(tr), "n_te": len(te),
    }


FRACCIONES = np.round(np.arange(0, 0.4001, 0.005), 3)  # rechazar del 0 % al 40 % de mayor riesgo


def curva_beneficio(p, y, canon, meses, L, c, fracciones=FRACCIONES) -> np.ndarray:
    """Beneficio neto (COP) de rechazar el f % de mayor riesgo = pérdidas evitadas − comisiones perdidas."""
    orden = np.argsort(-p)
    ahorro = np.cumsum(L * canon[orden] * (y[orden] == 1))
    perdida = np.cumsum(c * canon[orden] * meses[orden] * (y[orden] == 0))
    k = (fracciones * len(p)).astype(int)
    return np.where(k == 0, 0.0, (ahorro - perdida)[np.maximum(k - 1, 0)])


def umbral_optimo(M: dict, L: float, c: float) -> float:
    """Elige el umbral con 2019-2024 (fuera de pliegue), sin mirar 2025."""
    curva = curva_beneficio(M["prob_oof"], M["y_tr"], M["canon_tr"], M["meses_tr"], L, c)
    f_opt = FRACCIONES[np.argmax(curva)]
    return float(np.sort(M["prob_oof"])[::-1][max(int(len(M["prob_oof"]) * f_opt) - 1, 0)])


def resultado_politica(rechazar: np.ndarray, M: dict, L: float, c: float) -> dict:
    """Resultado de una política de rechazo aplicada a los contratos de 2025."""
    y, canon, meses = M["y_te"], M["canon_te"], M["meses_te"]
    ahorro = (L * canon * (rechazar & (y == 1))).sum()
    perdida = (c * canon * meses * (rechazar & (y == 0))).sum()
    return {
        "rechazo": rechazar.mean(),
        "evitados": int((rechazar & (y == 1)).sum()),
        "perdidos": int((rechazar & (y == 0)).sum()),
        "mora_aprobados": y[~rechazar].mean() if (~rechazar).any() else 0.0,
        "beneficio_M": (ahorro - perdida) / 1e6,
    }


def tasa_por(d: pd.DataFrame, col: str) -> pd.DataFrame:
    """Tasa de mora por categoría con intervalo de confianza del 95 % (aprox. normal)."""
    t = d.groupby(col, observed=True)["mora_90dias"].agg(contratos="size", tasa="mean").reset_index()
    t["ic"] = 1.96 * np.sqrt(t["tasa"] * (1 - t["tasa"]) / t["contratos"].clip(lower=1))
    return t


# ------------------------------------------------------------------------------
# 4. CARGA INICIAL
# ------------------------------------------------------------------------------
df, ES_REAL = cargar_datos()
M = entrenar(df)

TASA_GLOBAL = df["mora_90dias"].mean()
UNO_DE_CADA = 1 / TASA_GLOBAL
U_BASE = umbral_optimo(M, 5, 0.10)
RES_BASE = resultado_politica(M["p_te"] >= U_BASE, M, 5, 0.10)
MORA_2025 = M["y_te"].mean()
ORDEN_TE = np.argsort(-M["p_te"])


def captura(frac: float) -> tuple[float, float, float]:
    """(captura de la mora, lift, precisión) del top `frac` de contratos 2025 por riesgo."""
    k = max(int(len(ORDEN_TE) * frac), 1)
    y_top = M["y_te"][ORDEN_TE[:k]]
    cap = y_top.sum() / M["y_te"].sum()
    return cap, y_top.mean() / MORA_2025, y_top.mean()


# ------------------------------------------------------------------------------
# 5. BARRA LATERAL: fuente de datos y supuestos
# ------------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Acerca de esta app")
    st.markdown(
        f"**Datos:** {'CSV real del caso' if ES_REAL else 'sintéticos (falta el CSV)'} · "
        f"{num(len(df))} contratos únicos.\n\n"
        f"**Modelo:** regresión logística, 7 variables. Entrena 2019-2024 "
        f"({num(M['n_tr'])}) y se prueba en 2025 ({num(M['n_te'])}).\n\n"
        f"**ROC-AUC 2025:** {num(M['roc_auc'], 3)} · **PR-AUC:** {num(M['pr_auc'], 3)} "
        f"(azar = {num(MORA_2025, 3)})."
    )
    st.markdown("---")
    st.markdown("**Supuestos financieros (a validar con Finanzas)**")
    st.markdown("- Pérdida por moroso: 5 cánones (escenario base; 3 conservador, 8 severo)\n"
                "- Comisión de Hábitat: 10 % del canon\n"
                "- Cartera vigente: ≈9.000 contratos; equipo de cobranza: 4 gestores")
    st.markdown("---")
    st.caption("Programación Aplicada · Universidad de La Sabana · 2026")

if not ES_REAL:
    st.warning(
        f"No se encontró **{NOMBRE_CSV}** junto a app.py. Se muestran **datos sintéticos** "
        "con distribuciones coherentes con el caso; las cifras exactas del documento solo se reproducen con el CSV real.",
        icon="⚠️",
    )

# ------------------------------------------------------------------------------
# 6. PORTADA (HEADER + KPIs)
# ------------------------------------------------------------------------------
st.markdown(
    f"""
    <div class="hero">
      <div class="kicker">Inmobiliaria Hábitat · Riesgo de arrendamiento</div>
      <h1>¿Se puede ver venir a quién dejará de pagar el arriendo?</h1>
      <p>Análisis de {num(len(df))} contratos de arrendamiento (2019-2025): qué predice la mora de más de
      90 días y qué decisiones la reducen.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

k1, k2, k3, k4 = st.columns(4)
k1.markdown(kpi("Tasa global de mora >90 días", f"1 de cada {num(UNO_DE_CADA)}",
                f"{pct(TASA_GLOBAL, 2)} de los contratos"), unsafe_allow_html=True)
k2.markdown(kpi("Beneficio neto estimado 2025", f"≈ {num(RES_BASE['beneficio_M'])} M",
                "millones de COP · escenario base"), unsafe_allow_html=True)
k3.markdown(kpi("Contratos analizados", num(len(df)), "2019-2025, sin duplicados"), unsafe_allow_html=True)
k4.markdown(kpi("Mora de los aprobados", f"{num(MORA_2025 * 100, 1)} → {num(RES_BASE['mora_aprobados'] * 100, 1)} %",
                f"revisando al {pct(RES_BASE['rechazo'], 0)} más riesgoso"), unsafe_allow_html=True)

st.markdown(
    f"<p class='nota' style='margin-top:10px'>Cuando un arrendatario se atrasa más de 3 meses, Hábitat paga al "
    f"propietario con su propia caja. La buena noticia: la mora se concentra en dos señales que Hábitat ya conoce al "
    f"firmar.</p>",
    unsafe_allow_html=True,
)

tab_diag, tab_sim, tab_concl = st.tabs(
    ["📊  1 · Diagnóstico del riesgo", "🎛️  2 · Simulador de decisiones", "🧭  3 · Conclusiones"]
)

# ==============================================================================
# 7. SECCIÓN 1 · DIAGNÓSTICO (EDA)
# ==============================================================================
with tab_diag:
    st.markdown("#### Explore el riesgo por segmento")
    f1, f2, f3 = st.columns(3)
    ciudades = sorted(df["ciudad"].unique())
    sel_ciudad = f1.multiselect("Ciudad", ciudades, default=ciudades)
    sel_estrato = f2.multiselect("Estrato", sorted(df["estrato"].unique()), default=sorted(df["estrato"].unique()))
    sel_ingreso = f3.multiselect("Rango de ingreso mensual (COP)", TRAMOS_INGRESO[1], default=TRAMOS_INGRESO[1])

    d = df[df["ciudad"].isin(sel_ciudad) & df["estrato"].isin(sel_estrato) & df["tramo_ingreso"].isin(sel_ingreso)]

    if d.empty:
        st.info("No hay contratos con esa combinación de filtros. Amplíe la selección.")
    else:
        tasa_d = d["mora_90dias"].mean()
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Contratos en la selección", num(len(d)))
        m2.metric("Tasa de mora", pct(tasa_d, 2), f"{num((tasa_d - TASA_GLOBAL) * 100, 2)} pp vs. total",
                  delta_color="inverse")
        m3.metric("Equivale a", f"1 de cada {num(1 / tasa_d) if tasa_d > 0 else '—'}")
        m4.metric("Canon / ingreso mediano", pct(d["ratio_canon_ingreso"].median(), 0))

        # --- Factores de riesgo: puntaje y peso del arriendo ---
        st.markdown("#### Lo que sí predice la mora: capacidad de pago e historial")
        c1, c2 = st.columns(2)
        prom = tasa_d * 100

        t_s = tasa_por(d, "tramo_score")
        fig = go.Figure(go.Bar(
            x=t_s["tramo_score"].astype(str), y=t_s["tasa"] * 100,
            marker_color=[BORGONA if v > 2 * tasa_d else (ROJO if v > tasa_d else GRIS) for v in t_s["tasa"]],
            text=[num(v * 100, 1) for v in t_s["tasa"]], textposition="outside",
            customdata=t_s["contratos"], hovertemplate="Puntaje %{x}<br>Mora %{y:.1f} %<br>%{customdata:,} contratos<extra></extra>",
        ))
        fig.add_hline(y=prom, line_dash="dash", line_color=GRIS_OSCURO,
                      annotation_text=f"Promedio {num(prom, 1)} %", annotation_position="top right")
        fig.update_layout(xaxis_title="Puntaje crediticio", yaxis_title="Tasa de mora", yaxis_range=[0, max(t_s["tasa"].max() * 115, 10)])
        with c1:
            mostrar(estilo(fig, "Mora por puntaje crediticio: cae de forma escalonada", yfmt=" %"))
            st.caption(f"Del tramo más bajo ({pct(t_s['tasa'].iloc[0])}) al más alto ({pct(t_s['tasa'].iloc[-1])}): "
                       "a mayor puntaje, menor mora en cada escalón.")

        t_r = tasa_por(d, "tramo_ratio")
        fig = go.Figure(go.Bar(
            x=t_r["tramo_ratio"].astype(str), y=t_r["tasa"] * 100,
            marker_color=[BORGONA if v > 2 * tasa_d else (ROJO if v > tasa_d else GRIS) for v in t_r["tasa"]],
            text=[num(v * 100, 1) for v in t_r["tasa"]], textposition="outside",
            customdata=t_r["contratos"], hovertemplate="Canon/ingreso %{x}<br>Mora %{y:.1f} %<br>%{customdata:,} contratos<extra></extra>",
        ))
        fig.add_hline(y=prom, line_dash="dash", line_color=GRIS_OSCURO,
                      annotation_text=f"Promedio {num(prom, 1)} %", annotation_position="top left")
        fig.update_layout(xaxis_title="Arriendo como % del ingreso declarado", yaxis_title="Tasa de mora",
                          yaxis_range=[0, max(t_r["tasa"].max() * 112, 10)])
        with c2:
            mostrar(estilo(fig, "Mora por peso del arriendo: efecto umbral", yfmt=" %"))
            st.caption("Hasta el 40 % del ingreso la mora es baja; si el arriendo supera el ingreso declarado, casi todos caen en mora.")

        with st.expander("Otras señales con información: pagos tardíos previos, garantía y tipo de inmueble"):
            e1, e2, e3 = st.columns(3)
            for col_st, var, tit in [(e1, "pagos_tardios_txt", "Pagos tardíos previos"),
                                     (e2, "tipo_garantia", "Tipo de garantía"),
                                     (e3, "tipo_inmueble", "Tipo de inmueble")]:
                t = tasa_por(d, var).sort_values("tasa", ascending=var == "pagos_tardios_txt")
                if var == "pagos_tardios_txt":
                    t = t.sort_values(var)
                f = go.Figure(go.Bar(x=t[var].astype(str), y=t["tasa"] * 100, marker_color=ROJO,
                                     text=[num(v * 100, 1) for v in t["tasa"]], textposition="outside"))
                f.add_hline(y=prom, line_dash="dash", line_color=GRIS_OSCURO)
                f.update_layout(yaxis_range=[0, t["tasa"].max() * 125])
                with col_st:
                    mostrar(estilo(f, tit, alto=300, yfmt=" %"))

        # --- El mito del barrio ---
        st.markdown("#### El mito del barrio: la ciudad, el estrato y el canal no cambian el riesgo")
        g1, g2, g3 = st.columns(3)
        rango_info = {}
        for col_st, var, tit in [(g1, "ciudad", "Por ciudad"), (g2, "estrato", "Por estrato"),
                                 (g3, "canal_captacion", "Por canal de captación")]:
            t = tasa_por(d, var)
            if var != "estrato":
                t = t.sort_values("tasa", ascending=False)
            rango_info[tit] = (t["tasa"].max() - t["tasa"].min()) * 100 if len(t) > 1 else 0
            f = go.Figure(go.Bar(
                x=t[var].astype(str), y=t["tasa"] * 100, marker_color=GRIS,
                error_y=dict(type="data", array=t["ic"] * 100, color=GRIS_OSCURO, thickness=1),
                text=[num(v * 100, 1) for v in t["tasa"]], textposition="inside", insidetextanchor="end",
                textfont=dict(color="white"),
                hovertemplate="%{x}<br>Mora %{y:.1f} %<extra></extra>",
            ))
            f.add_hrect(y0=8, y1=9, fillcolor=ROSA, opacity=0.35, line_width=0)
            # Eje fijo a 0-45 % para comparar con la escala del puntaje (no exagerar diferencias)
            f.update_layout(yaxis_range=[0, 45], showlegend=False)
            with col_st:
                mostrar(estilo(f, tit, alto=300, yfmt=" %"))

        brecha_score = (t_s["tasa"].max() - t_s["tasa"].min()) * 100
        st.markdown(
            f"<div class='card no'><p><b>Lectura:</b> en todos los grupos la mora se mantiene en ≈8-9 de cada 100 "
            f"(franja rosada). La diferencia máxima es de {num(rango_info['Por ciudad'], 1)} pp entre ciudades, "
            f"{num(rango_info['Por estrato'], 1)} pp entre estratos y {num(rango_info['Por canal de captación'], 1)} pp entre canales, "
            f"frente a <b>{num(brecha_score, 1)} pp</b> entre tramos de puntaje. Las barras usan la misma escala que el "
            f"puntaje para no inflar diferencias pequeñas. <b>No hay que juzgar por el barrio.</b></p></div>",
            unsafe_allow_html=True,
        )

# ==============================================================================
# 8. SECCIÓN 2 · SIMULADOR DE DECISIONES
# ==============================================================================
with tab_sim:
    st.markdown("Ajuste los supuestos y las políticas: los resultados se recalculan sobre los "
                f"**{num(M['n_te'])} contratos de 2025**, que el modelo nunca vio al entrenar.")

    # ---------------- Decisión 1: aprobar ----------------
    st.markdown("### <span class='num'>1</span>Aprobar: ¿a quién no aprobar sin condiciones?", unsafe_allow_html=True)
    s1, s2, s3 = st.columns([1, 1, 1.2])
    L = s1.slider("Pérdida por moroso (cánones)", 3, 8, 5, help="Cánones que Hábitat pierde, en promedio, por cada contrato en mora. 3 = conservador, 5 = base, 8 = severo.")
    c = s2.slider("Comisión de Hábitat (% del canon)", 5, 15, 10) / 100
    u_opt = umbral_optimo(M, L, c)
    with s3:
        usar_opt = st.toggle("Usar el umbral óptimo según los costos", value=True)
        umbral = u_opt if usar_opt else st.slider("Umbral de probabilidad de mora", 0.05, 0.60, round(u_opt, 2), 0.005)
        st.caption(f"Umbral aplicado: probabilidad ≥ **{num(umbral, 3)}** (óptimo con estos costos: {num(u_opt, 3)}).")

    r = resultado_politica(M["p_te"] >= umbral, M, L, c)
    a1, a2, a3, a4, a5 = st.columns(5)
    a1.metric("Solicitudes revisadas", pct(r["rechazo"]))
    a2.metric("Mora de los aprobados", pct(r["mora_aprobados"], 2),
              f"{num((r['mora_aprobados'] - MORA_2025) * 100, 2)} pp", delta_color="inverse")
    a3.metric("Morosos evitados", num(r["evitados"]), f"{pct(r['evitados'] / M['y_te'].sum(), 0)} de la mora", delta_color="off")
    a4.metric("Buenos clientes perdidos", num(r["perdidos"]), delta_color="off")
    a5.metric("Beneficio neto", f"{num(r['beneficio_M'])} M COP")

    b1, b2 = st.columns([1.4, 1])
    curva = curva_beneficio(M["p_te"], M["y_te"], M["canon_te"], M["meses_te"], L, c) / 1e6
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=FRACCIONES * 100, y=curva, mode="lines", line=dict(color=BORGONA, width=3),
                             name="Beneficio neto 2025", hovertemplate="Rechazar %{x:.1f} %<br>%{y:,.0f} M COP<extra></extra>"))
    fig.add_trace(go.Scatter(x=[r["rechazo"] * 100], y=[r["beneficio_M"]], mode="markers",
                             marker=dict(size=14, color=ROJO, line=dict(color="white", width=2)), name="Su política"))
    fig.add_hline(y=0, line_color=GRIS)
    fig.update_layout(xaxis_title="% de solicitudes de mayor riesgo que no se aprueban sin condiciones",
                      yaxis_title="Millones de COP")
    with b1:
        mostrar(estilo(fig, "El beneficio sube hasta un óptimo y luego cae (se pierden buenos clientes)"))
    with b2:
        fig = go.Figure(go.Bar(
            x=["Situación actual", "Con su política"], y=[MORA_2025 * 100, r["mora_aprobados"] * 100],
            marker_color=[GRIS, BORGONA], text=[pct(MORA_2025, 2), pct(r["mora_aprobados"], 2)], textposition="outside", width=0.5,
        ))
        fig.update_layout(yaxis_range=[0, MORA_2025 * 140], yaxis_title="Mora de los contratos aprobados")
        mostrar(estilo(fig, "Mora de la cartera aprobada (2025)", yfmt=" %"))

    st.divider()

    # ---------------- Decisión 2: codeudor ----------------
    st.markdown("### <span class='num'>2</span>Codeudor: pedirlo según el riesgo, no a todos", unsafe_allow_html=True)
    ef = M["efectos"]
    h1, h2 = st.columns([1.3, 1])
    with h1:
        fig = go.Figure()
        colores_g = {"Sin garantía": GRIS_CLARO, "Depósito": ROSA, "Seguro de arrendamiento": ROJO, "Codeudor": BORGONA}
        for gg in GARANTIAS:
            t = ef[ef["garantia"] == gg].set_index("nivel").loc[NIVELES]
            fig.add_trace(go.Bar(x=["Riesgo bajo (50 %)", "Riesgo medio (30 %)", "Riesgo alto (20 %)"], y=t["mora"] * 100,
                                 name=gg, marker_color=colores_g[gg], text=[num(v * 100, 1) for v in t["mora"]],
                                 textposition="outside"))
        fig.update_layout(barmode="group", yaxis_title="Mora esperada si todos tuvieran esa garantía",
                          yaxis_range=[0, ef["mora"].max() * 118])
        mostrar(estilo(fig, "Mora esperada por garantía y nivel de riesgo (2019-2024)", alto=380, yfmt=" %"))
    with h2:
        mora_ef = ef.set_index(["nivel", "garantia"])["mora"]
        red = {nn: (mora_ef[(nn, "Sin garantía")] - mora_ef[(nn, "Codeudor")]) * 100 for nn in NIVELES}
        alto_sin, alto_cod = mora_ef[("Alto", "Sin garantía")], mora_ef[("Alto", "Codeudor")]
        st.markdown(
            f"<div class='card si'><h4>El codeudor vale más donde más riesgo hay</h4>"
            f"<p>Reducción de la mora con codeudor frente a no tener garantía:</p>"
            f"<ul><li>Riesgo bajo: <b>{num(red['Bajo'], 1)} pp</b></li>"
            f"<li>Riesgo medio: <b>{num(red['Medio'], 1)} pp</b></li>"
            f"<li>Riesgo alto: <b>{num(red['Alto'], 1)} pp</b> (de {pct(alto_sin)} a {pct(alto_cod)})</li></ul>"
            f"<p>Exigirlo a todos encarece la entrada sin beneficio en la mitad sana de la cartera; "
            f"no exigirlo nunca deja expuesto al 20 % más riesgoso.</p>"
            f"<p class='nota'>Asociaciones observacionales, no efectos causales: un piloto con grupo de control medirá el efecto real.</p></div>",
            unsafe_allow_html=True,
        )

    st.markdown("**Defina la política de garantía por nivel de riesgo** (riesgo calculado sin considerar la garantía):")
    OPC = ["Opcional", "Alguna garantía", "Codeudor"]
    p1, p2, p3 = st.columns(3)
    pol = {"Bajo": p1.selectbox("Riesgo bajo (50 % de menor riesgo)", OPC, index=0),
           "Medio": p2.selectbox("Riesgo medio (30 %)", OPC, index=1),
           "Alto": p3.selectbox("Riesgo alto (20 % de mayor riesgo)", OPC, index=2)}

    te = M["te"]
    g_actual = te["tipo_garantia"].values
    g_nueva = g_actual.copy()
    for nn, regla in pol.items():
        msk = M["nivel_te"] == nn
        if regla == "Codeudor":
            g_nueva[msk] = "Codeudor"
        elif regla == "Alguna garantía":
            g_nueva[msk & (g_actual == "Sin garantía")] = "Seguro de arrendamiento"
    cambia = g_nueva != g_actual
    te_cf = te.copy()
    te_cf["tipo_garantia"] = g_nueva
    p_cf = M["modelo"].predict_proba(te_cf)[:, 1]  # probabilidad recalculada con la garantía exigida

    filas = []
    for nn in NIVELES:
        msk = M["nivel_te"] == nn
        filas.append({"Nivel": nn, "Contratos": msk.sum(), "Deben reforzar garantía": cambia[msk].mean(),
                      "Mora esperada actual": M["p_te"][msk].mean(), "Mora esperada con la política": p_cf[msk].mean()})
    tab_pol = pd.DataFrame(filas)
    q1, q2, q3 = st.columns(3)
    q1.metric("Mora esperada de la cartera", pct(p_cf.mean(), 2),
              f"{num((p_cf.mean() - M['p_te'].mean()) * 100, 2)} pp", delta_color="inverse")
    q2.metric("Contratos que deben reforzar la garantía", pct(cambia.mean()), help="Fricción comercial: clientes a quienes se pediría una garantía más fuerte.")
    q3.metric("Mora esperada en riesgo alto", pct(tab_pol.iloc[2]["Mora esperada con la política"], 1),
              f"{num((tab_pol.iloc[2]['Mora esperada con la política'] - tab_pol.iloc[2]['Mora esperada actual']) * 100, 1)} pp",
              delta_color="inverse")
    tabla(
        tab_pol.style.format({"Contratos": lambda v: num(v), "Deben reforzar garantía": lambda v: pct(v),
                              "Mora esperada actual": lambda v: pct(v, 2), "Mora esperada con la política": lambda v: pct(v, 2)}))

    st.divider()

    # ---------------- Decisión 3: cobrar ----------------
    st.markdown("### <span class='num'>3</span>Cobrar: enfocar a los 4 gestores en los contratos más riesgosos", unsafe_allow_html=True)
    w1, w2, w3 = st.columns(3)
    cobertura = w1.slider("Cobertura de la cartera (%)", 5, 40, 10) / 100
    gestores = w2.number_input("Gestores de cobranza", 1, 20, 4)
    cartera = w3.number_input("Cartera vigente (contratos)", 1000, 50000, 9000, step=500)
    cap, lift, prec = captura(cobertura)
    v1, v2, v3, v4 = st.columns(4)
    v1.metric("Mora capturada", pct(cap), help="Porcentaje de todos los morosos de 2025 que caen en la lista priorizada.")
    v2.metric("Lift", f"{num(lift, 1)}×", help="Cuántas veces la mora de la lista supera la mora promedio.")
    v3.metric("Precisión", pct(prec), help="De cada 100 contratos gestionados, cuántos caen en mora.")
    v4.metric("Contratos por gestor", num(cartera * cobertura / gestores))

    x = np.linspace(0, 1, 201)
    y_cap = [0] + [captura(f)[0] for f in x[1:]]
    fig = go.Figure()
    fig.add_vrect(x0=10, x1=20, fillcolor=ROSA, opacity=0.35, line_width=0,
                  annotation_text="Capacidad de 4 gestores", annotation_position="top left")
    fig.add_trace(go.Scatter(x=x * 100, y=x * 100, mode="lines", line=dict(color=GRIS, dash="dash"), name="Al azar"))
    fig.add_trace(go.Scatter(x=x * 100, y=np.array(y_cap) * 100, mode="lines", line=dict(color=BORGONA, width=3),
                             name="Lista ordenada por el modelo", hovertemplate="Gestionar %{x:.0f} %<br>Captura %{y:.1f} %<extra></extra>"))
    fig.add_trace(go.Scatter(x=[cobertura * 100], y=[cap * 100], mode="markers+text",
                             marker=dict(size=13, color=ROJO, line=dict(color="white", width=2)),
                             text=[f"  {pct(cap)}"], textposition="middle right", name="Su cobertura"))
    fig.update_layout(xaxis_title="% de contratos gestionados (de mayor a menor riesgo)", yaxis_title="% de la mora capturada",
                      xaxis_range=[0, 100], yaxis_range=[0, 102])
    k_a, k_b = st.columns([1.4, 1])
    with k_a:
        mostrar(estilo(fig, "Curva de captura de la mora (2025)", alto=380))
    with k_b:
        top = np.zeros(len(te), bool)
        top[ORDEN_TE[: max(int(len(te) * cobertura), 1)]] = True
        perfil = pd.DataFrame({
            "Rasgo": ["Arriendo > 50 % del ingreso", "Puntaje < 550", "Sin garantía", "≥ 2 pagos tardíos previos"],
            "Lista priorizada": [(te["ratio_canon_ingreso"] > .5)[top].mean(), (te["score_crediticio"] < 550)[top].mean(),
                                 (te["tipo_garantia"] == "Sin garantía")[top].mean(), (te["pagos_tardios_previos"] >= 2)[top].mean()],
            "Resto": [(te["ratio_canon_ingreso"] > .5)[~top].mean(), (te["score_crediticio"] < 550)[~top].mean(),
                      (te["tipo_garantia"] == "Sin garantía")[~top].mean(), (te["pagos_tardios_previos"] >= 2)[~top].mean()],
        })
        st.markdown("**Perfil de la lista priorizada**")
        tabla(perfil.style.format({"Lista priorizada": lambda v: pct(v, 0), "Resto": lambda v: pct(v, 0)}))
        st.caption("Regla de oficina (árbol de 3 niveles): priorizar si el arriendo supera el 45 % del ingreso y, "
                   "además, el puntaje es ≤536 o el arriendo supera el 61 % del ingreso.")

    st.divider()

    # ---------------- Evaluador de una solicitud ----------------
    st.markdown("### Pruebe una solicitud")
    st.caption("Modelo de decisión de tres pasos: (1) riesgo base sin garantía → (2) garantía según el nivel → "
               f"(3) aprobar solo si la probabilidad con la garantía ofrecida es menor que {num(U_BASE, 3)}.")
    e1, e2, e3, e4, e5 = st.columns(5)
    sc = e1.number_input("Puntaje crediticio", 300, 950, 620, step=10)
    ra = e2.slider("Arriendo / ingreso (%)", 10, 150, 35) / 100
    pa = e3.selectbox("Pagos tardíos previos", [0, 1, 2, 3, 4], index=0)
    ga = e4.selectbox("Garantía ofrecida", GARANTIAS, index=0)
    inm = e5.selectbox("Tipo de inmueble", ["Apartamento", "Casa", "Local comercial", "Oficina", "Bodega"])

    def _sol(g):
        return pd.DataFrame([{"score_crediticio": sc, "score_nulo": 0, "ratio_acot": min(ra, 2), "pagos_acot": min(pa, 4),
                              "es_2020": 0, "tipo_garantia": g, "tipo_inmueble": inm}])

    p_sol = M["modelo"].predict_proba(_sol(ga))[0, 1]
    p_cod = M["modelo"].predict_proba(_sol("Codeudor"))[0, 1]
    rb = M["modelo_base"].predict_proba(_sol(ga))[0, 1]
    nivel_sol = "Bajo" if rb < M["cortes"][0] else ("Medio" if rb < M["cortes"][1] else "Alto")
    if p_sol >= U_BASE:
        decision = (f"✅ Aprobar **solo con codeudor** (con codeudor la probabilidad baja a {pct(p_cod)})"
                    if p_cod < U_BASE else "❌ No aprobar: incluso con codeudor supera el umbral")
    elif nivel_sol == "Alto" and ga != "Codeudor":
        decision = "⚠️ Aprobar **exigiendo codeudor** (riesgo alto)"
    elif nivel_sol == "Medio" and ga == "Sin garantía":
        decision = "⚠️ Aprobar **exigiendo alguna garantía**, idealmente codeudor (riesgo medio)"
    else:
        decision = "✅ Aprobar" + (" (garantía opcional)" if nivel_sol == "Bajo" else " con la garantía ofrecida")
    o1, o2, o3 = st.columns([1, 1, 2])
    o1.metric("Probabilidad de mora", pct(p_sol))
    o2.metric("Nivel de riesgo base", nivel_sol)
    o3.markdown(f"<div class='card si' style='margin-top:4px'><h4>Recomendación</h4><p>{decision}</p></div>",
                unsafe_allow_html=True)

# ==============================================================================
# 9. SECCIÓN 3 · NARRATIVA Y CONCLUSIONES
# ==============================================================================
with tab_concl:
    t_s_all = tasa_por(df, "tramo_score")
    t_r_all = tasa_por(df, "tramo_ratio")
    rng_ciudad = tasa_por(df, "ciudad")["tasa"]
    rng_estrato = tasa_por(df, "estrato")["tasa"]
    cap10, lift10, prec10 = captura(0.10)
    cap20 = captura(0.20)[0]

    st.markdown("### La historia en una página")
    st.markdown(
        f"<div class='card' style='border-left:5px solid {BORGONA}'>"
        f"<span class='tag'>El problema</span>"
        f"<p style='font-size:1.05rem;margin-top:8px'>1 de cada {num(UNO_DE_CADA)} contratos se atrasa más de 3 meses y Hábitat "
        f"paga al propietario de su propio bolsillo. La pregunta no es si habrá mora, sino <b>si se puede ver venir al firmar</b>. "
        f"Sí se puede: el modelo ordena el riesgo con un ROC-AUC de {num(M['roc_auc'], 2)} en contratos que nunca vio.</p></div>",
        unsafe_allow_html=True,
    )
    st.write("")
    s_a, s_b = st.columns(2)
    s_a.markdown(
        f"<div class='card si'><span class='tag'>Lo que sí importa</span>"
        f"<h4 style='margin-top:8px'>Capacidad de pago real e historial</h4><ul>"
        f"<li><b>Peso del arriendo:</b> por debajo del 40 % del ingreso, la mora es de {pct(t_r_all['tasa'].iloc[:3].min())}-"
        f"{pct(t_r_all['tasa'].iloc[:3].max())}; si supera el 100 %, llega a <b>{pct(t_r_all['tasa'].iloc[-1], 0)}</b>.</li>"
        f"<li><b>Puntaje crediticio:</b> {pct(t_s_all['tasa'].iloc[0], 0)} de mora con ≤450 puntos frente a "
        f"{pct(t_s_all['tasa'].iloc[-1])} con más de 800.</li>"
        f"<li>Suman: pagos tardíos previos, tipo de garantía y tipo de inmueble (bodegas y locales).</li></ul></div>",
        unsafe_allow_html=True,
    )
    s_b.markdown(
        f"<div class='card no'><span class='tag' style='background:{GRIS_CLARO};color:{GRIS_OSCURO}'>Lo que no importa</span>"
        f"<h4 style='margin-top:8px'>Las apariencias</h4><ul>"
        f"<li><b>Ciudad:</b> {pct(rng_ciudad.min())}-{pct(rng_ciudad.max())}.</li>"
        f"<li><b>Estrato:</b> {pct(rng_estrato.min())}-{pct(rng_estrato.max())}.</li>"
        f"<li><b>Canal de captación</b>, administración, duración, área y antigüedad: sin diferencias relevantes.</li>"
        f"<li>El riesgo es transversal: <b>no hay que juzgar por el barrio</b>. El modelo no usa estrato ni ciudad.</li></ul></div>",
        unsafe_allow_html=True,
    )

    st.markdown("### Las 3 decisiones")
    d1, d2, d3 = st.columns(3)
    d1.markdown(
        f"<div class='card si'><h4><span class='num'>1</span>Aprobar</h4>"
        f"<p>Revisar mejor al <b>{pct(RES_BASE['rechazo'], 0)}</b> más riesgoso (probabilidad ≥ {num(U_BASE, 3)}) baja la mora de los "
        f"aprobados de <b>{num(MORA_2025 * 100, 1)} a {num(RES_BASE['mora_aprobados'] * 100, 1)}</b> por cada 100 y duplica el "
        f"beneficio de una regla simple.</p></div>", unsafe_allow_html=True)
    d2.markdown(
        f"<div class='card si'><h4><span class='num'>2</span>Codeudor</h4>"
        f"<p>Pedirlo según el riesgo: en el grupo más riesgoso la mora pasa de <b>{pct(alto_sin, 0)} a {pct(alto_cod, 0)}</b>; "
        f"en el 50 % de menor riesgo la ganancia es pequeña ({num(red['Bajo'], 1)} pp) y puede ser opcional.</p></div>",
        unsafe_allow_html=True)
    d3.markdown(
        f"<div class='card si'><h4><span class='num'>3</span>Cobrar</h4>"
        f"<p>Enfocar a los 4 gestores en el 10 % más riesgoso captura el <b>{pct(cap10, 0)}</b> de la mora "
        f"(lift {num(lift10, 1)}×); con el 20 % de cobertura se llega al <b>{pct(cap20, 0)}</b>.</p></div>",
        unsafe_allow_html=True)

    st.write("")
    st.markdown(
        f"<div class='hero' style='padding:22px 28px'><div class='kicker'>Impacto estimado</div>"
        f"<h1 style='font-size:1.8rem'>≈ {num(RES_BASE['beneficio_M'])} millones de COP de beneficio neto con los contratos de 2025</h1>"
        f"<p>Evita {num(RES_BASE['evitados'])} de {num(M['y_te'].sum())} morosos a costa de {num(RES_BASE['perdidos'])} buenos "
        f"clientes. Escenario base: 5 cánones perdidos por moroso y comisión del 10 %.</p></div>",
        unsafe_allow_html=True,
    )

    l1, l2 = st.columns(2)
    l1.markdown(
        "<div class='card'><h4>Límites a tener presentes</h4><ul>"
        "<li><b>Costos supuestos:</b> el monto depende de la pérdida por moroso y la comisión; la conclusión se mantiene en los tres escenarios.</li>"
        "<li><b>Sesgo de selección:</b> solo hay contratos aprobados en el pasado.</li>"
        "<li><b>Garantía no causal:</b> sus efectos son asociaciones.</li>"
        "<li><b>Techo del modelo:</b> cerca de la mitad de los morosos no se detecta al firmar; el ingreso declarado es alterable.</li></ul></div>",
        unsafe_allow_html=True)
    l2.markdown(
        "<div class='card si'><h4>Próximo paso: piloto de 3 meses</h4><ul>"
        "<li>Aplicar el modelo de tres pasos en una parte de las solicitudes, con grupo de control.</li>"
        "<li>Validar con Finanzas la pérdida real por moroso y la comisión.</li>"
        "<li>Medir colocación, mora a 90 días y recuperación de la cobranza priorizada.</li>"
        "<li>Monitorear deriva del modelo y equidad por estrato cada trimestre.</li></ul></div>",
        unsafe_allow_html=True)

st.markdown(
    "<p class='nota' style='text-align:center;margin-top:28px'>Inmobiliaria Hábitat · Caso 4 · "
    "Elaboración propia a partir de inmobiliaria_arriendos.csv · Las asociaciones no implican causalidad.</p>",
    unsafe_allow_html=True,
)
