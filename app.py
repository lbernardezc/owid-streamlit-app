# app.py
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

DATA_URL = "https://raw.githubusercontent.com/owid/co2-data/master/owid-co2-data.csv"


# -----------------------------
# Carrega + preprocessament base
# -----------------------------
@st.cache_data(show_spinner=False)
def load_raw():
    df = pd.read_csv(DATA_URL)
    return df


@st.cache_data(show_spinner=False)
def prepare_countries(df: pd.DataFrame) -> pd.DataFrame:
    # Filtre robust: només ISO3 (països). Fora OWID_* i continents amb iso_code NaN
    iso = df["iso_code"].astype("string")
    mask_iso3 = iso.str.fullmatch(r"[A-Z]{3}").fillna(False)
    d = df[mask_iso3].copy()

    # Fora Antarctica (ATA) perquè no és un país sobirà i pot distorsionar lectures
    d = d[d["iso_code"] != "ATA"].copy()

    # Ens quedem amb el mínim necessari (i algun extra útil)
    cols = [
        "country", "iso_code", "year",
        "co2", "co2_per_capita",
        "population",
        "ghg_per_capita"
    ]
    cols = [c for c in cols if c in d.columns]
    d = d[cols].copy()

    # Recalculo co2_per_capita quan falti però tingui co2 i population
    # (co2 és en milions de tones, per això *1e6 per passar a tones)
    if "co2_per_capita" in d.columns and "co2" in d.columns and "population" in d.columns:
        m = (
            d["co2_per_capita"].isna()
            & d["co2"].notna()
            & d["population"].notna()
            & (d["population"] > 0)
        )
        d.loc[m, "co2_per_capita"] = (d.loc[m, "co2"] * 1e6) / d.loc[m, "population"]

    return d

@st.cache_data(show_spinner=False)
def prepare_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    # Mantinc files que NO són ISO3 (agregats / regions / world / income groups)
    iso = df["iso_code"].astype("string")
    mask_iso3 = iso.str.fullmatch(r"[A-Z]{3}").fillna(False)

    d = df[~mask_iso3].copy()

    cols = ["country","iso_code","year","co2","co2_per_capita","population","ghg_per_capita"]
    cols = [c for c in cols if c in d.columns]
    d = d[cols].copy()

    # Recalculo co2_per_capita quan falti i sigui derivable
    if "co2_per_capita" in d.columns and "co2" in d.columns and "population" in d.columns:
        m = (
            d["co2_per_capita"].isna()
            & d["co2"].notna()
            & d["population"].notna()
            & (d["population"] > 0)
        )
        d.loc[m, "co2_per_capita"] = (d.loc[m, "co2"] * 1e6) / d.loc[m, "population"]

    return d



def filter_base(df: pd.DataFrame, y0: int, y1: int, min_pop: int) -> pd.DataFrame:
    d = df[(df["year"] >= y0) & (df["year"] <= y1)].copy()
    if min_pop > 0 and "population" in d.columns:
        d = d[d["population"].fillna(0) >= min_pop].copy()
    return d


# -----------------------------
# Rànquing Top-N (trimmed)
# -----------------------------
def top_countries_trimmed_mean(
    df: pd.DataFrame,
    metric: str,
    y0: int,
    y1: int,
    top_n: int,
    min_pop: int,
    trim_q: float = 0.99,
) -> list[str]:
    """
    Rànquing robust: per a cada país, retallo (trim) la cua superior de la mètrica
    dins del període (per-country), i calculo la mitjana del que queda.
    Això evita que un any excepcional (p.ex. Kuwait 1991) domini el Top-N.
    """
    d = filter_base(df, y0, y1, min_pop).dropna(subset=[metric]).copy()

    def trimmed_mean_country(g: pd.DataFrame) -> float:
        s = g[metric].dropna()
        if len(s) == 0:
            return np.nan
        cap = s.quantile(trim_q)
        s2 = s[s <= cap]
        # Si per algun cas extrem s2 queda buit, torno a s
        if len(s2) == 0:
            s2 = s
        return float(s2.mean())

    score = d.groupby("country", as_index=True).apply(trimmed_mean_country).dropna()
    rank = score.sort_values(ascending=False).head(top_n).index.tolist()
    return rank


# -----------------------------
# Construcció datasets per gràfics
# -----------------------------
def make_timeseries(
    df: pd.DataFrame,
    metric: str,
    y0: int,
    y1: int,
    mode: str,
    manual_countries: list[str],
    top_n: int,
    min_pop: int,
    trim_q: float,
) -> tuple[pd.DataFrame, list[str]]:
    d = filter_base(df, y0, y1, min_pop).dropna(subset=[metric]).copy()

    if mode == "manual":
        sel = manual_countries or []
        d = d[d["country"].isin(sel)].copy()
        return d, sel

    # mode top (trimmed)
    rank = top_countries_trimmed_mean(df, metric, y0, y1, top_n, min_pop, trim_q=trim_q)
    d = d[d["country"].isin(rank)].copy()
    return d, rank


def make_scatter_year(
    df: pd.DataFrame,
    year0: int,
    min_pop: int,
) -> pd.DataFrame:
    d = df[df["year"] == year0].copy()
    need = ["co2", "co2_per_capita", "population"]
    for c in need:
        if c not in d.columns:
            d[c] = np.nan
    d = d.dropna(subset=["co2", "co2_per_capita", "population"]).copy()
    if min_pop > 0:
        d = d[d["population"] >= min_pop].copy()
    return d


# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="OWID CO₂ — Visualització", layout="wide")

raw = load_raw()
df_c = prepare_countries(raw)
df_agg = prepare_aggregates(raw)

METRIC_INFO = {
    "co2_per_capita": {
        "title": "CO₂ per càpita",
        "unit": "tones de CO₂ per persona (t/persona)",
        "desc": "Emissions territorials de CO₂ per persona (principalment combustibles fòssils i ciment).",
    },
    "co2": {
        "title": "CO₂ total",
        "unit": "milions de tones de CO₂ (Mt)",
        "desc": "Emissions territorials totals de CO₂ (combustibles fòssils i ciment).",
    },
    "ghg_per_capita": {
        "title": "Gasos d'efecte hivernacle per càpita (GHG)",
        "unit": "tones de CO₂ equivalent per persona (tCO₂e/persona, GWP100)",
        "desc": "Emissions totals de gasos d’efecte hivernacle convertides a CO₂ equivalent (inclou CO₂, CH₄, N₂O i gasos fluorats).",
    },
}



st.title("Desigualtats en emissions (OWID) — Sèries temporals i comparacions")
st.caption("Països ISO3. Controls per comparar països manualment o amb Top-N robust (trimmed).")

# Sidebar controls
st.sidebar.header("Controls")

metric = st.sidebar.selectbox(
    "Mètrica principal",
    options=[m for m in ["co2_per_capita", "co2", "ghg_per_capita"] if m in df_c.columns],
    index=0,
)
info = METRIC_INFO.get(metric, {})
st.info(f"**{info.get('title', metric)}** · Unitats: **{info.get('unit','')}**\n\n{info.get('desc','')}")

year_min = int(df_c["year"].min())
year_max = int(df_c["year"].max())
y0, y1 = st.sidebar.slider(
    "Rang d'anys",
    min_value=year_min,
    max_value=year_max,
    value=(1960, year_max),
    step=1,
)

mode = st.sidebar.radio("Mode de comparació", ["manual", "top"], index=0)

# microestats
use_min_pop = st.sidebar.checkbox("Excloure microestats (població mínima)", value=True)
min_pop = st.sidebar.select_slider(
    "Llindar de població",
    options=[0, 100_000, 500_000, 1_000_000, 5_000_000, 10_000_000],
    value=1_000_000 if use_min_pop else 0,
)
if not use_min_pop:
    min_pop = 0

# Top-N config
top_n = st.sidebar.slider("Top-N (només mode top)", 5, 30, 10, 1)
trim_q = st.sidebar.slider("Trim (quantil superior per país)", 0.90, 0.999, 0.99, 0.001)

# y axis view (linears, per llegibilitat)
cap_y = st.sidebar.checkbox("Limitar eix Y (p99) per llegibilitat", value=True)

# manual countries
all_countries = sorted(df_c["country"].unique().tolist())
default_manual = [c for c in ["Spain", "France", "Germany", "United States", "China"] if c in all_countries]
if mode == "manual":
    manual_countries = st.sidebar.multiselect(
        "Països (manual)",
        options=all_countries,
        default=default_manual,
    )
else:
    manual_countries = default_manual  # no es fa servir, però evito None

# Tabs
tab1, tab2, tab3 = st.tabs(["Sèrie temporal", "Scatter per any", "Agregats (regions)"])


with tab1:
    st.subheader("Sèrie temporal")

    ts, used_countries = make_timeseries(
        df_c, metric, y0, y1,
        mode=mode,
        manual_countries=manual_countries,
        top_n=top_n,
        min_pop=min_pop,
        trim_q=trim_q,
    )

    if len(ts) == 0:
        st.warning("No hi ha dades amb aquests filtres. Prova d'ampliar el rang o canviar el llindar de població.")
    else:
        # Figure
        labels_map = {
            "co2_per_capita": "CO₂ per càpita (t/persona)",
            "co2": "CO₂ total (Mt)",
            "ghg_per_capita": "GHG per càpita (tCO₂e/persona)",
        }

        fig = px.line(
            ts,
            x="year",
            y=metric,
            color="country",
            title=f"{METRIC_INFO[metric]['title']} — {mode} ({y0}–{y1})",
            labels={"year": "Any", metric: labels_map.get(metric, metric), "country": "País"},
        )


        # Linear scale. Opcionalment capem l'eix Y per fer-ho llegible.
        if cap_y:
            ymax = float(ts[metric].quantile(0.99))
            if np.isfinite(ymax) and ymax > 0:
                fig.update_yaxes(range=[0, ymax])

        fig.update_layout(legend_title_text="País")
        st.plotly_chart(fig, use_container_width=True)

        if mode == "top":
            st.caption(f"Top-{top_n} calculat amb **trimmed mean** per país dins el període (retallo per sobre del quantil {trim_q:.3f}).")
            st.write("Països seleccionats (Top):", ", ".join(sorted(used_countries)))
        else:
            st.caption("Mode manual: la selecció de països és lliure.")

with tab2:
    st.subheader("Scatter: CO₂ total vs CO₂ per càpita (any seleccionat)")

    year0 = st.slider("Any", min_value=y0, max_value=y1, value=min(y1, 2020), step=1)
    dY = make_scatter_year(df_c, year0, min_pop=min_pop)

    if len(dY) == 0:
        st.warning("No hi ha dades per aquest any amb els filtres actuals.")
    else:
        # Ressalto el mateix conjunt de països que la sèrie (manual o top)
        highlight = set(used_countries) if used_countries else set()
        dY["group"] = np.where(dY["country"].isin(highlight), "Seleccionats", "Altres")

        # Defineixo colors fixes perquè el contrast sigui clar:
        # - Seleccionats: vermell
        # - Altres: blau
        color_map = {
        "Seleccionats": "rgba(220, 20, 60, 0.85)",  # red tomato
        "Altres": "rgba(31, 119, 180, 0.55)",       # blau
        }

        fig2 = px.scatter(
            dY,
            x="co2_per_capita",
            y="co2",
            size="population",
            color="group",
            hover_name="country",
            title=f"Any {year0}: per càpita vs total (mida = població)",
            labels={
                "co2_per_capita": "CO₂ per càpita (t/persona)",
                "co2": "CO₂ total (Mt)",
                "population": "Població",
                "group": ""
            },
            color_discrete_map=color_map,
            category_orders={"group": ["Altres", "Seleccionats"]}  # llegenda en ordre lògic
        )

        # una mica de contorn perquè destaquin encara més
        fig2.update_traces(marker=dict(line=dict(width=0.5, color="rgba(0,0,0,0.2)")))

        if cap_y:
            ymax = float(dY["co2"].quantile(0.99))
            if np.isfinite(ymax) and ymax > 0:
                fig2.update_yaxes(range=[0, ymax])
            xmax = float(dY["co2_per_capita"].quantile(0.99))
            if np.isfinite(xmax) and xmax > 0:
                fig2.update_xaxes(range=[0, xmax])

        st.plotly_chart(fig2, use_container_width=True)

        st.caption("La selecció resaltada és la mateixa que a la sèrie temporal (manual o Top-N).")

with tab3:
    st.subheader("Agregats (World / continents / UE)")

    REGION_PRESET = [
    "World",
    "Europe","Asia", "Africa", "North America", "South America", "Oceania",
    ]

    metric_r = st.selectbox(
        "Mètrica (agregats)",
        options=[m for m in ["co2", "co2_per_capita", "ghg_per_capita"] if m in df_agg.columns],
        index=0,
        key="metric_regions"
    )

    # Llista de regions disponibles (preset + el que existeixi al dataset)
    available = sorted(df_agg["country"].dropna().unique().tolist())

    # Proposo un default amb les que realment existeixen
    default_regions = [r for r in REGION_PRESET if r in available]
    if not default_regions:
        default_regions = ["World"] if "World" in available else available[:5]

    regions = st.multiselect(
        "Regions / agregats a comparar",
        options=available,
        default=default_regions,
        help="Aquí hi ha també altres agregats (p.ex. income groups). Pots filtrar escrivint al quadre.",
        key="regions_multiselect"
    )

    dR = df_agg[(df_agg["year"] >= y0) & (df_agg["year"] <= y1)].copy()
    dR = dR[dR["country"].isin(regions)].dropna(subset=[metric_r]).copy()

    if len(dR) == 0:
        st.warning("No hi ha dades amb aquests filtres. Prova d'ampliar el rang o canviar les regions.")
    else:
        figR = px.line(
            dR,
            x="year",
            y=metric_r,
            color="country",
            title=f"Agregats — {metric_r} ({y0}–{y1})"
        )

        # Mateix control de llegibilitat (linear + cap p99)
        if cap_y:
            ymax = float(dR[metric_r].quantile(0.99))
            if np.isfinite(ymax) and ymax > 0:
                figR.update_yaxes(range=[0, ymax])

        st.plotly_chart(figR, width="stretch")
        st.caption("Agregats/regions segons OWID. Serveix per comparar macro-tendències sense entrar al detall país.")
