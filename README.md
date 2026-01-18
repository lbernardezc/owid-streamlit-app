# OWID CO₂ Explorer — Streamlit app

Visualització interactiva (Streamlit + Plotly) per explorar desigualtats i evolució de les emissions:
- **CO₂ total** (`co2`)
- **CO₂ per càpita** (`co2_per_capita`)
- **GHG per càpita** (`ghg_per_capita`, en tCO₂e/persona, GWP100)

📌 **App desplegada:** https://owid-app-app-vcqjszxjkyxi5hus55se3e.streamlit.app/  
📌 **Repositori:** https://github.com/lbernardezc/owid-streamlit-app

---

## Objectiu del projecte

Aquest projecte té com a objectiu facilitar l’exploració de:
1) **tendències temporals** (sèries 1960–2024) en emissions totals i per càpita,
2) **comparacions entre països** (manual o Top-N automàtic),
3) **comparacions per agregats** (World, continents, UE…),
4) diferències entre emissions **totals** vs **per càpita** (scatter per any).

---

## Dades

- **Font:** Our World in Data (OWID), *CO₂ and Greenhouse Gas Emissions* dataset.
- **Carrega:** l’app descarrega el CSV directament des del repositori públic d’OWID:
  https://github.com/owid/co2-data

---

## Com usar la visualització

### Pestanya 1 — Sèrie temporal
- Mode **manual**: selecció lliure de països.
- Mode **Top-N**: rànquing automàtic per la mètrica seleccionada.
- **Top-N robust**: s’utilitza una *trimmed mean* per país (retall del quantil superior) per reduir l’efecte de pics excepcionals en el rànquing.
- Opció de **limitar l’eix Y (p99)** per millorar la llegibilitat.

### Pestanya 2 — Scatter per any
- Comparació en un any concret: **CO₂ total vs CO₂ per càpita** (mida = població).
- Els països seleccionats a la sèrie temporal es ressalten al scatter per facilitar la lectura.

### Pestanya 3 — Agregats (regions)
- Comparació d’agregats (World / continents / UE…) per veure tendències macro.

---

## Execució en local

> Requisit: Python 3.10+ (recomanat 3.12)

```bash
pip install -r requirements.txt
streamlit run app.py
