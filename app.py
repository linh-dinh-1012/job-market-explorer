from __future__ import annotations

import os
import time
import ast
import tempfile
import urllib.parse
from datetime import datetime
from typing import Any, Optional, List, Dict

import pandas as pd
import streamlit as st

# =========================
# DATA pipeline scripts
# =========================
from data.ft_auth import get_ft_access_token
from data.collect_ft import fetch_all_ft_offers
from data.collect_wtj import fetch_wttj_offers
from data.preprocessing import preprocess_ft, preprocess_wttj

# =========================
# ANALYSIS modules
# =========================
from analysis.skills import analyze_skills_ft, analyze_skills_wttj
from analysis.salary import analyze_salary
from analysis.geography import create_ft_jobs_map, jobs_by_location
from analysis.matching import match_cv_market


# ==================================================
# CONFIG
# ==================================================
APP_NAME = "Explorateur du marché du travail"
TMP_PREFIX = "job_market_explorer_"
TMP_TTL_SECONDS = 60 * 60 

LIST_COLS = [
    "skills_hard_required",
    "skills_hard_optional",
    "skills_soft",
    "languages_required",
    "languages_optional",
]

# Location presets (URL-based)
LOC_PRESETS: Dict[str, Optional[dict]] = {
    "Toutes (FR)": None,
    "Île-de-France": {"state": "Île-de-France", "aroundQuery": "Île-de-France, France"},
    "Auvergne-Rhône-Alpes": {"state": "Auvergne-Rhône-Alpes", "aroundQuery": "Auvergne-Rhône-Alpes, France"},
    "Provence-Alpes-Côte d'Azur": {
        "state": "Provence-Alpes-Côte d'Azur",
        "aroundQuery": "Provence-Alpes-Côte d'Azur, France",
    },
    "Occitanie": {"state": "Occitanie", "aroundQuery": "Occitanie, France"},
    "Nouvelle-Aquitaine": {"state": "Nouvelle-Aquitaine", "aroundQuery": "Nouvelle-Aquitaine, France"},
    "Hauts-de-France": {"state": "Hauts-de-France", "aroundQuery": "Hauts-de-France, France"},
    "Grand Est": {"state": "Grand Est", "aroundQuery": "Grand Est, France"},
    "Pays de la Loire": {"state": "Pays de la Loire", "aroundQuery": "Pays de la Loire, France"},
    "Bretagne": {"state": "Bretagne", "aroundQuery": "Bretagne, France"},
}


# ==================================================
# UI — CSS/HTML helpers
# ==================================================
def inject_global_css() -> None:
    st.markdown(
        """
<style>
/* ---------- Base ---------- */
:root{
  --card:#FFFFFF;
  --muted:#64748B;
  --text:#0F172A;
  --blue:#2563EB;
  --blue2:#1D4ED8;
  --sky:#0EA5E9;
  --border: rgba(15,23,42,.10);
  --shadow: 0 14px 34px rgba(2, 8, 23, .10);
  --radius: 18px;
}

/* Main background (blue subtle) */
.stApp{
  background:
    radial-gradient(1200px 420px at 12% -10%, rgba(14,165,233,.20), transparent 60%),
    radial-gradient(1000px 420px at 88% -20%, rgba(37,99,235,.18), transparent 55%),
    linear-gradient(180deg, #F6FAFF 0%, #F7FBFF 30%, #F4F8FF 100%);
}

/* Layout width */
.block-container{
  padding-top: 1.0rem !important;
  padding-bottom: 2.2rem !important;
  max-width: 1380px;
}

/* Hide extra chrome spacing */
header[data-testid="stHeader"]{ background: transparent; }
div[data-testid="stToolbar"]{ right: 1rem; }

/* ---------- Topbar / Hero ---------- */
.jme-topbar{
  border-radius: 22px;
  padding: 18px 18px;
  background:
    radial-gradient(1200px 400px at 10% 0%, rgba(14,165,233,.35), transparent 60%),
    radial-gradient(900px 380px at 90% 10%, rgba(37,99,235,.35), transparent 55%),
    linear-gradient(90deg, rgba(37,99,235,.88), rgba(14,165,233,.78));
  color: #fff;
  box-shadow: var(--shadow);
  border: 1px solid rgba(255,255,255,.18);
}
.jme-topbar .row{
  display:flex; align-items:center; justify-content:space-between; gap:14px;
}
.jme-brand{ display:flex; align-items:center; gap:12px; }
.jme-logo{
  width:42px;height:42px;border-radius:14px;
  background: rgba(255,255,255,.18);
  display:flex; align-items:center; justify-content:center;
  border: 1px solid rgba(255,255,255,.22);
  font-size: 18px;
}
.jme-title{ font-size: 22px; font-weight: 900; line-height:1.1; }

/* ---------- Cards ---------- */
.card{
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  box-shadow: var(--shadow);
  padding: 16px 16px;
}
.card-title{
  font-weight:900;
  font-size: 15px;
  color: var(--text);
  margin-bottom: 10px;
  display:flex; align-items:center; gap:8px;
}
.card-sub{
  color: var(--muted);
  font-size: 12px;
  margin-top: -4px;
  margin-bottom: 8px;
}

/* KPI cards */
.kpi-grid{
  display:grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 12px;
  margin-top: 10px;
}
.kpi{
  background: linear-gradient(180deg, #FFFFFF, #F8FBFF);
  border: 1px solid rgba(37,99,235,.12);
  border-radius: 16px;
  padding: 14px 16px;
  box-shadow: 0 10px 24px rgba(2,8,23,.08);
}
.kpi .label{ font-size: 12px; color: var(--muted); }
.kpi .value{ font-size: 22px; font-weight: 900; color: #2563EB; margin-top: 2px; }
.kpi .hint{ font-size: 12px; color: rgba(37,99,235,.95); margin-top: 2px; font-weight:700; }

@media (max-width: 1150px){
  .kpi-grid{ grid-template-columns: repeat(2, minmax(0,1fr)); }
}
@media (max-width: 650px){
  .kpi-grid{ grid-template-columns: 1fr; }
}

/* ---------- Inputs polish ---------- */
div[data-testid="stTextInput"] input,
div[data-testid="stNumberInput"] input,
div[data-testid="stSelectbox"] div[role="combobox"],
div[data-testid="stTextArea"] textarea{
  border-radius: 14px !important;
}
div[data-testid="stButton"] button{
  border-radius: 14px !important;
  font-weight: 800 !important;
  padding: 10px 14px !important;
  border: 1px solid rgba(37,99,235,.25) !important;
}
div[data-testid="stButton"] button[kind="primary"]{
  background: linear-gradient(90deg, var(--blue), var(--sky)) !important;
  border: 1px solid rgba(255,255,255,.15) !important;
}
div[data-testid="stRadio"] label{
  background: rgba(37,99,235,.06);
  border: 1px solid rgba(37,99,235,.12);
  padding: 8px 12px;
  border-radius: 999px;
  margin-right: 8px;
}

/* Tabs polish */
div[data-testid="stTabs"] button{
  border-radius: 999px !important;
  padding: 10px 14px !important;
  font-weight: 900 !important;
}
</style>
        """,
        unsafe_allow_html=True,
    )


def topbar() -> None:
    st.markdown(
        f"""
<div class="jme-topbar">
  <div class="row">
    <div class="jme-brand">
      <div>
        <div class="jme-title">{APP_NAME}</div>
      </div>
    </div>
  </div>
</div>
        """,
        unsafe_allow_html=True,
    )


def card_open(title: str, icon: str = "", subtitle: str = "") -> None:
    sub_html = f"""<div class="card-sub">{subtitle}</div>""" if subtitle else ""
    st.markdown(
        f"""
<div class="card">
  <div class="card-title">{icon} {title}</div>
  {sub_html}
""",
        unsafe_allow_html=True,
    )


def card_close() -> None:
    st.markdown("</div>", unsafe_allow_html=True)


def kpi_row(items: List[dict]) -> None:
    cards = []
    for it in items:
        cards.append(
            f"""
<div class="kpi">
  <div class="label">{it.get("label","")}</div>
  <div class="value">{it.get("value","")}</div>
  <div class="hint">{it.get("hint","")}</div>
</div>
"""
        )
    st.markdown(f"""<div class="kpi-grid">{''.join(cards)}</div>""", unsafe_allow_html=True)


# ==================================================
# TEMP / CSV HELPERS
# ==================================================
def _safe_remove(path: Optional[str]) -> None:
    if path and os.path.exists(path):
        try:
            os.remove(path)
        except Exception:
            pass


def _cleanup_old_tmp(prefix: str = TMP_PREFIX, ttl_seconds: int = TMP_TTL_SECONDS) -> None:
    tmpdir = tempfile.gettempdir()
    now = time.time()
    try:
        for name in os.listdir(tmpdir):
            if not (name.startswith(prefix) and name.endswith(".csv")):
                continue
            p = os.path.join(tmpdir, name)
            try:
                if now - os.path.getmtime(p) > ttl_seconds:
                    os.remove(p)
            except Exception:
                continue
    except Exception:
        pass


def _parse_list_cell(v: Any) -> list:
    """Restore list columns after CSV reload (lists become strings)."""
    if isinstance(v, list):
        return v
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return []
    if isinstance(v, str):
        s = v.strip()
        if not s:
            return []
        if (s.startswith("[") and s.endswith("]")) or (s.startswith("(") and s.endswith(")")):
            try:
                x = ast.literal_eval(s)
                return x if isinstance(x, list) else []
            except Exception:
                return []
        if "," in s:
            return [t.strip() for t in s.split(",") if t.strip()]
        return [s]
    return []


def _ensure_list_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in LIST_COLS:
        if c not in df.columns:
            df[c] = [[] for _ in range(len(df))]
        else:
            df[c] = df[c].apply(_parse_list_cell)
    return df


def _save_temp_csv(df: pd.DataFrame, tag: str) -> str:
    tmpdir = tempfile.gettempdir()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(tmpdir, f"{TMP_PREFIX}{tag}_{ts}.csv")
    df.to_csv(path, index=False)
    return path


def _reload_snapshot(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = _ensure_list_cols(df)
    return df


# ==================================================
# FT helpers: flatten + keep coords for map
# ==================================================
def _flatten_ft(df_raw: pd.DataFrame) -> pd.DataFrame:
    records = df_raw.to_dict(orient="records")
    return pd.json_normalize(records)


def _attach_ft_coords(df_pre: pd.DataFrame, df_flat: pd.DataFrame) -> pd.DataFrame:
    df_pre = df_pre.copy()
    coords = df_flat[["id", "lieuTravail.latitude", "lieuTravail.longitude"]].copy()
    coords = coords.rename(columns={"lieuTravail.latitude": "latitude", "lieuTravail.longitude": "longitude"})
    coords["latitude"] = pd.to_numeric(coords["latitude"], errors="coerce")
    coords["longitude"] = pd.to_numeric(coords["longitude"], errors="coerce")
    return df_pre.merge(coords, on="id", how="left")


def _prepare_wttj_for_preprocess(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()
    if "industry_category" not in df.columns:
        df["industry_category"] = df.get("industry", "")
    for col in ["competences_techniques", "savoir_faire", "savoir_etre", "langues"]:
        if col not in df.columns:
            df[col] = [[] for _ in range(len(df))]
    return df


# ==================================================
# WTTJ URL helper
# ==================================================
def apply_wttj_location_to_url(base_url: str, loc: Optional[dict]) -> str:
    if not loc:
        return base_url
    parts = urllib.parse.urlsplit(base_url)
    qs = urllib.parse.parse_qs(parts.query, keep_blank_values=True)

    if "refinementList[offices.country_code][]" not in qs:
        qs["refinementList[offices.country_code][]"] = ["FR"]

    state = loc.get("state")
    around = loc.get("aroundQuery")

    if state:
        qs["refinementList[offices.state][]"] = [state]
    if around:
        qs["aroundQuery"] = [around]

    new_query = urllib.parse.urlencode(qs, doseq=True, safe="[]")
    return urllib.parse.urlunsplit((parts.scheme, parts.netloc, parts.path, new_query, parts.fragment))


# ==================================================
# STREAMLIT
# ==================================================
st.set_page_config(page_title=APP_NAME, layout="wide")
inject_global_css()
topbar()
_cleanup_old_tmp()

# Session init
if "df" not in st.session_state:
    st.session_state.df = None
if "tmp_csv" not in st.session_state:
    st.session_state.tmp_csv = None
if "source" not in st.session_state:
    st.session_state.source = None

st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

# Defaults (avoid UnboundLocalError across reruns)
ft_keywords = ""
ft_location = ""
ft_contract = ""
ft_max = 300
ft_step = 150

wttj_keywords_raw = ""
wttj_pages = 3
wttj_base_url = "https://www.welcometothejungle.com/fr/jobs?refinementList%5Boffices.country_code%5D%5B%5D=FR"
wttj_headless = True
wttj_sleep = 1.0
selected_loc_label = "Toutes (FR)"
use_exp_filter = False
wttj_min_exp: Optional[int] = None
wttj_max_exp: Optional[int] = None
wttj_contract_filters: List[str] = ["FULL_TIME"]

uploaded = None

# ==================================================
# FILTER CARD — FULL WIDTH (TOP)
# ==================================================
st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

source = st.radio(
    "Source des offres",
    ["France Travail", "Welcome to the Jungle"],
    horizontal=True,
    label_visibility="collapsed"
)

with st.form("load_form"):
    mode = st.selectbox("Mode", ["Live (API/Selenium)", "Télécharger un CSV"], index=0)

    if source == "France Travail":
        c1, c2, c3, c4 = st.columns([2, 2, 1, 1])
        ft_keywords = c1.text_input("Mots-clés", value=ft_keywords)
        ft_location = c2.text_input("Lieu (optionnel)", value=ft_location)
        ft_max = c3.number_input("Max results", min_value=50, max_value=2000, value=int(ft_max), step=50)
        ft_step = c4.number_input("Step", min_value=50, max_value=150, value=int(ft_step), step=10)
        ft_contract = st.multiselect(
            "Type de contract (optionnel)",
            ["CDI", "CDD", "Franchise", "Intérim", "Apprentissage", "Stage", "Profession libérale"]
        )

    else:
        c1, c2, c3 = st.columns([2, 1, 1])
        wttj_keywords_raw = c1.text_input("Mots-clés", value=wttj_keywords_raw)
        wttj_pages = c2.number_input("Max pages / keyword", min_value=1, max_value=20, value=int(wttj_pages), step=1)
        wttj_sleep = c3.slider("Sleep (s)", 0.0, 3.0, float(wttj_sleep), 0.1)

        selected_loc_label = st.selectbox("Lieu", list(LOC_PRESETS.keys()))
        wttj_contract_filters = st.multiselect(
            "Type de contract (optionnel)",
            ["INTERN", "FULL_TIME", "TEMPORAIN"],
        )

    if mode == "Télécharger snapshot CSV":
        uploaded = st.file_uploader("Télécharger un CSV snapshot", type=["csv"])

    submitted = st.form_submit_button("Lancer l'analyse", type="primary")

# Optional: clear snapshot button under the form
if st.session_state.tmp_csv and st.button("Effacer le snapshot", use_container_width=True):
    _safe_remove(st.session_state.tmp_csv)
    st.session_state.tmp_csv = None
    st.session_state.df = None
    st.session_state.source = None
    st.rerun()

card_close()

st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

# ==================================================
# SNAPSHOT + KPI 
# ==================================================

df_tmp = st.session_state.df
total_offres = len(df_tmp) if isinstance(df_tmp, pd.DataFrame) else None

pct_salary = None
pct_cdi = None
if isinstance(df_tmp, pd.DataFrame) and len(df_tmp) > 0:
    # % salary (best-effort)
    try:
        sal_tmp = analyze_salary(df_tmp, source=(st.session_state.source or source))
        pct_salary = sal_tmp["kpis"].get("pct_avec_salaire")
    except Exception:
        pct_salary = None

    # % CDI (best-effort)
    if "contract" in df_tmp.columns:
        try:
            pct_cdi = round(100 * (df_tmp["contract"].astype(str).str.upper().str.contains("CDI").mean()), 0)
            pct_cdi = int(pct_cdi)
        except Exception:
            pct_cdi = None

snap_ok = bool(st.session_state.tmp_csv and os.path.exists(st.session_state.tmp_csv))

kpi_row(
    [
        {
            "label": "Offres analysées",
            "value": f"{total_offres if total_offres is not None else '—'}",
            "hint": "Snapshot actif" if snap_ok else "En attente",
        },
        {"label": "Source", "value": st.session_state.source or "—", "hint": "Live pipeline"},
        {"label": "% offres avec salaire", "value": f"{pct_salary}%" if pct_salary is not None else "—", "hint": "selon parsing"},
        {"label": "% CDI", "value": f"{pct_cdi}%" if pct_cdi is not None else "—", "hint": "approx."},
    ]
)

st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

a1, a2, a3 = st.columns([1.2, 1.2, 1], gap="small")

if snap_ok:
    with open(st.session_state.tmp_csv, "rb") as f:
        a1.download_button(
            "Télécharger les données (CSV)",
            data=f,
            file_name="job_market_snapshot.csv",
            mime="text/csv",
            use_container_width=True,
        )
else:
    a1.button("Télécharger les données (CSV)", disabled=True, use_container_width=True)

a2.button("Télécharger le rapport", disabled=True, use_container_width=True)

if snap_ok:
    try:
        mtime = os.path.getmtime(st.session_state.tmp_csv)
        dt = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
        a3.markdown(
            f"<div style='margin-top:10px;color:#64748B;font-size:12px;'> {dt}</div>",
            unsafe_allow_html=True,
        )
    except Exception:
        pass

if snap_ok:
    st.success("Snapshot prêt ")
else:
    st.info("Chargez des données pour activer les analyses ")

card_close()

# ==================================================
# Handle submit (load / preprocess / snapshot)
# ==================================================
if submitted:
    # reset
    _safe_remove(st.session_state.tmp_csv)
    st.session_state.tmp_csv = None
    st.session_state.df = None
    st.session_state.source = None

    # --------------------
    # Upload mode
    # --------------------
    if mode == "Upload snapshot CSV (préprocessé)":
        with st.spinner("Chargement CSV..."):
            if uploaded is None:
                st.error("Veuillez uploader un CSV.")
                st.stop()
            df = pd.read_csv(uploaded)
            df = _ensure_list_cols(df)

    # --------------------
    # Live mode
    # --------------------
    else:
        if source == "France Travail":
            with st.spinner("France Travail: chargement..."):
                try:
                    token = get_ft_access_token()
                except KeyError:
                    st.error("FT_CLIENT_ID / FT_CLIENT_SECRET manquants dans les variables d'environnement.")
                    st.stop()

                df_raw = fetch_all_ft_offers(
                    token=token,
                    keywords=ft_keywords,
                    location=ft_location or None,
                    contract_type=ft_contract or None,
                    step=int(ft_step),
                    max_results=int(ft_max),
                )

                if df_raw.empty:
                    st.warning("Aucune offre FT trouvée.")
                    st.stop()

                df_flat = _flatten_ft(df_raw)
                df_pre = preprocess_ft(df_flat)
                df_pre = _attach_ft_coords(df_pre, df_flat)
                df = df_pre

        else:
            kws = [k.strip() for k in wttj_keywords_raw.split(",") if k.strip()]
            if not kws:
                st.error("Veuillez fournir au moins 1 mot-clé.")
                st.stop()

            loc = LOC_PRESETS.get(selected_loc_label)
            base_url_final = apply_wttj_location_to_url(wttj_base_url.strip(), loc)

            status = st.status("WTTJ: démarrage...", expanded=True)
            bar = st.progress(0)
            msg = st.empty()

            def on_progress(p: dict):
                pct = int((p.get("progress", 0) or 0) * 100)
                pct = max(0, min(100, pct))
                bar.progress(pct)
                m = p.get("msg", "")
                if m:
                    msg.write(m)

                stage = p.get("stage", "running")
                if stage == "done":
                    status.update(label=m or "Terminé", state="complete")
                else:
                    status.update(label=f"[{pct}%] {stage}", state="en progress")

            try:
                df_raw = fetch_wttj_offers(
                    keywords=kws,
                    base_url=base_url_final,
                    max_pages=int(wttj_pages),
                    headless=bool(wttj_headless),
                    sleep_time=float(wttj_sleep),
                    min_experience_years=wttj_min_exp,
                    max_experience_years=wttj_max_exp,
                    contract_filters=wttj_contract_filters if wttj_contract_filters else None,
                    on_progress=on_progress,
                )
            except Exception as e:
                status.update(label=f"WTTJ: erreur  {e}", state="error")
                st.stop()

            if df_raw.empty:
                st.warning("Aucune offre WTTJ trouvée (après filtres).")
                st.stop()

            df_raw2 = _prepare_wttj_for_preprocess(df_raw)
            df_pre = preprocess_wttj(df_raw2)
            df = df_pre

    # snapshot temp CSV -> reload -> store in session
    tag = "ft" if source == "France Travail" else "wttj"
    tmp_path = _save_temp_csv(df, tag=tag)
    df_reload = _reload_snapshot(tmp_path)

    st.session_state.tmp_csv = tmp_path
    st.session_state.df = df_reload
    st.session_state.source = source

    st.rerun()

# ==================================================
# Guard: stop before tabs if no data
# ==================================================
if st.session_state.df is None:
    st.stop()

df_source = st.session_state.df.copy()
source_loaded = st.session_state.source or source

st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

# ==================================================
# TABS
# ==================================================
tabs = st.tabs([" Market", " Skills", " Salary", "️ Geography", " CV Matching", " Related jobs"])

# --- Market
with tabs[0]:
    show_cols = [x for x in ["title", "company", "location", "contract", "salary", "url"] if x in df_source.columns]
    st.dataframe(df_source[show_cols].head(50), use_container_width=True, height=360)
    card_close()

    card_open("Top localisations", "")
    st.dataframe(jobs_by_location(df_source, top_n=20), use_container_width=True)
    card_close()

# --- Skills
with tabs[1]:
    card_open("Analyse des compétences", "")
    if source_loaded == "France Travail":
        skills = analyze_skills_ft(df_source)
        st.markdown("**Hard skills (required / optional)**")
        st.dataframe(skills["hard_skills"], use_container_width=True)

        st.markdown("**Soft skills**")
        st.dataframe(skills["soft_skills"], use_container_width=True)

        st.markdown("**Compétences émergentes**")
        st.dataframe(skills["emerging_skills"], use_container_width=True)
    else:
        skills = analyze_skills_wttj(df_source)
        st.markdown("**Hard skills**")
        st.dataframe(skills["hard_skills"], use_container_width=True)

        st.markdown("**Soft skills**")
        st.dataframe(skills["soft_skills"], use_container_width=True)
    card_close()

# --- Salary
with tabs[2]:
    card_open("Analyse des salaires", "")
    sal = analyze_salary(df_source, source=source_loaded)
    kpis = sal["kpis"]

    k1, k2, k3 = st.columns(3)
    k1.metric("Total offres", kpis["total_offres"])
    k2.metric("Offres avec salaire", kpis["offres_avec_salaire"])
    k3.metric("% avec salaire", f"{kpis['pct_avec_salaire']} %")
    card_close()

    card_open("Distribution (approx.)", "")
    if sal.get("distribution"):
        st.json(sal["distribution"])
    else:
        st.info("Pas de distribution exploitable (ou transparence uniquement).")
    card_close()

    with st.expander("Voir df_salary"):
        st.dataframe(sal["df_salary"].head(200), use_container_width=True)

# --- Geography
with tabs[3]:
    card_open("Répartition géographique", "️")
    if source_loaded == "France Travail":
        try:
            job_map = create_ft_jobs_map(df_source)
            st.components.v1.html(job_map._repr_html_(), height=600)
            st.caption("Carte: uniquement les offres avec latitude/longitude.")
        except Exception as e:
            st.warning(f"Carte non dispo: {e}")
            st.dataframe(jobs_by_location(df_source, top_n=30), use_container_width=True)
    else:
        st.info("WTTJ: pas de coordonnées dans ce pipeline → pas de carte.")
        st.dataframe(jobs_by_location(df_source, top_n=30), use_container_width=True)
    card_close()

# --- Matching
with tabs[4]:
    card_open("Matching CV  Offres", "", "Compare vos skills (et embeddings optionnels) aux annonces")
    with st.form("cv_form"):
        colA, colB, colC = st.columns(3)
        min_score = colA.slider("Score minimum", 0, 100, 40, 5)
        use_sem_skills = colB.checkbox("Semantic skills", value=True)
        use_desc_sem = colC.checkbox("Semantic description", value=True)
        skill_threshold = st.slider("Skill similarity threshold", 0.50, 0.95, 0.75, 0.05)

        skills_hard = st.text_input("Hard skills (virgules)", value="python, sql")
        skills_soft = st.text_input("Soft skills (virgules)", value="rigueur, communication")
        languages = st.text_input("Langues (virgules)", value="anglais, français")
        cv_text = st.text_area("Résumé / expériences (texte libre)", height=140)

        submitted_cv = st.form_submit_button(" Lancer le matching", type="primary")
    card_close()

    if submitted_cv:
        cv = {
            "skills_hard": [s.strip() for s in skills_hard.split(",") if s.strip()],
            "skills_soft": [s.strip() for s in skills_soft.split(",") if s.strip()],
            "languages": [s.strip() for s in languages.split(",") if s.strip()],
            "text": cv_text,
        }

        with st.spinner("Matching..."):
            results = match_cv_market(
                cv,
                df_source,
                min_score=float(min_score),
                use_semantic_skills=bool(use_sem_skills),
                skill_threshold=float(skill_threshold),
                use_description_semantic=bool(use_desc_sem),
            )

        if results.empty:
            st.warning("Aucune offre correspondante trouvée.")
        else:
            cols = [
                "score",
                "job_title",
                "company",
                "location",
                "source",
                "hard_required_coverage",
                "hard_optional_coverage",
                "soft_coverage",
                "languages_required_coverage",
                "description_similarity",
                "url",
            ]
            cols = [c for c in cols if c in results.columns]

            card_open("Résultats", "")
            st.dataframe(results[cols].head(50), use_container_width=True, height=420)
            card_close()

            with st.expander("Voir détail complet"):
                st.dataframe(results.head(50), use_container_width=True)

# --- Related job titles
with tabs[5]:
    card_open("Métiers proches (semantic)", "", "Charge un SentenceTransformer")
    query_title = st.text_input("Intitulé de poste recherché")
    min_sim = st.slider("Min similarity", 0.50, 0.95, 0.70, 0.05)
    top_n = st.slider("Top N", 5, 30, 10, 1)

    if st.button(" Charger le modèle & chercher", type="primary"):
        with st.spinner("Chargement + recherche..."):
            from analysis.job_titles import find_related_job_titles  # heavy import only here

            related = find_related_job_titles(
                df_source,
                query_title=query_title,
                top_n=int(top_n),
                min_similarity=float(min_sim),
            )

        if related.empty:
            st.info("Aucun titre proche trouvé.")
        else:
            st.dataframe(related, use_container_width=True)
    card_close()
