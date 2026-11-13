import base64
import json
import math
import os
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_echarts import st_echarts

BASE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(BASE, "models")

st.set_page_config(page_title="WWTP BOD & COD Predictor", page_icon="💧",
                   layout="wide", initial_sidebar_state="collapsed")

# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------
NEQS = {"bod": 80, "cod": 150}          # NEQS 2000, discharge to inland waters (reference lines only)
ACCENT, GOLD, BLUE, MUTED, TEXT = "#2dd4bf", "#fbbf24", "#60a5fa", "#9fb3ad", "#e8f3ef"
HEAT_COLORS = ["#0b3d3a", "#14756e", "#2fb49c", "#9be3c1", "#f4f1bb"]   # sequential, colour-blind safe

STAGE_LABELS = {"Inlet": "Inlet (raw influent)", "Aeration Tank": "Aeration tank", "Effluent": "Effluent (discharge)"}
STAGE_ORDER = ["Inlet", "Aeration Tank", "Effluent"]
SENSORS = [  # feature, label, min, max, step, format
    ("pH", "pH", 0.0, 14.0, 0.1, "%.2f"),
    ("Temperature (°C)", "Temperature (°C)", 0.0, 50.0, 0.1, "%.1f"),
    ("DO (mg/L)", "Dissolved oxygen (mg/L)", 0.0, 20.0, 0.1, "%.2f"),
    ("Turbidity (NTU)", "Turbidity (NTU)", 0.0, 3000.0, 1.0, "%.0f"),
    ("TDS (mg/L)", "Total dissolved solids (mg/L)", 0.0, 10000.0, 10.0, "%.0f"),
    ("EC (µS/cm)", "Electrical conductivity (µS/cm)", 0.0, 20000.0, 10.0, "%.0f"),
    ("TSS (mg/L)", "Total suspended solids (mg/L)", 0.0, 10000.0, 5.0, "%.0f"),
]
LANG = {
    "en": {"title": "Wastewater BOD & COD Predictor",
           "sub": "Estimate BOD and COD at any stage of the treatment train from seven field measurements",
           "btn": "Predict BOD and COD",
           "disc": "Research prototype, FYDP, IIUI 2022-2026. Estimates are not laboratory results. Confirm with laboratory analysis before any reporting."},
    "ur": {"title": "فضلہ پانی BOD اور COD پیش گو",
           "sub": "سات میدانی پیمائشوں سے BOD اور COD کا اندازہ",
           "btn": "BOD اور COD پیش گو کریں",
           "disc": "تحقیقی نمونہ۔ FYDP، IIUI 2022-2026۔ یہ اندازے لیبارٹری نتائج نہیں ہیں۔"},
}


# ----------------------------------------------------------------------------
# Assets
# ----------------------------------------------------------------------------
@st.cache_resource
def load_assets():
    with open(os.path.join(MODELS, "model_card.json")) as f:
        card = json.load(f)
    with open(os.path.join(MODELS, "heatmap_data.json")) as f:
        heat = json.load(f)
    bod = joblib.load(os.path.join(MODELS, "bod_model.joblib"))
    cod = joblib.load(os.path.join(MODELS, "cod_model.joblib"))
    return card, heat, bod, cod


card, heat, bod_model, cod_model = load_assets()
SENS_KEYS = [s[0] for s in SENSORS]

for k, v in {"lang": "en", "history": [], "last": None}.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ----------------------------------------------------------------------------
# Styling
# ----------------------------------------------------------------------------
def inject_css():
    bg_path = os.path.join(BASE, "assets", "bg.jpg")
    if os.path.exists(bg_path):
        with open(bg_path, "rb") as f:
            image = "url(data:image/jpeg;base64," + base64.b64encode(f.read()).decode() + ")"
    else:
        image = "linear-gradient(135deg,#0b1f1a,#10261f 50%,#06120e)"
    css = """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="css"], .stApp { font-family: 'Inter', 'Segoe UI', Arial, sans-serif; }
    .stApp { background: linear-gradient(rgba(5,12,10,.48), rgba(5,12,10,.80)), __IMG__;
             background-size: cover; background-position: center; background-attachment: fixed; }
    header[data-testid="stHeader"] { background: transparent; }
    .block-container { padding-top: 1.6rem; max-width: 1280px; }
    div[class*="st-key-card"] {
        background: rgba(14,29,25,.58); backdrop-filter: blur(14px); -webkit-backdrop-filter: blur(14px);
        border: 1px solid rgba(255,255,255,.10); border-radius: 18px; padding: 16px 18px; }
    [data-testid="stSidebar"] { background: rgba(8,18,15,.92); }
    .hero { padding: 6px 4px 14px 4px; }
    .hero-title { font-size: 2.1rem; font-weight: 700; color: #f1faf7; letter-spacing: -.01em; }
    .hero-sub { color: #9fb3ad; font-size: 1.02rem; margin-top: 2px; }
    .badges span { display: inline-block; margin: 10px 8px 0 0; padding: 4px 12px; border-radius: 999px;
        font-size: .76rem; color: #b6ece3; background: rgba(45,212,191,.10); border: 1px solid rgba(45,212,191,.28); }
    .card-title { font-size: .95rem; font-weight: 600; color: #e8f3ef; margin-bottom: 2px; }
    .card-note { font-size: .78rem; color: #9fb3ad; }
    .tile { padding: 12px 14px; border-radius: 14px; background: rgba(255,255,255,.04);
            border: 1px solid rgba(255,255,255,.08); height: 100%; }
    .tl { font-size: .74rem; color: #9fb3ad; text-transform: uppercase; letter-spacing: .04em; }
    .tv { font-size: 1.45rem; font-weight: 700; color: #f1faf7; margin-top: 2px; }
    .ts { font-size: .74rem; color: #9fb3ad; margin-top: 2px; }
    .legend-dot { display:inline-block; width:10px; height:10px; border-radius:50%; margin-right:6px; }
    div.stButton > button[kind="primary"] { background: linear-gradient(135deg,#14b8a6,#0d9488); border: 0;
        border-radius: 12px; font-weight: 600; padding: .6rem 1rem; }
    </style>
    """
    st.markdown(css.replace("__IMG__", image), unsafe_allow_html=True)


inject_css()
L = LANG[st.session_state.lang]


# ----------------------------------------------------------------------------
# Model helpers
# ----------------------------------------------------------------------------
def predict(stage, values):
    row = {f: float(values[f]) for f in SENS_KEYS}
    for s in card["stages"]:
        row[s] = 1.0 if s == stage else 0.0
    X = pd.DataFrame([row])[card["features"]]
    out = {}
    for tag, model in (("bod", bod_model), ("cod", cod_model)):
        v = max(0.0, float(model.predict(X)[0]))
        hw = float(card[tag]["interval_half_width_90"][stage])
        out[tag] = {"value": v, "lo": max(0.0, v - hw), "hi": v + hw,
                    "base": float(card[tag]["stage_mean"][stage])}
    return out


def out_of_range(stage, values, skip):
    flagged = []
    for f in SENS_KEYS:
        if f in skip:
            continue
        r = card["ranges"][stage][f]
        if values[f] < r["p5"] or values[f] > r["p95"]:
            flagged.append(f)
    return flagged


def nice_max(*vals):
    top = max(vals)
    step = 50 if top < 600 else 100
    return int(math.ceil(top / step) * step)


# ----------------------------------------------------------------------------
# Chart builders (ECharts options)
# ----------------------------------------------------------------------------
def gauge_option(name, d, limit):
    vmax = nice_max(d["hi"] * 1.1, limit * 1.3, d["value"] * 1.2)
    common = {"type": "gauge", "startAngle": 200, "endAngle": -20, "min": 0, "max": vmax,
              "radius": "95%", "center": ["50%", "62%"], "splitNumber": 5}
    main = dict(common, **{
        "axisLine": {"lineStyle": {"width": 16, "color": [
            [d["lo"] / vmax, "rgba(255,255,255,.10)"], [d["hi"] / vmax, ACCENT], [1, "rgba(255,255,255,.10)"]]}},
        "pointer": {"length": "60%", "width": 5, "itemStyle": {"color": "#f8fafc"}},
        "anchor": {"show": True, "size": 10, "itemStyle": {"color": "#f8fafc"}},
        "axisTick": {"show": False}, "splitLine": {"show": False},
        "axisLabel": {"color": MUTED, "distance": -34, "fontSize": 10},
        "title": {"show": False},
        "detail": {"valueAnimation": True, "formatter": "{value}", "fontSize": 34, "fontWeight": 700,
                   "color": "#f8fafc", "offsetCenter": [0, "42%"]},
        "data": [{"value": round(d["value"], 1), "name": name}]})
    limit_line = dict(common, **{
        "axisLine": {"show": False}, "axisTick": {"show": False}, "splitLine": {"show": False},
        "axisLabel": {"show": False}, "anchor": {"show": False}, "title": {"show": False}, "detail": {"show": False},
        "pointer": {"length": "95%", "width": 2, "itemStyle": {"color": GOLD}},
        "data": [{"value": limit}]})
    return {"series": [main, limit_line]}


def range_option(name, d, limit):
    vmax = nice_max(d["hi"] * 1.1, limit * 1.3, d["base"] * 1.2)
    return {
        "grid": {"left": 6, "right": 24, "top": 30, "bottom": 22, "containLabel": True},
        "tooltip": {"trigger": "item"},
        "xAxis": {"type": "value", "min": 0, "max": vmax, "axisLabel": {"color": MUTED},
                  "splitLine": {"lineStyle": {"color": "rgba(255,255,255,.08)"}}},
        "yAxis": {"type": "category", "data": [name], "axisLabel": {"show": False}, "axisTick": {"show": False},
                  "axisLine": {"show": False}},
        "series": [
            {"type": "bar", "stack": "r", "silent": True, "data": [round(d["lo"], 1)],
             "itemStyle": {"color": "transparent"}},
            {"type": "bar", "stack": "r", "name": "90% interval", "barWidth": 22, "data": [round(d["hi"] - d["lo"], 1)],
             "itemStyle": {"color": "rgba(45,212,191,.35)", "borderRadius": 6},
             "markLine": {"symbol": "none", "silent": True, "lineStyle": {"color": GOLD, "type": "dashed", "width": 2},
                          "label": {"formatter": "NEQS {c}", "color": GOLD}, "data": [{"xAxis": limit}]}},
            {"type": "scatter", "name": "Prediction", "data": [[round(d["value"], 1), name]], "symbolSize": 16,
             "itemStyle": {"color": "#f8fafc"}},
            {"type": "scatter", "name": "Stage average", "data": [[round(d["base"], 1), name]], "symbol": "diamond",
             "symbolSize": 15, "itemStyle": {"color": BLUE}},
        ]}


def heat_option(matrix, xs, ys, unit):
    flat = [v for row in matrix for v in row if v is not None]
    lo_v, hi_v = min(flat), max(flat)
    span = max(hi_v - lo_v, 1e-9)
    data = [{"value": [j, i, v], "label": {"color": "#062a25" if (v - lo_v) / span > 0.62 else "#ffffff"}}
            for i, row in enumerate(matrix) for j, v in enumerate(row) if v is not None]
    return {
        "tooltip": {"position": "top"},
        "grid": {"left": 4, "right": 10, "top": 6, "bottom": 52, "containLabel": True},
        "xAxis": {"type": "category", "data": xs, "axisLabel": {"color": MUTED}, "splitArea": {"show": False}},
        "yAxis": {"type": "category", "data": ys, "axisLabel": {"color": MUTED}},
        "visualMap": {"min": math.floor(min(flat)), "max": math.ceil(max(flat)), "calculable": False,
                      "orient": "horizontal", "left": "center", "bottom": 0, "textStyle": {"color": MUTED},
                      "inRange": {"color": HEAT_COLORS}},
        "series": [{"type": "heatmap", "data": data, "label": {"show": True, "fontSize": 10, "fontWeight": 600},
                    "itemStyle": {"borderColor": "rgba(0,0,0,.30)", "borderWidth": 1}}]}


def spark_option(xs, ys, color):
    return {"grid": {"left": 6, "right": 6, "top": 8, "bottom": 4},
            "tooltip": {"trigger": "axis"},
            "xAxis": {"type": "category", "show": False, "data": xs},
            "yAxis": {"type": "value", "show": False, "scale": True},
            "series": [{"type": "line", "data": ys, "smooth": True, "symbol": "circle", "symbolSize": 6,
                        "lineStyle": {"width": 2, "color": color}, "itemStyle": {"color": color},
                        "areaStyle": {"color": color, "opacity": 0.12}}]}


def tile(label, value, sub=""):
    return (f"<div class='tile'><div class='tl'>{label}</div><div class='tv'>{value}</div>"
            f"<div class='ts'>{sub}</div></div>")


# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## WWTP Predictor")
    choice = st.radio("Language / زبان", ["English", "اردو"], horizontal=True)
    st.session_state.lang = "ur" if choice == "اردو" else "en"
    L = LANG[st.session_state.lang]
    st.divider()
    st.markdown("### Reference limits")
    st.caption("NEQS 2000, discharge to inland waters. Shown as reference lines only. The tool gives no pass or fail verdict.")
    st.markdown("| Parameter | Limit |\n|---|---|\n| BOD | 80 mg/L |\n| COD | 150 mg/L |")
    st.divider()
    st.markdown("### How to use")
    st.markdown("1. Choose the sampling stage.\n2. Enter the seven readings.\n3. Press **Predict**.\n\n"
                "Read the bar: the white dot is the estimate, the shaded band is the 90% interval, "
                "the blue diamond is the typical value for that stage, and the gold line is the NEQS reference.")

# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------
st.markdown(
    f"<div class='hero'><div class='hero-title'>{L['title']}</div><div class='hero-sub'>{L['sub']}</div>"
    "<div class='badges'><span>Research prototype</span><span>Municipal WWTP, Islamabad</span>"
    "<span>Plant records Feb to Oct 2025 + lab campaign Dec 2025 to Feb 2026</span></div></div>",
    unsafe_allow_html=True)

left, right = st.columns([1, 2], gap="large")

# ----------------------------------------------------------------------------
# Inputs
# ----------------------------------------------------------------------------
with left:
    with st.container(key="card_inputs"):
        st.markdown("<div class='card-title'>Sample inputs</div>", unsafe_allow_html=True)
        stage = st.radio("Sampling stage", STAGE_ORDER, format_func=lambda s: STAGE_LABELS[s], key="stage")
        vals = {}
        for feat, label, lo, hi, step, fmt in SENSORS:
            typical = float(round(card["ranges"][stage][feat]["p50"], 2))
            vals[feat] = st.number_input(label, min_value=lo, max_value=hi, value=min(max(typical, lo), hi),
                                         step=step, format=fmt, key=f"{feat}_{stage}")
        unknown = st.multiselect("Not measured (stage typical value is used)", [s[0] for s in SENSORS], key=f"unk_{stage}")
        go = st.button(L["btn"], type="primary", use_container_width=True)

if go:
    used = dict(vals)
    for f in unknown:
        used[f] = float(card["ranges"][stage][f]["p50"])
    result = predict(stage, used)
    now = datetime.now()
    st.session_state.last = {"stage": stage, "out": result, "unknown": unknown,
                             "ood": out_of_range(stage, used, set(unknown)), "time": now.strftime("%d %b %Y, %H:%M")}
    st.session_state.history.append({"t": now.strftime("%H:%M:%S"), "stage": stage,
                                     "bod": result["bod"]["value"], "cod": result["cod"]["value"]})
    st.session_state.history = st.session_state.history[-10:]

# ----------------------------------------------------------------------------
# Results
# ----------------------------------------------------------------------------
with right:
    last = st.session_state.last
    if last is None:
        with st.container(key="card_empty"):
            st.markdown("<div class='card-title'>Results</div>", unsafe_allow_html=True)
            st.info("Choose a stage, enter the readings and press Predict to see the estimates, intervals and comparisons.")
    else:
        out = last["out"]
        if last["unknown"]:
            st.warning("Some inputs were not measured and were replaced by stage typical values. "
                       "The model was not validated with substituted inputs, so treat this estimate with extra caution.")
        if last["ood"]:
            st.warning("Inputs outside the usual range seen for this stage: " + ", ".join(last["ood"]) +
                       ". The model has little data there, so the estimate is less reliable.")
        g1, g2 = st.columns(2)
        for col, tag, name in ((g1, "bod", "BOD"), (g2, "cod", "COD")):
            d = out[tag]
            with col:
                with st.container(key=f"card_gauge_{tag}"):
                    st.markdown(f"<div class='card-title'>{name} estimate</div>"
                                f"<div class='card-note'>{STAGE_LABELS[last['stage']]} · {last['time']}</div>",
                                unsafe_allow_html=True)
                    st_echarts(gauge_option(name, d, NEQS[tag]), height="270px", key=f"g_{tag}_{last['time']}")
                    st.markdown(f"<div class='card-note'><b>90% interval:</b> {d['lo']:.0f} to {d['hi']:.0f} mg/L &nbsp;·&nbsp; "
                                f"<b>stage average:</b> {d['base']:.0f} mg/L &nbsp;·&nbsp; "
                                f"<span style='color:{GOLD}'>NEQS reference {NEQS[tag]} mg/L</span></div>",
                                unsafe_allow_html=True)
        with st.container(key="card_range"):
            st.markdown("<div class='card-title'>Estimate vs stage average</div>"
                        "<div class='card-note'>"
                        "<span class='legend-dot' style='background:#f8fafc'></span>Estimate "
                        f"<span class='legend-dot' style='background:{BLUE};margin-left:12px'></span>Stage average "
                        f"<span class='legend-dot' style='background:{ACCENT};margin-left:12px'></span>90% interval "
                        f"<span class='legend-dot' style='background:{GOLD};margin-left:12px'></span>NEQS reference"
                        "</div>", unsafe_allow_html=True)
            r1, r2 = st.columns(2)
            with r1:
                st.caption("BOD (mg/L)")
                st_echarts(range_option("BOD", out["bod"], NEQS["bod"]), height="120px", key=f"r_bod_{last['time']}")
            with r2:
                st.caption("COD (mg/L)")
                st_echarts(range_option("COD", out["cod"], NEQS["cod"]), height="120px", key=f"r_cod_{last['time']}")

# ----------------------------------------------------------------------------
# Heatmaps (recorded plant data, not live)
# ----------------------------------------------------------------------------
with st.container(key="card_heat"):
    h1, h2, h3 = st.columns([2, 1, 1])
    with h1:
        st.markdown("<div class='card-title'>Recorded plant patterns</div>"
                    f"<div class='card-note'>{heat['period']}. Averages of recorded values, not a live feed.</div>",
                    unsafe_allow_html=True)
    with h2:
        hp = st.radio("Parameter", ["BOD", "COD"], horizontal=True, key="hp")
    with h3:
        hs = st.selectbox("Stage for weekday view", STAGE_ORDER, format_func=lambda s: STAGE_LABELS[s],
                          index=2, key="hs")
    c1, c2 = st.columns(2)
    with c1:
        st.caption(f"{hp} by stage and month (mg/L)")
        st_echarts(heat_option(heat["month_stage"][hp], heat["months"], heat["stages"], "mg/L"),
                   height="260px", key=f"h1_{hp}")
    with c2:
        st.caption(f"{hp} by shift and weekday, {STAGE_LABELS[hs].lower()} (mg/L)")
        st_echarts(heat_option(heat["weekday_shift"][hp][hs], heat["weekdays"], heat["shifts"], "mg/L"),
                   height="260px", key=f"h2_{hp}_{hs}")

# ----------------------------------------------------------------------------
# Session trend + honest reliability panel
# ----------------------------------------------------------------------------
t1, t2 = st.columns([1, 1], gap="large")
with t1:
    with st.container(key="card_session"):
        st.markdown("<div class='card-title'>This session</div>"
                    "<div class='card-note'>Your last predictions in this browser session</div>", unsafe_allow_html=True)
        hist = st.session_state.history
        if len(hist) < 2:
            st.caption("Make at least two predictions to see a trend.")
        else:
            xs = [h["t"] for h in hist]
            s1, s2 = st.columns(2)
            with s1:
                st.caption("BOD (mg/L)")
                st_echarts(spark_option(xs, [round(h["bod"], 1) for h in hist], ACCENT), height="110px", key="sp_bod")
            with s2:
                st.caption("COD (mg/L)")
                st_echarts(spark_option(xs, [round(h["cod"], 1) for h in hist], BLUE), height="110px", key="sp_cod")
with t2:
    with st.container(key="card_trust"):
        v = card["validation"]
        st.markdown("<div class='card-title'>How much can you trust this?</div>"
                    "<div class='card-note'>Honest validation results (BOD / COD)</div>", unsafe_allow_html=True)
        a, b, c = st.columns(3)
        a.markdown(tile("Plant, unseen weeks R²", f"{v['plant_blocked_R2']['BOD']:.2f} / {v['plant_blocked_R2']['COD']:.2f}",
                        "week-blocked validation"), unsafe_allow_html=True)
        b.markdown(tile("Lab campaign R²", f"{v['lab_R2']['BOD']:.2f} / {v['lab_R2']['COD']:.2f}",
                        f"n = {v['lab_n']['BOD']} / {v['lab_n']['COD']}"), unsafe_allow_html=True)
        c.markdown(tile("Within effluent R²", f"{v['within_effluent_R2']['BOD']:.2f} / {v['within_effluent_R2']['COD']:.2f}",
                        "plant, unseen weeks"), unsafe_allow_html=True)
        st.markdown(
            f"<div class='card-note' style='margin-top:10px'>Answering with the stage average alone scores R² "
            f"{v['lab_stage_mean_R2']['BOD']:.2f} / {v['lab_stage_mean_R2']['COD']:.2f} on the lab campaign, "
            f"so most of the accuracy comes from knowing the stage. Inside a single stage, the sensor readings add "
            f"limited information, which is why the intervals are wide. Typical error on lab data is about "
            f"{v['lab_MAPE_pct']['BOD']:.0f}% (BOD) and {v['lab_MAPE_pct']['COD']:.0f}% (COD).</div>",
            unsafe_allow_html=True)

st.caption(L["disc"])
