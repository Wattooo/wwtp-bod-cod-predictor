import base64
import json
import math
import os
from datetime import datetime

import joblib
import pandas as pd
import streamlit as st
from streamlit_echarts import st_echarts

BASE = os.path.dirname(os.path.abspath(__file__))

st.set_page_config(page_title="BOD / COD Estimator", layout="wide", initial_sidebar_state="expanded")

# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------
NEQS = {"bod": 80, "cod": 150}                      # reference marks only
MOSS, OCHRE, BONE, MUTED = "#a8cf7e", "#e3ad45", "#edf1e8", "#a3b3a0"
HEAT = ["#16301f", "#2f5a36", "#5f8f4e", "#a8cf7e", "#ede7b5"]

STAGE_ORDER = ["Inlet", "Aeration Tank", "Effluent"]
STAGE_LABELS = {"Inlet": "Inlet", "Aeration Tank": "Aeration tank", "Effluent": "Effluent"}
SENSORS = [  # feature, short label, field label, min, max, step, format
    ("pH", "pH", "pH", 0.0, 14.0, 0.1, "%.2f"),
    ("Temperature (°C)", "Temperature", "Temp. (°C)", 0.0, 50.0, 0.1, "%.1f"),
    ("DO (mg/L)", "DO", "DO (mg/L)", 0.0, 20.0, 0.1, "%.2f"),
    ("Turbidity (NTU)", "Turbidity", "Turbidity (NTU)", 0.0, 3000.0, 1.0, "%.0f"),
    ("TDS (mg/L)", "TDS", "TDS (mg/L)", 0.0, 10000.0, 10.0, "%.0f"),
    ("EC (µS/cm)", "EC", "EC (µS/cm)", 0.0, 20000.0, 10.0, "%.0f"),
    ("TSS (mg/L)", "TSS", "TSS (mg/L)", 0.0, 10000.0, 5.0, "%.0f"),
]
SENS_KEYS = [s[0] for s in SENSORS]
SHORT = {s[0]: s[1] for s in SENSORS}


# ----------------------------------------------------------------------------
# Assets
# ----------------------------------------------------------------------------
@st.cache_resource
def load_assets():
    with open(os.path.join(BASE, "model_card.json")) as f:
        card = json.load(f)
    with open(os.path.join(BASE, "heatmap_data.json")) as f:
        heat = json.load(f)
    return card, heat, joblib.load(os.path.join(BASE, "bod_model.joblib")), joblib.load(os.path.join(BASE, "cod_model.joblib"))


card, heat, bod_model, cod_model = load_assets()
DATA_NOTE = card.get("data_note", "")

disp = card.get("display")
if disp:
    R2, ERR = disp["r2"], disp["typical_error_pct"]
else:
    R2 = card["validation"]["plant_blocked_R2"]
    ERR = {"BOD": card["bod"]["blocked_MAPE_pct"], "COD": card["cod"]["blocked_MAPE_pct"]}

for k, v in {"history": [], "last": None}.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ----------------------------------------------------------------------------
# Styling
# ----------------------------------------------------------------------------
def inject_css():
    bg = os.path.join(BASE, "bg.jpg")
    if os.path.exists(bg):
        with open(bg, "rb") as f:
            image = "url(data:image/jpeg;base64," + base64.b64encode(f.read()).decode() + ")"
    else:
        image = "linear-gradient(135deg,#0b1f1a,#10261f 50%,#06120e)"
    css = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow:wght@400;500;600&family=Barlow+Condensed:wght@500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');
html, body, .stApp, .stApp p, .stApp label, .stApp span, .stApp div { font-family: 'Barlow', 'Segoe UI', Arial, sans-serif; }
.stApp { background: linear-gradient(rgba(5,12,8,.40), rgba(5,12,8,.66)), __IMG__; background-size: cover;
  background-position: center; background-attachment: fixed; }
.stApp, .stApp p, .stApp li, .stApp label, .stApp span, .stApp h1, .stApp h2, .stApp h3, .stApp h4 { color: #edf1e8; }
header[data-testid="stHeader"] { background: transparent; }
.block-container { padding: 1.1rem 1.6rem 1.4rem 1.6rem; max-width: 1760px; }

/* glass panels */
div[class*="st-key-card"] { background: rgba(14,24,18,.52); backdrop-filter: blur(16px) saturate(1.1);
  -webkit-backdrop-filter: blur(16px) saturate(1.1); border: 1px solid rgba(237,241,232,.17); border-radius: 8px;
  padding: 14px 16px; box-sizing: border-box; }
div[class*="st-key-card"] .element-container, div[class*="st-key-card"] [data-testid="element-container"],
div[class*="st-key-card"] [data-testid="stElementContainer"], div[class*="st-key-card"] [data-testid="stHorizontalBlock"],
div[class*="st-key-card"] [data-testid="stVerticalBlock"] { width: 100% !important; max-width: 100% !important; }

/* type */
.top { display:flex; align-items:baseline; justify-content:space-between; margin: 2px 2px 12px 2px; }
.top .t { font-family:'Barlow Condensed','Arial Narrow','Roboto Condensed',sans-serif; font-weight:700; font-size:2.35rem; letter-spacing:.01em; color:#f3f6ee; line-height:1; }
.top .s { font-size:1rem; color:#b5c4b0; padding-right:130px; }
.ct { font-family:'Barlow Condensed','Arial Narrow','Roboto Condensed',sans-serif; font-weight:600; font-size:1.35rem; color:#f3f6ee; line-height:1.1; }
.ct .u { font-family:'Barlow',sans-serif; font-weight:400; font-size:.9rem; color:#a3b3a0; margin-left:6px; }
.sub { font-size:.88rem; color:#a3b3a0; margin:1px 0 4px 0; }
.stats { display:grid; grid-template-columns: 1.5fr 1fr 1fr; gap:10px; margin-top:4px; }
.stat { min-width:0; overflow:hidden; }
.sv { font-family:'Barlow Condensed','Arial Narrow','Roboto Condensed',sans-serif; font-weight:600; font-size:1.45rem; line-height:1.05; white-space:nowrap;
  font-variant-numeric: tabular-nums; color:#f3f6ee; }
.sc { font-size:.8rem; color:#a3b3a0; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.tiles { display:grid; grid-template-columns: repeat(2, minmax(0,1fr)); gap:10px; margin: 6px 0 8px 0; }
.tile { min-width:0; overflow:hidden; border:1px solid rgba(237,241,232,.15); border-radius:6px; padding:8px 10px; background:rgba(255,255,255,.04); }
.tile .tl { font-size:.82rem; color:#a3b3a0; white-space:nowrap; }
.tile .tv { font-family:'Barlow Condensed','Arial Narrow','Roboto Condensed',sans-serif; font-weight:600; font-size:1.9rem; line-height:1.1; color:#f3f6ee; white-space:nowrap; }
.note { font-size:.86rem; color:#c9d6c4; margin: 2px 2px 10px 2px; min-height:1.1rem; }
.note b { color:#e3ad45; font-weight:600; }
.foot { font-size:.8rem; color:#8fa08c; margin-top:6px; }

/* form controls */
[data-testid="stWidgetLabel"] p, [data-testid="stCheckbox"] label p { color:#dfe8da !important; font-size:.9rem !important; font-weight:500; }
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p { color:#a3b3a0 !important; font-size:.84rem; }
[data-baseweb="input"], [data-baseweb="base-input"] { background-color: rgba(255,255,255,.07) !important;
  border-color: rgba(237,241,232,.26) !important; border-radius: 5px !important; }
[data-baseweb="input"] input, [data-baseweb="base-input"] input { color:#f6f9f2 !important; -webkit-text-fill-color:#f6f9f2 !important;
  font-family:'IBM Plex Mono', monospace !important; font-size:1rem; padding-right:6px; }
[data-baseweb="input"] input:disabled { color:#7f917d !important; -webkit-text-fill-color:#7f917d !important; }
[data-testid="stNumberInputStepUp"], [data-testid="stNumberInputStepDown"] { display:none !important; }
[data-baseweb="select"] > div { background-color: rgba(255,255,255,.07) !important; border-color: rgba(237,241,232,.26) !important; }
[data-baseweb="select"] div, [data-baseweb="select"] span, [data-baseweb="select"] svg { color:#f6f9f2 !important; fill:#f6f9f2 !important; }
ul[role="listbox"] { background:#101c14 !important; } ul[role="listbox"] li, ul[role="listbox"] li * { color:#edf1e8 !important; }
[data-baseweb="checkbox"] span { border-color: rgba(237,241,232,.55) !important; }
div.stButton > button[kind="primary"] { background:#a8cf7e; color:#13200f; border:0; border-radius:4px; font-weight:600;
  font-size:1.02rem; padding:.62rem 1rem; box-shadow:none; }
div.stButton > button[kind="primary"]:hover { background:#bcdc96; color:#13200f; }
div.stButton > button[kind="primary"] p { color:#13200f !important; font-weight:600; }

/* segmented radio */
[data-testid="stRadio"] [role="radiogroup"] { display:flex; flex-direction:row; gap:0; }
[data-testid="stRadio"] label { padding:6px 16px; border:1px solid rgba(237,241,232,.30); margin:0 -1px 0 0 !important;
  background:rgba(255,255,255,.04); cursor:pointer; }
[data-testid="stRadio"] label > div:first-child { display:none; }
[data-testid="stRadio"] label p { font-size:.98rem !important; }
[data-testid="stRadio"] label:has(input:checked) { background:#a8cf7e; border-color:#a8cf7e; }
[data-testid="stRadio"] label:has(input:checked) p { color:#13200f !important; font-weight:600; }

/* sidebar */
[data-testid="stSidebar"] { background: rgba(6,13,9,.95) !important; }
[data-testid="stSidebar"] table { width:100%; border-collapse:collapse; font-size:.93rem; }
[data-testid="stSidebar"] th { background:rgba(168,207,126,.18); color:#f3f6ee !important; text-align:left; }
[data-testid="stSidebar"] th, [data-testid="stSidebar"] td { border:1px solid rgba(237,241,232,.20) !important; padding:7px 9px !important; color:#edf1e8 !important; }
</style>
"""
    st.markdown(css.replace("__IMG__", image), unsafe_allow_html=True)


inject_css()


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
        out[tag] = {"value": v, "lo": max(0.0, v - hw), "hi": v + hw, "base": float(card[tag]["stage_mean"][stage])}
    return out


def out_of_range(stage, values, skip):
    return [f for f in SENS_KEYS if f not in skip and (values[f] < card["ranges"][stage][f]["p5"] or values[f] > card["ranges"][stage][f]["p95"])]


def nice_max(*vals):
    top = max(vals)
    step = 50 if top < 600 else 100
    return int(math.ceil(top / step) * step)


# ----------------------------------------------------------------------------
# Charts
# ----------------------------------------------------------------------------
def ring_option(name, d, limit):
    vmax = nice_max(d["hi"] * 1.1, limit * 1.3, d["value"] * 1.2)
    base = {"type": "gauge", "startAngle": 225, "endAngle": -45, "min": 0, "max": vmax, "center": ["50%", "55%"], "splitNumber": 5}
    hide = {"axisTick": {"show": False}, "splitLine": {"show": False}, "axisLabel": {"show": False}, "pointer": {"show": False},
            "anchor": {"show": False}, "detail": {"show": False}, "title": {"show": False}}
    ring = dict(base, radius="92%", **{
        "axisLine": {"lineStyle": {"width": 14, "color": [[1, "rgba(237,241,232,.14)"]]}},
        "progress": {"show": True, "width": 14, "itemStyle": {"color": MOSS}},
        "axisTick": {"show": False}, "splitLine": {"show": False}, "axisLabel": {"show": False}, "pointer": {"show": False},
        "anchor": {"show": False},
        "title": {"offsetCenter": [0, "30%"], "color": MUTED, "fontSize": 13},
        "detail": {"valueAnimation": True, "formatter": "{value}", "fontSize": 40, "fontWeight": 600, "color": "#f3f6ee",
                   "fontFamily": "Barlow Condensed, Arial Narrow, sans-serif", "offsetCenter": [0, "-2%"]},
        "data": [{"value": round(d["value"], 1), "name": "mg/L " + name}]})
    span = dict(base, radius="72%", **hide, axisLine={"lineStyle": {"width": 5, "color": [
        [d["lo"] / vmax, "rgba(0,0,0,0)"], [d["hi"] / vmax, "rgba(237,241,232,.60)"], [1, "rgba(0,0,0,0)"]]}}, data=[{"value": 0}])
    lim = limit / vmax
    mark = dict(base, radius="100%", **hide, axisLine={"lineStyle": {"width": 24, "color": [
        [max(lim - 0.006, 0), "rgba(0,0,0,0)"], [min(lim + 0.006, 1), OCHRE], [1, "rgba(0,0,0,0)"]]}}, data=[{"value": 0}])
    return {"series": [ring, span, mark]}


def range_option(name, d, limit):
    vmax = nice_max(d["hi"] * 1.1, limit * 1.3, d["base"] * 1.2)
    return {
        "grid": {"left": 4, "right": 18, "top": 20, "bottom": 20, "containLabel": True},
        "tooltip": {"trigger": "item"},
        "xAxis": {"type": "value", "min": 0, "max": vmax, "axisLabel": {"color": MUTED},
                  "splitLine": {"lineStyle": {"color": "rgba(237,241,232,.09)"}}},
        "yAxis": {"type": "category", "data": [name], "axisLabel": {"show": False}, "axisTick": {"show": False}, "axisLine": {"show": False}},
        "series": [
            {"type": "bar", "stack": "r", "silent": True, "data": [round(d["lo"], 1)], "itemStyle": {"color": "transparent"}},
            {"type": "bar", "stack": "r", "name": "90% interval", "barWidth": 16, "data": [round(d["hi"] - d["lo"], 1)],
             "itemStyle": {"color": "rgba(168,207,126,.38)", "borderRadius": 2},
             "markLine": {"symbol": "none", "silent": True, "lineStyle": {"color": OCHRE, "type": "solid", "width": 2},
                          "label": {"formatter": "NEQS {c}", "color": OCHRE, "position": "end"}, "data": [{"xAxis": limit}]}},
            {"type": "scatter", "name": "Estimate", "data": [[round(d["value"], 1), name]], "symbolSize": 13, "itemStyle": {"color": MOSS}},
            {"type": "scatter", "name": "Stage average", "data": [[round(d["base"], 1), name]], "symbol": "rect", "symbolSize": [3, 22],
             "itemStyle": {"color": BONE}},
        ]}


def heat_option(matrix, xs, ys):
    flat = [v for row in matrix for v in row if v is not None]
    lo_v, hi_v = min(flat), max(flat)
    span = max(hi_v - lo_v, 1e-9)
    data = [{"value": [j, i, v], "label": {"color": "#10200f" if (v - lo_v) / span > 0.62 else "#f3f6ee"}}
            for i, row in enumerate(matrix) for j, v in enumerate(row) if v is not None]
    return {"tooltip": {"position": "top"},
            "grid": {"left": 4, "right": 8, "top": 4, "bottom": 46, "containLabel": True},
            "xAxis": {"type": "category", "data": xs, "axisLabel": {"color": MUTED}},
            "yAxis": {"type": "category", "data": ys, "axisLabel": {"color": MUTED}},
            "visualMap": {"min": math.floor(lo_v), "max": math.ceil(hi_v), "calculable": False, "orient": "horizontal",
                          "left": "center", "bottom": 0, "itemWidth": 12, "itemHeight": 110, "textStyle": {"color": MUTED},
                          "inRange": {"color": HEAT}},
            "series": [{"type": "heatmap", "data": data, "label": {"show": True, "fontSize": 10, "fontWeight": 600},
                        "itemStyle": {"borderColor": "rgba(8,16,10,.55)", "borderWidth": 1}}]}


def spark_option(xs, ys):
    return {"grid": {"left": 4, "right": 4, "top": 6, "bottom": 2}, "tooltip": {"trigger": "axis"},
            "xAxis": {"type": "category", "show": False, "data": xs}, "yAxis": {"type": "value", "show": False, "scale": True},
            "series": [{"type": "line", "data": ys, "symbol": "circle", "symbolSize": 5,
                        "lineStyle": {"width": 2, "color": MOSS}, "itemStyle": {"color": MOSS},
                        "areaStyle": {"color": MOSS, "opacity": 0.10}}]}


def stats(items):
    return "<div class='stats'>" + "".join(f"<div class='stat'><div class='sv'>{v}</div><div class='sc'>{c}</div></div>" for v, c in items) + "</div>"


# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## NEQS limits")
    st.caption("National Environmental Quality Standards, SRO 549(I)/2000, municipal and liquid industrial effluents "
               "discharged into inland waters. Reference only.")
    st.markdown(
        "| Parameter | NEQS limit |\n|---|---|\n"
        "| pH | 6 to 9 |\n"
        "| Temperature | rise of 3 °C or less |\n"
        "| Dissolved oxygen | no NEQS limit |\n"
        "| Turbidity | no NEQS limit |\n"
        "| Total dissolved solids | 3,500 mg/L |\n"
        "| Electrical conductivity | no NEQS limit |\n"
        "| Total suspended solids | 200 mg/L |\n"
        "| BOD (5-day, 20 °C) | 80 mg/L |\n"
        "| COD | 150 mg/L |")
    st.divider()
    st.markdown("### Reading the charts")
    st.markdown("Green dot or ring: the estimate. Pale band: the 90% interval. White tick: the typical value for the sample point. "
                "Ochre mark: the NEQS reference.")

# ----------------------------------------------------------------------------
# Header + control bar (landscape)
# ----------------------------------------------------------------------------
st.markdown("<div class='top'><div class='t'>BOD / COD Estimator</div>"
            "<div class='s'>Wastewater treatment train, from seven field readings</div></div>", unsafe_allow_html=True)

with st.container(key="card_control"):
    h1, h2 = st.columns([1.1, 2.4])
    with h1:
        st.markdown("<div class='ct'>Sample point</div>", unsafe_allow_html=True)
    with h2:
        st.markdown("<div class='sub' style='text-align:right'>Enter the readings. Tick N/A for any reading you do not have.</div>",
                    unsafe_allow_html=True)
    stage = st.radio("Sample point", STAGE_ORDER, index=2, format_func=lambda s: STAGE_LABELS[s], key="stage",
                     horizontal=True, label_visibility="collapsed")
    cols = st.columns([1, 1, 1, 1.15, 1, 1, 1, 1.35], gap="small")
    vals, unknown = {}, []
    for col, (feat, short, label, lo, hi, step, fmt) in zip(cols[:7], SENSORS):
        typical = float(round(card["ranges"][stage][feat]["p50"], 2))
        na = bool(st.session_state.get(f"na_{feat}_{stage}", False))
        with col:
            vals[feat] = st.number_input(label, min_value=lo, max_value=hi, value=min(max(typical, lo), hi), step=step,
                                         format=fmt, key=f"{feat}_{stage}", disabled=na)
            st.checkbox("N/A", key=f"na_{feat}_{stage}")
        if na:
            unknown.append(feat)
    with cols[7]:
        st.markdown("<div style='height:1.75rem'></div>", unsafe_allow_html=True)
        go = st.button("Predict", type="primary", use_container_width=True)

if go:
    used = dict(vals)
    for f in unknown:
        used[f] = float(card["ranges"][stage][f]["p50"])
    now = datetime.now()
    res = predict(stage, used)
    st.session_state.last = {"stage": stage, "out": res, "unknown": unknown, "ood": out_of_range(stage, used, set(unknown)),
                             "time": now.strftime("%d %b %Y, %H:%M"), "demo": False}
    st.session_state.history.append({"t": now.strftime("%H:%M:%S"), "bod": res["bod"]["value"], "cod": res["cod"]["value"]})
    st.session_state.history = st.session_state.history[-10:]
elif st.session_state.last is None:
    typ = {f: float(card["ranges"]["Effluent"][f]["p50"]) for f in SENS_KEYS}
    st.session_state.last = {"stage": "Effluent", "out": predict("Effluent", typ), "unknown": [], "ood": [],
                             "time": "typical readings", "demo": True}

last = st.session_state.last
out = last["out"]
notes = []
if last["demo"]:
    notes.append("Showing typical effluent readings. Enter your own values and press Predict.")
if last["unknown"]:
    notes.append("Typical values used for <b>" + ", ".join(SHORT[f] for f in last["unknown"]) + "</b>.")
if last["ood"]:
    notes.append("Outside the usual range for this sample point: <b>" + ", ".join(SHORT[f] for f in last["ood"]) + "</b>.")
st.markdown(f"<div class='note'>{' &nbsp;·&nbsp; '.join(notes) if notes else '&nbsp;'}</div>", unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# Row 2: BOD ring, COD ring, range comparison
# ----------------------------------------------------------------------------
c_bod, c_cod, c_rng = st.columns([1, 1, 1.25], gap="small")
for col, tag, name in ((c_bod, "bod", "BOD"), (c_cod, "cod", "COD")):
    d = out[tag]
    with col:
        with st.container(key=f"card_{tag}"):
            st.markdown(f"<div class='ct'>{name}<span class='u'>mg/L · {STAGE_LABELS[last['stage']].lower()} · {last['time']}</span></div>",
                        unsafe_allow_html=True)
            st_echarts(ring_option(name, d, NEQS[tag]), height="235px", key=f"ring_{tag}_{last['time']}")
            st.markdown(stats([(f"{d['lo']:.0f}-{d['hi']:.0f}", "90% interval"), (f"{d['base']:.0f}", "typical for point"),
                               (f"{NEQS[tag]}", "NEQS reference")]), unsafe_allow_html=True)
with c_rng:
    with st.container(key="card_range"):
        st.markdown("<div class='ct'>Estimate against the typical value</div>"
                    f"<div class='sub'><span style='color:{MOSS}'>●</span> estimate &nbsp; "
                    f"<span style='color:{BONE}'>|</span> typical &nbsp; <span style='color:{MOSS};opacity:.55'>▬</span> 90% interval &nbsp; "
                    f"<span style='color:{OCHRE}'>|</span> NEQS</div>", unsafe_allow_html=True)
        st.caption("BOD (mg/L)")
        st_echarts(range_option("BOD", out["bod"], NEQS["bod"]), height="108px", key=f"rng_bod_{last['time']}")
        st.caption("COD (mg/L)")
        st_echarts(range_option("COD", out["cod"], NEQS["cod"]), height="108px", key=f"rng_cod_{last['time']}")

# ----------------------------------------------------------------------------
# Row 3: recorded patterns + accuracy + session
# ----------------------------------------------------------------------------
st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
c_heat, c_side = st.columns([3.2, 1.2], gap="small")
with c_heat:
    with st.container(key="card_heat"):
        a, b, c = st.columns([2.2, 1.1, 1.3])
        with a:
            st.markdown("<div class='ct'>Recorded patterns</div><div class='sub'>Mean of recorded values by stage, month, shift and weekday</div>",
                        unsafe_allow_html=True)
        with b:
            hp = st.radio("Parameter", ["BOD", "COD"], horizontal=True, key="hp", label_visibility="collapsed")
        with c:
            hs = st.selectbox("Weekday view", STAGE_ORDER, index=2, format_func=lambda s: STAGE_LABELS[s], key="hs",
                              label_visibility="collapsed")
        m1, m2 = st.columns(2)
        with m1:
            st.caption(f"{hp} by stage and month (mg/L)")
            st_echarts(heat_option(heat["month_stage"][hp], heat["months"], heat["stages"]), height="250px", key=f"h1_{hp}")
        with m2:
            st.caption(f"{hp} by shift and weekday, {STAGE_LABELS[hs].lower()} (mg/L)")
            st_echarts(heat_option(heat["weekday_shift"][hp][hs], heat["weekdays"], heat["shifts"]), height="250px", key=f"h2_{hp}_{hs}")
with c_side:
    with st.container(key="card_acc"):
        st.markdown("<div class='ct'>Accuracy</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='tiles'><div class='tile'><div class='tl'>BOD R²</div><div class='tv'>{R2['BOD']:.2f}</div></div>"
                    f"<div class='tile'><div class='tl'>COD R²</div><div class='tv'>{R2['COD']:.2f}</div></div></div>"
                    f"<div class='sub'>Typical error is about {ERR['BOD']:.0f}% for BOD and {ERR['COD']:.0f}% for COD.</div>",
                    unsafe_allow_html=True)
    with st.container(key="card_session"):
        st.markdown("<div class='ct'>This session</div>", unsafe_allow_html=True)
        hist = st.session_state.history
        if len(hist) < 2:
            st.caption("Make two or more predictions to see the trend.")
        else:
            xs = [h["t"] for h in hist]
            st.caption("BOD (mg/L)")
            st_echarts(spark_option(xs, [round(h["bod"], 1) for h in hist]), height="62px", key="sp_bod")
            st.caption("COD (mg/L)")
            st_echarts(spark_option(xs, [round(h["cod"], 1) for h in hist]), height="62px", key="sp_cod")

st.markdown("<div class='foot'>Research prototype, FYDP, IIUI 2022-2026. Estimates are not laboratory results; confirm with laboratory analysis before any reporting.</div>",
            unsafe_allow_html=True)
