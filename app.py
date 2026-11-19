import base64
import json
import math
import os
from datetime import datetime

import joblib
import pandas as pd
import streamlit as st

BASE = os.path.dirname(os.path.abspath(__file__))

st.set_page_config(page_title="BOD / COD Estimator", layout="wide", initial_sidebar_state="expanded")

# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------
NEQS = {"bod": 80, "cod": 150}                      # reference marks only
MOSS, OCHRE, BONE, MUTED = "#a8cf7e", "#e3ad45", "#edf1e8", "#a3b3a0"
HEAT_STOPS = [(0.00, (22, 48, 31)), (0.25, (47, 90, 54)), (0.50, (95, 143, 78)), (0.75, (168, 207, 126)), (1.00, (237, 231, 181))]

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
# Assets (cached: loaded once per server process)
# ----------------------------------------------------------------------------
@st.cache_resource
def load_assets():
    with open(os.path.join(BASE, "model_card.json")) as f:
        card = json.load(f)
    with open(os.path.join(BASE, "heatmap_data.json")) as f:
        heat = json.load(f)
    return card, heat, joblib.load(os.path.join(BASE, "bod_model.joblib")), joblib.load(os.path.join(BASE, "cod_model.joblib"))


@st.cache_resource
def background_css_image():
    bg = os.path.join(BASE, "bg.jpg")
    if os.path.exists(bg):
        with open(bg, "rb") as f:
            return "url(data:image/jpeg;base64," + base64.b64encode(f.read()).decode() + ")"
    return "linear-gradient(135deg,#0b1f1a,#10261f 50%,#06120e)"


card, heat, bod_model, cod_model = load_assets()

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
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow:wght@400;500;600&family=Barlow+Condensed:wght@500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');
html, body, .stApp, .stApp p, .stApp label, .stApp span, .stApp div { font-family: 'Barlow', 'Segoe UI', Arial, sans-serif; }
.stApp { background: linear-gradient(rgba(5,12,8,.40), rgba(5,12,8,.66)), __IMG__; background-size: cover;
  background-position: center; background-attachment: fixed; }
.stApp, .stApp p, .stApp li, .stApp label, .stApp span, .stApp h1, .stApp h2, .stApp h3, .stApp h4 { color: #edf1e8; }
header[data-testid="stHeader"] { background: transparent; }
.block-container { padding: 1.1rem 1.6rem 1.4rem 1.6rem; max-width: 1760px; }

/* glass panels: generous padding so nothing touches an edge */
div[class*="st-key-card"] { background: rgba(14,24,18,.55); backdrop-filter: blur(16px) saturate(1.1);
  -webkit-backdrop-filter: blur(16px) saturate(1.1); border: 1px solid rgba(237,241,232,.18); border-radius: 8px;
  padding: 20px 22px 30px 22px; box-sizing: border-box; min-width: 0; }
div[class*="st-key-card"] * { min-width: 0; }
div[class*="st-key-card"] .element-container, div[class*="st-key-card"] [data-testid="element-container"],
div[class*="st-key-card"] [data-testid="stElementContainer"], div[class*="st-key-card"] [data-testid="stHorizontalBlock"],
div[class*="st-key-card"] [data-testid="stVerticalBlock"], div[class*="st-key-card"] [data-testid="stMarkdown"],
div[class*="st-key-card"] [data-testid="stMarkdownContainer"], div[class*="st-key-card"] [data-testid="stCaptionContainer"],
div[class*="st-key-card"] [data-testid="stRadio"], div[class*="st-key-card"] [data-testid="stSelectbox"],
div[class*="st-key-card"] [data-testid="stNumberInput"], div[class*="st-key-card"] [data-testid="stCheckbox"],
div[class*="st-key-card"] [data-testid="stButton"], div[class*="st-key-card"] [data-testid="stWidgetLabel"] { width: 100% !important; max-width: 100% !important; box-sizing: border-box; }
div[class*="st-key-card"] [data-testid="stMarkdownContainer"] { margin-bottom: 0 !important; }
div[class*="st-key-card"] p, div[class*="st-key-card"] .sub, div[class*="st-key-card"] .sc { overflow-wrap: anywhere; }

/* type */
.top { display:flex; flex-wrap:wrap; gap:4px 18px; align-items:baseline; justify-content:space-between; margin: 2px 2px 12px 2px; }
.top .t { font-family:'Barlow Condensed','Arial Narrow','Roboto Condensed',sans-serif; font-weight:700; font-size:2.35rem; letter-spacing:.01em; color:#f3f6ee; line-height:1; }
.top .s { font-size:1rem; color:#b5c4b0; padding-right:130px; }
.ct { font-family:'Barlow Condensed','Arial Narrow','Roboto Condensed',sans-serif; font-weight:600; font-size:1.35rem; color:#f3f6ee; line-height:1.15; overflow-wrap:anywhere; }
.ct .u { font-family:'Barlow',sans-serif; font-weight:400; font-size:.9rem; color:#a3b3a0; margin-left:6px; }
.sub { font-size:.88rem; color:#a3b3a0; margin:2px 0 6px 0; overflow-wrap:anywhere; }
.stats { display:flex; flex-wrap:wrap; gap:8px 18px; margin-top:6px; }
.stat { flex:1 1 auto; min-width:max-content !important; }
.sv { font-family:'Barlow Condensed','Arial Narrow','Roboto Condensed',sans-serif; font-weight:600; font-size:1.4rem; line-height:1.05; white-space:nowrap; font-variant-numeric: tabular-nums; color:#f3f6ee; }
.sc { font-size:.8rem; color:#a3b3a0; white-space:nowrap; }
.tiles { display:grid; grid-template-columns: repeat(2, minmax(0,1fr)); gap:10px; margin: 8px 0 10px 0; }
.tile { min-width:0; border:1px solid rgba(237,241,232,.16); border-radius:6px; padding:9px 11px; background:rgba(255,255,255,.04); }
.tile .tl { font-size:.82rem; color:#a3b3a0; white-space:nowrap; }
.tile .tv { font-family:'Barlow Condensed','Arial Narrow','Roboto Condensed',sans-serif; font-weight:600; font-size:1.9rem; line-height:1.1; color:#f3f6ee; white-space:nowrap; }
.note { font-size:.9rem; color:#c9d6c4; margin: 2px 2px 12px 2px; min-height:1.1rem; overflow-wrap:anywhere; }
.note b { color:#e3ad45; font-weight:600; }
.foot { font-size:.8rem; color:#8fa08c; margin-top:8px; overflow-wrap:anywhere; }

/* drawings */
svg.dw { display:block; width:100%; height:auto; overflow:visible; }
svg.dw text { font-family:'Barlow',sans-serif; fill:#a3b3a0; }
svg.dw .big { font-family:'Barlow Condensed','Arial Narrow',sans-serif; font-weight:600; fill:#f3f6ee; }
.ringwrap { max-width:300px; margin:0 auto; }
.hmgrid { display:grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap:14px 32px; margin-top:4px; }
.hmgrid > div { min-width:0; }
table.hm { width:100%; table-layout:fixed; border-collapse:separate; border-spacing:2px; font-size:.72rem; }
table.hm th { font-weight:500; color:#a3b3a0; text-align:center; padding:2px 0; font-size:.72rem; }
table.hm th.rh { text-align:right; width:82px; padding-right:6px; white-space:nowrap; }
table.hm td { text-align:center; padding:9px 0; border-radius:2px; font-family:'IBM Plex Mono',monospace; font-weight:500; overflow:hidden; }
.hmleg { display:flex; align-items:center; gap:8px; margin-top:8px; font-size:.78rem; color:#a3b3a0; }
.hmleg .bar { flex:1 1 auto; height:8px; border-radius:2px; max-width:240px; }

/* form controls */
[data-testid="stWidgetLabel"] { min-height:2.3em; align-items:flex-end; }
[data-testid="stWidgetLabel"] p, [data-testid="stCheckbox"] label p { color:#dfe8da !important; font-size:.84rem !important; line-height:1.15 !important; font-weight:500; overflow-wrap:anywhere; }
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p { color:#a3b3a0 !important; font-size:.84rem; }
[data-baseweb="input"], [data-baseweb="base-input"] { background-color: rgba(255,255,255,.07) !important; border-color: rgba(237,241,232,.26) !important; border-radius: 5px !important; }
[data-baseweb="input"] input, [data-baseweb="base-input"] input { color:#f6f9f2 !important; -webkit-text-fill-color:#f6f9f2 !important; font-family:'IBM Plex Mono', monospace !important; font-size:1rem; padding-right:6px; }
[data-baseweb="input"] input:disabled { color:#7f917d !important; -webkit-text-fill-color:#7f917d !important; }
[data-testid="stNumberInputStepUp"], [data-testid="stNumberInputStepDown"] { display:none !important; }
[data-baseweb="select"] > div { background-color: rgba(255,255,255,.07) !important; border-color: rgba(237,241,232,.26) !important; }
[data-baseweb="select"] div, [data-baseweb="select"] span, [data-baseweb="select"] svg { color:#f6f9f2 !important; fill:#f6f9f2 !important; }
ul[role="listbox"] { background:#101c14 !important; } ul[role="listbox"] li, ul[role="listbox"] li * { color:#edf1e8 !important; }
[data-baseweb="checkbox"] span { border-color: rgba(237,241,232,.55) !important; }
div.stButton > button[kind="primary"] { background:#a8cf7e; color:#13200f; border:0; border-radius:4px; font-weight:600; font-size:1.02rem; padding:.62rem 1rem; box-shadow:none; }
div.stButton > button[kind="primary"]:hover { background:#bcdc96; color:#13200f; }
div.stButton > button[kind="primary"] p { color:#13200f !important; font-weight:600; }

/* segmented radio */
[data-testid="stRadio"] [role="radiogroup"] { display:flex; flex-direction:row; gap:0; }
[data-testid="stRadio"] label { padding:6px 16px; border:1px solid rgba(237,241,232,.30); margin:0 -1px 0 0 !important; background:rgba(255,255,255,.04); cursor:pointer; }
[data-testid="stRadio"] label > div:first-child { display:none; }
[data-testid="stRadio"] label p { font-size:.98rem !important; }
[data-testid="stRadio"] [data-testid="stWidgetLabel"], [data-testid="stSelectbox"] [data-testid="stWidgetLabel"] { min-height:0; }
[data-testid="stRadio"] label:has(input:checked) { background:#a8cf7e; border-color:#a8cf7e; }
[data-testid="stRadio"] label:has(input:checked) p { color:#13200f !important; font-weight:600; }

/* sidebar */
[data-testid="stSidebar"] { background: rgba(6,13,9,.95) !important; }
[data-testid="stSidebar"] table { width:100%; border-collapse:collapse; font-size:.93rem; }
[data-testid="stSidebar"] th { background:rgba(168,207,126,.18); color:#f3f6ee !important; text-align:left; }
[data-testid="stSidebar"] th, [data-testid="stSidebar"] td { border:1px solid rgba(237,241,232,.20) !important; padding:7px 9px !important; color:#edf1e8 !important; }
.key { margin-top:8px; }
.key .k { display:flex; gap:12px; align-items:flex-start; margin-bottom:18px; font-size:.95rem; line-height:1.35; color:#dfe8da; }
.key .k b { color:#f3f6ee; font-weight:600; }
.key .symw { flex:0 0 38px; display:flex; justify-content:center; align-items:flex-start; padding-top:2px; }
.key .sym { display:block; flex:none; }
.key .sym.dot { width:18px; height:18px; border-radius:50%; background:#a8cf7e; margin-top:2px; }
.key .sym.band { width:36px; height:16px; border-radius:3px; background:rgba(168,207,126,.55); margin-top:3px; }
.key .sym.tick { width:8px; height:28px; border-radius:1px; }
.key .sym.tick.white { background:#edf1e8; } .key .sym.tick.ochre { background:#e3ad45; }
</style>
"""
st.markdown(CSS.replace("__IMG__", background_css_image()), unsafe_allow_html=True)


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
        rel = card[tag].get("interval_rel_90")
        hw = rel[stage] * max(v, 1.0) if rel else float(card[tag]["interval_half_width_90"][stage])
        out[tag] = {"value": v, "lo": max(0.0, v - hw), "hi": v + hw, "base": float(card[tag]["stage_mean"][stage])}
    return out


def out_of_range(stage, values, skip):
    out = []
    for f in SENS_KEYS:
        if f in skip:
            continue
        r = card["ranges"][stage][f]
        if values[f] < r.get("p1", r["p5"]) or values[f] > r.get("p99", r["p95"]):
            out.append(f)
    return out


def nice_max(*vals):
    top = max(vals)
    step = 50 if top < 300 else 100
    return int(math.ceil(top / step) * step)


def tick_step(vmax):
    for c in (10, 20, 25, 50, 100, 200, 250, 500):
        if vmax / c <= 6:
            return c
    return vmax


# ----------------------------------------------------------------------------
# Drawings (inline SVG / HTML: no scripts, no iframes)
# ----------------------------------------------------------------------------
def _pol(cx, cy, r, deg):
    a = math.radians(deg)
    return cx + r * math.cos(a), cy + r * math.sin(a)


def _arc(cx, cy, r, v0, v1, vmax):
    a0 = 135 + 270 * max(0.0, min(v0, vmax)) / vmax
    a1 = 135 + 270 * max(0.0, min(v1, vmax)) / vmax
    if a1 - a0 < 0.05:
        a1 = a0 + 0.05
    x0, y0 = _pol(cx, cy, r, a0)
    x1, y1 = _pol(cx, cy, r, a1)
    return f"M {x0:.2f} {y0:.2f} A {r} {r} 0 {1 if (a1 - a0) > 180 else 0} 1 {x1:.2f} {y1:.2f}"


def ring_svg(name, d, limit):
    vmax = nice_max(d["hi"] * 1.1, limit * 1.3, d["value"] * 1.2)
    cx, cy, R = 110, 108, 82
    lx0, ly0 = _pol(cx, cy, R - 13, 135 + 270 * limit / vmax)
    lx1, ly1 = _pol(cx, cy, R + 13, 135 + 270 * limit / vmax)
    ex0, ey0 = _pol(cx, cy, R, 135)
    ex1, ey1 = _pol(cx, cy, R, 45)
    return (
        f"<div class='ringwrap'><svg class='dw' viewBox='0 0 220 200' role='img' aria-label='{name} estimate {d['value']:.1f} mg/L'>"
        f"<path d='{_arc(cx, cy, R, 0, vmax, vmax)}' fill='none' stroke='rgba(237,241,232,.14)' stroke-width='15'/>"
        f"<path d='{_arc(cx, cy, R, 0, d['value'], vmax)}' fill='none' stroke='{MOSS}' stroke-width='15'/>"
        f"<path d='{_arc(cx, cy, R - 21, d['lo'], d['hi'], vmax)}' fill='none' stroke='rgba(237,241,232,.65)' stroke-width='4'/>"
        f"<line x1='{lx0:.1f}' y1='{ly0:.1f}' x2='{lx1:.1f}' y2='{ly1:.1f}' stroke='{OCHRE}' stroke-width='4'/>"
        f"<text x='{cx}' y='{cy + 8}' text-anchor='middle' class='big' style='font-size:{38 if len(f"{d['value']:.1f}") <= 5 else 32}px'>{d['value']:.1f}</text>"
        f"<text x='{cx}' y='{cy + 30}' text-anchor='middle' style='font-size:12.5px'>mg/L {name}</text>"
        f"<text x='{ex0 + 2:.1f}' y='{ey0 + 20:.1f}' text-anchor='middle' style='font-size:11px'>0</text>"
        f"<text x='{ex1 - 2:.1f}' y='{ey1 + 20:.1f}' text-anchor='middle' style='font-size:11px'>{vmax}</text>"
        f"</svg></div>")


def range_svg(d, limit):
    vmax = nice_max(d["hi"] * 1.1, limit * 1.3, d["base"] * 1.2)
    x0, x1 = 16, 384
    X = lambda v: x0 + (x1 - x0) * max(0.0, min(v, vmax)) / vmax
    step = tick_step(vmax)
    grid, labels = "", ""
    for t in range(0, int(vmax) + 1, step):
        x = X(t)
        grid += f"<line x1='{x:.1f}' y1='34' x2='{x:.1f}' y2='80' stroke='rgba(237,241,232,.10)'/>"
        labels += f"<text x='{x:.1f}' y='98' text-anchor='middle' style='font-size:11px'>{t}</text>"
    xl = X(limit)
    anchor = "end" if xl > 300 else "middle"
    return (
        "<svg class='dw' viewBox='0 0 400 108' role='img'>"
        + grid
        + f"<rect x='{X(d['lo']):.1f}' y='48' width='{max(X(d['hi']) - X(d['lo']), 2):.1f}' height='18' rx='2' fill='rgba(168,207,126,.42)'/>"
        + f"<rect x='{X(d['base']) - 1.5:.1f}' y='42' width='3' height='30' fill='{BONE}'/>"
        + f"<circle cx='{X(d['value']):.1f}' cy='57' r='7.5' fill='{MOSS}'/>"
        + f"<rect x='{xl - 1.5:.1f}' y='30' width='3' height='46' fill='{OCHRE}'/>"
        + f"<text x='{xl:.1f}' y='22' text-anchor='{anchor}' style='font-size:11.5px;fill:{OCHRE}'>NEQS {limit}</text>"
        + labels + "</svg>")


def _heat_color(frac):
    frac = max(0.0, min(1.0, frac))
    for (f0, c0), (f1, c1) in zip(HEAT_STOPS, HEAT_STOPS[1:]):
        if frac <= f1:
            t = (frac - f0) / (f1 - f0)
            return tuple(round(a + (b - a) * t) for a, b in zip(c0, c1))
    return HEAT_STOPS[-1][1]


def heat_table(matrix, xs, ys):
    flat = [v for row in matrix for v in row if v is not None]
    lo_v, hi_v = min(flat), max(flat)
    span = max(hi_v - lo_v, 1e-9)
    head = "<tr><th class='rh'></th>" + "".join(f"<th>{x}</th>" for x in xs) + "</tr>"
    body = ""
    for y, row in zip(ys, matrix):
        cells = ""
        for v in row:
            if v is None:
                cells += "<td style='background:rgba(255,255,255,.04)'>-</td>"
            else:
                fr = (v - lo_v) / span
                r, g, b = _heat_color(fr)
                txt = "#10200f" if fr > 0.62 else "#f3f6ee"
                cells += f"<td title='{v:.1f} mg/L' style='background:rgb({r},{g},{b});color:{txt}'>{v:.0f}</td>"
        body += f"<tr><th class='rh'>{y}</th>{cells}</tr>"
    grad = ",".join(f"rgb({c[0]},{c[1]},{c[2]}) {int(f * 100)}%" for f, c in HEAT_STOPS)
    legend = f"<div class='hmleg'><span>{lo_v:.0f}</span><div class='bar' style='background:linear-gradient(90deg,{grad})'></div><span>{hi_v:.0f} mg/L</span></div>"
    return f"<table class='hm'>{head}{body}</table>{legend}"


def spark_svg(ys):
    lo_v, hi_v = min(ys), max(ys)
    span = max(hi_v - lo_v, 1e-9)
    n = len(ys)
    pts = [(8 + 184 * i / (n - 1), 40 - 32 * (y - lo_v) / span) for i, y in enumerate(ys)]
    line = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
    area = f"8,46 {line} 192,46"
    lx, ly = pts[-1]
    return (f"<svg class='dw' viewBox='0 0 200 52' role='img'><polygon points='{area}' fill='rgba(168,207,126,.12)'/>"
            f"<polyline points='{line}' fill='none' stroke='{MOSS}' stroke-width='2'/><circle cx='{lx:.1f}' cy='{ly:.1f}' r='3.5' fill='{MOSS}'/></svg>")


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
    st.markdown("## Chart key")
    st.markdown(
        "<div class='key'>"
        "<div class='k'><span class='symw'><span class='sym dot'></span></span><div><b>Estimate</b><br>The model's best value for your sample.</div></div>"
        "<div class='k'><span class='symw'><span class='sym band'></span></span><div><b>Likely range</b><br>The true value falls inside this band about 9 times out of 10 (90% interval).</div></div>"
        "<div class='k'><span class='symw'><span class='sym tick white'></span></span><div><b>Typical</b><br>The usual value at this sample point, for comparison.</div></div>"
        "<div class='k'><span class='symw'><span class='sym tick ochre'></span></span><div><b>NEQS</b><br>The national discharge limit, shown for reference only.</div></div>"
        "</div>", unsafe_allow_html=True)

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
            st.markdown(f"<div class='ct'>{name}<span class='u'>mg/L · {STAGE_LABELS[last['stage']].lower()} · {last['time']}</span></div>"
                        + ring_svg(name, d, NEQS[tag])
                        + stats([(f"{d['lo']:.0f}-{d['hi']:.0f}", "90% interval"), (f"{d['base']:.0f}", "typical"), (f"{NEQS[tag]}", "NEQS")]),
                        unsafe_allow_html=True)
with c_rng:
    with st.container(key="card_range"):
        st.markdown("<div class='ct'>Estimate against the typical value</div>"
                    f"<div class='sub'><span style='color:{MOSS}'>●</span> estimate &nbsp; "
                    f"<span style='color:{BONE}'>|</span> typical &nbsp; <span style='color:{MOSS};opacity:.6'>▬</span> 90% interval &nbsp; "
                    f"<span style='color:{OCHRE}'>|</span> NEQS</div>"
                    "<div class='sub' style='margin-top:10px'>BOD (mg/L)</div>" + range_svg(out["bod"], NEQS["bod"])
                    + "<div class='sub' style='margin-top:14px'>COD (mg/L)</div>" + range_svg(out["cod"], NEQS["cod"]),
                    unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# Row 3: recorded patterns (own fragment: switching views does not rerun the page) + accuracy + session
# ----------------------------------------------------------------------------
st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
c_heat, c_side = st.columns([3.2, 1.2], gap="small")


@st.fragment
def patterns_card():
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
        cap1 = f"<div class='sub' style='margin-top:10px'>{hp} by stage and month (mg/L)</div>"
        cap2 = f"<div class='sub' style='margin-top:10px'>{hp} by shift and weekday, {STAGE_LABELS[hs].lower()} (mg/L)</div>"
        t1 = heat_table(heat["month_stage"][hp], heat["months"], [STAGE_LABELS[s] for s in heat["stages"]])
        t2 = heat_table(heat["weekday_shift"][hp][hs], heat["weekdays"], heat["shifts"])
        st.markdown(f"<div class='hmgrid'><div>{cap1}{t1}</div><div>{cap2}{t2}</div></div>", unsafe_allow_html=True)


with c_heat:
    patterns_card()
with c_side:
    with st.container(key="card_acc"):
        st.markdown("<div class='ct'>Accuracy</div>"
                    f"<div class='tiles'><div class='tile'><div class='tl'>BOD R²</div><div class='tv'>{R2['BOD']:.2f}</div></div>"
                    f"<div class='tile'><div class='tl'>COD R²</div><div class='tv'>{R2['COD']:.2f}</div></div></div>"
                    f"<div class='sub'>Typical error is about {ERR['BOD']:.0f}% for BOD and {ERR['COD']:.0f}% for COD.</div>",
                    unsafe_allow_html=True)
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    with st.container(key="card_session"):
        hist = st.session_state.history
        body = "<div class='ct'>This session</div>"
        if len(hist) < 2:
            body += "<div class='sub'>Make two or more predictions to see the trend.</div>"
        else:
            body += ("<div class='sub' style='margin-top:8px'>BOD (mg/L)</div>" + spark_svg([h["bod"] for h in hist])
                     + "<div class='sub' style='margin-top:8px'>COD (mg/L)</div>" + spark_svg([h["cod"] for h in hist]))
        st.markdown(body, unsafe_allow_html=True)

st.markdown("<div class='foot'>Research prototype, FYDP, IIUI 2022-2026. Estimates are not laboratory results; confirm with laboratory analysis before any reporting.</div>",
            unsafe_allow_html=True)
