
import streamlit as st
import numpy as np
import pandas as pd
import pickle
import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
from datetime import datetime

st.set_page_config(
    page_title="WWTP BOD/COD Predictor",
    page_icon="W",
    layout="wide",
    initial_sidebar_state="expanded"
)

try:
    from streamlit_echarts import st_echarts
    ECHARTS_AVAILABLE = True
except ImportError:
    ECHARTS_AVAILABLE = False

@st.cache_resource
def load_assets():
    base = os.path.dirname(__file__)
    with open(os.path.join(base, "final_model.pkl"), "rb") as f:
        model = pickle.load(f)
    with open(os.path.join(base, "scaler_final.pkl"), "rb") as f:
        scaler = pickle.load(f)
    with open(os.path.join(base, "bootstrap_ensemble.pkl"), "rb") as f:
        boot = pickle.load(f)
    with open(os.path.join(base, "shap_weights.json")) as f:
        shap_w = json.load(f)
    with open(os.path.join(base, "feature_stats.json")) as f:
        feat_stats = json.load(f)
    with open(os.path.join(base, "compliance_info.json")) as f:
        comp_info = json.load(f)
    return model, scaler, boot, shap_w, feat_stats, comp_info

model, scaler, boot_data, shap_weights, feat_stats, comp_info = load_assets()

FEATURE_COLS = comp_info["feature_cols"]
NEQS_BOD     = comp_info["NEQS_BOD"]
NEQS_COD     = comp_info["NEQS_COD"]

LANG = {
    "en": {
        "title": "Wastewater BOD & COD Predictor",
        "sub": "Enter surrogate field measurements to predict BOD and COD with NEQS 2000 compliance status",
        "field_section": "Field Measurements",
        "loc_label": "Sampling location (optional)",
        "loc_help": "Select for NEQS compliance guidance",
        "season_label": "Seasonal condition",
        "rain_label": "Rain event",
        "surrogates_header": "Surrogate parameters — enter all available values",
        "unknown": "Unknown",
        "predict_btn": "Predict BOD and COD",
        "completeness": "Data completeness",
        "feature_tab": "Feature Contribution",
        "percentile_tab": "Percentile Context",
        "cost_tab": "Cost-Benefit",
        "batch_tab": "Batch Mode",
        "trend_header": "Trend Tracker",
        "how_to": "How to use this tool",
        "disc": "Trained on CDA WWTP I-9/1, Islamabad (2009-2010 and 2025). Validate before applying to other facilities.",
        "disc_full": "Research prototype. FYDP, IIUI 2022-2026. Confirmatory laboratory analysis required for official NEQS compliance reporting.",
    },
    "ur": {
        "title": "فضلہ پانی BOD اور COD پیش گو",
        "sub": "BOD اور COD کی پیش گوئی کے لیے میدانی پیمائش درج کریں",
        "field_section": "میدانی پیمائش",
        "loc_label": "نمونہ لینے کی جگہ (اختیاری)",
        "loc_help": "NEQS رہنمائی کے لیے منتخب کریں",
        "season_label": "موسمی حالت",
        "rain_label": "بارش کا واقعہ",
        "surrogates_header": "معاون پیرامیٹر — تمام دستیاب قدریں درج کریں",
        "unknown": "معلوم نہیں",
        "predict_btn": "BOD اور COD پیش گو کریں",
        "completeness": "ڈیٹا مکمل ہونا",
        "feature_tab": "خصوصیت کا حصہ",
        "percentile_tab": "پرسینٹائل سیاق",
        "cost_tab": "لاگت فائدہ",
        "batch_tab": "بیچ موڈ",
        "trend_header": "رجحان ٹریکر",
        "how_to": "یہ ٹول کیسے استعمال کریں",
        "disc": "CDA WWTP I-9/1 اسلام آباد کے ڈیٹا پر تربیت یافتہ۔",
        "disc_full": "تحقیقی نمونہ۔ FYDP، IIUI 2022-2026۔",
    }
}

if "lang" not in st.session_state:
    st.session_state.lang = "en"
if "pred_history" not in st.session_state:
    st.session_state.pred_history = []
if "last_pred" not in st.session_state:
    st.session_state.last_pred = None

L = LANG[st.session_state.lang]

# ── predict_with_interval -- RAW input, no pre-scaling ───────────
def predict_with_interval(input_raw, confidence=0.95):
    boot_preds = []
    for m, sc in zip(boot_data["models"], boot_data["scalers"]):
        x_filled = np.where(np.isnan(input_raw), 0.0, input_raw)
        x_sc     = sc.transform(x_filled)
        x_sc[np.isnan(input_raw)] = np.nan
        boot_preds.append(m.predict(x_sc)[0])
    boot_preds = np.array(boot_preds)
    alpha  = (1 - confidence) / 2
    return (boot_preds.mean(axis=0),
            np.percentile(boot_preds, alpha * 100, axis=0),
            np.percentile(boot_preds, (1-alpha) * 100, axis=0))

def completeness_score(missing_features, target="bod"):
    weights = shap_weights[f"{target}_weights"]
    return max(0.0, 1.0 - sum(weights.get(f, 0) for f in missing_features))

def check_ood(feature_name, value):
    s = feat_stats.get(feature_name, {})
    return bool(s and (value < s["p5"] or value > s["p95"]))

def neqs_status(pred, lower, upper, limit):
    if pred <= limit:           return "COMPLIANT",     "normal"
    elif lower <= limit:        return "BORDERLINE",    "off"
    else:                       return "NON-COMPLIANT", "inverse"

# ── Gauge chart using matplotlib (fallback and primary) ──────────
def draw_gauge(value, limit, label, unit="mg/L"):
    fig, ax = plt.subplots(figsize=(4, 2.4),
                           subplot_kw=dict(aspect="equal"))
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")

    max_val   = max(limit * 2.5, value * 1.2)
    angle_val = 180 - (min(value, max_val) / max_val) * 180

    # Background arc sections
    from matplotlib.patches import Wedge
    # Green zone 0 to limit*0.75
    g_end = (limit * 0.75 / max_val) * 180
    ax.add_patch(Wedge((0.5, 0), 0.45, 180, 180-g_end,
                        width=0.12, color="#10b981", alpha=0.85))
    # Amber zone limit*0.75 to limit
    a_end = (limit / max_val) * 180
    ax.add_patch(Wedge((0.5, 0), 0.45, 180-g_end, 180-a_end,
                        width=0.12, color="#f59e0b", alpha=0.85))
    # Red zone limit to max
    ax.add_patch(Wedge((0.5, 0), 0.45, 180-a_end, 0,
                        width=0.12, color="#ef4444", alpha=0.85))

    # Needle
    import math
    rad = math.radians(angle_val)
    nx  = 0.5 + 0.35 * math.cos(rad)
    ny  = 0.0 + 0.35 * math.sin(rad)
    ax.annotate("", xy=(nx, ny), xytext=(0.5, 0),
                arrowprops=dict(arrowstyle="-|>", color="white",
                                lw=2, mutation_scale=12))
    ax.plot(0.5, 0, "o", color="white", markersize=5, zorder=5)

    # Value text
    colour = ("#10b981" if value <= limit * 0.75
              else "#f59e0b" if value <= limit
              else "#ef4444")
    ax.text(0.5, 0.22, f"{value:.1f}", ha="center", va="center",
            fontsize=16, fontweight="bold", color=colour,
            transform=ax.transAxes)
    ax.text(0.5, 0.08, unit, ha="center", va="center",
            fontsize=9, color="#9ca3af", transform=ax.transAxes)
    ax.text(0.5, -0.08, label, ha="center", va="center",
            fontsize=10, fontweight="bold", color="white",
            transform=ax.transAxes)
    ax.text(0.5, -0.22,
            f"NEQS limit: {limit} {unit}",
            ha="center", va="center",
            fontsize=8, color="#9ca3af", transform=ax.transAxes)

    ax.set_xlim(0, 1)
    ax.set_ylim(-0.35, 0.55)
    ax.axis("off")
    plt.tight_layout(pad=0)
    return fig

# ═══════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## WWTP Predictor")
    st.caption("Surrogate Parameter Decision Support Tool")

    lang_choice = st.radio("Language / زبان",
                           options=["English", "اردو"], horizontal=True)
    st.session_state.lang = "ur" if lang_choice == "اردو" else "en"
    L = LANG[st.session_state.lang]

    st.divider()
    st.markdown("### Settings")
    confidence   = st.slider("Prediction interval confidence",
                             min_value=0.80, max_value=0.99,
                             value=0.95, step=0.01)
    show_feature = st.checkbox("Show feature contribution", value=True)
    show_cost    = st.checkbox("Show cost-benefit estimate", value=True)

    st.divider()
    st.markdown("### NEQS 2000 Discharge Limits")
    st.markdown("""
| Parameter | Limit |
|-----------|-------|
| BOD | <= 80 mg/L |
| COD | <= 150 mg/L |
| TSS | <= 150 mg/L |
| pH | 6.0 -- 9.0 |
| Temperature | <= 40 °C |
| DO | >= 2.0 mg/L (effluent) |
| TDS | <= 3,500 mg/L |
| EC | <= 4,000 uS/cm |
| Turbidity | <= 50 NTU (treated) |
    """)

    st.divider()
    st.markdown("### Typical Parameter Ranges")
    st.markdown("""
**Inlet (raw wastewater)**
- BOD: 150 -- 350 mg/L
- COD: 300 -- 700 mg/L
- TSS: 100 -- 400 mg/L
- Turbidity: 80 -- 300 NTU
- TDS: 400 -- 1,200 mg/L
- EC: 600 -- 1,600 uS/cm
- pH: 6.5 -- 8.0
- DO: 0.0 -- 2.0 mg/L
- Temperature: 15 -- 30 °C

**Aeration Tank**
- BOD: 80 -- 300 mg/L
- COD: 200 -- 600 mg/L
- TSS: 1,500 -- 4,000 mg/L
- pH: 6.8 -- 7.6
- DO: 1.5 -- 4.0 mg/L

**Effluent (treated, NEQS target)**
- BOD: < 80 mg/L
- COD: < 150 mg/L
- TSS: < 150 mg/L
- Turbidity: < 50 NTU
- pH: 6.0 -- 9.0
- DO: > 2.0 mg/L
    """)

    st.divider()
    st.markdown("### Model Performance")
    st.markdown(f"""
| Metric | Value |
|--------|-------|
| Algorithm | XGBoost |
| BOD R2 | {comp_info["model_BOD_R2"]} |
| COD R2 | {comp_info["model_COD_R2"]} |
| Training rows | {comp_info.get("training_rows", 5578)} |
| Periods | 2009-2010 + 2025 |
    """)

    st.divider()
    st.markdown(f"### {L['trend_header']}")
    if not st.session_state.pred_history:
        st.caption("No readings yet")
    else:
        for i, r in enumerate(st.session_state.pred_history):
            arrow = ""
            if i > 0:
                diff  = r["bod"] - st.session_state.pred_history[i-1]["bod"]
                arrow = "↑" if diff > 5 else ("↓" if diff < -5 else "→")
            st.caption(f"{arrow} BOD {r['bod']:.0f} mg/L  {r['time']}")
        if len(st.session_state.pred_history) >= 3:
            last3 = st.session_state.pred_history[-3:]
            if (last3[1]["bod"] > last3[0]["bod"] and
                    last3[2]["bod"] > last3[1]["bod"]):
                st.warning("Trend alert: BOD increasing across last 3 readings.")

    st.divider()
    st.caption(L["disc"])

# ═══════════════════════════════════════════════════════════════
# MAIN PAGE
# ═══════════════════════════════════════════════════════════════
st.title(L["title"])
st.markdown(L["sub"])
st.divider()

tab_predict, tab_batch = st.tabs([L["predict_btn"], L["batch_tab"]])

# ═══════════════════════════════════════════════════════════════
# PREDICT TAB
# ═══════════════════════════════════════════════════════════════
with tab_predict:

    # How to use
    with st.expander(f"  {L['how_to']}", expanded=False):
        st.markdown("""
- **Enter your field readings** for any available surrogate parameters below.
  Tick **Unknown** for any value you do not have -- the model will still predict
  using the remaining inputs, but confidence will be lower.
- **TSS is the most important parameter** (55% predictive weight). If you can
  only measure one thing, measure TSS.
- **Read the result:** the gauge shows predicted BOD and COD against the
  NEQS 2000 discharge limit. Green = compliant. Amber = borderline.
  Red = non-compliant. Always confirm with a laboratory test before
  official regulatory reporting.
        """)

    with st.container(border=True):
        st.subheader(L["field_section"])

        col1, col2, col3 = st.columns(3)
        with col1:
            location = st.selectbox(
                L["loc_label"],
                ["Not specified / General",
                 "Effluent (discharge point)",
                 "Inlet (raw influent)",
                 "Aeration Tank (biological)"],
                help=L["loc_help"]
            )
            loc_key = "unknown"
            if "Effluent" in location:   loc_key = "Effluent"
            elif "Inlet" in location:    loc_key = "Inlet"
            elif "Aeration" in location: loc_key = "Aeration Tank"

        with col2:
            season     = st.selectbox(L["season_label"],
                                      ["Dry season / No rain",
                                       "Wet season / Monsoon"])
            is_monsoon = "Wet" in season

        with col3:
            rain_event = st.selectbox(L["rain_label"], ["No", "Yes"])
            rain_flag  = 1 if rain_event == "Yes" else 0

        st.markdown(f"**{L['surrogates_header']}**")

        col1, col2 = st.columns(2)
        with col1:
            ph_unknown = st.checkbox(L["unknown"], key="unk_ph")
            if not ph_unknown:
                ph_val = st.selectbox("pH (1 -- 14)",
                                      options=list(range(1, 15)), index=6)
                if ph_val < 5 or ph_val > 10:
                    st.warning("Outside typical wastewater range (5 -- 10)")
            else:
                ph_val = None

            temp_unknown = st.checkbox(L["unknown"], key="unk_temp")
            if not temp_unknown:
                temp_unit = st.radio("Temperature unit", ["C", "F"],
                                     horizontal=True)
                temp_raw  = st.number_input(
                    f"Temperature ({temp_unit})",
                    min_value=-10.0, max_value=150.0,
                    value=22.0, step=0.1)
                temp_val  = (temp_raw - 32) * 5/9 if temp_unit == "F" else temp_raw
            else:
                temp_val = None

            do_unknown = st.checkbox(L["unknown"], key="unk_do")
            do_val = None if do_unknown else st.number_input(
                "Dissolved Oxygen (mg/L)  [NEQS effluent: >= 2.0]",
                min_value=0.0, max_value=20.0, value=1.5, step=0.01)

            turb_unknown = st.checkbox(L["unknown"], key="unk_turb")
            turb_val = None if turb_unknown else st.number_input(
                "Turbidity (NTU)  [NEQS treated: <= 50]",
                min_value=0.0, max_value=3000.0, value=40.0, step=1.0)

        with col2:
            tds_unknown = st.checkbox(L["unknown"], key="unk_tds")
            tds_val = None if tds_unknown else st.number_input(
                "Total Dissolved Solids (mg/L)  [NEQS: <= 3,500]",
                min_value=0.0, max_value=5000.0, value=300.0, step=1.0)

            ec_unknown = st.checkbox(L["unknown"], key="unk_ec")
            if not ec_unknown:
                ec_unit = st.radio("EC unit", ["uS/cm", "mS/cm"],
                                   horizontal=True)
                ec_raw  = st.number_input(
                    f"Electrical Conductivity ({ec_unit})  [NEQS: <= 4,000 uS/cm]",
                    min_value=0.0, max_value=10000.0,
                    value=420.0, step=1.0)
                ec_val  = ec_raw * 1000 if ec_unit == "mS/cm" else ec_raw
            else:
                ec_val = None

            tss_unknown = st.checkbox(L["unknown"], key="unk_tss")
            tss_val = None if tss_unknown else st.number_input(
                "Total Suspended Solids (mg/L)  [NEQS: <= 150]  ★ Primary driver",
                min_value=0.0, max_value=10000.0, value=45.0, step=1.0)

    predict_btn = st.button(
        L["predict_btn"], type="primary", use_container_width=True
    )

    if predict_btn:
        monsoon_flag = 1 if is_monsoon else 0
        unknown_map  = {
            "pH": ph_unknown,
            "Temperature (°C)": temp_unknown,
            "DO (mg/L)": do_unknown,
            "Turbidity (NTU)": turb_unknown,
            "TDS (mg/L)": tds_unknown,
            "EC (µS/cm)": ec_unknown,
            "TSS (mg/L)": tss_unknown,
            "Monsoon_Flag": False,
            "Rain_Event_Flag": False,
        }
        raw_vals = {
            "pH": ph_val, "Temperature (°C)": temp_val,
            "DO (mg/L)": do_val, "Turbidity (NTU)": turb_val,
            "TDS (mg/L)": tds_val, "EC (µS/cm)": ec_val,
            "TSS (mg/L)": tss_val,
            "Monsoon_Flag": monsoon_flag,
            "Rain_Event_Flag": rain_flag,
        }
        missing_features = [c for c in FEATURE_COLS
                            if unknown_map.get(c, False)]

        input_raw = np.array([
            np.nan if unknown_map.get(col, False)
            else (raw_vals[col] or 0.0)
            for col in FEATURE_COLS
        ]).reshape(1, -1)

        ood_warnings = [
            col for col in FEATURE_COLS
            if not unknown_map.get(col, False)
            and raw_vals.get(col) is not None
            and check_ood(col, raw_vals[col])
        ]

        with st.spinner("Computing prediction and uncertainty range..."):
            mean_pred, lower, upper = predict_with_interval(
                input_raw, confidence)

        bod_pred  = float(mean_pred[0])
        cod_pred  = float(mean_pred[1])
        bod_lower = float(lower[0])
        bod_upper = float(upper[0])
        cod_lower = float(lower[1])
        cod_upper = float(upper[1])

        comp_bod = completeness_score(missing_features, "bod")
        comp_cod = completeness_score(missing_features, "cod")

        bod_label, bod_dt = neqs_status(
            bod_pred, bod_lower, bod_upper, NEQS_BOD)
        cod_label, cod_dt = neqs_status(
            cod_pred, cod_lower, cod_upper, NEQS_COD)

        st.session_state.last_pred = {
            "bod": bod_pred, "cod": cod_pred,
            "bod_lower": bod_lower, "bod_upper": bod_upper,
            "cod_lower": cod_lower, "cod_upper": cod_upper,
            "comp": comp_bod, "missing": missing_features,
            "loc": loc_key,
            "time": datetime.now().strftime("%H:%M:%S"),
            "confidence": confidence,
        }
        st.session_state.pred_history.append({
            "bod": bod_pred,
            "time": datetime.now().strftime("%H:%M:%S")
        })
        if len(st.session_state.pred_history) > 5:
            st.session_state.pred_history.pop(0)

        # ── Alerts ────────────────────────────────────────────────
        if ood_warnings:
            st.warning(f"Out-of-training-range inputs detected: "
                       f"{', '.join(ood_warnings)}. "
                       f"Predictions carry higher uncertainty.")
        if loc_key == "Effluent":
            st.info("Within-effluent prediction accuracy is limited "
                    "(R2 = 0.47). Use for compliance screening only, "
                    "not precise operational control.")
        if bod_pred <= NEQS_BOD and cod_pred <= NEQS_COD:
            st.success("Predicted values are within NEQS 2000 limits. "
                       "Confirmatory laboratory testing recommended "
                       "before official discharge certification.")
        if (tss_val is not None and tss_val >= 100
                and loc_key == "Effluent"):
            st.error(f"Operational Alert: Effluent TSS = {tss_val:.0f} mg/L "
                     f"exceeds the 100 mg/L trigger threshold. "
                     f"BOD non-compliance probability exceeds 96%.")
        if comp_bod < 0.5:
            st.warning(f"Data completeness: {comp_bod*100:.0f}%. "
                       f"High-importance surrogates are missing. "
                       f"Prediction interval is wider than usual.")

        # ── Timestamp watermark ────────────────────────────────────
        ts_str  = datetime.now().strftime("%d %b %Y, %H:%M")
        loc_str = loc_key if loc_key != "unknown" else "Location not specified"
        st.markdown(
            f"<div style='text-align:right; color:grey; "
            f"font-size:12px; margin-bottom:8px;'>"
            f"Screening generated at {ts_str} | {loc_str}"
            f"</div>",
            unsafe_allow_html=True
        )

        # ── Gauge charts ──────────────────────────────────────────
        st.markdown("#### Compliance Status")
        gc1, gc2 = st.columns(2)
        with gc1:
            fig_bod = draw_gauge(bod_pred, NEQS_BOD, "BOD")
            st.pyplot(fig_bod, use_container_width=True)
            plt.close(fig_bod)
        with gc2:
            fig_cod = draw_gauge(cod_pred, NEQS_COD, "COD")
            st.pyplot(fig_cod, use_container_width=True)
            plt.close(fig_cod)

        # ── Numeric result cards ──────────────────────────────────
        c1, c2 = st.columns(2)
        with c1:
            st.metric(label=f"BOD  --  {bod_label}",
                      value=f"{bod_pred:.1f} mg/L",
                      delta=f"NEQS limit: {NEQS_BOD} mg/L",
                      delta_color=bod_dt)
            st.caption(
                f"{int(confidence*100)}% Prediction Interval: "
                f"{bod_lower:.1f} -- {bod_upper:.1f} mg/L")
            st.progress(min(comp_bod, 1.0),
                        text=f"{L['completeness']}: "
                             f"{comp_bod*100:.0f}%")
        with c2:
            st.metric(label=f"COD  --  {cod_label}",
                      value=f"{cod_pred:.1f} mg/L",
                      delta=f"NEQS limit: {NEQS_COD} mg/L",
                      delta_color=cod_dt)
            st.caption(
                f"{int(confidence*100)}% Prediction Interval: "
                f"{cod_lower:.1f} -- {cod_upper:.1f} mg/L")
            st.progress(min(comp_cod, 1.0),
                        text=f"{L['completeness']}: "
                             f"{comp_cod*100:.0f}%")

        # ── Sub-tabs ──────────────────────────────────────────────
        sub1, sub2, sub3 = st.tabs([
            L["feature_tab"],
            L["percentile_tab"],
            L["cost_tab"]
        ])

        with sub1:
            if show_feature:
                shap_bod   = shap_weights["bod_weights"]
                feat_names = list(shap_bod.keys())
                feat_vals  = list(shap_bod.values())
                colours    = [
                    "#ef4444" if f in missing_features else "#3b82f6"
                    for f in feat_names
                ]
                sorted_p = sorted(
                    zip(feat_names, feat_vals, colours),
                    key=lambda x: x[1]
                )
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.barh([p[0] for p in sorted_p],
                        [p[1] for p in sorted_p],
                        color=[p[2] for p in sorted_p],
                        alpha=0.85)
                ax.set_xlabel("SHAP importance weight")
                ax.set_title(
                    "Feature contribution to BOD prediction "
                    "(red = missing input)")
                ax.grid(axis="x", alpha=0.3)
                fig.patch.set_alpha(0)
                ax.set_facecolor("none")
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
                st.caption(
                    "Based on SHAP analysis of the trained XGBoost model. "
                    "TSS accounts for 55% of predictive weight. "
                    "Red bars indicate missing inputs.")

        with sub2:
            pct_map  = {
                "pH": ph_val, "Temperature (°C)": temp_val,
                "DO (mg/L)": do_val, "Turbidity (NTU)": turb_val,
                "TDS (mg/L)": tds_val, "EC (µS/cm)": ec_val,
                "TSS (mg/L)": tss_val
            }
            stat_key = {
                "pH": "pH", "Temperature (°C)": "Temperature",
                "DO (mg/L)": "DO", "Turbidity (NTU)": "Turbidity",
                "TDS (mg/L)": "TDS", "EC (µS/cm)": "EC",
                "TSS (mg/L)": "TSS"
            }
            st.caption(
                "Shows where each entered value sits within the "
                "training data distribution (5th -- 95th percentile range).")
            for feat, val in pct_map.items():
                sk = stat_key[feat]
                s  = feat_stats.get(sk, {})
                if val is None or not s:
                    st.caption(f"{feat}: not provided")
                    continue
                lo  = s["min"]
                hi  = s["max"]
                pct = min(100, max(0,
                          (val - lo) / max(hi - lo, 1) * 100))
                st.caption(
                    f"{feat}: **{val:.1f}** "
                    f"({pct:.0f}th percentile of training data)")
                st.progress(pct / 100)

        with sub3:
            if show_cost:
                st.info(
                    "This section estimates potential cost savings "
                    "from using surrogate-based prediction to reduce "
                    "the frequency of expensive laboratory BOD/COD "
                    "tests. Adjust the inputs below to match your "
                    "facility. Actual savings depend on your "
                    "regulatory requirements and confirmatory "
                    "testing protocol.")
                cb1, cb2, cb3 = st.columns(3)
                with cb1:
                    n_tests = st.number_input(
                        "Lab tests per day",
                        min_value=1, max_value=50, value=9)
                with cb2:
                    currency = st.selectbox(
                        "Currency", ["PKR","USD","EUR","GBP"])
                    cost_per = st.number_input(
                        f"Cost per test ({currency})",
                        min_value=100, max_value=100000,
                        value=3500, step=100)
                with cb3:
                    reduction = st.slider(
                        "Test reduction (%)",
                        min_value=10, max_value=80, value=60)

                annual = n_tests * cost_per * 365
                saving = annual * reduction / 100
                sym    = {"PKR":"PKR ","USD":"$",
                          "EUR":"€","GBP":"£"}[currency]
                m1, m2, m3 = st.columns(3)
                m1.metric("Annual lab cost",
                          f"{sym}{annual:,.0f}")
                m2.metric("Estimated annual saving",
                          f"{sym}{saving:,.0f}")
                m3.metric("Tests replaced per day",
                          f"{int(n_tests*reduction/100)} of {n_tests}")

        st.caption(L["disc_full"])

# ═══════════════════════════════════════════════════════════════
# BATCH TAB
# ═══════════════════════════════════════════════════════════════
with tab_batch:
    st.subheader("Batch Prediction Mode")
    st.markdown(
        "Upload a CSV file containing multiple rows of surrogate "
        "measurements to receive BOD and COD predictions for all "
        "samples simultaneously. Useful for processing daily lab logs.")
    st.caption(
        "Expected CSV columns: pH, Temperature (C), DO (mg/L), "
        "Turbidity (NTU), TDS (mg/L), EC (uS/cm), TSS (mg/L), "
        "Monsoon_Flag (0/1), Rain_Event_Flag (0/1)")

    uploaded_file = st.file_uploader(
        "Upload CSV file", type=["csv"])

    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            st.success(f"Loaded {len(batch_df)} rows successfully.")

            results_list = []
            for _, row in batch_df.iterrows():
                inp = np.array([[
                    float(row.get(c, 0.0)) for c in FEATURE_COLS
                ]])
                mean_p, lo, hi = predict_with_interval(inp, 0.95)
                results_list.append({
                    "BOD Predicted (mg/L)": round(float(mean_p[0]), 1),
                    "BOD 95% Low":          round(float(lo[0]), 1),
                    "BOD 95% High":         round(float(hi[0]), 1),
                    "COD Predicted (mg/L)": round(float(mean_p[1]), 1),
                    "BOD NEQS Status": (
                        "Compliant" if mean_p[0] <= NEQS_BOD
                        else "Exceeds"),
                    "COD NEQS Status": (
                        "Compliant" if mean_p[1] <= NEQS_COD
                        else "Exceeds"),
                })

            result_df = pd.concat(
                [batch_df.reset_index(drop=True),
                 pd.DataFrame(results_list)],
                axis=1
            )
            st.dataframe(result_df, use_container_width=True)
            st.download_button(
                "Download Results CSV",
                data=result_df.to_csv(index=False),
                file_name=(
                    f"batch_predictions_"
                    f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    f".csv"),
                mime="text/csv"
            )
        except Exception as e:
            st.error(f"Error processing file: {e}")
    else:
        st.info(
            "Upload a CSV file above. Each row should contain "
            "surrogate parameter values. Columns not present in "
            "the file will be filled with training data means.")
