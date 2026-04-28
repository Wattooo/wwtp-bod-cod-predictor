
import streamlit as st
import numpy as np
import pandas as pd
import pickle, json, os, math
import matplotlib.pyplot as plt
from datetime import datetime

st.set_page_config(page_title="WWTP BOD/COD Predictor",
                   page_icon="W", layout="wide",
                   initial_sidebar_state="expanded")

@st.cache_resource
def load_assets():
    base = os.path.dirname(__file__)
    def r(f,m="rb"): return (pickle.load(open(os.path.join(base,f),m))
                              if m=="rb" else json.load(open(os.path.join(base,f))))
    return (r("final_model.pkl"),r("scaler_final.pkl"),
            r("bootstrap_ensemble.pkl"),
            r("shap_weights.json","r"),r("feature_stats.json","r"),
            r("compliance_info.json","r"))

model,scaler,boot_data,shap_weights,feat_stats,comp_info = load_assets()
FEATURE_COLS = comp_info["feature_cols"]
NEQS_BOD, NEQS_COD = comp_info["NEQS_BOD"], comp_info["NEQS_COD"]

LANG = {
  "en":{"title":"Wastewater BOD & COD Predictor",
    "sub":"Enter surrogate field measurements to predict BOD and COD with NEQS 2000 compliance status",
    "predict_btn":"Predict BOD and COD","unknown":"Unknown",
    "completeness":"Data completeness","feature_tab":"Feature Contribution",
    "cost_tab":"Cost-Benefit","trend_header":"Trend Tracker",
    "how_to":"How to use this tool",
    "disc":"Trained on CDA WWTP I-9/1, Islamabad (2009-2010 and 2025).",
    "disc_full":"Research prototype. FYDP, IIUI 2022-2026. Confirmatory laboratory analysis required for official NEQS compliance reporting."},
  "ur":{"title":"فضلہ پانی BOD اور COD پیش گو",
    "sub":"BOD اور COD کی پیش گوئی کے لیے میدانی پیمائش درج کریں",
    "predict_btn":"BOD اور COD پیش گو کریں","unknown":"معلوم نہیں",
    "completeness":"ڈیٹا مکمل ہونا","feature_tab":"خصوصیت",
    "cost_tab":"لاگت فائدہ","trend_header":"رجحان ٹریکر",
    "how_to":"یہ ٹول کیسے استعمال کریں",
    "disc":"CDA WWTP I-9/1 اسلام آباد کے ڈیٹا پر تربیت یافتہ۔",
    "disc_full":"تحقیقی نمونہ۔ FYDP، IIUI 2022-2026۔"}
}

for k,v in {"lang":"en","pred_history":[],"last_pred":None}.items():
    if k not in st.session_state: st.session_state[k]=v

L = LANG[st.session_state.lang]

def predict_with_interval(input_raw, confidence=0.95):
    preds=[]
    for m,sc in zip(boot_data["models"],boot_data["scalers"]):
        xf=np.where(np.isnan(input_raw),0.0,input_raw)
        xs=sc.transform(xf); xs[np.isnan(input_raw)]=np.nan
        preds.append(m.predict(xs)[0])
    preds=np.array(preds); a=(1-confidence)/2
    return preds.mean(0),np.percentile(preds,a*100,0),np.percentile(preds,(1-a)*100,0)

def completeness_score(missing,target="bod"):
    w=shap_weights[f"{target}_weights"]
    return max(0.0,1.0-sum(w.get(f,0) for f in missing))

def check_ood(feat,val):
    s=feat_stats.get(feat,{}); return bool(s and (val<s["p5"] or val>s["p95"]))

def neqs_status(pred,lo,hi,lim):
    if pred<=lim:    return "COMPLIANT","normal"
    elif lo<=lim:    return "BORDERLINE","off"
    else:            return "NON-COMPLIANT","inverse"

# ── Gauge -- dark background so white needle visible in both modes ─
def draw_gauge(value, limit, label):
    BG = "#1a1a2e"
    fig,ax=plt.subplots(figsize=(4,2.8),subplot_kw=dict(aspect="equal"))
    fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
    from matplotlib.patches import Wedge
    mx=max(limit*2.5, value*1.2)
    g=(limit*0.75/mx)*180; a=(limit/mx)*180
    ax.add_patch(Wedge((0.5,0),0.45,180,180-g,   width=0.13,color="#10b981",alpha=0.9))
    ax.add_patch(Wedge((0.5,0),0.45,180-g,180-a, width=0.13,color="#f59e0b",alpha=0.9))
    ax.add_patch(Wedge((0.5,0),0.45,180-a,0,     width=0.13,color="#ef4444",alpha=0.9))
    ang=math.radians(180-(min(value,mx)/mx)*180)
    nx,ny=0.5+0.34*math.cos(ang),0.34*math.sin(ang)
    ax.annotate("",xy=(nx,ny),xytext=(0.5,0),
                arrowprops=dict(arrowstyle="-|>",color="white",lw=2.5,mutation_scale=16))
    ax.plot(0.5,0,"o",color="white",markersize=7,zorder=5)
    col=("#10b981" if value<=limit*0.75 else "#f59e0b" if value<=limit else "#ef4444")
    ax.text(0.5,0.22,f"{value:.1f}",ha="center",va="center",
            fontsize=16,fontweight="bold",color=col,transform=ax.transAxes)
    ax.text(0.5,0.08,"mg/L",ha="center",va="center",
            fontsize=9,color="#9ca3af",transform=ax.transAxes)
    ax.text(0.5,-0.10,label,ha="center",va="center",
            fontsize=11,fontweight="bold",color="white",transform=ax.transAxes)
    ax.text(0.5,-0.26,f"NEQS limit: {limit} mg/L",ha="center",va="center",
            fontsize=8,color="#9ca3af",transform=ax.transAxes)
    ax.set_xlim(0,1); ax.set_ylim(-0.38,0.58); ax.axis("off")
    plt.tight_layout(pad=0); return fig

# ── Feature chart -- explicit dark bg, white text, teal bars ──────
def draw_feature_chart(missing):
    BG="#1a1a2e"; BAR="#14b8a6"; MISS="#ef4444"; TXT="white"
    sw=shap_weights["bod_weights"]
    sp=sorted(zip(sw.keys(),sw.values()),key=lambda x:x[1])
    cols=[MISS if f in missing else BAR for f,_ in sp]
    fig,ax=plt.subplots(figsize=(8,4))
    fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
    ax.barh([f for f,_ in sp],[v for _,v in sp],color=cols,alpha=0.9)
    ax.set_xlabel("SHAP importance weight",color=TXT,fontsize=10)
    ax.set_title("Feature contribution to BOD prediction (red = missing input)",
                 color=TXT,fontsize=10)
    ax.tick_params(colors=TXT,labelsize=9)
    ax.xaxis.label.set_color(TXT)
    for spine in ax.spines.values(): spine.set_edgecolor("#444")
    ax.grid(axis="x",alpha=0.2,color="#666")
    plt.tight_layout(); return fig

# ── SIDEBAR ───────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## WWTP Predictor")
    st.caption("Surrogate Parameter Decision Support Tool")
    lang_choice=st.radio("Language / زبان",["English","اردو"],horizontal=True)
    st.session_state.lang="ur" if lang_choice=="اردو" else "en"
    L=LANG[st.session_state.lang]

    st.divider()
    st.markdown("### Settings")
    confidence=st.slider(
        "Prediction interval confidence",0.80,0.99,0.95,0.01,
        help="Controls the width of the uncertainty range. At 95%, the model is 95% confident the true BOD/COD falls within the shown interval. Higher values give wider but more conservative intervals. 95% is the scientific standard for regulatory screening.")
    show_feature=st.checkbox("Show feature contribution",value=True)
    show_cost=st.checkbox("Show cost-benefit estimate",value=True)

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
| DO (effluent) | >= 2.0 mg/L |
| TDS | <= 3,500 mg/L |
| EC | <= 4,000 uS/cm |
| Turbidity (treated) | <= 50 NTU |
    """)

    st.divider()
    st.markdown("### Model Performance")
    st.markdown(f"""
| Metric | Value |
|--------|-------|
| Algorithm | XGBoost |
| BOD R2 | {comp_info["model_BOD_R2"]} |
| COD R2 | {comp_info["model_COD_R2"]} |
| Training rows | {comp_info.get("training_rows",5578)} |
| Periods | 2009-2010 + 2025 |
    """)

    st.divider()
    st.markdown(f"### {L['trend_header']}")
    if not st.session_state.pred_history:
        st.caption("No readings yet")
    else:
        for i,r in enumerate(st.session_state.pred_history):
            arrow=""
            if i>0:
                d=r["bod"]-st.session_state.pred_history[i-1]["bod"]
                arrow="↑" if d>5 else ("↓" if d<-5 else "→")
            st.caption(f"{arrow} BOD {r['bod']:.0f} mg/L  {r['time']}")
        if len(st.session_state.pred_history)>=3:
            l3=st.session_state.pred_history[-3:]
            if l3[1]["bod"]>l3[0]["bod"] and l3[2]["bod"]>l3[1]["bod"]:
                st.warning("Trend alert: BOD increasing across last 3 readings.")
    st.divider()
    st.caption(L["disc"])

# ── MAIN ─────────────────────────────────────────────────────────
st.title(L["title"])
st.markdown(L["sub"])
st.divider()

with st.expander(f"  {L['how_to']}",expanded=False):
    st.markdown("""
- **Enter your field readings** below. Tick **Unknown** for any unavailable value — the model still predicts using remaining inputs, but confidence will be lower.
- **TSS is the most important parameter** (55% of predictive weight). If only one measurement is possible, prioritise TSS.
- **Read the gauge:** green = compliant, amber = borderline, red = non-compliant. Always confirm with a laboratory test before official regulatory reporting.
    """)

with st.container(border=True):
    st.subheader("Field Measurements")
    c1,c2,c3=st.columns(3)
    with c1:
        location=st.selectbox("Sampling location (optional)",
            ["Not specified / General","Effluent (discharge point)",
             "Inlet (raw influent)","Aeration Tank (biological)"],
            help="Select for NEQS compliance guidance")
        loc_key=("Effluent" if "Effluent" in location else
                 "Inlet" if "Inlet" in location else
                 "Aeration Tank" if "Aeration" in location else "unknown")
    with c2:
        season=st.selectbox("Seasonal condition",
                            ["Dry season / No rain","Wet season / Monsoon"])
        is_monsoon="Wet" in season
    with c3:
        rain_flag=1 if st.selectbox("Rain event",["No","Yes"])=="Yes" else 0

    st.markdown("**Surrogate parameters — enter all available values**")
    col1,col2=st.columns(2)
    with col1:
        ph_unk=st.checkbox(L["unknown"],key="unk_ph")
        if not ph_unk:
            ph_val=st.selectbox("pH (1 -- 14)",list(range(1,15)),index=6)
            if ph_val<5 or ph_val>10: st.warning("Outside typical wastewater range (5 -- 10)")
        else: ph_val=None

        temp_unk=st.checkbox(L["unknown"],key="unk_temp")
        if not temp_unk:
            tu=st.radio("Temperature unit",["C","F"],horizontal=True)
            tr=st.number_input(f"Temperature ({tu})",-10.0,150.0,22.0,0.1)
            temp_val=(tr-32)*5/9 if tu=="F" else tr
        else: temp_val=None

        do_unk=st.checkbox(L["unknown"],key="unk_do")
        do_val=None if do_unk else st.number_input(
            "Dissolved Oxygen (mg/L)  [NEQS effluent: >= 2.0]",0.0,20.0,1.5,0.01)

        turb_unk=st.checkbox(L["unknown"],key="unk_turb")
        turb_val=None if turb_unk else st.number_input(
            "Turbidity (NTU)  [NEQS treated: <= 50]",0.0,3000.0,40.0,1.0)

    with col2:
        tds_unk=st.checkbox(L["unknown"],key="unk_tds")
        tds_val=None if tds_unk else st.number_input(
            "Total Dissolved Solids (mg/L)  [NEQS: <= 3,500]",0.0,5000.0,300.0,1.0)

        ec_unk=st.checkbox(L["unknown"],key="unk_ec")
        if not ec_unk:
            eu=st.radio("EC unit",["uS/cm","mS/cm"],horizontal=True)
            er=st.number_input(f"Electrical Conductivity ({eu})  [NEQS: <= 4,000 uS/cm]",
                               0.0,10000.0,420.0,1.0)
            ec_val=er*1000 if eu=="mS/cm" else er
        else: ec_val=None

        tss_unk=st.checkbox(L["unknown"],key="unk_tss")
        tss_val=None if tss_unk else st.number_input(
            "Total Suspended Solids (mg/L)  [NEQS: <= 150]  ★ Primary driver",
            0.0,10000.0,45.0,1.0)

predict_btn=st.button(L["predict_btn"],type="primary",use_container_width=True)

if predict_btn:
    unk_map={"pH":ph_unk,"Temperature (°C)":temp_unk,"DO (mg/L)":do_unk,
             "Turbidity (NTU)":turb_unk,"TDS (mg/L)":tds_unk,
             "EC (µS/cm)":ec_unk,"TSS (mg/L)":tss_unk,
             "Monsoon_Flag":False,"Rain_Event_Flag":False}
    raw={"pH":ph_val,"Temperature (°C)":temp_val,"DO (mg/L)":do_val,
         "Turbidity (NTU)":turb_val,"TDS (mg/L)":tds_val,"EC (µS/cm)":ec_val,
         "TSS (mg/L)":tss_val,"Monsoon_Flag":1 if is_monsoon else 0,
         "Rain_Event_Flag":rain_flag}

    missing=[c for c in FEATURE_COLS if unk_map.get(c,False)]
    inp=np.array([np.nan if unk_map.get(c,False) else (raw[c] or 0.0)
                  for c in FEATURE_COLS]).reshape(1,-1)
    ood=[c for c in FEATURE_COLS if not unk_map.get(c,False)
         and raw.get(c) is not None and check_ood(c,raw[c])]

    with st.spinner("Computing prediction and uncertainty range..."):
        mp,lo,hi=predict_with_interval(inp,confidence)

    bp,cp=float(mp[0]),float(mp[1])
    bl,bh=float(lo[0]),float(hi[0])
    cl,ch=float(lo[1]),float(hi[1])
    cb,cc=completeness_score(missing,"bod"),completeness_score(missing,"cod")
    blbl,bdt=neqs_status(bp,bl,bh,NEQS_BOD)
    clbl,cdt=neqs_status(cp,cl,ch,NEQS_COD)

    st.session_state.last_pred=dict(bod=bp,cod=cp,bod_lower=bl,bod_upper=bh,
        cod_lower=cl,cod_upper=ch,comp=cb,missing=missing,
        loc=loc_key,time=datetime.now().strftime("%H:%M:%S"),confidence=confidence)
    st.session_state.pred_history.append(
        dict(bod=bp,time=datetime.now().strftime("%H:%M:%S")))
    if len(st.session_state.pred_history)>5:
        st.session_state.pred_history.pop(0)

    if ood:
        st.warning(f"Out-of-training-range inputs: {', '.join(ood)}. Predictions carry higher uncertainty.")
    if loc_key=="Effluent":
        st.info("Within-effluent prediction accuracy is limited (R2 = 0.47). Use for compliance screening only.")
    if bp<=NEQS_BOD and cp<=NEQS_COD:
        st.success("Predicted values are within NEQS 2000 limits. Confirmatory laboratory testing recommended.")
    if tss_val and tss_val>=100 and loc_key=="Effluent":
        st.error(f"Operational Alert: Effluent TSS = {tss_val:.0f} mg/L exceeds 100 mg/L trigger. Non-compliance probability > 96%.")
    if cb<0.5:
        st.warning(f"Data completeness: {cb*100:.0f}%. High-importance surrogates are missing.")

    st.markdown(
        f"<div style='text-align:right;color:grey;font-size:12px;margin-bottom:8px;'>"
        f"Screening generated at {datetime.now().strftime('%d %b %Y, %H:%M')} "
        f"| {loc_key if loc_key!='unknown' else 'Location not specified'}</div>",
        unsafe_allow_html=True)

    st.markdown("#### Compliance Status")
    gc1,gc2=st.columns(2)
    with gc1:
        f1=draw_gauge(bp,NEQS_BOD,"BOD")
        st.pyplot(f1,use_container_width=True); plt.close(f1)
    with gc2:
        f2=draw_gauge(cp,NEQS_COD,"COD")
        st.pyplot(f2,use_container_width=True); plt.close(f2)

    r1,r2=st.columns(2)
    with r1:
        st.metric(f"BOD  --  {blbl}",f"{bp:.1f} mg/L",
                  f"NEQS limit: {NEQS_BOD} mg/L",delta_color=bdt)
        st.caption(f"{int(confidence*100)}% Prediction Interval: {bl:.1f} -- {bh:.1f} mg/L")
        st.progress(min(cb,1.0),text=f"{L['completeness']}: {cb*100:.0f}%")
    with r2:
        st.metric(f"COD  --  {clbl}",f"{cp:.1f} mg/L",
                  f"NEQS limit: {NEQS_COD} mg/L",delta_color=cdt)
        st.caption(f"{int(confidence*100)}% Prediction Interval: {cl:.1f} -- {ch:.1f} mg/L")
        st.progress(min(cc,1.0),text=f"{L['completeness']}: {cc*100:.0f}%")

    t1,t2=st.tabs([L["feature_tab"],L["cost_tab"]])
    with t1:
        if show_feature:
            fig=draw_feature_chart(missing)
            st.pyplot(fig,use_container_width=True); plt.close(fig)
            st.caption("TSS accounts for 55% of predictive weight. Red bars indicate missing inputs.")

    with t2:
        if show_cost:
            st.info("Estimates potential savings from using surrogate prediction to reduce laboratory testing frequency.")
            cb1,cb2,cb3=st.columns(3)
            with cb1: nt=st.number_input("Lab tests / day",1,50,9)
            with cb2:
                cur=st.selectbox("Currency",["PKR","USD","EUR","GBP"])
                cp_=st.number_input(f"Cost per test ({cur})",100,100000,3500,100)
            with cb3: red=st.slider("Test reduction (%)",10,80,60)
            ann=nt*cp_*365; sav=ann*red/100
            sym={"PKR":"PKR ","USD":"$","EUR":"€","GBP":"£"}[cur]
            m1,m2,m3=st.columns(3)
            m1.metric("Annual lab cost",f"{sym}{ann:,.0f}")
            m2.metric("Annual saving",  f"{sym}{sav:,.0f}")
            m3.metric("Tests replaced/day",f"{int(nt*red/100)} of {nt}")

    st.caption(L["disc_full"])
