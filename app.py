"""
⚡ FitPulse — Fitbit Fitness Analytics Dashboard
Bellabeat / Strava Fitness Case Study

Run:  streamlit run app.py
Needs the `data/` folder (created by index.ipynb) next to this file.
"""

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

st.set_page_config(page_title="FitPulse | Fitbit Analytics", page_icon="⚡", layout="wide")

DATA = Path(__file__).parent / "data"
WD = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
PALETTE = ["#00E5A8", "#7C4DFF", "#FF6B6B", "#FFC94D", "#4DA6FF", "#FF8AD8"]
STAGE = {1: "Asleep", 2: "Restless", 3: "Awake"}

# ------------------------------------------------------------------
# STYLE
# ------------------------------------------------------------------
st.markdown("""
<style>
.stApp {background: radial-gradient(1100px 600px at 8% -10%, #241a5e 0%, #0b0f1f 55%);}
section[data-testid="stSidebar"] {background: linear-gradient(180deg,#14102f,#0b0f1f); border-right:1px solid rgba(255,255,255,.08);}
.stApp, .stApp p, .stApp label, .stApp span, .stApp li, .stApp h1, .stApp h2, .stApp h3, .stApp h4 {color:#e8ecf7;}
.block-container {padding-top: 1.6rem; max-width: 1400px;}
.hero {padding:28px 34px; border-radius:24px; margin-bottom:18px;
       background: linear-gradient(120deg,#7C4DFF 0%,#00B8D4 58%,#00E5A8 100%); box-shadow:0 12px 40px rgba(124,77,255,.35);}
.hero h1 {margin:0; font-size:2.3rem; color:#fff !important; letter-spacing:.5px;}
.hero p {margin:6px 0 0 0; color:#f2fffb !important; font-size:1.02rem;}
.kpi {position:relative; padding:16px 18px 14px 22px; border-radius:18px; overflow:hidden; min-height:122px; margin-bottom:10px;
      background: linear-gradient(135deg,rgba(255,255,255,.07),rgba(255,255,255,.02)); border:1px solid rgba(255,255,255,.09);
      transition: transform .2s ease, box-shadow .2s ease;}
.kpi:hover {transform: translateY(-3px); box-shadow:0 10px 28px rgba(0,229,168,.18);}
.kpi::before {content:""; position:absolute; left:0; top:0; bottom:0; width:6px; background:linear-gradient(180deg,var(--c1),var(--c2));}
.kpi-icon {position:absolute; right:14px; top:10px; font-size:1.7rem; opacity:.9;}
.kpi-label {font-size:.78rem; text-transform:uppercase; letter-spacing:1.3px; color:#aab3d1 !important;}
.kpi-value {font-size:2rem; font-weight:800; line-height:1.25; background:linear-gradient(90deg,var(--c1),var(--c2));
            -webkit-background-clip:text; -webkit-text-fill-color:transparent;}
.kpi-sub {font-size:.8rem; color:#8f99bd !important;}
.mini {padding:9px 12px; border-radius:12px; margin-bottom:8px; background:rgba(255,255,255,.05); border:1px solid rgba(255,255,255,.08);}
.mini b {font-size:1.15rem; color:#00E5A8;}
.mini span {display:block; font-size:.7rem; color:#9aa4c7 !important; text-transform:uppercase; letter-spacing:1px;}
.insight {padding:18px 20px; border-radius:16px; margin-bottom:14px; background:rgba(255,255,255,.05);
          border:1px solid rgba(255,255,255,.09); border-left:6px solid var(--c);}
.insight h4 {margin:0 0 6px 0; color:#fff !important;}
.insight .rec {margin-top:8px; padding:8px 12px; border-radius:10px; background:rgba(0,229,168,.10); color:#b9ffe9 !important;}
.sec {font-size:1.25rem; font-weight:700; margin:18px 0 6px 0; padding-left:10px; border-left:5px solid #00E5A8;}
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------------
def kpi(icon, label, value, sub="", c1="#00E5A8", c2="#7C4DFF"):
    return (f'<div class="kpi" style="--c1:{c1};--c2:{c2}"><div class="kpi-icon">{icon}</div>'
            f'<div class="kpi-label">{label}</div><div class="kpi-value">{value}</div>'
            f'<div class="kpi-sub">{sub}</div></div>')


def kpi_row(cards):
    for col, c in zip(st.columns(len(cards)), cards):
        col.markdown(c, unsafe_allow_html=True)


def section(text):
    st.markdown(f'<div class="sec">{text}</div>', unsafe_allow_html=True)


def show(fig, height=380):
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      height=height, margin=dict(l=10, r=10, t=55, b=10), title_font_size=16,
                      legend=dict(bgcolor="rgba(0,0,0,0)"))
    try:
        st.plotly_chart(fig, width="stretch")
    except TypeError:
        st.plotly_chart(fig, use_container_width=True)


def table(df, **kw):
    try:
        st.dataframe(df, width="stretch", **kw)
    except TypeError:
        st.dataframe(df, use_container_width=True, **kw)


def scatter(df, x, y, **kw):
    """scatter with OLS trendline when there is enough data"""
    if len(df) > 3:
        try:
            return px.scatter(df, x=x, y=y, trendline="ols", trendline_color_override="#FF6B6B", **kw)
        except Exception:
            pass
    return px.scatter(df, x=x, y=y, **kw)


def corr_text(a, b):
    return f"{a.corr(b):.2f}" if len(a) > 2 else "n/a"


# ------------------------------------------------------------------
# DATA  (works with the raw-clean files in data/ — no other prep needed)
# ------------------------------------------------------------------
def norm(x):
    return x.lower().replace("_", "").replace("-", "").replace(" ", "")


def pick(*names, key=None):
    """find a csv in data/: exact name first (any letter-case), else any file whose name contains `key`"""
    files = sorted(DATA.glob("*.csv"), key=lambda p: ("clean" not in p.name.lower(), p.name.lower()))
    have = {p.name.lower(): p for p in files}
    for n in names:
        if n.lower() in have:
            return have[n.lower()]
    if key:
        for p in files:
            if key in norm(p.name):
                return p
    return None


def need(*names, key=None):
    p = pick(*names, key=key)
    if p is None:
        found = ", ".join(sorted(f.name for f in DATA.glob("*"))) or "(folder is empty)"
        st.error(f"❌ File not found in the `data/` folder: **{names[0]}**  \n"
                 f"Looking in: `{DATA}`  \nFiles found there: {found}")
        st.stop()
    return p


def to_dt(s):
    """parse dates in any of the formats used by these files"""
    if pd.api.types.is_datetime64_any_dtype(s):
        return s
    s = s.astype(str)
    sample = s.iloc[0].strip().upper() if len(s) else ""
    if sample.endswith(("AM", "PM")):
        return pd.to_datetime(s, format="%m/%d/%Y %I:%M:%S %p", errors="coerce")
    if "/" in sample and ":" not in sample:
        return pd.to_datetime(s, format="%m/%d/%Y", errors="coerce")
    return pd.to_datetime(s, errors="coerce")


def read(names, key=None, **kw):
    return pd.read_csv(need(*names, key=key), dtype={"Id": str}, **kw)


def save_cache(df, name):
    try:
        df.to_csv(DATA / name, index=False)      # speeds up next start
    except Exception:
        pass


@st.cache_data(show_spinner="Loading data… (first run aggregates the big files, please wait)")
def load():
    # ---- daily activity
    da = read(["dailyActivity_clean.csv", "daily_activity.csv", "dailyActivity_merged.csv"], key="dailyactivity")
    da = da.rename(columns={"ActivityDate": "Date"})
    da["Date"] = to_dt(da["Date"]).dt.normalize()
    da = da.drop_duplicates(["Id", "Date"])
    if "TotalActiveMinutes" not in da:
        da["TotalActiveMinutes"] = da["VeryActiveMinutes"] + da["FairlyActiveMinutes"] + da["LightlyActiveMinutes"]
    if "IsNonWearDay" not in da:
        da["IsNonWearDay"] = (da["TotalSteps"] == 0) & (da["Calories"] == 0)
    da["IsNonWearDay"] = da["IsNonWearDay"].astype(bool)

    # ---- sleep (per day)
    sl = read(["sleepDay_clean.csv", "sleep_day.csv", "sleepDay_merged.csv"], key="sleepday").rename(columns={"SleepDay": "Date"})
    sl["Date"] = to_dt(sl["Date"]).dt.normalize()
    sl = sl.drop_duplicates(["Id", "Date"])
    if "TimeAwakeInBed" not in sl:
        sl["TimeAwakeInBed"] = sl["TotalTimeInBed"] - sl["TotalMinutesAsleep"]

    # ---- weight
    wt = read(["weightLogInfo_clean.csv", "weight_log.csv", "weightLogInfo_merged.csv"], key="weight")
    wt["Date"] = to_dt(wt["Date"]).dt.normalize()
    wt = wt.drop_duplicates(["Id", "Date"])
    wt["IsManualReport"] = wt["IsManualReport"].astype(bool)

    # ---- hourly (steps + calories + intensities)
    p = pick("hourly.csv")
    if p is not None:
        hr = pd.read_csv(p, dtype={"Id": str})
        hr["ActivityHour"] = to_dt(hr["ActivityHour"])
    else:
        hs = read(["hourlySteps_clean.csv", "hourlySteps_merged.csv"], key="hourlysteps")
        hc = read(["hourlyCalories_clean.csv", "hourlyCalories_merged.csv"], key="hourlycalories")
        hi = read(["hourlyIntensities_clean.csv", "hourlyIntensities_merged.csv"], key="hourlyintensities")
        for d in (hs, hc, hi):
            d["ActivityHour"] = to_dt(d["ActivityHour"])
        hr = hs.merge(hc, on=["Id", "ActivityHour"]).merge(hi, on=["Id", "ActivityHour"])
    hr = hr.drop_duplicates(["Id", "ActivityHour"])
    hr["Date"] = hr["ActivityHour"].dt.normalize()
    hr["Hour"] = hr["ActivityHour"].dt.hour

    # ---- heart rate (seconds file -> hourly + daily, cached to data/)
    ph, pdly = pick("heartrate_hourly.csv"), pick("heartrate_daily.csv")
    if ph is not None and pdly is not None:
        hh = pd.read_csv(ph, dtype={"Id": str}, parse_dates=["Date"])
        hd = pd.read_csv(pdly, dtype={"Id": str}, parse_dates=["Date"])
    else:
        sec = pd.read_csv(need("heartrate_seconds_clean.csv", "heartrate_seconds_merged.csv", key="second"),
                          dtype={"Id": str}, usecols=["Id", "Time", "Value"])
        sec["Time"] = to_dt(sec["Time"])
        sec["Date"] = sec["Time"].dt.normalize()
        sec["Hour"] = sec["Time"].dt.hour
        hh = sec.groupby(["Id", "Date", "Hour"])["Value"].agg(AvgHR="mean", MinHR="min", MaxHR="max").reset_index()
        hd = sec.groupby(["Id", "Date"])["Value"].agg(AvgHR="mean", MinHR="min", MaxHR="max", Readings="count").reset_index()
        hh[["AvgHR", "MinHR", "MaxHR"]] = hh[["AvgHR", "MinHR", "MaxHR"]].round(1)
        hd[["AvgHR", "MinHR", "MaxHR"]] = hd[["AvgHR", "MinHR", "MaxHR"]].round(1)
        del sec
        save_cache(hh, "heartrate_hourly.csv")
        save_cache(hd, "heartrate_daily.csv")

    # ---- minute sleep
    ms = read(["minuteSleep_clean.csv", "minute_sleep.csv", "minuteSleep_merged.csv"], key="minutesleep")
    ms = ms.rename(columns={"date": "DateTime", "value": "SleepStage"})
    ms["DateTime"] = to_dt(ms["DateTime"])
    ms["Date"] = ms["DateTime"].dt.normalize()
    ms["Hour"] = ms["DateTime"].dt.hour
    ms["Stage"] = ms["SleepStage"].map(STAGE)

    # ---- METs (minute file -> hourly, cached to data/). Fitbit stores METs x 10
    pm = pick("mets_hourly.csv")
    if pm is not None:
        mt = pd.read_csv(pm, dtype={"Id": str}, parse_dates=["Date"])
    else:
        m = pd.read_csv(need("minuteMETsNarrow_clean.csv", "minuteMETsNarrow_merged.csv", key="minutemets"),
                        dtype={"Id": str}, usecols=["Id", "ActivityMinute", "METs"])
        m["ActivityMinute"] = to_dt(m["ActivityMinute"])
        m["METs"] = m["METs"] / 10
        m["Date"] = m["ActivityMinute"].dt.normalize()
        m["Hour"] = m["ActivityMinute"].dt.hour
        mt = m.groupby(["Id", "Date", "Hour"])["METs"].agg(AvgMETs="mean", MaxMETs="max").reset_index().round(2)
        del m
        save_cache(mt, "mets_hourly.csv")

    umap = {i: f"U{n + 1:02d}" for n, i in enumerate(sorted(da["Id"].unique()))}
    for df in (da, sl, wt, hr, hd, hh, ms, mt):
        df["User"] = df["Id"].map(umap)
        df["Weekday"] = pd.Categorical(df["Date"].dt.day_name(), categories=WD, ordered=True)
    da["SedentaryHours"] = da["SedentaryMinutes"] / 60
    sl["HoursAsleep"] = sl["TotalMinutesAsleep"] / 60
    sl["Efficiency"] = (sl["TotalMinutesAsleep"] / sl["TotalTimeInBed"] * 100).round(1)
    return da, sl, wt, hr, hd, hh, ms, mt, umap


da0, sl0, wt0, hr0, hd0, hh0, ms0, mt0, UMAP = load()


@st.cache_resource
def sql_conn():
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    for name, df in {"daily_activity": da0, "sleep_day": sl0, "weight_log": wt0, "hourly": hr0,
                     "heartrate_daily": hd0, "heartrate_hourly": hh0, "minute_sleep": ms0, "mets_hourly": mt0}.items():
        d = df.copy()
        d["Weekday"] = d["Weekday"].astype(str)
        for c in d.columns:
            if pd.api.types.is_datetime64_any_dtype(d[c]):
                d[c] = d[c].dt.strftime("%Y-%m-%d %H:%M:%S" if c in ("ActivityHour", "DateTime") else "%Y-%m-%d")
        d.to_sql(name, conn, index=False)
    return conn


# ------------------------------------------------------------------
# SIDEBAR : index + filters + mini KPIs
# ------------------------------------------------------------------
PAGES = ["🏠 Home", "👟 Steps & Distance", "🔥 Calories & Intensity", "🪑 Sedentary Behaviour",
         "😴 Sleep Lab", "❤️ Heart Rate & METs", "⚖️ Weight & BMI", "👤 User Explorer",
         "🗄️ SQL Lab", "💡 Insights", "🧼 Data Cleaning"]

st.sidebar.markdown("## ⚡ FitPulse")
st.sidebar.caption("Fitbit analytics · Bellabeat case study")
st.sidebar.markdown("### 📑 Index")
page = st.sidebar.radio("index", PAGES, label_visibility="collapsed")

st.sidebar.markdown("### 🎛️ Filters")
all_users = sorted(UMAP.values())
sel_users = st.sidebar.multiselect("Users", all_users, default=all_users)
dmin, dmax = da0["Date"].min().date(), da0["Date"].max().date()
rng = st.sidebar.date_input("Date range", (dmin, dmax), min_value=dmin, max_value=dmax)
start, end = (pd.Timestamp(rng[0]), pd.Timestamp(rng[1])) if len(rng) == 2 else (pd.Timestamp(dmin), pd.Timestamp(dmax))


def F(df, users=True):
    d = df[(df["Date"] >= start) & (df["Date"] <= end)]
    return d[d["User"].isin(sel_users)] if users else d


da, sl, wt, hr, hd, hh, ms, mt = (F(x) for x in (da0, sl0, wt0, hr0, hd0, hh0, ms0, mt0))
if da.empty:
    st.warning("No data for the selected filters — widen the users / date range in the sidebar.")
    st.stop()

st.sidebar.markdown("### 📌 Quick Stats")
st.sidebar.markdown(f'<div class="mini"><span>Users selected</span><b>{da["User"].nunique()}</b></div>'
                    f'<div class="mini"><span>User-days</span><b>{len(da):,}</b></div>'
                    f'<div class="mini"><span>Avg steps / day</span><b>{da["TotalSteps"].mean():,.0f}</b></div>'
                    f'<div class="mini"><span>Avg sleep</span><b>{sl["HoursAsleep"].mean() if len(sl) else 0:.1f} h</b></div>',
                    unsafe_allow_html=True)
st.sidebar.caption("Data: Fitbit Fitness Tracker · 33 users · Apr–May 2016")


def hero(title, sub):
    st.markdown(f'<div class="hero"><h1>{title}</h1><p>{sub}</p></div>', unsafe_allow_html=True)


def segment(s):
    return ("Sedentary (<5k)" if s < 5000 else "Lightly Active (5–7.5k)" if s < 7500
            else "Fairly Active (7.5–10k)" if s < 10000 else "Very Active (10k+)")


# ==================================================================
# PAGES
# ==================================================================
def page_home():
    hero("⚡ FitPulse Dashboard", "How do people really use their smart fitness devices? Explore steps, calories, sleep, heart rate and more.")
    pct_goal = (da["TotalSteps"] >= 10000).mean() * 100
    kpi_row([kpi("👥", "Users", da["User"].nunique(), "in current filter"),
             kpi("👟", "Avg Steps / Day", f"{da['TotalSteps'].mean():,.0f}", "target 10,000", "#FFC94D", "#FF6B6B"),
             kpi("🔥", "Avg Calories", f"{da['Calories'].mean():,.0f}", "kcal / day", "#FF6B6B", "#FF8AD8"),
             kpi("😴", "Avg Sleep", f"{sl['HoursAsleep'].mean():.1f} h" if len(sl) else "n/a", "recommended 7–9 h", "#4DA6FF", "#7C4DFF")])
    kpi_row([kpi("🎯", "Days ≥ 10k Steps", f"{pct_goal:.1f}%", "CDC-style goal", "#00E5A8", "#4DA6FF"),
             kpi("🪑", "Sedentary Hours", f"{da['SedentaryHours'].mean():.1f} h", "per day", "#FF8AD8", "#7C4DFF"),
             kpi("❤️", "Avg Heart Rate", f"{hd['AvgHR'].mean():.0f} bpm" if len(hd) else "n/a", f"{hd['User'].nunique()} users with HR", "#FF6B6B", "#FFC94D"),
             kpi("📅", "User-Days", f"{len(da):,}", f"{da['Date'].min():%d %b} → {da['Date'].max():%d %b}", "#7C4DFF", "#00E5A8")])

    c1, c2 = st.columns([2, 1])
    with c1:
        t = da.groupby("Date")["TotalSteps"].mean().reset_index()
        t["7-day avg"] = t["TotalSteps"].rolling(7, min_periods=1).mean()
        fig = go.Figure()
        fig.add_bar(x=t["Date"], y=t["TotalSteps"], name="Daily avg", marker_color="rgba(124,77,255,.55)")
        fig.add_scatter(x=t["Date"], y=t["7-day avg"], name="7-day avg", line=dict(color="#00E5A8", width=3))
        fig.update_layout(title="📈 Average Steps per Day (all selected users)")
        show(fig)
    with c2:
        fig = go.Figure(go.Indicator(mode="gauge+number", value=pct_goal, number={"suffix": "%"},
                                     title={"text": "Days meeting 10k steps"},
                                     gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#00E5A8"},
                                            "steps": [{"range": [0, 33], "color": "rgba(255,107,107,.35)"},
                                                      {"range": [33, 66], "color": "rgba(255,201,77,.35)"},
                                                      {"range": [66, 100], "color": "rgba(0,229,168,.25)"}]}))
        show(fig)

    c3, c4, c5 = st.columns(3)
    with c3:
        m = da[["SedentaryMinutes", "LightlyActiveMinutes", "FairlyActiveMinutes", "VeryActiveMinutes"]].mean()
        fig = px.pie(values=m.values, names=[n.replace("Minutes", "") for n in m.index], hole=.55,
                     color_discrete_sequence=PALETTE, title="⏱️ Where the 24 hours go")
        show(fig, 340)
    with c4:
        ua = da.groupby("User")["TotalSteps"].mean().apply(segment).value_counts().reset_index()
        ua.columns = ["Segment", "Users"]
        fig = px.bar(ua, x="Users", y="Segment", orientation="h", color="Segment", color_discrete_sequence=PALETTE,
                     title="🧬 User Segments (avg steps)")
        fig.update_layout(showlegend=False)
        show(fig, 340)
    with c5:
        fun = pd.DataFrame({"Data type": ["Activity", "Sleep", "Heart rate", "Weight"],
                            "Users": [da["User"].nunique(), sl["User"].nunique(), hd["User"].nunique(), wt["User"].nunique()]})
        fig = px.funnel(fun, x="Users", y="Data type", color_discrete_sequence=["#7C4DFF"], title="🔌 Feature Adoption")
        show(fig, 340)


def page_steps():
    hero("👟 Steps & Distance", "How much do people move, and when?")
    kpi_row([kpi("👟", "Avg Steps", f"{da['TotalSteps'].mean():,.0f}", "per user-day"),
             kpi("🏆", "Best Day", f"{da['TotalSteps'].max():,}", "max steps", "#FFC94D", "#FF6B6B"),
             kpi("📏", "Avg Distance", f"{da['TotalDistance'].mean():.2f} km", "per day", "#4DA6FF", "#7C4DFF"),
             kpi("🎯", "Days ≥ 10k", f"{(da['TotalSteps'] >= 10000).sum():,}", f"{(da['TotalSteps'] >= 10000).mean() * 100:.1f}% of days", "#00E5A8", "#4DA6FF")])
    c1, c2 = st.columns(2)
    with c1:
        w = da.groupby("Weekday", observed=True)["TotalSteps"].mean().reindex(WD).reset_index()
        fig = px.bar(w, x="Weekday", y="TotalSteps", color="TotalSteps", color_continuous_scale="Viridis", title="📆 Avg Steps by Weekday")
        show(fig)
    with c2:
        fig = px.histogram(da, x="TotalSteps", nbins=35, color_discrete_sequence=["#7C4DFF"], title="📊 Daily Steps Distribution")
        fig.add_vline(x=10000, line_dash="dash", line_color="#FF6B6B", annotation_text="10k goal")
        show(fig)
    h = hr.pivot_table(index="Weekday", columns="Hour", values="StepTotal", aggfunc="mean", observed=True).reindex(WD)
    fig = px.imshow(h, aspect="auto", color_continuous_scale="Turbo", labels=dict(x="Hour of day", y="", color="Avg steps"),
                    title="🔥 When do people walk? (weekday × hour heatmap)")
    show(fig, 380)
    c3, c4 = st.columns(2)
    with c3:
        fig = px.box(da, x="Weekday", y="TotalSteps", color="Weekday", category_orders={"Weekday": WD},
                     color_discrete_sequence=PALETTE, title="📦 Spread of Steps by Weekday")
        fig.update_layout(showlegend=False)
        show(fig)
    with c4:
        u = da.groupby("User")["TotalSteps"].mean().sort_values(ascending=False).reset_index()
        fig = px.bar(u, x="User", y="TotalSteps", color="TotalSteps", color_continuous_scale="Tealgrn", title="🏅 Avg Steps per User")
        show(fig)
    fig = scatter(da, "TotalSteps", "TotalDistance", opacity=.5, color_discrete_sequence=["#4DA6FF"], title="📐 Steps vs Distance")
    show(fig, 340)


def page_cal():
    hero("🔥 Calories & Intensity", "Energy burnt and how hard people work out.")
    da2 = da[~da["IsNonWearDay"]]
    kpi_row([kpi("🔥", "Avg Calories", f"{da2['Calories'].mean():,.0f}", "excl. non-wear days", "#FF6B6B", "#FFC94D"),
             kpi("💥", "Peak Day", f"{da['Calories'].max():,}", "kcal", "#FFC94D", "#FF6B6B"),
             kpi("🔗", "Steps ↔ Calories", corr_text(da2["TotalSteps"], da2["Calories"]), "correlation r", "#7C4DFF", "#00E5A8"),
             kpi("⚡", "Active Minutes", f"{da['TotalActiveMinutes'].mean():.0f}", "light+fair+very / day", "#00E5A8", "#4DA6FF")])
    c1, c2 = st.columns(2)
    with c1:
        fig = scatter(da2, "TotalSteps", "Calories", color="VeryActiveMinutes", opacity=.7,
                      color_continuous_scale="Plasma", title="👟 Steps vs Calories (colour = very-active minutes)")
        show(fig)
    with c2:
        h = hr.groupby("Hour")[["Calories", "AverageIntensity"]].mean().reset_index()
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_bar(x=h["Hour"], y=h["Calories"], name="Calories", marker_color="rgba(255,107,107,.7)")
        fig.add_scatter(x=h["Hour"], y=h["AverageIntensity"], name="Intensity", line=dict(color="#00E5A8", width=3), secondary_y=True)
        fig.update_layout(title="🕒 Calories & Intensity by Hour of Day")
        show(fig)
    c3, c4 = st.columns(2)
    with c3:
        w = da.groupby("Weekday", observed=True)[["LightlyActiveMinutes", "FairlyActiveMinutes", "VeryActiveMinutes"]].mean().reindex(WD).reset_index()
        fig = px.bar(w, x="Weekday", y=["LightlyActiveMinutes", "FairlyActiveMinutes", "VeryActiveMinutes"],
                     color_discrete_sequence=["#4DA6FF", "#FFC94D", "#FF6B6B"], title="💪 Active Minutes by Weekday (stacked)")
        show(fig)
    with c4:
        cols = ["TotalSteps", "TotalDistance", "VeryActiveMinutes", "FairlyActiveMinutes", "LightlyActiveMinutes", "SedentaryMinutes", "Calories"]
        fig = px.imshow(da[cols].corr().round(2), text_auto=True, color_continuous_scale="RdBu_r", zmin=-1, zmax=1, title="🧮 Correlation Matrix")
        show(fig)
    w2 = da.groupby("Weekday", observed=True)["Calories"].mean().reindex(WD).reset_index()
    fig = px.line(w2, x="Weekday", y="Calories", markers=True, title="📆 Avg Calories by Weekday", color_discrete_sequence=["#FF8AD8"])
    show(fig, 300)


def page_sed():
    hero("🪑 Sedentary Behaviour", "How much of the day is spent sitting still?")
    sed_pct = da["SedentaryMinutes"].mean() / 1440 * 100
    top = da.groupby("User")["SedentaryHours"].mean().idxmax()
    kpi_row([kpi("🪑", "Avg Sedentary", f"{da['SedentaryHours'].mean():.1f} h", "per day", "#FF8AD8", "#7C4DFF"),
             kpi("📉", "Share of Day", f"{sed_pct:.0f}%", "of 24 hours", "#FF6B6B", "#FFC94D"),
             kpi("🚨", "Most Sedentary", top, "highest avg hours", "#FFC94D", "#FF6B6B"),
             kpi("📴", "Non-wear Days", f"{int(da['IsNonWearDay'].sum())}", "0 steps & 0 kcal", "#4DA6FF", "#7C4DFF")])
    h = da.pivot_table(index="User", columns="Weekday", values="SedentaryHours", aggfunc="mean", observed=True)[WD]
    fig = px.imshow(h, aspect="auto", color_continuous_scale="RdYlBu_r", labels=dict(color="Hours"),
                    title="🌡️ Avg Sedentary Hours — User × Weekday")
    show(fig, 620)
    c1, c2 = st.columns(2)
    with c1:
        fig = px.histogram(da, x="SedentaryHours", nbins=30, color_discrete_sequence=["#FF8AD8"], title="📊 Distribution of Sedentary Hours")
        show(fig)
    with c2:
        fig = scatter(da, "SedentaryHours", "TotalSteps", opacity=.5, color_discrete_sequence=["#FFC94D"], title="🔀 Sedentary Hours vs Steps")
        show(fig)


def page_sleep():
    hero("😴 Sleep Lab", "Duration, efficiency and sleep stages.")
    if sl.empty:
        st.info("No sleep data for the current filter.")
        return
    kpi_row([kpi("😴", "Avg Sleep", f"{sl['HoursAsleep'].mean():.1f} h", "per night", "#4DA6FF", "#7C4DFF"),
             kpi("🛏️", "Avg Time in Bed", f"{sl['TotalTimeInBed'].mean() / 60:.1f} h", "per night", "#7C4DFF", "#FF8AD8"),
             kpi("⚙️", "Sleep Efficiency", f"{sl['Efficiency'].mean():.1f}%", "asleep ÷ in bed", "#00E5A8", "#4DA6FF"),
             kpi("⏰", "Nights < 7h", f"{(sl['HoursAsleep'] < 7).mean() * 100:.0f}%", "short sleep", "#FF6B6B", "#FFC94D"),
             kpi("💤", "Naps / Split Sleep", f"{(sl['TotalSleepRecords'] > 1).sum()}", "nights with >1 record", "#FFC94D", "#FF8AD8")])
    m = da.merge(sl[["Id", "Date", "HoursAsleep", "TotalTimeInBed", "TimeAwakeInBed", "Efficiency"]], on=["Id", "Date"])
    c1, c2 = st.columns(2)
    with c1:
        fig = px.histogram(sl, x="HoursAsleep", nbins=30, color_discrete_sequence=["#7C4DFF"], title="📊 Hours Asleep per Night")
        fig.add_vline(x=7, line_dash="dash", line_color="#FF6B6B", annotation_text="7h")
        fig.add_vline(x=9, line_dash="dash", line_color="#00E5A8", annotation_text="9h")
        show(fig)
    with c2:
        w = sl.groupby("Weekday", observed=True)[["HoursAsleep"]].mean().reindex(WD).reset_index()
        fig = px.bar(w, x="Weekday", y="HoursAsleep", color="HoursAsleep", color_continuous_scale="Purp", title="📆 Avg Sleep by Weekday")
        show(fig)
    c3, c4 = st.columns(2)
    with c3:
        fig = scatter(m, "SedentaryHours", "HoursAsleep", opacity=.6, color_discrete_sequence=["#FF8AD8"],
                      title=f"🪑 Sedentary Hours vs Sleep (r = {corr_text(m['SedentaryHours'], m['HoursAsleep'])})")
        show(fig)
    with c4:
        fig = scatter(m, "TotalSteps", "HoursAsleep", opacity=.6, color_discrete_sequence=["#00E5A8"],
                      title=f"👟 Steps vs Sleep (r = {corr_text(m['TotalSteps'], m['HoursAsleep'])})")
        show(fig)
    c5, c6 = st.columns(2)
    with c5:
        s = ms["Stage"].value_counts().reset_index()
        s.columns = ["Stage", "Minutes"]
        fig = px.pie(s, names="Stage", values="Minutes", hole=.55, color="Stage",
                     color_discrete_map={"Asleep": "#7C4DFF", "Restless": "#FFC94D", "Awake": "#FF6B6B"}, title="🌙 Sleep Stage Mix (minute data)")
        show(fig)
    with c6:
        s = ms.groupby(["Hour", "Stage"]).size().reset_index(name="Minutes")
        fig = px.bar(s, x="Hour", y="Minutes", color="Stage", color_discrete_map={"Asleep": "#7C4DFF", "Restless": "#FFC94D", "Awake": "#FF6B6B"},
                     title="🕰️ Sleep Minutes by Clock Hour")
        show(fig)
    u = sl.groupby("User")["HoursAsleep"].mean().sort_values().reset_index()
    fig = px.bar(u, x="User", y="HoursAsleep", color="HoursAsleep", color_continuous_scale="Blues", title="👥 Avg Sleep per User")
    fig.add_hline(y=7, line_dash="dash", line_color="#FF6B6B")
    show(fig, 320)


def page_heart():
    hero("❤️ Heart Rate & METs", "Cardio intensity from per-second heart rate and minute-level METs.")
    if hd.empty:
        st.info("No heart-rate data for the current filter (only 14 of 33 users have it).")
        return
    kpi_row([kpi("👥", "Users with HR", hd["User"].nunique(), "of 33 users", "#7C4DFF", "#00E5A8"),
             kpi("❤️", "Avg Heart Rate", f"{hd['AvgHR'].mean():.0f} bpm", "daily average", "#FF6B6B", "#FFC94D"),
             kpi("📈", "Peak Heart Rate", f"{hd['MaxHR'].max():.0f} bpm", "highest reading", "#FFC94D", "#FF6B6B"),
             kpi("🛌", "Lowest Reading", f"{hd['MinHR'].min():.0f} bpm", "resting proxy", "#4DA6FF", "#7C4DFF"),
             kpi("⚡", "Avg METs", f"{mt['AvgMETs'].mean():.2f}" if len(mt) else "n/a", "1.0 = resting", "#00E5A8", "#4DA6FF")])
    c1, c2 = st.columns(2)
    with c1:
        h = hh.groupby("Hour").agg(Avg=("AvgHR", "mean"), Min=("MinHR", "min"), Max=("MaxHR", "max")).reset_index()
        fig = go.Figure()
        fig.add_scatter(x=h["Hour"], y=h["Max"], line=dict(width=0), showlegend=False)
        fig.add_scatter(x=h["Hour"], y=h["Min"], fill="tonexty", fillcolor="rgba(124,77,255,.25)", line=dict(width=0), name="min–max range")
        fig.add_scatter(x=h["Hour"], y=h["Avg"], line=dict(color="#FF6B6B", width=3), name="avg")
        fig.update_layout(title="🕒 Heart Rate by Hour of Day")
        show(fig)
    with c2:
        t = hd.groupby("Date")["AvgHR"].mean().reset_index()
        fig = px.area(t, x="Date", y="AvgHR", color_discrete_sequence=["#FF6B6B"], title="📅 Daily Avg Heart Rate Trend")
        show(fig)
    c3, c4 = st.columns(2)
    with c3:
        m = da.merge(hd[["Id", "Date", "AvgHR"]], on=["Id", "Date"])
        fig = scatter(m, "VeryActiveMinutes", "AvgHR", opacity=.6, color_discrete_sequence=["#FFC94D"], title="💥 Very-Active Minutes vs Avg HR")
        show(fig)
    with c4:
        fig = px.box(hd, x="User", y="AvgHR", color="User", color_discrete_sequence=PALETTE, title="📦 Avg HR per User")
        fig.update_layout(showlegend=False)
        show(fig)
    if len(mt):
        h2 = mt.pivot_table(index="Weekday", columns="Hour", values="AvgMETs", aggfunc="mean", observed=True).reindex(WD)
        fig = px.imshow(h2, aspect="auto", color_continuous_scale="Inferno", labels=dict(x="Hour", y="", color="METs"),
                        title="🔥 Average METs — weekday × hour")
        show(fig, 360)


def page_weight():
    hero("⚖️ Weight & BMI", "Only a few users log weight — a big engagement gap.")
    if wt.empty:
        st.info("No weight logs for the current filter (only 8 users log weight).")
        return
    kpi_row([kpi("👥", "Users Logging", wt["User"].nunique(), "of 33 users", "#7C4DFF", "#00E5A8"),
             kpi("🧾", "Entries", len(wt), "weight logs", "#4DA6FF", "#7C4DFF"),
             kpi("⚖️", "Avg Weight", f"{wt['WeightKg'].mean():.1f} kg", "", "#00E5A8", "#4DA6FF"),
             kpi("📐", "Avg BMI", f"{wt['BMI'].mean():.1f}", "", "#FFC94D", "#FF6B6B"),
             kpi("✍️", "Manual Entries", f"{wt['IsManualReport'].mean() * 100:.0f}%", "vs auto-synced", "#FF8AD8", "#7C4DFF")])
    fig = px.line(wt.sort_values("Date"), x="Date", y="WeightKg", color="User", markers=True, title="📉 Weight Over Time")
    show(fig)
    c1, c2 = st.columns(2)
    with c1:
        b = pd.cut(wt["BMI"], [0, 18.5, 25, 30, 100], labels=["Underweight", "Normal", "Overweight", "Obese"]).value_counts().reset_index()
        b.columns = ["Category", "Entries"]
        fig = px.bar(b, x="Category", y="Entries", color="Category", color_discrete_sequence=["#4DA6FF", "#00E5A8", "#FFC94D", "#FF6B6B"], title="🧍 BMI Categories")
        fig.update_layout(showlegend=False)
        show(fig)
    with c2:
        wl = wt.sort_values("Date").groupby("User")["WeightKg"].agg(lambda s: s.iloc[-1] - s.iloc[0]).reset_index(name="Change (kg)")
        fig = px.bar(wl, x="User", y="Change (kg)", color="Change (kg)", color_continuous_scale="RdYlGn_r", title="↕️ Weight Change (last − first log)")
        show(fig)
    u = da.groupby("User")["TotalSteps"].mean().reset_index().merge(wt.groupby("User")["BMI"].mean().reset_index(), on="User")
    if len(u) > 1:
        fig = px.scatter(u, x="TotalSteps", y="BMI", text="User", size_max=18, color_discrete_sequence=["#00E5A8"], title="👟 Avg Steps vs Avg BMI (users who log weight)")
        fig.update_traces(textposition="top center", marker=dict(size=14))
        show(fig, 340)


def page_user():
    hero("👤 User Explorer", "Drill into one person's activity profile.")
    u = st.selectbox("Choose a user", all_users, format_func=lambda x: f"{x} · {[k for k, v in UMAP.items() if v == x][0]}")
    d, s, h, hb = (F(x, users=False) for x in (da0, sl0, hr0, hd0))
    d, s, h, hb = (x[x["User"] == u] for x in (d, s, h, hb))
    if d.empty:
        st.info("No data for this user in the chosen date range.")
        return
    kpi_row([kpi("📅", "Days Tracked", len(d), "in range"),
             kpi("👟", "Avg Steps", f"{d['TotalSteps'].mean():,.0f}", segment(d["TotalSteps"].mean()), "#FFC94D", "#FF6B6B"),
             kpi("🔥", "Avg Calories", f"{d['Calories'].mean():,.0f}", "kcal", "#FF6B6B", "#FF8AD8"),
             kpi("😴", "Avg Sleep", f"{s['HoursAsleep'].mean():.1f} h" if len(s) else "n/a", f"{len(s)} nights", "#4DA6FF", "#7C4DFF"),
             kpi("❤️", "Avg HR", f"{hb['AvgHR'].mean():.0f} bpm" if len(hb) else "n/a", "", "#FF6B6B", "#FFC94D")])
    pop = da0.groupby("User").agg(steps=("TotalSteps", "mean"), cal=("Calories", "mean"), very=("VeryActiveMinutes", "mean"),
                                  sed=("SedentaryHours", "mean"))
    pr = pop.rank(pct=True).loc[u] * 100
    pr["sed"] = 100 - pr["sed"]
    cats = ["Steps", "Calories", "Very-active min", "Low sedentary"]
    c1, c2 = st.columns([1, 2])
    with c1:
        fig = go.Figure(go.Scatterpolar(r=list(pr.values) + [pr.values[0]], theta=cats + [cats[0]], fill="toself",
                                        line=dict(color="#00E5A8"), fillcolor="rgba(0,229,168,.3)"))
        fig.update_layout(title="🕸️ Percentile vs all users", polar=dict(radialaxis=dict(range=[0, 100])))
        show(fig, 380)
    with c2:
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_bar(x=d["Date"], y=d["TotalSteps"], name="Steps", marker_color="rgba(124,77,255,.7)")
        fig.add_scatter(x=d["Date"], y=d["Calories"], name="Calories", line=dict(color="#FF6B6B", width=3), secondary_y=True)
        fig.update_layout(title="📈 Steps & Calories timeline")
        show(fig, 380)
    c3, c4 = st.columns(2)
    with c3:
        p = h.groupby("Hour")["StepTotal"].mean().reset_index()
        fig = px.area(p, x="Hour", y="StepTotal", color_discrete_sequence=["#FFC94D"], title="🕒 Typical day — avg steps by hour")
        show(fig, 320)
    with c4:
        if len(s):
            fig = px.bar(s, x="Date", y="HoursAsleep", color="HoursAsleep", color_continuous_scale="Purp", title="😴 Sleep per night")
            fig.add_hline(y=7, line_dash="dash", line_color="#FF6B6B")
            show(fig, 320)
        else:
            st.info("This user has no sleep records.")
    st.markdown("**Daily data**")
    table(d[["Date", "TotalSteps", "TotalDistance", "Calories", "VeryActiveMinutes", "SedentaryMinutes", "IsNonWearDay"]].sort_values("Date"), hide_index=True)


PRESETS = {
    "Avg steps & calories per user": "SELECT Id, ROUND(AVG(TotalSteps),0) AS avg_steps, ROUND(AVG(Calories),0) AS avg_calories\nFROM daily_activity GROUP BY Id ORDER BY avg_steps DESC;",
    "% of days meeting 10,000 steps": "SELECT ROUND(100.0*SUM(CASE WHEN TotalSteps>=10000 THEN 1 ELSE 0 END)/COUNT(*),1) AS pct_days_goal\nFROM daily_activity;",
    "Avg steps by weekday": "SELECT Weekday, ROUND(AVG(TotalSteps),0) AS avg_steps\nFROM daily_activity GROUP BY Weekday ORDER BY DayOfWeek;",
    "Peak activity hour": "SELECT Hour, ROUND(AVG(StepTotal),0) AS avg_steps\nFROM hourly GROUP BY Hour ORDER BY Hour;",
    "Sleep by sedentary bucket": "SELECT CASE WHEN a.SedentaryMinutes<600 THEN '1) <10h sedentary'\n            WHEN a.SedentaryMinutes<900 THEN '2) 10-15h sedentary' ELSE '3) 15h+ sedentary' END AS bucket,\n       ROUND(AVG(s.TotalMinutesAsleep)/60.0,2) AS avg_hours_asleep, COUNT(*) AS nights\nFROM daily_activity a JOIN sleep_day s ON a.Id=s.Id AND a.Date=s.Date\nGROUP BY bucket ORDER BY bucket;",
    "Sleep stage minutes": "SELECT CASE SleepStage WHEN 1 THEN 'Asleep' WHEN 2 THEN 'Restless' ELSE 'Awake' END AS stage, COUNT(*) AS minutes\nFROM minute_sleep GROUP BY SleepStage;",
    "Heart rate by hour": "SELECT Hour, ROUND(AVG(AvgHR),1) AS avg_hr FROM heartrate_hourly GROUP BY Hour ORDER BY Hour;",
    "Feature adoption": "SELECT (SELECT COUNT(DISTINCT Id) FROM daily_activity) AS activity_users,\n       (SELECT COUNT(DISTINCT Id) FROM sleep_day) AS sleep_users,\n       (SELECT COUNT(DISTINCT Id) FROM heartrate_daily) AS heart_users,\n       (SELECT COUNT(DISTINCT Id) FROM weight_log) AS weight_users;",
}


def page_sql():
    hero("🗄️ SQL Lab", "Run SQL directly on the cleaned data (in-memory SQLite).")
    conn = sql_conn()
    with st.expander("📚 Tables & columns"):
        for t in ["daily_activity", "sleep_day", "weight_log", "hourly", "heartrate_daily", "heartrate_hourly", "minute_sleep", "mets_hourly"]:
            cols = ", ".join(r[1] for r in conn.execute(f"PRAGMA table_info({t})"))
            st.markdown(f"**{t}** — `{cols}`")
    choice = st.selectbox("Preset queries", list(PRESETS))
    q = st.text_area("SQL", PRESETS[choice], height=170)
    if st.button("▶ Run query", type="primary"):
        try:
            res = pd.read_sql_query(q, conn)
            st.success(f"{len(res)} rows returned")
            table(res, hide_index=True)
            num = res.select_dtypes("number").columns.tolist()
            if len(res) > 1 and num and res.shape[1] >= 2:
                x = [c for c in res.columns if c not in num[:1]][0] if res.columns[0] in num[:1] and res.shape[1] > 1 else res.columns[0]
                fig = px.bar(res, x=x, y=num[0], color=num[0], color_continuous_scale="Viridis", title="Auto-chart")
                show(fig, 340)
            st.download_button("⬇ Download CSV", res.to_csv(index=False), "query_result.csv", "text/csv")
        except Exception as e:
            st.error(f"Query failed: {e}")


def page_insights():
    hero("💡 Insights & Recommendations", "Data-driven findings for the Bellabeat marketing team (update live with filters).")
    pct = (da["TotalSteps"] >= 10000).mean() * 100
    wk = da.groupby("Weekday", observed=True)["TotalSteps"].mean().reindex(WD)
    peak = hr.groupby("Hour")["StepTotal"].mean().idxmax()
    m = da.merge(sl[["Id", "Date", "HoursAsleep"]], on=["Id", "Date"])
    r = m["SedentaryHours"].corr(m["HoursAsleep"]) if len(m) > 2 else float("nan")
    cards = [
        ("#00E5A8", "🎯 Most days miss the 10k goal", f"Only {pct:.1f}% of user-days reach 10,000 steps (avg {da['TotalSteps'].mean():,.0f}).",
         "Offer personalised, progressive step goals instead of one flat target."),
        ("#FF8AD8", "🪑 The day is mostly sedentary", f"Users sit ≈ {da['SedentaryHours'].mean():.1f} h/day ({da['SedentaryMinutes'].mean() / 14.4:.0f}% of 24 h).",
         "Add gentle 'time to move' nudges after long inactive streaks."),
        ("#4DA6FF", "😴 Sleep falls short for many", f"Average sleep {sl['HoursAsleep'].mean():.1f} h; {(sl['HoursAsleep'] < 7).mean() * 100:.0f}% of nights are under 7 h." if len(sl) else "No sleep data in filter.",
         "Promote wind-down reminders and a sleep score in the app."),
        ("#FFC94D", "🔗 Sitting and sleep move together", f"Sedentary hours vs sleep correlation r = {r:.2f}." if r == r else "Not enough overlapping data.",
         "Bundle 'move more, sleep better' challenges."),
        ("#7C4DFF", "🕕 Activity has clear peak hours", f"Most steps happen around {peak}:00; lowest weekday is {wk.idxmin()} ({wk.min():,.0f} steps), highest {wk.idxmax()} ({wk.max():,.0f}).",
         f"Time push notifications around {peak}:00 and run weekend/low-day challenges."),
        ("#FF6B6B", "🔌 Feature adoption is uneven", f"Sleep: {sl['User'].nunique()} users · Heart rate: {hd['User'].nunique()} · Weight: {wt['User'].nunique()} out of {da['User'].nunique()} tracking activity.",
         "Simplify onboarding and prompt users to enable HR, sleep and weight logging."),
        ("#00B8D4", "📴 Some days have no data", f"{int(da['IsNonWearDay'].sum())} non-wear days ({da['IsNonWearDay'].mean() * 100:.1f}% of user-days).",
         "Send 'wear your tracker' and battery reminders to keep the streak alive."),
    ]
    for c, t, f, rec in cards:
        st.markdown(f'<div class="insight" style="--c:{c}"><h4>{t}</h4><div>{f}</div><div class="rec">✅ Recommendation: {rec}</div></div>',
                    unsafe_allow_html=True)


def page_cleaning():
    hero("🧼 Data Cleaning Summary", "What was done to every raw file before analysis.")
    rows = [("dailyActivity", 940, 940, 0, "Parsed dates, flagged 4 non-wear days, added Weekday & TotalActiveMinutes"),
            ("dailySteps", 940, 940, 0, "Parsed dates; 77 zero-step days kept"),
            ("dailyCalories", 940, 940, 0, "Parsed dates; 4 zero-calorie days kept"),
            ("dailyIntensities", 940, 940, 0, "Parsed dates; minutes ≤ 1440 verified"),
            ("sleepDay", 413, 410, 3, "Removed duplicates, added TimeAwakeInBed"),
            ("weightLogInfo", 67, 67, 0, "Dropped 'Fat' (97% missing), boolean IsManualReport"),
            ("hourlySteps", 22099, 22099, 0, "Parsed AM/PM datetime"),
            ("hourlyCalories", 22099, 22099, 0, "Parsed AM/PM datetime"),
            ("hourlyIntensities", 22099, 22099, 0, "Parsed AM/PM datetime; intensity ≤ 3 verified"),
            ("minuteStepsNarrow", 1325580, 1325580, 0, "Parsed datetime; no nulls/negatives"),
            ("minuteCaloriesNarrow", 1325580, 1325580, 0, "Parsed datetime; no nulls/negatives"),
            ("minuteIntensitiesWide", 21645, 21645, 0, "60 minute columns validated (0–3)"),
            ("minuteMETsNarrow", 1325580, 1325580, 0, "Values are METs × 10 (divided by 10 in this app)"),
            ("minuteSleep", 188521, 187978, 543, "Removed duplicates; value → SleepStage (1 asleep, 2 restless, 3 awake)"),
            ("heartrate_seconds", 2483658, 2483658, 0, "Valid range 36–203 bpm; aggregated to hourly/daily for this app")]
    df = pd.DataFrame(rows, columns=["File", "Raw rows", "Clean rows", "Duplicates dropped", "What was done"])
    kpi_row([kpi("📁", "Files Cleaned", len(df), "raw datasets"), kpi("🧹", "Duplicates Removed", int(df["Duplicates dropped"].sum()), "rows", "#FF6B6B", "#FFC94D"),
             kpi("👥", "Users", 33, "Fitbit participants", "#4DA6FF", "#7C4DFF")])
    table(df, hide_index=True)
    st.markdown("**Common steps:** `Id` → text · dates → datetime · duplicates removed · null / negative / range checks · derived columns added.")


{"🏠 Home": page_home, "👟 Steps & Distance": page_steps, "🔥 Calories & Intensity": page_cal,
 "🪑 Sedentary Behaviour": page_sed, "😴 Sleep Lab": page_sleep, "❤️ Heart Rate & METs": page_heart,
 "⚖️ Weight & BMI": page_weight, "👤 User Explorer": page_user, "🗄️ SQL Lab": page_sql,
 "💡 Insights": page_insights, "🧼 Data Cleaning": page_cleaning}[page]()