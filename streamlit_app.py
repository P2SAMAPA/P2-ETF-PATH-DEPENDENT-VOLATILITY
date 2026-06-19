import streamlit as st
import pandas as pd
import json
from huggingface_hub import HfFileSystem
import config
from us_calendar import next_trading_day

st.set_page_config(page_title="Path-Dependent Volatility Engine", layout="wide")

st.markdown("""
<style>
.main-header { font-size:2.4rem; font-weight:700; color:#4a235a; margin-bottom:0.3rem; }
.sub-header  { font-size:1.1rem; color:#555; margin-bottom:1.5rem; }
.uni-title   { font-size:1.4rem; font-weight:600; margin-top:1rem; margin-bottom:0.8rem;
               padding-left:0.5rem; border-left:5px solid #7d3c98; }
.etf-card    { background:linear-gradient(135deg,#4a235a 0%,#7d3c98 100%); color:white;
               border-radius:14px; padding:1rem; margin:0.4rem; text-align:center;
               box-shadow:0 4px 6px rgba(0,0,0,0.2); }
.win-card    { background:linear-gradient(135deg,#1a5276 0%,#4a235a 100%); color:white;
               border-radius:14px; padding:1rem; margin:0.4rem; text-align:center;
               box-shadow:0 4px 6px rgba(0,0,0,0.2); }
.etf-ticker  { font-size:1.3rem; font-weight:bold; }
.etf-score   { font-size:0.88rem; margin-top:0.25rem; opacity:0.9; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">〰️ Path-Dependent Volatility Engine</div>',
            unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Guyon & Lekeufack (2023) PDV model · '
    'Leverage effect (K₁) + Vol clustering (K₂) · '
    'Exponential path kernels · Multi-window cross-sectional z-score ranking</div>',
    unsafe_allow_html=True)

st.sidebar.markdown("## 〰️ PDV Engine")
st.sidebar.markdown(f"**Next Trading Day:** `{next_trading_day()}`")
st.sidebar.markdown(f"**Windows:** {config.WINDOWS}")
st.sidebar.markdown(
    f"**K₁ lag:** {config.PDV_K1}d (α={config.PDV_ALPHA1}) | "
    f"**K₂ lag:** {config.PDV_K2}d (α={config.PDV_ALPHA2})")
st.sidebar.markdown(
    f"**Weights:** PDV ratio {config.WEIGHT_PDV_RATIO:.0%} | "
    f"Path sign {config.WEIGHT_PATH_SIGN:.0%} | "
    f"Vol surprise {config.WEIGHT_VOL_SURPRISE:.0%}")

HF_TOKEN    = config.HF_TOKEN
OUTPUT_REPO = config.OUTPUT_REPO


@st.cache_data(ttl=3600)
def list_repo_files():
    fs = HfFileSystem(token=HF_TOKEN)
    try:
        return [f["name"] for f in fs.ls(f"datasets/{OUTPUT_REPO}",
                                          detail=True, recursive=True)
                if f["type"] == "file"]
    except Exception as e:
        return [f"Error: {e}"]


def find_latest(files, prefix):
    matches = sorted([f for f in files if f.endswith(".json") and prefix in f],
                     reverse=True)
    return matches[0] if matches else None


@st.cache_data(ttl=3600)
def load_json(path):
    fs = HfFileSystem(token=HF_TOKEN)
    try:
        with fs.open(path, "r") as f:
            return json.load(f)
    except Exception as e:
        return {"error": str(e)}


files     = list_repo_files()
tab1_path = find_latest(files, "pdv_engine_2")
tab2_path = find_latest(files, "pdv_engine_windows_")

if not tab1_path:
    st.error("No results found. Run trainer.py first.")
    st.stop()

data1 = load_json(tab1_path)
if "error" in data1:
    st.error(f"Error loading data: {data1['error']}")
    st.stop()

data2      = load_json(tab2_path) if tab2_path else None
universes1 = data1["universes"]
universes2 = data2["universes"] if data2 and "error" not in data2 else None

st.sidebar.markdown(f"**Run date:** `{data1.get('run_date','?')}`")

tab1, tab2 = st.tabs(["🏆 Best Window per ETF", "🔍 Explore by Window"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.header("🏆 Top ETFs — Path-Dependent Volatility Signal")

    with st.expander("📖 PDV Methodology (Guyon & Lekeufack 2023)", expanded=True):
        st.markdown("""
Guyon & Lekeufack (2023) showed that realised volatility is **mostly path-dependent**:
the recent sequence of returns, not just their statistical moments, predicts future vol.

**PDV model:**
```
σ²(t) ≈ a₀  +  a₁·K₁(r,t)  +  a₂·K₂(r²,t)
```

| Kernel | Formula | Economic meaning |
|--------|---------|-----------------|
| **K₁** (fast, 5d) | Σⱼ w₁ⱼ·r(t−j) | **Leverage effect**: recent losses → higher vol |
| **K₂** (slow, 63d) | Σⱼ w₂ⱼ·r²(t−j) | **Vol clustering**: recent high variance → high vol |

**Score components:**

| Component | Weight | Signal logic |
|-----------|--------|-------------|
| PDV ratio (PDV/EWMA) | 45% | PDV < EWMA → vol cooling → **buy** |
| Path sign (−sign K₁) | 35% | Recent drawdown → contrarian **buy** |
| Vol surprise (EWMA−PDV)/EWMA | 20% | PDV underpredicts → **caution** |

All components are cross-sectionally z-scored per universe per window.
        """)

    for universe_name, uni_data in universes1.items():
        top_etfs = uni_data.get("top_etfs", [])
        if not top_etfs:
            continue
        st.markdown(
            f'<div class="uni-title">{universe_name.replace("_", " ").title()}</div>',
            unsafe_allow_html=True)
        cols = st.columns(3)
        for idx, etf in enumerate(top_etfs):
            with cols[idx]:
                st.markdown(f"""
<div class="etf-card">
  <div class="etf-ticker">{etf['ticker']}</div>
  <div class="etf-score">PDV score = {etf['pdv_score']:.4f}</div>
  <div class="etf-score">best window = {etf.get('best_window','N/A')}d</div>
</div>
""", unsafe_allow_html=True)

        with st.expander(f"📋 Full ranking — {universe_name}"):
            full = uni_data.get("full_scores", {})
            if full:
                rows = []
                for t, info in full.items():
                    score = info.get("score", info) if isinstance(info, dict) else info
                    win   = info.get("best_window", "N/A") if isinstance(info, dict) else "N/A"
                    rows.append({"ETF": t, "PDV Score": score, "Best Window (d)": win})
                df = pd.DataFrame(rows).sort_values("PDV Score", ascending=False)
                st.dataframe(df, use_container_width=True, hide_index=True)
        st.divider()

    st.caption(
        f"Run date: {data1.get('run_date','?')} · "
        "Guyon & Lekeufack (2023) PDV framework · "
        "Scores are cross-sectional z-scores.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.header("🔍 Explore PDV Rankings by Window")

    if not universes2:
        st.warning("Window-level detail not found. Re-run trainer to generate "
                   "`pdv_engine_windows_<date>.json`.")
        st.stop()

    all_wins = set()
    for ud in universes2.values():
        all_wins.update(ud.get("windows", {}).keys())
    win_options = sorted([int(w) for w in all_wins])

    if not win_options:
        st.error("No window data available.")
        st.stop()

    default_idx  = win_options.index(63) if 63 in win_options else 0
    selected_win = st.selectbox(
        "Select lookback window",
        options=win_options,
        index=default_idx,
        format_func=lambda w: f"{w}d  (~{round(w/21)} months)",
    )
    win_key = str(selected_win)

    with st.expander("ℹ️ Window guidance", expanded=False):
        st.markdown("""
- **63d** — primary PDV signal window; K₁ and K₂ kernels operate at short lags
- **126d** — 6-month window; balances path memory with recent regime
- **252d** — 1-year window; captures full vol cycle; most stable PDV estimates
- **504d** — 2-year window; structural path dependencies; slow leverage mean-reversion
        """)

    st.markdown(f"### PDV Rankings at **{selected_win}d** window")

    for universe_name in ["FI_COMMODITIES", "EQUITY_SECTORS", "COMBINED"]:
        label = {
            "FI_COMMODITIES": "🏦 FI & Commodities",
            "EQUITY_SECTORS": "📈 Equity Sectors",
            "COMBINED":       "🌐 Combined",
        }.get(universe_name, universe_name)

        st.markdown(f'<div class="uni-title">{label}</div>', unsafe_allow_html=True)

        uni_data = universes2.get(universe_name, {})
        win_data = uni_data.get("windows", {}).get(win_key)

        if not win_data:
            st.info(f"No data for {universe_name} at {selected_win}d.")
            st.divider()
            continue

        cols = st.columns(3)
        for idx, etf in enumerate(win_data.get("top_etfs", [])):
            with cols[idx]:
                st.markdown(f"""
<div class="win-card">
  <div class="etf-ticker">{etf['ticker']}</div>
  <div class="etf-score">PDV score = {etf['pdv_score']:.4f}</div>
  <div class="etf-score">window = {selected_win}d</div>
</div>
""", unsafe_allow_html=True)

        with st.expander(f"📋 Full ranking — {label} @ {selected_win}d"):
            rows = win_data.get("full_ranking", [])
            if rows:
                df = pd.DataFrame(rows, columns=["ETF", "PDV Score"])
                df.insert(0, "Rank", range(1, len(df) + 1))
                st.dataframe(df, use_container_width=True, hide_index=True)

        st.divider()

    st.caption(f"Window: {selected_win}d · Run date: {data2.get('run_date','?')}")
