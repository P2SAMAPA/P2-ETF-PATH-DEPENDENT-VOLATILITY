# 〰️ P2-ETF-PATH-DEPENDENT-VOLATILITY

**Path-Dependent Volatility Engine — Guyon & Lekeufack (2023) PDV Framework**

Part of the **P2Quant Engine Suite** · [P2SAMAPA](https://github.com/P2SAMAPA)

---

## What This Engine Does

This engine implements the **Path-Dependent Volatility (PDV)** model from Guyon
& Lekeufack (2023), which demonstrated that realised volatility is *mostly
path-dependent*: the recent sequence of returns — not just their variance —
predicts future volatility better than GARCH or rough-vol models at short horizons.

### Why PDV for ETF Ranking?

Standard vol models (GARCH, EWMA) treat volatility as a function of recent
squared returns. PDV additionally captures:

1. **The leverage effect via the return path** — not just the sign of today's return,
   but the weighted sequence of recent returns (K₁)
2. **Volatility clustering via the squared-return path** — the weighted history of
   realised variance (K₂)

ETFs where PDV forecasts *lower* vol than the EWMA benchmark are flagged as
potentially oversold (vol spike has already occurred). ETFs where the recent
return path shows a leverage-induced vol compression are contrarian buys.

---

## Theory

### PDV Model

```
σ²(t) ≈ a₀  +  a₁·K₁(r, t)  +  a₂·K₂(r², t)
```

**K₁ — Signed path kernel (leverage effect):**
```
K₁(r, t) = Σⱼ₌₀ᵏ¹⁻¹  w₁ⱼ · r(t−j)
w₁ⱼ = (1−α₁) · α₁ʲ       (fast exponential decay, α₁=0.85)
```

**K₂ — Squared path kernel (vol clustering):**
```
K₂(r², t) = Σⱼ₌₀ᵏ²⁻¹  w₂ⱼ · r²(t−j)
w₂ⱼ = (1−α₂) · α₂ʲ       (slow exponential decay, α₂=0.97)
```

| Kernel | Lag k | Decay α | Captures |
|--------|-------|---------|---------|
| K₁ | 5d | 0.85 | Leverage effect — recent drawdowns raise vol |
| K₂ | 63d | 0.97 | Vol clustering — high past variance persists |

**Key empirical results (Guyon & Lekeufack 2023):**
- K₁ coefficient a₁ is **negative**: recent losses → higher future vol
- K₂ coefficient a₂ is **positive**: recent variance → higher future vol
- PDV outperforms GARCH(1,1) and rough vol at horizons of 1–5 days

---

## Score Construction

```
score_i = 0.45 · pdv_ratio_i  +  0.35 · path_sign_i  +  0.20 · vol_surprise_i
```

| Component | Formula | Interpretation |
|-----------|---------|----------------|
| **PDV ratio** | −(PDV_vol/EWMA_vol − 1) | PDV < EWMA → vol cooling → **positive** |
| **Path sign** | −sign(K₁) | Recent drawdown (negative K₁) → contrarian **positive** |
| **Vol surprise** | (EWMA − PDV) / EWMA | EWMA overstates → mean reversion → **positive** |

Final score: **cross-sectional z-score** per universe per window.

---

## Hyperparameters

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `PDV_K1` | 5d | Fast kernel lag (leverage effect) |
| `PDV_K2` | 63d | Slow kernel lag (vol clustering) |
| `PDV_ALPHA1` | 0.85 | Fast kernel decay rate |
| `PDV_ALPHA2` | 0.97 | Slow kernel decay rate |
| `WEIGHT_PDV_RATIO` | 0.45 | PDV/EWMA ratio weight |
| `WEIGHT_PATH_SIGN` | 0.35 | Path momentum sign weight |
| `WEIGHT_VOL_SURPRISE` | 0.20 | Vol surprise weight |

---

## Universes

| Universe | Tickers |
|---|---|
| FI_COMMODITIES | TLT, VCIT, LQD, HYG, VNQ, GLD, SLV |
| EQUITY_SECTORS | SPY, QQQ, XLK, XLF, XLE, XLV, XLI, XLY, XLP, XLU, GDX, XME, IWF, XSD, XBI, IWM, IWD, IWO, XLB, XLRE |
| COMBINED | All of the above |

## Rolling Windows

```
63d · 126d · 252d · 504d
```

---

## Repository Structure

```
P2-ETF-PATH-DEPENDENT-VOLATILITY/
├── config.py          # Universes, PDV hyperparameters, score weights
├── data_manager.py    # HuggingFace loader → price DataFrames
├── pdv_engine.py      # Core PDV: path kernels K₁, K₂ → score components
├── trainer.py         # Orchestrator: load → score → build JSON → upload
├── push_results.py    # HfApi.upload_file wrapper
├── streamlit_app.py   # Two-tab Streamlit dashboard
├── us_calendar.py     # US trading calendar helper
├── requirements.txt
└── .github/
    └── workflows/
        └── daily.yml  # Scheduled run 23:30 UTC Mon–Fri
```

---

## Output JSON Schemas

### Tab 1 — `pdv_engine_YYYY-MM-DD.json`

```json
{
  "run_date": "2026-06-18",
  "universes": {
    "FI_COMMODITIES": {
      "top_etfs": [
        {"ticker": "GLD", "pdv_score": 1.45, "best_window": 63}
      ],
      "full_scores": {
        "GLD": {"score": 1.45, "best_window": 63}
      }
    }
  }
}
```

### Tab 2 — `pdv_engine_windows_YYYY-MM-DD.json`

```json
{
  "run_date": "2026-06-18",
  "universes": {
    "FI_COMMODITIES": {
      "windows": {
        "63":  {"top_etfs": [...], "full_ranking": [["GLD", 1.45], ...]},
        "252": {"top_etfs": [...], "full_ranking": [...]}
      }
    }
  }
}
```

---

## Setup

```bash
git clone https://github.com/P2SAMAPA/P2-ETF-PATH-DEPENDENT-VOLATILITY
cd P2-ETF-PATH-DEPENDENT-VOLATILITY
pip install -r requirements.txt

export HF_TOKEN=hf_...
python trainer.py
streamlit run streamlit_app.py
```

**Required GitHub secret:** `HF_TOKEN`

**Required HuggingFace dataset repo:** `P2SAMAPA/p2-etf-pdv-results`

---

## Relationship to Other Vol Engines

| Engine | Vol Model | Horizon | Key feature |
|--------|-----------|---------|-------------|
| ROUGH-VOL | Rough Bergomi / fBM | Medium | Fractional Brownian motion, Hurst H<0.5 |
| ROUGH-PATH | Signature features | Medium | Path signatures on return stream |
| **PDV** | Path-dependent kernels | **Short (1–5d)** | **Return path K₁ + squared path K₂** |
| HAR-RV | Heterogeneous AR | Daily | Realised variance at multiple horizons |
| JUMP-DIFFUSION | Merton jump-diffusion | Event-driven | Discontinuous price jumps |

PDV is the empirically strongest at **short horizons (1–5 days)** and is the
most recently validated (2023), making it genuinely novel vs. the rest of the suite.

---

## References

- Guyon, J. & Lekeufack, J. (2023). Volatility is (mostly) path-dependent.
  *Quantitative Finance*, 23(9), 1221–1258.
- Gatheral, J., Jaisson, T. & Rosenbaum, M. (2018). Volatility is rough.
  *Quantitative Finance*, 18(6), 933–949.
- Cont, R. (2001). Empirical properties of asset returns: stylized facts and
  statistical issues. *Quantitative Finance*, 1(2), 223–236.
- Black, F. (1976). Studies of stock price volatility changes. Proceedings of
  the 1976 American Statistical Association.
