import os

HF_TOKEN    = os.environ.get("HF_TOKEN", "")
DATA_REPO   = "P2SAMAPA/fi-etf-macro-signal-master-data"
OUTPUT_REPO = "P2SAMAPA/p2-etf-pdv-results"

UNIVERSES = {
    "FI_COMMODITIES": ["TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV"],
    "EQUITY_SECTORS": [
        "SPY", "QQQ", "XLK", "XLF", "XLE", "XLV", "XLI", "XLY",
        "XLP", "XLU", "GDX", "XME", "IWF", "XSD", "XBI",
        "IWM", "IWD", "IWO", "XLB", "XLRE",
    ],
    "COMBINED": [
        "TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV",
        "SPY", "QQQ", "XLK", "XLF", "XLE", "XLV", "XLI", "XLY",
        "XLP", "XLU", "GDX", "XME", "IWF", "XSD", "XBI",
        "IWM", "IWD", "IWO", "XLB", "XLRE",
    ],
}

# ── Rolling windows (trading days) ────────────────────────────────────────────
WINDOWS = [63, 126, 252, 504]

# ── PDV model hyperparameters (Guyon & Lekeufack 2023) ────────────────────────

# Short-term kernel lag (k1): recent return path lookback
# Captures the immediate leverage/momentum effect
PDV_K1 = 5          # trading days (~1 week)

# Long-term kernel lag (k2): slow-moving variance path lookback
# Captures the vol-of-vol clustering effect
PDV_K2 = 63         # trading days (~1 quarter)

# Exponential kernel decay rates for weighting past returns
# alpha1: fast decay (more weight on very recent returns)
# alpha2: slow decay (spread more evenly over past)
PDV_ALPHA1 = 0.85   # fast kernel
PDV_ALPHA2 = 0.97   # slow kernel

# Minimum observations required within a window to compute PDV
MIN_OBS = 30

# ── Score construction ────────────────────────────────────────────────────────
# PDV score = weighted combination of:
#   pdv_ratio    : PDV-predicted vol / realised vol  (low = oversold, high = overbought)
#   path_sign    : sign of recent return path momentum (long/short signal)
#   vol_surprise : diff between PDV forecast and simple EWMA vol (regime signal)
WEIGHT_PDV_RATIO    = 0.45
WEIGHT_PATH_SIGN    = 0.35
WEIGHT_VOL_SURPRISE = 0.20

TOP_N = 3
