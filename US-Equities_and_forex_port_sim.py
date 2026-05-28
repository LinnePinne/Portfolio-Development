import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ta
from itertools import product
import copy
from dataclasses import dataclass
from math import sqrt

plt.style.use("default")

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)
pd.set_option("display.max_colwidth", None)
pd.set_option("display.max_rows", 200)

# ============================================================
# GLOBAL CONFIG
# ============================================================

ACCOUNT_CCY = "USD"
START_CAPITAL = 50000.0
RECOVERY_REFERENCE_BALANCE = 50000.0

MAX_GROSS_EXPOSURE_TOTAL = 12.5

STRATEGY_TARGETS = {
    "TF_EQ": 1.125,
    "MR_EQ": 0.875,
    "TF_FX": 2.5,
    "MR_FX": 8.0,
}

TF_EQ_MARKET_WEIGHTS = {"US500": 0.34, "US100": 0.29, "US30": 0.37}
MR_EQ_MARKET_WEIGHTS = {"US500": 1/3,  "US100": 1/3,  "US30": 1/3}

TF_FX_MARKET_WEIGHTS = {
    "EURJPY": 0.3567,
    "GBPJPY": 0.2456,
    "USDJPY": 0.3977,
}

MR_FX_MARKET_WEIGHTS = {
    "EURCHF": 0.229429,
    "EURCAD": 0.205126,
    "GBPCHF": 0.196246,
    "EURUSD": 0.190355,
    "USDCHF": 0.178844,
}

STRATEGY_MARKET_WEIGHTS = {
    "TF_EQ": TF_EQ_MARKET_WEIGHTS,
    "MR_EQ": MR_EQ_MARKET_WEIGHTS,
    "TF_FX": TF_FX_MARKET_WEIGHTS,
    "MR_FX": MR_FX_MARKET_WEIGHTS,
}

# ============================================================
# FTMO / PIPELINE SETTINGS
# ============================================================

EVAL_EXPOSURE_FACTOR = 0.75
FUNDED_EXPOSURE_FACTOR = 0.75

SOFT_CUTOFF_DAILY = -0.045

CHALLENGE_TARGET = 0.10
VERIFICATION_TARGET = 0.05
DAILY_LOSS_LIMIT = 0.05
MAX_LOSS_LIMIT = 0.10
MIN_TRADING_DAYS_EVAL = 4
PAYOUT_WAIT_DAYS = 14

BOOT_N_ITER = 20000
BOOT_BLOCK_LEN = 20
BOOT_HORIZON_DAYS = 365 * 3
BOOT_SEED = 42

# ============================================================
# EQUITY INDEX DATA CONFIG
# ============================================================

EQUITY_MARKETS = [
    {"name": "US500", "csv_1h": "US500_1H_2012-2025.csv", "csv_1d": "US500_1D_2012-2025.csv"},
    {"name": "US100", "csv_1h": "US100_1H_2012-2025.csv", "csv_1d": "US100_1D_2012-2025.csv"},
    {"name": "US30",  "csv_1h": "US30_1H_2012-2025.csv",  "csv_1d": "US30_1D_2012-2025.csv"},
]

POINT_VALUE = {"US500": 1.0, "US100": 1.0, "US30": 1.0}

HALF = 0.5

# Equity index cost model (points)
EQ_SLIPPAGE_POINTS = 0.5
EQ_FIXED_SPREAD_POINTS = 0.8
EQ_COMM_POINTS_PER_SIDE = 0.05

def eq_comm_round_turn_points():
    return 2.0 * EQ_COMM_POINTS_PER_SIDE

# ============================================================
# MARKET-SPECIFIC VOL OVERLAY SETTINGS
# ============================================================

USE_REGIME_OVERLAY = True
REGIME_LOOKBACK_DAYS = 20

# Ny struktur:
# (Regime, Strategy, Market) -> multiplier
REGIME_MARKET_MULTIPLIERS = {
    # ExtremeVol overlays
    ("ExtremeVol", "TF_EQ", "US100"): 0.50,
    ("ExtremeVol", "TF_EQ", "US500"): 1.00,
    ("ExtremeVol", "TF_EQ", "US30"): 1.00,

    ("ExtremeVol", "MR_FX", "USDCHF"): 1.00,
    ("ExtremeVol", "MR_FX", "EURUSD"): 1.00,
    ("ExtremeVol", "MR_FX", "EURCHF"): 1.00,
    ("ExtremeVol", "MR_FX", "GBPCHF"): 1.00,
    ("ExtremeVol", "MR_FX", "EURCAD"): 0.65,

    # LowVol overlays för TF_FX
    ("LowVol", "TF_FX", "EURJPY"): 0.35,
    ("LowVol", "TF_FX", "GBPJPY"): 0.35,
    ("LowVol", "TF_FX", "USDJPY"): 1.00,
}
# ============================================================
# FX CONFIG
# ============================================================

# ---------- TF FX ----------
FX_TF_MARKETS = [
    {
        "name": "USDJPY",
        "csv": "USDJPY_1H_2012-2026.csv",
        "pip_size": 0.01,
        "spread_points_per_pip": 10.0,
        "cost_model": {"slippage_pips": 0.12, "fixed_spread_pips": 0.45, "comm_pips_per_side": 0.25},
    },
    {
        "name": "EURJPY",
        "csv": "EURJPY_1H_2012-2026.csv",
        "pip_size": 0.01,
        "spread_points_per_pip": 10.0,
        "cost_model": {"slippage_pips": 0.35, "fixed_spread_pips": 0.90, "comm_pips_per_side": 0.25},
    },
    {
        "name": "GBPJPY",
        "csv": "GBPJPY_1H_2012-2026.csv",
        "pip_size": 0.01,
        "spread_points_per_pip": 10.0,
        "cost_model": {"slippage_pips": 0.50, "fixed_spread_pips": 1.20, "comm_pips_per_side": 0.25},
    },
]

# ---------- MR FX ----------
MR_FX_SESSION_START = "00:00:00"
MR_FX_SESSION_END = "07:00:00"
MR_FX_VWAP_RESET = "20:00:00"
MR_FX_ENTRY_STD = 2.25
MR_FX_EXIT_STD = 0.75
MR_FX_COMM_PIPS_PER_SIDE = 0.25

MR_FX_DEFAULT_COSTS = {"slippage_pips": 0.10, "spread_pips": 0.20}
MR_FX_COSTS_BY_MARKET = {
    "EURUSD": {"slippage_pips": 0.10, "spread_pips": 0.20},
    "GBPUSD": {"slippage_pips": 0.12, "spread_pips": 0.25},  # rate-only
    "USDCHF": {"slippage_pips": 0.10, "spread_pips": 0.22},
    "EURCHF": {"slippage_pips": 0.10, "spread_pips": 0.22},
    "GBPCHF": {"slippage_pips": 0.15, "spread_pips": 0.30},
    "EURCAD": {"slippage_pips": 0.15, "spread_pips": 0.30},
    "USDCAD": {"slippage_pips": 0.12, "spread_pips": 0.25},  # rate-only
}

MR_FX_PIP_SIZE = {
    "EURUSD": 0.0001,
    "USDCHF": 0.0001,
    "EURCHF": 0.0001,
    "GBPCHF": 0.0001,
    "EURCAD": 0.0001,
    "GBPUSD": 0.0001,  # rate-only
    "USDCAD": 0.0001,  # rate-only
}

MR_FX_TRADE_MARKETS = [
    {"name": "EURUSD", "csv": "EURUSD_1H_2012-2026.csv"},
    {"name": "USDCHF", "csv": "USDCHF_1H_2012-2026.csv"},
    {"name": "EURCHF", "csv": "EURCHF_1H_2012-2026.csv"},
    {"name": "GBPCHF", "csv": "GBPCHF_1H_2012-2026.csv"},
    {"name": "EURCAD", "csv": "EURCAD_1H_2012-2026.csv"},
]

FX_RATE_ONLY_MARKETS = [
    {"name": "GBPUSD", "csv": "GBPUSD_1H_2012-2026.csv"},
    {"name": "USDCAD", "csv": "USDCAD_1H_2012-2026.csv"},
]

# ============================================================
# INSTRUMENT SPECS
# ============================================================

@dataclass
class InstrumentSpec:
    symbol: str
    asset_class: str           # "INDEX" or "FX"
    base_ccy: str | None
    quote_ccy: str
    contract_size: float = 1.0

INSTRUMENTS = {
    # Indices
    "US500":  InstrumentSpec("US500", "INDEX", None, "USD", 1.0),
    "US100":  InstrumentSpec("US100", "INDEX", None, "USD", 1.0),
    "US30":   InstrumentSpec("US30",  "INDEX", None, "USD", 1.0),

    # FX traded
    "USDJPY": InstrumentSpec("USDJPY", "FX", "USD", "JPY", 1.0),
    "EURJPY": InstrumentSpec("EURJPY", "FX", "EUR", "JPY", 1.0),
    "GBPJPY": InstrumentSpec("GBPJPY", "FX", "GBP", "JPY", 1.0),

    "EURUSD": InstrumentSpec("EURUSD", "FX", "EUR", "USD", 1.0),
    "USDCHF": InstrumentSpec("USDCHF", "FX", "USD", "CHF", 1.0),
    "EURCHF": InstrumentSpec("EURCHF", "FX", "EUR", "CHF", 1.0),
    "GBPCHF": InstrumentSpec("GBPCHF", "FX", "GBP", "CHF", 1.0),
    "EURCAD": InstrumentSpec("EURCAD", "FX", "EUR", "CAD", 1.0),

    # FX rate-only
    "GBPUSD": InstrumentSpec("GBPUSD", "FX", "GBP", "USD", 1.0),
    "USDCAD": InstrumentSpec("USDCAD", "FX", "USD", "CAD", 1.0),
}

# ============================================================
# DATA HELPERS
# ============================================================

def load_market_df(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp")
    elif "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.set_index("datetime")
    else:
        raise ValueError(f"Hittar ingen 'timestamp' eller 'datetime' i {csv_path}")

    df = df.sort_index()

    required_cols = {"open", "high", "low", "close"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"{csv_path} måste innehålla: {required_cols}")

    if df.index.has_duplicates:
        agg_map = {"open": "first", "high": "max", "low": "min", "close": "last"}
        if "volume" in df.columns:
            agg_map["volume"] = "sum"
        elif "tick_volume" in df.columns:
            agg_map["tick_volume"] = "sum"
        df = df.groupby(df.index).agg(agg_map).sort_index()

    return df

def normalize_weights(w: dict) -> dict:
    s = float(sum(w.values()))
    if s <= 0:
        raise ValueError("Weights summerar till 0.")
    return {k: float(v) / s for k, v in w.items()}

def parse_fx_symbol(symbol: str) -> tuple[str, str]:
    return symbol[:3], symbol[3:]

# ============================================================
# FX CONVERSION
# ============================================================

def get_close_asof(close_matrix: pd.DataFrame, symbol: str, ts: pd.Timestamp) -> float:
    if symbol not in close_matrix.columns:
        raise ValueError(f"Saknar close-serie för {symbol}")
    val = close_matrix[symbol].asof(ts)
    if pd.isna(val):
        raise ValueError(f"Saknar {symbol}-pris vid {ts}")
    return float(val)

def convert_to_usd(amount: float, ccy: str, ts: pd.Timestamp, close_matrix: pd.DataFrame) -> float:
    if ccy == "USD":
        return float(amount)

    direct = f"{ccy}USD"
    inverse = f"USD{ccy}"

    if direct in close_matrix.columns:
        px = get_close_asof(close_matrix, direct, ts)
        return float(amount) * px

    if inverse in close_matrix.columns:
        px = get_close_asof(close_matrix, inverse, ts)
        return float(amount) / px

    raise ValueError(f"Saknar FX-konvertering {ccy}->USD vid {ts}")

def usd_per_base_unit(symbol: str, ts: pd.Timestamp, close_matrix: pd.DataFrame) -> float:
    spec = INSTRUMENTS[symbol]
    if spec.asset_class == "INDEX":
        px = get_close_asof(close_matrix, symbol, ts)
        pv = float(POINT_VALUE[symbol])
        return px * pv

    return convert_to_usd(1.0, spec.base_ccy, ts, close_matrix)

def quote_to_usd(amount_quote: float, quote_ccy: str, ts: pd.Timestamp, close_matrix: pd.DataFrame) -> float:
    return convert_to_usd(amount_quote, quote_ccy, ts, close_matrix)

# ============================================================
# STANDARDIZED TRADE HELPERS
# ============================================================

def make_trade_row(
    market: str,
    strategy: str,
    direction: str,
    entry_signal_time,
    entry_fill_time,
    exit_fill_time,
    entry_price: float,
    exit_price: float,
    exit_reason: str,
    comm_rt_price: float,
) -> dict:
    spec = INSTRUMENTS[market]
    return {
        "Market": market,
        "Strategy": strategy,
        "AssetClass": spec.asset_class,
        "BaseCcy": spec.base_ccy,
        "QuoteCcy": spec.quote_ccy,
        "Direction": direction.upper(),
        "Entry Signal Time": pd.to_datetime(entry_signal_time) if entry_signal_time is not None else pd.NaT,
        "Entry Fill Time": pd.to_datetime(entry_fill_time),
        "Exit Fill Time": pd.to_datetime(exit_fill_time),
        "Entry Price": float(entry_price),
        "Exit Price": float(exit_price),
        "Exit Reason": exit_reason,
        "Comm RT Price": float(comm_rt_price),
    }

# ============================================================
# EQUITY TF (1H)
# ============================================================

def generate_trades_tf_eq(
    market_name: str,
    df: pd.DataFrame,
    exit_confirm_bars: int = 10,
    adx_threshold: float = 15,
    ema_fast_len: int = 70,
    ema_slow_len: int = 120,
) -> pd.DataFrame:
    df = df.copy()

    df["ema_fast"] = df["close"].ewm(span=ema_fast_len, adjust=False).mean()
    df["ema_slow"] = df["close"].ewm(span=ema_slow_len, adjust=False).mean()

    df["adx"] = ta.trend.ADXIndicator(
        high=df["high"], low=df["low"], close=df["close"],
        window=14, fillna=False
    ).adx()

    use_spread_col = "spread_points" in df.columns

    def spread_points(row) -> float:
        return float(row["spread_points"]) if use_spread_col else EQ_FIXED_SPREAD_POINTS

    trades = []

    in_position = False
    entry_price = None
    entry_signal_time = None
    entry_fill_time = None
    exit_breach_count = 0

    idx = df.index.to_list()

    for i in range(1, len(df) - 1):
        ts = idx[i]
        row = df.iloc[i]
        prev_row = df.iloc[i - 1]
        next_row = df.iloc[i + 1]

        if np.isnan(row["adx"]) or np.isnan(row["ema_slow"]) or np.isnan(prev_row["ema_slow"]):
            continue

        # Exit
        if in_position:
            if row["ema_fast"] <= row["ema_slow"]:
                exit_breach_count += 1
            else:
                exit_breach_count = 0

            if exit_breach_count >= exit_confirm_bars:
                spr = spread_points(next_row)
                exit_fill_price = float(next_row["open"] - HALF * spr - EQ_SLIPPAGE_POINTS)

                trades.append(make_trade_row(
                    market=market_name,
                    strategy="TF_EQ",
                    direction="LONG",
                    entry_signal_time=entry_signal_time,
                    entry_fill_time=entry_fill_time,
                    exit_fill_time=idx[i + 1],
                    entry_price=entry_price,
                    exit_price=exit_fill_price,
                    exit_reason=f"ema_recross_{exit_confirm_bars}bars",
                    comm_rt_price=eq_comm_round_turn_points(),
                ))

                in_position = False
                entry_price = None
                entry_signal_time = None
                entry_fill_time = None
                exit_breach_count = 0

            if in_position:
                continue

        # Entry
        adx_ok = row["adx"] > adx_threshold
        cross_up = (prev_row["ema_fast"] < prev_row["ema_slow"]) and (row["ema_fast"] > row["ema_slow"])

        if cross_up and adx_ok:
            spr = spread_points(next_row)
            entry_fill_price = float(next_row["open"] + HALF * spr + EQ_SLIPPAGE_POINTS)

            in_position = True
            entry_signal_time = ts
            entry_fill_time = idx[i + 1]
            entry_price = entry_fill_price
            exit_breach_count = 0

    if in_position:
        last_row = df.iloc[-1]
        spr = float(last_row["spread_points"]) if "spread_points" in df.columns else EQ_FIXED_SPREAD_POINTS
        exit_fill_price = float(last_row["close"] - HALF * spr - EQ_SLIPPAGE_POINTS)

        trades.append(make_trade_row(
            market=market_name,
            strategy="TF_EQ",
            direction="LONG",
            entry_signal_time=entry_signal_time,
            entry_fill_time=entry_fill_time,
            exit_fill_time=df.index[-1],
            entry_price=entry_price,
            exit_price=exit_fill_price,
            exit_reason="forced_exit_end_of_test",
            comm_rt_price=eq_comm_round_turn_points(),
        ))

    return pd.DataFrame(trades)

# ============================================================
# EQUITY MR (Daily signal -> later aligned to 1H)
# ============================================================

def generate_trades_mr_eq_daily(
    market_name: str,
    df: pd.DataFrame,
    ema_fast_len: int = 20,
    ema_slow_len: int = 250,
    pullback_frac: float = 0.20,
) -> pd.DataFrame:
    df = df.copy()
    df["ema_fast"] = df["close"].ewm(span=ema_fast_len, adjust=False).mean()
    df["ema_slow"] = df["close"].ewm(span=ema_slow_len, adjust=False).mean()

    use_spread_col = "spread_points" in df.columns

    def spread_points(row) -> float:
        return float(row["spread_points"]) if use_spread_col else EQ_FIXED_SPREAD_POINTS

    trades = []

    in_position = False
    entry_signal_time = None
    entry_fill_time = None
    entry_price = None

    idx = df.index.to_list()

    for i in range(1, len(df) - 1):
        ts = idx[i]
        row = df.iloc[i]
        next_row = df.iloc[i + 1]

        if np.isnan(row["ema_slow"]) or np.isnan(row["ema_fast"]):
            continue

        # Exit
        if in_position:
            if float(row["high"]) >= float(row["ema_fast"]):
                spr = spread_points(next_row)
                exit_fill_price = float(next_row["open"] - HALF * spr - EQ_SLIPPAGE_POINTS)

                trades.append(make_trade_row(
                    market=market_name,
                    strategy="MR_EQ",
                    direction="LONG",
                    entry_signal_time=entry_signal_time,
                    entry_fill_time=entry_fill_time,
                    exit_fill_time=idx[i + 1],
                    entry_price=entry_price,
                    exit_price=exit_fill_price,
                    exit_reason="ema_fast_touch",
                    comm_rt_price=eq_comm_round_turn_points(),
                ))

                in_position = False
                entry_signal_time = None
                entry_fill_time = None
                entry_price = None

        if in_position:
            continue

        close_px = float(row["close"])
        ema_fast = float(row["ema_fast"])
        ema_slow = float(row["ema_slow"])
        high = float(row["high"])
        low = float(row["low"])

        bullish_pullback_regime = (close_px < ema_fast) and (close_px > ema_slow)
        deep_pullback = close_px < (low + pullback_frac * (high - low))

        if bullish_pullback_regime and deep_pullback:
            spr = spread_points(next_row)
            entry_fill_price = float(next_row["open"] + HALF * spr + EQ_SLIPPAGE_POINTS)

            in_position = True
            entry_signal_time = ts
            entry_fill_time = idx[i + 1]
            entry_price = entry_fill_price

    if in_position:
        last_row = df.iloc[-1]
        spr = float(last_row["spread_points"]) if "spread_points" in df.columns else EQ_FIXED_SPREAD_POINTS
        exit_fill_price = float(last_row["close"] - HALF * spr - EQ_SLIPPAGE_POINTS)

        trades.append(make_trade_row(
            market=market_name,
            strategy="MR_EQ",
            direction="LONG",
            entry_signal_time=entry_signal_time,
            entry_fill_time=entry_fill_time,
            exit_fill_time=df.index[-1],
            entry_price=entry_price,
            exit_price=exit_fill_price,
            exit_reason="forced_exit_end_of_test",
            comm_rt_price=eq_comm_round_turn_points(),
        ))

    return pd.DataFrame(trades)

def align_trades_to_hourly_opens_for_index(
    trades_df: pd.DataFrame,
    hourly_df: pd.DataFrame,
    half: float = HALF,
    slippage_points: float = EQ_SLIPPAGE_POINTS,
    fixed_spread_points: float = EQ_FIXED_SPREAD_POINTS,
) -> pd.DataFrame:
    if trades_df.empty:
        return trades_df.copy()

    t = trades_df.copy()
    t["Entry Fill Time"] = pd.to_datetime(t["Entry Fill Time"])
    t["Exit Fill Time"] = pd.to_datetime(t["Exit Fill Time"])

    hdf = hourly_df.sort_index()
    hidx = hdf.index

    use_spread_col = "spread_points" in hdf.columns

    def next_ts(ts):
        pos = hidx.searchsorted(ts, side="left")
        if pos >= len(hidx):
            return hidx[-1]
        return hidx[pos]

    def spread_at(ts):
        if use_spread_col:
            return float(hdf.loc[ts, "spread_points"])
        return float(fixed_spread_points)

    new_entry_ts, new_exit_ts, new_entry_px, new_exit_px = [], [], [], []

    for _, r in t.iterrows():
        direction = str(r["Direction"]).upper()

        et = next_ts(r["Entry Fill Time"])
        xt = next_ts(r["Exit Fill Time"])

        eopen = float(hdf.loc[et, "open"])
        xopen = float(hdf.loc[xt, "open"])

        spr_e = spread_at(et)
        spr_x = spread_at(xt)

        if direction == "LONG":
            efill = eopen + half * spr_e + slippage_points
            xfill = xopen - half * spr_x - slippage_points
        else:
            efill = eopen - half * spr_e - slippage_points
            xfill = xopen + half * spr_x + slippage_points

        new_entry_ts.append(et)
        new_exit_ts.append(xt)
        new_entry_px.append(efill)
        new_exit_px.append(xfill)

    t["Entry Fill Time"] = new_entry_ts
    t["Exit Fill Time"] = new_exit_ts
    t["Entry Price"] = new_entry_px
    t["Exit Price"] = new_exit_px
    return t

# ============================================================
# FX TF (1H)
# ============================================================

def fx_tf_atr(df_in: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df_in["high"]
    low = df_in["low"]
    close = df_in["close"]
    prev_close = close.shift(1)

    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)

    atr = tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    return atr

def generate_trades_tf_fx(
    market_name: str,
    df: pd.DataFrame,
    pip_size: float,
    spread_points_per_pip: float,
    cost_model: dict,
) -> pd.DataFrame:
    df = df.copy()

    if "volume" in df.columns:
        vol_col = "volume"
    elif "tick_volume" in df.columns:
        vol_col = "tick_volume"
    else:
        df["volume_dummy"] = 1.0
        vol_col = "volume_dummy"

    slippage_pips = float(cost_model.get("slippage_pips", 0.0))
    fixed_spread_pips = float(cost_model.get("fixed_spread_pips", 0.0))
    comm_pips_per_side = float(cost_model.get("comm_pips_per_side", 0.0))

    def pips_to_price(pips: float) -> float:
        return float(pips) * float(pip_size)

    def commission_round_turn_price() -> float:
        return 2.0 * pips_to_price(comm_pips_per_side)

    def asia_range(df_in: pd.DataFrame) -> pd.DataFrame:
        data = df_in.copy()
        session = data.between_time("00:00", "07:00")

        daily_range = session.groupby(session.index.date).agg(
            asia_high=("high", "max"),
            asia_low=("low", "min")
        )

        data["date"] = data.index.date
        data = data.merge(daily_range, left_on="date", right_index=True, how="left")
        data["asia_range"] = data["asia_high"] - data["asia_low"]
        data["asia_mid"] = (data["asia_high"] + data["asia_low"]) / 2
        data.loc[data.index.hour < 8, ["asia_high", "asia_low", "asia_range", "asia_mid"]] = np.nan
        data.drop(columns="date", inplace=True)
        return data

    def pct_rank_last(x):
        s = pd.Series(x)
        return s.rank(pct=True).iloc[-1]

    df = asia_range(df)

    asia_valid = df[df["asia_range"].notna()].copy()
    daily_asia = asia_valid.groupby(asia_valid.index.date).first()[["asia_range"]].copy()
    daily_asia.index = pd.to_datetime(daily_asia.index)
    daily_asia["asia_range_pct_rank"] = (
        daily_asia["asia_range"]
        .rolling(window=252, min_periods=50)
        .apply(pct_rank_last, raw=False)
    )

    df["date_only"] = pd.to_datetime(df.index.date)
    df = df.merge(
        daily_asia[["asia_range_pct_rank"]],
        left_on="date_only",
        right_index=True,
        how="left"
    )
    df.drop(columns="date_only", inplace=True)

    df["atr"] = fx_tf_atr(df, period=14)
    df["atr_ma"] = df["atr"].rolling(75).mean()

    session_start_t = pd.to_datetime("08:00:00").time()
    session_end_t = pd.to_datetime("09:00:00").time()

    def in_session(ts) -> bool:
        t = ts.time()
        return (t >= session_start_t) and (t < session_end_t)

    use_spread_pips_col = "spread_pips" in df.columns
    use_spread_points_col = "spread_points" in df.columns

    def get_spread_pips(row) -> float:
        if use_spread_pips_col:
            return float(row["spread_pips"])
        if use_spread_points_col:
            return float(row["spread_points"]) / float(spread_points_per_pip)
        return float(fixed_spread_pips)

    trades = []
    in_position = False
    pos_direction = None
    entry_price = None
    entry_time = None
    entry_signal_time = None

    idx_list = df.index.to_list()

    for i in range(1, len(df) - 1):
        ts = idx_list[i]
        row = df.iloc[i]
        next_row = df.iloc[i + 1]

        # Exit
        if in_position:
            exit_price = None
            exit_reason = None

            atr_now = row["atr"]
            atr_ma_now = row["atr_ma"]

            if pos_direction == "LONG":
                if np.isfinite(atr_now) and np.isfinite(atr_ma_now) and atr_now < atr_ma_now:
                    spread_pips = get_spread_pips(next_row)
                    spread_px = pips_to_price(spread_pips)
                    slip_px = pips_to_price(slippage_pips)
                    exit_price = float(next_row["open"] - HALF * spread_px - slip_px)
                    exit_reason = "atr_exit"

            else:
                if np.isfinite(atr_now) and np.isfinite(atr_ma_now) and atr_now < atr_ma_now:
                    spread_pips = get_spread_pips(next_row)
                    spread_px = pips_to_price(spread_pips)
                    slip_px = pips_to_price(slippage_pips)
                    exit_price = float(next_row["open"] + HALF * spread_px + slip_px)
                    exit_reason = "atr_exit"

            if exit_price is not None:
                trades.append(make_trade_row(
                    market=market_name,
                    strategy="TF_FX",
                    direction=pos_direction,
                    entry_signal_time=entry_signal_time,
                    entry_fill_time=entry_time,
                    exit_fill_time=idx_list[i + 1],
                    entry_price=entry_price,
                    exit_price=exit_price,
                    exit_reason=exit_reason,
                    comm_rt_price=commission_round_turn_price(),
                ))

                in_position = False
                pos_direction = None
                entry_price = None
                entry_time = None
                entry_signal_time = None
                continue

        if not in_session(ts):
            continue

        if in_position:
            continue

        close_price = row["close"]
        asia_low = row["asia_low"]
        asia_high = row["asia_high"]
        atr_now = row["atr"]
        atr_ma_now = row["atr_ma"]
        asia_range_pct_rank = row["asia_range_pct_rank"]

        if not np.isfinite(asia_high) or not np.isfinite(asia_low):
            continue
        if not np.isfinite(atr_now) or not np.isfinite(atr_ma_now):
            continue
        if not np.isfinite(asia_range_pct_rank):
            continue

        atr_filter = atr_now > atr_ma_now
        bullish_breakout = close_price > asia_high
        bearish_breakout = close_price < asia_low

        long_signal = bullish_breakout and atr_filter
        short_signal = bearish_breakout and atr_filter

        if market_name == "GBPJPY":
            long_entry = long_signal and asia_range_pct_rank > 0.75
            short_entry = short_signal and asia_range_pct_rank < 0.65
        else:
            long_entry = long_signal and asia_range_pct_rank > 0.7
            short_entry = short_signal and asia_range_pct_rank > 0.7

        if long_entry:
            spread_pips = get_spread_pips(next_row)
            spread_px = pips_to_price(spread_pips)
            slip_px = pips_to_price(slippage_pips)

            pos_direction = "LONG"
            entry_signal_time = ts
            entry_time = idx_list[i + 1]
            entry_price = float(next_row["open"] + HALF * spread_px + slip_px)
            in_position = True

        elif short_entry:
            spread_pips = get_spread_pips(next_row)
            spread_px = pips_to_price(spread_pips)
            slip_px = pips_to_price(slippage_pips)

            pos_direction = "SHORT"
            entry_signal_time = ts
            entry_time = idx_list[i + 1]
            entry_price = float(next_row["open"] - HALF * spread_px - slip_px)
            in_position = True

    return pd.DataFrame(trades)

# ============================================================
# FX MR (1H)
# ============================================================

def compute_session_anchored_vwap_and_std(data: pd.DataFrame, vol_col: str, reset_time: str):
    df_v = data.copy()

    tp = (df_v["high"] + df_v["low"] + df_v["close"]) / 3.0
    vol = df_v[vol_col].astype(float).fillna(0.0)

    df_v["tp"] = tp
    df_v["vol"] = vol
    df_v["tp_vol"] = df_v["tp"] * df_v["vol"]

    rt = pd.to_datetime(reset_time).time()
    session_date = df_v.index.floor("D")
    session_date = session_date.where(df_v.index.time >= rt, session_date - pd.Timedelta(days=1))
    df_v["session_date"] = session_date

    g = df_v.groupby("session_date", sort=False)
    df_v["cum_tp_vol"] = g["tp_vol"].cumsum()
    df_v["cum_vol"] = g["vol"].cumsum()

    vwap = df_v["cum_tp_vol"] / df_v["cum_vol"].replace(0.0, np.nan)
    std = g["tp"].transform(lambda x: x.expanding().std(ddof=0))
    return vwap, std

def generate_trades_mr_fx(
    market_name: str,
    df: pd.DataFrame,
    pip_size: float,
    spread_points_per_pip: float = 10.0,
) -> pd.DataFrame:
    df = df.copy()

    if "volume" in df.columns:
        vol_col = "volume"
    elif "tick_volume" in df.columns:
        vol_col = "tick_volume"
    else:
        df["volume_dummy"] = 1.0
        vol_col = "volume_dummy"

    df["VWAP"], df["TP_STD"] = compute_session_anchored_vwap_and_std(df, vol_col, MR_FX_VWAP_RESET)

    session_start_t = pd.to_datetime(MR_FX_SESSION_START).time()
    session_end_t = pd.to_datetime(MR_FX_SESSION_END).time()

    def in_session(ts) -> bool:
        t = ts.time()
        if session_start_t < session_end_t:
            return (t >= session_start_t) and (t < session_end_t)
        return (t >= session_start_t) or (t < session_end_t)

    costs = MR_FX_COSTS_BY_MARKET.get(market_name, MR_FX_DEFAULT_COSTS)
    slippage_pips = float(costs["slippage_pips"])
    fixed_spread_pips = float(costs["spread_pips"])
    comm = float(MR_FX_COMM_PIPS_PER_SIDE)

    def pips_to_price(pips: float) -> float:
        return float(pips) * float(pip_size)

    def commission_round_turn_price() -> float:
        return 2.0 * pips_to_price(comm)

    use_spread_pips_col = "spread_pips" in df.columns
    use_spread_points_col = "spread_points" in df.columns

    def get_spread_pips(row) -> float:
        if use_spread_pips_col:
            return float(row["spread_pips"])
        if use_spread_points_col:
            return float(row["spread_points"]) / float(spread_points_per_pip)
        return float(fixed_spread_pips)

    trades = []
    in_position = False
    pos_direction = None
    entry_price = None
    entry_time = None
    entry_signal_time = None

    idx_list = df.index.to_list()

    for i in range(1, len(df) - 1):
        ts = idx_list[i]
        row = df.iloc[i]
        prev_row = df.iloc[i - 1]
        next_row = df.iloc[i + 1]

        # Exit
        if in_position:
            exit_price = None
            exit_reason = None

            vwap = row["VWAP"]
            std = row["TP_STD"]

            if np.isfinite(vwap) and np.isfinite(std) and std != 0:
                if pos_direction == "LONG":
                    final_level = vwap - MR_FX_EXIT_STD * std
                    if row["close"] >= final_level:
                        spread_pips = get_spread_pips(next_row)
                        spread_px = pips_to_price(spread_pips)
                        slip_px = pips_to_price(slippage_pips)
                        exit_price = float(next_row["open"] - HALF * spread_px - slip_px)
                        exit_reason = "final_exit"
                else:
                    final_level = vwap + MR_FX_EXIT_STD * std
                    if row["close"] <= final_level:
                        spread_pips = get_spread_pips(next_row)
                        spread_px = pips_to_price(spread_pips)
                        slip_px = pips_to_price(slippage_pips)
                        exit_price = float(next_row["open"] + HALF * spread_px + slip_px)
                        exit_reason = "final_exit"

            if exit_price is not None:
                trades.append(make_trade_row(
                    market=market_name,
                    strategy="MR_FX",
                    direction=pos_direction,
                    entry_signal_time=entry_signal_time,
                    entry_fill_time=entry_time,
                    exit_fill_time=idx_list[i + 1],
                    entry_price=entry_price,
                    exit_price=exit_price,
                    exit_reason=exit_reason,
                    comm_rt_price=commission_round_turn_price(),
                ))

                in_position = False
                pos_direction = None
                entry_price = None
                entry_time = None
                entry_signal_time = None
                continue

        if not in_session(ts) or not in_session(idx_list[i + 1]):
            continue

        if in_position:
            continue

        close_price = row["close"]
        prev_close = prev_row["close"]
        next_open = float(next_row["open"])
        vwap = row["VWAP"]
        std = row["TP_STD"]

        if not np.isfinite(vwap) or not np.isfinite(std) or std == 0:
            continue
        if not np.isfinite(prev_row["VWAP"]) or not np.isfinite(prev_row["TP_STD"]):
            continue

        upper_band = vwap + MR_FX_ENTRY_STD * std
        lower_band = vwap - MR_FX_ENTRY_STD * std
        prev_upper_band = prev_row["VWAP"] + MR_FX_ENTRY_STD * prev_row["TP_STD"]
        prev_lower_band = prev_row["VWAP"] - MR_FX_ENTRY_STD * prev_row["TP_STD"]

        upper_band_break = (prev_close < prev_upper_band) and (close_price > upper_band)
        lower_band_break = (prev_close > prev_lower_band) and (close_price < lower_band)

        if lower_band_break:
            spread_pips = get_spread_pips(next_row)
            spread_px = pips_to_price(spread_pips)
            slip_px = pips_to_price(slippage_pips)

            pos_direction = "LONG"
            entry_signal_time = ts
            entry_time = idx_list[i + 1]
            entry_price = next_open + HALF * spread_px + slip_px
            in_position = True

        elif upper_band_break:
            spread_pips = get_spread_pips(next_row)
            spread_px = pips_to_price(spread_pips)
            slip_px = pips_to_price(slippage_pips)

            pos_direction = "SHORT"
            entry_signal_time = ts
            entry_time = idx_list[i + 1]
            entry_price = next_open - HALF * spread_px - slip_px
            in_position = True

    return pd.DataFrame(trades)

# ============================================================
# PORTFOLIO ENGINE (USD MTM, one position per market+strategy)
# ============================================================

def gross_notional_usd_at_ts(market: str, units: float, ts: pd.Timestamp, close_matrix: pd.DataFrame) -> float:
    spec = INSTRUMENTS[market]

    if spec.asset_class == "INDEX":
        px = get_close_asof(close_matrix, market, ts)
        return abs(units * px * POINT_VALUE[market])

    return abs(units * usd_per_base_unit(market, ts, close_matrix))

def unrealized_pnl_usd(
    market: str,
    direction: str,
    units: float,
    entry_price: float,
    current_price: float,
    ts: pd.Timestamp,
    close_matrix: pd.DataFrame,
) -> float:
    sign = 1.0 if direction == "LONG" else -1.0
    spec = INSTRUMENTS[market]

    if spec.asset_class == "INDEX":
        return sign * (current_price - entry_price) * units * POINT_VALUE[market]

    pnl_quote = sign * (current_price - entry_price) * units
    return quote_to_usd(pnl_quote, spec.quote_ccy, ts, close_matrix)

def commission_usd(
    market: str,
    units: float,
    comm_rt_price: float,
    ts: pd.Timestamp,
    close_matrix: pd.DataFrame,
) -> float:
    spec = INSTRUMENTS[market]

    if spec.asset_class == "INDEX":
        return comm_rt_price * units * POINT_VALUE[market]

    comm_quote = comm_rt_price * units
    return quote_to_usd(comm_quote, spec.quote_ccy, ts, close_matrix)

def units_from_target_notional_usd(
    market: str,
    ts: pd.Timestamp,
    target_notional_usd: float,
    close_matrix: pd.DataFrame,
) -> float:
    if target_notional_usd <= 0:
        return 0.0

    spec = INSTRUMENTS[market]

    if spec.asset_class == "INDEX":
        px = get_close_asof(close_matrix, market, ts)
        denom = px * POINT_VALUE[market]
        return target_notional_usd / denom if denom > 0 else 0.0

    usd_per_unit = usd_per_base_unit(market, ts, close_matrix)
    return target_notional_usd / usd_per_unit if usd_per_unit > 0 else 0.0

def build_close_matrix(market_dfs_1h: dict) -> pd.DataFrame:
    all_index = None
    for _, df in market_dfs_1h.items():
        dfx = df.sort_index()
        all_index = dfx.index if all_index is None else all_index.union(dfx.index)

    all_index = pd.DatetimeIndex(all_index.sort_values().unique())

    closes = pd.DataFrame(index=all_index)
    for mkt, df in market_dfs_1h.items():
        closes[mkt] = df["close"].reindex(all_index).ffill()

    return closes

def build_vol_regime_series_from_equity_proxy(
    equity_series: pd.Series | None = None,
    close_series: pd.Series | None = None,
    lookback: int = 20,
    labels: tuple[str, str, str, str] = ("LowVol", "MidVol", "HighVol", "ExtremeVol"),
) -> pd.Series:
    """
    Bygger daglig vol-regimserie från antingen:
      - equity_series (om ni vill använda portföljens egen vol)
      - close_series (om ni vill använda en proxy, t.ex. US500 close)

    För entry overlay i backtest rekommenderas en exogen proxy, t.ex. US500.
    """
    if equity_series is None and close_series is None:
        raise ValueError("Ange equity_series eller close_series")

    if equity_series is not None:
        daily_eq = equity_series.resample("1D").last().dropna()
        daily_ret = daily_eq.pct_change().dropna()
    else:
        daily_close = close_series.resample("1D").last().dropna()
        daily_ret = daily_close.pct_change().dropna()

    vol = daily_ret.rolling(lookback).std().dropna()
    if vol.empty:
        return pd.Series(dtype="object")

    q1 = vol.quantile(0.25)
    q2 = vol.quantile(0.50)
    q3 = vol.quantile(0.75)

    low_lbl, mid_lbl, high_lbl, extreme_lbl = labels

    regime = pd.Series(index=vol.index, dtype="object")
    regime[vol <= q1] = low_lbl
    regime[(vol > q1) & (vol <= q2)] = mid_lbl
    regime[(vol > q2) & (vol <= q3)] = high_lbl
    regime[vol > q3] = extreme_lbl

    return regime.dropna()

def strategy_target_with_regime_overlay(
    strategy: str,
    market: str,
    base_target: float,
    market_regime_label: str | None,
) -> float:
    """
    Market-specific regime overlay.

    Overlay triggas av den marknadens egen regim.
    Lookup:
      (Regime, Strategy, Market) -> multiplier

    Om ingen regel finns används 1.0.
    """
    target = float(base_target)

    if not USE_REGIME_OVERLAY:
        return target

    if market_regime_label is None:
        return target

    mult = REGIME_MARKET_MULTIPLIERS.get(
        (str(market_regime_label), str(strategy), str(market)),
        1.0
    )
    target *= float(mult)

    return target

def build_portfolio_mtm_usd_multi_strategy(
    market_dfs_1h: dict,
    trades_df: pd.DataFrame,
    start_capital: float,
    max_gross_exposure: float,
    strategy_targets: dict,
    strategy_market_weights: dict,
    market_regime_series_daily: dict | None = None,
) -> tuple:
    """
    Multi-asset portfolio engine i USD.

    Returnerar:
      1) equity_series
      2) realized_series
      3) daily_returns
      4) open_pos_series
      5) gross_exposure_series
      6) strategy_equity_df
      7) strategy_pnl_1h_df
      8) strategy_pnl_1d_df
      9) strategy_gross_df
    """
    tr = trades_df.copy()
    tr["Entry Fill Time"] = pd.to_datetime(tr["Entry Fill Time"])
    tr["Exit Fill Time"] = pd.to_datetime(tr["Exit Fill Time"])

    required = {
        "Market", "Strategy", "Direction",
        "Entry Fill Time", "Exit Fill Time",
        "Entry Price", "Exit Price"
    }
    missing = required - set(tr.columns)
    if missing:
        raise ValueError(f"Trades saknar kolumner: {missing}")

    strategies = sorted(strategy_targets.keys())

    smw = {}
    for strat, w in strategy_market_weights.items():
        smw[strat] = normalize_weights(w)

    close_matrix = build_close_matrix(market_dfs_1h)
    all_index = close_matrix.index

    entries = tr.sort_values(["Entry Fill Time", "Market", "Strategy"]).groupby("Entry Fill Time")
    exits = tr.sort_values(["Exit Fill Time", "Market", "Strategy"]).groupby("Exit Fill Time")

    positions = {}
    realized_equity = float(start_capital)
    realized_by_strategy = {s: 0.0 for s in strategies}

    equity_path = []
    realized_path = []
    gross_exposure_path = []
    open_positions_path = []

    strategy_equity_snapshots = []
    strategy_gross_snapshots = []

    def current_unrealized_and_gross_by_strategy(ts: pd.Timestamp):
        unreal_by_strategy = {s: 0.0 for s in strategies}
        gross_by_strategy = {s: 0.0 for s in strategies}

        for (mkt, strat), pos in positions.items():
            px = get_close_asof(close_matrix, mkt, ts)

            pnl_u = unrealized_pnl_usd(
                market=mkt,
                direction=pos["direction"],
                units=pos["units"],
                entry_price=pos["entry_price"],
                current_price=px,
                ts=ts,
                close_matrix=close_matrix,
            )

            gross_u = gross_notional_usd_at_ts(
                market=mkt,
                units=pos["units"],
                ts=ts,
                close_matrix=close_matrix,
            )

            unreal_by_strategy[strat] += pnl_u
            gross_by_strategy[strat] += gross_u

        total_unreal = sum(unreal_by_strategy.values())
        total_gross = sum(gross_by_strategy.values())

        return total_unreal, total_gross, unreal_by_strategy, gross_by_strategy

    def market_regime_at_ts(market: str, ts: pd.Timestamp) -> str | None:
        if market_regime_series_daily is None:
            return None

        s = market_regime_series_daily.get(market)
        if s is None or len(s) == 0:
            return None

        d = pd.Timestamp(ts).normalize()
        val = s.asof(d)
        if pd.isna(val):
            return None
        return str(val)

    for ts in all_index:
        # 1) Exits
        if ts in exits.groups:
            block = exits.get_group(ts)

            for _, t in block.iterrows():
                mkt = t["Market"]
                strat = t["Strategy"]
                key = (mkt, strat)

                if key not in positions:
                    continue

                pos = positions[key]

                pnl_real = unrealized_pnl_usd(
                    market=mkt,
                    direction=pos["direction"],
                    units=pos["units"],
                    entry_price=pos["entry_price"],
                    current_price=float(t["Exit Price"]),
                    ts=ts,
                    close_matrix=close_matrix,
                )

                comm_real = commission_usd(
                    market=mkt,
                    units=pos["units"],
                    comm_rt_price=float(t.get("Comm RT Price", 0.0)),
                    ts=ts,
                    close_matrix=close_matrix,
                )

                pnl_net = pnl_real - comm_real

                realized_equity += pnl_net
                realized_by_strategy[strat] += pnl_net

                del positions[key]

        # 2) Entries
        if ts in entries.groups:
            block = entries.get_group(ts)

            unreal_now, gross_now, _, _ = current_unrealized_and_gross_by_strategy(ts)
            equity_now = realized_equity + unreal_now

            for _, t in block.iterrows():
                mkt = t["Market"]
                strat = t["Strategy"]
                direction = str(t["Direction"]).upper()
                key = (mkt, strat)

                if key in positions:
                    continue

                current_market_regime = market_regime_at_ts(mkt, ts)

                base_target = float(strategy_targets.get(strat, 0.0))
                strat_target = strategy_target_with_regime_overlay(
                    strategy=strat,
                    market=mkt,
                    base_target=base_target,
                    market_regime_label=current_market_regime,
                )
                if strat_target <= 0:
                    continue

                w_mkt = float(smw.get(strat, {}).get(mkt, 0.0))
                if w_mkt <= 0:
                    continue

                remaining_notional_total = max(0.0, equity_now * max_gross_exposure - gross_now)
                if remaining_notional_total <= 0:
                    continue

                desired_notional = equity_now * strat_target * w_mkt
                position_notional = min(desired_notional, remaining_notional_total)

                units = units_from_target_notional_usd(
                    market=mkt,
                    ts=ts,
                    target_notional_usd=position_notional,
                    close_matrix=close_matrix,
                )

                if units <= 0:
                    continue

                positions[key] = {
                    "units": float(units),
                    "entry_price": float(t["Entry Price"]),
                    "direction": direction,
                    "comm_rt_price": float(t.get("Comm RT Price", 0.0)),
                }

                gross_now += position_notional

        # 3) Snapshot
        unreal, gross, unreal_by_strategy, gross_by_strategy = current_unrealized_and_gross_by_strategy(ts)
        equity = realized_equity + unreal

        equity_by_strategy = {
            s: realized_by_strategy[s] + unreal_by_strategy[s]
            for s in strategies
        }

        equity_path.append((ts, equity))
        realized_path.append((ts, realized_equity))
        open_positions_path.append((ts, len(positions)))
        gross_exposure_path.append((ts, gross / equity if equity > 0 else 0.0))

        strategy_equity_snapshots.append((ts, equity_by_strategy.copy()))
        strategy_gross_snapshots.append((ts, gross_by_strategy.copy()))

    equity_series = pd.Series(
        [v for _, v in equity_path],
        index=pd.DatetimeIndex([t for t, _ in equity_path]),
        name="PortfolioEquity"
    )

    realized_series = pd.Series(
        [v for _, v in realized_path],
        index=pd.DatetimeIndex([t for t, _ in realized_path]),
        name="RealizedEquity"
    )

    open_pos_series = pd.Series(
        [v for _, v in open_positions_path],
        index=pd.DatetimeIndex([t for t, _ in open_positions_path]),
        name="OpenPositions"
    )

    gross_exposure_series = pd.Series(
        [v for _, v in gross_exposure_path],
        index=pd.DatetimeIndex([t for t, _ in gross_exposure_path]),
        name="GrossExposurePct"
    )

    strategy_equity_df = pd.DataFrame(
        [row for _, row in strategy_equity_snapshots],
        index=pd.DatetimeIndex([ts for ts, _ in strategy_equity_snapshots])
    ).sort_index()

    strategy_gross_df = pd.DataFrame(
        [row for _, row in strategy_gross_snapshots],
        index=pd.DatetimeIndex([ts for ts, _ in strategy_gross_snapshots])
    ).sort_index()

    strategy_pnl_1h_df = strategy_equity_df.diff().fillna(0.0)

    strategy_pnl_1d_df = strategy_pnl_1h_df.resample("1D").sum()
    strategy_pnl_1d_df["TOTAL"] = strategy_pnl_1d_df.sum(axis=1)

    daily_equity = equity_series.resample("1D").last().dropna()
    daily_returns = daily_equity.pct_change().dropna()

    return (
        equity_series,
        realized_series,
        daily_returns,
        open_pos_series,
        gross_exposure_series,
        strategy_equity_df,
        strategy_pnl_1h_df,
        strategy_pnl_1d_df,
        strategy_gross_df,
    )

# ============================================================
# METRICS / DD HELPERS
# ============================================================

def portfolio_metrics_from_equity(equity_series: pd.Series, daily_returns: pd.Series, trading_days=252) -> dict:
    n_days = (equity_series.index[-1] - equity_series.index[0]).days
    years = n_days / 365.25 if n_days > 0 else np.nan
    cagr = (equity_series.iloc[-1] / equity_series.iloc[0]) ** (1.0 / years) - 1.0 if years and years > 0 else np.nan

    roll_max = equity_series.cummax()
    dd = equity_series / roll_max - 1.0
    max_dd = dd.min()

    mu = daily_returns.mean()
    sd = daily_returns.std(ddof=1)
    sharpe = (mu / sd) * np.sqrt(trading_days) if sd and sd > 0 else np.nan
    calmar = (cagr / abs(max_dd)) if pd.notna(cagr) and pd.notna(max_dd) and max_dd < 0 else np.nan

    return {
        "Equity Start": float(equity_series.iloc[0]),
        "Equity End": float(equity_series.iloc[-1]),
        "CAGR": float(cagr) if pd.notna(cagr) else np.nan,
        "Max Drawdown %": float(max_dd * 100.0) if pd.notna(max_dd) else np.nan,
        "Sharpe (ann.)": float(sharpe) if pd.notna(sharpe) else np.nan,
        "Calmar": float(calmar) if pd.notna(calmar) else np.nan,
        "Avg Daily Return": float(mu) if pd.notna(mu) else np.nan,
        "Daily Vol": float(sd) if pd.notna(sd) else np.nan,
    }

def max_drawdown_pct(equity: pd.Series) -> float:
    roll_max = equity.cummax()
    dd = equity / roll_max - 1.0
    return float(dd.min())

def compute_intraday_drawdowns(equity_series: pd.Series) -> pd.DataFrame:
    eq = equity_series.copy().dropna()
    eq.index = pd.to_datetime(eq.index)

    rows = []
    for day, s in eq.groupby(eq.index.date):
        if len(s) < 2:
            continue

        day_start = float(s.iloc[0])
        running_max = s.cummax()
        intraday_dd = (s - running_max) / day_start

        rows.append({
            "date": pd.Timestamp(day),
            "DayStartEquity": day_start,
            "MinIntradayDD": float(intraday_dd.min()),
        })

    return pd.DataFrame(rows).set_index("date")

def daily_loss_statistics(
    intraday_dd_df: pd.DataFrame,
    soft_limit: float = -0.05,
    hard_limit: float = -0.10,
) -> dict:
    n = len(intraday_dd_df)
    if n == 0:
        raise ValueError("Ingen intraday drawdown-data.")

    breach_5 = intraday_dd_df["MinIntradayDD"] <= soft_limit
    breach_10 = intraday_dd_df["MinIntradayDD"] <= hard_limit

    return {
        "Days": n,
        "WorstDayDD": float(intraday_dd_df["MinIntradayDD"].min()),
        "AvgWorstDD": float(intraday_dd_df["MinIntradayDD"].mean()),
        "P(Breach -5%)": float(breach_5.mean()),
        "P(Breach -10%)": float(breach_10.mean()),
        "Count Breach -5%": int(breach_5.sum()),
        "Count Breach -10%": int(breach_10.sum()),
    }

# ===============
# FTMO HELPERS
# ===============

def dynamic_soft_cutoff_from_drawdown(
    current_equity: float,
    reference_balance: float,
) -> float:
    """
    Returnerar soft cutoff beroende på var kontot ligger relativt start_balance.

    Kontozoner:
      - [-10%, -7%)   -> cutoff -0.75%
      - [-7%, -4.5%)  -> cutoff -1.50%
      - [-4.5%, 0%)   -> cutoff -3.00%

    Om equity >= start_balance (breakeven eller bättre) kan man antingen:
      - behålla -3.00%
      - eller återgå till någon standard, t.ex. -4.00%

    Här sätter vi -3.00% även vid >= breakeven, men kan enkelt ändra detta.
    """
    dd = current_equity / reference_balance - 1.0

    #if dd < -0.07:
        #return -0.01   # väldigt defensiv när kontot är pressat
    if dd < -0.080:
        return -0.045
    elif dd < -0.070:
        return -0.045
    else:
        return -0.045

def dynamic_exposure_from_drawdown(current_equity: float, reference_balance: float) -> float:
    dd = current_equity / reference_balance - 1.0

    if dd < -0.07:
        return 0.40
    elif dd < -0.04:
        return 0.40
    elif dd < -0.02:
        return 0.80
    elif dd < 0.00:
        return 1.00
    else:
        return 1.25

def dd_bucket(dd):
    if dd < -0.07:
        return "<-7%"
    elif dd < -0.04:
        return "-7%_-4%"
    elif dd < -0.02:
        return "-4%_-2%"
    elif dd < 0.0:
        return "-2%_0%"
    elif dd < 0.025:
        return "0%_2.5%"
    elif dd < 0.05:
        return "2.5%_5%"
    else:
        return ">5%"

BEST_EXPOSURE_MAP = {
    "<-7%": 0.4,
    "-7%_-4%": 0.4,
    "-4%_-2%": 0.8,
    "-2%_0%": 1.0,
    "0%_2.5%": 1.25,
    "2.5%_5%": 1.25,
    ">5%": 1.25,
}
RANK3_EXPOSURE_MAP = {
    "<-7%": 0.4,
    "-7%_-4%": 0.8,
    "-4%_-2%": 0.8,
    "-2%_0%": 1.0,
    "0%_2.5%": 1.25,
    "2.5%_5%": 1.25,
    ">5%": 1.25,
}
RANK4_EXPOSURE_MAP = {
    "<-7%": 0.4,
    "-7%_-4%": 0.4,
    "-4%_-2%": 1.0,
    "-2%_0%": 1.0,
    "0%_2.5%": 1.25,
    "2.5%_5%": 1.25,
    ">5%": 1.25,
}
RANK9_EXPOSURE_MAP = {
    "<-7%": 0.4,
    "-7%_-4%": 0.4,
    "-4%_-2%": 0.8,
    "-2%_0%": 1.0,
    "0%_2.5%": 1.0,
    "2.5%_5%": 1.25,
    ">5%": 1.25,
}
RANK17_EXPOSURE_MAP = {
    "<-7%": 0.4,
    "-7%_-4%": 0.4,
    "-4%_-2%": 0.4,
    "-2%_0%": 0.8,
    "0%_2.5%": 1.0,
    "2.5%_5%": 1.0,
    ">5%": 1.25,
}
def dynamic_exposure_from_map(
    current_equity: float,
    reference_balance: float,
    exposure_map: dict[str, float] | None = None,
) -> float:
    if exposure_map is None:
        exposure_map = RANK4_EXPOSURE_MAP

    dd = current_equity / reference_balance - 1.0
    bucket = dd_bucket(dd)
    return float(exposure_map[bucket])

@dataclass
class FTMOParams2:
    challenge_target: float = 0.10
    verification_target: float = 0.05
    daily_loss_limit: float = 0.05
    max_loss_limit: float = 0.10
    min_trading_days_eval: int = 4
    payout_wait_days: int = 14
    soft_cutoff: float = -0.045
    eval_exposure_factor: float = 0.75
    funded_exposure_factor: float = 0.75

def build_day_path_library(
    equity_1h: pd.Series,
    balance_1h: pd.Series,
    trades_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Bygger ett day library med separata serier för:
      1) FTMO-risk relativt dagens start balance
      2) equity-evolution relativt dagens öppnings-equity
    """
    eq = equity_1h.dropna().copy()
    eq.index = pd.to_datetime(eq.index)
    eq = eq.sort_index()

    bal = balance_1h.reindex(eq.index).ffill().copy()
    bal.index = pd.to_datetime(bal.index)
    bal = bal.sort_index()

    tr = trades_df.copy()
    tr["Entry Fill Time"] = pd.to_datetime(tr["Entry Fill Time"])
    entry_dates = set(pd.Timestamp(d) for d in tr["Entry Fill Time"].dt.normalize().unique())

    rows = []
    for day, s in eq.groupby(eq.index.normalize()):
        s = s.sort_index()
        if len(s) < 2:
            continue

        day_bal = bal.loc[s.index]
        day_start_balance = float(day_bal.iloc[0])
        day_open_equity = float(s.iloc[0])

        if not np.isfinite(day_start_balance) or day_start_balance <= 0:
            continue
        if not np.isfinite(day_open_equity) or day_open_equity <= 0:
            continue

        # För FTMO daily risk
        path_from_balance = (s / day_start_balance - 1.0).values.astype(float)

        # För equity evolution
        path_from_equity_open = (s / day_open_equity - 1.0).values.astype(float)
        day_end_return_from_equity_open = float(path_from_equity_open[-1])

        rows.append({
            "date": day,
            "IsTradingDay": bool(day in entry_dates),
            "IntradayPathFromBalance": path_from_balance,
            "IntradayPathFromEquityOpen": path_from_equity_open,
            "DayEndReturnFromEquityOpen": day_end_return_from_equity_open,
        })

    lib = pd.DataFrame(rows).set_index("date").sort_index()
    return lib

def block_bootstrap_day_indices(sample_len: int, universe_len: int, block_len: int, rng: np.random.Generator) -> np.ndarray:
    out = []
    while len(out) < sample_len:
        start = int(rng.integers(0, universe_len))
        block = [(start + j) % universe_len for j in range(block_len)]
        out.extend(block)
    return np.array(out[:sample_len], dtype=int)

def apply_exposure_and_cutoff(
    frac_path: np.ndarray,
    exposure_factor: float,
    soft_cutoff: float | None,
) -> tuple[np.ndarray, float]:
    """
    frac_path[t] = equity_t / day_start_balance - 1

    Används för FTMO-risk:
      - soft_cutoff mäts från dagens start balance
      - om cutoff träffas antas alla positioner stängas
        och inga fler entries resten av dagen
    """
    p = frac_path.astype(float) * float(exposure_factor)

    if len(p) == 0:
        return p, np.nan

    if soft_cutoff is None:
        return p, float(np.min(p))

    hit = np.where(p <= soft_cutoff)[0]
    if hit.size > 0:
        j = int(hit[0])
        p[j:] = soft_cutoff

    worst = float(np.min(p))
    return p, worst

def simulate_stage_from_daypaths(
    day_sample: pd.DataFrame,
    start_balance: float,
    target_frac: float,
    daily_loss_limit_frac: float,
    max_loss_limit_frac: float,
    min_trading_days: int,
    exposure_factor: float,
    soft_cutoff: float,
    exposure_map: dict[str, float] | None = None,
) -> dict:
    """
    Simulerar Challenge/Verification på dags-path nivå.

    Viktigt:
      - day paths är relativa mot dagens start-balance
      - soft_cutoff mäts från dagens start-balance
      - daily loss check mäts också från dagens start-balance
      - om soft_cutoff träffas antas:
          * alla öppna positioner stängas
          * inga nya entries resten av dagen
    """
    bal0 = float(start_balance)
    equity = bal0

    trading_days = 0
    days_elapsed = 0

    target_equity = bal0 * (1.0 + target_frac)
    max_loss_floor = bal0 * (1.0 - max_loss_limit_frac)

    for dt, row in day_sample.iterrows():
        days_elapsed += 1

        risk_path = row["IntradayPathFromBalance"]
        day_end_return_eq_open = float(row["DayEndReturnFromEquityOpen"])

        dd_mult = dynamic_exposure_from_map(
            current_equity=equity,
            reference_balance=bal0,
            exposure_map=exposure_map,
        )

        effective_exposure = exposure_factor * dd_mult

        current_soft_cutoff = dynamic_soft_cutoff_from_drawdown(
            current_equity=equity,
            reference_balance=bal0,
        )

        adj_risk_path, worst_frac = apply_exposure_and_cutoff(
            frac_path=risk_path,
            exposure_factor=effective_exposure,
            soft_cutoff=current_soft_cutoff,
        )

        day_start_balance = equity
        daily_loss_abs = day_start_balance * daily_loss_limit_frac

        min_equity_intraday = day_start_balance * (1.0 + worst_frac)
        intraday_drop_abs = day_start_balance - min_equity_intraday

        if intraday_drop_abs > daily_loss_abs:
            return {
                "passed": False, "failed": True, "fail_reason": "MaxDailyLoss",
                "days_elapsed": days_elapsed, "trading_days": trading_days,
                "end_equity": float(min_equity_intraday), "pass_day": None
            }

        end_frac = day_end_return_eq_open * effective_exposure

        if current_soft_cutoff is not None and np.min(adj_risk_path) <= current_soft_cutoff:
            end_frac = float(adj_risk_path[-1])

        equity = day_start_balance * (1.0 + end_frac)

        if bool(row["IsTradingDay"]):
            trading_days += 1

        if equity < max_loss_floor:
            return {
                "passed": False, "failed": True, "fail_reason": "MaxLoss",
                "days_elapsed": days_elapsed, "trading_days": trading_days,
                "end_equity": float(equity), "pass_day": None
            }

        if equity >= target_equity and trading_days >= min_trading_days:
            return {
                "passed": True, "failed": False, "fail_reason": None,
                "days_elapsed": days_elapsed, "trading_days": trading_days,
                "end_equity": float(equity), "pass_day": days_elapsed
            }

    return {
        "passed": False, "failed": False, "fail_reason": "HorizonEnded",
        "days_elapsed": days_elapsed, "trading_days": trading_days,
        "end_equity": float(equity), "pass_day": None
    }

def simulate_funded_to_payout_from_daypaths(
    day_sample: pd.DataFrame,
    start_balance: float,
    daily_loss_limit_frac: float,
    max_loss_limit_frac: float,
    payout_wait_days: int,
    exposure_factor: float,
    soft_cutoff: float,
    exposure_map: dict[str, float] | None = None,
) -> dict:
    """
    Simulerar funded tills payout eligibility.

    Viktigt:
      - day paths är relativa mot dagens start-balance
      - soft_cutoff mäts från dagens start-balance
      - daily loss check mäts också från dagens start-balance
    """
    bal0 = float(start_balance)
    equity = bal0

    daily_loss_abs = bal0 * daily_loss_limit_frac
    max_loss_floor = bal0 * (1.0 - max_loss_limit_frac)

    days_elapsed = 0
    first_trade_day_idx = None

    for dt, row in day_sample.iterrows():
        days_elapsed += 1

        if first_trade_day_idx is None and bool(row["IsTradingDay"]):
            first_trade_day_idx = days_elapsed

        risk_path = row["IntradayPathFromBalance"]
        day_end_return_eq_open = float(row["DayEndReturnFromEquityOpen"])

        dd_mult = dynamic_exposure_from_map(
            current_equity=equity,
            reference_balance=bal0,
            exposure_map=exposure_map,
        )

        effective_exposure = exposure_factor * dd_mult

        current_soft_cutoff = dynamic_soft_cutoff_from_drawdown(
            current_equity=equity,
            reference_balance=bal0,
        )

        adj_risk_path, worst_frac = apply_exposure_and_cutoff(
            frac_path=risk_path,
            exposure_factor=effective_exposure,
            soft_cutoff=current_soft_cutoff,
        )

        day_start_balance = equity
        daily_loss_abs = day_start_balance * daily_loss_limit_frac

        min_equity_intraday = day_start_balance * (1.0 + worst_frac)
        intraday_drop_abs = day_start_balance - min_equity_intraday

        if intraday_drop_abs > daily_loss_abs:
            return {
                "eligible": False,
                "fail_reason": "MaxDailyLoss",
                "days_elapsed": days_elapsed,
                "eligible_day": None
            }

        end_frac = day_end_return_eq_open * effective_exposure

        if current_soft_cutoff is not None and np.min(adj_risk_path) <= current_soft_cutoff:
            end_frac = float(adj_risk_path[-1])

        equity = day_start_balance * (1.0 + end_frac)

        if equity < max_loss_floor:
            return {
                "eligible": False,
                "fail_reason": "MaxLoss",
                "days_elapsed": days_elapsed,
                "eligible_day": None
            }

        if first_trade_day_idx is not None:
            if (days_elapsed - first_trade_day_idx) >= payout_wait_days and equity > bal0:
                return {
                    "eligible": True,
                    "fail_reason": None,
                    "days_elapsed": days_elapsed,
                    "eligible_day": days_elapsed
                }

    return {
        "eligible": False,
        "fail_reason": "HorizonEnded",
        "days_elapsed": days_elapsed,
        "eligible_day": None
    }
def run_bootstrap_ftmo_pipeline_daypaths(
    day_lib: pd.DataFrame,
    start_balance: float,
    n_iter: int,
    horizon_days: int,
    block_len: int,
    seed: int,
    params: FTMOParams2,
    exposure_map: dict[str, float] | None = None,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base = day_lib.copy()
    n = len(base)
    if n < 2 * block_len:
        raise ValueError("För få dagar i day_lib relativt block_len.")

    results = []
    L = min(horizon_days, n)

    for k in range(n_iter):
        idx = block_bootstrap_day_indices(
            sample_len=L,
            universe_len=n,
            block_len=block_len,
            rng=rng,
        )
        sample = base.iloc[idx].copy()

        ch = simulate_stage_from_daypaths(
            day_sample=sample,
            start_balance=start_balance,
            target_frac=params.challenge_target,
            daily_loss_limit_frac=params.daily_loss_limit,
            max_loss_limit_frac=params.max_loss_limit,
            min_trading_days=params.min_trading_days_eval,
            exposure_factor=params.eval_exposure_factor,
            soft_cutoff=params.soft_cutoff,
            exposure_map=exposure_map,
        )
        if not ch["passed"]:
            results.append({
                "iter": k,
                "passed_challenge": False,
                "passed_verification": False,
                "reached_payout": False,
                "days_to_challenge": ch["days_elapsed"],
                "days_to_verification": np.nan,
                "days_to_payout": np.nan,
                "fail_stage": "Challenge",
                "fail_reason": ch["fail_reason"],
            })
            continue

        rem = sample.iloc[ch["days_elapsed"]:]
        if len(rem) < 50:
            idx2 = block_bootstrap_day_indices(
            sample_len=L,
            universe_len=n,
            block_len=block_len,
            rng=rng,
        )
            rem = base.iloc[idx2].copy()

        ver = simulate_stage_from_daypaths(
            day_sample=rem,
            start_balance=start_balance,
            target_frac=params.verification_target,
            daily_loss_limit_frac=params.daily_loss_limit,
            max_loss_limit_frac=params.max_loss_limit,
            min_trading_days=params.min_trading_days_eval,
            exposure_factor=params.eval_exposure_factor,
            soft_cutoff=params.soft_cutoff,
            exposure_map=exposure_map,
        )
        if not ver["passed"]:
            results.append({
                "iter": k,
                "passed_challenge": True,
                "passed_verification": False,
                "reached_payout": False,
                "days_to_challenge": ch["days_elapsed"],
                "days_to_verification": ver["days_elapsed"],
                "days_to_payout": np.nan,
                "fail_stage": "Verification",
                "fail_reason": ver["fail_reason"],
            })
            continue

        rem2 = rem.iloc[ver["days_elapsed"]:]
        if len(rem2) < 50:
            idx3 = block_bootstrap_day_indices(
            sample_len=L,
            universe_len=n,
            block_len=block_len,
            rng=rng,
        )
            rem2 = base.iloc[idx3].copy()

        fund = simulate_funded_to_payout_from_daypaths(
            day_sample=rem2,
            start_balance=start_balance,
            daily_loss_limit_frac=params.daily_loss_limit,
            max_loss_limit_frac=params.max_loss_limit,
            payout_wait_days=params.payout_wait_days,
            exposure_factor=params.funded_exposure_factor,
            soft_cutoff=params.soft_cutoff,
        )

        results.append({
            "iter": k,
            "passed_challenge": True,
            "passed_verification": True,
            "reached_payout": bool(fund["eligible"]),
            "days_to_challenge": ch["days_elapsed"],
            "days_to_verification": ver["days_elapsed"],
            "days_to_payout": fund["eligible_day"] if fund["eligible"] else np.nan,
            "fail_stage": "Funded" if not fund["eligible"] else None,
            "fail_reason": fund["fail_reason"] if not fund["eligible"] else None,
        })

    return pd.DataFrame(results)

def summarize_ftmo_bootstrap_results(sim_df: pd.DataFrame) -> dict:
    """
    Tar resultat-DataFrame från run_bootstrap_ftmo_pipeline_daypaths()
    och räknar ut samma summary-mått som ni brukar printa.
    """

    df = sim_df.copy()

    # Challenge
    challenge_pass = df["passed_challenge"] == True
    p_pass_challenge = float(challenge_pass.mean())

    # Verification conditional on challenge pass
    verification_pass = challenge_pass & (df["passed_verification"] == True)
    p_pass_verification = (
        float(verification_pass.sum() / challenge_pass.sum())
        if challenge_pass.sum() > 0 else np.nan
    )

    # Payout conditional on verification pass
    payout_pass = verification_pass & (df["reached_payout"] == True)
    p_reach_payout = (
        float(payout_pass.sum() / verification_pass.sum())
        if verification_pass.sum() > 0 else np.nan
    )

    def summarize_days(series: pd.Series) -> dict:
        s = pd.to_numeric(series, errors="coerce").dropna()
        if len(s) == 0:
            return {
                "count": 0,
                "mean": np.nan,
                "median": np.nan,
                "p05": np.nan,
                "p95": np.nan,
            }
        return {
            "count": int(len(s)),
            "mean": float(s.mean()),
            "median": float(s.median()),
            "p05": float(s.quantile(0.05)),
            "p95": float(s.quantile(0.95)),
        }

    challenge_days = summarize_days(df.loc[challenge_pass, "days_to_challenge"])
    verification_days = summarize_days(df.loc[verification_pass, "days_to_verification"])
    payout_days = summarize_days(df.loc[payout_pass, "days_to_payout"])

    return {
        "p_pass_challenge": p_pass_challenge,
        "p_pass_verification": p_pass_verification,
        "p_reach_payout": p_reach_payout,
        "time_to_event": {
            "Challenge": challenge_days,
            "Verification": verification_days,
            "Payout": payout_days,
        },
    }

def optimize_dynamic_exposure_map(
    day_lib: pd.DataFrame,
    params: FTMOParams2,
    start_balance: float,
    n_iter: int,
    horizon_days: int,
    block_len: int,
    seed: int,
    candidate_values: list[float] | None = None,
    max_configs: int | None = None,
):
    """
    Testar många exposure maps och rankar dem efter FTMO-resultat.

    candidate_values:
        möjliga exposure-nivåer per bucket
    max_configs:
        valfri cap om ni vill stoppa efter N kombinationer
    """
    if candidate_values is None:
        candidate_values = [0.40, 0.60, 0.80, 1.00, 1.25]

    bucket_order = ["<-7%", "-7%_-4%", "-4%_-2%", "-2%_0%", "0%_2.5%", "2.5%_5%", ">5%"]

    # Monotonicity:
    # vi vill normalt inte att exposure blir lägre när drawdown förbättras,
    # så vi filtrerar till icke-avtagande profiler
    all_configs = []
    for vals in product(candidate_values, repeat=len(bucket_order)):
        is_monotone = all(vals[i] <= vals[i + 1] for i in range(len(vals) - 1))
        if not is_monotone:
            continue

        exposure_map = dict(zip(bucket_order, vals))
        all_configs.append(exposure_map)

    if max_configs is not None:
        all_configs = all_configs[:max_configs]

    rows = []
    total = len(all_configs)

    for i, exposure_map in enumerate(all_configs, start=1):
        sim = run_bootstrap_ftmo_pipeline_daypaths(
            day_lib=day_lib,
            start_balance=start_balance,
            n_iter=n_iter,
            horizon_days=horizon_days,
            block_len=block_len,
            seed=seed,
            params=params,
            exposure_map=exposure_map,
        )

        summary = summarize_ftmo_bootstrap_results(sim)

        payout_prob = summary["p_reach_payout"]
        challenge_prob = summary["p_pass_challenge"]
        verification_prob = summary["p_pass_verification"]

        challenge_median = summary["time_to_event"]["Challenge"]["median"] if summary["time_to_event"]["Challenge"]["count"] > 0 else np.nan
        verification_median = summary["time_to_event"]["Verification"]["median"] if summary["time_to_event"]["Verification"]["count"] > 0 else np.nan
        payout_median = summary["time_to_event"]["Payout"]["median"] if summary["time_to_event"]["Payout"]["count"] > 0 else np.nan

        # Enkel score: hög payout-prob bra, kortare challenge-tid bra
        # skydda mot division med 0/nan
        if np.isfinite(payout_median) and payout_median > 0:
            payout_speed_score = payout_prob / payout_median
        else:
            payout_speed_score = np.nan

        if np.isfinite(challenge_median) and challenge_median > 0:
            challenge_speed_score = challenge_prob / challenge_median
        else:
            challenge_speed_score = np.nan

        overall_score = (
            0.50 * payout_prob
            + 0.20 * challenge_prob
            + 0.10 * verification_prob
            + 50.0 * (0.0 if pd.isna(payout_speed_score) else payout_speed_score)
            + 20.0 * (0.0 if pd.isna(challenge_speed_score) else challenge_speed_score)
        )

        row = {
            "Config": str(exposure_map),
            "Overall Score": overall_score,
            "P(Pass Challenge)": challenge_prob,
            "P(Pass Verification)": verification_prob,
            "P(Reach 1st payout)": payout_prob,
            "Challenge Median Days": challenge_median,
            "Verification Median Days": verification_median,
            "Payout Median Days": payout_median,
        }

        for b in bucket_order:
            row[f"Exp {b}"] = exposure_map[b]

        rows.append(row)

        if i % 25 == 0 or i == total:
            print(f"Exposure optimization progress: {i}/{total}")

    out = pd.DataFrame(rows).sort_values("Overall Score", ascending=False).reset_index(drop=True)
    return out

def summarize_times(df: pd.DataFrame, col: str) -> dict:
    s = df[col].dropna()
    if s.empty:
        return {"count": 0}
    return {
        "count": int(s.shape[0]),
        "mean": float(s.mean()),
        "median": float(s.median()),
        "p05": float(s.quantile(0.05)),
        "p95": float(s.quantile(0.95)),
    }

# ============================================================
# REGIME / CRISIS ANALYSIS HELPERS
# ============================================================

CRISIS_PERIODS = {
    "Taper_2013": ("2013-05-15", "2013-07-15"),
    "CHF_Event_2015": ("2015-01-01", "2015-02-15"),
    "Brexit_2016": ("2016-06-01", "2016-07-15"),
    "Volmageddon_2018": ("2018-01-20", "2018-02-20"),
    "Covid_2020": ("2020-02-20", "2020-04-30"),
    "Inflation_2022": ("2022-01-01", "2022-10-31"),
    "BankStress_2023": ("2023-03-01", "2023-04-15"),
    "YenCarryStress_2024": ("2024-04-01", "2024-05-15"),
    "SpringShock_2025": ("2025-03-15", "2025-04-20"),
}

def _safe_sharpe_from_daily_returns(daily_ret: pd.Series, annualization: int = 252) -> float:
    daily_ret = daily_ret.dropna()
    if len(daily_ret) < 2:
        return np.nan

    mu = daily_ret.mean()
    sd = daily_ret.std(ddof=1)
    if sd <= 0 or pd.isna(sd):
        return np.nan

    return float((mu / sd) * np.sqrt(annualization))

def _safe_max_drawdown_from_returns(daily_ret: pd.Series) -> float:
    daily_ret = daily_ret.dropna()
    if daily_ret.empty:
        return np.nan

    eq = (1.0 + daily_ret).cumprod()
    roll_max = eq.cummax()
    dd = eq / roll_max - 1.0
    return float(dd.min())

def metrics_for_period(equity_series: pd.Series, start=None, end=None) -> dict:
    """
    Metrics på en delperiod av portfolio equity.
    """
    s = equity_series.copy().dropna().sort_index()

    if start is not None:
        s = s[s.index >= pd.Timestamp(start)]
    if end is not None:
        s = s[s.index <= pd.Timestamp(end)]

    if len(s) < 10:
        return {}

    daily_eq = s.resample("1D").last().dropna()
    if len(daily_eq) < 10:
        return {}

    daily_ret = daily_eq.pct_change().dropna()

    out = portfolio_metrics_from_equity(s, daily_ret)
    out["Worst Day"] = float(daily_ret.min()) if not daily_ret.empty else np.nan
    out["Best Day"] = float(daily_ret.max()) if not daily_ret.empty else np.nan
    out["Days"] = int(len(daily_eq))
    out["Start"] = str(daily_eq.index.min().date())
    out["End"] = str(daily_eq.index.max().date())
    return out

def evaluate_crisis_periods(equity_series: pd.Series, crisis_periods: dict | None = None) -> pd.DataFrame:
    """
    Kör period-metrics på fördefinierade stressfönster.
    """
    if crisis_periods is None:
        crisis_periods = CRISIS_PERIODS

    rows = []
    for name, (start, end) in crisis_periods.items():
        m = metrics_for_period(equity_series, start, end)
        if m:
            m["Period"] = name
            rows.append(m)

    if not rows:
        return pd.DataFrame()

    cols_first = ["Period", "Start", "End", "Days"]
    out = pd.DataFrame(rows)
    other_cols = [c for c in out.columns if c not in cols_first]
    out = out[cols_first + other_cols].set_index("Period").sort_index()
    return out

def build_vol_regime_series(
    equity_series: pd.Series,
    lookback: int = 20,
    labels: tuple[str, str, str, str] = ("LowVol", "MidVol", "HighVol", "ExtremeVol"),
) -> pd.Series:
    """
    Bygger daglig vol-regimserie från portfolio equity.
    Regim definieras via rullande realized vol och kvartiler.
    """
    daily_eq = equity_series.resample("1D").last().dropna()
    daily_ret = daily_eq.pct_change().dropna()

    vol = daily_ret.rolling(lookback).std()
    vol = vol.dropna()

    if vol.empty:
        return pd.Series(dtype="object")

    q1 = vol.quantile(0.25)
    q2 = vol.quantile(0.50)
    q3 = vol.quantile(0.75)

    low_lbl, mid_lbl, high_lbl, extreme_lbl = labels

    regime = pd.Series(index=vol.index, dtype="object")
    regime[vol <= q1] = low_lbl
    regime[(vol > q1) & (vol <= q2)] = mid_lbl
    regime[(vol > q2) & (vol <= q3)] = high_lbl
    regime[vol > q3] = extreme_lbl

    return regime.dropna()

def build_market_vol_regime_series(
    market_dfs_1h: dict,
    lookback: int = 20,
    labels: tuple[str, str, str, str] = ("LowVol", "MidVol", "HighVol", "ExtremeVol"),
) -> dict:
    """
    Bygger en daglig vol-regimserie för varje marknad, baserat på den marknadens egen daily vol.

    Returnerar:
      market_regimes = {
          "US500": pd.Series(...),
          "USDCHF": pd.Series(...),
          ...
      }
    """
    low_lbl, mid_lbl, high_lbl, extreme_lbl = labels
    market_regimes = {}

    for market, df in market_dfs_1h.items():
        close = df["close"].copy().dropna()
        daily_close = close.resample("1D").last().dropna()
        daily_ret = daily_close.pct_change().dropna()

        vol = daily_ret.rolling(lookback).std().dropna()
        if vol.empty:
            market_regimes[market] = pd.Series(dtype="object")
            continue

        q1 = vol.quantile(0.25)
        q2 = vol.quantile(0.50)
        q3 = vol.quantile(0.75)

        regime = pd.Series(index=vol.index, dtype="object")
        regime[vol <= q1] = low_lbl
        regime[(vol > q1) & (vol <= q2)] = mid_lbl
        regime[(vol > q2) & (vol <= q3)] = high_lbl
        regime[vol > q3] = extreme_lbl

        market_regimes[market] = regime.dropna()

    return market_regimes

def evaluate_by_vol_regime(equity_series: pd.Series, regime_series: pd.Series) -> pd.DataFrame:
    """
    Portfolio-metrics per vol-regim på daglig return-nivå.
    """
    daily_eq = equity_series.resample("1D").last().dropna()
    daily_ret = daily_eq.pct_change().dropna()

    aligned = pd.DataFrame({
        "ret": daily_ret,
        "regime": regime_series.reindex(daily_ret.index)
    }).dropna()

    rows = []
    for reg, grp in aligned.groupby("regime"):
        if len(grp) < 20:
            continue

        r = grp["ret"].dropna()
        mu = r.mean()
        sd = r.std(ddof=1)
        max_dd = _safe_max_drawdown_from_returns(r)

        # CAGR-liknande annualisering på regimsubsample
        eq = (1.0 + r).cumprod()
        years = len(r) / 252.0
        cagr_like = (eq.iloc[-1] ** (1.0 / years) - 1.0) if years > 0 and eq.iloc[-1] > 0 else np.nan
        calmar_like = (cagr_like / abs(max_dd)) if pd.notna(cagr_like) and pd.notna(max_dd) and max_dd < 0 else np.nan

        rows.append({
            "Regime": reg,
            "Days": int(len(r)),
            "Avg Daily Return": float(mu),
            "Daily Vol": float(sd),
            "Sharpe-like": float((mu / sd) * np.sqrt(252)) if sd > 0 else np.nan,
            "Worst Day": float(r.min()),
            "Best Day": float(r.max()),
            "Max DD %": float(max_dd * 100.0) if pd.notna(max_dd) else np.nan,
            "CAGR-like": float(cagr_like) if pd.notna(cagr_like) else np.nan,
            "Calmar-like": float(calmar_like) if pd.notna(calmar_like) else np.nan,
        })

    if not rows:
        return pd.DataFrame()

    order = ["LowVol", "MidVol", "HighVol", "ExtremeVol"]
    out = pd.DataFrame(rows).set_index("Regime")
    out = out.reindex([r for r in order if r in out.index] + [r for r in out.index if r not in order])
    return out

def market_regime_table_by_own_vol(
    trades_df: pd.DataFrame,
    market_dfs_1h: dict,
    strategy_targets: dict,
    strategy_market_weights: dict,
    start_capital: float,
    max_gross_exposure: float,
    market_regime_series_daily: dict,
) -> pd.DataFrame:
    """
    Kör varje (strategy, market) separat genom portfolio-motorn och
    utvärderar sedan den komponentens daily PnL mot JUST den marknadens egen vol-regim.
    """

    rows = []
    strategies = sorted(strategy_targets.keys())

    for strat in strategies:
        market_weights = strategy_market_weights.get(strat, {})

        for market, w in market_weights.items():
            if float(w) <= 0:
                continue

            sub_trades = trades_df[
                (trades_df["Strategy"] == strat) &
                (trades_df["Market"] == market)
            ].copy()

            if sub_trades.empty:
                continue

            sub_targets = {strat: float(strategy_targets[strat])}
            sub_weights = {strat: {market: 1.0}}

            try:
                (
                    sub_equity,
                    sub_realized,
                    sub_daily_returns,
                    sub_open_pos,
                    sub_gross_exp,
                    sub_strategy_equity_df,
                    sub_strategy_pnl_1h_df,
                    sub_strategy_pnl_1d_df,
                    sub_strategy_gross_df,
                ) = build_portfolio_mtm_usd_multi_strategy(
                    market_dfs_1h=market_dfs_1h,
                    trades_df=sub_trades,
                    start_capital=start_capital,
                    max_gross_exposure=max_gross_exposure,
                    strategy_targets=sub_targets,
                    strategy_market_weights=sub_weights,
                    market_regime_series_daily=None,  # analys utan extra overlay här
                )
            except Exception as e:
                print(f"[WARN] Kunde inte köra {strat} / {market}: {e}")
                continue

            if strat not in sub_strategy_pnl_1d_df.columns:
                continue

            regime_series = market_regime_series_daily.get(market)
            if regime_series is None or regime_series.empty:
                continue

            daily_pnl = sub_strategy_pnl_1d_df[[strat]].rename(columns={strat: "pnl"}).copy()
            daily_pnl["Regime"] = regime_series.reindex(daily_pnl.index)
            daily_pnl = daily_pnl.dropna()

            if daily_pnl.empty:
                continue

            for reg, grp in daily_pnl.groupby("Regime"):
                pnl = grp["pnl"].dropna()
                if len(pnl) < 20:
                    continue

                equity = pnl.cumsum()
                avg = pnl.mean()
                vol = pnl.std(ddof=1)
                sharpe_like = (avg / vol) * np.sqrt(252) if vol and vol > 0 else np.nan

                roll_max = equity.cummax()
                dd_abs = equity - roll_max
                max_dd_abs = float(dd_abs.min()) if len(dd_abs) > 0 else np.nan

                rows.append({
                    "Strategy": strat,
                    "Market": market,
                    "Regime": reg,
                    "Days": int(len(pnl)),
                    "Total PnL": float(pnl.sum()),
                    "Avg Daily PnL": float(avg),
                    "Daily Vol": float(vol) if pd.notna(vol) else np.nan,
                    "Sharpe-like": float(sharpe_like) if pd.notna(sharpe_like) else np.nan,
                    "Worst Day": float(pnl.min()),
                    "Best Day": float(pnl.max()),
                    "Positive Days %": float((pnl > 0).mean()),
                    "Negative Days %": float((pnl < 0).mean()),
                    "Max DD Abs": max_dd_abs,
                })

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows)

    regime_order = {"LowVol": 0, "MidVol": 1, "HighVol": 2, "ExtremeVol": 3}
    out["_reg_ord"] = out["Regime"].map(regime_order).fillna(999)

    out = (
        out.sort_values(["Strategy", "Market", "_reg_ord", "Regime"])
           .drop(columns="_reg_ord")
           .reset_index(drop=True)
    )
    return out

def _strategy_metrics_from_daily_pnl(pnl_series: pd.Series) -> dict:
    """
    Metrics för en enskild strategis dagliga PnL-serie.
    """
    s = pnl_series.dropna()
    if len(s) < 20:
        return {}

    equity = s.cumsum()
    avg = s.mean()
    vol = s.std(ddof=1)
    sharpe_like = (avg / vol) * np.sqrt(252) if vol > 0 else np.nan

    roll_max = equity.cummax()
    dd_abs = equity - roll_max
    max_dd_abs = float(dd_abs.min()) if len(dd_abs) > 0 else np.nan

    pos_days = float((s > 0).mean())
    neg_days = float((s < 0).mean())

    return {
        "Days": int(len(s)),
        "Total PnL": float(s.sum()),
        "Avg Daily PnL": float(avg),
        "Daily Vol": float(vol) if pd.notna(vol) else np.nan,
        "Sharpe-like": float(sharpe_like) if pd.notna(sharpe_like) else np.nan,
        "Worst Day": float(s.min()),
        "Best Day": float(s.max()),
        "Positive Days %": pos_days,
        "Negative Days %": neg_days,
        "Max DD Abs": float(max_dd_abs) if pd.notna(max_dd_abs) else np.nan,
    }

def strategy_regime_table(strategy_pnl_1d_df: pd.DataFrame, regime_series: pd.Series) -> pd.DataFrame:
    """
    Metrics per strategi inom varje vol-regim.
    strategy_pnl_1d_df ska innehålla strategikolumner och ev. TOTAL.
    """
    pnl = strategy_pnl_1d_df.copy()
    if "TOTAL" in pnl.columns:
        pnl = pnl.drop(columns="TOTAL")

    pnl = pnl.reindex(regime_series.index)

    rows = []
    unique_regimes = regime_series.dropna().unique().tolist()

    for strat in pnl.columns:
        for reg in unique_regimes:
            s = pnl.loc[regime_series == reg, strat].dropna()
            m = _strategy_metrics_from_daily_pnl(s)
            if m:
                m["Strategy"] = strat
                m["Regime"] = reg
                rows.append(m)

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows)
    regime_order = {"LowVol": 0, "MidVol": 1, "HighVol": 2, "ExtremeVol": 3}
    out["_reg_ord"] = out["Regime"].map(regime_order).fillna(999)
    out = out.sort_values(["Strategy", "_reg_ord", "Regime"]).drop(columns="_reg_ord").reset_index(drop=True)
    return out

def market_regime_table(
    trades_df: pd.DataFrame,
    market_dfs_1h: dict,
    strategy_targets: dict,
    strategy_market_weights: dict,
    start_capital: float,
    max_gross_exposure: float,
    regime_series: pd.Series,
) -> pd.DataFrame:
    """
    Bygger daily PnL per (strategy, market) genom att köra portfolio-motorn
    en komponent i taget, men med samma sizinglogik som fullportföljen.

    Returnerar en tabell med metrics per:
      Strategy, Market, Regime
    """

    rows = []

    # Bara riktiga strategier, inte TOTAL etc
    strategies = sorted(strategy_targets.keys())

    # loop över varje strategi och market som har vikt > 0
    for strat in strategies:
        market_weights = strategy_market_weights.get(strat, {})
        for market, w in market_weights.items():
            if float(w) <= 0:
                continue

            # filtrera trades till bara denna strategy+market
            sub_trades = trades_df[
                (trades_df["Strategy"] == strat) &
                (trades_df["Market"] == market)
            ].copy()

            if sub_trades.empty:
                continue

            # targets/weights för en isolerad körning
            sub_targets = {strat: float(strategy_targets[strat])}
            sub_weights = {strat: {market: 1.0}}

            try:
                (
                    sub_equity,
                    sub_realized,
                    sub_daily_returns,
                    sub_open_pos,
                    sub_gross_exp,
                    sub_strategy_equity_df,
                    sub_strategy_pnl_1h_df,
                    sub_strategy_pnl_1d_df,
                    sub_strategy_gross_df,
                ) = build_portfolio_mtm_usd_multi_strategy(
                    market_dfs_1h=market_dfs_1h,
                    trades_df=sub_trades,
                    start_capital=start_capital,
                    max_gross_exposure=max_gross_exposure,
                    strategy_targets=sub_targets,
                    strategy_market_weights=sub_weights,
                )
            except Exception as e:
                print(f"[WARN] Kunde inte köra {strat} / {market}: {e}")
                continue

            if strat not in sub_strategy_pnl_1d_df.columns:
                continue

            daily_pnl = sub_strategy_pnl_1d_df[[strat]].rename(columns={strat: "pnl"}).copy()
            daily_pnl["Regime"] = regime_series.reindex(daily_pnl.index)
            daily_pnl = daily_pnl.dropna()

            if daily_pnl.empty:
                continue

            for reg, grp in daily_pnl.groupby("Regime"):
                pnl = grp["pnl"].dropna()
                if len(pnl) < 20:
                    continue

                equity = pnl.cumsum()
                avg = pnl.mean()
                vol = pnl.std(ddof=1)
                sharpe_like = (avg / vol) * np.sqrt(252) if vol and vol > 0 else np.nan

                roll_max = equity.cummax()
                dd_abs = equity - roll_max
                max_dd_abs = float(dd_abs.min()) if len(dd_abs) > 0 else np.nan

                rows.append({
                    "Strategy": strat,
                    "Market": market,
                    "Regime": reg,
                    "Days": int(len(pnl)),
                    "Total PnL": float(pnl.sum()),
                    "Avg Daily PnL": float(avg),
                    "Daily Vol": float(vol) if pd.notna(vol) else np.nan,
                    "Sharpe-like": float(sharpe_like) if pd.notna(sharpe_like) else np.nan,
                    "Worst Day": float(pnl.min()),
                    "Best Day": float(pnl.max()),
                    "Positive Days %": float((pnl > 0).mean()),
                    "Negative Days %": float((pnl < 0).mean()),
                    "Max DD Abs": max_dd_abs,
                })

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows)

    regime_order = {"LowVol": 0, "MidVol": 1, "HighVol": 2, "ExtremeVol": 3}
    out["_reg_ord"] = out["Regime"].map(regime_order).fillna(999)

    out = out.sort_values(["Strategy", "Market", "_reg_ord", "Regime"]).drop(columns="_reg_ord").reset_index(drop=True)
    return out

from itertools import product
import copy

def build_overlay_multiplier_grid() -> list[dict]:
    """
    Bygger ett rimligt första sökrutnät för de overlays som redan visat edge.
    """
    tf_eq_us100_vals = [0.35, 0.50, 0.65]
    mr_fx_eurcad_vals = [0.35, 0.50, 0.65]
    tf_fx_eurjpy_low_vals = [0.35, 0.50, 0.65]
    tf_fx_gbpjpy_low_vals = [0.35, 0.50, 0.65]

    grid = []
    for a, b, c, d in product(
        tf_eq_us100_vals,
        mr_fx_eurcad_vals,
        tf_fx_eurjpy_low_vals,
        tf_fx_gbpjpy_low_vals,
    ):
        grid.append({
            ("ExtremeVol", "TF_EQ", "US100"): a,
            ("ExtremeVol", "TF_EQ", "US500"): 1.0,
            ("ExtremeVol", "TF_EQ", "US30"): 1.0,

            ("ExtremeVol", "MR_FX", "USDCHF"): 1.0,
            ("ExtremeVol", "MR_FX", "EURUSD"): 1.0,
            ("ExtremeVol", "MR_FX", "EURCHF"): 1.0,
            ("ExtremeVol", "MR_FX", "GBPCHF"): 1.0,
            ("ExtremeVol", "MR_FX", "EURCAD"): b,

            ("LowVol", "TF_FX", "EURJPY"): c,
            ("LowVol", "TF_FX", "GBPJPY"): d,
            ("LowVol", "TF_FX", "USDJPY"): 1.0,
        })
    return grid


def overlay_config_name(mult_map: dict) -> str:
    return (
        f"US100x{mult_map[('ExtremeVol','TF_EQ','US100')]:.2f}_"
        f"EURCADx{mult_map[('ExtremeVol','MR_FX','EURCAD')]:.2f}_"
        f"EURJPYx{mult_map[('LowVol','TF_FX','EURJPY')]:.2f}_"
        f"GBPJPYx{mult_map[('LowVol','TF_FX','GBPJPY')]:.2f}"
    )


def compute_portfolio_score(
    metrics: dict,
    ftmo_daily_loss_stats: dict,
    p_challenge: float,
    p_verification: float,
    p_payout: float,
) -> float:
    """
    Huvudscore för att ranka kandidater.
    Justera vikterna efter vad ni prioriterar.
    """
    cagr = float(metrics.get("CAGR", 0.0))
    sharpe = float(metrics.get("Sharpe (ann.)", 0.0))
    calmar = float(metrics.get("Calmar", 0.0))
    max_dd_pct = float(metrics.get("Max Drawdown %", 0.0))   # negativ
    worst_day_dd = float(ftmo_daily_loss_stats.get("WorstDayDD", 0.0))  # negativ
    p_breach_5 = float(ftmo_daily_loss_stats.get("P(Breach -5%)", 0.0))

    # score: högre är bättre
    score = (
        2.5 * calmar
        + 1.5 * sharpe
        + 1.0 * cagr
        + 1.75 * p_payout
        + 0.50 * p_verification
        + 0.25 * p_challenge
        - 1.25 * abs(max_dd_pct / 100.0)
        - 1.50 * abs(worst_day_dd)
        - 8.00 * p_breach_5
    )
    return float(score)


def compute_ftmo_score_only(
    ftmo_daily_loss_stats: dict,
    p_challenge: float,
    p_verification: float,
    p_payout: float,
) -> float:
    """
    Alternativ score om ni vill ranka enbart för FTMO.
    """
    worst_day_dd = float(ftmo_daily_loss_stats.get("WorstDayDD", 0.0))
    p_breach_5 = float(ftmo_daily_loss_stats.get("P(Breach -5%)", 0.0))

    score = (
        3.0 * p_payout
        + 1.0 * p_verification
        + 0.5 * p_challenge
        - 3.0 * p_breach_5
        - 1.0 * abs(worst_day_dd)
    )
    return float(score)


def run_single_overlay_backtest(
    multipliers: dict,
    market_dfs_1h: dict,
    portfolio_trades: pd.DataFrame,
    strategy_targets: dict,
    strategy_market_weights: dict,
    start_capital: float,
    max_gross_exposure_total: float,
    ftmo_params: FTMOParams2,
    boot_n_iter: int,
    boot_horizon_days: int,
    boot_block_len: int,
    boot_seed: int,
) -> dict:
    """
    Kör EN overlay-kandidat och returnerar en resultatrad.
    Kräver att build_portfolio_mtm_usd_multi_strategy läser global REGIME_MARKET_MULTIPLIERS.
    """

    global REGIME_MARKET_MULTIPLIERS

    old_multipliers = copy.deepcopy(REGIME_MARKET_MULTIPLIERS)
    REGIME_MARKET_MULTIPLIERS = copy.deepcopy(multipliers)

    try:
        market_regime_series_daily = build_market_vol_regime_series(
            market_dfs_1h=market_dfs_1h,
            lookback=REGIME_LOOKBACK_DAYS,
        )

        (
            equity_series,
            realized_equity_series,
            daily_returns,
            open_pos_series,
            gross_exposure_series,
            strategy_equity_df,
            strategy_pnl_1h_df,
            strategy_pnl_1d_df,
            strategy_gross_df,
        ) = build_portfolio_mtm_usd_multi_strategy(
            market_dfs_1h=market_dfs_1h,
            trades_df=portfolio_trades,
            start_capital=start_capital,
            max_gross_exposure=max_gross_exposure_total,
            strategy_targets=strategy_targets,
            strategy_market_weights=strategy_market_weights,
            market_regime_series_daily=market_regime_series_daily,
        )

        metrics = portfolio_metrics_from_equity(equity_series, daily_returns)

        intraday_dd_df = compute_intraday_drawdowns(equity_series)
        ftmo_daily_loss_stats = daily_loss_statistics(
            intraday_dd_df,
            soft_limit=-0.05,
            hard_limit=-0.10,
        )

        day_lib = build_day_path_library(
            equity_1h=equity_series,
            trades_df=portfolio_trades,
        )

        sim2 = run_bootstrap_ftmo_pipeline_daypaths(
            day_lib=day_lib,
            start_balance=start_capital,
            n_iter=boot_n_iter,
            horizon_days=boot_horizon_days,
            block_len=boot_block_len,
            seed=boot_seed,
            params=ftmo_params,
        )

        p_ch = float(sim2["passed_challenge"].mean())
        p_ver = float(sim2["passed_verification"].mean())
        p_pay = float(sim2["reached_payout"].mean())

        score_main = compute_portfolio_score(
            metrics=metrics,
            ftmo_daily_loss_stats=ftmo_daily_loss_stats,
            p_challenge=p_ch,
            p_verification=p_ver,
            p_payout=p_pay,
        )

        score_ftmo = compute_ftmo_score_only(
            ftmo_daily_loss_stats=ftmo_daily_loss_stats,
            p_challenge=p_ch,
            p_verification=p_ver,
            p_payout=p_pay,
        )

        row = {
            "Config": overlay_config_name(multipliers),
            "Portfolio Score": score_main,
            "FTMO Score": score_ftmo,

            "CAGR": float(metrics.get("CAGR", np.nan)),
            "Sharpe": float(metrics.get("Sharpe (ann.)", np.nan)),
            "Calmar": float(metrics.get("Calmar", np.nan)),
            "Max DD %": float(metrics.get("Max Drawdown %", np.nan)),
            "Equity End": float(metrics.get("Equity End", np.nan)),

            "WorstDayDD": float(ftmo_daily_loss_stats.get("WorstDayDD", np.nan)),
            "P(Breach -5%)": float(ftmo_daily_loss_stats.get("P(Breach -5%)", np.nan)),
            "Count Breach -5%": int(ftmo_daily_loss_stats.get("Count Breach -5%", 0)),

            "P(Pass Challenge)": p_ch,
            "P(Pass Verification)": p_ver,
            "P(Reach 1st payout)": p_pay,

            "Avg Gross Exposure %": float(gross_exposure_series.mean() * 100.0),
            "Max Gross Exposure %": float(gross_exposure_series.max() * 100.0),

            "EV_TF_EQ_US100": multipliers[("ExtremeVol", "TF_EQ", "US100")],
            "EV_MR_FX_EURCAD": multipliers[("ExtremeVol", "MR_FX", "EURCAD")],
            "LV_TF_FX_EURJPY": multipliers[("LowVol", "TF_FX", "EURJPY")],
            "LV_TF_FX_GBPJPY": multipliers[("LowVol", "TF_FX", "GBPJPY")],
        }

        return row

    finally:
        REGIME_MARKET_MULTIPLIERS = old_multipliers


def run_overlay_grid_search(
    market_dfs_1h: dict,
    portfolio_trades: pd.DataFrame,
    strategy_targets: dict,
    strategy_market_weights: dict,
    start_capital: float,
    max_gross_exposure_total: float,
    ftmo_params: FTMOParams2,
    boot_n_iter_search: int = 3000,
    boot_horizon_days: int = BOOT_HORIZON_DAYS,
    boot_block_len: int = BOOT_BLOCK_LEN,
    boot_seed: int = BOOT_SEED,
) -> pd.DataFrame:
    """
    Kör grid search över overlay multipliers.
    Använd lägre bootstrap-iter här än i finalkörning för att få rimlig runtime.
    """
    grid = build_overlay_multiplier_grid()
    rows = []

    print(f"\n[GridSearch] Number of overlay configs: {len(grid)}")

    for i, multipliers in enumerate(grid, start=1):
        print(f"[GridSearch] Running {i}/{len(grid)}: {overlay_config_name(multipliers)}")

        try:
            row = run_single_overlay_backtest(
                multipliers=multipliers,
                market_dfs_1h=market_dfs_1h,
                portfolio_trades=portfolio_trades,
                strategy_targets=strategy_targets,
                strategy_market_weights=strategy_market_weights,
                start_capital=start_capital,
                max_gross_exposure_total=max_gross_exposure_total,
                ftmo_params=ftmo_params,
                boot_n_iter=boot_n_iter_search,
                boot_horizon_days=boot_horizon_days,
                boot_block_len=boot_block_len,
                boot_seed=boot_seed,
            )
            rows.append(row)
        except Exception as e:
            print(f"[GridSearch][WARN] Failed config {overlay_config_name(multipliers)}: {e}")

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows)
    out = out.sort_values(["Portfolio Score", "FTMO Score"], ascending=False).reset_index(drop=True)
    return out

# ============================================================
# MAIN
# ============================================================

def main():
    # ----------------------------
    # 1) Load data
    # ----------------------------
    market_dfs_1h = {}
    market_dfs_1d = {}

    # Equities
    for m in EQUITY_MARKETS:
        name = m["name"]
        market_dfs_1h[name] = load_market_df(m["csv_1h"])
        market_dfs_1d[name] = load_market_df(m["csv_1d"])

    # FX TF
    for m in FX_TF_MARKETS:
        market_dfs_1h[m["name"]] = load_market_df(m["csv"])

    # FX MR traded
    for m in MR_FX_TRADE_MARKETS:
        market_dfs_1h[m["name"]] = load_market_df(m["csv"])

    # FX rate-only for conversions
    for m in FX_RATE_ONLY_MARKETS:
        market_dfs_1h[m["name"]] = load_market_df(m["csv"])

    # ----------------------------
    # 2) Generate trades
    # ----------------------------
    all_trades = []

    # Equity TF + MR
    for m in EQUITY_MARKETS:
        name = m["name"]

        t_tf_eq = generate_trades_tf_eq(
            market_name=name,
            df=market_dfs_1h[name],
            exit_confirm_bars=10,
            adx_threshold=15,
            ema_fast_len=70,
            ema_slow_len=120,
        )
        if not t_tf_eq.empty:
            all_trades.append(t_tf_eq)

        t_mr_eq_daily = generate_trades_mr_eq_daily(
            market_name=name,
            df=market_dfs_1d[name],
            ema_fast_len=20,
            ema_slow_len=250,
            pullback_frac=0.20,
        )
        if not t_mr_eq_daily.empty:
            t_mr_eq_1h = align_trades_to_hourly_opens_for_index(t_mr_eq_daily, market_dfs_1h[name])
            all_trades.append(t_mr_eq_1h)

    # FX TF
    for m in FX_TF_MARKETS:
        t_tf_fx = generate_trades_tf_fx(
            market_name=m["name"],
            df=market_dfs_1h[m["name"]],
            pip_size=float(m["pip_size"]),
            spread_points_per_pip=float(m["spread_points_per_pip"]),
            cost_model=m["cost_model"],
        )
        if not t_tf_fx.empty:
            all_trades.append(t_tf_fx)

    # FX MR
    for m in MR_FX_TRADE_MARKETS:
        name = m["name"]
        t_mr_fx = generate_trades_mr_fx(
            market_name=name,
            df=market_dfs_1h[name],
            pip_size=float(MR_FX_PIP_SIZE[name]),
            spread_points_per_pip=10.0,
        )
        if not t_mr_fx.empty:
            all_trades.append(t_mr_fx)

    if not all_trades:
        raise RuntimeError("Inga trades genererades.")

    portfolio_trades = pd.concat(all_trades, ignore_index=True)
    portfolio_trades["Entry Fill Time"] = pd.to_datetime(portfolio_trades["Entry Fill Time"])
    portfolio_trades["Exit Fill Time"] = pd.to_datetime(portfolio_trades["Exit Fill Time"])
    portfolio_trades = portfolio_trades.sort_values(["Entry Fill Time", "Market", "Strategy"]).reset_index(drop=True)

    print("\n--- TRADE COUNTS BY STRATEGY ---")
    print(portfolio_trades.groupby("Strategy").size())

    print("\n--- TRADE COUNTS BY STRATEGY x MARKET ---")
    print(portfolio_trades.groupby(["Strategy", "Market"]).size())

    # ============================================================
    # REGIME OVERLAY INPUT
    # ============================================================

    regime_series_daily = build_vol_regime_series_from_equity_proxy(
        close_series=market_dfs_1h["US500"]["close"],
        lookback=REGIME_LOOKBACK_DAYS,
    )

    print("\n--- REGIME OVERLAY DAY COUNTS ---")
    print(regime_series_daily.value_counts())

    # ============================================================
    # MARKET-SPECIFIC VOL REGIMES
    # ============================================================

    market_regime_series_daily = build_market_vol_regime_series(
        market_dfs_1h=market_dfs_1h,
        lookback=REGIME_LOOKBACK_DAYS,
    )

    print("\n--- MARKET REGIME DAY COUNTS (sample) ---")
    for m in ["US500", "US100", "US30", "EURUSD", "USDCHF", "EURCHF", "GBPCHF", "EURCAD"]:
        if m in market_regime_series_daily:
            print(f"\n{m}")
            print(market_regime_series_daily[m].value_counts())


    print("\n--- EXTREMEVOL FRACTION BY MARKET ---")
    for m in ["US500", "US100", "US30", "EURUSD", "USDCHF", "EURCHF", "GBPCHF", "EURCAD"]:
        s = market_regime_series_daily.get(m)
        if s is None or len(s) == 0:
            continue
        frac = (s == "ExtremeVol").mean()
        print(f"{m}: {frac:.2%}")

    print("\n--- REGIME MARKET MULTIPLIERS ---")
    for k, v in REGIME_MARKET_MULTIPLIERS.items():
        print(f"{k}: {v}")

    # ----------------------------
    # 3) Portfolio simulation
    # ----------------------------
    (
        equity_series,
        realized_equity_series,
        daily_returns,
        open_pos_series,
        gross_exposure_series,
        strategy_equity_df,
        strategy_pnl_1h_df,
        strategy_pnl_1d_df,
        strategy_gross_df,
    ) = build_portfolio_mtm_usd_multi_strategy(
        market_dfs_1h=market_dfs_1h,
        trades_df=portfolio_trades,
        start_capital=START_CAPITAL,
        max_gross_exposure=MAX_GROSS_EXPOSURE_TOTAL,
        strategy_targets=STRATEGY_TARGETS,
        strategy_market_weights=STRATEGY_MARKET_WEIGHTS,
        market_regime_series_daily=market_regime_series_daily,
    )

    def summarize_strategy_attribution(strategy_pnl_1d_df: pd.DataFrame) -> pd.DataFrame:
        pnl = strategy_pnl_1d_df.drop(columns="TOTAL").copy()

        out = pd.DataFrame(index=pnl.columns)
        out["Total PnL"] = pnl.sum()
        out["Avg Daily PnL"] = pnl.mean()
        out["Daily Vol"] = pnl.std(ddof=1)
        out["Best Day"] = pnl.max()
        out["Worst Day"] = pnl.min()
        out["Positive Days %"] = (pnl > 0).mean()
        out["PnL Contribution %"] = out["Total PnL"] / out["Total PnL"].sum()

        sharpe_like = out["Avg Daily PnL"] / out["Daily Vol"]
        out["Sharpe-like Daily"] = sharpe_like.replace([np.inf, -np.inf], np.nan)

        return out.sort_values("Total PnL", ascending=False)

    attr_table = summarize_strategy_attribution(strategy_pnl_1d_df)
    print("\n--- STRATEGY ATTRIBUTION ---")
    print(attr_table)

    print("\n--- STRATEGY CORRELATION (DAILY PnL) ---")
    print(strategy_pnl_1d_df.drop(columns="TOTAL").corr())

    plt.figure(figsize=(12, 5))
    for col in strategy_equity_df.columns:
        plt.plot(strategy_equity_df.index, strategy_equity_df[col], label=col)
    plt.title("Equity by Strategy")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 5))
    for col in strategy_pnl_1d_df.drop(columns="TOTAL").columns:
        plt.plot(strategy_pnl_1d_df.index, strategy_pnl_1d_df[col].cumsum(), label=col)
    plt.title("Cumulative Daily PnL by Strategy")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # ----------------------------
    # 4) Metrics
    # ----------------------------
    metrics = portfolio_metrics_from_equity(equity_series, daily_returns)
    mtm_dd = max_drawdown_pct(equity_series)
    closed_dd = max_drawdown_pct(realized_equity_series)

    print("\n--- COMBINED PORTFOLIO METRICS (USD, 1H master) ---")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")

    print("\nSanity: Open positions at end:", int(open_pos_series.iloc[-1]))
    print("Avg gross exposure %:", float(gross_exposure_series.mean() * 100.0))
    print("Max gross exposure %:", float(gross_exposure_series.max() * 100.0))
    print(f"Max DD% (MTM equity): {mtm_dd * 100:.2f}%")
    print(f"Max DD% (Closed trades only): {closed_dd * 100:.2f}%")

    # ============================================================
    # REGIME / CRISIS ANALYSIS
    # ============================================================
    '''
    vol_regimes = build_vol_regime_series(equity_series, lookback=20)

    vol_table = evaluate_by_vol_regime(equity_series, vol_regimes)
    print("\n--- PORTFOLIO METRICS BY VOL REGIME ---")
    print(vol_table)

    crisis_table = evaluate_crisis_periods(equity_series, CRISIS_PERIODS)
    print("\n--- PORTFOLIO METRICS BY CRISIS PERIOD ---")
    print(crisis_table)

    strat_regime = strategy_regime_table(strategy_pnl_1d_df, vol_regimes)
    print("\n--- STRATEGY METRICS BY VOL REGIME ---")
    print(strat_regime)

    print("\n--- VOL REGIME DAY COUNTS ---")
    print(vol_regimes.value_counts())

    market_regime = market_regime_table(
        trades_df=portfolio_trades,
        market_dfs_1h=market_dfs_1h,
        strategy_targets=STRATEGY_TARGETS,
        strategy_market_weights=STRATEGY_MARKET_WEIGHTS,
        start_capital=START_CAPITAL,
        max_gross_exposure=MAX_GROSS_EXPOSURE_TOTAL,
        regime_series=vol_regimes,
    )
    
    print("\n--- MARKET METRICS BY VOL REGIME ---")
    print(market_regime.to_string(index=False))
    
    
    market_regime_table = market_regime_table_by_own_vol(
        trades_df=portfolio_trades,
        market_dfs_1h=market_dfs_1h,
        strategy_targets=STRATEGY_TARGETS,
        strategy_market_weights=STRATEGY_MARKET_WEIGHTS,
        start_capital=START_CAPITAL,
        max_gross_exposure=MAX_GROSS_EXPOSURE_TOTAL,
        market_regime_series_daily=market_regime_series_daily,
    )
    
    print("\n--- MARKET METRICS BY OWN VOL REGIME ---")
    print(market_regime_table.to_string(index=False))
    '''
    # ----------------------------
    # 5) FTMO analysis
    # ----------------------------
    intraday_dd_df = compute_intraday_drawdowns(equity_series)
    daily_loss_stats = daily_loss_statistics(
        intraday_dd_df,
        soft_limit=-0.05,
        hard_limit=-0.10,
    )

    print("\n--- FTMO DAILY LOSS ANALYSIS (MTM, 1H) ---")
    for k, v in daily_loss_stats.items():
        if isinstance(v, float):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")

    balance_1h = realized_equity_series.reindex(equity_series.index).ffill()

    day_lib = build_day_path_library(
        equity_1h=equity_series,
        balance_1h=balance_1h,
        trades_df=portfolio_trades,
    )

    print(day_lib.head())

    # --- SANITY CHECK: daily returns vs bootstrap day-end returns ---
    actual_daily = equity_series.resample("1D").last().dropna().pct_change().dropna()

    day_end_returns = pd.Series(
        [float(row["DayEndReturnFromEquityOpen"]) for _, row in day_lib.iterrows()],
        index=day_lib.index,
        name="BootstrapDayEndReturn"
    )

    print("\n--- SANITY CHECK: ACTUAL DAILY RETURNS ---")
    print(actual_daily.describe(percentiles=[0.05, 0.5, 0.95]))

    print("\n--- SANITY CHECK: BOOTSTRAP DAY-END RETURNS ---")
    print(day_end_returns.describe(percentiles=[0.05, 0.5, 0.95]))

    params2 = FTMOParams2(
        challenge_target=CHALLENGE_TARGET,
        verification_target=VERIFICATION_TARGET,
        daily_loss_limit=DAILY_LOSS_LIMIT,
        max_loss_limit=MAX_LOSS_LIMIT,
        min_trading_days_eval=MIN_TRADING_DAYS_EVAL,
        payout_wait_days=PAYOUT_WAIT_DAYS,
        soft_cutoff=SOFT_CUTOFF_DAILY,
        eval_exposure_factor=EVAL_EXPOSURE_FACTOR,
        funded_exposure_factor=FUNDED_EXPOSURE_FACTOR,
    )
    '''
    sim_debug = run_bootstrap_ftmo_pipeline_daypaths(
        day_lib=day_lib,
        start_balance=START_CAPITAL,
        n_iter=100,
        horizon_days=BOOT_HORIZON_DAYS,
        block_len=BOOT_BLOCK_LEN,
        seed=BOOT_SEED,
        params=params2,
    )

    print(sim_debug.columns.tolist())

    exposure_opt = optimize_dynamic_exposure_map(
        day_lib=day_lib,
        params=params2,
        start_balance=START_CAPITAL,
        n_iter=3000,
        horizon_days=BOOT_HORIZON_DAYS,
        block_len=BOOT_BLOCK_LEN,
        seed=BOOT_SEED,
        candidate_values=[0.40, 0.80, 1.00, 1.25],
        max_configs=80,
    )

    print("\n--- DYNAMIC EXPOSURE OPTIMIZATION: TOP 20 ---")
    print(exposure_opt.head(20).to_string(index=False))

    
    best_exposure_map = {
        "<-7%": 0.40,
        "-7%_-4%": 0.60,
        "-4%_-2%": 0.80,
        "-2%_0%": 1.00,
        "0%_2.5%": 1.00,
        "2.5%_5%": 1.25,
        ">5%": 1.25,
    }

    sim_best = run_bootstrap_ftmo_pipeline_daypaths(
        day_lib=day_lib,
        params=params2,
        exposure_map=best_exposure_map,
    )
    '''

    sim2 = run_bootstrap_ftmo_pipeline_daypaths(
        day_lib=day_lib,
        start_balance=START_CAPITAL,
        n_iter=BOOT_N_ITER,
        horizon_days=BOOT_HORIZON_DAYS,
        block_len=BOOT_BLOCK_LEN,
        seed=BOOT_SEED,
        params=params2,
        exposure_map=BEST_EXPOSURE_MAP,  # 🔥 viktigt
    )

    p_ch = float(sim2["passed_challenge"].mean())
    p_ver = float(sim2["passed_verification"].mean())
    p_pay = float(sim2["reached_payout"].mean())

    print("\n--- FTMO PIPELINE (DayPath Bootstrap) ---")
    print(f"Cutoff: {SOFT_CUTOFF_DAILY:.2%}, Eval factor: {EVAL_EXPOSURE_FACTOR:.2f}, Funded factor: {FUNDED_EXPOSURE_FACTOR:.2f}")
    print(f"P(Pass Challenge):    {p_ch:.4f}")
    print(f"P(Pass Verification): {p_ver:.4f}")
    print(f"P(Reach 1st payout):  {p_pay:.4f}")

    print("\n--- TIME TO EVENT (days) ---")
    print("Challenge:", summarize_times(sim2[sim2["passed_challenge"]], "days_to_challenge"))
    print("Verification:", summarize_times(sim2[sim2["passed_verification"]], "days_to_verification"))
    print("Payout:", summarize_times(sim2[sim2["reached_payout"]], "days_to_payout"))

    print("\n--- FAIL STAGE COUNTS ---")
    print(sim2["fail_stage"].value_counts(dropna=False))

    print("\n--- FAIL REASON COUNTS (non-null) ---")
    print(sim2["fail_reason"].dropna().value_counts())

    # ============================================================
    # OVERLAY GRID SEARCH
    # ============================================================

    RUN_OVERLAY_GRID_SEARCH = False

    if RUN_OVERLAY_GRID_SEARCH:
        grid_results = run_overlay_grid_search(
            market_dfs_1h=market_dfs_1h,
            portfolio_trades=portfolio_trades,
            strategy_targets=STRATEGY_TARGETS,
            strategy_market_weights=STRATEGY_MARKET_WEIGHTS,
            start_capital=START_CAPITAL,
            max_gross_exposure_total=MAX_GROSS_EXPOSURE_TOTAL,
            ftmo_params=params2,
            boot_n_iter_search=3000,   # snabbare sökning
            boot_horizon_days=BOOT_HORIZON_DAYS,
            boot_block_len=BOOT_BLOCK_LEN,
            boot_seed=BOOT_SEED,
        )

        print("\n--- OVERLAY GRID SEARCH: TOP 20 BY PORTFOLIO SCORE ---")
        print(grid_results.head(20).to_string(index=False))

        print("\n--- OVERLAY GRID SEARCH: TOP 20 BY FTMO SCORE ---")
        print(
            grid_results.sort_values("FTMO Score", ascending=False)
            .head(20)
            .to_string(index=False)
        )

    # ----------------------------
    # 6) Plots
    # ----------------------------
    plt.figure(figsize=(12, 5))
    plt.plot(equity_series.index, equity_series.values, label="MTM Equity")
    plt.plot(realized_equity_series.index, realized_equity_series.values, label="Realized Equity", alpha=0.8)
    plt.title("Combined Portfolio Equity (TF_EQ + MR_EQ + TF_FX + MR_FX)")
    plt.xlabel("Time")
    plt.ylabel("Equity ($)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 3))
    plt.plot(open_pos_series.index, open_pos_series.values)
    plt.title("Open Positions")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 3))
    plt.plot(gross_exposure_series.index, gross_exposure_series.values)
    plt.title("Gross Exposure % (Total)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 4))
    plt.hist(intraday_dd_df["MinIntradayDD"], bins=50)
    plt.axvline(-0.05, color="orange", linestyle="--", label="-5%")
    plt.axvline(-0.10, color="red", linestyle="--", label="-10%")
    plt.title("Worst Intraday Drawdown per Day (MTM)")
    plt.xlabel("Intraday Drawdown")
    plt.ylabel("Days")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    print("\n--- WORST INTRADAY DD DAYS ---")
    print(intraday_dd_df.sort_values("MinIntradayDD").head(10))

if __name__ == "__main__":
    main()
