import json
import time
from datetime import datetime, date
from typing import Optional, Tuple

from datetime import timezone, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import ta
import MetaTrader5 as mt5

from MT5exec_USequities_Forex import (
    ExecConfig,
    ensure_initialized,
    ensure_symbol,
    account_snapshot,
    get_position,
    close_position_market,
    open_long_by_notional,
    open_short_by_notional,
)

BROKER_TZ = timezone(timedelta(hours=3))   # fast UTC+3 året runt
LOCAL_TZ = ZoneInfo("Europe/Stockholm")

def broker_ts_to_stockholm(ts: pd.Timestamp) -> pd.Timestamp:
    """
    Tolkar naiv MT5-timestamp som broker time (fast UTC+2)
    och konverterar till Europe/Stockholm.
    """
    if ts.tzinfo is None:
        ts = ts.tz_localize(BROKER_TZ)
    else:
        ts = ts.tz_convert(BROKER_TZ)

    return ts.tz_convert(LOCAL_TZ)

def log_bar_times(label: str, ts: pd.Timestamp):
    st = broker_ts_to_stockholm(ts)
    print(f"[{now_str()}] {label} broker_ts={ts} stockholm_ts={st}")

# ==========================
# CONFIG
# ==========================

STATE_FILE = "live_state.json"

TF_EQ_MARKETS = [
    {"name": "US500", "symbol": "US500.cash"},
    {"name": "US100", "symbol": "US100.cash"},
    {"name": "US30",  "symbol": "US30.cash"},
]

MR_EQ_MARKETS = [
    {"name": "US500", "symbol": "US500.cash"},
    {"name": "US100", "symbol": "US100.cash"},
    {"name": "US30",  "symbol": "US30.cash"},
]

TF_FX_MARKETS = [
    {"name": "EURJPY", "symbol": "EURJPY"},
    {"name": "GBPJPY", "symbol": "GBPJPY"},
    {"name": "USDJPY", "symbol": "USDJPY"},
]

MR_FX_MARKETS = [
    {"name": "EURCHF", "symbol": "EURCHF"},
    {"name": "EURCAD", "symbol": "EURCAD"},
    {"name": "GBPCHF", "symbol": "GBPCHF"},
    {"name": "EURUSD", "symbol": "EURUSD"},
    {"name": "USDCHF", "symbol": "USDCHF"},
]

MAX_GROSS_EXPOSURE_TOTAL = 12.5

STRATEGY_TARGETS_BASE = {
    "TF_EQ": 1.125,
    "MR_EQ": 0.875,
    "TF_FX": 2.5,
    "MR_FX": 8.0,
}

TF_EQ_MARKET_WEIGHTS = {
    "US500": 0.34,
    "US100": 0.29,
    "US30": 0.37,
}

MR_EQ_MARKET_WEIGHTS = {
    "US500": 1 / 3,
    "US100": 1 / 3,
    "US30": 1 / 3,
}

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

USE_REGIME_OVERLAY = True
REGIME_LOOKBACK_DAYS = 20

REGIME_MARKET_MULTIPLIERS = {
    ("ExtremeVol", "TF_EQ", "US100"): 0.50,
    ("ExtremeVol", "TF_EQ", "US500"): 1.00,
    ("ExtremeVol", "TF_EQ", "US30"): 1.00,

    ("ExtremeVol", "MR_FX", "USDCHF"): 1.00,
    ("ExtremeVol", "MR_FX", "EURUSD"): 1.00,
    ("ExtremeVol", "MR_FX", "EURCHF"): 1.00,
    ("ExtremeVol", "MR_FX", "GBPCHF"): 1.00,
    ("ExtremeVol", "MR_FX", "EURCAD"): 0.65,

    ("LowVol", "TF_FX", "EURJPY"): 0.35,
    ("LowVol", "TF_FX", "GBPJPY"): 0.35,
    ("LowVol", "TF_FX", "USDJPY"): 1.00,
}

MODE = "EVAL"

EVAL_EXPOSURE_FACTOR = 2.00
FUNDED_EXPOSURE_FACTOR = 1.00

SOFT_CUTOFF_DAILY = -0.04

START_BALANCE_FOR_LIMITS = 50_000
DAILY_LOSS_LIMIT_ABS_FRAC = 0.05
MAX_LOSS_LIMIT_ABS_FRAC = 0.10

TF_EQ_PARAMS = dict(
    exit_confirm_bars=10,
    adx_threshold=15,
    ema_fast_len=70,
    ema_slow_len=120,
)

MR_EQ_PARAMS = dict(
    ema_fast_len=20,
    ema_slow_len=250,
    pullback_frac=0.20,
)

TF_FX_PARAMS = dict(
    session_start="08:00:00",
    session_end="09:00:00",
)

MR_FX_PARAMS = dict(
    session_start="00:00:00",
    session_end="07:00:00",
    vwap_reset="20:00:00",
    entry_std=2.25,
    exit_std=0.75,
)

H1_BARS = 900
D1_BARS = 500

SLEEP_SECONDS = 1.0
HEARTBEAT_EVERY_SEC = 60
HALF = 0.5


# ==========================
# STATE
# ==========================

def load_state():
    try:
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return {}


def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def now_str():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ==========================
# MT5 DATA
# ==========================

def mt5_timeframe(tf):
    return mt5.TIMEFRAME_H1 if tf == "H1" else mt5.TIMEFRAME_D1


def fetch_ohlc(symbol, tf, n, min_bars=120):
    ensure_symbol(symbol)

    rates = mt5.copy_rates_from_pos(symbol, mt5_timeframe(tf), 0, n)
    if rates is None or len(rates) < min_bars:
        return pd.DataFrame()

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df = df.set_index("time")

    cols = ["open", "high", "low", "close"]
    if "tick_volume" in df.columns:
        cols.append("tick_volume")
    if "real_volume" in df.columns:
        cols.append("real_volume")
    if "spread" in df.columns:
        cols.append("spread")

    return df[cols].copy()


def last_closed_bar(df):
    return df.iloc[-2]


def prev_closed_bar(df):
    return df.iloc[-3]


# ==========================
# MAGIC
# ==========================
MAGIC_MAP = {
    ("US500", "TF_EQ"): 11001,
    ("US100", "TF_EQ"): 11002,
    ("US30",  "TF_EQ"): 11003,

    ("US500", "MR_EQ"): 21001,
    ("US100", "MR_EQ"): 21002,
    ("US30",  "MR_EQ"): 21003,

    ("EURJPY", "TF_FX"): 31001,
    ("GBPJPY", "TF_FX"): 31002,
    ("USDJPY", "TF_FX"): 31003,

    ("EURCHF", "MR_FX"): 41001,
    ("EURCAD", "MR_FX"): 41002,
    ("GBPCHF", "MR_FX"): 41003,
    ("EURUSD", "MR_FX"): 41004,
    ("USDCHF", "MR_FX"): 41005,
}


def magic_for(market_name, strategy):
    return MAGIC_MAP[(market_name, strategy)]


# ==========================
# HELPERS
# ==========================

def clamp_time_series_index_unique(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_index()
    if not df.index.has_duplicates:
        return df

    agg = {"open": "first", "high": "max", "low": "min", "close": "last"}
    if "tick_volume" in df.columns:
        agg["tick_volume"] = "sum"
    if "real_volume" in df.columns:
        agg["real_volume"] = "sum"
    if "spread" in df.columns:
        agg["spread"] = "last"

    return df.groupby(df.index).agg(agg).sort_index()


def in_session(ts: pd.Timestamp, session_start: str, session_end: str) -> bool:
    """
    Sessionerna definieras i Stockholm-tid.
    MT5-barens timestamp tolkas först som broker time (fast UTC+2),
    och konverteras sedan till Stockholm-tid.
    """
    start_t = pd.to_datetime(session_start).time()
    end_t = pd.to_datetime(session_end).time()
    local_ts = broker_ts_to_stockholm(ts)
    t = local_ts.time()

    if start_t < end_t:
        return (t >= start_t) and (t < end_t)

    return (t >= start_t) or (t < end_t)


def position_direction(pos) -> Optional[str]:
    if pos is None:
        return None
    if pos.type == mt5.POSITION_TYPE_BUY:
        return "LONG"
    if pos.type == mt5.POSITION_TYPE_SELL:
        return "SHORT"
    return None


def pct_rank_last(x):
    s = pd.Series(x)
    return s.rank(pct=True).iloc[-1]


def ATR(df_in: pd.DataFrame, period: int = 14, method: str = "wilder") -> pd.Series:
    high = df_in["high"]
    low = df_in["low"]
    close = df_in["close"]
    prev_close = close.shift(1)

    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)

    if method.lower() == "sma":
        return tr.rolling(window=period, min_periods=period).mean()

    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def asia_range(df_in: pd.DataFrame) -> pd.DataFrame:
    """
    Bygger Asia range på Stockholm-tid, även om df.index är broker/MT5-tid.
    Behåller originalindex i output.
    """
    data = df_in.copy()

    if not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError("DataFrame måste ha DateTimeIndex")

    # Konvertera varje broker-ts till Stockholm-ts
    stockholm_index = pd.DatetimeIndex([broker_ts_to_stockholm(ts) for ts in data.index])

    # Temporary helper frame with Stockholm index for session slicing
    tmp = data.copy()
    tmp["stockholm_ts"] = stockholm_index
    tmp = tmp.set_index("stockholm_ts")

    # Asia session in Stockholm time
    session = tmp.between_time("00:00", "07:00")

    # Group by Stockholm calendar day
    daily_range = session.groupby(session.index.date).agg(
        asia_high=("high", "max"),
        asia_low=("low", "min")
    )

    # Map back to original dataframe via Stockholm calendar date
    data["stockholm_date"] = stockholm_index.date
    data = data.merge(
        daily_range,
        left_on="stockholm_date",
        right_index=True,
        how="left"
    )

    data["asia_range"] = data["asia_high"] - data["asia_low"]
    data["asia_mid"] = (data["asia_high"] + data["asia_low"]) / 2.0

    # For bars before 08:00 Stockholm, hide same-day Asia values
    before_8 = pd.Index(stockholm_index.hour) < 8
    data.loc[before_8, ["asia_high", "asia_low", "asia_range", "asia_mid"]] = np.nan

    data.drop(columns="stockholm_date", inplace=True)

    return data


def compute_session_anchored_vwap_and_std(data: pd.DataFrame, vol_col: str, reset_time: str):
    df_v = data.copy()

    tp = (df_v["high"] + df_v["low"] + df_v["close"]) / 3.0
    vol = df_v[vol_col].astype(float).fillna(0.0)

    df_v["tp"] = tp
    df_v["vol"] = vol
    df_v["tp_vol"] = df_v["tp"] * df_v["vol"]

    stockholm_index = pd.DatetimeIndex([broker_ts_to_stockholm(ts) for ts in df_v.index])

    rt = pd.to_datetime(reset_time).time()

    session_date = stockholm_index.floor("D")
    session_date = session_date.where(stockholm_index.time >= rt, session_date - pd.Timedelta(days=1))
    df_v["session_date"] = session_date

    g = df_v.groupby("session_date", sort=False)

    df_v["cum_tp_vol"] = g["tp_vol"].cumsum()
    df_v["cum_vol"] = g["vol"].cumsum()

    vwap = df_v["cum_tp_vol"] / df_v["cum_vol"].replace(0.0, np.nan)
    std = g["tp"].transform(lambda x: x.expanding().std(ddof=0))

    return vwap, std


# ==========================
# INDICATORS
# ==========================

def compute_tf_eq_indicators(df):
    df = df.copy()
    df["ema_fast"] = df["close"].ewm(span=TF_EQ_PARAMS["ema_fast_len"], adjust=False).mean()
    df["ema_slow"] = df["close"].ewm(span=TF_EQ_PARAMS["ema_slow_len"], adjust=False).mean()

    df["adx"] = ta.trend.ADXIndicator(
        df["high"], df["low"], df["close"], window=14
    ).adx()

    return df


def compute_mr_eq_indicators(df):
    df = df.copy()
    df["ema_fast"] = df["close"].ewm(span=MR_EQ_PARAMS["ema_fast_len"], adjust=False).mean()
    df["ema_slow"] = df["close"].ewm(span=MR_EQ_PARAMS["ema_slow_len"], adjust=False).mean()
    return df


def compute_tf_fx_indicators(df):
    df = clamp_time_series_index_unique(df.copy())
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

    df["atr"] = ATR(df, period=14, method="wilder")
    df["atr_ma"] = df["atr"].rolling(75).mean()

    return df


def compute_mr_fx_indicators(df):
    df = clamp_time_series_index_unique(df.copy())

    if "real_volume" in df.columns and df["real_volume"].sum() > 0:
        vol_col = "real_volume"
    elif "tick_volume" in df.columns:
        vol_col = "tick_volume"
    else:
        df["tick_volume"] = 1.0
        vol_col = "tick_volume"

    df["VWAP"], df["TP_STD"] = compute_session_anchored_vwap_and_std(
        df, vol_col, MR_FX_PARAMS["vwap_reset"]
    )
    df["VWAP_prev"] = df["VWAP"].shift(1)
    df["TP_STD_prev"] = df["TP_STD"].shift(1)

    return df


# ==========================
# SIGNALS
# ==========================

def tf_eq_entry(prev, prev2):
    adx_ok = prev["adx"] > TF_EQ_PARAMS["adx_threshold"]
    cross = (prev2["ema_fast"] < prev2["ema_slow"]) and (prev["ema_fast"] > prev["ema_slow"])
    return bool(adx_ok and cross)


def tf_eq_exit(prev):
    return prev["ema_fast"] <= prev["ema_slow"]


def mr_eq_entry(prev):
    close = prev["close"]
    ema_fast = prev["ema_fast"]
    ema_slow = prev["ema_slow"]

    high = prev["high"]
    low = prev["low"]

    regime = (close < ema_fast) and (close > ema_slow)
    deep = close < (low + MR_EQ_PARAMS["pullback_frac"] * (high - low))

    return bool(regime and deep)


def mr_eq_exit(prev):
    return prev["high"] >= prev["ema_fast"]


def tf_fx_entry(market_name: str, prev):
    if not in_session(prev.name, TF_FX_PARAMS["session_start"], TF_FX_PARAMS["session_end"]):
        return None

    if not np.isfinite(prev["asia_high"]) or not np.isfinite(prev["asia_low"]):
        return None
    if not np.isfinite(prev["atr"]) or not np.isfinite(prev["atr_ma"]):
        return None
    if not np.isfinite(prev["asia_range_pct_rank"]):
        return None

    atr_filter = prev["atr"] > prev["atr_ma"]
    bullish_breakout = prev["close"] > prev["asia_high"]
    bearish_breakout = prev["close"] < prev["asia_low"]

    if market_name == "GBPJPY":
        long_ok = bullish_breakout and atr_filter and prev["asia_range_pct_rank"] > 0.75
        short_ok = bearish_breakout and atr_filter and prev["asia_range_pct_rank"] < 0.65
    else:
        long_ok = bullish_breakout and atr_filter and prev["asia_range_pct_rank"] > 0.70
        short_ok = bearish_breakout and atr_filter and prev["asia_range_pct_rank"] > 0.70

    if long_ok:
        return "LONG"
    if short_ok:
        return "SHORT"
    return None


def tf_fx_exit(prev, direction: str) -> bool:
    if not np.isfinite(prev["atr"]) or not np.isfinite(prev["atr_ma"]):
        return False
    return bool(prev["atr"] < prev["atr_ma"])


def mr_fx_entry(prev, prev2):
    if not in_session(prev.name, MR_FX_PARAMS["session_start"], MR_FX_PARAMS["session_end"]):
        return None

    if not np.isfinite(prev["VWAP"]) or not np.isfinite(prev["TP_STD"]) or prev["TP_STD"] == 0:
        return None
    if not np.isfinite(prev2["VWAP"]) or not np.isfinite(prev2["TP_STD"]) or prev2["TP_STD"] == 0:
        return None

    entry_std = MR_FX_PARAMS["entry_std"]

    upper_band = prev["VWAP"] + entry_std * prev["TP_STD"]
    lower_band = prev["VWAP"] - entry_std * prev["TP_STD"]

    prev_upper = prev2["VWAP"] + entry_std * prev2["TP_STD"]
    prev_lower = prev2["VWAP"] - entry_std * prev2["TP_STD"]

    upper_break = (prev2["close"] < prev_upper) and (prev["close"] > upper_band)
    lower_break = (prev2["close"] > prev_lower) and (prev["close"] < lower_band)

    if lower_break:
        return "LONG"
    if upper_break:
        return "SHORT"
    return None


def mr_fx_exit(prev, direction: str) -> bool:
    if not np.isfinite(prev["VWAP"]) or not np.isfinite(prev["TP_STD"]) or prev["TP_STD"] == 0:
        return False

    exit_std = MR_FX_PARAMS["exit_std"]

    if direction == "LONG":
        final_level = prev["VWAP"] - exit_std * prev["TP_STD"]
        return bool(prev["close"] >= final_level)

    final_level = prev["VWAP"] + exit_std * prev["TP_STD"]
    return bool(prev["close"] <= final_level)


# ==========================
# EXPOSURE / OVERLAY
# ==========================

def strategy_targets():
    factor = EVAL_EXPOSURE_FACTOR if MODE == "EVAL" else FUNDED_EXPOSURE_FACTOR
    return {k: v * factor for k, v in STRATEGY_TARGETS_BASE.items()}


def open_positions_gross_notional():
    gross = 0.0
    pos = mt5.positions_get()

    if pos is None:
        return 0.0

    for p in pos:
        gross += abs(float(p.price_current) * float(p.volume))

    return gross


def remaining_capacity(equity):
    gross = open_positions_gross_notional()
    cap = equity * MAX_GROSS_EXPOSURE_TOTAL
    return max(0.0, cap - gross)


def current_market_vol_regime(symbol: str, lookback_days: int = REGIME_LOOKBACK_DAYS) -> Optional[str]:
    df = fetch_ohlc(symbol, "D1", D1_BARS, min_bars=max(lookback_days + 30, 80))
    if df.empty or len(df) < lookback_days + 20:
        return None

    daily_close = df["close"].dropna()
    daily_ret = daily_close.pct_change().dropna()
    if len(daily_ret) < lookback_days + 10:
        return None

    vol = daily_ret.rolling(lookback_days).std().dropna()
    if vol.empty:
        return None

    v = float(vol.iloc[-1])
    q1 = float(vol.quantile(0.25))
    q2 = float(vol.quantile(0.50))
    q3 = float(vol.quantile(0.75))

    if v <= q1:
        return "LowVol"
    if v <= q2:
        return "MidVol"
    if v <= q3:
        return "HighVol"
    return "ExtremeVol"


def get_cached_market_regime(state, market_name: str, symbol: str) -> Optional[str]:
    cache = state.setdefault("market_regime_cache", {})
    today = date.today().isoformat()

    rec = cache.get(market_name)
    if rec and rec.get("date") == today:
        return rec.get("regime")

    regime = current_market_vol_regime(symbol)
    cache[market_name] = {
        "date": today,
        "regime": regime,
    }
    return regime


def strategy_target_with_regime_overlay(
    strategy: str,
    market: str,
    base_target: float,
    market_regime_label: Optional[str],
) -> float:
    target = float(base_target)

    if not USE_REGIME_OVERLAY:
        return target

    if market_regime_label is None:
        return target

    mult = REGIME_MARKET_MULTIPLIERS.get(
        (str(market_regime_label), str(strategy), str(market)),
        1.0,
    )
    return target * float(mult)


def desired_notional(
    equity: float,
    strat: str,
    market: str,
    market_regime_label: Optional[str] = None,
) -> float:
    targets = strategy_targets()
    base_target = float(targets[strat])

    adj_target = strategy_target_with_regime_overlay(
        strategy=strat,
        market=market,
        base_target=base_target,
        market_regime_label=market_regime_label,
    )

    w = float(STRATEGY_MARKET_WEIGHTS[strat][market])
    return float(equity * adj_target * w)


def overlay_info(strategy: str, market: str, regime: Optional[str], equity: float) -> Tuple[float, float]:
    targets = strategy_targets()
    base_target = float(targets[strategy])
    w = float(STRATEGY_MARKET_WEIGHTS[strategy][market])

    base_notional = equity * base_target * w
    adj_target = strategy_target_with_regime_overlay(strategy, market, base_target, regime)
    adj_notional = equity * adj_target * w

    return float(base_notional), float(adj_notional)


# ==========================
# RISK
# ==========================
def update_day_peak_balance_reference(state, balance: float):
    """
    FTMO-style internal risk reference:
    starts from balance at midnight, but if balance increases during the day,
    the reference ratchets upward.
    """
    ref = state.get("day_peak_balance_reference")
    if ref is None:
        state["day_peak_balance_reference"] = float(balance)
        return

    if float(balance) > float(ref):
        state["day_peak_balance_reference"] = float(balance)

def day_drawdown(day_start, equity):
    return equity / day_start - 1


def risk_gate(state, snap, cfg=None):
    """
    Returns:
      allow_entries: bool
      must_flatten: bool

    Rule:
      Flatten all if equity is <= 4% below today's highest balance reference.
      Balance reference starts at midnight balance and ratchets upward if
      account balance increases during the day.
    """
    equity = float(snap["equity"])
    balance = float(snap["balance"])

    # Update today's highest balance reference
    update_day_peak_balance_reference(state, balance)

    ref_balance = float(state.get("day_peak_balance_reference", balance))
    day_start_balance = float(state.get("day_start_balance", balance))

    # Internal daily kill threshold: -4% from intraday highest balance reference
    soft_floor = ref_balance * (1.0 + SOFT_CUTOFF_DAILY)  # SOFT_CUTOFF_DAILY = -0.04

    # Absolute max loss floor (still keep this if you want account-wide protection)
    max_floor = START_BALANCE_FOR_LIMITS * (1.0 - MAX_LOSS_LIMIT_ABS_FRAC)

    must_flatten = False
    allow_entries = True

    # Already flattened once today -> no new entries rest of day
    if state.get("risk_flattened_today", False):
        allow_entries = False

    # Daily rule breach based on equity vs intraday peak balance reference
    if equity <= soft_floor:
        must_flatten = True
        allow_entries = False

    # Hard max loss protection
    if equity <= max_floor:
        must_flatten = True
        allow_entries = False

    print(
        f"[RISK] balance={balance:.2f} equity={equity:.2f} "
        f"day_start_balance={day_start_balance:.2f} "
        f"peak_balance_ref={ref_balance:.2f} "
        f"soft_floor={soft_floor:.2f} max_floor={max_floor:.2f} "
        f"allow_entries={allow_entries} must_flatten={must_flatten}"
    )

    return allow_entries, must_flatten

def flatten_all_known_positions(cfg):
    """
    Closes all positions managed by this bot, across all strategies/markets.
    """
    all_markets = TF_EQ_MARKETS + MR_EQ_MARKETS + TF_FX_MARKETS + MR_FX_MARKETS
    seen = set()

    for m in all_markets:
        name = m["name"]
        sym = m["symbol"]

        for strat in ["TF_EQ", "MR_EQ", "TF_FX", "MR_FX"]:
            key = (name, strat)
            if key not in MAGIC_MAP:
                continue

            magic = magic_for(name, strat)
            uniq = (sym, magic)
            if uniq in seen:
                continue
            seen.add(uniq)

            try:
                close_position_market(sym, magic, cfg, "RISK_FLATTEN")
            except Exception as e:
                print(f"[RISK FLATTEN ERROR] symbol={sym} magic={magic} err={e}")

# ==========================
# STRATEGY EXECUTION
# ==========================

def run_tf_eq(cfg, state, allow_entries):
    snap = account_snapshot()
    equity = float(snap["equity"])
    capacity = remaining_capacity(equity)

    bc_map = state.setdefault("tf_eq_bc", {})

    for m in TF_EQ_MARKETS:
        name = m["name"]
        sym = m["symbol"]

        magic = magic_for(name, "TF_EQ")
        pos = get_position(sym, magic)

        df = fetch_ohlc(sym, "H1", H1_BARS)
        if df.empty:
            continue

        df = compute_tf_eq_indicators(df)
        prev = last_closed_bar(df)
        prev2 = prev_closed_bar(df)

        bc = bc_map.get(name, 0)

        if pos is not None:
            if tf_eq_exit(prev):
                bc += 1
            else:
                bc = 0

            bc_map[name] = bc

            if bc >= TF_EQ_PARAMS["exit_confirm_bars"]:
                close_position_market(sym, magic, cfg, "TF_EQ_EXIT")
                bc_map[name] = 0

            continue

        if not allow_entries or capacity <= 0:
            continue

        if tf_eq_entry(prev, prev2):
            regime = get_cached_market_regime(state, name, sym)
            base_notional, adj_notional = overlay_info("TF_EQ", name, regime, equity)
            notional = min(adj_notional, capacity)

            print(f"[{now_str()}] TF_EQ {name} regime={regime} base_notional={base_notional:.2f} adj_notional={adj_notional:.2f}")

            ok, vol = open_long_by_notional(sym, notional, magic, cfg, "TF_EQ_ENTRY")
            if ok:
                capacity -= notional


def run_mr_eq(cfg, state, allow_entries):
    snap = account_snapshot()
    equity = float(snap["equity"])
    capacity = remaining_capacity(equity)

    mr_eq_processed = state.setdefault("mr_eq_processed_d1", {})

    for m in MR_EQ_MARKETS:
        name = m["name"]
        sym = m["symbol"]

        magic = magic_for(name, "MR_EQ")
        pos = get_position(sym, magic)

        df = fetch_ohlc(sym, "D1", D1_BARS)
        if df.empty:
            continue

        df = compute_mr_eq_indicators(df)
        prev = last_closed_bar(df)
        bar_id = str(pd.Timestamp(prev.name).date())

        if mr_eq_processed.get(name) == bar_id:
            continue

        if pos is not None:
            if mr_eq_exit(prev):
                close_position_market(sym, magic, cfg, "MR_EQ_EXIT")
            mr_eq_processed[name] = bar_id
            continue

        if not allow_entries or capacity <= 0:
            mr_eq_processed[name] = bar_id
            continue

        if mr_eq_entry(prev):
            regime = get_cached_market_regime(state, name, sym)
            base_notional, adj_notional = overlay_info("MR_EQ", name, regime, equity)
            notional = min(adj_notional, capacity)

            print(f"[{now_str()}] MR_EQ {name} regime={regime} base_notional={base_notional:.2f} adj_notional={adj_notional:.2f}")

            ok, vol = open_long_by_notional(sym, notional, magic, cfg, "MR_EQ_ENTRY")
            if ok:
                capacity -= notional

        mr_eq_processed[name] = bar_id


def run_tf_fx(cfg, state, allow_entries):
    snap = account_snapshot()
    equity = float(snap["equity"])
    capacity = remaining_capacity(equity)

    processed = state.setdefault("tf_fx_processed_h1", {})

    for m in TF_FX_MARKETS:
        name = m["name"]
        sym = m["symbol"]

        magic = magic_for(name, "TF_FX")
        pos = get_position(sym, magic)
        direction = position_direction(pos)

        df = fetch_ohlc(sym, "H1", H1_BARS)
        if df.empty:
            continue

        df = compute_tf_fx_indicators(df)

        prev = last_closed_bar(df)
        bar_id = str(pd.Timestamp(prev.name))

        # DEBUG
        #log_bar_times(name, prev.name)

        if processed.get(name) == bar_id:
            continue

        if pos is not None:
            if tf_fx_exit(prev, direction):
                close_position_market(sym, magic, cfg, f"TF_FX_EXIT_{direction}")
            processed[name] = bar_id
            continue

        if not allow_entries or capacity <= 0:
            processed[name] = bar_id
            continue

        signal = tf_fx_entry(name, prev)
        if signal is None:
            processed[name] = bar_id
            continue

        regime = get_cached_market_regime(state, name, sym)
        base_notional, adj_notional = overlay_info("TF_FX", name, regime, equity)
        notional = min(adj_notional, capacity)

        print(f"[{now_str()}] TF_FX {name} regime={regime} signal={signal} base_notional={base_notional:.2f} adj_notional={adj_notional:.2f}")

        if signal == "LONG":
            ok, vol = open_long_by_notional(sym, notional, magic, cfg, "TF_FX_ENTRY_LONG")
        else:
            ok, vol = open_short_by_notional(sym, notional, magic, cfg, "TF_FX_ENTRY_SHORT")

        if ok:
            capacity -= notional

        processed[name] = bar_id


def run_mr_fx(cfg, state, allow_entries):
    snap = account_snapshot()
    equity = float(snap["equity"])
    capacity = remaining_capacity(equity)

    processed = state.setdefault("mr_fx_processed_h1", {})

    for m in MR_FX_MARKETS:
        name = m["name"]
        sym = m["symbol"]

        magic = magic_for(name, "MR_FX")
        pos = get_position(sym, magic)
        direction = position_direction(pos)

        df = fetch_ohlc(sym, "H1", H1_BARS)
        if df.empty:
            continue

        df = compute_mr_fx_indicators(df)

        prev = last_closed_bar(df)
        prev2 = prev_closed_bar(df)
        bar_id = str(pd.Timestamp(prev.name))

        if processed.get(name) == bar_id:
            continue

        if pos is not None:
            if mr_fx_exit(prev, direction):
                close_position_market(sym, magic, cfg, f"MR_FX_EXIT_{direction}")
            processed[name] = bar_id
            continue

        if not allow_entries or capacity <= 0:
            processed[name] = bar_id
            continue

        signal = mr_fx_entry(prev, prev2)
        if signal is None:
            processed[name] = bar_id
            continue

        regime = get_cached_market_regime(state, name, sym)
        base_notional, adj_notional = overlay_info("MR_FX", name, regime, equity)
        notional = min(adj_notional, capacity)

        print(f"[{now_str()}] MR_FX {name} regime={regime} signal={signal} base_notional={base_notional:.2f} adj_notional={adj_notional:.2f}")

        if signal == "LONG":
            ok, vol = open_long_by_notional(sym, notional, magic, cfg, "MR_FX_ENTRY_LONG")
        else:
            ok, vol = open_short_by_notional(sym, notional, magic, cfg, "MR_FX_ENTRY_SHORT")

        if ok:
            capacity -= notional

        processed[name] = bar_id


# ==========================
# MAIN LOOP
# ==========================

def main():
    ensure_initialized()

    cfg = ExecConfig()
    state = load_state()

    state.setdefault("day_start_equity", None)
    state.setdefault("day_start_date", None)
    state.setdefault("day_start_balance", None)
    state.setdefault("day_peak_balance_reference", None)
    state.setdefault("risk_flattened_today", False)
    state.setdefault("tf_eq_bc", {})
    state.setdefault("mr_eq_processed_d1", {})
    state.setdefault("tf_fx_processed_h1", {})
    state.setdefault("mr_fx_processed_h1", {})
    state.setdefault("market_regime_cache", {})

    print("LIVE BOT STARTED")

    for m in TF_EQ_MARKETS:
        print(f"{m['name']} TF_EQ magic={magic_for(m['name'], 'TF_EQ')}")
    for m in MR_EQ_MARKETS:
        print(f"{m['name']} MR_EQ magic={magic_for(m['name'], 'MR_EQ')}")
    for m in TF_FX_MARKETS:
        print(f"{m['name']} TF_FX magic={magic_for(m['name'], 'TF_FX')}")
    for m in MR_FX_MARKETS:
        print(f"{m['name']} MR_FX magic={magic_for(m['name'], 'MR_FX')}")

    print("\n--- REGIME MARKET MULTIPLIERS ---")
    for k, v in REGIME_MARKET_MULTIPLIERS.items():
        print(f"{k}: {v}")

    last_heartbeat = 0

    while True:
        try:
            snap = account_snapshot()
            equity = float(snap["equity"])

            today = date.today().isoformat()
            if state.get("day_start_date") != today:
                state["day_start_date"] = today
                state["day_start_equity"] = equity
                state["day_start_balance"] = float(snap["balance"])
                state["day_peak_balance_reference"] = float(snap["balance"])
                state["risk_flattened_today"] = False
                state["market_regime_cache"] = {}

                print(
                    f"New day: balance={state['day_start_balance']:.2f} "
                    f"equity={state['day_start_equity']:.2f}"
                )

            allow, must_flatten = risk_gate(state, snap, cfg)

            if must_flatten and not state.get("risk_flattened_today", False):
                print(f"[{now_str()}] RISK FLATTEN TRIGGERED")
                flatten_all_known_positions(cfg)
                state["risk_flattened_today"] = True
                allow = False

            run_tf_eq(cfg, state, allow)
            run_mr_eq(cfg, state, allow)
            run_tf_fx(cfg, state, allow)
            run_mr_fx(cfg, state, allow)

            save_state(state)

            now = time.time()
            if now - last_heartbeat > HEARTBEAT_EVERY_SEC:
                last_heartbeat = now
                dd = day_drawdown(state["day_start_equity"], equity)
                print(f"[{now_str()}] equity={equity:.2f} dd={dd:.2%} allow={allow}")

        except Exception as e:
            print("ERROR:", e)

        time.sleep(SLEEP_SECONDS)


if __name__ == "__main__":
    main()
