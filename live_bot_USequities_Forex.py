import json
import time
from datetime import datetime, date
from datetime import datetime, date, timezone
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Optional, Tuple
import os
import atexit

import numpy as np
import pandas as pd
import ta
import MetaTrader5 as mt5

from mt5_exec import (
    ExecConfig,
    ensure_initialized,
    ensure_symbol,
    account_snapshot,
    get_position,
    close_position_market,
    open_long_by_notional,
    open_short_by_notional,
    position_usd_notional,
    mt5_healthcheck,
    reconnect_mt5,
    append_trade_lifecycle_log,
)


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

EVAL_EXPOSURE_FACTOR = 0.75
FUNDED_EXPOSURE_FACTOR = 0.75

SOFT_CUTOFF_DAILY = -0.045

START_BALANCE = 50_000
DAILY_LOSS_LIMIT_ABS_FRAC = 0.05
MAX_LOSS_LIMIT_ABS_FRAC = 0.10

DYNAMIC_EXPOSURE_MAP = {
    "<-7%": 0.4,
    "-7%_-4%": 0.4,
    "-4%_-2%": 0.8,
    "-2%_0%": 1.0,
    "0%_2.5%": 1.25,
    "2.5%_5%": 1.25,
    ">5%": 1.25,
}

MAX_ENTRY_DRIFT = {
    "TF_EQ": {
        "US500": 8.0,
        "US100": 25.0,
        "US30": 70.0,
    },
    "MR_EQ": {
        "US500": 8.0,
        "US100": 25.0,
        "US30": 70.0,
    },
    "TF_FX": {
        "EURJPY": 0.12,
        "GBPJPY": 0.18,
        "USDJPY": 0.10,
    },
    "MR_FX": {
        "EURCHF": 0.0008,
        "EURCAD": 0.0012,
        "GBPCHF": 0.0012,
        "EURUSD": 0.0008,
        "USDCHF": 0.0008,
    },
}

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
# All session times below are interpreted in broker/server time.
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

MAX_CONSECUTIVE_LOOP_ERRORS = 10
ERROR_COOLDOWN_SECONDS = 60

LOCK_FILE = "live_bot.lock"

MAX_ENTRIES_PER_SYMBOL_PER_DAY = 3
BLOCK_REENTRY_ON_SAME_BAR = True
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
    tmp = f"{STATE_FILE}.tmp"
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp, STATE_FILE)


def now_str():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def acquire_lock_or_die():
    if os.path.exists(LOCK_FILE):
        try:
            with open(LOCK_FILE, "r") as f:
                existing_pid = f.read().strip()
        except Exception:
            existing_pid = "unknown"

        raise RuntimeError(
            f"Lock file already exists: {LOCK_FILE}. "
            f"Another bot instance may be running. Existing PID={existing_pid}"
        )

    with open(LOCK_FILE, "w") as f:
        f.write(str(os.getpid()))

    def _cleanup():
        try:
            if os.path.exists(LOCK_FILE):
                os.remove(LOCK_FILE)
        except Exception:
            pass

    atexit.register(_cleanup)
# ==========================
# MT5 DATA
# ==========================

def mt5_timeframe(tf):
    return mt5.TIMEFRAME_H1 if tf == "H1" else mt5.TIMEFRAME_D1


def fetch_ohlc(symbol, tf, n, min_bars=120):
    ensure_symbol(symbol)

    rates = mt5.copy_rates_from_pos(symbol, mt5_timeframe(tf), 0, n)
    if rates is None:
        print(f"[{now_str()}] FETCH OHLC NONE symbol={symbol} tf={tf} n={n}")
        return pd.DataFrame()

    if len(rates) < min_bars:
        print(
            f"[{now_str()}] FETCH OHLC TOO SHORT "
            f"symbol={symbol} tf={tf} got={len(rates)} min_bars={min_bars}"
        )
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
def broker_now() -> datetime:
    """
    Approx broker/server time using latest available tick timestamp.
    Falls back to local time if unavailable.
    """
    for symbol in ["EURUSD", "USDJPY", "US500.cash", "EURCHF"]:
        try:
            tick = mt5.symbol_info_tick(symbol)
            if tick is not None and getattr(tick, "time", None):
                return datetime.fromtimestamp(int(tick.time))
        except Exception:
            pass

    return datetime.now()

def broker_today_str() -> str:
    return broker_now().date().isoformat()

def now_str():
    return broker_now().strftime("%Y-%m-%d %H:%M:%S")

def latest_closed_bar_time(symbol: str, tf: str) -> Optional[pd.Timestamp]:
    df = fetch_ohlc(symbol, tf, 5, min_bars=3)
    if df.empty or len(df) < 3:
        return None
    return pd.Timestamp(df.index[-2])


def compute_global_bar_clock(symbols, tf: str) -> Optional[str]:
    """
    Uses the first symbol with valid data as a clock source.
    Assumes symbols in same asset group have aligned bar times.
    """
    for sym in symbols:
        try:
            ts = latest_closed_bar_time(sym, tf)
            if ts is not None:
                return str(ts)
        except Exception as e:
            print(f"[{now_str()}] BAR CLOCK ERROR symbol={sym} tf={tf} err={e}")
    return None

def entry_circuit_break_active(state) -> bool:
    until_ts = float(state.get("entry_circuit_break_until", 0.0) or 0.0)
    return time.time() < until_ts

def get_bot_positions_grouped():
    positions = mt5.positions_get()
    grouped = {}

    if positions is None:
        return grouped

    allowed_magics = set(MAGIC_MAP.values())

    for p in positions:
        magic = int(getattr(p, "magic", 0))
        if magic not in allowed_magics:
            continue

        key = (str(p.symbol), magic)
        grouped.setdefault(key, []).append(p)

    return grouped

def reconcile_on_startup(state):
    """
    Basic startup reconciliation between persisted state and broker reality.
    Logs anomalies but does not mutate positions.
    """
    print(f"[{now_str()}] STARTUP RECONCILIATION BEGIN")

    grouped = get_bot_positions_grouped()

    for (symbol, magic), plist in grouped.items():
        if len(plist) > 1:
            print(
                f"[{now_str()}] RECON WARNING multiple positions "
                f"symbol={symbol} magic={magic} count={len(plist)}"
            )
            for p in plist:
                print(
                    f"[{now_str()}] RECON POSITION "
                    f"symbol={p.symbol} magic={p.magic} ticket={p.ticket} "
                    f"type={p.type} volume={p.volume} price_open={p.price_open}"
                )

    if not grouped:
        print(f"[{now_str()}] RECON no bot-managed open positions found")
    else:
        total = sum(len(v) for v in grouped.values())
        print(f"[{now_str()}] RECON found bot-managed open positions={total}")

    today = date.today().isoformat()
    if state.get("day_start_date") not in (None, today):
        print(
            f"[{now_str()}] RECON resetting stale day state "
            f"stored_day={state.get('day_start_date')} today={today}"
        )
        state["day_start_date"] = today
        state["day_start_equity"] = None
        state["day_start_balance"] = None
        state["day_peak_balance_reference"] = None
        state["risk_flattened_today"] = False
        state["market_regime_cache"] = {}

    print(f"[{now_str()}] STARTUP RECONCILIATION END")


def trade_key(symbol: str, magic: int) -> str:
    return f"{symbol}__{magic}"


def register_open_trade(state, strategy: str, market: str, symbol: str, magic: int, direction: str, volume: float, entry_price: float):
    reg = state.setdefault("open_trade_registry", {})
    reg[trade_key(symbol, magic)] = {
        "strategy": strategy,
        "market": market,
        "symbol": symbol,
        "magic": int(magic),
        "direction": direction,
        "entry_time": now_str(),
        "entry_price": float(entry_price),
        "volume": float(volume),
    }


def unregister_open_trade(state, symbol: str, magic: int):
    reg = state.setdefault("open_trade_registry", {})
    return reg.pop(trade_key(symbol, magic), None)


def current_position_mid_price(symbol: str) -> Optional[float]:
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        return None

    bid = float(tick.bid)
    ask = float(tick.ask)
    if bid > 0 and ask > 0:
        return (bid + ask) / 2.0
    return None

def finalize_trade_log(cfg, state, symbol: str, magic: int, exit_reason: str):
    rec = unregister_open_trade(state, symbol, magic)
    if rec is None:
        return

    exit_price = current_position_mid_price(symbol)
    if exit_price is None:
        exit_price = 0.0

    entry_price = float(rec["entry_price"])
    volume = float(rec["volume"])
    direction = rec["direction"]

    if direction == "LONG":
        pnl = (exit_price - entry_price) * volume if exit_price > 0 else 0.0
    else:
        pnl = (entry_price - exit_price) * volume if exit_price > 0 else 0.0

    try:
        t0 = datetime.strptime(rec["entry_time"], "%Y-%m-%d %H:%M:%S")
        t1 = broker_now()
        holding_seconds = int((t1 - t0).total_seconds())
    except Exception:
        holding_seconds = ""

    append_trade_lifecycle_log(cfg, {
        "ts": now_str(),
        "strategy": rec["strategy"],
        "market": rec["market"],
        "symbol": rec["symbol"],
        "magic": rec["magic"],
        "direction": direction,
        "entry_time": rec["entry_time"],
        "exit_time": now_str(),
        "holding_seconds": holding_seconds,
        "entry_price": entry_price,
        "exit_price": exit_price,
        "volume": volume,
        "pnl": pnl,
        "exit_reason": exit_reason,
    })

def entries_today_key(symbol: str, magic: int, day_str: str) -> str:
    return f"{day_str}__{symbol}__{magic}"


def increment_entries_today(state, symbol: str, magic: int):
    day_str = broker_today_str()
    key = entries_today_key(symbol, magic, day_str)
    d = state.setdefault("entries_today", {})
    d[key] = int(d.get(key, 0)) + 1


def entries_today_count(state, symbol: str, magic: int) -> int:
    day_str = broker_today_str()
    key = entries_today_key(symbol, magic, day_str)
    return int(state.setdefault("entries_today", {}).get(key, 0))


def mark_exit_bar(state, symbol: str, magic: int, bar_id: str):
    d = state.setdefault("last_exit_bar", {})
    d[trade_key(symbol, magic)] = bar_id


def exited_this_bar(state, symbol: str, magic: int, bar_id: str) -> bool:
    d = state.setdefault("last_exit_bar", {})
    return d.get(trade_key(symbol, magic)) == bar_id


def entry_allowed_by_daily_limits(state, symbol: str, magic: int) -> bool:
    count = entries_today_count(state, symbol, magic)
    if count >= MAX_ENTRIES_PER_SYMBOL_PER_DAY:
        print(
            f"[{now_str()}] ENTRY BLOCKED daily max entries reached "
            f"symbol={symbol} magic={magic} count={count}"
        )
        return False
    return True

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
    start_t = pd.to_datetime(session_start).time()
    end_t = pd.to_datetime(session_end).time()
    t = ts.time()

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

def current_entry_side_price(symbol: str, signal: str) -> Optional[float]:
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        return None

    if signal == "LONG":
        return float(tick.ask)
    return float(tick.bid)


def entry_drift_ok(
    strategy: str,
    market: str,
    symbol: str,
    signal: str,
    signal_price: float,
) -> Tuple[bool, Optional[float], Optional[float]]:
    """
    Returnerar:
      ok, current_price, max_drift

    signal_price = prisnivån när signalen observerades
    current_price = faktisk live entry-side price just nu
    """
    current_price = current_entry_side_price(symbol, signal)
    if current_price is None:
        return False, None, None

    max_drift = MAX_ENTRY_DRIFT.get(strategy, {}).get(market)
    if max_drift is None:
        return True, current_price, None

    ok = abs(float(current_price) - float(signal_price)) <= float(max_drift)
    return ok, float(current_price), float(max_drift)

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
    Assumes df index is already in broker/server time.
    Asia session defined as 00:00-07:00 broker time.
    """
    data = df_in.copy()

    session = data.between_time("00:00", "07:00")
    daily_range = session.groupby(session.index.date).agg(
        asia_high=("high", "max"),
        asia_low=("low", "min")
    )

    data["date"] = data.index.date
    data = data.merge(
        daily_range,
        left_on="date",
        right_index=True,
        how="left"
    )

    data["asia_range"] = data["asia_high"] - data["asia_low"]
    data["asia_mid"] = (data["asia_high"] + data["asia_low"]) / 2

    data.loc[data.index.hour < 8, ["asia_high", "asia_low", "asia_range", "asia_mid"]] = np.nan
    data.drop(columns="date", inplace=True)

    return data


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

def ensure_log_dirs(cfg):
    import os
    for path in [
        cfg.log_csv_path,
        cfg.event_log_csv_path,
        cfg.lifecycle_log_csv_path,
    ]:
        d = os.path.dirname(path)
        if d and not os.path.exists(d):
            os.makedirs(d, exist_ok=True)
            
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
def dd_bucket_from_balance(current_equity: float, reference_balance: float) -> str:
    dd = current_equity / reference_balance - 1.0

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


def dynamic_exposure_multiplier(
    current_equity: float,
    reference_balance: float = START_BALANCE,
) -> float:
    bucket = dd_bucket_from_balance(current_equity, reference_balance)
    return float(DYNAMIC_EXPOSURE_MAP[bucket])


def strategy_targets(current_equity: float):
    """
    Bas-targets * global dynamic exposure factor.
    Regime-overlay appliceras separat per market/strategy senare.
    """
    base_factor = EVAL_EXPOSURE_FACTOR if MODE == "EVAL" else FUNDED_EXPOSURE_FACTOR
    dd_factor = dynamic_exposure_multiplier(current_equity)

    total_factor = base_factor * dd_factor

    return {k: float(v) * total_factor for k, v in STRATEGY_TARGETS_BASE.items()}


def open_positions_gross_notional():
    gross = 0.0
    positions = mt5.positions_get()

    if positions is None:
        return 0.0

    for p in positions:
        try:
            gross += float(position_usd_notional(p))
        except Exception as e:
            print(f"[GROSS NOTIONAL ERROR] symbol={getattr(p, 'symbol', '?')} err={e}")

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
    """
    Tar already dynamically scaled base_target och lägger på market-specific regime overlay.
    """
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
    """
    Final desired notional:
      equity
    * dynamic exposure adjusted strategy target
    * market weight
    * optional regime overlay
    """
    targets = strategy_targets(equity)
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
    """
    Debug helper:
      base_notional = dynamic exposure only
      adj_notional  = dynamic exposure + regime overlay
    """
    targets = strategy_targets(equity)
    base_target = float(targets[strategy])
    w = float(STRATEGY_MARKET_WEIGHTS[strategy][market])

    base_notional = equity * base_target * w

    adj_target = strategy_target_with_regime_overlay(
        strategy=strategy,
        market=market,
        base_target=base_target,
        market_regime_label=regime,
    )
    adj_notional = equity * adj_target * w

    return float(base_notional), float(adj_notional)

def sizing_debug_info(strategy: str, market: str, regime: Optional[str], equity: float) -> dict:
    mode_factor = EVAL_EXPOSURE_FACTOR if MODE == "EVAL" else FUNDED_EXPOSURE_FACTOR
    dd_bucket = dd_bucket_from_balance(equity, START_BALANCE)
    dd_factor = dynamic_exposure_multiplier(equity, START_BALANCE)

    regime_mult = 1.0
    if USE_REGIME_OVERLAY and regime is not None:
        regime_mult = float(
            REGIME_MARKET_MULTIPLIERS.get((str(regime), str(strategy), str(market)), 1.0)
        )

    base_target_raw = float(STRATEGY_TARGETS_BASE[strategy])
    market_weight = float(STRATEGY_MARKET_WEIGHTS[strategy][market])

    final_factor = mode_factor * dd_factor * regime_mult

    base_notional = equity * base_target_raw * mode_factor * dd_factor * market_weight
    adj_notional = equity * base_target_raw * mode_factor * dd_factor * regime_mult * market_weight

    return {
        "mode_factor": float(mode_factor),
        "dd_bucket": dd_bucket,
        "dd_factor": float(dd_factor),
        "regime_mult": float(regime_mult),
        "final_factor": float(final_factor),
        "base_target_raw": float(base_target_raw),
        "market_weight": float(market_weight),
        "base_notional": float(base_notional),
        "adj_notional": float(adj_notional),
    }
# ==========================
# RISK
# ==========================
def get_all_bot_positions():
    positions = mt5.positions_get()
    if positions is None:
        return []

    allowed_magics = set(MAGIC_MAP.values())
    out = []

    for p in positions:
        if int(getattr(p, "magic", 0)) in allowed_magics:
            out.append(p)

    return out


def has_open_bot_positions() -> bool:
    return len(get_all_bot_positions()) > 0

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

    # Internal daily kill threshold: -4.5% from intraday highest balance reference
    soft_floor = ref_balance * (1.0 + SOFT_CUTOFF_DAILY)  # SOFT_CUTOFF_DAILY = -0.045

    # Absolute max loss floor (still keep this if you want account-wide protection)
    max_floor = START_BALANCE * (1.0 - MAX_LOSS_LIMIT_ABS_FRAC)

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

def flatten_all_known_positions(cfg) -> bool:
    """
    Attempts to close all bot-managed positions.
    Returns True if no bot-managed positions remain afterwards.
    """
    positions = get_all_bot_positions()

    if not positions:
        print(f"[{now_str()}] RISK FLATTEN no bot positions found")
        return True

    all_close_attempts_ok = True

    for p in positions:
        sym = str(p.symbol)
        magic = int(p.magic)
        ticket = int(p.ticket)

        try:
            ok = close_position_market(sym, magic, cfg, "RISK_FLATTEN")
            if not ok:
                all_close_attempts_ok = False
                print(
                    f"[{now_str()}] RISK FLATTEN close failed "
                    f"symbol={sym} magic={magic} ticket={ticket}"
                )
        except Exception as e:
            all_close_attempts_ok = False
            print(
                f"[{now_str()}] RISK FLATTEN exception "
                f"symbol={sym} magic={magic} ticket={ticket} err={e}"
            )

    remaining = get_all_bot_positions()
    if remaining:
        print(f"[{now_str()}] RISK FLATTEN incomplete remaining={len(remaining)}")
        for p in remaining:
            print(
                f"[{now_str()}] STILL OPEN symbol={p.symbol} "
                f"magic={p.magic} ticket={p.ticket} volume={p.volume}"
            )
        return False

    print(f"[{now_str()}] RISK FLATTEN complete")
    return all_close_attempts_ok

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
        bar_id = str(pd.Timestamp(prev.name))
        print(f"prev bar timestamp = {prev.name}")
        def sweden_now():
            return datetime.now(ZoneInfo("Europe/Stockholm"))
        print(
            f"SWEDEN_NOW={sweden_now()} "
            f"BOT_NOW={now_str()} "
            f"PREV_BAR={prev.name}"
        )
        bc = int(bc_map.get(name, 0))

        if pos is not None:
            if tf_eq_exit(prev):
                bc += 1
            else:
                bc = 0

            bc_map[name] = bc

            if bc >= TF_EQ_PARAMS["exit_confirm_bars"]:
                ok = close_position_market(sym, magic, cfg, "TF_EQ_EXIT")
                if ok:
                    finalize_trade_log(cfg, state, sym, magic, "TF_EQ_EXIT")
                    mark_exit_bar(state, sym, magic, bar_id)
                    bc_map[name] = 0

            continue

        bc_map[name] = 0

        if not allow_entries or capacity <= 0:
            continue

        if not entry_allowed_by_daily_limits(state, sym, magic):
            continue

        if BLOCK_REENTRY_ON_SAME_BAR and exited_this_bar(state, sym, magic, bar_id):
            print(
                f"[{now_str()}] TF_EQ {name} entry blocked same-bar reentry "
                f"symbol={sym} magic={magic} bar_id={bar_id}"
            )
            continue

        if tf_eq_entry(prev, prev2):
            signal_price = float(prev["close"])
            drift_ok, current_px, max_drift = entry_drift_ok(
                strategy="TF_EQ",
                market=name,
                symbol=sym,
                signal="LONG",
                signal_price=signal_price,
            )

            if not drift_ok:
                print(
                    f"[{now_str()}] TF_EQ {name} entry blocked by drift guard "
                    f"signal_price={signal_price:.5f} current_price={current_px} max_drift={max_drift}"
                )
                continue

            regime = get_cached_market_regime(state, name, sym)
            base_notional, adj_notional = overlay_info("TF_EQ", name, regime, equity)
            notional = min(adj_notional, capacity)

            if notional <= 0:
                continue

            dbg = sizing_debug_info("TF_EQ", name, regime, equity)
            print(
                f"[{now_str()}] TF_EQ {name} regime={regime} "
                f"mode_factor={dbg['mode_factor']:.2f} "
                f"dd_bucket={dbg['dd_bucket']} dd_factor={dbg['dd_factor']:.2f} "
                f"regime_mult={dbg['regime_mult']:.2f} final_factor={dbg['final_factor']:.4f} "
                f"base_notional={dbg['base_notional']:.2f} adj_notional={dbg['adj_notional']:.2f}"
            )

            ok, vol = open_long_by_notional(sym, notional, magic, cfg, "TF_EQ_ENTRY")
            if ok:
                entry_px = current_entry_side_price(sym, "LONG") or signal_price
                register_open_trade(state, "TF_EQ", name, sym, magic, "LONG", vol, entry_px)
                increment_entries_today(state, sym, magic)
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
                ok = close_position_market(sym, magic, cfg, "MR_EQ_EXIT")
                if ok:
                    finalize_trade_log(cfg, state, sym, magic, "MR_EQ_EXIT")
                    mark_exit_bar(state, sym, magic, bar_id)

            mr_eq_processed[name] = bar_id
            continue

        if not allow_entries or capacity <= 0:
            mr_eq_processed[name] = bar_id
            continue

        if not entry_allowed_by_daily_limits(state, sym, magic):
            mr_eq_processed[name] = bar_id
            continue

        if BLOCK_REENTRY_ON_SAME_BAR and exited_this_bar(state, sym, magic, bar_id):
            print(
                f"[{now_str()}] MR_EQ {name} entry blocked same-bar reentry "
                f"symbol={sym} magic={magic} bar_id={bar_id}"
            )
            mr_eq_processed[name] = bar_id
            continue

        if mr_eq_entry(prev):
            signal_price = float(prev["close"])
            drift_ok, current_px, max_drift = entry_drift_ok(
                strategy="MR_EQ",
                market=name,
                symbol=sym,
                signal="LONG",
                signal_price=signal_price,
            )

            if not drift_ok:
                print(
                    f"[{now_str()}] MR_EQ {name} entry blocked by drift guard "
                    f"signal_price={signal_price:.5f} current_price={current_px} max_drift={max_drift}"
                )
                mr_eq_processed[name] = bar_id
                continue

            regime = get_cached_market_regime(state, name, sym)
            base_notional, adj_notional = overlay_info("MR_EQ", name, regime, equity)
            notional = min(adj_notional, capacity)

            if notional <= 0:
                mr_eq_processed[name] = bar_id
                continue

            dbg = sizing_debug_info("MR_EQ", name, regime, equity)
            print(
                f"[{now_str()}] MR_EQ {name} regime={regime} "
                f"mode_factor={dbg['mode_factor']:.2f} "
                f"dd_bucket={dbg['dd_bucket']} dd_factor={dbg['dd_factor']:.2f} "
                f"regime_mult={dbg['regime_mult']:.2f} final_factor={dbg['final_factor']:.4f} "
                f"base_notional={dbg['base_notional']:.2f} adj_notional={dbg['adj_notional']:.2f}"
            )

            ok, vol = open_long_by_notional(sym, notional, magic, cfg, "MR_EQ_ENTRY")
            if ok:
                entry_px = current_entry_side_price(sym, "LONG") or signal_price
                register_open_trade(state, "MR_EQ", name, sym, magic, "LONG", vol, entry_px)
                increment_entries_today(state, sym, magic)
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

        if processed.get(name) == bar_id:
            continue

        if pos is not None:
            if tf_fx_exit(prev, direction):
                exit_reason = f"TF_FX_EXIT_{direction}"
                ok = close_position_market(sym, magic, cfg, exit_reason)
                if ok:
                    finalize_trade_log(cfg, state, sym, magic, exit_reason)
                    mark_exit_bar(state, sym, magic, bar_id)

            processed[name] = bar_id
            continue

        if not allow_entries or capacity <= 0:
            processed[name] = bar_id
            continue

        if not entry_allowed_by_daily_limits(state, sym, magic):
            processed[name] = bar_id
            continue

        if BLOCK_REENTRY_ON_SAME_BAR and exited_this_bar(state, sym, magic, bar_id):
            print(
                f"[{now_str()}] TF_FX {name} entry blocked same-bar reentry "
                f"symbol={sym} magic={magic} bar_id={bar_id}"
            )
            processed[name] = bar_id
            continue

        signal = tf_fx_entry(name, prev)
        if signal is None:
            processed[name] = bar_id
            continue

        signal_price = float(prev["close"])
        drift_ok, current_px, max_drift = entry_drift_ok(
            strategy="TF_FX",
            market=name,
            symbol=sym,
            signal=signal,
            signal_price=signal_price,
        )

        if not drift_ok:
            print(
                f"[{now_str()}] TF_FX {name} entry blocked by drift guard "
                f"signal={signal} signal_price={signal_price:.5f} current_price={current_px} max_drift={max_drift}"
            )
            processed[name] = bar_id
            continue

        regime = get_cached_market_regime(state, name, sym)
        base_notional, adj_notional = overlay_info("TF_FX", name, regime, equity)
        notional = min(adj_notional, capacity)

        if notional <= 0:
            processed[name] = bar_id
            continue

        dbg = sizing_debug_info("TF_FX", name, regime, equity)
        print(
            f"[{now_str()}] TF_FX {name} regime={regime} signal={signal} "
            f"mode_factor={dbg['mode_factor']:.2f} "
            f"dd_bucket={dbg['dd_bucket']} dd_factor={dbg['dd_factor']:.2f} "
            f"regime_mult={dbg['regime_mult']:.2f} final_factor={dbg['final_factor']:.4f} "
            f"base_notional={dbg['base_notional']:.2f} adj_notional={dbg['adj_notional']:.2f}"
        )

        if signal == "LONG":
            ok, vol = open_long_by_notional(sym, notional, magic, cfg, "TF_FX_ENTRY_LONG")
            if ok:
                entry_px = current_entry_side_price(sym, "LONG") or signal_price
                register_open_trade(state, "TF_FX", name, sym, magic, "LONG", vol, entry_px)
                increment_entries_today(state, sym, magic)
                capacity -= notional
        else:
            ok, vol = open_short_by_notional(sym, notional, magic, cfg, "TF_FX_ENTRY_SHORT")
            if ok:
                entry_px = current_entry_side_price(sym, "SHORT") or signal_price
                register_open_trade(state, "TF_FX", name, sym, magic, "SHORT", vol, entry_px)
                increment_entries_today(state, sym, magic)
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
                exit_reason = f"MR_FX_EXIT_{direction}"
                ok = close_position_market(sym, magic, cfg, exit_reason)
                if ok:
                    finalize_trade_log(cfg, state, sym, magic, exit_reason)
                    mark_exit_bar(state, sym, magic, bar_id)

            processed[name] = bar_id
            continue

        if not allow_entries or capacity <= 0:
            processed[name] = bar_id
            continue

        if not entry_allowed_by_daily_limits(state, sym, magic):
            processed[name] = bar_id
            continue

        if BLOCK_REENTRY_ON_SAME_BAR and exited_this_bar(state, sym, magic, bar_id):
            print(
                f"[{now_str()}] MR_FX {name} entry blocked same-bar reentry "
                f"symbol={sym} magic={magic} bar_id={bar_id}"
            )
            processed[name] = bar_id
            continue

        signal = mr_fx_entry(prev, prev2)
        if signal is None:
            processed[name] = bar_id
            continue

        signal_price = float(prev["close"])
        drift_ok, current_px, max_drift = entry_drift_ok(
            strategy="MR_FX",
            market=name,
            symbol=sym,
            signal=signal,
            signal_price=signal_price,
        )

        if not drift_ok:
            print(
                f"[{now_str()}] MR_FX {name} entry blocked by drift guard "
                f"signal={signal} signal_price={signal_price:.5f} current_price={current_px} max_drift={max_drift}"
            )
            processed[name] = bar_id
            continue

        regime = get_cached_market_regime(state, name, sym)
        base_notional, adj_notional = overlay_info("MR_FX", name, regime, equity)
        notional = min(adj_notional, capacity)

        if notional <= 0:
            processed[name] = bar_id
            continue

        dbg = sizing_debug_info("MR_FX", name, regime, equity)
        print(
            f"[{now_str()}] MR_FX {name} regime={regime} signal={signal} "
            f"mode_factor={dbg['mode_factor']:.2f} "
            f"dd_bucket={dbg['dd_bucket']} dd_factor={dbg['dd_factor']:.2f} "
            f"regime_mult={dbg['regime_mult']:.2f} final_factor={dbg['final_factor']:.4f} "
            f"base_notional={dbg['base_notional']:.2f} adj_notional={dbg['adj_notional']:.2f}"
        )

        if signal == "LONG":
            ok, vol = open_long_by_notional(sym, notional, magic, cfg, "MR_FX_ENTRY_LONG")
            if ok:
                entry_px = current_entry_side_price(sym, "LONG") or signal_price
                register_open_trade(state, "MR_FX", name, sym, magic, "LONG", vol, entry_px)
                increment_entries_today(state, sym, magic)
                capacity -= notional
        else:
            ok, vol = open_short_by_notional(sym, notional, magic, cfg, "MR_FX_ENTRY_SHORT")
            if ok:
                entry_px = current_entry_side_price(sym, "SHORT") or signal_price
                register_open_trade(state, "MR_FX", name, sym, magic, "SHORT", vol, entry_px)
                increment_entries_today(state, sym, magic)
                capacity -= notional

        processed[name] = bar_id


# ==========================
# MAIN LOOP
# ==========================

def main():

    acquire_lock_or_die()
    ensure_initialized()

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    cfg = ExecConfig(
    log_csv_path=os.path.join(BASE_DIR, "logs/trade_log.csv"),
    event_log_csv_path=os.path.join(BASE_DIR, "logs/execution_events.csv"),
    lifecycle_log_csv_path=os.path.join(BASE_DIR, "logs/trade_lifecycle.csv"),
    )

    ensure_log_dirs(cfg)   # ← viktigt att du har denna också
    
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
    state.setdefault("mt5_connection_failures", 0)
    state.setdefault("mt5_last_health_status", "unknown")
    state.setdefault("consecutive_loop_errors", 0)
    state.setdefault("entry_circuit_break_until", 0.0)
    state.setdefault("last_h1_bar_clock", None)
    state.setdefault("last_d1_bar_clock", None)
    state.setdefault("open_trade_registry", {})
    state.setdefault("entries_today", {})
    state.setdefault("last_exit_bar", {})

    reconcile_on_startup(state)

    print(f"[{now_str()}] LIVE BOT STARTED pid={os.getpid()}")

    h1_symbols = [m["symbol"] for m in TF_EQ_MARKETS + TF_FX_MARKETS + MR_FX_MARKETS]
    d1_symbols = [m["symbol"] for m in MR_EQ_MARKETS]

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
            healthy, health_msg = mt5_healthcheck()
            state["mt5_last_health_status"] = health_msg

            if not healthy:
                state["mt5_connection_failures"] = int(state.get("mt5_connection_failures", 0)) + 1
                print(
                    f"[{now_str()}] MT5 HEALTHCHECK FAILED "
                    f"status={health_msg} failures={state['mt5_connection_failures']}"
                )

                re_ok = reconnect_mt5()
                if re_ok:
                    print(f"[{now_str()}] MT5 RECONNECT SUCCESS")
                    state["mt5_connection_failures"] = 0
                    state["mt5_last_health_status"] = "reconnected"
                else:
                    print(f"[{now_str()}] MT5 RECONNECT FAILED")
                    save_state(state)
                    time.sleep(SLEEP_SECONDS)
                    continue
            else:
                state["mt5_connection_failures"] = 0

            snap = account_snapshot()
            equity = float(snap["equity"])

            today = broker_today_str()
            if state.get("day_start_date") != today:
                state["day_start_date"] = today
                state["day_start_equity"] = equity
                state["day_start_balance"] = float(snap["balance"])
                state["day_peak_balance_reference"] = float(snap["balance"])
                state["risk_flattened_today"] = False
                state["market_regime_cache"] = {}
                state["entries_today"] = {}
                state["last_exit_bar"] = {}

                print(
                    f"New day: balance={state['day_start_balance']:.2f} "
                    f"equity={state['day_start_equity']:.2f}"
                )

            allow, must_flatten = risk_gate(state, snap, cfg)

            if entry_circuit_break_active(state):
                print(
                    f"[{now_str()}] ENTRY CIRCUIT BREAK ACTIVE "
                    f"until={state.get('entry_circuit_break_until')}"
                )
                allow = False

            if must_flatten:
                print(f"[{now_str()}] RISK FLATTEN TRIGGERED")
                flatten_ok = flatten_all_known_positions(cfg)

                if flatten_ok and not has_open_bot_positions():
                    state["risk_flattened_today"] = True
                    print(f"[{now_str()}] RISK STATE locked for rest of day")
                else:
                    state["risk_flattened_today"] = False
                    print(f"[{now_str()}] RISK FLATTEN not fully complete, will retry next loop")

                allow = False

            h1_bar_clock = compute_global_bar_clock(h1_symbols, "H1")
            d1_bar_clock = compute_global_bar_clock(d1_symbols, "D1")

            new_h1_bar = h1_bar_clock is not None and h1_bar_clock != state.get("last_h1_bar_clock")
            new_d1_bar = d1_bar_clock is not None and d1_bar_clock != state.get("last_d1_bar_clock")

            if new_h1_bar:
                print(f"[{now_str()}] NEW H1 BAR DETECTED bar={h1_bar_clock}")
                run_tf_eq(cfg, state, allow)
                run_tf_fx(cfg, state, allow)
                run_mr_fx(cfg, state, allow)
                state["last_h1_bar_clock"] = h1_bar_clock

            if new_d1_bar:
                print(f"[{now_str()}] NEW D1 BAR DETECTED bar={d1_bar_clock}")
                run_mr_eq(cfg, state, allow)
                state["last_d1_bar_clock"] = d1_bar_clock

            save_state(state)

            now = time.time()
            if now - last_heartbeat > HEARTBEAT_EVERY_SEC:
                last_heartbeat = now
                dd = day_drawdown(state["day_start_equity"], equity)
                dd_bucket = dd_bucket_from_balance(equity, START_BALANCE)
                dd_mult = dynamic_exposure_multiplier(equity, START_BALANCE)

                print(
                    f"[{now_str()}] equity={equity:.2f} dd={dd:.2%} allow={allow} "
                    f"dd_bucket={dd_bucket} dd_mult={dd_mult:.2f}"
                )


        except Exception as e:

            state["consecutive_loop_errors"] = int(state.get("consecutive_loop_errors", 0)) + 1
            print(
                f"[{now_str()}] LOOP ERROR err={e} "
                f"consecutive={state['consecutive_loop_errors']}"
            )

            if state["consecutive_loop_errors"] >= MAX_CONSECUTIVE_LOOP_ERRORS:
                state["entry_circuit_break_until"] = time.time() + ERROR_COOLDOWN_SECONDS
                print(
                    f"[{now_str()}] ENTRY CIRCUIT BREAK TRIGGERED "
                    f"for {ERROR_COOLDOWN_SECONDS}s"
                )

            try:
                save_state(state)
            except Exception as save_err:
                print(f"[{now_str()}] STATE SAVE ERROR after exception err={save_err}")

        state["consecutive_loop_errors"] = 0

        time.sleep(SLEEP_SECONDS)


if __name__ == "__main__":
    main()
