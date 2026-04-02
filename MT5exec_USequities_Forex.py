import time
import csv
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Tuple, Dict

import MetaTrader5 as mt5


# ==========================
# CONFIG
# ==========================

@dataclass
class ExecConfig:
    deviation_points: int = 50
    order_filling: int = mt5.ORDER_FILLING_IOC
    retries: int = 5
    retry_sleep_sec: float = 0.5
    min_margin_level: float = 150.0
    log_csv_path: str = "trade_log.csv"
    event_log_csv_path: str = "execution_events.csv"
    lifecycle_log_csv_path: str = "trade_lifecycle.csv"
    max_tick_age_sec_fx: int = 10
    max_tick_age_sec_index: int = 20


@dataclass
class SymbolMeta:
    name: str
    contract_size: float
    tick_size: float
    tick_value: float
    vol_min: float
    vol_max: float
    vol_step: float
    currency_base: str
    currency_profit: str
    currency_margin: str

MAX_SPREAD_BY_SYMBOL = {
    # Indices
    "US500.cash": 1.5,
    "US100.cash": 3.0,
    "US30.cash": 8.0,

    # TF_FX
    "EURJPY": 0.03,
    "GBPJPY": 0.05,
    "USDJPY": 0.03,

    # MR_FX
    "EURCHF": 0.00025,
    "EURCAD": 0.00035,
    "GBPCHF": 0.00035,
    "EURUSD": 0.00020,
    "USDCHF": 0.00020,
}
# ==========================
# UTILS
# ==========================

def _ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def ensure_initialized():
    if not mt5.initialize():
        raise RuntimeError("MT5 initialize() failed")

    acc = mt5.account_info()
    if acc is None:
        raise RuntimeError("MT5 account_info() failed")

def mt5_healthcheck() -> Tuple[bool, str]:
    """
    Basic MT5 health check:
    - terminal connected
    - account info available
    """
    try:
        term = mt5.terminal_info()
        if term is None:
            return False, "terminal_info_none"

        acc = mt5.account_info()
        if acc is None:
            return False, "account_info_none"

        connected = getattr(term, "connected", None)
        if connected is False:
            return False, "terminal_disconnected"

        trade_allowed = getattr(term, "trade_allowed", None)
        if trade_allowed is False:
            return False, "terminal_trade_not_allowed"

        return True, "ok"

    except Exception as e:
        return False, f"healthcheck_exception:{e}"

def reconnect_mt5(pause_sec: float = 2.0) -> bool:
    """
    Hard reconnect attempt.
    """
    try:
        mt5.shutdown()
    except Exception:
        pass

    time.sleep(pause_sec)

    try:
        ok = mt5.initialize()
        if not ok:
            return False

        acc = mt5.account_info()
        return acc is not None
    except Exception:
        return False

def ensure_symbol(symbol: str) -> SymbolMeta:
    if not mt5.symbol_select(symbol, True):
        raise RuntimeError(f"Cannot select symbol: {symbol}")

    info = mt5.symbol_info(symbol)
    if info is None:
        raise RuntimeError(f"symbol_info None: {symbol}")

    return SymbolMeta(
        name=symbol,
        contract_size=float(info.trade_contract_size),
        tick_size=float(info.trade_tick_size),
        tick_value=float(info.trade_tick_value),
        vol_min=float(info.volume_min),
        vol_max=float(info.volume_max),
        vol_step=float(info.volume_step),
        currency_base=str(getattr(info, "currency_base", "") or ""),
        currency_profit=str(getattr(info, "currency_profit", "") or ""),
        currency_margin=str(getattr(info, "currency_margin", "") or ""),
    )

def _symbol_max_tick_age_sec(cfg: ExecConfig, symbol: str) -> int:
    if symbol.endswith(".cash"):
        return int(cfg.max_tick_age_sec_index)
    return int(cfg.max_tick_age_sec_fx)


def tick_age_seconds(symbol: str) -> Optional[float]:
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        return None

    tick_ts = getattr(tick, "time", None)
    if tick_ts is None:
        return None

    return max(0.0, time.time() - float(tick_ts))


def tick_is_fresh(cfg: ExecConfig, symbol: str) -> Tuple[bool, Optional[float], Optional[float]]:
    age = tick_age_seconds(symbol)
    if age is None:
        return False, None, None

    max_age = float(_symbol_max_tick_age_sec(cfg, symbol))
    return age <= max_age, age, max_age

def _mid_price(symbol: str) -> Optional[float]:
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        return None
    bid = float(tick.bid)
    ask = float(tick.ask)
    if bid <= 0 and ask <= 0:
        return None
    if bid > 0 and ask > 0:
        return (bid + ask) / 2.0
    return ask if ask > 0 else bid


def _fx_rate_to_usd(ccy: str) -> Optional[float]:
    """
    Returns USD per 1 unit of ccy.
    Example:
      EUR -> price of EURUSD
      CHF -> 1 / USDCHF
      JPY -> 1 / USDJPY
    """
    ccy = str(ccy).upper()
    if ccy == "USD":
        return 1.0

    direct = f"{ccy}USD"
    px = _mid_price(direct)
    if px is not None and px > 0:
        return float(px)

    inverse = f"USD{ccy}"
    px = _mid_price(inverse)
    if px is not None and px > 0:
        return float(1.0 / px)

    return None

# ==========================
# SPREAD GUARD
# ==========================

def current_spread(symbol: str) -> float:
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        raise RuntimeError(f"No tick for symbol: {symbol}")
    return float(tick.ask) - float(tick.bid)


def max_allowed_spread(symbol: str) -> Optional[float]:
    return MAX_SPREAD_BY_SYMBOL.get(symbol)

SYMBOL_EXECUTION_CONFIG = {
    "US500.cash": {"deviation_points": 80, "retries": 5, "retry_sleep_sec": 0.5},
    "US100.cash": {"deviation_points": 120, "retries": 5, "retry_sleep_sec": 0.5},
    "US30.cash":  {"deviation_points": 180, "retries": 5, "retry_sleep_sec": 0.5},

    "EURJPY": {"deviation_points": 30, "retries": 4, "retry_sleep_sec": 0.35},
    "GBPJPY": {"deviation_points": 35, "retries": 4, "retry_sleep_sec": 0.35},
    "USDJPY": {"deviation_points": 25, "retries": 4, "retry_sleep_sec": 0.35},

    "EURCHF": {"deviation_points": 20, "retries": 4, "retry_sleep_sec": 0.35},
    "EURCAD": {"deviation_points": 25, "retries": 4, "retry_sleep_sec": 0.35},
    "GBPCHF": {"deviation_points": 25, "retries": 4, "retry_sleep_sec": 0.35},
    "EURUSD": {"deviation_points": 15, "retries": 4, "retry_sleep_sec": 0.25},
    "USDCHF": {"deviation_points": 20, "retries": 4, "retry_sleep_sec": 0.25},
}

def execution_params_for_symbol(cfg: ExecConfig, symbol: str) -> Dict:
    base = {
        "deviation_points": cfg.deviation_points,
        "retries": cfg.retries,
        "retry_sleep_sec": cfg.retry_sleep_sec,
        "order_filling": cfg.order_filling,
    }

    override = SYMBOL_EXECUTION_CONFIG.get(symbol, {})
    base.update(override)
    return base

def spread_guard_ok(symbol: str) -> bool:
    max_spread = max_allowed_spread(symbol)
    if max_spread is None:
        return True

    spr = current_spread(symbol)
    return spr <= float(max_spread)

# ==========================
# ACCOUNT
# ==========================

def account_snapshot() -> Dict[str, float]:
    acc = mt5.account_info()
    if acc is None:
        raise RuntimeError("account_info failed")

    return {
        "equity": float(acc.equity),
        "balance": float(acc.balance),
        "margin": float(acc.margin),
        "margin_free": float(acc.margin_free),
        "margin_level": float(acc.margin_level) if acc.margin_level else 0.0,
    }


# ==========================
# LOGGING
# ==========================
def append_trade_lifecycle_log(cfg: ExecConfig, row: Dict):
    fieldnames = [
        "ts",
        "strategy",
        "market",
        "symbol",
        "magic",
        "direction",
        "entry_time",
        "exit_time",
        "holding_seconds",
        "entry_price",
        "exit_price",
        "volume",
        "pnl",
        "exit_reason",
    ]

    file_exists = False
    try:
        with open(cfg.lifecycle_log_csv_path, "r", encoding="utf-8"):
            file_exists = True
    except Exception:
        pass

    with open(cfg.lifecycle_log_csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        safe_row = {k: row.get(k, "") for k in fieldnames}
        writer.writerow(safe_row)

def append_trade_log(cfg: ExecConfig, row: Dict):
    fieldnames = [
        "ts",
        "symbol",
        "side",
        "volume",
        "magic",
        "ticket",
        "comment",
        "price",
        "retcode",
        "order",
        "deal",
        "spread",
    ]

    file_exists = False
    try:
        with open(cfg.log_csv_path, "r", encoding="utf-8"):
            file_exists = True
    except Exception:
        pass

    with open(cfg.log_csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        safe_row = {k: row.get(k, "") for k in fieldnames}
        writer.writerow(safe_row)

def append_execution_event(cfg: ExecConfig, row: Dict):
    fieldnames = [
        "ts",
        "event_type",
        "symbol",
        "side",
        "volume",
        "magic",
        "comment",
        "attempt",
        "requested_price",
        "fill_price",
        "spread",
        "retcode",
        "order",
        "deal",
        "position_ticket",
        "exception",
        "bid",
        "ask",
    ]

    file_exists = False
    try:
        with open(cfg.event_log_csv_path, "r", encoding="utf-8"):
            file_exists = True
    except Exception:
        pass

    with open(cfg.event_log_csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        safe_row = {k: row.get(k, "") for k in fieldnames}
        writer.writerow(safe_row)

# ==========================
# POSITION HELPERS
# ==========================
def get_positions_by_magic(symbol: str, magic: int):
    pos = mt5.positions_get(symbol=symbol)
    if pos is None:
        return []

    out = []
    for p in pos:
        if int(p.magic) == int(magic):
            out.append(p)

    return out

def get_position(symbol: str, magic: int):
    matches = get_positions_by_magic(symbol, magic)

    if not matches:
        return None

    if len(matches) > 1:
        print(
            f"[{_ts()}] WARNING multiple positions found "
            f"symbol={symbol} magic={magic} count={len(matches)}"
        )

    return matches[0]


def get_position(symbol: str, magic: int):
    pos = mt5.positions_get(symbol=symbol)

    if pos is None:
        return None

    for p in pos:
        if int(p.magic) == int(magic):
            return p

    return None


# ==========================
# SIZING
# ==========================
def position_usd_notional(position) -> float:
    """
    Approx current USD notional for an open MT5 position.
    Uses the same logic as entry sizing.
    """
    symbol = str(position.symbol)
    volume = float(position.volume)

    meta = ensure_symbol(symbol)

    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        # fallback to position price_open if live tick is unavailable
        price = float(getattr(position, "price_open", 0.0) or 0.0)
    else:
        bid = float(tick.bid)
        ask = float(tick.ask)
        if bid > 0 and ask > 0:
            price = (bid + ask) / 2.0
        else:
            price = float(getattr(position, "price_open", 0.0) or 0.0)

    if price <= 0:
        return 0.0

    per_lot = usd_notional_per_lot(symbol, meta, price)
    return abs(per_lot * volume)

def round_volume(meta: SymbolMeta, volume: float) -> float:
    if volume <= 0:
        return 0.0

    volume = max(meta.vol_min, min(meta.vol_max, volume))

    step = meta.vol_step
    if step <= 0:
        return float(volume)

    steps = int(volume / step)  # round DOWN for safety
    vol = steps * step

    if vol < meta.vol_min:
        return 0.0

    vol = max(meta.vol_min, min(meta.vol_max, vol))
    return float(vol)


def usd_notional_per_lot(symbol: str, meta: SymbolMeta, price: float) -> float:
    """
    Approx USD notional per 1.0 lot.

    FX:
      - XXXUSD: contract_size * price
      - USDXXX: contract_size
      - Crosses: contract_size * (USD per 1 unit base)

    Non-FX / CFD fallback:
      - abs(price * contract_size)
    """
    base = meta.currency_base.upper()
    profit = meta.currency_profit.upper()

    # FX-like symbol if broker exposes base/profit currencies
    if len(base) == 3 and len(profit) == 3:
        if profit == "USD":
            return abs(meta.contract_size * price)

        if base == "USD":
            return abs(meta.contract_size)

        base_to_usd = _fx_rate_to_usd(base)
        if base_to_usd is not None and base_to_usd > 0:
            return abs(meta.contract_size * base_to_usd)

    # fallback for indices / CFDs
    return abs(price * meta.contract_size)


def notional_to_volume(meta: SymbolMeta, notional_usd: float, price: float, symbol: str) -> float:
    denom = usd_notional_per_lot(symbol, meta, price)

    if denom <= 0:
        return 0.0

    volume = notional_usd / denom
    return round_volume(meta, volume)


# ==========================
# ORDER CORE
# ==========================

def _order_type(side: str):
    side = side.upper()

    if side == "BUY":
        return mt5.ORDER_TYPE_BUY
    if side == "SELL":
        return mt5.ORDER_TYPE_SELL

    raise ValueError("side must be BUY or SELL")


def _price(symbol: str, side: str):
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        raise RuntimeError(f"No tick for symbol={symbol}")

    side = side.upper()
    if side == "BUY":
        return float(tick.ask)
    return float(tick.bid)


def send_market_order(
    symbol: str,
    side: str,
    volume: float,
    magic: int,
    cfg: ExecConfig,
    comment: str = "",
    position_ticket: Optional[int] = None,
    enforce_spread_guard: bool = True,
) -> bool:

    side = side.upper()
    meta = ensure_symbol(symbol)
    volume = round_volume(meta, volume)

    if volume <= 0:
        append_execution_event(cfg, {
            "ts": _ts(),
            "event_type": "ORDER_SKIPPED_ZERO_VOLUME",
            "symbol": symbol,
            "side": side,
            "volume": volume,
            "magic": magic,
            "comment": comment,
            "position_ticket": position_ticket,
        })
        print(f"[{_ts()}] ORDER SKIP volume<=0 symbol={symbol} side={side}")
        return False

    fresh_ok, tick_age, max_tick_age = tick_is_fresh(cfg, symbol)
    if not fresh_ok:
        append_execution_event(cfg, {
            "ts": _ts(),
            "event_type": "STALE_TICK_BLOCK",
            "symbol": symbol,
            "side": side,
            "volume": volume,
            "magic": magic,
            "comment": comment,
            "position_ticket": position_ticket,
            "exception": f"tick_age={tick_age}, max_tick_age={max_tick_age}",
        })
        print(
            f"[{_ts()}] STALE TICK BLOCK "
            f"symbol={symbol} tick_age={tick_age} max_allowed={max_tick_age}"
        )
        return False

    tick0 = mt5.symbol_info_tick(symbol)
    bid0 = float(tick0.bid) if tick0 else None
    ask0 = float(tick0.ask) if tick0 else None
    spread0 = (ask0 - bid0) if (bid0 is not None and ask0 is not None) else None

    if enforce_spread_guard:
        max_spread = max_allowed_spread(symbol)
        if max_spread is not None:
            try:
                spr = current_spread(symbol)
            except Exception as e:
                append_execution_event(cfg, {
                    "ts": _ts(),
                    "event_type": "SPREAD_GUARD_ERROR",
                    "symbol": symbol,
                    "side": side,
                    "volume": volume,
                    "magic": magic,
                    "comment": comment,
                    "position_ticket": position_ticket,
                    "exception": str(e),
                    "bid": bid0,
                    "ask": ask0,
                    "spread": spread0,
                })
                print(f"[SPREAD GUARD ERROR] symbol={symbol} err={e}")
                return False

            if spr > float(max_spread):
                append_execution_event(cfg, {
                    "ts": _ts(),
                    "event_type": "SPREAD_BLOCK",
                    "symbol": symbol,
                    "side": side,
                    "volume": volume,
                    "magic": magic,
                    "comment": comment,
                    "position_ticket": position_ticket,
                    "spread": spr,
                    "bid": bid0,
                    "ask": ask0,
                })
                print(
                    f"[SPREAD GUARD] blocked order "
                    f"symbol={symbol} spread={spr:.6f} max={float(max_spread):.6f} "
                    f"comment={comment}"
                )
                return False

    exec_params = execution_params_for_symbol(cfg, symbol)
    deviation_points = int(exec_params["deviation_points"])
    retries = int(exec_params["retries"])
    retry_sleep_sec = float(exec_params["retry_sleep_sec"])
    order_filling = int(exec_params["order_filling"])

    for attempt in range(1, retries + 1):
        try:
            price = _price(symbol, side)

            append_execution_event(cfg, {
                "ts": _ts(),
                "event_type": "ORDER_ATTEMPT",
                "symbol": symbol,
                "side": side,
                "volume": volume,
                "magic": magic,
                "comment": comment,
                "attempt": attempt,
                "requested_price": price,
                "spread": spread0,
                "position_ticket": position_ticket,
                "bid": bid0,
                "ask": ask0,
            })

            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": volume,
                "type": _order_type(side),
                "price": price,
                "deviation": deviation_points,
                "magic": magic,
                "comment": comment[:30],
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": order_filling,
            }

            if position_ticket is not None:
                request["position"] = int(position_ticket)

            result = mt5.order_send(request)

            if result is None:
                append_execution_event(cfg, {
                    "ts": _ts(),
                    "event_type": "ORDER_SEND_NONE",
                    "symbol": symbol,
                    "side": side,
                    "volume": volume,
                    "magic": magic,
                    "comment": comment,
                    "attempt": attempt,
                    "requested_price": price,
                    "position_ticket": position_ticket,
                    "bid": bid0,
                    "ask": ask0,
                    "spread": spread0,
                })
                print(f"[{_ts()}] order_send returned None symbol={symbol} side={side} attempt={attempt}")
                time.sleep(retry_sleep_sec)
                continue

            fill_price = float(getattr(result, "price", 0.0) or 0.0)
            order_id = getattr(result, "order", "")
            deal_id = getattr(result, "deal", "")
            retcode = getattr(result, "retcode", "")

            if result.retcode == mt5.TRADE_RETCODE_DONE:
                append_execution_event(cfg, {
                    "ts": _ts(),
                    "event_type": "ORDER_FILLED",
                    "symbol": symbol,
                    "side": side,
                    "volume": volume,
                    "magic": magic,
                    "comment": comment,
                    "attempt": attempt,
                    "requested_price": price,
                    "fill_price": fill_price,
                    "spread": spread0,
                    "retcode": retcode,
                    "order": order_id,
                    "deal": deal_id,
                    "position_ticket": position_ticket,
                    "bid": bid0,
                    "ask": ask0,
                })

                append_trade_log(cfg, {
                    "ts": _ts(),
                    "symbol": symbol,
                    "side": side,
                    "volume": volume,
                    "magic": magic,
                    "ticket": position_ticket,
                    "comment": comment,
                    "price": fill_price if fill_price > 0 else price,
                    "retcode": retcode,
                    "order": order_id,
                    "deal": deal_id,
                    "spread": spread0,
                })
                return True

            append_execution_event(cfg, {
                "ts": _ts(),
                "event_type": "ORDER_REJECTED",
                "symbol": symbol,
                "side": side,
                "volume": volume,
                "magic": magic,
                "comment": comment,
                "attempt": attempt,
                "requested_price": price,
                "fill_price": fill_price,
                "spread": spread0,
                "retcode": retcode,
                "order": order_id,
                "deal": deal_id,
                "position_ticket": position_ticket,
                "bid": bid0,
                "ask": ask0,
            })

            print(
                f"[{_ts()}] ORDER FAIL symbol={symbol} side={side} volume={volume:.4f} "
                f"retcode={result.retcode} comment={comment} attempt={attempt}"
            )

        except Exception as e:
            append_execution_event(cfg, {
                "ts": _ts(),
                "event_type": "ORDER_EXCEPTION",
                "symbol": symbol,
                "side": side,
                "volume": volume,
                "magic": magic,
                "comment": comment,
                "attempt": attempt,
                "position_ticket": position_ticket,
                "exception": str(e),
                "bid": bid0,
                "ask": ask0,
                "spread": spread0,
            })
            print(f"[{_ts()}] ORDER EXCEPTION symbol={symbol} side={side} err={e}")

        time.sleep(retry_sleep_sec)

    return False

# ==========================
# CLOSE POSITION (SAFE)
# ==========================

def close_position_market(
    symbol: str,
    magic: int,
    cfg: ExecConfig,
    comment: str = "CLOSE"
) -> bool:
    p = get_position(symbol, magic)

    if p is None:
        print(f"[{_ts()}] CLOSE SKIP no position symbol={symbol} magic={magic}")
        return True

    volume = float(p.volume)

    if p.type == mt5.POSITION_TYPE_BUY:
        side = "SELL"
    else:
        side = "BUY"

    ticket = int(p.ticket)

    ok = send_market_order(
        symbol=symbol,
        side=side,
        volume=volume,
        magic=magic,
        cfg=cfg,
        comment=comment,
        position_ticket=ticket,
        enforce_spread_guard=False,
    )

    return ok


# ==========================
# PANIC CLOSE ALL
# ==========================

def close_all_positions(cfg: ExecConfig):
    positions = mt5.positions_get()

    if positions is None:
        return

    for p in positions:
        symbol = p.symbol
        volume = float(p.volume)
        ticket = int(p.ticket)

        if p.type == mt5.POSITION_TYPE_BUY:
            side = "SELL"
        else:
            side = "BUY"

        send_market_order(
            symbol=symbol,
            side=side,
            volume=volume,
            magic=int(p.magic),
            cfg=cfg,
            comment="PANIC_CLOSE",
            position_ticket=ticket,
            enforce_spread_guard=False,
        )


# ==========================
# ENTRY
# ==========================

def _entry_allowed_by_margin(cfg: ExecConfig) -> bool:
    snap = account_snapshot()
    ml = float(snap["margin_level"])

    # Some brokers return 0 when margin is near zero or not meaningful.
    if ml > 0 and ml < cfg.min_margin_level:
        print(f"[{_ts()}] ORDER BLOCKED low margin_level={ml:.2f}")
        return False

    return True


def open_long_by_notional(
    symbol: str,
    notional_usd: float,
    magic: int,
    cfg: ExecConfig,
    comment: str = ""
) -> Tuple[bool, float]:
    meta = ensure_symbol(symbol)

    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        return False, 0.0

    if not _entry_allowed_by_margin(cfg):
        return False, 0.0

    price = (float(tick.bid) + float(tick.ask)) / 2.0
    volume = notional_to_volume(meta, notional_usd, price, symbol)

    if volume <= 0:
        return False, 0.0

    ok = send_market_order(
        symbol=symbol,
        side="BUY",
        volume=volume,
        magic=magic,
        cfg=cfg,
        comment=comment,
        enforce_spread_guard=True,
    )

    return ok, volume


def open_short_by_notional(
    symbol: str,
    notional_usd: float,
    magic: int,
    cfg: ExecConfig,
    comment: str = ""
) -> Tuple[bool, float]:
    meta = ensure_symbol(symbol)

    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        return False, 0.0

    if not _entry_allowed_by_margin(cfg):
        return False, 0.0

    price = (float(tick.bid) + float(tick.ask)) / 2.0
    volume = notional_to_volume(meta, notional_usd, price, symbol)

    if volume <= 0:
        return False, 0.0

    ok = send_market_order(
        symbol=symbol,
        side="SELL",
        volume=volume,
        magic=magic,
        cfg=cfg,
        comment=comment,
        enforce_spread_guard=True,
    )

    return ok, volume
