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


# ==========================
# POSITION HELPERS
# ==========================

def get_positions(symbol: Optional[str] = None):
    if symbol:
        return mt5.positions_get(symbol=symbol)
    return mt5.positions_get()


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
        print(f"[{_ts()}] ORDER SKIP volume<=0 symbol={symbol} side={side}")
        return False

    if enforce_spread_guard:
        max_spread = max_allowed_spread(symbol)
        if max_spread is not None:
            try:
                spr = current_spread(symbol)
            except Exception as e:
                print(f"[SPREAD GUARD ERROR] symbol={symbol} err={e}")
                return False

            if spr > float(max_spread):
                print(
                    f"[SPREAD GUARD] blocked order "
                    f"symbol={symbol} spread={spr:.6f} max={float(max_spread):.6f} "
                    f"comment={comment}"
                )
                return False

    for attempt in range(cfg.retries):
        try:
            price = _price(symbol, side)

            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": volume,
                "type": _order_type(side),
                "price": price,
                "deviation": cfg.deviation_points,
                "magic": magic,
                "comment": comment[:30],
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": cfg.order_filling,
            }

            # Important for hedging accounts / explicit close
            if position_ticket is not None:
                request["position"] = int(position_ticket)

            result = mt5.order_send(request)

            if result is None:
                print(f"[{_ts()}] order_send returned None symbol={symbol} side={side} attempt={attempt+1}")
                time.sleep(cfg.retry_sleep_sec)
                continue

            if result.retcode == mt5.TRADE_RETCODE_DONE:
                append_trade_log(cfg, {
                    "ts": _ts(),
                    "symbol": symbol,
                    "side": side,
                    "volume": volume,
                    "magic": magic,
                    "ticket": position_ticket,
                    "comment": comment,
                    "price": price,
                    "retcode": result.retcode,
                })
                return True

            print(
                f"[{_ts()}] ORDER FAIL symbol={symbol} side={side} volume={volume:.4f} "
                f"retcode={result.retcode} comment={comment} attempt={attempt+1}"
            )

        except Exception as e:
            print(f"[{_ts()}] ORDER EXCEPTION symbol={symbol} side={side} err={e}")

        time.sleep(cfg.retry_sleep_sec)

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
