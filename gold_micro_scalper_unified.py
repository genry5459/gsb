#!/usr/bin/env python3
"""
Gold Micro Scalper - unified single-file bot for XAUUSD on Capital.com.

What this file includes:
  - Capital.com REST connector
  - Indicator pipeline
  - Micro-scalping strategies
  - Risk management with multi-TP and trailing stop
  - GradientBoosting ML filter
  - Adaptive strategy ranking
  - Realistic backtest with spread, commission, and walk-forward scoring
  - Live / dry-run / backtest entrypoint

This is intentionally a single file so it can be dropped into a repository
without importing a multi-file package.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import signal
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from http import HTTPStatus
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, List, Optional, Sequence, Tuple
import urllib.error
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd

try:
    import ta
except ImportError as exc:  # pragma: no cover
    ta = None  # type: ignore
    _TA_IMPORT_ERROR = exc
else:
    _TA_IMPORT_ERROR = None


# ──────────────────────────────────────────────────────────────────────────────
# Environment loading
# ──────────────────────────────────────────────────────────────────────────────


def load_env_file(path: Path) -> None:
    """Load simple KEY=VALUE pairs from a .env file if present."""
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


HERE = Path(__file__).resolve().parent
load_env_file(HERE / ".env")


# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class OandaConfig:
    api_token: str = os.getenv("OANDA_API_TOKEN", "")
    account_id: str = os.getenv("OANDA_ACCOUNT_ID", "")
    demo: bool = os.getenv("OANDA_DEMO", "true").lower() == "true"


@dataclass
class TradingConfig:
    symbol: str = os.getenv("SYMBOL", "XAUUSD")
    timeframe: str = os.getenv("TIMEFRAME", "5m")
    initial_capital: float = float(os.getenv("INITIAL_CAPITAL", "100000"))
    risk_pct: float = float(os.getenv("RISK_PCT", "0.005"))
    leverage: float = float(os.getenv("LEVERAGE", "200"))
    lot_size: float = float(os.getenv("LOT_SIZE", "0.10"))
    use_dynamic_sizing: bool = os.getenv("USE_DYNAMIC_SIZING", "true").lower() == "true"

    # Micro-scalping defaults based on ATR multiples rather than fixed percent targets.
    sl_atr_mult: float = float(os.getenv("SL_ATR_MULT", "0.90"))
    tp1_atr_mult: float = float(os.getenv("TP1_ATR_MULT", "0.70"))
    tp2_atr_mult: float = float(os.getenv("TP2_ATR_MULT", "1.15"))
    tp3_atr_mult: float = float(os.getenv("TP3_ATR_MULT", "1.70"))
    tp1_close: float = float(os.getenv("TP1_CLOSE", "0.50"))
    tp2_close: float = float(os.getenv("TP2_CLOSE", "0.30"))
    tp3_close: float = float(os.getenv("TP3_CLOSE", "0.20"))
    trailing_activate_rr: float = float(os.getenv("TRAILING_ACTIVATE_RR", "0.60"))
    trailing_step_atr: float = float(os.getenv("TRAILING_STEP_ATR", "0.25"))
    breakeven_buffer_atr: float = float(os.getenv("BREAKEVEN_BUFFER_ATR", "0.05"))

    cooldown_bars: int = int(os.getenv("COOLDOWN_BARS", "2"))
    max_daily_loss_pct: float = float(os.getenv("MAX_DAILY_LOSS_PCT", "0.03"))
    max_concurrent: int = int(os.getenv("MAX_CONCURRENT", "1"))
    max_hold_bars: int = int(os.getenv("MAX_HOLD_BARS", "10"))
    max_consecutive_losses: int = int(os.getenv("MAX_CONSECUTIVE_LOSSES", "3"))

    # Execution filters
    max_spread_abs: float = float(os.getenv("MAX_SPREAD_ABS", "5.0"))
    max_spread_atr_frac: float = float(os.getenv("MAX_SPREAD_ATR_FRAC", "0.20"))
    max_volatility_atr_pct: float = float(os.getenv("MAX_ATR_PCT", "0.0075"))
    min_atr_pct: float = float(os.getenv("MIN_ATR_PCT", "0.0006"))
    session_start_utc: int = int(os.getenv("SESSION_START_UTC", "6"))
    session_end_utc: int = int(os.getenv("SESSION_END_UTC", "21"))

    # ML
    ml_enabled: bool = os.getenv("ML_ENABLED", "true").lower() == "true"
    ml_retrain_interval: int = int(os.getenv("ML_RETRAIN_TRADES", "25"))
    ml_min_samples: int = int(os.getenv("ML_MIN_SAMPLES", "80"))
    ml_threshold: float = float(os.getenv("ML_THRESHOLD", "0.55"))

    # Strategy control
    min_signal_strength: int = int(os.getenv("MIN_SIGNAL_STRENGTH", "1"))
    best_strategy_lookback: int = int(os.getenv("BEST_STRATEGY_LOOKBACK", "600"))
    min_live_strategy_score: float = float(os.getenv("MIN_LIVE_STRATEGY_SCORE", "0.05"))


@dataclass
class ServerConfig:
    poll_interval: int = int(os.getenv("POLL_INTERVAL", "10"))
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    log_file: str = os.getenv("LOG_FILE", "gold_micro_scalper.log")
    state_file: str = os.getenv("STATE_FILE", "gold_micro_scalper_state.json")


@dataclass
class BacktestConfig:
    commission_roundtrip_per_lot: float = float(os.getenv("COMMISSION_RT_PER_LOT", "7.0"))
    slippage_atr_frac: float = float(os.getenv("SLIPPAGE_ATR_FRAC", "0.04"))
    estimated_spread_atr_frac: float = float(os.getenv("ESTIMATED_SPREAD_ATR_FRAC", "0.06"))
    starting_equity: float = float(os.getenv("BACKTEST_STARTING_EQUITY", "100000"))
    walk_forward_train_frac: float = float(os.getenv("WF_TRAIN_FRAC", "0.70"))
    walk_forward_test_frac: float = float(os.getenv("WF_TEST_FRAC", "0.15"))
    warmup_bars: int = int(os.getenv("BACKTEST_WARMUP_BARS", "200"))
    max_trades_per_day: int = int(os.getenv("MAX_TRADES_PER_DAY", "999"))
    use_estimated_spread: bool = os.getenv("BACKTEST_USE_SPREAD", "true").lower() == "true"


OANDA = OandaConfig()
TRADING = TradingConfig()
SERVER = ServerConfig()
BACKTEST = BacktestConfig()


# ──────────────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────────────


def setup_logging() -> None:
    fmt = "%(asctime)s | %(name)-14s | %(levelname)-8s | %(message)s"
    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(SERVER.log_file, encoding="utf-8"),
    ]
    logging.basicConfig(
        level=getattr(logging, SERVER.log_level.upper(), logging.INFO),
        format=fmt,
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
    )


log = logging.getLogger("GoldMicroScalp")


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def clamp(value: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, value)))


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        if isinstance(value, (float, int, np.floating, np.integer)):
            if pd.isna(value):
                return default
            return float(value)
        return float(value)
    except Exception:
        return default


def timeframe_to_yf_interval(tf: str) -> str:
    tf = tf.lower().strip()
    mapping = {
        "1m": "1m",
        "2m": "2m",
        "5m": "5m",
        "15m": "15m",
        "30m": "30m",
        "60m": "60m",
        "1h": "60m",
        "4h": "1h",
        "1d": "1d",
    }
    return mapping.get(tf, "5m")


def yfinance_period_for_interval(interval: str, requested_days: int) -> str:
    interval = interval.lower()
    if interval in {"1m", "2m"}:
        return "7d"
    if interval == "5m":
        return f"{min(requested_days, 60)}d"
    if interval in {"15m", "30m", "60m", "1h"}:
        return f"{min(requested_days, 730)}d"
    return f"{max(requested_days, 365)}d"


def utc_hour_index(index: pd.Index) -> pd.Series:
    if not isinstance(index, pd.DatetimeIndex):
        return pd.Series([0] * len(index), index=index)
    dt = index
    if dt.tz is None:
        return pd.Series(dt.hour, index=index)
    return pd.Series(dt.tz_convert(timezone.utc).hour, index=index)


def percentile_rank(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) else np.nan,
        raw=False,
    )


def ensure_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    cols = {c.lower(): c for c in df.columns}
    rename = {}
    for key in ["open", "high", "low", "close", "volume"]:
        if key in cols:
            rename[cols[key]] = key.capitalize()
    out = df.rename(columns=rename).copy()
    missing = [x for x in ["Open", "High", "Low", "Close", "Volume"] if x not in out.columns]
    if missing:
        raise ValueError(f"Missing required OHLCV columns: {missing}")
    if not isinstance(out.index, pd.DatetimeIndex):
        out.index = pd.to_datetime(out.index, utc=True, errors="coerce")
    out = out.sort_index().dropna(subset=["Open", "High", "Low", "Close"])
    if "Volume" not in out.columns:
        out["Volume"] = 0.0
    out["Volume"] = out["Volume"].fillna(0.0)
    return out


def extract_trade_date(ts: pd.Timestamp) -> pd.Timestamp:
    if ts.tzinfo is None:
        return pd.Timestamp(ts.date())
    return pd.Timestamp(ts.tz_convert(timezone.utc).date())


# ──────────────────────────────────────────────────────────────────────────────
# Capital.com REST connector
# ──────────────────────────────────────────────────────────────────────────────


TIMEFRAME_MAP = {
    "1m": "M1",
    "2m": "M2",
    "5m": "M5",
    "15m": "M15",
    "30m": "M30",
    "1H": "H1",
    "4H": "H4",
    "1D": "D",
}


@dataclass
class OrderResult:
    success: bool
    deal_id: str = ""
    error: str = ""
    raw: dict = field(default_factory=dict)


@dataclass
class SimpleResponse:
    status_code: int
    headers: Dict[str, str]
    body: bytes

    @property
    def text(self) -> str:
        return self.body.decode("utf-8", errors="replace")

    def json(self) -> Dict[str, Any]:
        txt = self.text.strip()
        return json.loads(txt) if txt else {}


class SimpleHttpSession:
    """Tiny requests-like wrapper on top of urllib."""

    def request(self, method: str, url: str, **kwargs) -> SimpleResponse:
        headers = dict(kwargs.pop("headers", {}) or {})
        params = kwargs.pop("params", None)
        json_data = kwargs.pop("json", None)
        timeout = kwargs.pop("timeout", 20)
        if kwargs:
            raise TypeError(f"Unsupported kwargs for SimpleHttpSession: {sorted(kwargs.keys())}")
        if params:
            query = urllib.parse.urlencode(params, doseq=True)
            url = f"{url}&{query}" if "?" in url else f"{url}?{query}"
        body = None
        if json_data is not None:
            body = json.dumps(json_data).encode("utf-8")
            headers.setdefault("Content-Type", "application/json")
        req = urllib.request.Request(url, data=body, headers=headers, method=method.upper())
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return SimpleResponse(
                    status_code=getattr(resp, "status", HTTPStatus.OK),
                    headers=dict(resp.headers.items()),
                    body=resp.read(),
                )
        except urllib.error.HTTPError as exc:
            return SimpleResponse(
                status_code=int(getattr(exc, "code", 500)),
                headers=dict(exc.headers.items()) if exc.headers else {},
                body=exc.read() if hasattr(exc, "read") else b"",
            )


class OandaClient:
    DEMO_URL = "https://api-fxpractice.oanda.com"
    LIVE_URL = "https://api-fxtrade.oanda.com"

    def __init__(self, api_token: str, account_id: str, demo: bool = True):
        self.api_token = api_token
        self.account_id = account_id
        self.demo = demo
        self.base_url = self.DEMO_URL if demo else self.LIVE_URL
        self._session = SimpleHttpSession()

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json",
        }

    def _request(self, method: str, path: str, **kwargs) -> dict:
        url = f"{self.base_url}{path}"
        kwargs.setdefault("timeout", 20)
        kwargs.setdefault("headers", self._headers())
        resp = self._session.request(method, url, **kwargs)
        if resp.status_code == 204:
            return {}
        if resp.status_code >= 400:
            raise RuntimeError(f"API {resp.status_code}: {resp.text[:500]}")
        try:
            return resp.json()
        except Exception:
            return {}

    def get_candles(self, instrument: str, granularity: str, count: int = 500) -> List[Dict[str, Any]]:
        params = {
            "count": count,
            "granularity": granularity,
            "price": "M",
        }
        data = self._request("GET", f"/v3/instruments/{instrument}/candles", params=params)
        rows = []
        for c in data.get("candles", []):
            if not c.get("complete"):
                continue
            try:
                ts = c["time"]
                mid = c["mid"]
                stamp = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                rows.append(
                    {
                        "time": stamp,
                        "open": safe_float(mid["o"]),
                        "high": safe_float(mid["h"]),
                        "low": safe_float(mid["l"]),
                        "close": safe_float(mid["c"]),
                        "volume": safe_float(c.get("volume", 0)),
                    }
                )
            except Exception:
                continue
        return rows

    def get_market_price(self, instrument: str) -> Dict[str, float]:
        data = self._request("GET", f"/v3/accounts/{self.account_id}/pricing", params={"instruments": instrument})
        prices = data.get("prices", [])
        if not prices:
            return {"bid": 0.0, "ask": 0.0, "spread": 0.0, "instrument": instrument}
        p = prices[0]
        bid = safe_float(p.get("bid"))
        ask = safe_float(p.get("ask"))
        return {
            "bid": bid,
            "ask": ask,
            "spread": max(0.0, ask - bid),
            "instrument": instrument,
        }

    def get_account_info(self) -> Dict[str, float]:
        data = self._request("GET", f"/v3/accounts/{self.account_id}")
        acc = data.get("account", {})
        return {
            "balance": safe_float(acc.get("balance")),
            "available": safe_float(acc.get("marginAvailable")),
            "pnl": safe_float(acc.get("unrealizedPL")),
            "margin": safe_float(acc.get("marginUsed")),
        }

    def get_positions(self) -> List[Dict[str, Any]]:
        data = self._request("GET", f"/v3/accounts/{self.account_id}/openTrades")
        trades = data.get("trades", [])
        positions = []
        for t in trades:
            units = float(t.get("units", 0))
            positions.append({
                "dealId": t["id"],
                "direction": "BUY" if units > 0 else "SELL",
                "size": abs(units),
                "openLevel": safe_float(t.get("price")),
                "instrument": t.get("instrument"),
                "stopLevel": safe_float(t.get("stopLossOrder", {}).get("price")) if t.get("stopLossOrder") else None,
            })
        return positions

    def open_position(
        self,
        instrument: str,
        direction: str,
        size: float,
        stop_level: Optional[float] = None,
        limit_level: Optional[float] = None,
    ) -> OrderResult:
        units = str(size) if direction.upper() == "BUY" else str(-size)
        order = {
            "type": "MARKET",
            "instrument": instrument,
            "units": units,
            "timeInForce": "FOK",
        }
        if stop_level is not None:
            order["stopLossOnFill"] = {
                "price": str(round(float(stop_level), 2)),
                "timeInForce": "GTC",
            }
        if limit_level is not None:
            order["takeProfitOnFill"] = {
                "price": str(round(float(limit_level), 2)),
                "timeInForce": "GTC",
            }
        try:
            data = self._request("POST", f"/v3/accounts/{self.account_id}/orders", json={"order": order})
            tx = data.get("orderFillTransaction") or data.get("orderCreateTransaction")
            if tx:
                deal_id = tx.get("id", "")
                return OrderResult(success=True, deal_id=deal_id, raw=data)
            return OrderResult(success=False, error="No transaction in response", raw=data)
        except Exception as exc:
            return OrderResult(success=False, error=str(exc))

    def modify_position(
        self,
        deal_id: str,
        stop_level: Optional[float] = None,
        limit_level: Optional[float] = None,
    ) -> OrderResult:
        body: Dict[str, Any] = {}
        if stop_level is not None:
            body["stopLoss"] = {
                "price": str(round(float(stop_level), 2)),
                "timeInForce": "GTC",
            }
        if limit_level is not None:
            body["takeProfit"] = {
                "price": str(round(float(limit_level), 2)),
                "timeInForce": "GTC",
            }
        try:
            data = self._request("PUT", f"/v3/accounts/{self.account_id}/trades/{deal_id}/orders", json=body)
            return OrderResult(success=True, raw=data)
        except Exception as exc:
            return OrderResult(success=False, error=str(exc))

    def close_position(self, deal_id: str, units: Optional[float] = None) -> OrderResult:
        try:
            payload: Optional[Dict[str, Any]] = None
            if units is not None:
                payload = {"units": str(units)}
            data = self._request("PUT", f"/v3/accounts/{self.account_id}/trades/{deal_id}/close", json=payload)
            return OrderResult(success=True, deal_id=deal_id, raw=data)
        except Exception as exc:
            return OrderResult(success=False, error=str(exc))


# ──────────────────────────────────────────────────────────────────────────────
# Indicators
# ──────────────────────────────────────────────────────────────────────────────


def require_ta() -> None:
    if ta is None:
        raise ImportError(
            "The 'ta' package is required. Install dependencies first."
        ) from _TA_IMPORT_ERROR


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add the indicator set used by the strategies and ML layer."""
    require_ta()
    df = ensure_ohlcv(df).copy()
    c, h, l, v = df["Close"], df["High"], df["Low"], df["Volume"]

    # Returns / regime features
    df["ret_1"] = c.pct_change()
    df["ret_3"] = c.pct_change(3)
    df["ret_6"] = c.pct_change(6)
    df["hour_utc"] = utc_hour_index(df.index)
    df["is_london"] = ((df["hour_utc"] >= 7) & (df["hour_utc"] <= 11)).astype(int)
    df["is_ny"] = ((df["hour_utc"] >= 13) & (df["hour_utc"] <= 17)).astype(int)
    df["is_overlap"] = ((df["hour_utc"] >= 13) & (df["hour_utc"] <= 16)).astype(int)

    # Momentum
    df["rsi_7"] = ta.momentum.RSIIndicator(c, window=7).rsi()
    df["rsi_14"] = ta.momentum.RSIIndicator(c, window=14).rsi()
    stoch = ta.momentum.StochasticOscillator(h, l, c, window=14, smooth_window=3)
    df["stoch_k"] = stoch.stoch()
    df["stoch_d"] = stoch.stoch_signal()
    srsi = ta.momentum.StochRSIIndicator(c, window=14, smooth1=3, smooth2=3)
    df["stoch_rsi_k"] = srsi.stochrsi_k()
    df["willr"] = ta.momentum.WilliamsRIndicator(h, l, c, lbp=14).williams_r()
    df["cci"] = ta.trend.CCIIndicator(h, l, c, window=20).cci()
    df["roc_3"] = ta.momentum.ROCIndicator(c, window=3).roc()
    df["roc_5"] = ta.momentum.ROCIndicator(c, window=5).roc()
    df["mfi"] = ta.volume.MFIIndicator(h, l, c, v, window=14).money_flow_index()

    # Trend
    macd = ta.trend.MACD(c, window_slow=26, window_fast=12, window_sign=9)
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_hist"] = macd.macd_diff()
    for w in [3, 5, 8, 13, 20, 21, 34, 50, 100, 200]:
        df[f"ema_{w}"] = ta.trend.EMAIndicator(c, window=w).ema_indicator()
    adx = ta.trend.ADXIndicator(h, l, c, window=14)
    df["adx"] = adx.adx()
    df["adx_pos"] = adx.adx_pos()
    df["adx_neg"] = adx.adx_neg()

    # Ichimoku
    ich = ta.trend.IchimokuIndicator(h, l, window1=9, window2=26, window3=52)
    df["ich_a"] = ich.ichimoku_a()
    df["ich_b"] = ich.ichimoku_b()
    df["ich_conv"] = ich.ichimoku_conversion_line()
    df["ich_base"] = ich.ichimoku_base_line()

    # Volatility
    bb = ta.volatility.BollingerBands(c, window=20, window_dev=2)
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_lower"] = bb.bollinger_lband()
    df["bb_mid"] = bb.bollinger_mavg()
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"]
    df["atr_14"] = ta.volatility.AverageTrueRange(h, l, c, window=14).average_true_range()
    kc = ta.volatility.KeltnerChannel(h, l, c, window=20, window_atr=10)
    df["kc_upper"] = kc.keltner_channel_hband()
    df["kc_lower"] = kc.keltner_channel_lband()
    df["atr_pct"] = df["atr_14"] / c
    df["tr"] = pd.concat(
        [(h - l), (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1
    ).max(axis=1)
    df["tr_pct"] = df["tr"] / c

    # Volume / flow
    df["obv"] = ta.volume.OnBalanceVolumeIndicator(c, v).on_balance_volume()
    df["vol_sma_20"] = v.rolling(20).mean()
    df["vol_ratio"] = v / df["vol_sma_20"]
    df["vol_z"] = (v - v.rolling(20).mean()) / v.rolling(20).std(ddof=0)
    df["cmf"] = ta.volume.ChaikinMoneyFlowIndicator(h, l, c, v, window=20).chaikin_money_flow()
    df["vwap"] = (v * (h + l + c) / 3).cumsum() / v.cumsum()
    df["vwap_dist_atr"] = (c - df["vwap"]) / df["atr_14"]

    # Supertrend approximation
    hl2 = (h + l) / 2
    atr = df["atr_14"]
    upband = hl2 + (1.5 * atr)
    dnband = hl2 - (1.5 * atr)
    direction = np.where(c > upband.shift(1), 1, np.where(c < dnband.shift(1), -1, np.nan))
    df["supertrend_dir"] = pd.Series(direction, index=df.index).ffill().fillna(1).astype(int)

    # Derived
    df["ema_spread"] = (df["ema_5"] - df["ema_20"]) / c * 10000
    df["price_vs_vwap"] = (c - df["vwap"]) / c * 10000
    df["rsi_divergence"] = df["rsi_7"] - df["rsi_14"]
    df["stoch_cross"] = df["stoch_k"] - df["stoch_d"]
    df["range_pct"] = (h - l) / c
    df["roll_high_10"] = h.rolling(10).max()
    df["roll_low_10"] = l.rolling(10).min()
    df["donchian_breakout_up"] = (c > df["roll_high_10"].shift(1)).astype(int)
    df["donchian_breakout_dn"] = (c < df["roll_low_10"].shift(1)).astype(int)

    return df


# ──────────────────────────────────────────────────────────────────────────────
# Strategies
# ──────────────────────────────────────────────────────────────────────────────


class BaseStrategy:
    name = "Base"
    description = ""
    min_bars = 60

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        raise NotImplementedError


class TrendPullbackScalp(BaseStrategy):
    name = "TrendPullback"
    description = "EMA stack + pullback to value + ADX + VWAP"
    min_bars = 80

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        sig = pd.Series(0, index=df.index)
        bull = (df["ema_5"] > df["ema_8"]) & (df["ema_8"] > df["ema_20"]) & (df["ema_20"] > df["ema_50"])
        bear = (df["ema_5"] < df["ema_8"]) & (df["ema_8"] < df["ema_20"]) & (df["ema_20"] < df["ema_50"])
        pull_long = (df["Close"] <= df["ema_20"]) | (df["Low"] <= df["ema_20"])
        pull_short = (df["Close"] >= df["ema_20"]) | (df["High"] >= df["ema_20"])
        confirm_long = (df["rsi_14"].between(45, 60)) & (df["adx"] > 20) & (df["Close"] > df["vwap"]) & (df["vol_ratio"] > 0.9)
        confirm_short = (df["rsi_14"].between(40, 55)) & (df["adx"] > 20) & (df["Close"] < df["vwap"]) & (df["vol_ratio"] > 0.9)
        long_reclaim = (df["Close"] > df["ema_8"]) & (df["Close"].shift(1) <= df["ema_8"].shift(1))
        short_reclaim = (df["Close"] < df["ema_8"]) & (df["Close"].shift(1) >= df["ema_8"].shift(1))
        sig[bull & pull_long & confirm_long & long_reclaim] = 1
        sig[bear & pull_short & confirm_short & short_reclaim] = -1
        return sig


class VWAPReversionScalp(BaseStrategy):
    name = "VWAPReversion"
    description = "VWAP extension + RSI/CMF fade"
    min_bars = 60

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        sig = pd.Series(0, index=df.index)
        ext = df["vwap_dist_atr"]
        long_setup = (ext < -1.2) & (df["rsi_7"] < 32) & (df["adx"] < 22) & (df["cmf"] > -0.03) & (df["vol_ratio"] > 0.9)
        short_setup = (ext > 1.2) & (df["rsi_7"] > 68) & (df["adx"] < 22) & (df["cmf"] < 0.03) & (df["vol_ratio"] > 0.9)
        long_trigger = (df["Close"] > df["Open"]) & (df["Close"].shift(1) < df["Open"].shift(1))
        short_trigger = (df["Close"] < df["Open"]) & (df["Close"].shift(1) > df["Open"].shift(1))
        sig[long_setup & long_trigger] = 1
        sig[short_setup & short_trigger] = -1
        return sig


class SqueezeBreakoutScalp(BaseStrategy):
    name = "SqueezeBreakout"
    description = "Low BB width / low ATR then volume breakout"
    min_bars = 100

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        sig = pd.Series(0, index=df.index)
        squeeze = df["bb_width"] < df["bb_width"].rolling(120, min_periods=60).quantile(0.18)
        atr_cool = df["atr_pct"] < df["atr_pct"].rolling(120, min_periods=60).quantile(0.30)
        vol_up = df["vol_ratio"] > 1.35
        up_break = (df["Close"] > df["bb_upper"]) & (df["Close"] > df["roll_high_10"].shift(1))
        dn_break = (df["Close"] < df["bb_lower"]) & (df["Close"] < df["roll_low_10"].shift(1))
        sig[squeeze.shift(1).fillna(False) & atr_cool & up_break & vol_up & (df["adx"] > 18) & (df["Close"] > df["vwap"])] = 1
        sig[squeeze.shift(1).fillna(False) & atr_cool & dn_break & vol_up & (df["adx"] > 18) & (df["Close"] < df["vwap"])] = -1
        return sig


class OrderflowImpulseScalp(BaseStrategy):
    name = "OrderflowImpulse"
    description = "OBV slope + CMF + volume impulse + EMA bias"
    min_bars = 80

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        sig = pd.Series(0, index=df.index)
        obv_up = df["obv"] > df["obv"].rolling(10).mean()
        obv_dn = df["obv"] < df["obv"].rolling(10).mean()
        ema_bias_long = df["ema_8"] > df["ema_20"]
        ema_bias_short = df["ema_8"] < df["ema_20"]
        long = obv_up & (df["cmf"] > 0.04) & (df["vol_ratio"] > 1.25) & ema_bias_long & (df["Close"] > df["vwap"])
        short = obv_dn & (df["cmf"] < -0.04) & (df["vol_ratio"] > 1.25) & ema_bias_short & (df["Close"] < df["vwap"])
        sig[long & (df["adx"] > 18) & (df["rsi_14"] > 52)] = 1
        sig[short & (df["adx"] > 18) & (df["rsi_14"] < 48)] = -1
        return sig


class SupertrendFlipScalp(BaseStrategy):
    name = "SupertrendFlip"
    description = "Supertrend flip + ADX confirmation"
    min_bars = 70

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        sig = pd.Series(0, index=df.index)
        up = (df["supertrend_dir"] == 1) & (df["supertrend_dir"].shift(1) == -1)
        dn = (df["supertrend_dir"] == -1) & (df["supertrend_dir"].shift(1) == 1)
        sig[up & (df["adx"] > 20) & (df["Close"] > df["ema_20"]) & (df["vol_ratio"] > 0.9)] = 1
        sig[dn & (df["adx"] > 20) & (df["Close"] < df["ema_20"]) & (df["vol_ratio"] > 0.9)] = -1
        return sig


class MomentumContinuationScalp(BaseStrategy):
    name = "MomentumContinuation"
    description = "ROC + MACD + RSI momentum follow-through"
    min_bars = 80

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        sig = pd.Series(0, index=df.index)
        long = (
            (df["roc_3"] > 0.05)
            & (df["macd_hist"] > df["macd_hist"].shift(1))
            & (df["rsi_14"] > 54)
            & (df["ema_5"] > df["ema_20"])
            & (df["vol_ratio"] > 1.1)
        )
        short = (
            (df["roc_3"] < -0.05)
            & (df["macd_hist"] < df["macd_hist"].shift(1))
            & (df["rsi_14"] < 46)
            & (df["ema_5"] < df["ema_20"])
            & (df["vol_ratio"] > 1.1)
        )
        sig[long & (df["adx"] > 18)] = 1
        sig[short & (df["adx"] > 18)] = -1
        return sig


ALL_STRATEGIES: List[BaseStrategy] = [
    TrendPullbackScalp(),
    VWAPReversionScalp(),
    SqueezeBreakoutScalp(),
    OrderflowImpulseScalp(),
    SupertrendFlipScalp(),
    MomentumContinuationScalp(),
]

STRATEGY_MAP = {s.name: s for s in ALL_STRATEGIES}


# ──────────────────────────────────────────────────────────────────────────────
# Risk management
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class OpenPosition:
    deal_id: str = ""
    direction: str = ""
    entry_price: float = 0.0
    entry_time: float = 0.0
    size: float = 0.0
    sl_price: float = 0.0
    tp1: float = 0.0
    tp2: float = 0.0
    tp3: float = 0.0
    current_sl: float = 0.0
    remaining_frac: float = 1.0
    tp1_hit: bool = False
    tp2_hit: bool = False
    trailing_active: bool = False
    trail_sl: float = 0.0
    bars_held: int = 0


class RiskManager:
    def __init__(self, cfg: TradingConfig):
        self.cfg = cfg
        self.position: Optional[OpenPosition] = None
        self.daily_pnl: float = 0.0
        self.daily_start: float = cfg.initial_capital
        self.last_loss_time: float = 0.0
        self.total_trades: int = 0
        self.total_wins: int = 0
        self.last_trade_time: float = 0.0
        self.consecutive_losses: int = 0

    def reset_day(self, balance: float) -> None:
        self.daily_pnl = 0.0
        self.daily_start = balance

    def calc_stop_distance(self, price: float, atr: float) -> float:
        atr = max(atr, price * 0.0001)
        return max(atr * self.cfg.sl_atr_mult, price * 0.0002)

    def calc_lot_size(self, price: float, atr: float, account_balance: float) -> float:
        if not self.cfg.use_dynamic_sizing:
            return round(self.cfg.lot_size, 2)
        stop_distance = self.calc_stop_distance(price, atr)
        risk_usd = account_balance * self.cfg.risk_pct
        lots = risk_usd / max(stop_distance, 1e-9)
        lots = clamp(lots, 0.01, 20.0)
        return round(lots, 2)

    def calc_levels(self, entry_price: float, direction: str, atr: float) -> Dict[str, float]:
        stop_distance = self.calc_stop_distance(entry_price, atr)
        if direction.upper() == "BUY":
            return {
                "sl": entry_price - stop_distance,
                "tp1": entry_price + atr * self.cfg.tp1_atr_mult,
                "tp2": entry_price + atr * self.cfg.tp2_atr_mult,
                "tp3": entry_price + atr * self.cfg.tp3_atr_mult,
            }
        return {
            "sl": entry_price + stop_distance,
            "tp1": entry_price - atr * self.cfg.tp1_atr_mult,
            "tp2": entry_price - atr * self.cfg.tp2_atr_mult,
            "tp3": entry_price - atr * self.cfg.tp3_atr_mult,
        }

    def open(self, deal_id: str, direction: str, entry_price: float, size: float, levels: Dict[str, float]) -> None:
        self.position = OpenPosition(
            deal_id=deal_id,
            direction=direction,
            entry_price=entry_price,
            entry_time=time.time(),
            size=size,
            sl_price=levels["sl"],
            tp1=levels["tp1"],
            tp2=levels["tp2"],
            tp3=levels["tp3"],
            current_sl=levels["sl"],
            remaining_frac=1.0,
        )

    def _is_long(self) -> bool:
        return self.position is not None and self.position.direction.upper() == "BUY"

    def can_enter(
        self,
        spread: float,
        price: Optional[float] = None,
        atr: Optional[float] = None,
        current_time: Optional[pd.Timestamp] = None,
    ) -> bool:
        if self.position is not None:
            return False
        if self.daily_pnl < -self.cfg.max_daily_loss_pct * self.daily_start:
            return False
        if self.consecutive_losses >= self.cfg.max_consecutive_losses:
            return False
        if time.time() - self.last_loss_time < self.cfg.cooldown_bars * 60:
            return False
        if spread > self.cfg.max_spread_abs:
            return False
        if atr is not None and price is not None:
            atr_pct = atr / max(price, 1e-9)
            if atr_pct < self.cfg.min_atr_pct or atr_pct > self.cfg.max_volatility_atr_pct:
                return False
            if spread > atr * self.cfg.max_spread_atr_frac:
                return False
        if current_time is not None:
            hour = current_time.hour if current_time.tzinfo is None else current_time.tz_convert(timezone.utc).hour
            if hour < self.cfg.session_start_utc or hour > self.cfg.session_end_utc:
                return False
        return True

    def _trail_update(self, current_price: float, high: float, low: float) -> None:
        pos = self.position
        if pos is None:
            return
        is_long = pos.direction.upper() == "BUY"
        if is_long:
            unrealized_rr = (high - pos.entry_price) / max(pos.entry_price - pos.sl_price, 1e-9)
        else:
            unrealized_rr = (pos.entry_price - low) / max(pos.sl_price - pos.entry_price, 1e-9)
        if unrealized_rr >= self.cfg.trailing_activate_rr and not pos.trailing_active:
            pos.trailing_active = True
            if is_long:
                pos.trail_sl = pos.entry_price + self.cfg.breakeven_buffer_atr * max(pos.entry_price - pos.sl_price, 1e-9)
            else:
                pos.trail_sl = pos.entry_price - self.cfg.breakeven_buffer_atr * max(pos.sl_price - pos.entry_price, 1e-9)
        if pos.trailing_active:
            if is_long:
                new_trail = high - self.cfg.trailing_step_atr * max(pos.entry_price - pos.sl_price, 1e-9)
                pos.trail_sl = max(pos.trail_sl, new_trail)
                pos.current_sl = max(pos.current_sl, pos.trail_sl)
            else:
                new_trail = low + self.cfg.trailing_step_atr * max(pos.sl_price - pos.entry_price, 1e-9)
                pos.trail_sl = min(pos.trail_sl, new_trail)
                pos.current_sl = min(pos.current_sl, pos.trail_sl)

    def check(self, current_price: float, high: float, low: float) -> Dict[str, Any]:
        if self.position is None:
            return {"action": "hold"}
        pos = self.position
        pos.bars_held += 1
        is_long = pos.direction.upper() == "BUY"

        self._trail_update(current_price, high, low)

        # Conservative candle sequencing:
        # long: assume low can hit SL before high hits TP on the same candle.
        # short: assume high can hit SL before low hits TP on the same candle.
        if is_long:
            if low <= pos.current_sl:
                return {"action": "full_close", "reason": "SL"}
            if not pos.tp1_hit and high >= pos.tp1:
                pos.tp1_hit = True
                pos.remaining_frac -= self.cfg.tp1_close
                pos.current_sl = max(pos.current_sl, pos.entry_price + self.cfg.breakeven_buffer_atr * (pos.tp1 - pos.entry_price))
                return {"action": "partial_close", "close_frac": self.cfg.tp1_close, "reason": "TP1"}
            if not pos.tp2_hit and high >= pos.tp2:
                pos.tp2_hit = True
                pos.remaining_frac -= self.cfg.tp2_close
                return {"action": "partial_close", "close_frac": self.cfg.tp2_close, "reason": "TP2"}
            if high >= pos.tp3:
                return {"action": "full_close", "reason": "TP3"}
        else:
            if high >= pos.current_sl:
                return {"action": "full_close", "reason": "SL"}
            if not pos.tp1_hit and low <= pos.tp1:
                pos.tp1_hit = True
                pos.remaining_frac -= self.cfg.tp1_close
                pos.current_sl = min(pos.current_sl, pos.entry_price - self.cfg.breakeven_buffer_atr * (pos.entry_price - pos.tp1))
                return {"action": "partial_close", "close_frac": self.cfg.tp1_close, "reason": "TP1"}
            if not pos.tp2_hit and low <= pos.tp2:
                pos.tp2_hit = True
                pos.remaining_frac -= self.cfg.tp2_close
                return {"action": "partial_close", "close_frac": self.cfg.tp2_close, "reason": "TP2"}
            if low <= pos.tp3:
                return {"action": "full_close", "reason": "TP3"}

        if pos.bars_held >= self.cfg.max_hold_bars:
            return {"action": "full_close", "reason": "TIME"}
        return {"action": "hold"}

    def record_close(self, pnl: float) -> None:
        self.total_trades += 1
        self.daily_pnl += pnl
        self.last_trade_time = time.time()
        if pnl > 0:
            self.total_wins += 1
            self.consecutive_losses = 0
        else:
            self.last_loss_time = time.time()
            self.consecutive_losses += 1
        self.position = None


# ──────────────────────────────────────────────────────────────────────────────
# ML layer
# ──────────────────────────────────────────────────────────────────────────────


FEATURE_COLS = [
    "rsi_7",
    "rsi_14",
    "stoch_k",
    "stoch_rsi_k",
    "cci",
    "willr",
    "mfi",
    "macd_hist",
    "adx",
    "adx_pos",
    "adx_neg",
    "bb_width",
    "atr_14",
    "atr_pct",
    "vol_ratio",
    "vol_z",
    "cmf",
    "roc_3",
    "roc_5",
    "stoch_cross",
    "ema_spread",
    "price_vs_vwap",
    "rsi_divergence",
    "vwap_dist_atr",
]


class MLEngine:
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.model = None
        self.scaler = None
        self.trained = False
        self.sample_buffer: Deque[Tuple[List[float], int]] = deque(maxlen=4000)

    def extract_features(self, df: pd.DataFrame, idx: int) -> Optional[List[float]]:
        if idx < 0 or idx >= len(df):
            return None
        row = df.iloc[idx]
        if any(pd.isna(row.get(c)) for c in FEATURE_COLS):
            return None
        return [safe_float(row[c]) for c in FEATURE_COLS]

    def add_sample(self, features: Optional[List[float]], won: bool) -> None:
        if features is not None:
            self.sample_buffer.append((features, 1 if won else 0))

    def train(self) -> bool:
        if not self.enabled:
            return False
        if len(self.sample_buffer) < 50:
            return False
        try:
            from sklearn.ensemble import GradientBoostingClassifier
            from sklearn.metrics import accuracy_score
            from sklearn.preprocessing import StandardScaler
        except Exception as exc:  # pragma: no cover
            log.warning("Skipping ML training: %s", exc)
            return False
        try:
            X = np.array([x for x, _ in self.sample_buffer], dtype=float)
            y = np.array([y for _, y in self.sample_buffer], dtype=int)
            self.scaler = StandardScaler()
            Xs = self.scaler.fit_transform(X)
            self.model = GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=3,
                subsample=0.8,
                random_state=42,
            )
            self.model.fit(Xs, y)
            self.trained = True
            pred = self.model.predict(Xs)
            acc = accuracy_score(y, pred)
            log.info("ML trained on %d samples | acc=%.1f%%", len(y), acc * 100)
            return True
        except Exception as exc:
            log.warning("ML training failed: %s", exc)
            return False

    def predict(self, df: pd.DataFrame, idx: int) -> float:
        if not self.trained or self.model is None or self.scaler is None:
            return 0.5
        feat = self.extract_features(df, idx)
        if feat is None:
            return 0.5
        try:
            X = self.scaler.transform([feat])
            proba = self.model.predict_proba(X)[0]
            return float(proba[1]) if len(proba) > 1 else float(proba[0])
        except Exception:
            return 0.5

    def should_trade(self, df: pd.DataFrame, idx: int, signal: int, threshold: float = 0.55) -> bool:
        if not self.trained:
            return True
        win_prob = self.predict(df, idx)
        if signal == 1:
            return win_prob >= threshold
        if signal == -1:
            return win_prob <= (1 - threshold)
        return False


# ──────────────────────────────────────────────────────────────────────────────
# Adaptive strategy ranking
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class StrategyScore:
    name: str
    trades: int = 0
    wins: int = 0
    total_pnl: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    sharpe: float = 0.0
    sortino: float = 0.0
    expectancy: float = 0.0
    max_drawdown: float = 0.0
    composite: float = 0.0
    recent_pnls: Deque[float] = field(default_factory=lambda: deque(maxlen=100))


class AdaptiveManager:
    def __init__(self, strategy_names: List[str]):
        self.scores: Dict[str, StrategyScore] = {name: StrategyScore(name=name) for name in strategy_names}
        self.active = strategy_names[0] if strategy_names else ""
        self.switches = 0

    def record_trade(self, strategy_name: str, pnl: float) -> None:
        sc = self.scores.setdefault(strategy_name, StrategyScore(name=strategy_name))
        sc.trades += 1
        sc.total_pnl += pnl
        sc.recent_pnls.append(pnl)
        if pnl > 0:
            sc.wins += 1
        sc.win_rate = sc.wins / max(sc.trades, 1)
        pnls = np.array(list(sc.recent_pnls), dtype=float)
        if len(pnls) >= 3:
            wins_sum = pnls[pnls > 0].sum()
            loss_sum = abs(pnls[pnls < 0].sum())
            sc.profit_factor = wins_sum / loss_sum if loss_sum > 0 else wins_sum if wins_sum > 0 else 0.0
            mean_r = pnls.mean()
            std_r = pnls.std(ddof=0)
            sc.sharpe = (mean_r / std_r) * np.sqrt(len(pnls)) if std_r > 0 else 0.0
            neg = pnls[pnls < 0]
            ds = neg.std(ddof=0) if len(neg) > 1 else 0.0
            sc.sortino = (mean_r / ds) * np.sqrt(len(pnls)) if ds > 0 else sc.sharpe
            sc.expectancy = pnls.mean()
        equity = np.cumsum(pnls) if len(pnls) else np.array([0.0])
        peak = np.maximum.accumulate(equity)
        dd = equity - peak
        sc.max_drawdown = float(dd.min()) if len(dd) else 0.0
        sh = clamp(sc.sharpe / 3.0, -1.0, 1.0)
        so = clamp(sc.sortino / 3.0, -1.0, 1.0)
        pf = clamp(sc.profit_factor / 4.0, 0.0, 1.0)
        wr = clamp(sc.win_rate, 0.0, 1.0)
        act = clamp(sc.trades / 20.0, 0.0, 1.0)
        dd_penalty = clamp(abs(sc.max_drawdown) / max(abs(sc.total_pnl) + 1e-9, 1.0), 0.0, 1.0)
        sc.composite = 0.30 * sh + 0.20 * so + 0.20 * wr + 0.15 * pf + 0.10 * act - 0.15 * dd_penalty

    def get_best(self) -> str:
        ranked = sorted(self.scores.values(), key=lambda s: s.composite, reverse=True)
        if not ranked:
            return self.active
        best = ranked[0].name
        if best != self.active:
            log.info("Strategy switch: %s -> %s", self.active, best)
            self.active = best
            self.switches += 1
        return best

    def get_table(self) -> str:
        ranked = sorted(self.scores.values(), key=lambda s: s.composite, reverse=True)
        lines = [
            f"{'Strategy':<22s} {'Trades':>6s} {'Win%':>6s} {'PF':>6s} {'Sharpe':>7s} {'Score':>7s} {'PnL':>12s}",
            "-" * 82,
        ]
        for sc in ranked:
            lines.append(
                f"{sc.name:<22s} {sc.trades:>6d} {sc.win_rate:>5.0%} {sc.profit_factor:>6.2f} "
                f"{sc.sharpe:>7.2f} {sc.composite:>7.3f} ${sc.total_pnl:>11,.2f}"
            )
        return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# Backtesting
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class TradeRecord:
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    direction: str
    strategy: str
    entry_price: float
    exit_price: float
    size: float
    pnl_usd: float
    pnl_r: float
    reason: str
    bars_held: int


@dataclass
class BacktestResult:
    strategy_name: str
    trades: List[TradeRecord]
    equity_curve: pd.Series
    metrics: Dict[str, float]


def load_yfinance_data(symbol: str, timeframe: str, days: int = 60) -> pd.DataFrame:
    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover
        raise ImportError("yfinance is required for backtesting data fetch") from exc
    interval = timeframe_to_yf_interval(timeframe)
    period = yfinance_period_for_interval(interval, days)
    ticker = "GC=F" if symbol.upper().startswith("XAU") else symbol
    log.info("Downloading %s for backtest: interval=%s period=%s", ticker, interval, period)
    df = yf.download(ticker, interval=interval, period=period, progress=False, auto_adjust=False)
    if df is None or len(df) == 0:
        raise ValueError("No data returned from yfinance")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns={c: c.capitalize() for c in df.columns})
    if "Adj Close" in df.columns:
        df = df.drop(columns=["Adj Close"])
    if "Volume" not in df.columns:
        df["Volume"] = 0.0
    df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
    df = df.dropna(subset=["Open", "High", "Low", "Close"]).sort_index()
    return df[["Open", "High", "Low", "Close", "Volume"]]


class Backtester:
    def __init__(self, trading: TradingConfig, backtest: BacktestConfig):
        self.trading = trading
        self.backtest = backtest

    def _estimate_spread(self, atr: float, price: float) -> float:
        if not self.backtest.use_estimated_spread:
            return 0.0
        return max(price * 0.00002, atr * self.backtest.estimated_spread_atr_frac)

    def _slippage(self, atr: float) -> float:
        return atr * self.backtest.slippage_atr_frac

    def _bar_is_tradeable(self, bar: pd.Series, atr: float, price: float) -> bool:
        if price <= 0 or atr <= 0:
            return False
        atr_pct = atr / price
        if atr_pct < self.trading.min_atr_pct or atr_pct > self.trading.max_volatility_atr_pct:
            return False
        spread = self._estimate_spread(atr, price)
        if spread > self.trading.max_spread_abs:
            return False
        if spread > atr * self.trading.max_spread_atr_frac:
            return False
        ts = bar.name if isinstance(bar.name, pd.Timestamp) else None
        if isinstance(ts, pd.Timestamp):
            hour = ts.hour if ts.tzinfo is None else ts.tz_convert(timezone.utc).hour
            if hour < self.trading.session_start_utc or hour > self.trading.session_end_utc:
                return False
        return True

    def _trade_pnl(
        self,
        direction: str,
        entry: float,
        exit_price: float,
        size: float,
        portion: float,
        spread: float,
        commission_roundtrip_per_lot: float,
    ) -> float:
        gross = (exit_price - entry) * size * portion if direction == "BUY" else (entry - exit_price) * size * portion
        cost = commission_roundtrip_per_lot * size * portion
        return gross - cost

    def _simulate_trade(
        self,
        df: pd.DataFrame,
        strategy_name: str,
        signal_idx: int,
        signal: int,
        equity: float,
        day_trade_count: Dict[pd.Timestamp, int],
    ) -> Optional[TradeRecord]:
        entry_idx = signal_idx + 1
        if entry_idx >= len(df):
            return None
        entry_bar = df.iloc[entry_idx]
        signal_bar = df.iloc[signal_idx]
        atr = safe_float(signal_bar.get("atr_14"), 0.0)
        price_ref = safe_float(entry_bar["Open"])
        if atr <= 0 or price_ref <= 0:
            return None
        if not self._bar_is_tradeable(entry_bar, atr, price_ref):
            return None
        spread = self._estimate_spread(atr, price_ref)
        slippage = self._slippage(atr)
        direction = "BUY" if signal == 1 else "SELL"
        entry = price_ref + spread / 2 + slippage if direction == "BUY" else price_ref - spread / 2 - slippage
        risk_usd = equity * self.trading.risk_pct
        stop_dist = max(atr * self.trading.sl_atr_mult, price_ref * 0.0002)
        size = clamp(risk_usd / max(stop_dist, 1e-9), 0.01, 20.0)
        levels = RiskManager(self.trading).calc_levels(entry, direction, atr)
        sl = levels["sl"]
        tp1 = levels["tp1"]
        tp2 = levels["tp2"]
        tp3 = levels["tp3"]
        remaining = 1.0
        pnl_usd = 0.0
        pnl_r = 0.0
        tp1_hit = False
        tp2_hit = False
        bars_held = 0
        reason = "TIME"
        entry_date = extract_trade_date(df.index[entry_idx])
        day_trade_count.setdefault(entry_date, 0)
        if day_trade_count[entry_date] >= self.backtest.max_trades_per_day:
            return None

        for j in range(entry_idx, len(df)):
            bar = df.iloc[j]
            high = safe_float(bar["High"])
            low = safe_float(bar["Low"])
            close = safe_float(bar["Close"])
            bars_held += 1
            if direction == "BUY":
                # conservative order: stop before targets on the same bar
                if low <= sl:
                    exit_price = sl - spread / 2
                    pnl_usd += self._trade_pnl(direction, entry, exit_price, size, remaining, spread, self.backtest.commission_roundtrip_per_lot)
                    reason = "SL"
                    break
                if not tp1_hit and high >= tp1:
                    portion = self.trading.tp1_close
                    exit_price = tp1 - spread / 2
                    pnl_usd += self._trade_pnl(direction, entry, exit_price, size, portion, spread, self.backtest.commission_roundtrip_per_lot)
                    remaining -= portion
                    tp1_hit = True
                    sl = max(sl, entry + self.trading.breakeven_buffer_atr * atr)
                if not tp2_hit and high >= tp2:
                    portion = self.trading.tp2_close
                    exit_price = tp2 - spread / 2
                    pnl_usd += self._trade_pnl(direction, entry, exit_price, size, portion, spread, self.backtest.commission_roundtrip_per_lot)
                    remaining -= portion
                    tp2_hit = True
                if high >= tp3:
                    exit_price = tp3 - spread / 2
                    pnl_usd += self._trade_pnl(direction, entry, exit_price, size, remaining, spread, self.backtest.commission_roundtrip_per_lot)
                    reason = "TP3"
                    break
            else:
                if high >= sl:
                    exit_price = sl + spread / 2
                    pnl_usd += self._trade_pnl(direction, entry, exit_price, size, remaining, spread, self.backtest.commission_roundtrip_per_lot)
                    reason = "SL"
                    break
                if not tp1_hit and low <= tp1:
                    portion = self.trading.tp1_close
                    exit_price = tp1 + spread / 2
                    pnl_usd += self._trade_pnl(direction, entry, exit_price, size, portion, spread, self.backtest.commission_roundtrip_per_lot)
                    remaining -= portion
                    tp1_hit = True
                    sl = min(sl, entry - self.trading.breakeven_buffer_atr * atr)
                if not tp2_hit and low <= tp2:
                    portion = self.trading.tp2_close
                    exit_price = tp2 + spread / 2
                    pnl_usd += self._trade_pnl(direction, entry, exit_price, size, portion, spread, self.backtest.commission_roundtrip_per_lot)
                    remaining -= portion
                    tp2_hit = True
                if low <= tp3:
                    exit_price = tp3 + spread / 2
                    pnl_usd += self._trade_pnl(direction, entry, exit_price, size, remaining, spread, self.backtest.commission_roundtrip_per_lot)
                    reason = "TP3"
                    break

            if bars_held >= self.trading.max_hold_bars:
                exit_price = close - spread / 2 if direction == "BUY" else close + spread / 2
                pnl_usd += self._trade_pnl(direction, entry, exit_price, size, remaining, spread, self.backtest.commission_roundtrip_per_lot)
                reason = "TIME"
                break

        risk_per_lot = max(stop_dist, 1e-9)
        pnl_r = pnl_usd / max(risk_per_lot * size, 1e-9)
        exit_time = df.index[min(j, len(df) - 1)]
        day_trade_count[entry_date] += 1
        return TradeRecord(
            entry_time=df.index[entry_idx],
            exit_time=exit_time,
            direction=direction,
            strategy=strategy_name,
            entry_price=entry,
            exit_price=exit_price,
            size=size,
            pnl_usd=pnl_usd,
            pnl_r=pnl_r,
            reason=reason,
            bars_held=bars_held,
        )

    def run_strategy(self, df: pd.DataFrame, strategy: BaseStrategy, initial_equity: Optional[float] = None) -> BacktestResult:
        if initial_equity is None:
            initial_equity = self.backtest.starting_equity
        signals = strategy.generate_signals(df).fillna(0).astype(int)
        equity = float(initial_equity)
        equity_points = []
        trades: List[TradeRecord] = []
        day_trade_count: Dict[pd.Timestamp, int] = {}
        for i in range(max(self.backtest.warmup_bars, strategy.min_bars), len(df) - 1):
            sig = int(signals.iloc[i])
            if sig == 0:
                equity_points.append(equity)
                continue
            trade = self._simulate_trade(df, strategy.name, i, sig, equity, day_trade_count)
            if trade is None:
                equity_points.append(equity)
                continue
            trades.append(trade)
            equity += trade.pnl_usd
            equity_points.append(equity)
        if len(equity_points) < len(df):
            equity_points.extend([equity] * (len(df) - len(equity_points)))
        equity_curve = pd.Series(equity_points[: len(df)], index=df.index)
        metrics = self._metrics(trades, equity_curve, initial_equity)
        return BacktestResult(strategy.name, trades, equity_curve, metrics)

    def _metrics(self, trades: List[TradeRecord], equity_curve: pd.Series, initial_equity: float) -> Dict[str, float]:
        if not trades:
            return {
                "trades": 0.0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "net_pnl": 0.0,
                "return_pct": 0.0,
                "max_drawdown_pct": 0.0,
                "sharpe": 0.0,
                "sortino": 0.0,
                "expectancy": 0.0,
                "avg_hold_bars": 0.0,
                "score": -999.0,
            }
        pnls = np.array([t.pnl_usd for t in trades], dtype=float)
        wins = pnls[pnls > 0]
        losses = pnls[pnls < 0]
        net = pnls.sum()
        win_rate = (pnls > 0).mean()
        profit_factor = wins.sum() / abs(losses.sum()) if len(losses) else float(wins.sum()) if len(wins) else 0.0
        ret = equity_curve.pct_change().dropna()
        sharpe = (ret.mean() / ret.std(ddof=0)) * np.sqrt(len(ret)) if len(ret) > 2 and ret.std(ddof=0) > 0 else 0.0
        downside = ret[ret < 0]
        sortino = (ret.mean() / downside.std(ddof=0)) * np.sqrt(len(ret)) if len(downside) > 2 and downside.std(ddof=0) > 0 else sharpe
        peak = equity_curve.cummax()
        drawdown = (equity_curve - peak) / peak
        max_dd = float(drawdown.min()) if len(drawdown) else 0.0
        avg_hold = float(np.mean([t.bars_held for t in trades]))
        expectancy = float(pnls.mean())
        score = (
            0.30 * clamp(sharpe / 3.0, -1.0, 1.0)
            + 0.20 * clamp(sortino / 3.0, -1.0, 1.0)
            + 0.20 * clamp(win_rate, 0.0, 1.0)
            + 0.15 * clamp(profit_factor / 4.0, 0.0, 1.0)
            + 0.10 * clamp((net / initial_equity) * 10.0, -1.0, 1.0)
            + 0.05 * clamp(len(trades) / 50.0, 0.0, 1.0)
            - 0.20 * clamp(abs(max_dd) / 0.10, 0.0, 1.0)
        )
        return {
            "trades": float(len(trades)),
            "win_rate": float(win_rate),
            "profit_factor": float(profit_factor),
            "net_pnl": float(net),
            "return_pct": float((equity_curve.iloc[-1] / initial_equity - 1) * 100),
            "max_drawdown_pct": float(max_dd * 100),
            "sharpe": float(sharpe),
            "sortino": float(sortino),
            "expectancy": float(expectancy),
            "avg_hold_bars": float(avg_hold),
            "score": float(score),
        }

    def walk_forward_rank(self, df: pd.DataFrame, strategies: Sequence[BaseStrategy]) -> pd.DataFrame:
        n = len(df)
        train_end = int(n * self.backtest.walk_forward_train_frac)
        test_end = int(n * (self.backtest.walk_forward_train_frac + self.backtest.walk_forward_test_frac))
        train = df.iloc[:train_end].copy()
        test = df.iloc[train_end:test_end].copy() if test_end > train_end else df.iloc[train_end:].copy()
        if len(train) < 300:
            train = df.copy()
            test = df.copy()
        rows = []
        for strat in strategies:
            train_res = self.run_strategy(train, strat, initial_equity=self.backtest.starting_equity)
            test_res = self.run_strategy(test, strat, initial_equity=self.backtest.starting_equity) if len(test) > 0 else train_res
            row = {
                "strategy": strat.name,
                "train_score": train_res.metrics["score"],
                "test_score": test_res.metrics["score"],
                "trades": int(test_res.metrics["trades"]),
                "win_rate": test_res.metrics["win_rate"],
                "pf": test_res.metrics["profit_factor"],
                "return_pct": test_res.metrics["return_pct"],
                "max_dd_pct": test_res.metrics["max_drawdown_pct"],
                "sharpe": test_res.metrics["sharpe"],
                "sortino": test_res.metrics["sortino"],
                "expectancy": test_res.metrics["expectancy"],
                "avg_hold": test_res.metrics["avg_hold_bars"],
            }
            rows.append(row)
        table = pd.DataFrame(rows).sort_values(["test_score", "train_score"], ascending=False).reset_index(drop=True)
        return table


# ──────────────────────────────────────────────────────────────────────────────
# Bot
# ──────────────────────────────────────────────────────────────────────────────


TF_SECONDS = {
    "1m": 60,
    "2m": 120,
    "5m": 300,
    "15m": 900,
    "30m": 1800,
    "1H": 3600,
    "4H": 14400,
    "1D": 86400,
}


def _check_env_or_die() -> None:
    """Validate that required OANDA credentials are present."""
    missing = []
    api_token = os.getenv("OANDA_API_TOKEN", "")
    account_id = os.getenv("OANDA_ACCOUNT_ID", "")
    if not api_token:
        missing.append("OANDA_API_TOKEN")
    if not account_id:
        missing.append("OANDA_ACCOUNT_ID")
    if missing:
        log.critical("=" * 60)
        log.critical("MISSING ENVIRONMENT VARIABLES: %s", ", ".join(missing))
        log.critical("Please add them in Railway Dashboard:")
        log.critical("  Project -> Your Service -> Variables")
        log.critical("=" * 60)
        sys.exit(1)
    log.info("ENV OK: OANDA_API_TOKEN is set (length=%d)", len(api_token))
    log.info("ENV OK: OANDA_ACCOUNT_ID is set (%s)", account_id)


class MicroScalpBot:
    def _format_instrument(self, symbol: str) -> str:
        if "_" in symbol:
            return symbol.upper()
        if symbol.upper().startswith("XAU"):
            return "XAU_USD"
        if len(symbol) == 6:
            return symbol[:3].upper() + "_" + symbol[3:].upper()
        return symbol.upper()

    def __init__(self):
        self.client = OandaClient(
            api_token=OANDA.api_token,
            account_id=OANDA.account_id,
            demo=OANDA.demo,
        )
        self.instrument = ""
        self.risk = RiskManager(TRADING)
        self.ml = MLEngine(enabled=TRADING.ml_enabled)
        self.adaptive = AdaptiveManager([s.name for s in ALL_STRATEGIES])
        self.strategy_names = [s.name for s in ALL_STRATEGIES]
        self.active_strategy_idx = 0
        self.active_strategy_score = -999.0
        self.state_file = HERE / SERVER.state_file
        self.last_candle_time: Optional[pd.Timestamp] = None
        self.bars_since_last_trade = 999
        self.trade_count = 0
        self.last_strategy_refresh = 0.0

    def _active_name(self) -> str:
        return self.strategy_names[self.active_strategy_idx]

    def start(self) -> None:
        log.info("=" * 72)
        log.info("Gold Micro Scalper")
        log.info("=" * 72)
        if not CAPITAL.api_key or not CAPITAL.login or not CAPITAL.password:
            log.warning("Capital.com credentials are incomplete. Live mode will fail until .env is filled.")
        account = self.client.get_account_info()
        if account:
            log.info("Account balance: $%.2f", account.get("balance", 0.0))
            log.info("Available: $%.2f", account.get("available", 0.0))
        self.instrument = self._format_instrument(TRADING.symbol)
        log.info("Trading %s (Instrument: %s)", TRADING.symbol, self.instrument)
        log.info("Timeframe: %s | leverage: %.0fx | risk: %.2f%%", TRADING.timeframe, TRADING.leverage, TRADING.risk_pct * 100)
        self._load_state()
        if TRADING.ml_enabled:
            self._initial_ml_train()

    def _initial_ml_train(self) -> None:
        try:
            df = self.fetch_candles(max(400, TRADING.best_strategy_lookback))
            df = compute_indicators(df).dropna()
            if len(df) < TRADING.ml_min_samples:
                return
            for strat in ALL_STRATEGIES:
                signals = strat.generate_signals(df).fillna(0).astype(int)
                for i in range(10, len(df) - 5):
                    sig = int(signals.iloc[i])
                    if sig == 0:
                        continue
                    future = df["Close"].iloc[i + 1 : min(i + 6, len(df))]
                    if len(future) == 0:
                        continue
                    entry = df["Close"].iloc[i]
                    if sig == 1:
                        won = future.max() > entry
                    else:
                        won = future.min() < entry
                    feats = self.ml.extract_features(df, i)
                    self.ml.add_sample(feats, won)
            self.ml.train()
        except Exception as exc:
            log.info("Initial ML warmup skipped: %s", exc)

    def _load_state(self) -> None:
        if not self.state_file.exists():
            return
        try:
            data = json.loads(self.state_file.read_text(encoding="utf-8"))
            self.active_strategy_idx = int(data.get("active_strategy_idx", 0))
            self.trade_count = int(data.get("trade_count", 0))
        except Exception:
            pass

    def _save_state(self) -> None:
        payload = {
            "active_strategy_idx": self.active_strategy_idx,
            "trade_count": self.trade_count,
            "saved_at": datetime.utcnow().isoformat(),
        }
        try:
            self.state_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except Exception:
            pass

    def fetch_candles(self, count: int = 500) -> pd.DataFrame:
        raw = self.client.get_candles(self.instrument, TRADING.timeframe, count)
        if not raw:
            raise ValueError("No candle data received")
        df = pd.DataFrame(raw)
        df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
        df = df.set_index("time").sort_index()
        return ensure_ohlcv(df)

    def get_current_price(self) -> Dict[str, float]:
        return self.client.get_market_price(self.instrument)

    def _refresh_strategy(self, df: pd.DataFrame) -> None:
        if time.time() - self.last_strategy_refresh < 300:
            return
        self.last_strategy_refresh = time.time()
        bt = Backtester(TRADING, BACKTEST)
        ranking = bt.walk_forward_rank(df.iloc[-min(len(df), TRADING.best_strategy_lookback):].copy(), ALL_STRATEGIES)
        if len(ranking) == 0:
            return
        best_name = str(ranking.iloc[0]["strategy"])
        self.active_strategy_score = safe_float(ranking.iloc[0].get("test_score"), -999.0)
        if best_name in self.strategy_names:
            self.active_strategy_idx = self.strategy_names.index(best_name)
            log.info("Selected best strategy: %s", best_name)
            log.info("\n%s", ranking.head(6).to_string(index=False))

    def get_signal(self, df: pd.DataFrame) -> int:
        strat = ALL_STRATEGIES[self.active_strategy_idx]
        sig = strat.generate_signals(df)
        return int(sig.iloc[-1]) if len(sig) else 0

    def _estimate_live_pnl(self, pos: OpenPosition, exit_price: float, close_frac: float) -> float:
        if pos.direction.upper() == "BUY":
            return (exit_price - pos.entry_price) * pos.size * close_frac
        return (pos.entry_price - exit_price) * pos.size * close_frac

    def _manage_position(self, current_price: float, df: pd.DataFrame) -> None:
        pos = self.risk.position
        if pos is None:
            return
        high = safe_float(df["High"].iloc[-1])
        low = safe_float(df["Low"].iloc[-1])
        action = self.risk.check(current_price, high, low)
        if action["action"] == "hold":
            if pos.current_sl != pos.sl_price and pos.deal_id:
                self.client.modify_position(pos.deal_id, stop_level=round(pos.current_sl, 2))
                pos.sl_price = pos.current_sl
            return
        if action["action"] == "partial_close":
            close_frac = float(action.get("close_frac", 0.0))
            if pos.deal_id:
                result = self.client.close_position(pos.deal_id)
                if result.success:
                    remaining_size = max(pos.size * pos.remaining_frac, 0.0)
                    if remaining_size > 0.01:
                        new_result = self.client.open_position(
                            epic=self.epic,
                            direction=pos.direction,
                            size=remaining_size,
                            stop_level=round(pos.current_sl, 2),
                        )
                        if new_result.success:
                            pos.deal_id = new_result.deal_id
                            pos.size = remaining_size
                    pnl = self._estimate_live_pnl(pos, current_price, close_frac)
                    self.risk.record_close(pnl)
                    feats = self.ml.extract_features(df, len(df) - 1)
                    self.ml.add_sample(feats, pnl > 0)
                    self.adaptive.record_trade(self._active_name(), pnl)
                    log.info("Partial close (%s): pnl=$%.2f", action.get("reason", ""), pnl)
            return
        if action["action"] == "full_close" and pos.deal_id:
            result = self.client.close_position(pos.deal_id)
            if result.success:
                pnl = self._estimate_live_pnl(pos, current_price, 1.0)
                self.risk.record_close(pnl)
                feats = self.ml.extract_features(df, len(df) - 1)
                self.ml.add_sample(feats, pnl > 0)
                self.adaptive.record_trade(self._active_name(), pnl)
                log.info("Closed (%s): pnl=$%.2f", action.get("reason", ""), pnl)

    def run_once(self) -> None:
        df = self.fetch_candles(600)
        df = compute_indicators(df).dropna()
        if len(df) < 100:
            return
        current_bar_time = df.index[-1]
        if self.last_candle_time is not None and current_bar_time == self.last_candle_time:
            return
        self.last_candle_time = current_bar_time
        self.bars_since_last_trade += 1
        self._refresh_strategy(df)
        price_info = self.get_current_price()
        bid, ask, spread = price_info["bid"], price_info["ask"], price_info["spread"]
        atr = safe_float(df["atr_14"].iloc[-1])
        if self.risk.position is not None:
            self._manage_position(bid, df)
            return
        if self.active_strategy_score < TRADING.min_live_strategy_score:
            log.info(
                "Skipping trade: best strategy score %.3f below threshold %.3f",
                self.active_strategy_score,
                TRADING.min_live_strategy_score,
            )
            return
        if not self.risk.can_enter(spread, price=bid, atr=atr, current_time=current_bar_time):
            return
        signal = self.get_signal(df)
        if signal == 0:
            return
        if TRADING.ml_enabled and self.ml.trained and not self.ml.should_trade(df, len(df) - 1, signal, TRADING.ml_threshold):
            log.info("ML rejected signal %s from %s", signal, self._active_name())
            return
        account = self.client.get_account_info()
        balance = account.get("balance", TRADING.initial_capital)
        entry = ask if signal == 1 else bid
        direction = "BUY" if signal == 1 else "SELL"
        size = self.risk.calc_lot_size(entry, atr, balance)
        levels = self.risk.calc_levels(entry, direction, atr)
        result = self.client.open_position(
            instrument=self.instrument,
            direction=direction,
            size=size,
            stop_level=round(levels["sl"], 2),
            limit_level=round(levels["tp1"], 2),
        )
        if result.success:
            self.risk.open(result.deal_id, direction, entry, size, levels)
            self.trade_count += 1
            self.bars_since_last_trade = 0
            log.info("TRADE #%d | %s | %s | size=%.2f | entry=%.2f | strat=%s", self.trade_count, direction, self.epic, size, entry, self._active_name())
            self._save_state()
        else:
            log.warning("Order failed: %s", result.error)

    def run_dry_run(self) -> None:
        df = self.fetch_candles(600)
        df = compute_indicators(df).dropna()
        self._refresh_strategy(df)
        signal = self.get_signal(df)
        price = self.get_current_price()
        sig_text = {1: "BUY", -1: "SELL", 0: "HOLD"}[signal]
        log.info(
            "[%s] Signal=%s bid=%.2f ask=%.2f spread=%.2f atr=%.2f",
            self._active_name(),
            sig_text,
            price["bid"],
            price["ask"],
            price["spread"],
            safe_float(df["atr_14"].iloc[-1]),
        )


# ──────────────────────────────────────────────────────────────────────────────
# Backtest runner and report
# ──────────────────────────────────────────────────────────────────────────────


def print_backtest_report(ranking: pd.DataFrame, detailed: Dict[str, BacktestResult]) -> None:
    print("\n" + "=" * 104)
    print("BACKTEST REPORT")
    print("=" * 104)
    if len(ranking):
        cols = ["strategy", "trades", "win_rate", "pf", "return_pct", "max_dd_pct", "sharpe", "sortino", "expectancy", "avg_hold", "test_score"]
        view = ranking.copy()
        for col in ["win_rate", "return_pct", "max_dd_pct", "expectancy", "avg_hold", "sharpe", "sortino", "test_score", "pf"]:
            if col in view.columns:
                pass
        print(view.to_string(index=False, justify="left", max_rows=12))
        best = view.iloc[0]
        print("-" * 104)
        print(f"Best strategy: {best['strategy']}")
        print(f"Return: {best['return_pct']:.2f}% | Win rate: {best['win_rate']:.1%} | PF: {best['pf']:.2f} | Max DD: {best['max_dd_pct']:.2f}%")
    else:
        print("No ranking data.")
    print("=" * 104)
    for name, res in detailed.items():
        if not res.trades:
            continue
        m = res.metrics
        print(
            f"{name:<22s} trades={int(m['trades']):>4d} "
            f"win={m['win_rate']:.1%} pf={m['profit_factor']:.2f} "
            f"ret={m['return_pct']:.2f}% dd={m['max_drawdown_pct']:.2f}% "
            f"sharpe={m['sharpe']:.2f} score={m['score']:.3f}"
        )


def run_backtest(days: int = 60, symbol: Optional[str] = None, timeframe: Optional[str] = None) -> None:
    symbol = symbol or TRADING.symbol
    timeframe = timeframe or TRADING.timeframe
    df = load_yfinance_data(symbol, timeframe, days=days)
    df = compute_indicators(df).dropna()
    if len(df) < 300:
        raise ValueError("Not enough data after indicators")
    bt = Backtester(TRADING, BACKTEST)
    ranking = bt.walk_forward_rank(df, ALL_STRATEGIES)
    detailed: Dict[str, BacktestResult] = {}
    for strat in ALL_STRATEGIES:
        detailed[strat.name] = bt.run_strategy(df, strat, initial_equity=BACKTEST.starting_equity)
    print_backtest_report(ranking, detailed)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────


_shutdown = False


def signal_handler(sig, frame):  # type: ignore[override]
    global _shutdown
    log.info("Shutdown signal received.")
    _shutdown = True


def run_live() -> None:
    bot = MicroScalpBot()
    bot.start()
    log.info("Polling every %ss", SERVER.poll_interval)
    while not _shutdown:
        try:
            bot.run_once()
        except Exception as exc:
            log.exception("Live loop error: %s", exc)
            time.sleep(20)
        time.sleep(SERVER.poll_interval)
    log.info("Bot stopped.")


def run_dry() -> None:
    bot = MicroScalpBot()
    bot.start()
    while not _shutdown:
        try:
            bot.run_dry_run()
        except Exception as exc:
            log.exception("Dry-run error: %s", exc)
        time.sleep(SERVER.poll_interval)


def main() -> None:
    parser = argparse.ArgumentParser(description="Gold Micro Scalper - single file bot")
    parser.add_argument("--backtest", action="store_true", help="Run backtest only")
    parser.add_argument("--dry-run", action="store_true", help="Paper-trading signals only")
    parser.add_argument("--days", type=int, default=60, help="Backtest lookback days for yfinance")
    parser.add_argument("--symbol", type=str, default=TRADING.symbol, help="Trading symbol")
    parser.add_argument("--timeframe", type=str, default=TRADING.timeframe, help="Trading timeframe")
    args = parser.parse_args()

    setup_logging()
    _check_env_or_die()
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    if args.backtest:
        run_backtest(days=args.days, symbol=args.symbol, timeframe=args.timeframe)
    elif args.dry_run:
        run_dry()
    else:
        run_live()


if __name__ == "__main__":
    main()
