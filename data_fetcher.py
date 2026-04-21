"""
데이터 수집 모듈
- 실서버: pyupbit API 사용
- 테스트/오프라인: 시뮬레이션 데이터 자동 생성
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta


# ── 시뮬레이션 데이터 생성 (테스트용) ─────────────────────

def _make_ohlcv(
    n: int,
    interval_minutes: int,
    start_price: float = 50_000_000,
    volatility: float = 0.003,
    seed: int = 42,
) -> pd.DataFrame:
    """
    GBM(기하 브라운 운동) 기반 OHLCV 시뮬레이션
    """
    rng = np.random.default_rng(seed)
    end = datetime.now().replace(second=0, microsecond=0)
    start = end - timedelta(minutes=interval_minutes * n)
    dates = pd.date_range(start=start, periods=n, freq=f"{interval_minutes}min")

    returns = rng.normal(0.0002, volatility, n)
    close = start_price * np.cumprod(1 + returns)

    high = close * (1 + rng.uniform(0, volatility * 1.5, n))
    low = close * (1 - rng.uniform(0, volatility * 1.5, n))
    open_ = np.roll(close, 1)
    open_[0] = start_price
    volume = rng.uniform(0.5, 5.0, n) * 1e6 / close

    df = pd.DataFrame({
        "date": dates,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })
    return df


def _make_daily(n: int = 60, start_price: float = 50_000_000) -> pd.DataFrame:
    return _make_ohlcv(n, 1440, start_price, volatility=0.025, seed=99)


# ── pyupbit 래퍼 ───────────────────────────────────────────

def _try_import_pyupbit():
    try:
        import pyupbit
        return pyupbit
    except ImportError:
        return None


def fetch_ohlcv(
    ticker: str = "KRW-BTC",
    interval: str = "minute1",
    count: int = 200,
    use_mock: bool = False,
) -> pd.DataFrame:
    """
    interval: "minute1" | "minute5" | "day"
    use_mock=True 이면 시뮬레이션 데이터 반환
    """
    if not use_mock:
        pyupbit = _try_import_pyupbit()
        if pyupbit:
            try:
                df = pyupbit.get_ohlcv(ticker, interval=interval, count=count)
                if df is not None and len(df) > 0:
                    df = df.reset_index().rename(columns={"index": "date"})
                    df.columns = [c.lower() for c in df.columns]
                    return df[["date", "open", "high", "low", "close", "volume"]]
            except Exception as e:
                print(f"[pyupbit 오류] {e} → 시뮬레이션 데이터 사용")

    interval_map = {"minute1": 1, "minute5": 5, "day": 1440}
    minutes = interval_map.get(interval, 1)
    seed_map = {"minute1": 42, "minute5": 77, "day": 99}
    seed = seed_map.get(interval, 42)

    return _make_ohlcv(count, minutes, seed=seed)


def fetch_current_price(ticker: str = "KRW-BTC", use_mock: bool = False) -> float:
    if not use_mock:
        pyupbit = _try_import_pyupbit()
        if pyupbit:
            try:
                price = pyupbit.get_current_price(ticker)
                if price:
                    return float(price)
            except Exception:
                pass
    rng = np.random.default_rng()
    return 50_000_000 * (1 + rng.normal(0, 0.001))


def fetch_balance(upbit_obj, ticker: str = "KRW-BTC", use_mock: bool = False) -> dict:
    if not use_mock and upbit_obj:
        try:
            krw = upbit_obj.get_balance("KRW")
            coin = upbit_obj.get_balance(ticker.replace("KRW-", ""))
            return {"krw": krw or 0, "coin": coin or 0}
        except Exception:
            pass
    return {"krw": 1_000_000, "coin": 0.0}


def get_upbit_client(access_key: str, secret_key: str):
    pyupbit = _try_import_pyupbit()
    if pyupbit and access_key and secret_key:
        try:
            return pyupbit.Upbit(access_key, secret_key)
        except Exception:
            pass
    return None


def place_order(
    upbit_obj,
    ticker: str,
    side: str,       # "buy" | "sell"
    amount_krw: float = 0,
    volume: float = 0,
    use_mock: bool = True,
) -> dict:
    """
    실제 주문 or 모의 주문
    side="buy"  → amount_krw 금액만큼 시장가 매수
    side="sell" → volume 수량만큼 시장가 매도
    """
    if use_mock:
        return {
            "mock": True,
            "side": side,
            "ticker": ticker,
            "amount": amount_krw if side == "buy" else volume,
            "status": "simulated",
        }
    if upbit_obj:
        try:
            if side == "buy":
                return upbit_obj.buy_market_order(ticker, amount_krw)
            else:
                return upbit_obj.sell_market_order(ticker, volume)
        except Exception as e:
            return {"error": str(e)}
    return {"error": "업비트 클라이언트 없음"}
