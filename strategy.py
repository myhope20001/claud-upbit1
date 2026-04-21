"""
복합 자동매매 전략 엔진
#1 MA Cross + #2 MACD + #3 RSI + #5 변동성 돌파
5분봉: 추세 필터 (MA Cross, MACD, 변동성 돌파)
1분봉: 진입 타이밍 (RSI, MA Cross)
"""

import pandas as pd
import numpy as np


# ── 지표 계산 ──────────────────────────────────────────────

def calc_ma(df: pd.DataFrame, short: int = 5, long: int = 20) -> pd.DataFrame:
    df = df.copy()
    df["ma_short"] = df["close"].rolling(short).mean()
    df["ma_long"] = df["close"].rolling(long).mean()
    df["ma_cross"] = df["ma_short"] > df["ma_long"]
    df["golden_cross"] = df["ma_cross"] & ~df["ma_cross"].shift(1).fillna(False)
    df["dead_cross"] = ~df["ma_cross"] & df["ma_cross"].shift(1).fillna(False)
    return df


def calc_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    df = df.copy()
    ema_fast = df["close"].ewm(span=fast, adjust=False).mean()
    ema_slow = df["close"].ewm(span=slow, adjust=False).mean()
    df["macd"] = ema_fast - ema_slow
    df["macd_signal"] = df["macd"].ewm(span=signal, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]
    df["macd_bull"] = df["macd"] > df["macd_signal"]
    return df


def calc_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    df = df.copy()
    delta = df["close"].diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))
    return df


def calc_volatility_breakout(df_daily: pd.DataFrame, k: float = 0.5) -> pd.DataFrame:
    """
    변동성 돌파 목표가 계산 (일봉 기준)
    목표가 = 당일 시가 + (전일 고가 - 전일 저가) × K
    """
    df = df_daily.copy()
    df["prev_range"] = df["high"].shift(1) - df["low"].shift(1)
    df["target_price"] = df["open"] + df["prev_range"] * k
    return df


# ── 신호 판단 ──────────────────────────────────────────────

def check_buy_signal(
    df_1m: pd.DataFrame,
    df_5m: pd.DataFrame,
    target_price: float,
    params: dict,
) -> dict:
    """
    복합 매수 조건 체크 (5개 조건)
    Returns: {signal: bool, conditions: dict, score: int}
    """
    if len(df_1m) < 30 or len(df_5m) < 30:
        return {"signal": False, "conditions": {}, "score": 0}

    latest_1m = df_1m.iloc[-1]
    latest_5m = df_5m.iloc[-1]
    current_price = latest_1m["close"]

    conditions = {
        "5m_ma_cross":   bool(latest_5m.get("ma_cross", False)),
        "5m_macd_bull":  bool(latest_5m.get("macd_bull", False)),
        "5m_vol_break":  bool(target_price > 0 and current_price >= target_price),
        "1m_ma_cross":   bool(latest_1m.get("ma_cross", False)),
        "1m_rsi_ok":     bool(
            latest_1m.get("rsi", 100) < params.get("rsi_buy_threshold", 60)
        ),
    }

    score = sum(conditions.values())
    required = params.get("min_conditions", 4)
    signal = score >= required

    return {
        "signal": signal,
        "conditions": conditions,
        "score": score,
        "current_price": current_price,
        "target_price": target_price,
        "rsi_1m": latest_1m.get("rsi"),
        "macd_5m": latest_5m.get("macd"),
        "ma_short_5m": latest_5m.get("ma_short"),
        "ma_long_5m": latest_5m.get("ma_long"),
    }


def check_sell_signal(
    df_1m: pd.DataFrame,
    df_5m: pd.DataFrame,
    buy_price: float,
    params: dict,
) -> dict:
    """
    매도 조건 체크 (OR — 하나라도 해당 시 청산)
    """
    if len(df_1m) < 30 or buy_price <= 0:
        return {"signal": False, "reason": None}

    latest_1m = df_1m.iloc[-1]
    latest_5m = df_5m.iloc[-1]
    current_price = latest_1m["close"]

    pnl_pct = (current_price - buy_price) / buy_price * 100

    reasons = []
    if pnl_pct <= -abs(params.get("stop_loss_pct", 2.0)):
        reasons.append(f"손절 ({pnl_pct:.2f}%)")
    if pnl_pct >= params.get("take_profit_pct", 3.0):
        reasons.append(f"익절 ({pnl_pct:.2f}%)")
    if latest_1m.get("rsi", 0) > params.get("rsi_sell_threshold", 75):
        reasons.append(f"RSI 과매수 ({latest_1m['rsi']:.1f})")
    if not latest_5m.get("macd_bull", True):
        reasons.append("5m MACD 하락 전환")
    if not latest_1m.get("ma_cross", True):
        reasons.append("1m 데드크로스")

    return {
        "signal": len(reasons) > 0,
        "reason": reasons[0] if reasons else None,
        "all_reasons": reasons,
        "current_price": current_price,
        "pnl_pct": pnl_pct,
    }


# ── 백테스팅 ───────────────────────────────────────────────

def run_backtest(
    df_1m: pd.DataFrame,
    df_5m: pd.DataFrame,
    df_daily: pd.DataFrame,
    params: dict,
    initial_balance: float = 1_000_000,
) -> dict:
    """
    단순 이벤트 기반 백테스트
    """
    df_1m = calc_ma(calc_rsi(df_1m), params["ma_short"], params["ma_long"])
    df_5m = calc_macd(calc_ma(df_5m, params["ma_short"], params["ma_long"]))
    df_daily = calc_volatility_breakout(df_daily, params["vol_k"])

    balance = initial_balance
    position = 0.0
    buy_price = 0.0
    trades = []
    equity_curve = []

    # 5분봉 인덱스를 1분봉에 맞춰 리샘플
    df_5m_idx = df_5m.set_index("date") if "date" in df_5m.columns else df_5m
    df_daily_idx = df_daily.set_index("date") if "date" in df_daily.columns else df_daily

    for i in range(30, len(df_1m)):
        row_1m = df_1m.iloc[:i]
        ts = df_1m.iloc[i]["date"] if "date" in df_1m.columns else df_1m.index[i]
        current_price = df_1m.iloc[i]["close"]

        # 가장 가까운 5분봉 슬라이스
        try:
            mask_5m = df_5m_idx.index <= ts
            slice_5m = df_5m_idx[mask_5m].tail(30)
        except Exception:
            slice_5m = df_5m.iloc[:30]

        # 당일 변동성 돌파 목표가
        try:
            date_str = str(ts)[:10]
            row_d = df_daily_idx[df_daily_idx.index.astype(str).str[:10] == date_str]
            target_price = float(row_d["target_price"].iloc[-1]) if len(row_d) else 0.0
        except Exception:
            target_price = 0.0

        # 보유 중: 매도 체크
        if position > 0:
            sell = check_sell_signal(row_1m, slice_5m, buy_price, params)
            if sell["signal"]:
                sell_amount = position * current_price * (1 - 0.0005)
                profit = sell_amount - (position * buy_price)
                trades.append({
                    "type": "sell",
                    "date": str(ts),
                    "price": current_price,
                    "amount": sell_amount,
                    "profit": profit,
                    "profit_pct": sell["pnl_pct"],
                    "reason": sell["reason"],
                    "balance": balance + sell_amount,
                })
                balance += sell_amount
                position = 0.0
                buy_price = 0.0

        # 미보유: 매수 체크
        elif position == 0:
            sig = check_buy_signal(row_1m, slice_5m, target_price, params)
            if sig["signal"]:
                invest = balance * params.get("invest_ratio", 0.2)
                fee = invest * 0.0005
                position = (invest - fee) / current_price
                buy_price = current_price
                balance -= invest
                trades.append({
                    "type": "buy",
                    "date": str(ts),
                    "price": current_price,
                    "amount": invest,
                    "profit": 0,
                    "profit_pct": 0,
                    "reason": f"조건 {sig['score']}개 충족",
                    "balance": balance,
                })

        equity = balance + position * current_price
        equity_curve.append({"date": str(ts), "equity": equity})

    # 미청산 포지션 강제 종료
    if position > 0:
        final_price = df_1m.iloc[-1]["close"]
        balance += position * final_price * (1 - 0.0005)

    trade_df = pd.DataFrame(trades)
    sells = trade_df[trade_df["type"] == "sell"] if len(trade_df) else pd.DataFrame()

    total_return = (balance - initial_balance) / initial_balance * 100
    win_rate = (
        len(sells[sells["profit"] > 0]) / len(sells) * 100 if len(sells) > 0 else 0
    )
    avg_profit = sells["profit_pct"].mean() if len(sells) > 0 else 0
    max_profit = sells["profit_pct"].max() if len(sells) > 0 else 0
    max_loss = sells["profit_pct"].min() if len(sells) > 0 else 0

    equity_df = pd.DataFrame(equity_curve)
    if len(equity_df):
        rolling_max = equity_df["equity"].cummax()
        drawdown = (equity_df["equity"] - rolling_max) / rolling_max * 100
        max_drawdown = drawdown.min()
    else:
        max_drawdown = 0

    return {
        "trades": trade_df,
        "equity_curve": equity_df,
        "final_balance": balance,
        "total_return": total_return,
        "win_rate": win_rate,
        "avg_profit_pct": avg_profit,
        "max_profit_pct": max_profit,
        "max_loss_pct": max_loss,
        "max_drawdown": max_drawdown,
        "trade_count": len(sells),
        "buy_count": len(trade_df[trade_df["type"] == "buy"]) if len(trade_df) else 0,
    }
