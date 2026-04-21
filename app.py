"""
업비트 복합 자동매매 대시보드
실행: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime
import time

from strategy import (
    calc_ma, calc_macd, calc_rsi, calc_volatility_breakout,
    check_buy_signal, check_sell_signal, run_backtest,
)
from data_fetcher import (
    fetch_ohlcv, fetch_current_price, fetch_balance,
    get_upbit_client, place_order,
)

# ── 페이지 설정 ────────────────────────────────────────────
st.set_page_config(
    page_title="업비트 복합 자동매매",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .metric-label { font-size: 12px; color: #888; }
    .metric-value { font-size: 22px; font-weight: 600; }
    .signal-buy  { background: #d4edda; color: #155724; padding: 8px 14px; border-radius: 8px; font-weight: 600; }
    .signal-sell { background: #f8d7da; color: #721c24; padding: 8px 14px; border-radius: 8px; font-weight: 600; }
    .signal-hold { background: #e2e3e5; color: #383d41; padding: 8px 14px; border-radius: 8px; font-weight: 600; }
    .cond-ok   { color: #28a745; font-weight: 600; }
    .cond-fail { color: #dc3545; font-weight: 600; }
    .stTabs [data-baseweb="tab"] { font-size: 14px; }
</style>
""", unsafe_allow_html=True)


# ── 세션 상태 초기화 ───────────────────────────────────────
def init_session():
    defaults = {
        "running": False,
        "position": 0.0,
        "buy_price": 0.0,
        "balance": 1_000_000,
        "trades": [],
        "log": [],
        "upbit": None,
        "last_signal": None,
        "tick": 0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session()


# ── 사이드바 ───────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ 설정")

    st.subheader("거래 설정")
    ticker = st.selectbox("코인", ["KRW-BTC", "KRW-ETH", "KRW-XRP", "KRW-SOL", "KRW-DOGE"])
    use_mock = st.toggle("모의 거래 모드", value=True, help="OFF 시 실제 주문 실행")

    st.subheader("API 키 (실거래 시 필요)")
    access_key = st.text_input("Access Key", type="password", placeholder="업비트 발급 키")
    secret_key = st.text_input("Secret Key", type="password", placeholder="업비트 발급 키")
    if st.button("API 연결"):
        client = get_upbit_client(access_key, secret_key)
        if client:
            st.session_state.upbit = client
            st.success("연결 성공")
        else:
            st.warning("pyupbit 미설치 또는 키 오류 → 모의 모드 유지")

    st.subheader("전략 파라미터")
    with st.expander("이동평균 (MA)", expanded=True):
        ma_short = st.slider("단기 MA", 3, 20, 5)
        ma_long  = st.slider("장기 MA", 10, 60, 20)

    with st.expander("MACD"):
        macd_fast   = st.slider("빠른 EMA", 5, 20, 12)
        macd_slow   = st.slider("느린 EMA", 15, 40, 26)
        macd_signal = st.slider("시그널", 5, 15, 9)

    with st.expander("RSI"):
        rsi_period = st.slider("RSI 기간", 7, 21, 14)
        rsi_buy    = st.slider("매수 기준 (미만)", 40, 70, 60)
        rsi_sell   = st.slider("매도 기준 (초과)", 60, 90, 75)

    with st.expander("변동성 돌파 (#5)"):
        vol_k = st.slider("K 값", 0.1, 1.0, 0.5, 0.05)

    with st.expander("리스크 관리"):
        stop_loss    = st.slider("손절 (%)", 0.5, 5.0, 2.0, 0.5)
        take_profit  = st.slider("익절 (%)", 1.0, 10.0, 3.0, 0.5)
        invest_ratio = st.slider("투자 비율 (%)", 10, 100, 20, 5)
        min_conds    = st.slider("최소 충족 조건 수", 3, 5, 4)

    params = {
        "ma_short": ma_short, "ma_long": ma_long,
        "macd_fast": macd_fast, "macd_slow": macd_slow, "macd_signal": macd_signal,
        "rsi_period": rsi_period,
        "rsi_buy_threshold": rsi_buy, "rsi_sell_threshold": rsi_sell,
        "vol_k": vol_k,
        "stop_loss_pct": stop_loss, "take_profit_pct": take_profit,
        "invest_ratio": invest_ratio / 100,
        "min_conditions": min_conds,
    }

    st.subheader("자동매매 제어")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("▶ 시작", use_container_width=True, type="primary"):
            st.session_state.running = True
            st.session_state.log.append(f"[{datetime.now().strftime('%H:%M:%S')}] 자동매매 시작")
    with col2:
        if st.button("■ 정지", use_container_width=True):
            st.session_state.running = False
            st.session_state.log.append(f"[{datetime.now().strftime('%H:%M:%S')}] 자동매매 정지")

    refresh = st.slider("갱신 주기 (초)", 5, 60, 10)


# ── 데이터 로드 & 지표 계산 ────────────────────────────────
@st.cache_data(ttl=10)
def load_data(ticker, use_mock, _tick):
    df_1m    = fetch_ohlcv(ticker, "minute1", 200, use_mock)
    df_5m    = fetch_ohlcv(ticker, "minute5", 200, use_mock)
    df_daily = fetch_ohlcv(ticker, "day",      60, use_mock)
    return df_1m, df_5m, df_daily


def apply_indicators(df_1m, df_5m, df_daily, params):
    df_1m = calc_rsi(calc_ma(df_1m, params["ma_short"], params["ma_long"]), params["rsi_period"])
    df_5m = calc_macd(calc_ma(df_5m, params["ma_short"], params["ma_long"]))
    df_daily = calc_volatility_breakout(df_daily, params["vol_k"])
    return df_1m, df_5m, df_daily


# ── 차트 ───────────────────────────────────────────────────
def make_price_chart(df, df_ind, title, show_ma=True):
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        row_heights=[0.55, 0.25, 0.20],
        vertical_spacing=0.03,
    )

    # 캔들
    fig.add_trace(go.Candlestick(
        x=df["date"], open=df["open"], high=df["high"],
        low=df["low"], close=df["close"],
        name="가격",
        increasing_line_color="#26a69a",
        decreasing_line_color="#ef5350",
    ), row=1, col=1)

    if show_ma and "ma_short" in df_ind.columns:
        fig.add_trace(go.Scatter(x=df_ind["date"], y=df_ind["ma_short"],
            line=dict(color="#1976D2", width=1.2), name=f"MA{params['ma_short']}"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_ind["date"], y=df_ind["ma_long"],
            line=dict(color="#FB8C00", width=1.2), name=f"MA{params['ma_long']}"), row=1, col=1)

    # MACD (있으면)
    if "macd" in df_ind.columns:
        fig.add_trace(go.Scatter(x=df_ind["date"], y=df_ind["macd"],
            line=dict(color="#7B1FA2", width=1), name="MACD"), row=2, col=1)
        fig.add_trace(go.Scatter(x=df_ind["date"], y=df_ind["macd_signal"],
            line=dict(color="#F06292", width=1), name="Signal"), row=2, col=1)
        colors = ["#26a69a" if v >= 0 else "#ef5350" for v in df_ind["macd_hist"].fillna(0)]
        fig.add_trace(go.Bar(x=df_ind["date"], y=df_ind["macd_hist"],
            marker_color=colors, name="Hist"), row=2, col=1)

    # RSI (있으면)
    if "rsi" in df_ind.columns:
        fig.add_trace(go.Scatter(x=df_ind["date"], y=df_ind["rsi"],
            line=dict(color="#0288D1", width=1.2), name="RSI"), row=3, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red",   line_width=0.8, row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", line_width=0.8, row=3, col=1)

    fig.update_layout(
        title=title, height=520,
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", y=1.02, x=0),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    fig.update_yaxes(title_text="가격", row=1, col=1)
    fig.update_yaxes(title_text="MACD", row=2, col=1)
    fig.update_yaxes(title_text="RSI",  row=3, col=1)
    return fig


# ── 메인 화면 ──────────────────────────────────────────────
st.title("📈 업비트 복합 자동매매 — 1분봉 + 5분봉")

mode_badge = "🟡 모의거래" if use_mock else "🔴 실거래"
run_badge  = "🟢 실행 중" if st.session_state.running else "⚪ 정지"
st.caption(f"{mode_badge}  |  {run_badge}  |  {ticker}  |  마지막 갱신: {datetime.now().strftime('%H:%M:%S')}")

# 자동 갱신 루프
if st.session_state.running:
    st.session_state.tick += 1

df_1m_raw, df_5m_raw, df_daily_raw = load_data(ticker, use_mock, st.session_state.tick)
df_1m, df_5m, df_daily = apply_indicators(df_1m_raw, df_5m_raw, df_daily_raw, params)

try:
    target_today = float(df_daily["target_price"].iloc[-1])
except Exception:
    target_today = 0.0

current_price = float(df_1m["close"].iloc[-1])
signal_info   = check_buy_signal(df_1m, df_5m, target_today, params)
sell_info     = check_sell_signal(df_1m, df_5m, st.session_state.buy_price, params) \
                if st.session_state.position > 0 else {"signal": False}

# 자동 주문 실행
if st.session_state.running:
    if st.session_state.position == 0 and signal_info["signal"]:
        invest = st.session_state.balance * params["invest_ratio"]
        result = place_order(st.session_state.upbit, ticker, "buy", amount_krw=invest, use_mock=use_mock)
        if "error" not in result:
            fee = invest * 0.0005
            st.session_state.position = (invest - fee) / current_price
            st.session_state.buy_price = current_price
            st.session_state.balance -= invest
            msg = f"[{datetime.now().strftime('%H:%M:%S')}] 매수 {current_price:,.0f}원 / 조건 {signal_info['score']}개"
            st.session_state.trades.append({"type": "매수", "price": current_price,
                "time": datetime.now().strftime('%H:%M:%S'), "reason": signal_info.get("reason", "")})
            st.session_state.log.append(msg)

    elif st.session_state.position > 0 and sell_info["signal"]:
        sell_val = st.session_state.position * current_price * (1 - 0.0005)
        result = place_order(st.session_state.upbit, ticker, "sell",
                             volume=st.session_state.position, use_mock=use_mock)
        if "error" not in result:
            pnl = sell_val - st.session_state.position * st.session_state.buy_price
            msg = f"[{datetime.now().strftime('%H:%M:%S')}] 매도 {current_price:,.0f}원 | {sell_info['reason']} | PnL: {pnl:+,.0f}원"
            st.session_state.trades.append({"type": "매도", "price": current_price,
                "time": datetime.now().strftime('%H:%M:%S'), "reason": sell_info.get("reason","")})
            st.session_state.log.append(msg)
            st.session_state.balance += sell_val
            st.session_state.position = 0.0
            st.session_state.buy_price = 0.0


# ── 상단 지표 카드 ─────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("현재가", f"{current_price:,.0f}원")
c2.metric("RSI (1m)", f"{df_1m['rsi'].iloc[-1]:.1f}" if "rsi" in df_1m else "-")
c3.metric("MACD (5m)", f"{df_5m['macd'].iloc[-1]:.0f}" if "macd" in df_5m else "-")
c4.metric("변동성 목표가", f"{target_today:,.0f}원" if target_today > 0 else "-")
c5.metric("잔고", f"{st.session_state.balance:,.0f}원")


# ── 신호 표시 ──────────────────────────────────────────────
st.markdown("---")
sig_col, cond_col = st.columns([1, 2])

with sig_col:
    st.subheader("현재 신호")
    if st.session_state.position > 0:
        pnl_pct = (current_price - st.session_state.buy_price) / st.session_state.buy_price * 100
        st.markdown(f'<div class="signal-buy">📦 보유 중 | 매수가 {st.session_state.buy_price:,.0f}원 | {pnl_pct:+.2f}%</div>', unsafe_allow_html=True)
        if sell_info["signal"]:
            st.markdown(f'<div class="signal-sell">🔴 매도 신호: {sell_info.get("reason","")}</div>', unsafe_allow_html=True)
    elif signal_info["signal"]:
        st.markdown(f'<div class="signal-buy">🟢 매수 신호 ({signal_info["score"]}/5개 충족)</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="signal-hold">⚪ 대기 중 ({signal_info.get("score",0)}/5개 충족)</div>', unsafe_allow_html=True)

with cond_col:
    st.subheader("조건 체크")
    conds = signal_info.get("conditions", {})
    labels = {
        "5m_ma_cross":  "5분봉 MA 골든크로스",
        "5m_macd_bull": "5분봉 MACD 상승",
        "5m_vol_break": "5분봉 변동성 돌파",
        "1m_ma_cross":  "1분봉 MA 골든크로스",
        "1m_rsi_ok":    f"1분봉 RSI < {rsi_buy}",
    }
    cols = st.columns(len(labels))
    for i, (k, label) in enumerate(labels.items()):
        ok = conds.get(k, False)
        icon = "✅" if ok else "❌"
        cols[i].markdown(f"**{icon}**<br><small>{label}</small>", unsafe_allow_html=True)


# ── 탭 ─────────────────────────────────────────────────────
st.markdown("---")
tab1, tab2, tab3, tab4 = st.tabs(["📊 차트", "🔁 백테스팅", "📋 거래 로그", "ℹ️ 전략 설명"])

with tab1:
    t1c1, t1c2 = st.columns(2)
    with t1c1:
        st.plotly_chart(make_price_chart(df_1m_raw.tail(120), df_1m.tail(120), f"{ticker} — 1분봉"), use_container_width=True)
    with t1c2:
        st.plotly_chart(make_price_chart(df_5m_raw.tail(80),  df_5m.tail(80),  f"{ticker} — 5분봉"), use_container_width=True)


with tab2:
    st.subheader("백테스팅")
    bt_col1, bt_col2, bt_col3 = st.columns([1, 1, 1])
    with bt_col1:
        bt_initial = st.number_input("초기 자금 (원)", value=1_000_000, step=100_000)
    with bt_col2:
        bt_candles = st.selectbox("1분봉 데이터 수", [500, 1000, 2000, 5000], index=1)
    with bt_col3:
        st.write("")
        st.write("")
        run_bt = st.button("백테스팅 실행", type="primary", use_container_width=True)

    if run_bt:
        with st.spinner("백테스팅 실행 중..."):
            bt_1m = fetch_ohlcv(ticker, "minute1", bt_candles, True)
            bt_5m = fetch_ohlcv(ticker, "minute5", bt_candles // 5, True)
            bt_d  = fetch_ohlcv(ticker, "day", 60, True)
            result = run_backtest(bt_1m, bt_5m, bt_d, params, bt_initial)

        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("최종 수익률", f"{result['total_return']:+.2f}%",
                  delta_color="normal" if result['total_return'] >= 0 else "inverse")
        m2.metric("승률", f"{result['win_rate']:.1f}%")
        m3.metric("총 거래 수", f"{result['trade_count']}회")
        m4.metric("평균 수익", f"{result['avg_profit_pct']:.2f}%")
        m5.metric("최대 수익", f"{result['max_profit_pct']:.2f}%")
        m6.metric("최대 낙폭 (MDD)", f"{result['max_drawdown']:.2f}%")

        if len(result["equity_curve"]) > 0:
            eq_df = result["equity_curve"]
            fig_eq = px.area(eq_df, x="date", y="equity",
                             title="자산 곡선 (Equity Curve)",
                             labels={"equity": "자산 (원)", "date": "시간"},
                             color_discrete_sequence=["#1976D2"])
            fig_eq.add_hline(y=bt_initial, line_dash="dash", line_color="gray", line_width=0.8)
            fig_eq.update_layout(height=300, margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig_eq, use_container_width=True)

        if len(result["trades"]) > 0:
            sells = result["trades"][result["trades"]["type"] == "sell"]
            if len(sells):
                fig_pnl = px.bar(sells, x="date", y="profit_pct",
                                 color=sells["profit_pct"].apply(lambda x: "수익" if x > 0 else "손실"),
                                 color_discrete_map={"수익": "#26a69a", "손실": "#ef5350"},
                                 title="거래별 손익 (%)",
                                 labels={"profit_pct": "수익률 (%)", "date": "날짜"})
                fig_pnl.update_layout(height=280, margin=dict(l=0, r=0, t=40, b=0), showlegend=False)
                st.plotly_chart(fig_pnl, use_container_width=True)

            with st.expander("전체 거래 내역 보기"):
                st.dataframe(result["trades"], use_container_width=True)


with tab3:
    st.subheader("실시간 로그")
    log_container = st.container()
    with log_container:
        if st.session_state.log:
            for line in reversed(st.session_state.log[-50:]):
                st.text(line)
        else:
            st.info("자동매매를 시작하면 로그가 여기에 표시됩니다.")

    if st.session_state.trades:
        st.subheader("체결 내역")
        st.dataframe(pd.DataFrame(st.session_state.trades), use_container_width=True)

    if st.button("로그 초기화"):
        st.session_state.log = []
        st.session_state.trades = []
        st.rerun()


with tab4:
    st.subheader("복합 전략 설명")
    st.markdown("""
### 사용 전략 조합

| 번호 | 전략 | 역할 | 봉 |
|------|------|------|----|
| #1 | MA 크로스 | 추세 방향 확인 | 5분봉 + 1분봉 |
| #2 | MACD | 추세 강도 확인 | 5분봉 |
| #3 | RSI | 과매수 진입 방지 | 1분봉 |
| #5 | 변동성 돌파 | 모멘텀 진입 | 일봉 기준 목표가 |

### 매수 조건 (AND — 4개 이상 충족 시)
1. **[5분봉]** MA5 > MA20 (골든크로스 상태)
2. **[5분봉]** MACD선 > 시그널선 (상승 추세)
3. **[5분봉]** 현재가 >= 시가 + 전일 Range × K (변동성 돌파)
4. **[1분봉]** MA5 > MA20 (단기 방향 일치)
5. **[1분봉]** RSI < 60 (과매수 아닌 구간)

### 매도 조건 (OR — 하나라도 해당 시 즉시 청산)
- 손절: 매수가 대비 -2% 이하
- 익절: 매수가 대비 +3% 이상
- RSI > 75 (단기 과매수)
- 5분봉 MACD 하락 전환
- 1분봉 데드크로스

### 주의사항
> ⚠️ 본 프로그램은 교육·테스트 목적입니다. 실제 투자 시 반드시 백테스팅을 충분히 하고, 소액으로 시작하세요.
> 암호화폐 투자는 원금 손실 위험이 있습니다.
    """)

# 자동 갱신
if st.session_state.running:
    time.sleep(refresh)
    st.rerun()
