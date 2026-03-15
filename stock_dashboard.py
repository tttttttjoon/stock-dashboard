"""
🇰🇷 한국 주식 분석 웹 대시보드 v3.3
================================
실행:
  streamlit run stock_dashboard.py

설치:
  pip install streamlit pykrx pandas plotly ta xgboost scikit-learn yfinance tensorflow feedparser
"""

import warnings
warnings.filterwarnings("ignore")
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pykrx import stock
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ta
import yfinance as yf
import feedparser
import re
from urllib.parse import quote

st.set_page_config(page_title="🇰🇷 한국 주식 분석", page_icon="📈", layout="wide")
st.title("🇰🇷 한국 주식 분석 대시보드 v3.3")

# ──────────────────────────────────────────
# 사이드바
# ──────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ 분석 설정")
    st.subheader("📌 종목 선택")

    POPULAR = {
        "삼성전자": "005930", "SK하이닉스": "000660",
        "LG에너지솔루션": "373220", "카카오": "035720",
        "네이버": "035420", "현대차": "005380",
        "셀트리온": "068270", "기아": "000270",
        "POSCO홀딩스": "005490", "KB금융": "105560",
    }

    quick = st.selectbox("빠른 선택", ["직접 입력"] + list(POPULAR.keys()))
    if quick == "직접 입력":
        main_ticker = st.text_input("종목코드 입력", value="005930", max_chars=6)
    else:
        main_ticker = POPULAR[quick]
        st.info(f"선택된 코드: {main_ticker}")

    st.subheader("📊 비교 종목 (선택사항)")
    compare_input = st.text_input("비교할 종목코드 (쉼표로 구분)", placeholder="예: 000660, 035720", value="")
    compare_tickers = [t.strip() for t in compare_input.split(",") if t.strip()] if compare_input else []

    st.subheader("📅 기간 설정")
    period = st.selectbox("조회 기간", ["3개월", "6개월", "1년", "2년"], index=2)
    days = {"3개월": 90, "6개월": 180, "1년": 365, "2년": 730}[period]
    END      = datetime.today().strftime("%Y%m%d")
    START    = (datetime.today() - timedelta(days=days)).strftime("%Y%m%d")
    START_DT = datetime.today() - timedelta(days=days)

    st.subheader("🤖 AI 예측")
    show_xgb     = st.checkbox("XGBoost 예측", value=True)
    show_prophet = st.checkbox("Prophet 예측", value=True)
    show_lstm    = st.checkbox("LSTM 예측", value=True)
    predict_days = st.slider("예측 기간 (일)", 10, 60, 30)

    st.markdown("---")
    run_btn = st.button("🔍 분석 시작", type="primary", use_container_width=True)

# ──────────────────────────────────────────
# 데이터 로드
# ──────────────────────────────────────────
@st.cache_data(ttl=3600)
def load_stock(ticker, start, end):
    try:
        df = stock.get_market_ohlcv(start, end, ticker)
        df.columns = ["시가", "고가", "저가", "종가", "거래량", "등락률"]
        df.index = pd.to_datetime(df.index)
        try:
            name = stock.get_market_ticker_name(ticker)
            name = name.strip() if isinstance(name, str) and name.strip() else ticker
        except:
            name = ticker
        return df, name
    except:
        return None, ticker

@st.cache_data(ttl=3600)
def load_macro(start_dt, end_dt):
    kospi, fx = None, None
    try:
        k = yf.download("^KS11", start=start_dt, end=end_dt, progress=False)
        if isinstance(k.columns, pd.MultiIndex): k.columns = k.columns.get_level_values(0)
        kospi = k["Close"]
    except: pass
    try:
        f = yf.download("KRW=X", start=start_dt, end=end_dt, progress=False)
        if isinstance(f.columns, pd.MultiIndex): f.columns = f.columns.get_level_values(0)
        fx = f["Close"]
    except: pass
    return kospi, fx

def calc_indicators(df):
    df = df.copy()
    df["MA5"]    = df["종가"].rolling(5).mean()
    df["MA20"]   = df["종가"].rolling(20).mean()
    df["MA60"]   = df["종가"].rolling(60).mean()
    bb           = ta.volatility.BollingerBands(close=df["종가"], window=20, window_dev=2)
    df["BB_상단"] = bb.bollinger_hband()
    df["BB_하단"] = bb.bollinger_lband()
    df["RSI"]    = ta.momentum.RSIIndicator(close=df["종가"], window=14).rsi()
    macd         = ta.trend.MACD(close=df["종가"])
    df["MACD"]   = macd.macd()
    df["Signal"] = macd.macd_signal()
    df["Hist"]   = macd.macd_diff()
    return df

def get_signal_summary(df):
    latest = df.dropna(subset=["RSI", "MACD"]).iloc[-1]
    prev   = df.dropna(subset=["RSI", "MACD"]).iloc[-2] if len(df) >= 2 else latest
    signals, score = [], 0

    rsi = latest["RSI"]
    if rsi < 30:   signals.append(("RSI", f"🔵 과매도 ({rsi:.1f})", "매수", +1)); score += 1
    elif rsi > 70: signals.append(("RSI", f"🔴 과매수 ({rsi:.1f})", "매도", -1)); score -= 1
    else:          signals.append(("RSI", f"⚪ 중립 ({rsi:.1f})", "중립", 0))

    mc, ms, pc, ps = latest["MACD"], latest["Signal"], prev["MACD"], prev["Signal"]
    if pc < ps and mc > ms:   signals.append(("MACD", "📈 골든크로스 발생!", "매수", +2)); score += 2
    elif pc > ps and mc < ms: signals.append(("MACD", "📉 데드크로스 발생!", "매도", -2)); score -= 2
    elif mc > ms:             signals.append(("MACD", "📈 골든크로스 유지", "매수", +1)); score += 1
    else:                     signals.append(("MACD", "📉 데드크로스 유지", "매도", -1)); score -= 1

    price, bb_u, bb_l = latest["종가"], latest["BB_상단"], latest["BB_하단"]
    if price < bb_l:              signals.append(("볼린저밴드", "🔵 하단 이탈", "매수", +1)); score += 1
    elif price > bb_u:            signals.append(("볼린저밴드", "🔴 상단 돌파", "매도", -1)); score -= 1
    elif price > (bb_u + bb_l)/2: signals.append(("볼린저밴드", "⚪ 중간선 위", "중립", 0))
    else:                         signals.append(("볼린저밴드", "⚪ 중간선 아래", "중립", 0))

    ma5, ma20, ma60 = latest["MA5"], latest["MA20"], latest["MA60"]
    if ma5 > ma20 > ma60:   signals.append(("이동평균", "📈 정배열", "매수", +1)); score += 1
    elif ma5 < ma20 < ma60: signals.append(("이동평균", "📉 역배열", "매도", -1)); score -= 1
    else:                   signals.append(("이동평균", "⚪ 혼조세", "중립", 0))

    if score >= 3:    overall = ("🟢 강한 매수 신호", "#00C851")
    elif score >= 1:  overall = ("🔵 약한 매수 신호", "#33b5e5")
    elif score <= -3: overall = ("🔴 강한 매도 신호", "#ff4444")
    elif score <= -1: overall = ("🟠 약한 매도 신호", "#ffbb33")
    else:             overall = ("⚪ 중립 / 관망",    "#aaaaaa")
    return signals, score, overall

# ──────────────────────────────────────────
# 뉴스 로드 (URL 인코딩 수정)
# ──────────────────────────────────────────
@st.cache_data(ttl=1800)
def load_news(stock_name, ticker):
    news_list = []
    query = quote(f"{stock_name} 주가")
    rss   = f"https://news.google.com/rss/search?q={query}&hl=ko&gl=KR&ceid=KR:ko"
    feed  = feedparser.parse(rss)

    # 단순 주가 전망 기사 필터링 키워드
    skip_keywords = [
        "목표가", "얼마 간다", "조 간다", "만원 간다",
        "전망↑", "전망↓", "상향", "하향", "매수", "매도",
        "투자의견", "리포트"
    ]

    seen_titles = []  # 중복 체크용

    for entry in feed.entries[:50]:  # 더 많이 가져와서 필터링
        pt = entry.get("published_parsed")
        if pt:
            pub_dt = datetime(pt[0], pt[1], pt[2], pt[3], pt[4], pt[5])
        else:
            pub_dt = datetime.now()

        title   = re.sub(r"<[^>]+>", "", entry.get("title", "제목 없음"))
        link    = entry.get("link", "#")
        source  = entry.get("source", {}).get("title", "")
        pub_str = pub_dt.strftime("%Y-%m-%d %H:%M")

        # 단순 주가 전망 기사 스킵
        if any(kw in title for kw in skip_keywords):
            continue

        # 중복 제거 (앞 15글자가 비슷하면 스킵)
        short_title = title[:15]
        if short_title in seen_titles:
            continue
        seen_titles.append(short_title)

        news_list.append({
            "제목": title,
            "링크": link,
            "출처": source,
            "날짜": pub_str,
            "datetime": pub_dt
        })

        if len(news_list) >= 10:
            break

    news_list.sort(key=lambda x: x["datetime"], reverse=True)
    return news_list

# ──────────────────────────────────────────
# 메인
# ──────────────────────────────────────────
if run_btn or "df_main" not in st.session_state:
    st.session_state["ticker"]   = main_ticker
    st.session_state["compares"] = compare_tickers

if st.session_state.get("ticker"):
    ticker = st.session_state["ticker"]

    with st.spinner("📥 데이터 수집 중..."):
        df, name = load_stock(ticker, START, END)
    if df is None or df.empty:
        st.error(f"❌ '{ticker}' 데이터를 불러오지 못했어요.")
        st.stop()

    df = calc_indicators(df)

    with st.spinner("🌐 코스피/환율 수집 중..."):
        kospi_s, fx_s = load_macro(START_DT, datetime.today())
        if kospi_s is not None: df["코스피"] = kospi_s.reindex(df.index).ffill()
        if fx_s    is not None: df["환율"]   = fx_s.reindex(df.index).ffill()

    latest = df.dropna(subset=["RSI", "MACD"]).iloc[-1]

    # 상단 카드
    st.subheader(f"📋 {name} ({ticker}) 현황")
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    price = latest["종가"]
    prev  = df["종가"].iloc[-2] if len(df) >= 2 else price
    chg, chg_pct = price - prev, (price - prev) / prev * 100
    rsi_val, macd_val, sig_val = latest["RSI"], latest["MACD"], latest["Signal"]
    bb_u, bb_l = latest["BB_상단"], latest["BB_하단"]

    c1.metric("현재가", f"{price:,.0f}원", f"{chg:+,.0f} ({chg_pct:+.2f}%)")
    c2.metric("RSI", f"{rsi_val:.1f}", "과매수🔴" if rsi_val>70 else "과매도🔵" if rsi_val<30 else "중립⚪")
    c3.metric("MACD", "골든크로스📈" if macd_val>sig_val else "데드크로스📉", f"{macd_val:.0f}/{sig_val:.0f}")
    c4.metric("볼린저밴드", "상단돌파🔴" if price>bb_u else "하단이탈🔵" if price<bb_l else "밴드내⚪", f"{bb_l:,.0f}~{bb_u:,.0f}")
    if "코스피" in df and df["코스피"].notna().any(): c5.metric("코스피", f"{df['코스피'].dropna().iloc[-1]:,.2f}")
    if "환율"   in df and df["환율"].notna().any():   c6.metric("환율",   f"{df['환율'].dropna().iloc[-1]:,.1f}원")

    st.markdown("---")

    # 종합 신호
    st.subheader("📊 지표 종합 신호")
    signals, score, (overall_text, overall_color) = get_signal_summary(df)
    st.markdown(f"""
        <div style='background:linear-gradient(135deg,#1e1e2e,#2a2a3e);border:2px solid {overall_color};
        border-radius:12px;padding:20px 30px;margin-bottom:20px;text-align:center;'>
            <div style='font-size:28px;font-weight:bold;color:{overall_color};'>{overall_text}</div>
            <div style='font-size:14px;color:#aaa;margin-top:8px;'>
                종합 점수: {score:+d}점 &nbsp;|&nbsp; 기준일: {df.index[-1].strftime('%Y-%m-%d')}
            </div>
            <div style='font-size:12px;color:#666;margin-top:6px;'>
                ⚠️ 투자 판단은 본인 책임입니다. 이 신호는 참고용입니다.
            </div>
        </div>""", unsafe_allow_html=True)

    scols = st.columns(4)
    cmap  = {"매수": "#00C851", "매도": "#ff4444", "중립": "#aaaaaa"}
    for i, (ind, desc, action, pts) in enumerate(signals):
        color = cmap[action]
        scols[i].markdown(f"""
            <div style='background:#1e1e2e;border:1px solid {color};border-radius:10px;
            padding:14px;text-align:center;height:110px;'>
                <div style='font-size:13px;color:#aaa;margin-bottom:6px;'>{ind}</div>
                <div style='font-size:13px;color:{color};font-weight:bold;'>{desc}</div>
                <div style='font-size:12px;color:{color};margin-top:6px;'>{action} ({pts:+d}점)</div>
            </div>""", unsafe_allow_html=True)

    with st.expander("💡 신호 읽는 법"):
        st.markdown("""
        | 점수 | 의미 |
        |------|------|
        | +3 이상 | 🟢 강한 매수 신호 |
        | +1 ~ +2 | 🔵 약한 매수 신호 |
        | 0 | ⚪ 중립 / 관망 |
        | -1 ~ -2 | 🟠 약한 매도 신호 |
        | -3 이하 | 🔴 강한 매도 신호 |
        > 지표는 **후행 지표**예요. 뉴스/실적/금리 등 외부 요인도 함께 고려하세요.
        """)

    st.markdown("---")

    # 메인 차트
    st.subheader("📈 주가 차트  ← 마우스를 올리면 모든 지표가 한번에 나와요!")

    def make_full_hover(row):
        price = row.get("종가", 0)
        vol   = row.get("거래량", 0)
        rsi   = row.get("RSI", float("nan"))
        macd  = row.get("MACD", float("nan"))
        sig   = row.get("Signal", float("nan"))
        bb_u  = row.get("BB_상단", float("nan"))
        bb_l  = row.get("BB_하단", float("nan"))
        ma5   = row.get("MA5", float("nan"))
        ma20  = row.get("MA20", float("nan"))
        ma60  = row.get("MA60", float("nan"))

        rsi_txt  = f"RSI {rsi:.1f} {'🔴과매수' if rsi>70 else '🔵과매도' if rsi<30 else '⚪중립'}" if pd.notna(rsi) else "RSI -"
        macd_txt = f"MACD {'📈골든' if pd.notna(macd) and pd.notna(sig) and macd>sig else '📉데드'} ({macd:.0f}/{sig:.0f})" if pd.notna(macd) else "MACD -"
        bb_txt   = f"BB {'🔴상단돌파' if price>bb_u else '🔵하단이탈' if price<bb_l else '⚪밴드내'}" if pd.notna(bb_u) else "BB -"

        return (
            f"<b>💰 주가</b><br>"
            f"종가: {price:,.0f}원<br>"
            f"MA5: {ma5:,.0f} | MA20: {ma20:,.0f} | MA60: {ma60:,.0f}<br>"
            f"BB상단: {bb_u:,.0f} | BB하단: {bb_l:,.0f}<br>"
            f"━━━━━━━━━━━━━━━━<br>"
            f"<b>📊 기술적 지표</b><br>"
            f"{rsi_txt}<br>"
            f"{macd_txt}<br>"
            f"{bb_txt}<br>"
            f"━━━━━━━━━━━━━━━━<br>"
            f"<b>📦 거래량: {vol:,.0f}주</b>"
            f"<extra></extra>"
        )

    hover_texts = [make_full_hover(df.iloc[i]) for i in range(len(df))]

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        row_heights=[0.6, 0.2, 0.2],
        vertical_spacing=0.02,
        subplot_titles=[
            f"{name} ({ticker}) 주가",
            "RSI (14)  |  70↑ 과매수  /  30↓ 과매도",
            "MACD  |  Signal 상향돌파 → 매수신호"
        ]
    )

    fig.add_trace(go.Scatter(
        x=df.index, y=df["종가"], mode="lines",
        line=dict(color="rgba(0,0,0,0)", width=0),
        name="전체정보", text=hover_texts,
        hovertemplate="%{text}", showlegend=False,
    ), row=1, col=1)

    fig.add_trace(go.Candlestick(
        x=df.index, open=df["시가"], high=df["고가"],
        low=df["저가"], close=df["종가"], name="주가",
        increasing_line_color="#FF3B30",
        decreasing_line_color="#007AFF",
        hoverinfo="skip",
    ), row=1, col=1)

    for col, color, label in [("MA5","#FFD60A","MA5"),("MA20","#FF9F0A","MA20"),("MA60","#30D158","MA60")]:
        fig.add_trace(go.Scatter(x=df.index, y=df[col], line=dict(color=color, width=1.5), name=label, hoverinfo="skip"), row=1, col=1)

    fig.add_trace(go.Scatter(x=df.index, y=df["BB_상단"], line=dict(color="rgba(150,150,150,0.5)", width=1, dash="dot"), name="BB 상단", hoverinfo="skip"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["BB_하단"], fill="tonexty", fillcolor="rgba(150,150,150,0.1)", line=dict(color="rgba(150,150,150,0.5)", width=1, dash="dot"), name="BB 하단", hoverinfo="skip"), row=1, col=1)

    fig.add_trace(go.Scatter(x=df.index, y=df["RSI"], line=dict(color="#BF5AF2", width=2), name="RSI", hoverinfo="skip"), row=2, col=1)
    fig.add_hline(y=70, line=dict(color="red",  dash="dash", width=1), row=2, col=1)
    fig.add_hline(y=30, line=dict(color="blue", dash="dash", width=1), row=2, col=1)
    fig.add_hrect(y0=70, y1=100, fillcolor="red",  opacity=0.05, row=2, col=1)
    fig.add_hrect(y0=0,  y1=30,  fillcolor="blue", opacity=0.05, row=2, col=1)

    fig.add_trace(go.Scatter(x=df.index, y=df["MACD"],   line=dict(color="#FF9F0A", width=2),              name="MACD",   hoverinfo="skip"), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["Signal"], line=dict(color="#FF3B30", width=1.5, dash="dot"), name="Signal", hoverinfo="skip"), row=3, col=1)
    hist_colors = ["#FF3B30" if v >= 0 else "#007AFF" for v in df["Hist"].fillna(0)]
    fig.add_trace(go.Bar(x=df.index, y=df["Hist"], marker_color=hist_colors, name="히스토그램", hoverinfo="skip"), row=3, col=1)

    fig.update_layout(
        template="plotly_dark", height=800,
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", y=1.02),
        margin=dict(l=10, r=10, t=60, b=10),
        hovermode="x",
        hoverlabel=dict(bgcolor="#1e1e2e", bordercolor="#30D158", font=dict(size=12), namelength=-1),
    )
    fig.update_xaxes(showspikes=True, spikemode="across+toaxis", spikesnap="cursor", spikecolor="#aaaaaa", spikethickness=1, spikedash="solid")
    fig.update_yaxes(title_text="가격(원)", row=1, col=1)
    fig.update_yaxes(title_text="RSI",      row=2, col=1, range=[0, 100])
    fig.update_yaxes(title_text="MACD",     row=3, col=1)

    st.plotly_chart(fig, use_container_width=True)

    # 비교 종목
    compare_tickers = st.session_state.get("compares", [])
    if compare_tickers:
        st.markdown("---")
        st.subheader("📊 종목 비교 차트")
        fig_cmp = go.Figure()
        base = df["종가"].iloc[0]
        fig_cmp.add_trace(go.Scatter(x=df.index, y=(df["종가"]/base*100), name=f"{name} ({ticker})", line=dict(width=2), hovertemplate=f"{name}: %{{y:.1f}}<extra></extra>"))
        for ct in compare_tickers:
            cdf, cname = load_stock(ct, START, END)
            if cdf is not None and not cdf.empty:
                cbase = cdf["종가"].iloc[0]
                fig_cmp.add_trace(go.Scatter(x=cdf.index, y=(cdf["종가"]/cbase*100), name=f"{cname} ({ct})", line=dict(width=2), hovertemplate=f"{cname}: %{{y:.1f}}<extra></extra>"))
            else:
                st.warning(f"⚠️ '{ct}' 데이터를 불러오지 못했어요.")
        fig_cmp.add_hline(y=100, line=dict(color="gray", dash="dash", width=1))
        fig_cmp.update_layout(title="📊 수익률 비교 (시작일 = 100 기준)", template="plotly_dark", height=400, hovermode="x unified", yaxis_title="상대 수익률 (%)", margin=dict(l=10, r=10, t=60, b=10))
        st.plotly_chart(fig_cmp, use_container_width=True)

    # ──────────────────────────────────────────
    # AI 예측
    # ──────────────────────────────────────────
    st.markdown("---")
    st.subheader("🤖 AI 예측")

    ai_cols = [n for n, show in [("XGBoost", show_xgb), ("Prophet", show_prophet), ("LSTM", show_lstm)] if show]
    cols    = st.columns(len(ai_cols)) if ai_cols else []

    def make_ai_chart(title, x_actual, y_actual, x_test, y_test_pred, x_future, y_future, last_date, mape):
        fig_ai = go.Figure()
        fig_ai.add_trace(go.Scatter(x=x_actual, y=y_actual, name="실제", line=dict(color="#FF9F0A", width=2), hovertemplate="실제: %{y:,.0f}원<extra></extra>"))
        fig_ai.add_trace(go.Scatter(x=x_test, y=y_test_pred, name="테스트예측", line=dict(color="#BF5AF2", width=1.5, dash="dot"), hovertemplate="테스트: %{y:,.0f}원<extra></extra>"))
        fig_ai.add_trace(go.Scatter(x=x_future, y=y_future, name="미래예측", line=dict(color="#30D158", width=2.5), hovertemplate="예측: %{y:,.0f}원<extra></extra>"))
        fig_ai.add_trace(go.Scatter(x=[last_date, last_date], y=[min(y_actual)*0.95, max(y_actual)*1.05], mode="lines", line=dict(color="gray", dash="dash", width=1), name="예측시작", hoverinfo="skip"))
        fig_ai.update_layout(
            title=f"{title} | 오차율: {mape:.2f}%",
            template="plotly_dark", height=380,
            hovermode="x unified",
            xaxis=dict(showspikes=True, spikemode="across+toaxis", spikecolor="#aaaaaa", spikethickness=1, spikedash="solid"),
            margin=dict(l=10, r=10, t=50, b=10)
        )
        return fig_ai

    # ── XGBoost (급등락 가중치 적용)
    if show_xgb:
        with cols[ai_cols.index("XGBoost")]:
            with st.spinner("XGBoost 예측 중..."):
                try:
                    from xgboost import XGBRegressor
                    from sklearn.metrics import mean_absolute_percentage_error

                    fcols  = ["종가","거래량","RSI","MACD","Signal","MA5","MA20","MA60"]
                    df_xgb = df[fcols].dropna().copy()
                    W = 5

                    rows = []
                    for i in range(W, len(df_xgb)):
                        row = {f"{c}_lag{d+1}": df_xgb.iloc[i-W+d][c] for d in range(W) for c in fcols}
                        row["target"] = df_xgb.iloc[i]["종가"]
                        rows.append(row)

                    ml = pd.DataFrame(rows)
                    X, y = ml.drop("target", axis=1), ml["target"]
                    sp   = int(len(X) * 0.8)

                    # ✅ 급등락 가중치 계산
                    # 전일 대비 변동률 계산
                    returns    = df_xgb["종가"].pct_change().abs().fillna(0)
                    # 변동률 기반 샘플 가중치 (급등락일수록 높은 가중치)
                    # rolling 평균 대비 몇 배인지로 정규화
                    mean_vol   = returns.rolling(20).mean().fillna(returns.mean())
                    # 가중치 = 변동률 / 평균변동률 (최소 1.0, 최대 5.0 클리핑)
                    weights_raw = (returns / mean_vol.replace(0, 1e-6)).clip(1.0, 3.0)
                    # 학습 데이터에 맞게 슬라이싱
                    sample_weights = weights_raw.iloc[W:].values[:sp]

                    mdl = XGBRegressor(
                        n_estimators=500,
                        learning_rate=0.05,
                        max_depth=4,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        min_child_weight=3,
                        random_state=42,
                        verbosity=0
                    )
                    mdl.fit(X.iloc[:sp], y.iloc[:sp], sample_weight=sample_weights)
                    yp   = mdl.predict(X.iloc[sp:])
                    mape = mean_absolute_percentage_error(y.iloc[sp:], yp) * 100

                    # 미래 예측
                    cur = df_xgb.tail(W).values.copy()
                    std = df_xgb["종가"].pct_change().std()
                    fp  = []
                    for _ in range(predict_days):
                        row  = {f"{c}_lag{d+1}": cur[d][ci] for d in range(W) for ci, c in enumerate(fcols)}
                        pred = mdl.predict(pd.DataFrame([row]))[0]
                        pred = pred + pred * std * np.random.randn()
                        fp.append(pred)
                        nr = cur[-1].copy(); nr[0] = pred
                        cur = np.vstack([cur[1:], nr])

                    ld = pd.to_datetime(df.index[-1])
                    fd = pd.bdate_range(start=ld + timedelta(days=1), periods=predict_days)
                    td = df_xgb.index[sp + W:]

                    fig_xgb = make_ai_chart("XGBoost (급등락 가중치)", df_xgb.index, df_xgb["종가"].values, td, yp, fd, fp, ld, mape)
                    st.plotly_chart(fig_xgb, use_container_width=True)
                    xc = (fp[-1] - df_xgb["종가"].iloc[-1]) / df_xgb["종가"].iloc[-1] * 100
                    st.metric(f"XGBoost {predict_days}일 후", f"{fp[-1]:,.0f}원", f"{xc:+.2f}%")
                except ImportError:
                    st.warning("pip install xgboost scikit-learn")

    # ── Prophet
    if show_prophet:
        with cols[ai_cols.index("Prophet")]:
            with st.spinner("Prophet 예측 중..."):
                try:
                    from prophet import Prophet
                    pdf = df[["종가"]].reset_index(); pdf.columns = ["ds","y"]; pdf["ds"] = pd.to_datetime(pdf["ds"])
                    m   = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True, changepoint_prior_scale=0.05)
                    m.fit(pdf)
                    fut = m.make_future_dataframe(periods=predict_days, freq="B")
                    fc  = m.predict(fut)
                    fig_p = go.Figure()
                    fig_p.add_trace(go.Scatter(x=pdf["ds"], y=pdf["y"], name="실제", line=dict(color="#FF9F0A", width=2), hovertemplate="실제: %{y:,.0f}원<extra></extra>"))
                    fig_p.add_trace(go.Scatter(x=fc["ds"], y=fc["yhat"], name="예측", line=dict(color="#30D158", width=2, dash="dot"), hovertemplate="예측: %{y:,.0f}원<extra></extra>"))
                    fig_p.add_trace(go.Scatter(x=fc["ds"].tolist()+fc["ds"].tolist()[::-1], y=fc["yhat_upper"].tolist()+fc["yhat_lower"].tolist()[::-1], fill="toself", fillcolor="rgba(48,209,88,0.15)", line=dict(color="rgba(255,255,255,0)"), name="예측구간", hovertemplate="예측구간<extra></extra>"))
                    fig_p.add_trace(go.Scatter(x=[pdf["ds"].max(), pdf["ds"].max()], y=[float(pdf["y"].min()), float(pdf["y"].max())], mode="lines", line=dict(color="gray", dash="dash", width=1), name="예측시작", hoverinfo="skip"))
                    fig_p.update_layout(title=f"Prophet ({predict_days}일)", template="plotly_dark", height=380, hovermode="x unified", xaxis=dict(showspikes=True, spikemode="across+toaxis", spikecolor="#aaaaaa", spikethickness=1, spikedash="solid"), margin=dict(l=10, r=10, t=50, b=10))
                    st.plotly_chart(fig_p, use_container_width=True)
                    pl  = fc[fc["ds"] > pdf["ds"].max()]["yhat"].iloc[-1]
                    pc2 = (pl - pdf["y"].iloc[-1]) / pdf["y"].iloc[-1] * 100
                    st.metric(f"Prophet {predict_days}일 후", f"{pl:,.0f}원", f"{pc2:+.2f}%")
                except ImportError:
                    st.warning("pip install prophet")

    # ── LSTM
    if show_lstm:
       with cols[ai_cols.index("LSTM")]:
        st.markdown("""
            <div style='background:#1e1e2e;border:1px solid #555;border-radius:10px;
            padding:20px;text-align:center;height:380px;display:flex;
            flex-direction:column;justify-content:center;'>
                <div style='font-size:40px;margin-bottom:16px;'>🤖</div>
                <div style='font-size:16px;color:#aaa;margin-bottom:8px;'>LSTM 예측</div>
                <div style='font-size:13px;color:#666;'>
                    Cloud 환경에서는 미지원이에요.<br>
                    로컬 PC에서 실행하면 사용 가능해요!
                </div>
            </div>
        """, unsafe_allow_html=True)

    # 최근 데이터 테이블
    st.markdown("---")
    st.subheader("📄 최근 데이터")
    disp = df[["시가","고가","저가","종가","거래량","등락률","RSI","MACD"]].tail(20).copy().iloc[::-1]
    disp.index = disp.index.strftime("%Y-%m-%d")
    disp["등락률"] = disp["등락률"].map("{:.2f}%".format)
    disp["RSI"]    = disp["RSI"].map("{:.1f}".format)
    disp["MACD"]   = disp["MACD"].map("{:.1f}".format)
    st.dataframe(disp, use_container_width=True)

    # ──────────────────────────────────────────
    # 📰 최근 뉴스
    # ──────────────────────────────────────────
    st.markdown("---")
    st.subheader(f"📰 {name} 최근 뉴스")

    with st.spinner("뉴스 수집 중..."):
        news_list = load_news(name, ticker)

    if news_list:
        for news in news_list[:10]:
            st.markdown(
                f"""
                <div style='
                    background:#1e1e2e;
                    border-left:3px solid #30D158;
                    border-radius:8px;
                    padding:12px 16px;
                    margin-bottom:10px;
                '>
                    <a href='{news["링크"]}' target='_blank' style='
                        color:#ffffff;font-size:14px;
                        font-weight:bold;text-decoration:none;
                    '>📌 {news["제목"]}</a>
                    <div style='font-size:12px;color:#888;margin-top:6px;'>
                        {news["출처"]} &nbsp;|&nbsp; {news["날짜"]}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
    else:
        st.info("뉴스를 찾지 못했어요. 잠시 후 다시 시도해주세요.")

else:
    st.info("👈 왼쪽 사이드바에서 종목을 선택하고 '분석 시작' 버튼을 눌러주세요!")
