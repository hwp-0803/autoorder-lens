"""AutoOrder Lens | business-facing demand intelligence dashboard.

Forecasting uses FRED/Census aggregate nominal orders. This is not a SKU,
unit-sales, inventory-optimization, or independently validated production system.
"""
from __future__ import annotations

from html import escape
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from fred_data import DOWNLOAD_URL, fetch_fred_orders
from forecast_engine import forecast_current, load_backtest, load_orders
from monitoring_service import build_alert_monitor
from evaluation_engine import evaluate_short, evaluate_long
from business_insights import market_summary, scenario_table, market_interpretation
from market_context import CONTEXT_SOURCES, fetch_indicator, economic_context
from data_quality import audit_availability
from validation_integrity import check_backtest_actuals
from reporting import make_performance_report, validation_manifest
from policy_events import (CURATED_EVENTS, before_after_yoy, before_after_ape,
                           event_table, parse_event_csv)

ROOT = Path(__file__).resolve().parent
NAVY = "#182C46"
BLUE = "#3B66D9"
TEAL = "#178C84"
ORANGE = "#CF863B"
RED = "#C45158"
MUTED = "#78869B"
GRID = "#E9EDF4"

st.set_page_config(
    page_title="AutoOrder Lens | Demand Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
:root { --ink:#182C46;--muted:#74839A;--blue:#3B66D9;--line:#E7EBF2;--bg:#F7F9FC; }
html, body, [class*="css"], [data-testid="stAppViewContainer"] {
    font-family: "Pretendard", "Malgun Gothic", "Apple SD Gothic Neo", system-ui, sans-serif;
}
.stApp {background:var(--bg);color:var(--ink)}
.block-container {max-width:1570px;padding:2.65rem 3.25rem 5rem;}
[data-testid="stSidebar"] {background:#FFFFFF;border-right:1px solid #E6EBF2;}
[data-testid="stSidebar"] .block-container {padding:1.4rem 1.35rem 4.5rem;}
[data-testid="stSidebarUserContent"] {overflow-y:auto!important;padding-bottom:2rem;}
[data-testid="stSidebar"] {scrollbar-width:thin;}
[data-testid="stSidebar"] .stCaption, [data-testid="stSidebar"] [data-testid="stCaptionContainer"] {line-height:1.65;}
[data-testid="stSidebar"] .panel-body {font-size:.84rem;line-height:1.8;}
/* The side rail must remain usable at standard laptop display heights. */
[data-testid="stSidebar"] .stRadio label {font-size:.89rem;}
header[data-testid="stHeader"] {background:transparent;}
[data-testid="stToolbar"] {visibility:visible;}
[data-testid="stSidebarCollapsedControl"], [data-testid="stSidebarCollapseButton"] { visibility:visible!important;opacity:1!important;pointer-events:auto!important;z-index:999999!important; }
h1,h2,h3,h4 {color:var(--ink);letter-spacing:-.038em;}
h1 {font-weight:760 !important}
p {line-height:1.68;}
[data-testid="stTabs"] [role="tablist"] {gap:2.0rem;border-bottom:1px solid #DEE5EF;margin-top:12px;}
[data-testid="stTabs"] [role="tab"] {font-weight:650;font-size:.96rem;padding:.72rem .05rem .86rem;}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {color:var(--blue);}
[data-testid="stDataFrame"] {border:1px solid var(--line);border-radius:12px;overflow:hidden;}
[data-testid="stDownloadButton"] button, .stButton button {border-radius:9px;font-weight:630;}
div[data-testid="stExpander"] {border:1px solid var(--line);border-radius:12px;background:#FFFFFF;}
div[data-testid="stRadio"] label {font-size:.88rem;}
.brand {display:flex;align-items:center;gap:10px;margin:0 0 .7rem;}
.brand-emblem {display:flex;align-items:center;justify-content:center;width:33px;height:33px;background:#1E3A5F;
  color:white;border-radius:9px;font-size:18px;font-weight:700;}
.brand-name {font-size:1.03rem;letter-spacing:.04em;font-weight:800;color:#173351;}
.brand-sub {font-size:.68rem;font-weight:650;color:#8894A4;letter-spacing:.12em;margin-top:3px;}
.side-cap {font-size:.67rem;color:#8694A8;font-weight:700;letter-spacing:.12em;text-transform:uppercase;margin:1.1rem 0 .45rem;}
.eyebrow {display:block;font-size:.71rem;letter-spacing:.15em;font-weight:750;color:#4670B6;
 line-height:1.7;white-space:normal;overflow:visible;margin:0 0 .62rem;}
.page-title {font-size:2.23rem;font-weight:790;color:#172D49;letter-spacing:-.05em;line-height:1.35;margin:.02rem 0 .36rem;}
.page-sub {font-size:.95rem;color:#718198;line-height:1.65;margin:0 0 1.2rem;}
.tagrow {display:flex;gap:8px;flex-wrap:wrap;margin:0 0 1.1rem;}
.tag {font-size:.78rem;font-weight:700;padding:5px 10px;border:1px solid #DFE7F1;border-radius:7px;
 background:#fff;color:#546985;letter-spacing:.01em;}
.status-strip {background:#EFF4FB;border:1px solid #DCE6F3;color:#3B597F;padding:11px 15px;
 font-size:.84rem;line-height:1.65;border-radius:10px;margin:0 0 20px;}
.status-strip b {color:#254A7A;}
.status-strip.warn {background:#FFF7EA;border-color:#F1DFC0;color:#98682E;}
.metrics-grid {display:grid;grid-template-columns:repeat(4,1fr);gap:13px;margin:0 0 16px;}
.kpi {background:white;border:1px solid #E7EBF2;border-radius:14px;min-height:119px;
 padding:17px 19px;box-shadow:0 2px 12px rgba(18,38,70,.025);}
.kpi .label {font-size:.78rem;color:#8290A3;font-weight:600;line-height:1.5;}
.kpi .value {font-size:1.61rem;color:#182D49;font-weight:780;letter-spacing:-.045em;
 margin:8px 0 6px;overflow-wrap:anywhere;}
.kpi .support {font-size:.79rem;color:#8A97A8;line-height:1.5;}
.kpi .value.emphasis {color:#315DC5;}
.section-head {font-size:1.28rem;font-weight:760;color:#182D49;margin:1.35rem 0 .22rem;letter-spacing:-.035em;}
.section-sub {color:#7B899C;font-size:.85rem;line-height:1.6;margin:0 0 1.05rem;}
.panel {background:#fff;border:1px solid #E7EBF2;border-radius:14px;padding:20px 22px;margin:0 0 16px;}
.panel-title {font-size:.84rem;font-weight:800;color:#233954;margin:0 0 .7rem;}
.panel-body {font-size:.91rem;color:#68788F;line-height:1.78;}
.panel-body b {color:#243D5D;}
.status-chip {display:inline-flex;align-items:center;gap:5px;border-radius:7px;padding:5px 9px;font-size:.75rem;font-weight:750;}
.status-chip.ok {background:#E8F5F2;color:#17776D;}
.status-chip.wait {background:#F0F3F7;color:#61738B;}
.status-chip.watch {background:#FFF3E1;color:#A46B21;}
.status-chip.high {background:#FCECEF;color:#B53F4C;}
.stat-row {display:flex;justify-content:space-between;align-items:center;border-bottom:1px solid #EEF1F5;
 padding:11px 0;font-size:.85rem;color:#687890;gap:15px;}
.stat-row b {color:#263C59;white-space:nowrap;}
.stat-row:last-child {border-bottom:none;}
.usage-note {padding:11px 15px;border:1px solid #E2EAF6;border-radius:10px;background:#F0F5FC;
 color:#365679;font-size:.83rem;line-height:1.7;margin:.65rem 0 .9rem;}
.usage-note b {color:#244A7A;}
.usage-hint {font-size:.8rem;color:#697A90;line-height:1.6;margin:.25rem 0 .8rem;}
.footnote {font-size:.82rem;color:#8492A4;line-height:1.75;margin:.75rem 0 1.1rem;}
.detail-bar {padding:12px 15px;border-radius:10px;border-left:3px solid #4679CB;background:#EFF5FE;
 font-size:.84rem;color:#3A5E8F;line-height:1.75;margin:.5rem 0 1rem;}
.detail-bar.amber {border-left-color:#CD8E42;background:#FFF7EA;color:#895F2B;}
.detail-bar.red {border-left-color:#C45158;background:#FFF0F1;color:#93404B;}
.section-divider {height:1px;background:#E7EBF2;margin:1.5rem 0;}
.footerline {margin-top:2.5rem;padding-top:1rem;border-top:1px solid #E5E9F0;
 color:#8390A1;font-size:.72rem;line-height:1.7;}
@media (max-width: 1200px) {
 .block-container {padding:2rem 1.45rem 4rem;}
 .metrics-grid {grid-template-columns:repeat(2,1fr)}
 .kpi .value {font-size:1.43rem;}
}
@media (max-width: 680px) {
 .page-title{font-size:1.65rem;}
 .metrics-grid{grid-template-columns:1fr 1fr;gap:8px}
 .kpi{padding:13px 12px;min-height:100px}
 .kpi .value{font-size:1.24rem}
}

.insight-label {font-size:.73rem;text-transform:uppercase;letter-spacing:.075em;font-weight:780;color:#58759E;margin-bottom:.55rem;}
.insight-title {font-size:1.0rem;line-height:1.5;font-weight:790;color:#203858;margin-bottom:.45rem;}
.insight-body {font-size:.88rem;line-height:1.75;color:#60748E;}
.insight-card {background:#FFFFFF;border:1px solid #E5EBF3;border-radius:13px;padding:18px 18px 16px;min-height:175px;}
.trend-breakdown {font-size:.83rem;line-height:1.8;color:#62758E;}
.trend-breakdown strong {color:#203B5B;}
@media (max-width: 850px) {.insight-card {min-height:auto;}}
.context-kpi {background:#fff;border:1px solid #E7EBF2;border-radius:13px;padding:16px;}
.context-kpi .title {color:#74839A;font-weight:650;font-size:.82rem;line-height:1.5;}
.context-kpi .number {font-size:1.5rem;font-weight:760;letter-spacing:-.03em;color:#203857;margin:.42rem 0;}
.context-kpi .desc {font-size:.79rem;color:#7D8BA0;line-height:1.5;}
.warning-note {font-size:.82rem;color:#7C6A4C;line-height:1.75;margin:9px 0 15px;}

</style>
""", unsafe_allow_html=True)


def find_file(names: list[str]):
    for parent in (ROOT, ROOT.parent, Path.cwd()):
        for name in names:
            path = parent / name
            if path.is_file():
                return path
    return None


def choose_source(upload, names):
    if upload is not None:
        return upload, f"업로드: {upload.name}"
    path = find_file(names)
    return (path, f"로컬 파일: {path.name}") if path else (None, None)


def series_hash(y):
    return hashlib.sha256(pd.util.hash_pandas_object(y, index=True).values.tobytes()).hexdigest()


def fmt_amount(million: float, digits=1):
    if not math.isfinite(float(million)):
        return "—"
    return f"{float(million) / 100:,.{digits}f}억 달러"


def fmt_signed(value, decimals=1):
    return "—" if value is None or not np.isfinite(value) else f"{value:+,.{decimals}f}%"


def html_card(label, value, support, emphasis=False):
    return (f'<div class="kpi"><div class="label">{escape(label)}</div>'
            f'<div class="value{" emphasis" if emphasis else ""}">{escape(value)}</div>'
            f'<div class="support">{escape(support)}</div></div>')


def info_box(title, body):
    return (f'<div class="panel"><div class="panel-title">{escape(title)}</div>'
            f'<div class="panel-body">{body}</div></div>')


def status_chip(text, kind='wait'):
    return f'<span class="status-chip {kind}">{escape(text)}</span>'


def section(title, description=None):
    st.markdown(f'<div class="section-head">{escape(title)}</div>', unsafe_allow_html=True)
    if description:
        st.markdown(f'<div class="section-sub">{escape(description)}</div>', unsafe_allow_html=True)


def csv_download(label, df, filename, key):
    st.download_button(label, df.to_csv(index=False).encode('utf-8-sig'),
                       file_name=filename, mime='text/csv', use_container_width=False, key=key)


def base_chart(height=410):
    fig = go.Figure()
    fig.update_layout(
        height=height, margin=dict(l=4,r=14,t=45,b=18),
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Malgun Gothic, Arial, sans-serif',color='#67778D',size=12),
        hovermode='x unified',
        xaxis=dict(showgrid=False,zeroline=False,linecolor=GRID,tickformat='%Y-%m',
                   ticks='outside',tickfont=dict(size=11)),
        yaxis=dict(gridcolor=GRID,zeroline=False,tickformat=',.0f',tickfont=dict(size=11)),
        legend=dict(orientation='h',y=1.11,yanchor='bottom',x=0,xanchor='left',
                    font=dict(size=11),itemwidth=35),
        bargap=.28,
    )
    return fig


def forecast_chart(y, bundle, interval, history=36):
    fig = base_chart(height=440)
    forecast = bundle.long
    past = y.iloc[-int(history):]
    fig.add_trace(go.Scatter(x=forecast.index,y=forecast[f'upper_{interval}'],
        mode='lines',line=dict(width=0),showlegend=False,hoverinfo='skip'))
    fig.add_trace(go.Scatter(x=forecast.index,y=forecast[f'lower_{interval}'],
        mode='lines',fill='tonexty',fillcolor='rgba(62,102,204,.105)',
        line=dict(width=0),name=f'{interval}% 예측구간',hoverinfo='skip'))
    fig.add_trace(go.Scatter(x=past.index,y=past.values,name='실제 신규주문액',
        mode='lines',line=dict(color=NAVY,width=2.7),
        hovertemplate='%{x|%Y-%m}<br>%{y:,.0f} 백만 달러<extra></extra>'))
    fig.add_trace(go.Scatter(x=[past.index[-1],*forecast.index],
        y=[float(past.iloc[-1]),*forecast.forecast.values],name='12개월 전망 · AR6',
        mode='lines+markers',line=dict(color=BLUE,width=2.5,dash='dash'),
        marker=dict(size=5),hovertemplate='%{x|%Y-%m}<br>%{y:,.0f} 백만 달러<extra></extra>'))
    fig.add_trace(go.Scatter(x=bundle.short.index,y=bundle.short.forecast.values,
        name='1개월 전망 · AR3',mode='markers',marker=dict(color=TEAL,size=13,symbol='diamond',
        line=dict(color='white',width=1.8)),hovertemplate='%{x|%Y-%m}<br>%{y:,.0f} 백만 달러<extra></extra>'))
    fig.add_vline(x=past.index[-1].timestamp()*1000,line_dash='dot',line_color='#B7C3D1')
    fig.update_yaxes(title='백만 달러',title_font=dict(size=11,color=MUTED))
    return fig


def alert_chart(alerts):
    fig=base_chart(340)
    fig.add_hline(y=0,line_color='#BAC5D2',line_width=1)
    fig.add_trace(go.Bar(x=alerts.Month,y=alerts.bias_pct,name='월별 예측편향',
        marker_color=[BLUE if n>=0 else '#E1A25C' for n in alerts.bias_pct],
        hovertemplate='%{x|%Y-%m}<br>%{y:+.2f}%<extra></extra>'))
    for lvl,c in [('주의',ORANGE),('강화 경고',RED)]:
        d=alerts.loc[alerts.alert_level.eq(lvl)]
        if not d.empty:
            fig.add_trace(go.Scatter(x=d.Month,y=d.bias_pct,mode='markers',name=lvl,
                marker=dict(color=c,size=12,symbol='diamond',line=dict(color='#FFFFFF',width=1.5)),
                hovertemplate='%{x|%Y-%m}<br>%{y:+.2f}%<extra></extra>'))
    fig.update_yaxes(title='편향 (%)',title_font=dict(size=11,color=MUTED))
    return fig


def performance_bars(table, historic_col, recent_col):
    """Grouped model-error comparison; periods explicitly separated on x-axis."""
    names=list(table.index)
    fig=base_chart(290)
    fig.add_trace(go.Bar(x=names,y=table[historic_col].values,name='과거 평가',
        marker_color=BLUE,text=[f'{x:.2f}%' if pd.notna(x) else '' for x in table[historic_col]],
        textposition='outside',textfont=dict(size=11)))
    if recent_col in table.columns:
        vals=table[recent_col].values
        fig.add_trace(go.Bar(x=names,y=vals,name='2026년 평가',
            marker_color=TEAL,text=[f'{x:.2f}%' if pd.notna(x) else '' for x in vals],
            textposition='outside',textfont=dict(size=11)))
    fig.update_layout(barmode='group',margin=dict(l=2,r=10,t=30,b=12),
                      legend=dict(orientation='h',x=0,y=1.09),
                      yaxis=dict(gridcolor=GRID,zeroline=False,range=[0,None],title='MAPE (%)'))
    return fig


def alert_donut(alerts):
    nums = [int(alerts.alert_level.eq(label).sum()) for label in ('정상','주의','강화 경고')]
    fig=go.Figure(go.Pie(labels=['조건 미충족','주의','강화 경고'],values=nums,
        hole=.72,sort=False,textinfo='none',direction='clockwise',
        marker=dict(colors=['#D8E0ED','#DEAA69','#C45158'],line=dict(color='white',width=2)),
        hovertemplate='%{label} %{value}개월 (%{percent})<extra></extra>'))
    fig.update_layout(height=285,paper_bgcolor='rgba(0,0,0,0)',margin=dict(l=0,r=0,t=6,b=32),
        legend=dict(orientation='h',x=.5,xanchor='center',y=-.08,yanchor='top',font=dict(size=11)),
        annotations=[dict(x=.5,y=.51,text=f'<b>{nums[1]+nums[2]} / {sum(nums)}</b>',
                          showarrow=False,font=dict(size=23,color=NAVY)),
                     dict(x=.5,y=.40,text='경고 발생 월',showarrow=False,
                          font=dict(size=11,color=MUTED))])
    return fig


def scenario_chart(scenario_data: pd.DataFrame):
    """Compare baseline and user-adjusted monthly values in 100 million USD."""
    fig = base_chart(height=320)
    months = scenario_data['Month']
    fig.add_trace(go.Scatter(
        x=months, y=scenario_data['baseline_million']/100,
        name='모델 기본 전망', mode='lines+markers',
        line=dict(color=NAVY, width=2.5), marker=dict(size=6),
        hovertemplate='%{x|%Y-%m}<br>%{y:,.1f}억 달러<extra></extra>'
    ))
    fig.add_trace(go.Scatter(
        x=months, y=scenario_data['scenario_million']/100,
        name='사용자 가정 적용', mode='lines+markers',
        line=dict(color=BLUE, width=2.5, dash='dash'), marker=dict(size=6),
        hovertemplate='%{x|%Y-%m}<br>%{y:,.1f}억 달러<extra></extra>'
    ))
    fig.update_yaxes(title='억 달러', title_font=dict(size=11,color=MUTED))
    return fig


def comparison_chart(frame: pd.DataFrame):
    """Different coverage metrics: compare only their YOY rates, not index levels."""
    fig = base_chart(height=355)
    series_map = [
        ("orders_yoy", "명목 신규주문액", NAVY),
        ("PPI_yoy", "부품 생산자물가(PPI)", ORANGE),
        ("IP_yoy", "자동차·부품 생산지수", TEAL),
    ]
    for col, name, color in series_map:
        fig.add_trace(go.Scatter(x=frame.index, y=frame[col], name=name,
            mode="lines+markers", line=dict(color=color, width=2.4),
            marker=dict(size=4), hovertemplate="%{x|%Y-%m}<br>%{y:+.2f}%<extra>"+name+"</extra>"))
    fig.add_hline(y=0, line_color="#B6C2D3", line_width=1)
    fig.update_yaxes(title="전년 동월 대비 (%)", title_font=dict(size=11,color=MUTED))
    return fig


def context_kpi(title: str, value: str, detail: str):
    return (f'<div class="context-kpi"><div class="title">{escape(title)}</div>'
            f'<div class="number">{escape(value)}</div><div class="desc">{escape(detail)}</div></div>')


def insight_card(category: str, title: str, text: str):
    return (
        '<div class="insight-card">'
        f'<div class="insight-label">{escape(category)}</div>'
        f'<div class="insight-title">{escape(title)}</div>'
        f'<div class="insight-body">{escape(text)}</div>'
        '</div>'
    )


@st.cache_data(ttl=900, show_spinner='공식 시계열을 조회하고 있습니다.')
def fred_cached():
    return fetch_fred_orders(), datetime.now(timezone.utc)


@st.cache_data(ttl=900, show_spinner=False)
def context_series_cached():
    """Fetch independent indicators concurrently; preserve partial failures."""
    series, failures = {}, {}
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {pool.submit(fetch_indicator, meta['id'], timeout=11): alias
                   for alias, meta in CONTEXT_SOURCES.items()}
        for future in as_completed(futures):
            alias = futures[future]
            try:
                series[alias] = future.result()
            except Exception as exc:
                failures[alias] = str(exc)
    return series, failures


@st.cache_data(show_spinner='예측모델을 계산하고 있습니다.')
def forecast_cached(y):
    return forecast_current(y)


@st.cache_data(ttl=3600, show_spinner='과거 월별 예측오차와 경고 이력을 자동 분석하고 있습니다.')
def monitor_cached(y):
    """Automatically backtest and alert on the exact current source snapshot.

    Streamlit's cache key includes the entire Pandas Series (dates and values),
    so newly published or revised observations trigger a fresh assessment.
    """
    return build_alert_monitor(y)


@st.cache_data(show_spinner='모델을 동일한 기준으로 재평가하고 있습니다.')
def short_cached(y):
    return evaluate_short(y)


@st.cache_data(show_spinner='12개월 시계열 검증을 수행하고 있습니다.')
def long_cached(y):
    return evaluate_long(y)


# Sidebar / data inputs
with st.sidebar:
    st.markdown('''<div class="brand"><div class="brand-emblem">◈</div>
      <div><div class="brand-name">AUTOORDER LENS</div><div class="brand-sub">DEMAND INTELLIGENCE</div></div></div>''',
      unsafe_allow_html=True)
    st.markdown('<div class="side-cap">DATA SOURCE · 원천자료</div>',unsafe_allow_html=True)
    mode=st.radio('주문액 데이터', ['공식 시계열 자동 조회','로컬 CSV 이용'],label_visibility='collapsed')
    if st.button('최신 데이터 다시 확인',use_container_width=True,
        disabled=mode!='공식 시계열 자동 조회'):
        fred_cached.clear()
        context_series_cached.clear()
        st.rerun()
    with st.expander('데이터 파일 연결',expanded=False):
        uploaded=st.file_uploader('신규주문액 CSV',type=['csv'],
            help="date/Month 및 orders(백만 달러) 열이 필요합니다.")
        st.caption('연결하지 않으면 자동으로 FRED 또는 프로젝트 내 CSV를 사용합니다.')
    with st.expander('자동 모니터링 안내'):
        st.caption('예측오차와 경고 이력은 원천자료 변경 시 자동으로 계산됩니다. 별도 경고 CSV는 필요하지 않습니다. FRED 원천자료는 최신 데이터 다시 확인 버튼으로 갱신할 수 있습니다.')
    with st.expander('과거 모델 성능 자료 연결 (선택)',expanded=False):
        short_file=st.file_uploader('단기 예측 검증 결과',type=['csv'],key='upload_short')
        long_file=st.file_uploader('장기 예측 검증 결과',type=['csv'],key='upload_long')
        older_file=st.file_uploader('과거 추가 검증 결과',type=['csv'],key='upload_older')
    st.markdown('<div class="side-cap">MODEL POLICY · 운영 정책</div>',unsafe_allow_html=True)
    st.markdown('<div class="panel-body"><b>단기 · SARIMA AR3</b><br>1개월 앞 전망 / 연 1회 계수 학습<br><br>'
        '<b>장기 · SARIMA AR6</b><br>12개월 연속 전망 / 전망 시점 학습</div>',unsafe_allow_html=True)
    st.divider()
    st.caption('데이터: 미국 Census M3 · FRED UBTPNO\n\n단위: 명목 백만 달러 · 비계절조정')
    with st.expander('지표 범위'):
        st.caption('미국 산업 전체의 명목 신규주문액이며, 특정 기업의 발주 수량·판매량·재고량은 아닙니다.')

# Load real data only
source, source_desc=choose_source(uploaded,['data/fred_UBTPNO.csv','orders_backlog_inventory.csv','orders_clean.csv'])
y=None
fallback_error=None
loaded=''
source_checked_at_utc=None
if mode=='공식 시계열 자동 조회':
    try:
        y, source_checked_at_utc=fred_cached()
        loaded='FRED UBTPNO · 최신 공개자료 조회'
    except Exception as exc:
        fallback_error=str(exc)
if y is None and source is not None:
    try:
        y=load_orders(source)
        loaded=source_desc
    except Exception as exc:
        st.error(f'데이터 형식을 확인해 주십시오. 상세 오류: {exc}')
        st.stop()
if y is None:
    st.error('데이터를 불러올 수 없습니다. 네트워크 연결을 확인하거나 신규주문액 CSV를 업로드해 주십시오.')
    if fallback_error:
        with st.expander('연결 오류 상세'):
            st.code(fallback_error)
    st.stop()

# Align model training period with experiments
train_start=pd.Timestamp('2005-02-01')
if y.index.min()<train_start:
    y=y.loc[train_start:]
if y.index.min()>train_start:
    st.warning(f'현 데이터 시작월은 {y.index.min():%Y-%m}입니다. 기존 백테스트 조건과 다르므로 정확도 비교에 유의해 주십시오.')

fingerprint=series_hash(y)
last=y.index[-1]
next_month=last+pd.offsets.MonthBegin(1)
this_month=pd.Timestamp.today().to_period('M').to_timestamp()
report_pending=next_month<this_month
if fallback_error:
    st.warning(f'공식 FRED 연결 실패 · 저장 자료 사용 중({escape(loaded)}). 최신 공식 실적이 아닐 수 있습니다. 자료 기준월을 확인해 주세요.')

try:
    bundle=forecast_cached(y)
except Exception as exc:
    st.error(f'예측모델을 계산하지 못했습니다. 입력자료 및 Python 환경을 확인해 주십시오. ({exc})')
    st.stop()
if not bundle.short_converged or not bundle.long_converged:
    st.warning('일부 모델의 최적화가 수렴하지 않았습니다. 해당 전망을 의사결정에 사용하기 전에 모델 상태를 점검해 주십시오.')

try:
    demand = market_summary(y, bundle.long)
except Exception as exc:
    st.error(f'수요 전망의 비교기간을 계산하지 못했습니다. 자료 연속성과 예측기간을 확인해 주십시오. ({exc})')
    st.stop()

# The monitoring calculation is an automatic part of the product data pipeline.
# No manual CSV or user-triggered backtest is required.
alerts = None
monitor_error = None
try:
    alerts = monitor_cached(y)
except Exception as exc:
    # Never claim the latest period is normal when assessment failed.
    monitor_error = str(exc)

alert_current = bool(alerts is not None and len(alerts) and alerts.Month.max() == last)
alert_last = alerts.iloc[-1] if alert_current else None
if alert_current:
    raw_status = alert_last.alert_level
    status = '경고 조건 미충족' if raw_status == '정상' else f'{alert_last.alert_direction} · {raw_status}'
    status_kind = ('ok' if raw_status == '정상' else 'high' if raw_status == '강화 경고' else 'watch')
else:
    status = '모니터링 확인 불가'
    status_kind = 'wait'

# Header / overview KPI
st.markdown('''<span class="eyebrow">MANUFACTURING  /  DEMAND INTELLIGENCE</span>
  <div class="page-title">자동차부품 신규주문액 분석</div>
  <div class="page-sub">미국 제조업 신규주문액 전망, 예측오차 점검 및 운영 의사결정 지원</div>''',unsafe_allow_html=True)
source_checked_local=(pd.Timestamp(source_checked_at_utc).tz_convert('Asia/Seoul').strftime('%Y-%m-%d %H:%M KST') if source_checked_at_utc else None)
st.markdown(f'''<div class="tagrow"><span class="tag">FRED UBTPNO</span>
  <span class="tag">관측 {y.index[0]:%Y-%m} — {last:%Y-%m}</span>
  <span class="tag">월별 · 명목 백만 달러</span><span class="tag">{escape(loaded)}</span><span class="tag">{escape("FRED 최근 조회 · "+source_checked_local) if source_checked_local else "공식 실시간 조회 아님 · 저장 자료"}</span></div>''',
  unsafe_allow_html=True)

if report_pending:
    strip=(f'<div class="status-strip warn"><b>자료 발표 대기</b>  ·  최신 관측은 {last:%Y-%m}입니다. '
           f'{next_month:%Y-%m}은 달력상 이미 지난 월이지만 원자료에 아직 관측치가 없습니다. '
           '아래 수치는 해당 월에 대한 미발표 실적 추정치입니다.</div>')
else:
    strip=(f'<div class="status-strip"><b>데이터 확인</b>  ·  {last:%Y-%m}까지 실적을 반영했습니다. '
           f'다음 추정 대상은 {next_month:%Y-%m}입니다. 출처의 발표 지연 및 과거 수치 개정 가능성이 있습니다.</div>')
st.markdown(strip,unsafe_allow_html=True)

yoy=demand.latest_yoy_pct
kpi_html=''.join([
    html_card('최근 월 신규주문액',fmt_amount(y.iloc[-1]),f'기준월 {last:%Y-%m} · 전년 동월 대비 {fmt_signed(yoy)}'),
    html_card('다음 1개월 추정',fmt_amount(bundle.short.forecast.iloc[0]),f'AR3 · {next_month:%Y-%m}',True),
    html_card('향후 12개월 추정 합계',fmt_amount(bundle.long.forecast.sum(),0),f'AR6 · 직전 12개월 실적 대비 {fmt_signed(demand.next_twelve_yoy_pct)}'),
    html_card('누적 편향 모니터',status,f'실적 기준 {last:%Y-%m}' if alert_current else '계산 상태 점검 필요'),
])
st.markdown(f'<div class="metrics-grid">{kpi_html}</div>',unsafe_allow_html=True)

# Product navigation
overview_tab,forecast_tab,context_tab,policy_tab,alert_tab,model_tab,data_tab=st.tabs([
    '운영 개요','수요 전망 · 시나리오','산업 변화 분석','정책 이벤트','편향 모니터링','모델 성능','데이터 · 운영기준'
])

with overview_tab:
    section('자동 수요 분석 요약',
            f'{last:%Y-%m}까지 확인된 실적과 AR6의 향후 12개월 추정값을 분리하여 해석합니다.')
    assessments = market_interpretation(
        demand,
        str(alert_last.alert_level) if alert_current else None,
        str(alert_last.alert_direction) if alert_current else None,
    )
    label_titles = [
        f'전년 동월 {fmt_signed(demand.latest_yoy_pct)}',
        f'향후 12개월 {fmt_signed(demand.next_twelve_yoy_pct)}',
        status,
    ]
    insight_cols = st.columns(3, gap='small')
    for col, (group, narrative), headline in zip(insight_cols, assessments, label_titles):
        with col:
            st.markdown(insight_card(group, headline, narrative),unsafe_allow_html=True)
    st.caption('자동 분석은 확인된 원천자료와 예측값을 비교한 규칙 기반 요약입니다. 가격과 물량의 기여도를 분리하거나 산업 변화의 원인을 확정하지 않습니다.')

    l,r=st.columns([2.3,1],gap='large')
    with l:
        section('신규주문액 추이 및 향후 12개월 전망',
                '실제 관측값과 AR6 장기 전망을 동일한 단위로 비교합니다.')
        st.plotly_chart(forecast_chart(y,bundle,95,36),use_container_width=True,
                        config={'displayModeBar':False})
    with r:
        section('운영 점검 요약', '현재 데이터에서 확인할 항목')
        st.markdown('<div class="panel">'
             f'<div class="panel-title">최신 관측 요약</div>'
             f'<div class="stat-row"><span>관측 기준월</span><b>{last:%Y-%m}</b></div>'
             f'<div class="stat-row"><span>전년 동월 대비</span><b>{escape(fmt_signed(yoy))}</b></div>'
             f'<div class="stat-row"><span>최근 3개월 전년 동기 대비</span><b>{escape(fmt_signed(demand.recent_three_yoy_pct))}</b></div>'
             f'<div class="stat-row"><span>최신 경고 상태</span>{status_chip(status,status_kind)}</div>'
             f'<div class="stat-row"><span>장기 기준 학습월</span><b>{bundle.long_refit_at:%Y-%m}</b></div>'
             '</div>',unsafe_allow_html=True)
        if demand.latest_yoy_pct > 0:
            order_prompt = '전년 동월 대비 주문액 증가의 지속성과 고객사별 발주 집중 여부를 확인합니다.'
        elif demand.latest_yoy_pct < 0:
            order_prompt = '전년 동월 대비 주문액 감소의 일시성 및 고객사별 주문 변경 여부를 확인합니다.'
        else:
            order_prompt = '최근 주문액 변화가 일시적인지 확인합니다.'
        latest_policy_event = max((e for e in CURATED_EVENTS if pd.Timestamp(e.date) <= last),
                                  key=lambda e: e.date, default=None)
        if latest_policy_event is not None:
            st.markdown(info_box('최근 기록된 정책·통상 이벤트',
                f'<b>{latest_policy_event.date:%Y-%m-%d}</b> · '
                f'{escape(latest_policy_event.title)}<br>'
                '공식 출처에 근거한 정적 사건 이력입니다. 새 정책을 실시간 감시하거나 예측에 자동 반영하지 않습니다. 자세한 정보는 「정책 이벤트」 탭을 확인하십시오.'), unsafe_allow_html=True)
        st.markdown(info_box('운영 점검 권고',
            f'<b>발주 검토</b> · {escape(order_prompt)}<br>'
            '<b>가격 확인</b> · 명목 주문액 변화와 부품 생산자물가지수(PPI) 움직임을 구분하여 살펴봅니다.<br>'
            '<b>계획 검토</b> · 자체 발주·출하·재고·생산능력 자료와 비교한 후 업무에 반영합니다.'),
            unsafe_allow_html=True)
    st.markdown('<div class="footnote">예측치는 산업 전체의 명목 신규주문액입니다. 특정 기업의 주문 수량·생산계획·재고 최적화 결과로 해석할 수 없습니다.</div>',
                unsafe_allow_html=True)

with forecast_tab:
    section('향후 수요 전망', '실제 신규주문액과 향후 12개월 전망을 비교합니다. 설정은 그래프 표시와 가정 분석에만 적용됩니다.')
    st.markdown(
        '<div class="panel">'
        '<div class="panel-title">동일 기간 비교 · 산업 신규주문액 (명목)</div>'
        f'<div class="stat-row"><span>직전 12개월 실제 합계</span><b>{fmt_amount(demand.previous_twelve_sum,0)}</b></div>'
        f'<div class="stat-row"><span>향후 12개월 전망 합계 · AR6</span><b>{fmt_amount(demand.next_twelve_sum,0)}</b></div>'
        f'<div class="stat-row"><span>기준 기간 대비 예상 증감률</span><b>{fmt_signed(demand.next_twelve_yoy_pct)}</b></div>'
        '</div>', unsafe_allow_html=True
    )
    ctl1,ctl2=st.columns(2,gap='large')
    with ctl1:
        interval=st.radio(
            '미래 예측구간 · 불확실성 범위', options=[80,95],index=1,horizontal=True,
            format_func=lambda x: f'{x}% · {"좁은 범위" if x==80 else "넓은 범위"}',
            help='예측구간은 장기 AR6 모델이 미래 신규주문액의 불확실성을 표현한 범위입니다. 95%는 80%보다 넓으며, 모델의 정확도가 95%라는 의미가 아닙니다.'
        )
        st.caption('80%는 비교적 좁은 범위, 95%는 더 넓은 범위를 보여줍니다. 모델 적중률이나 보증 수준이 아닙니다.')
    with ctl2:
        hist_months=st.select_slider(
            '과거 관측자료 표시기간', options=[24,36,48,60],value=36,
            format_func=lambda x:f'최근 {x}개월',
            help='그래프에 표시할 과거 실제 신규주문액의 길이입니다. 기간을 변경해도 예측모델의 학습 데이터, 예측값, 향후 12개월 전망 기간은 바뀌지 않습니다.'
        )
        st.caption(f'최근 {hist_months}개월의 실제 기록을 표시합니다. 미래 전망은 항상 12개월입니다.')

    first_forecast=bundle.long.iloc[0]
    forecast_month=bundle.long.index[0]
    st.markdown(
        f'<div class="usage-note"><b>현재 설정의 의미</b> · 그래프 왼쪽에는 최근 {hist_months}개월 실적을 표시합니다. '
        f'오른쪽의 색이 채워진 영역은 <b>{interval}% 모형 예측구간</b>입니다. '
        f'예를 들어 {forecast_month:%Y-%m} 전망은 <b>{fmt_amount(first_forecast.forecast)}</b>이며, '
        f'선택한 구간은 <b>{fmt_amount(first_forecast[f"lower_{interval}"])}~{fmt_amount(first_forecast[f"upper_{interval}"])}</b>입니다. '
        '예측구간은 미래의 확정 범위나 모델의 정확도 점수를 뜻하지 않습니다.</div>',
        unsafe_allow_html=True
    )
    st.plotly_chart(forecast_chart(y,bundle,interval,hist_months),use_container_width=True,
                    config={'displayModeBar':True,'displaylogo':False})
    with st.expander('전년 동월 대비 월별 전망 변화 확인', expanded=False):
        comparison = demand.month_comparison.copy()
        comparison['대상 월'] = comparison['Month'].dt.strftime('%Y-%m')
        monthly_view = comparison[['대상 월','year_ago_actual','forecast','delta_million','yoy_pct']].copy()
        monthly_view.columns = ['대상 월','전년 동월 실적 (백만 달러)','AR6 전망 (백만 달러)','전망 차이 (백만 달러)','전년 동월 대비 전망 (%)']
        st.dataframe(monthly_view.style.format({
            '전년 동월 실적 (백만 달러)':'{:,.0f}',
            'AR6 전망 (백만 달러)':'{:,.0f}',
            '전망 차이 (백만 달러)':'{:+,.0f}',
            '전년 동월 대비 전망 (%)':'{:+.1f}%'
        }), hide_index=True, use_container_width=True)
        st.caption('전년 동월의 확정 실적과 해당 월 AR6 전망값을 비교합니다. 예측값의 증감이며 실제 미래 실적이 아닙니다.')
    with st.expander('예측구간 80%·95%와 표시기간의 차이 알아보기',expanded=False):
        st.markdown(
            '**과거 표시기간**은 과거 관측값을 그래프에서 몇 개월 보여줄지 정합니다. '
            '24개월을 선택하면 최근 2년, 60개월을 선택하면 최근 5년의 실적을 표시하며, **미래 예측값은 바뀌지 않습니다.**\n\n'
            '**80%·95% 예측구간**은 각 미래 월의 주문액이 변동할 수 있는 범위를 모델이 추정한 결과입니다. '
            '일반적으로 95% 구간이 더 넓습니다. 이 숫자는 **예측 정확도 80% 또는 95%를 뜻하지 않습니다.** '
            '모형 가정이 적절하고 구간이 보정되어 있을 때 반복적인 예측의 실제 포함 비율을 해석할 수 있으며, 현재 표시 구간의 실측 보정 여부는 별도로 검증해야 합니다.\n\n'
            '**향후 12개월**은 위 두 설정과 관계없이 고정되어 있습니다. '
            '화면의 음영은 계획상 허용 손실, 안전재고 또는 주문 보장 범위가 아닙니다.'
        )
    st.markdown('<div class="footnote">그래프 단위는 백만 달러입니다. 예측구간은 SARIMA 모형 기반 추정치이며, 별도의 실측 보정이나 예측구간 커버리지 검증으로 확정된 수치가 아닙니다.</div>',
                unsafe_allow_html=True)

    left,right=st.columns([1.6,1],gap='large')
    with left:
        section('월별 전망 명세', '시점별 예측액과 선택한 구간의 하한·상한을 확인할 수 있습니다.')
        tbl=bundle.long.copy()
        tbl.insert(0,'대상 월',tbl.index.strftime('%Y-%m'))
        out=tbl[['대상 월','forecast',f'lower_{interval}',f'upper_{interval}']].copy()
        out.columns=['대상 월','예측액 (백만 달러)','구간 하한','구간 상한']
        st.dataframe(out.style.format({'예측액 (백만 달러)':'{:,.0f}',
                                        '구간 하한':'{:,.0f}','구간 상한':'{:,.0f}'}),
                     use_container_width=True,hide_index=True,height=430)
        dl=tbl.rename(columns={'forecast':'AR6_forecast_million_usd'}).reset_index()
        dl=dl.rename(columns={'index':'Month'})
        csv_download('월별 전망 CSV 내려받기',dl,'autoorder_forecast_12months.csv','dl_forecast')
    with right:
        section('전망 시나리오', '예측 결과에 가상의 증가·감소율을 적용하여 금액 영향을 살펴봅니다.')
        st.markdown(
            '<div class="usage-note"><b>이 기능은 무엇인가요?</b> · 기본 AR6 전망에 '
            '사용자가 지정한 비율을 곱하는 <b>가정 분석</b>입니다. '
            '예를 들어 +10%를 선택하면 12개월의 예상 주문액이 모두 기본 전망보다 10% 늘어난 상황을 계산합니다. '
            '모델이 미래 수요를 다시 예측한 결과는 아닙니다.</div>',
            unsafe_allow_html=True
        )
        shock=st.slider(
            '기본 전망 대비 주문액 변화 가정',min_value=-15,max_value=15,value=0,step=1,
            format='%d%%',help='−10%는 기본 전망의 90%, +10%는 기본 전망의 110%를 가정합니다. 향후 12개월 예측액에 동일한 비율을 적용하며, AR3·AR6 모델 자체는 변하지 않습니다.'
        )
        st.caption('0% = 기본 전망 · +10% = 주문액이 10% 많은 경우 · −10% = 10% 적은 경우')
        factor=1+shock/100
        baseline=float(bundle.long.forecast.sum())
        scenario=baseline*factor
        st.markdown(f'<div class="panel">'
             f'<div class="panel-title">향후 12개월 주문액 합계 비교</div>'
             f'<div class="stat-row"><span>모델 기본 전망</span><b>{escape(fmt_amount(baseline,0))}</b></div>'
             f'<div class="stat-row"><span>사용자 시나리오 ({shock:+d}%)</span><b>{escape(fmt_amount(scenario,0))}</b></div>'
             f'<div class="stat-row"><span>기본 대비 차이</span><b>{escape(fmt_amount(scenario-baseline,0))}</b></div>'
             '</div>',unsafe_allow_html=True)
        if shock == 0:
            st.markdown('<div class="detail-bar">현재는 <b>기본 전망(0%)</b>입니다. 조정률을 변경하면 가정된 주문액의 차이만 즉시 계산되며, 모델 예측값은 그대로 유지됩니다.</div>',
                        unsafe_allow_html=True)
        else:
            st.markdown(
                f'<div class="detail-bar"><b>적용 계산</b> · 기본 12개월 전망 '
                f'{escape(fmt_amount(baseline,0))} × {factor:.2f} = '
                f'{escape(fmt_amount(scenario,0))}. '
                '동일한 가정을 각 월에 적용한 결과로, 실제 신규주문액이나 제조업 생산 수량을 예측한 추가 모델은 아닙니다.</div>',
                unsafe_allow_html=True
            )
        with st.expander('시나리오 분석의 활용 예시와 주의사항',expanded=False):
            st.markdown(
                '- **상향 시나리오(+10%)**: 산업 신규주문액이 기본 전망보다 10% 높아질 경우 금액 변화 검토\n'
                '- **하향 시나리오(−10%)**: 기본 전망보다 10% 낮아질 경우 금액 변화 검토\n'
                '- **적용 범위**: 미래 12개월의 금액에 같은 비율을 적용합니다. 실제 수요 증가 원인, 물가 영향, 생산 능력은 추정하지 않습니다.\n'
                '- **주의**: 시나리오를 선택해도 AR3 단기 예측, AR6 기본 전망, 80%·95% 예측구간과 경고 기준은 변경되지 않습니다.'
            )
        st.markdown(info_box('모델 실행 정책',
            f'<b>단기 AR3</b> · {bundle.short_last_refit:%Y-%m}까지 모델 계수 학습, 이후 월별 관측치 반영.<br>'
            f'<b>장기 AR6</b> · {bundle.long_refit_at:%Y-%m}까지의 실제값으로 학습한 뒤 12개월 연속 전망.'),
            unsafe_allow_html=True)

    st.markdown('<div class="section-divider"></div>',unsafe_allow_html=True)
    section('월별 시나리오 금액 영향', '사용자 가정이 향후 각 월의 주문액 추정에 미치는 금액 차이를 확인합니다. 단위는 명목 달러입니다.')
    scenario_df = scenario_table(bundle.long, shock)
    if shock == 0:
        st.info('조정률 0%: 모델 기본 전망을 사용하고 있습니다. 상단에서 변화율을 조정하면 월별 영향 그래프와 비교표가 표시됩니다.')
    else:
        st.plotly_chart(scenario_chart(scenario_df), use_container_width=True,
                        config={'displayModeBar':False})
    details = scenario_df.copy()
    details['대상 월'] = details['Month'].dt.strftime('%Y-%m')
    details['기본 전망 (억 달러)'] = details['baseline_million']/100
    details['가정 적용 (억 달러)'] = details['scenario_million']/100
    details['기본 대비 차이 (억 달러)'] = details['delta_million']/100
    if shock != 0:
        st.dataframe(details[['대상 월','기본 전망 (억 달러)','가정 적용 (억 달러)','기본 대비 차이 (억 달러)']].style.format({
            '기본 전망 (억 달러)':'{:,.1f}',
            '가정 적용 (억 달러)':'{:,.1f}',
            '기본 대비 차이 (억 달러)':'{:+,.1f}',
        }), hide_index=True, use_container_width=True, height=330)
        csv_download('월별 시나리오 금액 CSV 내려받기',scenario_df,
                     'autoorder_monthly_scenario.csv','dl_scenario')
    st.caption('동일한 조정률을 12개월 전체에 적용한 산술적 민감도 분석입니다. 추가 수요예측, 재고·생산량 추천, 실제 매출 또는 이익 추정이 아닙니다.')

availability_audit = None
context = None

with context_tab:
    section('산업 변화 분석',
            '신규주문액 변화와 부품 가격·산업생산 지표를 동일한 월 기준으로 비교합니다. 각 지표의 산업 범위는 서로 다릅니다.')
    st.markdown(
        '<div class="usage-note"><b>분석 방법</b> · 미국 Census의 명목 신규주문액(UBTPNO), '
        '미국 BLS의 자동차부품 생산자물가지수(PCU33633363), 미국 연준의 자동차·부품 산업생산지수(IPG3361T3N)를 '
        '각각 별도로 조회합니다. 예측모델 AR3·AR6의 입력 변수는 변경하지 않습니다.</div>',
        unsafe_allow_html=True
    )
    with st.spinner('관련 산업지표를 확인하고 있습니다.'):
        series, errors = context_series_cached()
    if errors:
        st.warning('일부 보조지표를 최신 자료로 조회하지 못했습니다. 확인되지 않은 지표는 분석 결과로 표시하지 않습니다.')
        with st.expander('보조지표 연결 상태 확인'):
            for alias, reason in errors.items():
                st.write(f"{CONTEXT_SOURCES[alias]['title']}: {reason}")
            st.caption('네트워크를 확인한 뒤 왼쪽의 최신 데이터 다시 확인을 선택하면 재조회합니다.')
    if len(series) == 2:
        try:
            context = economic_context(y, series['PPI'], series['IP'])
            availability_audit = audit_availability(y, series['PPI'], series['IP'], context.observation_month)
        except Exception as exc:
            st.warning(f'세 지표를 동일한 월로 비교할 수 없습니다: {exc}')
        else:
            comparable = context.observation_month
            st.markdown(
                f'<div class="detail-bar"><b>비교 기준월 · {comparable:%Y-%m}</b><br>'
                '아래 증감률 3개는 모두 이 월의 전년 동월과 비교한 값입니다. '
                '지표별 발표 시기가 달라도 다른 월의 수치를 섞거나 누락된 주문액을 추정해 채우지 않습니다.</div>',
                unsafe_allow_html=True,
            )
            if availability_audit.differing_publication_months or comparable < last:
                st.info('일부 지표의 최신 공개월이 다릅니다. 아래 표에서 최신 공개월과 실제 분석에 사용한 공통월을 구분해 확인할 수 있습니다.')
            with st.expander('각 지표의 최신 공개월 · 분석 적용월 확인', expanded=True):
                st.dataframe(
                    availability_audit.status.drop(columns=['최신월과 비교월 차이']),
                    use_container_width=True, hide_index=True,
                )
            context_data = [
                ('명목 신규주문액 변화', fmt_signed(context.orders_yoy), 'Census UBTPNO · 금액 기준'),
                ('부품 생산자물가지수 변화', fmt_signed(context.ppi_yoy), 'BLS PCU33633363 · 가격 참고'),
                ('자동차·부품 생산지수 변화', fmt_signed(context.production_yoy), 'FRB IPG3361T3N · 생산활동 참고'),
            ]
            cols = st.columns(3, gap='small')
            for col,(name,value,desc) in zip(cols,context_data):
                with col:
                    st.markdown(context_kpi(name,value,desc),unsafe_allow_html=True)
            st.caption(f'값의 기준월 {comparable:%Y-%m} · 주문액 최신월 {last:%Y-%m} · PPI 최신월 {context.latest_ppi_month:%Y-%m} · 생산지수 최신월 {context.latest_production_month:%Y-%m} · 모든 숫자는 공통월에 맞춰 계산')
            section('산업 지표 동향','각 지표의 절대 수준이 아니라 전년 동월 대비 변화율을 비교합니다.')
            st.plotly_chart(comparison_chart(context.comparison),use_container_width=True,
                config={'displayModeBar':False})
            st.markdown('<div class="section-divider"></div>',unsafe_allow_html=True)
            section('가격 변화를 고려한 참고 계산',
                    '명목 주문액의 증가율과 부품 가격지수의 증가율을 이용해 단순 실질화 대리지표를 계산합니다.')
            v = context.proxy_price_adjusted_yoy
            st.markdown(
                '<div class="panel">'
                '<div class="panel-title">가격 조정 참고지표 · 원인 분해 결과 아님</div>'
                f'<div class="stat-row"><span>명목 주문액 전년 동월 대비</span><b>{fmt_signed(context.orders_yoy)}</b></div>'
                f'<div class="stat-row"><span>부품 PPI 전년 동월 대비</span><b>{fmt_signed(context.ppi_yoy)}</b></div>'
                f'<div class="stat-row"><span>가격 조정 참고 비율</span><b>{fmt_signed(v)}</b></div>'
                '</div>',unsafe_allow_html=True
            )
            st.markdown(
                '<div class="detail-bar amber"><b>해석 주의</b> · 이 결과는 주문액의 물량 증가율이 아닙니다. '
                '주문액 지표에는 차체·트레일러·부품이 포함되지만 선택한 PPI는 자동차부품 제조업 가격지수입니다. '
                '산업 분류와 구성 품목이 달라 가격과 물량의 기여도를 정확하게 분리할 수 없으며, '
                '생산지수도 주문액에서 직접 산출한 물량이 아닙니다. 인과적 원인을 확정하거나 생산 수량을 계산하지 않습니다.</div>',
                unsafe_allow_html=True
            )
            data_view = context.comparison.tail(12).copy().reset_index()
            data_view = data_view.rename(columns={data_view.columns[0]: '기준월'})
            data_view['기준월']=pd.to_datetime(data_view['기준월']).dt.strftime('%Y-%m')
            data_view=data_view.rename(columns={
                'orders_yoy':'명목 신규주문액 (%)', 'PPI_yoy':'부품 PPI (%)',
                'IP_yoy':'자동차·부품 산업생산 (%)'})
            with st.expander('최근 12개월 동기 비교 상세자료', expanded=False):
                st.dataframe(data_view.style.format({c:'{:+.2f}%' for c in data_view.columns[1:]}),
                    use_container_width=True,hide_index=True)
                csv_download('산업 지표 비교 CSV 내려받기',data_view,
                    'autoorder_industry_context.csv','dl_industry')
    else:
        st.info('산업 변화 분석은 PPI와 산업생산지수를 모두 가져온 경우에만 표시합니다. 현재 확인 가능한 자료만 사용하며 누락 값을 추정해 채우지 않습니다.')
    with st.expander('원천 지표 및 발표 주기 확인',expanded=False):
        for definition in CONTEXT_SOURCES.values():
            st.markdown(f"**{definition['title']}** · `{definition['id']}` · {definition['units']} · {definition['publisher']}  \n[FRED 원천자료]({definition['url']})")
        st.caption('지표별 갱신 시점과 과거 값 개정은 다를 수 있습니다. 보조지표는 설명용 참고 자료이며 시계열 예측모델에 직접 입력하지 않습니다.')


with policy_tab:
    section('산업정책·선거 이벤트 모니터',
            '공식 문서에 기록된 정책 시행일과 선거·협정 검토 시점을 신규주문액 추이와 나란히 확인합니다.')
    st.markdown(
        '<div class="usage-note"><b>독립적인 이벤트 추적</b> · 정책 시행 또는 선거가 관측되었다는 사실은 확인할 수 있지만, '
        '그 사건이 실제 신규주문액을 얼마나 변화시켰는지는 별도의 인과분석이 필요합니다. '
        '아래 이벤트는 AR3·AR6 예측에 자동 반영되지 않으며, 정책 뉴스를 실시간 감시하는 기능도 아닙니다.</div>',
        unsafe_allow_html=True,
    )
    with st.expander('사용자 지정 정책 이벤트 추가 (선택)', expanded=False):
        st.caption('새 이벤트를 직접 기록하려면 date,title,category,description,source_url 열로 구성한 CSV를 업로드하십시오. '
                   '이벤트는 현재 화면에서만 사용하며 공식 검증을 거친 사건으로 취급하지 않습니다.')
        sample = pd.DataFrame([{
            'date': '2026-10-01', 'title': '입력 예시 · 추가 정책 검토',
            'category': '기타', 'description': '영향 가능 경로를 간략히 기재',
            'source_url': 'https://www.trade.gov/'
        }])
        st.download_button('이벤트 입력 양식 CSV', sample.to_csv(index=False).encode('utf-8-sig'),
                           file_name='autoorder_policy_event_template.csv', mime='text/csv',
                           key='dl_policy_sample')
        extra_file = st.file_uploader('이벤트 기록 CSV', type=['csv'], key='event_upload')
    extra_events = []
    if extra_file is not None:
        try:
            extra_events = parse_event_csv(extra_file)
            st.info(f'사용자 제공 이벤트 {len(extra_events)}개가 추가되었습니다. 해당 이벤트의 사실관계는 별도 확인이 필요합니다.')
        except Exception as exc:
            st.error(f'사용자 이벤트 파일을 읽을 수 없습니다: {exc}')
    events = sorted([*CURATED_EVENTS, *extra_events], key=lambda item: item.date)
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    section('주요 사건 목록', '사건 날짜·성격·원문 출처를 구분합니다. 시행일과 선거·검토일은 같은 의미가 아닙니다.')
    category_options = ['전체', *sorted(set(e.category for e in events))]
    selected_category = st.selectbox('사건 유형', category_options, key='policy_category',
                                     help='정책 시행 및 선거·협정 검토 이벤트를 구분해 표시합니다.')
    shown = [e for e in events if selected_category == '전체' or e.category == selected_category]
    st.dataframe(event_table(shown), use_container_width=True, hide_index=True, height=250,
                 column_config={'원문 URL': st.column_config.LinkColumn('공식 원문 또는 제공 출처', display_text='원문 열기')})
    csv_download('현재 이벤트 목록 CSV', event_table(shown), 'autoorder_policy_events.csv', 'dl_policy_events')
    st.caption('기본 목록은 2026-11-02에 확인한 미국 정부 1차 자료를 이용한 정적 등록 자료입니다. '
               '후속 개정·예외·새 정책을 자동 수집하지 않습니다.')

    section('사건별 주문액 변화 확인', '선택한 사건의 시행·발생 전후 주문액 흐름을 비교합니다.')
    default_event = next((i for i, e in enumerate(events) if e.event_id == 'autoparts2025'), 0)
    chosen = st.selectbox('검토할 정책·사건', events, index=default_event, key='policy_selected',
                          format_func=lambda e: f'{e.date:%Y-%m-%d}  |  {e.title}')
    c1, c2 = st.columns([1.25, 1], gap='large')
    with c1:
        fig = go.Figure()
        center = pd.Timestamp(chosen.date.year, chosen.date.month, 1)
        start = center - pd.DateOffset(months=24)
        end = center + pd.DateOffset(months=24)
        subset = y.loc[(y.index >= start) & (y.index <= end)]
        if subset.empty:
            st.info('이 사건 날짜는 현재 주문액 관측 범위 밖에 있습니다.')
        else:
            fig.add_trace(go.Scatter(x=subset.index, y=subset.values / 100,
                                     mode='lines+markers', name='명목 신규주문액',
                                     line=dict(color=NAVY, width=2.5), marker=dict(size=4),
                                     hovertemplate='%{x|%Y-%m}<br>%{y:,.1f}억 달러<extra></extra>'))
            fig.add_shape(type='line', x0=center, x1=center,
                          y0=0, y1=1, yref='paper',
                          line=dict(color=ORANGE, width=2, dash='dash'))
            fig.add_annotation(x=center, y=1.06, yref='paper', text='사건 시점',
                               showarrow=False, font=dict(color=ORANGE, size=12))
            fig.update_layout(template='plotly_white', height=360, margin=dict(l=8, r=8, t=30, b=15),
                              xaxis_title=None, yaxis_title='명목 신규주문액 (억 달러)',
                              showlegend=False, hovermode='x unified',
                              plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            fig.update_yaxes(gridcolor=GRID, rangemode='tozero')
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar':False})
        st.caption('사건일이 포함된 월을 경계로 표시한 참고 차트입니다. 차트의 동시 변화는 정책 효과를 입증하지 않습니다.')
    with c2:
        st.markdown(info_box('사건 정보',
            f'<b>{escape(chosen.title)}</b><br>'
            f'날짜: {chosen.date:%Y-%m-%d} · {escape(chosen.kind)}<br>'
            f'검토 경로: {escape(chosen.channel)}<br>'
            f'출처: {escape(chosen.source)}<br>'
            f'{escape(chosen.note)}'), unsafe_allow_html=True)
        st.link_button('해당 사건의 원문 열기', chosen.source_url,
                       use_container_width=True, type='secondary')
        if not chosen.verified:
            st.warning('사용자 등록 사건입니다. 내용·출처를 별도로 확인한 뒤 의사결정에 활용하십시오.')

    section('정책 전후 주문액·예측오차 비교',
            '사건 발생월을 포함하여 이전 12개월과 이후 12개월의 월별 전년 동월 대비 증가율 평균을 비교합니다.')
    comparison = before_after_yoy(y, chosen.date)
    a, b, c = st.columns(3, gap='small')
    a.metric('사건 이전 12개월 평균 증가율',
             fmt_signed(comparison['before_yoy']) if np.isfinite(comparison['before_yoy']) else '자료 부족')
    b.metric('사건 이후 12개월 평균 증가율',
             fmt_signed(comparison['after_yoy']) if np.isfinite(comparison['after_yoy']) else '자료 부족')
    c.metric('기간 간 차이',
             f"{comparison['difference_pp']:+.2f}%p" if np.isfinite(comparison['difference_pp']) else '평가 보류')
    st.caption(f"전후 비교에 확보된 월 수: 이전 {comparison['before_count']}/12개월 · 이후 {comparison['after_count']}/12개월. "
               '12개월 전 기간이 모두 확인된 경우에만 평균을 표시합니다. 현재 개정된 원천자료 기준이며 과거 최초 발표본은 아닙니다.')
    err_comp = before_after_ape(alerts, chosen.date)
    if np.isfinite(err_comp['before_mape']) and np.isfinite(err_comp['after_mape']):
        st.markdown(
            f'<div class="detail-bar"><b>단기 AR3 예측오차 참고</b> · 이전 12개월 MAPE '
            f"{err_comp['before_mape']:.2f}% → 이후 12개월 {err_comp['after_mape']:.2f}%. "
            '이는 정책을 추가한 모델의 성능이 아니라, 기존 AR3 예측의 기간별 오차입니다.</div>',
            unsafe_allow_html=True,
        )
    else:
        st.info('해당 사건 전후로 완결된 AR3 월별 예측오차 자료가 없어 오차 비교를 생략했습니다. '
                '현재 주문액 자료만으로는 정책 반영 모델의 정확도를 평가할 수 없습니다.')
    st.markdown('<div class="detail-bar amber"><b>분석 한계</b> · 관세 또는 선거 날짜의 전후 비교는 인과효과가 아닙니다. '
                '코로나19, 경기, 환율, 가격, 공급망 변화 및 다른 정책과 중첩될 수 있습니다. '
                '기업 발주·재고 계획은 내부 자료와 전문가 검토를 함께 사용하십시오.</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="footnote">이 화면은 확인된 정책·정치 이벤트의 이력과 신규주문액 변화를 함께 보여주는 기능입니다. '
                '미래 정책 변화 탐지, 법률 자문 또는 관세 영향액 산정 기능이 아니며 AR3·AR6의 모델 계수나 출력값을 조정하지 않습니다.</div>',
                unsafe_allow_html=True)

with alert_tab:
    section('예측 편향 모니터링','연속 과소·과대예측의 방향과 크기를 기준으로 점검 신호를 제공합니다.')
    st.caption('현재 연결된 주문액 원천자료로 AR3의 월별 과거 예측을 자동 재현하고 경고를 평가합니다. 신규 공개월이나 기존 수치가 갱신되면 재계산합니다. 초기 계산에는 다소 시간이 걸릴 수 있습니다.')
    if alerts is None:
        st.markdown('<div class="detail-bar amber">경고 상태를 현재 확인할 수 없습니다. 예측 기능과는 별도이며, 원천자료와 계산 환경을 확인해 주십시오.</div>',
                    unsafe_allow_html=True)
        if monitor_error:
            with st.expander('경고 계산 오류 상세'):
                st.code(monitor_error)
    else:
        if not alert_current:
            st.markdown(f'<div class="detail-bar amber">경고 기록의 마지막 관측월은 {alerts.Month.max():%Y-%m}입니다. 현재 원천자료({last:%Y-%m})와 일치하지 않아 최신 경고 상태로 사용하지 않습니다.</div>',unsafe_allow_html=True)
        # 2022~2023 established warning thresholds; those months were not
        # genuinely assessed for alert status and must not inflate the ratio.
        eligible_alerts = alerts.loc[alerts.Year >= 2024].copy()
        flagged=eligible_alerts.loc[eligible_alerts.alert_level.ne('정상')]
        s1,s2,s3=st.columns(3)
        s1.metric('자동 평가 기준월',f'{alerts.Month.max():%Y-%m}')
        s2.metric('현재 조건',status if alert_current else '확인 불가')
        s3.metric('경고 발생 월',f'{len(flagged)} / {len(eligible_alerts)}개월')
        st.plotly_chart(alert_chart(alerts),use_container_width=True,
                        config={'displayModeBar':False})
        col1,col2=st.columns([1,1.5],gap='large')
        with col1:
            section('경고 상태 분포')
            st.plotly_chart(alert_donut(eligible_alerts),use_container_width=True,
                            config={'displayModeBar':False})
        with col2:
            section('지표 해석')
            n_ok=int(eligible_alerts.alert_level.eq('정상').sum())
            n_watch=int(eligible_alerts.alert_level.eq('주의').sum())
            n_high=int(eligible_alerts.alert_level.eq('강화 경고').sum())
            st.markdown('<div class="panel">'
                f'<div class="stat-row"><span>경고 조건 미충족</span><b>{n_ok}개월</b></div>'
                f'<div class="stat-row"><span>주의</span><b>{n_watch}개월</b></div>'
                f'<div class="stat-row"><span>강화 경고</span><b>{n_high}개월</b></div>'
                '</div>',unsafe_allow_html=True)
            st.markdown('<div class="footnote">원형 차트는 경고 조건별 월 수의 구성입니다. 산업이 정상인 비율이나 모델의 예측 적중률이 아닙니다.</div>',unsafe_allow_html=True)
        section('경고 발생 이력')
        if flagged.empty:
            st.caption('설정된 조건을 충족하는 과거 월이 없습니다.')
        else:
            events=flagged[['Month','alert_direction','alert_level','bias_pct','avg_2','avg_3']].copy()
            events.columns=['월','방향','단계','월별 편향 (%)','2개월 평균 (%)','3개월 평균 (%)']
            events['월']=events['월'].dt.strftime('%Y-%m')
            st.dataframe(events.style.format({k:'{:+.2f}' for k in events.columns[-3:]},na_rep='—'),
                hide_index=True,use_container_width=True)
        csv_download('경고 이력 CSV 내려받기',alerts,'autoorder_alert_history.csv','dl_alert')
        st.markdown('<div class="section-divider"></div>',unsafe_allow_html=True)
        section('경고 판단 기준')
        a,b=st.columns(2,gap='large')
        with a:
            st.markdown(info_box('1단계 · 주의',
                '최근 <b>2개월 연속</b> 같은 방향의 예측편향이 발생하고 해당 2개월 평균이 과거 참고 기준을 초과할 때 표시합니다.'),
                unsafe_allow_html=True)
        with b:
            st.markdown(info_box('2단계 · 강화 경고',
                '최근 <b>3개월 연속</b> 같은 방향의 예측편향이 발생하고 해당 3개월 평균이 과거 참고 기준을 초과할 때 표시합니다. 중복 시 강화 경고를 우선합니다.'),