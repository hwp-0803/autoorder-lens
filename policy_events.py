"""Verified, curated policy/election event register and non-causal comparisons.

All events are labels for interpreting an existing nominal manufacturing-orders
series. No automatic prediction of policy effects, causal estimation, or event
feed subscription takes place. Users may add a session-local CSV of events.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from urllib.parse import urlparse
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PolicyEvent:
    event_id: str
    date: date
    title: str
    category: str
    kind: str
    channel: str
    source: str
    source_url: str
    note: str
    verified: bool = True


# Official primary sources checked 2026-11-02. These are historical dates only;
# changes, exemptions, subsequent revisions and specific products must be checked
# in the linked documents before use in a procurement decision.
CURATED_EVENTS = (
    PolicyEvent(
        'steel2018', date(2018, 3, 23), '미국 철강·알루미늄 Section 232 관세',
        '관세', '시행', '원재료 조달·수입가격·공급망 비용 경로',
        '미국 관세국경보호청(CBP)',
        'https://content.govdelivery.com/accounts/USDHSCBP/bulletins/1e43e86',
        '대상 국가·품목 및 예외는 시점에 따라 달라집니다.'),
    PolicyEvent(
        'usmca2020', date(2020, 7, 1), 'USMCA 무역협정 발효',
        '무역협정', '발효', '북미 원산지 규정·역내 부품 조달 경로',
        '미국 무역대표부(USTR)',
        'https://ustr.gov/trade-agreements/free-trade-agreements/united-states-mexico-canada-agreement',
        '코로나19 충격과 기간이 겹치므로 전후 변화를 협정 효과로 해석하지 않습니다.'),
    PolicyEvent(
        'election2024', date(2024, 11, 5), '미국 대통령선거',
        '선거', '선거 실시', '정책 불확실성·기업 투자심리 관련 검토 지점',
        '미국 연방선거위원회(FEC)',
        'https://www.fec.gov/documents/5644/2024presgeresults.pdf',
        '선거일 자체가 관세·보조금 정책의 시행일은 아닙니다.'),
    PolicyEvent(
        'autoparts2025', date(2025, 5, 3), '미국 일부 수입 자동차부품 Section 232 관세',
        '관세', '시행', '부품 수입원가·대체조달·발주 시점·현지 생산 관련 경로',
        '미국 상무부(ITA)',
        'https://www.trade.gov/press-release/department-commerce-announces-new-auto-tariff-offset-process',
        '25% 관세 발표·적용 내용과 관련 예외 및 오프셋 제도를 원문에서 확인해야 합니다.'),
    PolicyEvent(
        'usmca2026review', date(2026, 7, 1), 'USMCA 공동 검토',
        '무역협정', '공식 검토', '향후 협정 개정·원산지 규정·통상 불확실성 점검',
        '미국 무역대표부(USTR)',
        'https://www.ustr.gov/about/policy-offices/press-office/press-releases/2026/july/ambassador-greer-issues-statement-usmca-joint-review',
        'USTR 발표 시점에 현행 형태 갱신 합의는 이루어지지 않았으나 협정은 계속 유효합니다. 즉각적인 관세 시행 사건이 아닙니다.'),
)


def _month_start(d: date | pd.Timestamp | str) -> pd.Timestamp:
    p = pd.Timestamp(d)
    return pd.Timestamp(year=p.year, month=p.month, day=1)


def before_after_yoy(orders: pd.Series, event_date: date | pd.Timestamp | str,
                     months: int = 12) -> dict:
    """Compare mean YoY in strictly separate pre vs post windows.

    Require *all* months in each window; do not extrapolate or impute. Uses
    current revised data, not historic published vintages. The event-month is
    the first post window month, regardless of the day within the month.
    """
    if not isinstance(orders, pd.Series) or not isinstance(orders.index, pd.DatetimeIndex):
        raise ValueError('월별 날짜 인덱스의 주문액 시계열이 필요합니다.')
    if months < 1 or months > 36:
        raise ValueError('비교 기간은 1~36개월이어야 합니다.')
    idx = _month_start(event_date)
    y = orders.sort_index().astype(float).asfreq('MS')
    yoy = (y.div(y.shift(12)) - 1) * 100
    before_idx = pd.date_range(end=idx - pd.DateOffset(months=1), periods=months, freq='MS')
    after_idx = pd.date_range(start=idx, periods=months, freq='MS')
    pre = yoy.reindex(before_idx)
    post = yoy.reindex(after_idx)
    pre_count = int(pre.notna().sum())
    post_count = int(post.notna().sum())
    pre_mean = float(pre.mean()) if pre_count == months else np.nan
    post_mean = float(post.mean()) if post_count == months else np.nan
    return {
        'event_month': idx,
        'months': months,
        'before_count': pre_count,
        'after_count': post_count,
        'before_yoy': pre_mean,
        'after_yoy': post_mean,
        'difference_pp': post_mean - pre_mean if np.isfinite(pre_mean) and np.isfinite(post_mean) else np.nan,
    }


def before_after_ape(backtest: pd.DataFrame | None,
                     event_date: date | pd.Timestamp | str, months: int = 12) -> dict:
    """Forecast absolute-error comparison only if all months exist, without causality."""
    if backtest is None or not {'Month', 'APE'}.issubset(backtest.columns):
        return {'before_count': 0, 'after_count': 0, 'before_mape': np.nan, 'after_mape': np.nan}
    df = backtest[['Month', 'APE']].copy()
    df['Month'] = pd.to_datetime(df['Month']).dt.to_period('M').dt.to_timestamp()
    if df['Month'].duplicated().any():
        raise ValueError('월별 예측오차 기록에 중복 월이 있습니다.')
    s = df.set_index('Month')['APE'].astype(float)
    start = _month_start(event_date)
    before = s.reindex(pd.date_range(end=start - pd.DateOffset(months=1), periods=months, freq='MS'))
    after = s.reindex(pd.date_range(start=start, periods=months, freq='MS'))
    n1, n2 = int(before.notna().sum()), int(after.notna().sum())
    return {
        'before_count': n1, 'after_count': n2,
        'before_mape': float(before.mean()) if n1 == months else np.nan,
        'after_mape': float(after.mean()) if n2 == months else np.nan,
    }


def event_table(events: list[PolicyEvent] | tuple[PolicyEvent, ...]) -> pd.DataFrame:
    return pd.DataFrame([{
        '시점': e.date.strftime('%Y-%m-%d'),
        '구분': e.category,
        '사건': e.title,
        '사건 성격': e.kind,
        '출처 상태': '공식 자료 확인' if e.verified else '사용자 제공 · 별도 확인 필요',
        '원문 URL': e.source_url,
    } for e in events])


def parse_event_csv(source, max_events: int = 30) -> list[PolicyEvent]:
    """Session-local user CSV, never elevate user-provided records to verified."""
    if hasattr(source, 'size') and source.size > 512_000:
        raise ValueError('이벤트 CSV 파일 크기는 500KB 이하여야 합니다.')
    raw = pd.read_csv(source).fillna('')
    required = {'date', 'title', 'category', 'description', 'source_url'}
    if not required.issubset(raw.columns):
        raise ValueError('CSV 열: date,title,category,description,source_url 형식이 필요합니다.')
    if len(raw) > max_events:
        raise ValueError(f'추가할 수 있는 이벤트는 최대 {max_events}개입니다.')
    out = []
    for i, (_, row) in enumerate(raw.iterrows()):
        d = pd.to_datetime(str(row['date']), format='%Y-%m-%d', errors='raise').date()
        title = str(row['title']).strip()
        category = str(row['category']).strip()
        description = str(row['description']).strip()
        url = str(row['source_url']).strip()
        u = urlparse(url)
        if not title or len(title) > 110 or not category or len(category) > 30 or len(description) > 300:
            raise ValueError(f'{i+1}행의 제목·분류·내용 길이 또는 필수값을 확인하십시오.')
        if u.scheme != 'https' or not u.netloc or u.username or u.password:
            raise ValueError(f'{i+1}행의 출처는 https URL이어야 합니다.')
        out.append(PolicyEvent(f'custom-{i}', d, title, category, '사용자 등록',
                               description, '사용자 제공', url,
                               '공식 확인을 거치지 않은 사용자 입력 이벤트입니다.', verified=False))
    return out