"""Transparent observation-month reconciliation for business users.

This module never imputes missing months, shifts publication vintages, or
claims that differently defined US industry indicators share an industry scope.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


INDICATOR_LABELS = (
    ('orders', '명목 신규주문액', 'UBTPNO'),
    ('PPI', '자동차부품 PPI', 'PCU33633363'),
    ('IP', '자동차·부품 산업생산지수', 'IPG3361T3N'),
)


def _month_distance(a: pd.Timestamp, b: pd.Timestamp) -> int:
    """a minus b in calendar months."""
    return (a.year - b.year) * 12 + a.month - b.month


@dataclass(frozen=True)
class AvailabilityAudit:
    common_month: pd.Timestamp
    latest_orders_month: pd.Timestamp
    latest_ppi_month: pd.Timestamp
    latest_production_month: pd.Timestamp
    status: pd.DataFrame
    differing_publication_months: bool


def audit_availability(orders: pd.Series, ppi: pd.Series, production: pd.Series,
                       comparison_month: pd.Timestamp) -> AvailabilityAudit:
    """Show each original last-observed month and the *same* analysis month.

    Caller supplies economic_context(...).observation_month, which is defined
    only from months with a valid 12-month YoY for all three indicators.
    """
    series_map = {'orders': orders, 'PPI': ppi, 'IP': production}
    common = pd.Timestamp(comparison_month).to_period('M').to_timestamp()
    rows = []
    last_dates = {}
    for key, title, series_id in INDICATOR_LABELS:
        s = series_map[key]
        if not isinstance(s, pd.Series) or not isinstance(s.index, pd.DatetimeIndex) or s.empty:
            raise ValueError(f'{series_id}: 관측 시계열이 비어 있거나 유효하지 않습니다.')
        if s.index.has_duplicates:
            raise ValueError(f'{series_id}: 날짜가 중복됩니다.')
        available = s.dropna()
        if available.empty:
            raise ValueError(f'{series_id}: 사용 가능한 관측치가 없습니다.')
        latest = pd.Timestamp(available.index.max()).to_period('M').to_timestamp()
        if latest < common or common not in available.index:
            raise ValueError(f'{series_id}: 공통 비교월의 실적이 없습니다.')
        prev_year = common - pd.DateOffset(years=1)
        if prev_year not in available.index:
            raise ValueError(f'{series_id}: 공통 비교월의 전년 동월 실적이 없습니다.')
        last_dates[key] = latest
        rows.append({
            '지표': title,
            'FRED 코드': series_id,
            '조회된 최신월': latest.strftime('%Y-%m'),
            '비교에 사용한 월': common.strftime('%Y-%m'),
            '최신월과 비교월 차이': _month_distance(latest, common),
            '기준월 상태': '동일' if latest == common else '선공개 자료 제외',
        })
    return AvailabilityAudit(
        common_month=common,
        latest_orders_month=last_dates['orders'],
        latest_ppi_month=last_dates['PPI'],
        latest_production_month=last_dates['IP'],
        status=pd.DataFrame(rows),
        differing_publication_months=len(set(last_dates.values())) > 1,
    )