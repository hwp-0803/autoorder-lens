"""Explainable business indicators for AutoOrder Lens.

All levels are nominal million USD for the U.S. Census/FRED series UBTPNO.
Insights are transparent arithmetic based only on input observed data and
scenario forecasts. This module does not infer vehicle/part quantities,
company-specific demand, production capacity, or causal effects.
"""
from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd


def yoy_change(current: float, previous: float) -> float:
    """Percent change. Empty denominator or non-finite inputs return NaN."""
    if not np.isfinite(current) or not np.isfinite(previous) or previous <= 0:
        return float('nan')
    return (float(current) / float(previous) - 1.0) * 100.0


@dataclass(frozen=True)
class MarketSummary:
    observed_month: pd.Timestamp
    latest_yoy_pct: float
    recent_three_yoy_pct: float
    next_twelve_sum: float
    previous_twelve_sum: float
    next_twelve_yoy_pct: float
    forecast_up_months: int
    forecast_down_months: int
    peak_month: pd.Timestamp
    peak_yoy_pct: float
    trough_month: pd.Timestamp
    trough_yoy_pct: float
    month_comparison: pd.DataFrame


def market_summary(y: pd.Series, long_forecast: pd.DataFrame) -> MarketSummary:
    """Compare complete next 12 months to identical months one year earlier."""
    if y.empty or len(y) < 24:
        raise ValueError('최소 24개월의 실제 신규주문액이 필요합니다.')
    if len(long_forecast) != 12 or not long_forecast.index.equals(pd.date_range(y.index[-1]+pd.offsets.MonthBegin(1), periods=12, freq='MS')):
        raise ValueError('장기 전망은 최신 관측월 다음 달부터 이어지는 12개월이어야 합니다.')
    if 'forecast' not in long_forecast.columns:
        raise ValueError('장기 전망에 forecast 열이 없습니다.')
    levels = long_forecast['forecast'].astype(float)
    last = y.index[-1]
    latest_yoy = yoy_change(float(y.iloc[-1]), float(y.loc[last-pd.DateOffset(months=12)]))
    recent_three_yoy = yoy_change(float(y.iloc[-3:].sum()), float(y.iloc[-15:-12].sum()))
    past = y.iloc[-12:].astype(float)
    same_months = levels.index - pd.DateOffset(months=12)
    prior = y.reindex(same_months)
    if prior.isna().any():
        raise ValueError('전망의 전년 동월 실적을 확인할 수 없습니다.')
    growth = (levels.to_numpy()/prior.to_numpy() - 1)*100
    frame = pd.DataFrame({
        'Month': levels.index,
        'forecast': levels.to_numpy(),
        'year_ago_actual': prior.to_numpy(),
        'yoy_pct': growth,
        'delta_million': levels.to_numpy() - prior.to_numpy(),
    })
    return MarketSummary(
        observed_month=last,
        latest_yoy_pct=latest_yoy,
        recent_three_yoy_pct=recent_three_yoy,
        next_twelve_sum=float(levels.sum()),
        previous_twelve_sum=float(past.sum()),
        next_twelve_yoy_pct=yoy_change(float(levels.sum()), float(past.sum())),
        forecast_up_months=int((growth > 0).sum()),
        forecast_down_months=int((growth < 0).sum()),
        peak_month=pd.Timestamp(frame.iloc[int(np.argmax(growth))]['Month']),
        peak_yoy_pct=float(np.max(growth)),
        trough_month=pd.Timestamp(frame.iloc[int(np.argmin(growth))]['Month']),
        trough_yoy_pct=float(np.min(growth)),
        month_comparison=frame,
    )


def scenario_table(long_forecast: pd.DataFrame, adjustment_pct: float) -> pd.DataFrame:
    """Fixed proportional scenario, NOT a new statistical forecast."""
    if not np.isfinite(adjustment_pct) or adjustment_pct < -100:
        raise ValueError('시나리오 조정률을 확인해 주십시오.')
    base = long_forecast['forecast'].to_numpy(dtype=float)
    if not np.isfinite(base).all() or (base <= 0).any():
        raise ValueError('유효한 장기 전망 금액이 필요합니다.')
    scenario = base * (1+adjustment_pct/100)
    return pd.DataFrame({
        'Month': pd.DatetimeIndex(long_forecast.index),
        'baseline_million': base,
        'scenario_million': scenario,
        'delta_million': scenario - base,
    })


def market_interpretation(s: MarketSummary, alert_level: str | None, alert_direction: str | None) -> list[tuple[str, str]]:
    """Facts and grounded checkpoints; no presumed causal or SKU conclusions."""
    facts = [
        ('확인된 실적', f'최근 관측월 신규주문액의 전년 동월 대비 변화율은 {s.latest_yoy_pct:+.1f}%이며, 최근 3개월 합계는 전년 동일 3개월 대비 {s.recent_three_yoy_pct:+.1f}%입니다.'),
        ('향후 12개월 전망', f'AR6가 추정한 다음 12개월 신규주문액 합계는 직전 12개월 실적 대비 {s.next_twelve_yoy_pct:+.1f}%입니다. 월별 전년 동월 대비 전망은 증가 {s.forecast_up_months}개월, 감소 {s.forecast_down_months}개월입니다.'),
    ]
    if alert_level is None:
        facts.append(('모니터링 상태', '예측오차 자동 평가 결과를 확인할 수 없습니다. 현재 주문액 전망과 별개로 경고 계산 상태를 점검해야 합니다.'))
    elif alert_level == '정상':
        facts.append(('예측오차 점검', '최근 실적은 설정된 연속 예측편향 경고 조건을 충족하지 않았습니다. 신규주문액의 증가·감소 여부와는 다른 지표입니다.'))
    else:
        direction = '예측 대비 실제 주문액 증가 방향' if alert_direction == '주문 증가' else '예측 대비 실제 주문액 감소 방향'
        facts.append(('예측오차 점검', f'{direction}의 {alert_level} 조건이 충족되었습니다. 고객사 발주, 가격 및 데이터 변경 여부를 추가 확인하십시오.'))
    return facts