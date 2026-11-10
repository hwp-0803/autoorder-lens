"""Automatic, source-aligned monitoring for AutoOrder Lens.

This module contains no Streamlit UI dependencies so that source integrity,
no-future-data calculations, and cache invalidation can be tested separately.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from forecast_engine import add_alerts, backtest_ar3


def build_alert_monitor(y: pd.Series, start_year: int = 2022) -> pd.DataFrame:
    """Build current warning status from the same actual orders used by forecasts.

    The input is the complete currently available monthly series. Each historic
    one-step prediction is generated using only observations up to that month.
    Backtest uses annual fit and monthly state updates, consistent with the
    short-horizon production forecasting policy. Historic vintages are *not*
    available: revised series are retrospectively evaluated at today's values.

    After the UI sees a changed `y`, a Streamlit cache keyed by the whole series
    forces a new run. On reruns with unchanged input, cached results are reused.
    """
    if not isinstance(y, pd.Series) or y.empty:
        raise ValueError('월별 주문액 시계열이 비어 있습니다.')
    if not isinstance(y.index, pd.DatetimeIndex):
        raise ValueError('월별 주문액 날짜 인덱스가 필요합니다.')
    if not y.index.is_monotonic_increasing or y.index.has_duplicates:
        raise ValueError('월별 주문액 날짜는 오름차순이고 중복이 없어야 합니다.')
    if not np.isfinite(y.to_numpy(dtype=float)).all():
        raise ValueError('주문액에 결측치 또는 무한대 값이 있습니다.')
    monitor = backtest_ar3(y, start_year=start_year)
    if monitor.empty:
        raise ValueError('자동 경고를 계산할 월별 예측오차 자료가 없습니다.')
    if monitor['Month'].duplicated().any():
        raise ValueError('월별 예측오차 결과에 중복 월이 있습니다.')
    if monitor['Month'].iloc[-1] != y.index[-1]:
        raise ValueError('모델 평가의 최신 월이 원천 주문액의 최신 월과 다릅니다.')
    obs = y.reindex(pd.DatetimeIndex(monitor['Month'])).to_numpy(dtype=float)
    if not np.allclose(obs, monitor['Actual'].to_numpy(dtype=float), atol=0.01, rtol=0):
        raise ValueError('모델 평가 실제값이 원천 주문액과 다릅니다.')
    if not (np.isfinite(monitor['Forecast'].to_numpy(dtype=float))).all():
        raise ValueError('모델 평가 중 비정상 예측값이 있습니다.')
    return add_alerts(monitor)