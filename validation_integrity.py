"""Ensure imported performance tables are not silently mixed with newer FRED data."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ActualsConsistency:
    status: str  # matching / conflicting / unverified
    count_matched: int
    count_conflicting: int
    count_unavailable: int
    comment: str


def check_backtest_actuals(frame: pd.DataFrame, orders: pd.Series,
                           tolerance_million: float = 0.1) -> ActualsConsistency:
    """Cross-check actual values in imported evaluations against loaded orders.

    A file lacking Month/Actual can still be an exploratory historical benchmark
    but cannot claim to be checked against the current source vintage.
    """
    if not {'Month', 'Actual'}.issubset(frame.columns):
        return ActualsConsistency('unverified', 0, 0, 0,
            '파일에 Month 또는 Actual 열이 없어 현재 실적과의 동일성 검증이 불가능합니다.')
    dates = pd.to_datetime(frame['Month'], errors='coerce')
    raw_actuals = pd.to_numeric(frame['Actual'], errors='coerce')
    if dates.isna().any() or raw_actuals.isna().any() or not np.isfinite(raw_actuals).all():
        raise ValueError('외부 평가 파일에 유효하지 않은 실제값 또는 날짜가 있습니다.')
    months = dates.dt.to_period('M').dt.to_timestamp()
    actual_ref = orders.reindex(pd.DatetimeIndex(months))
    missing = actual_ref.isna().to_numpy()
    mismatched = ((abs(raw_actuals.to_numpy(dtype=float) - actual_ref.to_numpy(dtype=float)) > tolerance_million)
                  & ~missing)
    conflicts = int(mismatched.sum())
    missed = int(missing.sum())
    matched = len(frame) - conflicts - missed
    if conflicts:
        return ActualsConsistency('conflicting', matched, conflicts, missed,
            f'기존 평가파일과 최신 주문액이 {conflicts}개 행에서 불일치합니다. 현재 원천자료로 다시 검증해 주십시오.')
    if missed:
        return ActualsConsistency('unverified', matched, conflicts, missed,
            f'현재 주문액 자료에 없는 평가 대상 {missed}개 행은 동일성을 확인하지 못했습니다.')
    return ActualsConsistency('matching', matched, 0, 0,
        f'실제 주문액 {matched}개 평가 행이 현재 조회한 자료와 일치합니다. 단, 공표 당시 데이터 빈티지의 동일성은 보장하지 않습니다.')