"""Optional, read-only economic context for aggregate US automotive order forecasts.

PPI and industrial production are independent FRED series. They have different
industry coverage and update schedules from UBTPNO. No SKU or unit estimate is
made, and none of the context data are used in the AR3/AR6 forecasts.
"""
from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

CONTEXT_SOURCES = {
    "PPI": {
        "id": "PCU33633363",
        "title": "자동차부품 제조업 생산자물가지수",
        "units": "2003년 12월=100",
        "publisher": "미국 노동통계국(BLS)",
        "url": "https://fred.stlouisfed.org/series/PCU33633363",
    },
    "IP": {
        "id": "IPG3361T3N",
        "title": "자동차·부품 제조업 산업생산지수",
        "units": "2017년=100",
        "publisher": "미국 연방준비제도(FRB)",
        "url": "https://fred.stlouisfed.org/series/IPG3361T3N",
    },
}


def read_fred_indicator_csv(raw: bytes, series_id: str) -> pd.Series:
    """Strict parsing; no fill-forward of missing monthly records."""
    if not raw or len(raw) > 6_000_000:
        raise ValueError("자료가 비어 있거나 크기 제한을 초과했습니다.")
    frame = pd.read_csv(BytesIO(raw))
    frame.columns = [str(c).strip().lstrip("\ufeff") for c in frame.columns]
    date_col = next((col for col in ("observation_date", "DATE", "Date", "date") if col in frame.columns), None)
    if date_col is None or series_id not in frame.columns:
        raise ValueError(f"FRED 응답에서 날짜 또는 {series_id} 열을 찾지 못했습니다.")
    dates = pd.to_datetime(frame[date_col], errors="raise")
    if dates.dt.day.ne(1).any():
        raise ValueError("월별 자료가 아닌 날짜를 발견했습니다.")
    if dates.duplicated().any():
        raise ValueError("중복된 날짜가 있습니다.")
    values = pd.to_numeric(frame[series_id].replace(".", np.nan), errors="coerce")
    valid = values.notna() & np.isfinite(values) & (values > 0)
    if not valid.any():
        raise ValueError("조회된 유효 관측치가 없습니다.")
    cleaned = pd.Series(values.to_numpy(dtype=float), index=pd.DatetimeIndex(dates), name=series_id).sort_index()
    cleaned = cleaned.loc[cleaned.first_valid_index():cleaned.last_valid_index()]
    if cleaned.isna().any() or (cleaned <= 0).any():
        raise ValueError("원천자료에 내부 결측치 또는 0 이하 값이 있습니다.")
    if cleaned.index.to_period("M").nunique() != len(cleaned):
        raise ValueError("중복 월이 있습니다.")
    return cleaned.asfreq("MS")


def fetch_indicator(series_id: str, timeout: int = 16) -> pd.Series:
    if series_id not in {s["id"] for s in CONTEXT_SOURCES.values()}:
        raise ValueError("허용되지 않은 산업 보조지표 식별자입니다.")
    url = "https://fred.stlouisfed.org/graph/fredgraph.csv?" + urlencode({"id": series_id, "cosd": "2004-01-01"})
    request = Request(url, headers={
        "User-Agent": "Mozilla/5.0 (compatible; AutoOrderLens/5.0; CSV reader)",
        "Accept": "text/csv,text/plain,*/*",
    })
    with urlopen(request, timeout=timeout) as response:
        raw = response.read(6_000_001)
    return read_fred_indicator_csv(raw, series_id)


@dataclass
class ContextOverview:
    observation_month: pd.Timestamp
    orders_yoy: float
    ppi_yoy: float
    production_yoy: float
    proxy_price_adjusted_yoy: float
    latest_ppi_month: pd.Timestamp
    latest_production_month: pd.Timestamp
    comparison: pd.DataFrame


def economic_context(orders: pd.Series, ppi: pd.Series, production: pd.Series) -> ContextOverview:
    """Year-over-year trends on exact same observed months only.

    The derived price-adjusted metric is strictly a rough proxy from mismatched
    aggregate series; it must never be labelled physical volume or causal impact.
    """
    for name, series in (("orders", orders), ("PPI", ppi), ("IP", production)):
        if not isinstance(series, pd.Series) or not isinstance(series.index, pd.DatetimeIndex):
            raise ValueError(f"{name}은 날짜 인덱스가 있는 월별 시계열이어야 합니다.")
        if series.index.has_duplicates or series.empty:
            raise ValueError(f"{name}에 중복 또는 빈 데이터가 있습니다.")
    frame = pd.concat([orders.rename("orders"), ppi.rename("PPI"), production.rename("IP")], axis=1)
    frame = frame.loc[:orders.index.max()].sort_index()
    # Do not shift the whole joined series after dropping missing observations.
    # Calculate monthly YoY only when the exact 12-month previous values exist.
    for column in ("orders", "PPI", "IP"):
        frame[f"{column}_yoy"] = (frame[column] / frame[column].shift(12) - 1.0) * 100
    frame = frame.replace([np.inf, -np.inf], np.nan)
    valid = frame.dropna(subset=["orders_yoy", "PPI_yoy", "IP_yoy"]).copy()
    if valid.empty:
        raise ValueError("세 지표의 동일 월 전년 동월 비교 자료가 충분하지 않습니다.")
    last = valid.iloc[-1]
    adjusted_proxy = ((1 + last["orders_yoy"] / 100) / (1 + last["PPI_yoy"] / 100) - 1) * 100
    if not np.isfinite(adjusted_proxy):
        raise ValueError("가격 참고지표 계산에 실패했습니다.")
    return ContextOverview(
        observation_month=pd.Timestamp(valid.index[-1]),
        orders_yoy=float(last["orders_yoy"]),
        ppi_yoy=float(last["PPI_yoy"]),
        production_yoy=float(last["IP_yoy"]),
        proxy_price_adjusted_yoy=float(adjusted_proxy),
        latest_ppi_month=pd.Timestamp(ppi.dropna().index.max()),
        latest_production_month=pd.Timestamp(production.dropna().index.max()),
        comparison=valid[["orders_yoy", "PPI_yoy", "IP_yoy"]].tail(30).copy(),
    )