"""Fetch current FRED UBTPNO observations without an API key.

FRED graph CSV is a public download endpoint, NOT a guaranteed real-time push API.
If the endpoint is offline or changes, the Streamlit app falls back to local CSV.
"""
from __future__ import annotations

from io import BytesIO
from urllib.request import Request, urlopen
import pandas as pd

from forecast_engine import load_orders

SERIES_ID = "UBTPNO"
# Keep the EXACT model-comparison training start used in the original project.
MODEL_START = "2005-02-01"
DOWNLOAD_URL = (
    "https://fred.stlouisfed.org/graph/fredgraph.csv"
    f"?id={SERIES_ID}&cosd={MODEL_START}"
)


def fetch_fred_orders(timeout: int = 18) -> pd.Series:
    """Download monthly USD-million orders and validate no gaps, duplicates or bad values."""
    request = Request(
        DOWNLOAD_URL,
        headers={
            "User-Agent": "Mozilla/5.0 (compatible; AutoOrderLens/1.1; FRED CSV reader)",
            "Accept": "text/csv,text/plain,*/*",
        },
    )
    with urlopen(request, timeout=timeout) as response:
        content = response.read(6_000_001)
    if len(content) > 6_000_000:
        raise ValueError("FRED 응답 크기가 비정상적입니다.")

    try:
        df = pd.read_csv(BytesIO(content))
    except Exception as exc:
        raise ValueError("FRED 데이터가 CSV 형식이 아닙니다.") from exc

    df.columns = [str(c).strip().lstrip("\ufeff") for c in df.columns]
    if SERIES_ID not in df.columns:
        raise ValueError("FRED 응답에 UBTPNO 열이 없습니다. CSV 다운로드 형식을 확인하세요.")
    date_col = next(
        (c for c in ("observation_date", "DATE", "Date", "date") if c in df.columns),
        None,
    )
    if date_col is None:
        raise ValueError("FRED 응답에 날짜 열이 없습니다.")

    numeric = pd.to_numeric(df[SERIES_ID], errors="coerce")
    # FRED can include missing pre-series rows or trailing unpublished dates.
    # Only trailing/leading blank rows are removed; missing months INSIDE series are errors.
    keep = numeric.notna()
    if not keep.any():
        raise ValueError("FRED가 아직 유효한 주문액 관측치를 제공하지 않습니다.")
    positions = keep.to_numpy().nonzero()[0]
    cleaned = df.iloc[positions.min():positions.max() + 1].copy()
    cleaned = cleaned.rename(columns={date_col: "Month", SERIES_ID: "orders"})
    cleaned = cleaned[["Month", "orders"]]
    cleaned["Month"] = pd.to_datetime(cleaned["Month"], errors="raise")
    cleaned = cleaned[cleaned["Month"] >= pd.Timestamp(MODEL_START)]
    if cleaned.empty:
        raise ValueError("FRED 데이터에 2005-02 이후 주문액이 없습니다.")
    series = load_orders(BytesIO(cleaned.to_csv(index=False).encode("utf-8")))
    if series.index.min() != pd.Timestamp(MODEL_START):
        raise ValueError("FRED 자료 시작월이 2005-02와 다릅니다. 임의로 훈련 기간을 바꾸지 않습니다.")
    return series