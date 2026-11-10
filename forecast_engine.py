"""Forecasting and alert calculations for US motor vehicle bodies, trailers & parts orders.

All order amounts are million USD, NOT vehicle counts.
No external API or hidden data dependencies.
"""
from __future__ import annotations

from dataclasses import dataclass
import io
from pathlib import Path
from typing import BinaryIO

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

COVID_MONTHS = ("2020-04", "2020-05", "2020-06", "2020-07")


def load_orders(source: str | Path | BinaryIO) -> pd.Series:
    """Parse a CSV with one date column and an `orders` amount column."""
    if hasattr(source, "seek"):
        source.seek(0)
    raw = pd.read_csv(source, encoding="utf-8-sig")
    raw.columns = [str(c).strip().replace("\ufeff", "") for c in raw.columns]
    if "orders" not in raw.columns:
        raise ValueError("CSV에 'orders' 열이 필요합니다. (단위: 백만 달러)")
    candidate = next((c for c in ["Month", "DATE", "Date", "date", "month"] if c in raw), raw.columns[0])
    if candidate == "orders":
        raise ValueError("CSV에서 날짜 열을 찾을 수 없습니다.")
    dt = pd.to_datetime(raw[candidate], errors="coerce")
    values = pd.to_numeric(raw["orders"], errors="coerce")
    if dt.isna().any() or values.isna().any():
        raise ValueError("날짜 또는 orders 값에 해석할 수 없는 결측/문자 값이 있습니다.")
    if dt.dt.day.ne(1).any():
        # FRED monthly exports may use YYYY-MM-DD; require month-start semantics.
        dt = dt.dt.to_period("M").dt.to_timestamp()
    s = pd.Series(values.to_numpy(dtype=float), index=pd.DatetimeIndex(dt), name="orders").sort_index()
    if s.index.duplicated().any():
        raise ValueError("같은 월이 두 번 존재합니다. 중복 월을 확인해 주세요.")
    if (s <= 0).any():
        raise ValueError("주문액이 0 이하인 달이 있습니다.")
    if len(s) < 80:
        raise ValueError("계절 SARIMA 학습에 필요한 과거 월별 자료가 부족합니다. 최소 80개월을 준비해 주세요.")
    s = s.asfreq("MS")
    if s.isna().any():
        missing = ", ".join(s[s.isna()].index.strftime("%Y-%m").tolist()[:5])
        raise ValueError(f"중간에 누락된 월이 있습니다: {missing}")
    return s


def covid_exog(idx: pd.DatetimeIndex) -> pd.DataFrame:
    idx = pd.DatetimeIndex(idx)
    x = pd.DataFrame(index=idx)
    for month in COVID_MONTHS:
        x["covid_" + month.replace("-", "_")] = (idx == pd.Timestamp(month)).astype(float)
    return x


def fit_model(y: pd.Series, lag: int):
    if lag not in (3, 6):
        raise ValueError("지원되는 AR 지연은 3 또는 6개월입니다.")
    model = SARIMAX(
        y,
        exog=covid_exog(y.index),
        order=([lag], 1, 1),
        seasonal_order=(0, 1, 1, 12),
        trend="n",
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    return model.fit(disp=False, maxiter=200)


def forecast_table(fitted, future_index: pd.DatetimeIndex) -> pd.DataFrame:
    results = fitted.get_forecast(steps=len(future_index), exog=covid_exog(future_index))
    result = pd.DataFrame({"forecast": results.predicted_mean.to_numpy()}, index=future_index)
    for level in (80, 95):
        ci = results.conf_int(alpha=1 - level / 100)
        result[f"lower_{level}"] = ci.iloc[:, 0].to_numpy()
        result[f"upper_{level}"] = ci.iloc[:, 1].to_numpy()
    return result


@dataclass
class ForecastBundle:
    short: pd.DataFrame
    long: pd.DataFrame
    short_converged: bool
    long_converged: bool
    short_last_refit: pd.Timestamp
    long_refit_at: pd.Timestamp


def forecast_current(y: pd.Series) -> ForecastBundle:
    """1-step AR3: Jan refit + monthly append. 12-step AR6: refit at origin.

    Forecasts are for AFTER the last observed month. Does not use future actuals.
    """
    last = y.index[-1]
    next_month = last + pd.offsets.MonthBegin(1)
    short_refit_date = pd.Timestamp(year=next_month.year - 1, month=12, day=1)
    train_short = y.loc[:short_refit_date]
    if len(train_short) < 80:
        raise ValueError("현재 연도 이전의 자료가 부족해 AR3 모델을 학습할 수 없습니다.")
    short_fit = fit_model(train_short, lag=3)
    short_converged = bool(short_fit.mle_retvals.get("converged", True))
    extra = y.loc[y.index > short_refit_date]
    if not extra.empty:
        short_fit = short_fit.append(extra, exog=covid_exog(extra.index), refit=False)
    short_future_idx = pd.date_range(start=next_month, periods=1, freq="MS")
    short_df = forecast_table(short_fit, short_future_idx)

    long_fit = fit_model(y, lag=6)
    long_future_idx = pd.date_range(start=next_month, periods=12, freq="MS")
    long_df = forecast_table(long_fit, long_future_idx)
    return ForecastBundle(
        short=short_df,
        long=long_df,
        short_converged=short_converged,
        long_converged=bool(long_fit.mle_retvals.get("converged", True)),
        short_last_refit=short_refit_date,
        long_refit_at=last,
    )


def load_monitor(source: str | Path | BinaryIO) -> pd.DataFrame:
    if hasattr(source, "seek"):
        source.seek(0)
    df = pd.read_csv(source, encoding="utf-8-sig", parse_dates=["Month"])
    needed = {"Month", "Actual", "Forecast"}
    if not needed.issubset(df.columns):
        raise ValueError(f"모니터링 CSV는 {sorted(needed)} 열이 필요합니다.")
    df = df.sort_values("Month").drop_duplicates("Month", keep="last").reset_index(drop=True)
    df["Actual"] = pd.to_numeric(df["Actual"], errors="raise")
    df["Forecast"] = pd.to_numeric(df["Forecast"], errors="raise")
    if df["Actual"].le(0).any():
        raise ValueError("Actual은 양수여야 합니다.")
    return df


def backtest_ar3(y: pd.Series, start_year: int = 2022) -> pd.DataFrame:
    """One-month-ahead backtest: annual coefficient refit + monthly observed update."""
    rows = []
    end_year = y.index[-1].year
    for year in range(start_year, end_year + 1):
        train = y.loc[:f"{year-1}-12"]
        test = y.loc[f"{year}-01":f"{year}-12"]
        if test.empty or len(train) < 80:
            continue
        state = fit_model(train, lag=3)
        for date, actual in test.items():
            exog_next = covid_exog(pd.DatetimeIndex([date]))
            prediction = state.get_forecast(steps=1, exog=exog_next)
            p = float(prediction.predicted_mean.iloc[0])
            ci = prediction.conf_int(alpha=0.05)
            lower, upper = (float(ci.iloc[0, 0]), float(ci.iloc[0, 1]))
            rows.append({"Month": date, "Actual": float(actual), "Forecast": p,
                         "Error": float(actual - p), "APE": 100 * abs(actual-p) / actual,
                         "Outside95": not (lower <= actual <= upper),
                         "Lower95": lower, "Upper95": upper})
            state = state.append(y.loc[[date]], exog=exog_next, refit=False)
    if not rows:
        raise ValueError("2022년 이후 평가 데이터가 부족합니다.")
    return pd.DataFrame(rows)


def add_alerts(monitor: pd.DataFrame) -> pd.DataFrame:
    """Attach directional, 2/3-month alert statuses. Calibrate in periods preceding evaluation.

    2024-25 thresholds from 2022-23; 2026+ thresholds from 2022-25.
    Intended as exploratory business monitoring, NOT validated anomaly probabilities.
    """
    df = monitor.sort_values("Month").copy().reset_index(drop=True)
    df["Year"] = df["Month"].dt.year
    df["bias_pct"] = 100 * (df["Actual"] - df["Forecast"]) / df["Actual"]
    df["alert_level"] = "정상"
    df["alert_direction"] = "—"
    df["alert_text"] = "연속 편향 기준 미충족"
    df["alert_threshold"] = np.nan
    for w in (2, 3):
        df[f"avg_{w}"] = df["bias_pct"].rolling(w, min_periods=w).mean()
        for sign, label in ((1, "주문 증가"), (-1, "주문 감소")):
            magnitude = sign * df[f"avg_{w}"]
            all_same_direction = (sign * df["bias_pct"] > 0).rolling(w, min_periods=w).sum().eq(w)
            for calib_start, calib_end, eval_start, eval_end in (
                (2022, 2023, 2024, 2025),
                (2022, 2025, 2026, 2100),
            ):
                historical = magnitude.loc[df["Year"].between(calib_start, calib_end)].dropna()
                if len(historical) < 8:
                    continue
                threshold = max(0., float(historical.quantile(.95)))
                mask = (df["Year"].between(eval_start, eval_end)
                        & all_same_direction & magnitude.gt(threshold))
                if w == 2:
                    mask &= df["alert_level"].eq("정상")
                df.loc[mask, "alert_level"] = "강화 경고" if w == 3 else "주의"
                df.loc[mask, "alert_direction"] = label
                df.loc[mask, "alert_text"] = (
                    f"{w}개월 연속 {('과소' if sign == 1 else '과대')}예측 + 편향 기준 초과"
                )
                df.loc[mask, "alert_threshold"] = threshold
    return df


def load_backtest(source: str | Path | BinaryIO, kind: str) -> pd.DataFrame:
    if hasattr(source, "seek"):
        source.seek(0)
    df = pd.read_csv(source, encoding="utf-8-sig")
    if kind == "short":
        needed = {"Year", "Model", "APE"}
    elif kind == "long":
        needed = {"Origin", "Model", "APE"}
    else:
        raise ValueError("kind는 short 또는 long이어야 합니다.")
    if not needed.issubset(df.columns):
        raise ValueError(f"{kind} 평가 CSV에 필요한 열: {sorted(needed)}")
    return df