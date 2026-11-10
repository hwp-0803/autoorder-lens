"""Refresh public FRED observations and derived model forecasts.

Never write simulated data to public datasets.
"""
from __future__ import annotations
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fred_data import DOWNLOAD_URL, SERIES_ID, fetch_fred_orders
from forecast_engine import forecast_current
from monitoring_service import build_alert_monitor


def format_snapshot(series: pd.Series) -> str:
    return (series.rename("orders").to_frame().rename_axis("Month").reset_index()
            .to_csv(index=False, date_format="%Y-%m-%d"))


def build_forecast_rows(series: pd.Series) -> tuple[str, dict]:
    bundle = forecast_current(series)
    asof = series.index[-1].strftime("%Y-%m")
    tables = []
    for label, forecast in (("AR3", bundle.short), ("AR6", bundle.long)):
        table = forecast.rename_axis("Month").reset_index()
        table.insert(0, "model", label)
        table.insert(1, "source_last_observation", asof)
        tables.append(table)
    combined = pd.concat(tables, ignore_index=True)
    assert len(bundle.short) == 1 and len(bundle.long) == 12
    assert pd.to_datetime(combined["Month"]).min() > series.index[-1]
    info = {
        "short_model": "SARIMA AR3: 1 month ahead, annually refit",
        "long_model": "SARIMA AR6: fixed-origin next 12 months",
        "short_last_refit": bundle.short_last_refit.strftime("%Y-%m"),
        "long_refit_at": bundle.long_refit_at.strftime("%Y-%m"),
        "short_converged": bool(bundle.short_converged),
        "long_converged": bool(bundle.long_converged),
        "forecasts_are_estimates": True
    }
    return combined.to_csv(index=False), info


def build_monitor_rows(series: pd.Series) -> str:
    result = build_alert_monitor(series)
    assert not result.empty and result.Month.max() == series.index[-1]
    return result.to_csv(index=False, date_format="%Y-%m-%d")


def main() -> int:
    series = fetch_fred_orders()
    folder = ROOT / "data"
    folder.mkdir(parents=True, exist_ok=True)
    observations = format_snapshot(series)
    source_file = folder / "fred_UBTPNO.csv"
    forecast_file = folder / "forecasts_AR3_AR6.csv"
    monitor_file = folder / "monitoring_AR3.csv"
    status_file = folder / "latest_status.json"
    previous = source_file.read_text(encoding="utf-8") if source_file.exists() else None
    if previous == observations and all(p.is_file() for p in (forecast_file, monitor_file, status_file)):
        print(f"No change to official FRED data; latest month {series.index[-1]:%Y-%m}")
        return 0

    # Build complete outputs before replacing previous snapshots.
    forecast_csv, model_info = build_forecast_rows(series)
    monitor_csv = build_monitor_rows(series)
    metadata = {
        "series_id": SERIES_ID,
        "source_url": DOWNLOAD_URL,
        "unit": "nominal million USD, not seasonally adjusted",
        "latest_observation_month": series.index[-1].strftime("%Y-%m"),
        "latest_value_million_usd": float(series.iloc[-1]),
        "number_of_observations": int(len(series)),
        "snapshot_generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "warning_assessed_through": series.index[-1].strftime("%Y-%m"),
        **model_info,
        "warning": "Model predictions are not observed sales, production or inventory."
    }
    payload = {
        source_file: observations,
        forecast_file: forecast_csv,
        monitor_file: monitor_csv,
        status_file: json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
    }
    for path, content in payload.items():
        path.with_name(path.name + ".tmp").write_text(content, encoding="utf-8")
    for path in payload:
        os.replace(path.with_name(path.name + ".tmp"), path)
    print(f"Refreshed FRED snapshots at observation {series.index[-1]:%Y-%m}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
