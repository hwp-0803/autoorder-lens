"""Retrospective 2026-09 forecast: training through 2026-08, using today's revised FRED historical values."""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
from fred_data import fetch_fred_orders
from forecast_engine import forecast_current

ROOT=Path(__file__).resolve().parents[1]
TARGET=pd.Timestamp("2026-09-01")

def main():
    y=fetch_fred_orders().loc["2005-02-01":]
    train=y.loc[:"2026-08-01"]
    if train.empty or train.index[-1]!=pd.Timestamp("2026-08-01"):
        raise ValueError("2026-08 history unavailable")
    if TARGET not in y.index:
        raise ValueError(f"FRED 2026-09 official observation is not yet available. Last: {y.index[-1]:%Y-%m}")
    bundle=forecast_current(train)
    actual=float(y.loc[TARGET])
    def scores(pred):
        return dict(predicted_million_usd=round(pred,2),error_million_usd=round(pred-actual,2),
                    absolute_percentage_error=round(100*abs(pred-actual)/actual,4),
                    bias="overforecast" if pred>actual else "underforecast" if pred<actual else "equal")
    d={
        "type":"RETROSPECTIVE RECONSTRUCTION, not stored original prediction",
        "historical_data":"current revised UBTPNO observations (possible vintage differences)",
        "training_last_month":"2026-08","actual_month":"2026-09",
        "latest_available_month":y.index[-1].strftime("%Y-%m"),
        "unit":"nominal million USD",
        "actual":round(actual,2),
        "AR3_1_month":scores(float(bundle.short.loc[TARGET,"forecast"])),
        "AR6_first_month_of_12":scores(float(bundle.long.loc[TARGET,"forecast"])),
        "seasonal_naive_previous_year":scores(float(train.loc[TARGET-pd.DateOffset(years=1)])),
        "caveat":"Not timestamped forecast issued in Oct 2026. Revised data may differ."
    }
    out=ROOT/"analysis"/"2026-09-retrospective.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(d,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(d,ensure_ascii=False,indent=2))

if __name__=="__main__":
    main()
