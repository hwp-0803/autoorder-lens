"""Reproducible, optional backtests; never train on months being evaluated."""
from __future__ import annotations
import numpy as np
import pandas as pd
from forecast_engine import fit_model, covid_exog


def evaluate_short(y: pd.Series, years=range(2022, 2027)) -> pd.DataFrame:
    rows = []
    for year in years:
        training = y.loc[:f'{year-1}-12']
        testing = y.loc[f'{year}-01':f'{year}-12']
        if len(training) < 80 or testing.empty:
            continue
        for lag in (3, 6):
            state = fit_model(training, lag)
            for date, actual in testing.items():
                exog = covid_exog(pd.DatetimeIndex([date]))
                pred = float(state.forecast(1, exog=exog).iloc[0])
                rows.append(dict(Year=year, Month=date, Model=f'AR{lag}', Actual=float(actual),
                                 Forecast=pred, APE=abs(float(actual)-pred)/float(actual)*100,
                                 Origin=f'{year}-01', Horizon=1))
                state = state.append(testing.loc[[date]], exog=exog, refit=False)
    return pd.DataFrame(rows)


def evaluate_long(y: pd.Series) -> pd.DataFrame:
    """Quarterly 12-month rolling origins, only where full observed horizon exists.

    2026-Jan is included as partial Jan-Aug if these observations exist.
    """
    origins = list(pd.date_range('2022-01-01', '2025-07-01', freq='QS-JAN'))
    origins.append(pd.Timestamp('2026-01-01'))
    rows = []
    for origin in origins:
        train = y.loc[y.index < origin]
        test = y.loc[origin: origin + pd.DateOffset(months=11)]
        if len(train) < 80 or test.empty:
            continue
        preds = {}
        for lag in (3, 6):
            fitted = fit_model(train, lag)
            preds[f'AR{lag}'] = fitted.forecast(len(test), exog=covid_exog(test.index)).to_numpy()
        preds['앙상블 50:50'] = (preds['AR3']+preds['AR6'])/2
        for name, vals in preds.items():
            for h, (date, act, pred) in enumerate(zip(test.index, test.to_numpy(), vals), 1):
                rows.append(dict(Origin=origin.strftime('%Y-%m'), Month=date, Horizon=h,
                                 Model=name, Actual=float(act), Forecast=float(pred),
                                 APE=100*abs(float(act)-float(pred))/float(act)))
    return pd.DataFrame(rows)