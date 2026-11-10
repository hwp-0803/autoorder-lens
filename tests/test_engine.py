import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from forecast_engine import load_orders, forecast_current, add_alerts
from evaluation_engine import evaluate_short


def series():
    idx=pd.date_range('2005-02-01','2022-03-01',freq='MS')
    n=len(idx)
    rng=np.random.default_rng(42)
    s=(23000 + 35*np.arange(n) + 1400*np.sin(2*np.pi*idx.month/12)
       +rng.normal(0,500,n))
    return pd.Series(s,index=idx,name='orders')


def test_load_orders_and_forecast(tmp_path):
    y=series()
    f=tmp_path/'orders.csv'
    y.to_frame().rename_axis('Month').to_csv(f)
    read=load_orders(f)
    assert read.index.equals(y.index)
    bundle=forecast_current(read)
    assert len(bundle.short)==1
    assert len(bundle.long)==12
    assert bundle.short.index[0]==pd.Timestamp('2022-04-01')
    assert bundle.short_last_refit==pd.Timestamp('2021-12-01')
    assert bundle.long_refit_at==pd.Timestamp('2022-03-01')
    assert np.isfinite(bundle.long.values).all()
    assert (bundle.long['lower_95'] <= bundle.long['upper_95']).all()


def test_alert_directions_and_prioritization():
    months=pd.date_range('2022-01-01', '2026-08-01',freq='MS')
    # Quiet calibration years, a non-alert year, then sharp 2026 monthly bias.
    error=np.zeros(len(months),dtype=float)
    error[(months>=pd.Timestamp('2026-02'))&(months<=pd.Timestamp('2026-04'))]=0.03
    error[(months>=pd.Timestamp('2026-05'))&(months<=pd.Timestamp('2026-08'))]=-0.03
    actual=np.full(len(months),100.0)
    pred=actual*(1-error)
    obs=pd.DataFrame({'Month':months,'Actual':actual,'Forecast':pred})
    alerted=add_alerts(obs)
    m=alerted.set_index('Month')
    assert m.loc['2026-03-01','alert_level']=='주의'
    assert m.loc['2026-04-01','alert_level']=='강화 경고'
    assert m.loc['2026-04-01','alert_direction']=='주문 증가'
    assert m.loc['2026-08-01','alert_level']=='강화 경고'
    assert m.loc['2026-08-01','alert_direction']=='주문 감소'


def test_short_backtest_does_not_use_future(tmp_path):
    y=series()
    results=evaluate_short(y,years=[2022])
    assert not results.empty
    assert results['Month'].min()==pd.Timestamp('2022-01-01')
    assert len(results)==6  # 3 months * 2 models
    assert set(results.Model)=={'AR3','AR6'}