from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from forecast_engine import forecast_current, add_alerts


def make_data():
    i = pd.date_range('2005-02-01', '2026-08-01', freq='MS')
    rng = np.random.default_rng(7)
    return pd.Series(29000 + 17*np.arange(len(i))
                     + 950*np.sin(2*np.pi*i.month/12)
                     + rng.normal(0, 360, len(i)), index=i, name='orders')


def test_forecast_horizon_and_units():
    result = forecast_current(make_data())
    assert result.short.index[0] == pd.Timestamp('2026-09-01')
    assert result.long.index[0] == pd.Timestamp('2026-09-01')
    assert result.long.index[-1] == pd.Timestamp('2027-08-01')
    assert result.short_last_refit == pd.Timestamp('2025-12-01')
    assert result.long_refit_at == pd.Timestamp('2026-08-01')
    assert result.short.iloc[0]['forecast'] > 10000
    assert (result.long['lower_95'] <= result.long['upper_95']).all()
    assert (result.long['lower_80'] <= result.long['upper_80']).all()
    assert (result.long['lower_80'] >= result.long['lower_95']).all()
    assert (result.long['upper_80'] <= result.long['upper_95']).all()


def test_no_market_normality_claim_in_app():
    app = (Path(__file__).resolve().parents[1] / 'app.py').read_text()
    assert '경고 조건 미충족' in app
    assert '데이터 미연결' not in app or '평가 데이터 미연결' in app
    assert '95% 예측구간' in app or '95% 하한' in app or '예측구간' in app
    assert 'ㅋㅋㅋ' not in app
    assert '!!!' not in app