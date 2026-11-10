"""Guardrail: customers must be able to understand forecasting controls."""
from pathlib import Path

APP = (Path(__file__).resolve().parents[1] / 'app.py').read_text(encoding='utf-8')


def test_prediction_interval_is_not_misrepresented_as_accuracy():
    assert '모델 적중률이나 보증 수준이 아닙니다' in APP
    assert '예측 정확도 80% 또는 95%' in APP
    assert '모형 예측구간' in APP


def test_history_and_scenarios_are_described_as_non_model_changes():
    assert '미래 전망은 항상 12개월입니다' in APP
    assert '미래 예측값은 바뀌지 않습니다' in APP
    assert '모델이 미래 수요를 다시 예측한 결과는 아닙니다' in APP
    assert '시나리오를 선택해도 AR3 단기 예측' in APP