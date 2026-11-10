# AutoOrder Lens — Demand Intelligence

자동차 차체·트레일러·부품 제조업의 **미국 월별 명목 신규주문액**(FRED `UBTPNO`)을 분석하는 공개 데이터 기반 수요예측 대시보드입니다.

- 단기: SARIMA AR3, 다음 1개월 전망
- 장기: SARIMA AR6, 향후 12개월 전망
- 편향 모니터링, 정책 이벤트, 산업보조지표 및 시나리오 탐색
- 원천자료 공개 갱신 시 재조회 및 예측 재계산

## 실행

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## 공개 웹사이트 배포

[Streamlit Community Cloud](https://share.streamlit.io)에서 이 GitHub 저장소의 `main` / `app.py`를 선택하고 Public으로 배포합니다. Python 3.12 권장.

이 프로젝트는 취업 포트폴리오·연구 및 데모용입니다. 원천자료는 산업별 명목 **금액**이며 특정 기업의 생산대수·실제 발주량이나 재고 최적화 성과가 아닙니다. 정책 이벤트는 인과효과를 입증하지 않습니다. 원천자료의 공개 지연·개정 가능성 및 외부 검증의 한계를 고려하세요.

GitHub Actions는 소스 변경 시 테스트를 수행하고, 별도 예약 작업으로 FRED 공개 자료를 확인하도록 구성합니다. 실시간 갱신 보장 서비스는 아닙니다.
