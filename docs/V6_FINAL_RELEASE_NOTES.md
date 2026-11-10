# AutoOrder Lens v6.0 — Final Demo Candidate

작성일: 2026-11-02 (KST)

## 이번 버전의 범위

### 기존 기능 유지

- **단기 AR3**: 매년 1월 계수 재학습, 월별 실제값 갱신, 1개월 앞 예측
- **장기 AR6**: 현 관측자료로 재학습한 12개월 고정시점 예측
- FRED UBTPNO 자동 조회, 명목 신규주문액 단위 표시, 80%·95% 모형 기반 예측구간
- 2개월 주의·3개월 강화 편향 모니터링(초기 재평가 후 캐시)
- PPI·산업생산 비교 시 공통 관측월 기준, 외부 평가 CSV와 최신 원천 값 대조
- 0% 시나리오 시 중복 그래프 비노출, 시나리오 값은 사용자 가정으로 분리
- 모델 성능 재평가 및 HTML 검증 보고서 출력

### 추가: 정책 이벤트

- 별도 **정책 이벤트** 탭: 공식 관세·무역협정 사건과 선거·공동 검토 시점 구분
- 2018-03-23 철강·알루미늄 Section 232 관세 시행
- 2020-07-01 USMCA 발효
- 2024-11-05 미국 대통령선거(정책 시행 사건 아님)
- 2025-05-03 일부 수입 자동차부품 Section 232 관세 시행
- 2026-07-01 USMCA 공동 검토(협정이 곧바로 종료되거나 새로운 관세가 시행된 것이 아님)
- 사건별 명목 주문액 그래프(전후 최대 24개월), 이전/이후 12개월의 평균 YoY, **완결되지 않은 12개월 기간은 통계 표시 보류**
- 충분한 AR3 과거 예측오차가 있을 때만 정책 전후 MAPE를 참고 제공
- 정적 공식 목록 외에 CSV로 사용자가 별도 사건을 입력 가능. 사용자 입력은 별도 검증 필요로 표시하고 세션에서만 활용
- 운영 개요에 최신 등록 통상 사건의 의미 있는 링크/안내 추가

**중요:** 정책 시행과 시계열 변화의 상관·시차를 묘사할 뿐, 정책의 인과효과를 식별하거나 선거/정책 기여율을 예측하지 않습니다. AR3·AR6 모델 입력·계수·예측값을 정책 이벤트로 자동 수정하지 않습니다. 공식 사건 목록은 정적 수동 검증 버전이며 실시간 정책 뉴스 모니터링·알림 기능이 아닙니다.

## 공식 이벤트 출처

| 사건 | 공식 출처 |
|---|---|
| 2018 Section 232 | [미국 CBP 공지](https://content.govdelivery.com/accounts/USDHSCBP/bulletins/1e43e86) |
| 2020 USMCA 발효 | [미국 USTR 협정 개요](https://ustr.gov/trade-agreements/free-trade-agreements/united-states-mexico-canada-agreement) |
| 2024 대통령선거 | [미국 FEC 선거 결과](https://www.fec.gov/documents/5644/2024presgeresults.pdf) |
| 2025 자동차부품 관세 | [미국 상무부 관련 발표](https://www.trade.gov/press-release/department-commerce-announces-new-auto-tariff-offset-process) |
| 2026 USMCA 공동 검토 | [미국 USTR 공동검토 발표](https://www.ustr.gov/about/policy-offices/press-office/press-releases/2026/july/ambassador-greer-issues-statement-usmca-joint-review) |

모두 2026-11-02 기준으로 확인한 1차 자료입니다. 정책별 적용 대상, 예외, 후속 개정과 계약상 실무 영향은 반드시 최신 문서로 재확인하십시오.

## 설치

1. 이전 폴더를 백업하십시오.
2. `AutoOrder_Lens_Final_v6.zip` 압축을 풀고, `automotive_forecast_dashboard` 폴더를 프로젝트 폴더 안에 배치합니다.
3. `run_dashboard.bat`를 실행합니다. 배치 파일 실행이 차단되면 해당 폴더에서 다음 명령을 사용합니다.

```cmd
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

4. 브라우저에서 `http://localhost:8501`을 엽니다.
5. **정책 이벤트** 탭에서 2025년 관세 사건과 공식 링크·전후 12개월 자료를 확인합니다.

`orders_backlog_inventory.csv` 또는 `orders_clean.csv`는 프로젝트의 상위 폴더에 그대로 두어도 됩니다. 네트워크가 가능하면 앱이 FRED를 먼저 조회합니다.

## 검증

- `python -m pytest tests -q` : **34 passed** (기존 28 + 새 기능 6)
- `python -m py_compile *.py` : 통과
- 공개 주문액 스냅샷(2005-01~2026-08) 대조: 2018 및 2025년 전후 YoY 비교값이 기존 분석 결과와 일치
- 웹 렌더링·실시간 FRED 호출·Windows 실행·다중 사용자 부하는 **이 환경에서 검증하지 못함**

## 상업 운영 전 필수 검증

이 프로젝트는 **판매용 데모 후보**로서 로그인, 조직별 접근권한, 감사로그, 자동 배포, 다중 사용자 성능, 운영 장애 대응, 데이터 상업적 재배포 권한과 법무 검토가 완료된 유료 SaaS가 아닙니다. `docs/COMMERCIAL_READINESS.md` 체크리스트를 참조하십시오. AR3·AR6는 탐색적 백테스트로 선택했으며 실시간 빈티지/미래 자료 기반 외부 검증이 추가로 필요합니다.