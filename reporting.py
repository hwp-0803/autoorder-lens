"""Client-friendly, self-contained performance report generated only from real backtests."""
from __future__ import annotations

from datetime import datetime
from html import escape
import math
import pandas as pd



def _validation_inventory(short: pd.DataFrame | None, long: pd.DataFrame | None) -> str:
    """Derive the exact evaluated month/origin ranges from the supplied rows."""
    parts: list[str] = []
    if short is not None and not short.empty:
        dates = pd.to_datetime(short['Month'], errors='coerce') if 'Month' in short else pd.Series(dtype='datetime64[ns]')
        dates = dates.dropna()
        if len(dates):
            parts.append(f"단기: {dates.min():%Y-%m}~{dates.max():%Y-%m}, 평가 행 {len(short):,}개, 평가월 {dates.dt.to_period('M').nunique():,}개월")
        else:
            years = pd.to_numeric(short['Year'], errors='raise')
            parts.append(f"단기: 평가연도 {int(years.min())}~{int(years.max())}, 평가 행 {len(short):,}개 (평가월 정보 없음)")
    if long is not None and not long.empty:
        starts = pd.to_datetime(long['Origin'].astype(str) + '-01', errors='coerce')
        starts = starts.dropna()
        n_origins = long['Origin'].astype(str).nunique()
        n_months = len(pd.to_datetime(long['Month'], errors='coerce').dropna().dt.to_period('M').unique()) if 'Month' in long else None
        horizon = pd.to_numeric(long['Horizon'], errors='coerce').max() if 'Horizon' in long else None
        suffix = f", 실제 평가 대상 {n_months:,}개월" if n_months is not None else ''
        suffix += f", 최대 {int(horizon)}개월 앞" if pd.notna(horizon) else ''
        parts.append(f"장기: 시작점 {starts.min():%Y-%m}~{starts.max():%Y-%m}, 시작점 {n_origins:,}개{suffix}, 평가 행 {len(long):,}개")
    return '<br>'.join(escape(part) for part in parts)


def validation_manifest(short: pd.DataFrame | None, long: pd.DataFrame | None,
                        asof: pd.Timestamp, source_label: str) -> dict[str, str | int]:
    """Machine-readable audit fields for the client report and UI."""
    return {
        '기준 관측월': pd.Timestamp(asof).strftime('%Y-%m'),
        '평가자료': source_label,
        '단기 평가행': 0 if short is None else len(short),
        '장기 평가행': 0 if long is None else len(long),
        '단기 모델': 'SARIMAX AR(3), d=1, MA(1), 계절(0,1,1,12), 코로나 월별 더미',
        '장기 모델': 'SARIMAX AR(6), d=1, MA(1), 계절(0,1,1,12), 코로나 월별 더미',
        '평가 단위': '명목 백만 달러',
    }

def _summary_frame(data: pd.DataFrame | None, kind: str) -> pd.DataFrame:
    if data is None or data.empty:
        return pd.DataFrame()
    required = {"Model", "APE", "Year" if kind == "short" else "Origin"}
    if not required.issubset(data.columns):
        raise ValueError(f"{kind} 성능 데이터에 필수 열이 없습니다.")
    frame = data.copy()
    frame["APE"] = pd.to_numeric(frame["APE"], errors="raise")
    if not frame["APE"].map(math.isfinite).all() or frame["APE"].lt(0).any():
        raise ValueError("성능 데이터에 유효하지 않은 오차율이 있습니다.")
    frame["Model"] = frame["Model"].replace({"앙상블_50_50": "앙상블 50:50", "앙상블": "앙상블 50:50"})
    if kind == "short":
        frame["Year"] = pd.to_numeric(frame["Year"], errors="raise").astype(int)
        past = frame.loc[frame.Year.between(2022, 2025)].groupby("Model").APE.mean()
        recent = frame.loc[frame.Year.eq(2026)].groupby("Model").APE.mean()
        out = pd.concat([past.rename("2022~2025 평균 MAPE"), recent.rename("2026 MAPE")], axis=1)
    else:
        frame["Origin"] = frame["Origin"].astype(str)
        origin_means = frame.groupby(["Origin", "Model"]).APE.mean().unstack()
        past = origin_means.drop(index="2026-01", errors="ignore").mean()
        recent = origin_means.loc["2026-01"] if "2026-01" in origin_means.index else pd.Series(dtype=float)
        out = pd.concat([past.rename("과거 시작점 평균 MAPE"), recent.rename("2026-01 시작점 MAPE")], axis=1)
    return out.sort_index()


def _html_table(table: pd.DataFrame) -> str:
    if table.empty:
        return '<p class="empty">검증 데이터가 없어 성능을 제시하지 않았습니다.</p>'
    headers = "".join(f"<th>{escape(str(c))}</th>" for c in table.columns)
    rows = []
    for model, values in table.iterrows():
        cells = "".join(f"<td>{float(v):.3f}%</td>" if pd.notna(v) else '<td>평가 불가</td>' for v in values)
        rows.append(f"<tr><th>{escape(str(model))}</th>{cells}</tr>")
    return f'<table><thead><tr><th>모델</th>{headers}</tr></thead><tbody>{"".join(rows)}</tbody></table>'


def make_performance_report(short: pd.DataFrame | None, long: pd.DataFrame | None,
                            asof: pd.Timestamp, source_label: str) -> bytes:
    """For print/save-as-PDF, no artificial headline accuracy values."""
    st = _summary_frame(short, "short")
    lg = _summary_frame(long, "long")
    if st.empty and lg.empty:
        raise ValueError("리포트 생성을 위해 실제 백테스트 결과가 필요합니다.")
    safe_source = escape(source_label)
    safe_date = escape(pd.Timestamp(asof).strftime("%Y-%m"))
    inventory = _validation_inventory(short, long)
    generated_at = escape(datetime.now().strftime('%Y-%m-%d %H:%M'))
    html = f'''<!doctype html><html lang="ko"><head><meta charset="utf-8">
    <meta name="viewport" content="width=device-width,initial-scale=1">
    <title>AutoOrder Lens | 모델 성능 검증 보고서</title><style>
    :root{{--ink:#182C46;--sub:#60738D;--line:#DEE7F1}}
    *{{box-sizing:border-box}}body{{font-family:'Malgun Gothic',Arial,sans-serif;color:var(--ink);margin:0;background:#F4F7FA}}
    main{{max-width:920px;margin:32px auto;padding:48px;background:white}}
    .brand{{letter-spacing:.14em;color:#3464C7;font-size:11px;font-weight:800}}
    h1{{font-size:30px;margin:12px 0 10px}}h2{{font-size:20px;margin:36px 0 12px}}
    .meta{{color:var(--sub);font-size:13px;line-height:1.9;padding:14px 0;border-bottom:1px solid var(--line)}}
    p,li{{line-height:1.85;font-size:14px;color:#40566F}}table{{border-collapse:collapse;width:100%;margin:18px 0 30px}}
    th,td{{padding:12px 14px;border-bottom:1px solid var(--line);text-align:right;font-size:13px}}
    th:first-child{{text-align:left}}thead{{background:#F0F4FA}}.fine{{color:#78879A;font-size:12px}}
    .note{{border-left:3px solid #3464C7;background:#F2F6FC;padding:12px 18px}}
    @media print{{body{{background:white}}main{{padding:0;margin:0;max-width:none}}@page{{size:A4;margin:17mm}}}}
    </style></head><body><main>
    <div class="brand">AUTOORDER LENS / MODEL ASSURANCE</div>
    <h1>수요예측 모델 성능 검증 보고서</h1>
    <div class="meta">기준 관측월: {safe_date} &nbsp; | &nbsp; 평가자료: {safe_source}<br>
    대상 지표: UBTPNO · 미국 자동차 차체·트레일러·부품 제조업 신규주문액 (명목 백만 달러)<br>
    보고서 생성: {generated_at} · 검증자료 실적 기준월: {safe_date}</div>
    <h2>0. 평가 범위와 데이터 계보</h2>
    <p>평가자료: {safe_source}<br>{inventory}</p>
    <p>단기 운용: AR3, 1개월 예측·연 1회 계수 재학습·매월 실제값 반영.<br>
    장기 운용: AR6, 12개월 고정시점 예측·예측 시작월마다 과거 실적만 사용.<br>
    공통 계절 구조: 12개월 계절 차분 및 계절 MA(1), 2020년 4~7월 코로나 더미.</p>
    <h2>1. 평가 방법</h2>
    <p>단기 예측은 매년 1회 계수를 재추정하고 월별 신규 실적을 반영하여 다음 1개월을 예측합니다.
    장기 예측은 예측 시작점에서 향후 12개월을 실제값 갱신 없이 연속 예측합니다.
    MAPE는 실적 대비 절대백분율오차의 평균이며 수치가 낮을수록 해당 평가구간의 오차가 작습니다.</p>
    <h2>2. 단기 예측 성능</h2>{_html_table(st)}
    <h2>3. 장기 예측 성능</h2>{_html_table(lg)}
    <h2>4. 결과 해석 및 한계</h2>
    <div class="note"><p><b>운영 선택:</b> 단기 AR3 / 장기 AR6. 위 표는 연결된 평가자료에서 계산한 값이며,
    모든 기간에서 우위가 보장되는 결과는 아닙니다.</p></div>
    <ul><li>과거 실적은 공표 당시 빈티지가 아닌 현재 취득한 개정 시계열일 수 있습니다.</li>
    <li>12개월 전망 시작점별 평가기간이 겹치므로 결과를 독립 표본의 반복 성공으로 해석할 수 없습니다.</li>
    <li>모델과 경고 기준은 기존 평가기간을 탐색적으로 활용하여 선정되었습니다.</li>
    <li>2026년 등 최신 기간은 12개월 평가가 완결되지 않았을 수 있으며, 평가 월 수와 비교기간을 확인해야 합니다.</li>
    <li>외부 CSV를 연결한 경우 산출물과 원천 시계열의 동일 버전·빈티지를 보장하지 않습니다.</li>
    <li>산업 전체 명목 주문액을 예측하며 기업별 주문 수량, 생산량, 재고·이익을 추정하지 않습니다.</li></ul>
    <p class="fine">출처: U.S. Census Bureau M3 via FRED (UBTPNO). 이 보고서는 연결된 실제 평가 데이터로 생성됩니다.
    브라우저에서 인쇄 → PDF 저장으로 보관할 수 있습니다.</p>
    </main></body></html>'''
    return html.encode("utf-8")