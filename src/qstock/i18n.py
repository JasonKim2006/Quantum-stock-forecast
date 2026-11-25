"""Korean / English text for the dashboard and its charts, plus number and date formatting."""

from __future__ import annotations

import pandas as pd

_SYMBOLS = {"USD": "$", "CAD": "C$", "KRW": "₩", "JPY": "¥", "EUR": "€", "GBP": "£", "HKD": "HK$"}
_NO_DECIMALS = {"KRW", "JPY"}


def money(currency: str, x: float) -> str:
    sym = _SYMBOLS.get(currency, "")
    body = f"{x:,.0f}" if currency in _NO_DECIMALS else f"{x:,.2f}"
    return f"{sym}{body}" if sym else f"{body} {currency}"


def pct(x: float, digits: int = 1) -> str:
    return f"{x * 100:.{digits}f}%"


def date(lang: str, d) -> str:
    d = pd.Timestamp(d)
    if lang == "ko":
        return f"{d.year}년 {d.month}월 {d.day}일"
    return f"{d:%b} {d.day}, {d.year}"


def t(lang: str, key: str, **kw) -> str:
    """Look up a string. Only formatted when arguments are given, because Plotly
    hover templates use %{...} placeholders of their own."""
    s = STR[key][lang]
    return s.format(**kw) if kw else s


STR: dict[str, dict[str, str]] = {
    # ------------------------------------------------------------------ page
    "brand": {"ko": "양자 주가 예측", "en": "Quantum Stock Forecast"},
    "tagline": {"ko": '이 종목이 실제로 움직여 온 기록으로 앞으로의 가격 범위와 그 확률을 보여 줘요. 같은 계산을 Qiskit 양자 회로로도 해요.', "en": 'See the range a stock is likely to trade in, built from how it has actually moved, and how likely each part is. The same numbers are also computed on a Qiskit quantum circuit.'},
    "settings": {"ko": "설정", "en": "Settings"},
    "demo_toggle": {"ko": "데모 데이터로 보기 (인터넷 없이)", "en": "Use demo data (works offline)"},
    "demo_help": {
        "ko": "장이 닫혀 있거나 Yahoo Finance에 접속이 안 될 때 써요. 가상의 시장 데이터예요.",
        "en": "For when the market is closed or Yahoo Finance is unreachable. The data is simulated.",
    },
    "ticker_label": {"ko": "종목", "en": "Stock"},
    "ticker_ph": {"ko": "티커 입력 (예: AAPL, 005930.KS)", "en": "Enter a ticker (e.g. AAPL, SHOP.TO)"},
    "popular": {"ko": "많이 찾는 종목", "en": "Popular"},
    "horizon_label": {"ko": "언제의 가격이 궁금하세요?", "en": "How far ahead?"},
    "h7": {"ko": "1주 후", "en": "1 week"},
    "h30": {"ko": "1개월 후", "en": "1 month"},
    "h91": {"ko": "3개월 후", "en": "3 months"},
    "h182": {"ko": "6개월 후", "en": "6 months"},
    "h365": {"ko": "1년 후", "en": "1 year"},
    "demo_banner": {"ko": "데모 데이터예요. 실제 시장 가격이 아니에요.", "en": "Demo data. These are not real market prices."},
    "live_caption": {
        "ko": "Yahoo Finance 기준이고 약 15분 늦은 가격이에요. {date} 종가까지 반영했어요.",
        "en": "Prices from Yahoo Finance, about 15 minutes delayed, through the close on {date}.",
    },
    "load_error": {
        "ko": "{ticker} 데이터를 불러오지 못했어요. 티커 철자를 확인하거나, 오른쪽 위 설정에서 데모 데이터를 켜 보세요.",
        "en": "Couldn't load {ticker}. Check the spelling, or turn on demo data under Settings (top right).",
    },
    "headline": {
        "ko": "{date}, {ticker}의 가격은 90% 확률로 {lo}에서 {hi} 사이에 있을 거예요.",
        "en": "By {date}, {ticker} has a 90% chance of trading between {lo} and {hi}.",
    },
    "sub_more": {
        "ko": "지금 가격은 {spot}이에요. 옵션 시장은 지난 1년보다 큰 움직임을 예상하고 있어요.",
        "en": "It trades at {spot} now. The options market expects bigger moves than the past year.",
    },
    "sub_less": {
        "ko": "지금 가격은 {spot}이에요. 옵션 시장은 지난 1년보다 작은 움직임을 예상하고 있어요.",
        "en": "It trades at {spot} now. The options market expects smaller moves than the past year.",
    },
    "sub_noopt": {
        "ko": "지금 가격은 {spot}이에요. 이 종목은 쓸 수 있는 옵션 가격이 없어서 지난 1년의 움직임으로 계산했어요.",
        "en": "It trades at {spot} now. No usable option prices for this stock, so the forecast uses the past year's moves.",
    },
    "disclaimer": {
        "ko": "오를지 내릴지를 맞히는 모델은 아니에요. 가격이 움직일 수 있는 범위와 그 확률을 보여 줘요. 투자 조언이 아니에요.",
        "en": "This model doesn't call direction. It shows how far the price could move and how likely each range is. "
        "Not investment advice.",
    },
    "chart_hint": {
        "ko": "그래프 위에 마우스를 올리면 각 선과 영역이 무엇인지 설명이 나와요.",
        "en": "Hover over the chart to see what each line and shaded area means.",
    },
    # ------------------------------------------------------------------ facts
    "fact_above": {"ko": "{target} 이상으로 끝날 확률", "en": "Chance it ends above {target}"},
    "fact_below": {"ko": "{target} 이하로 끝날 확률", "en": "Chance it ends below {target}"},
    "target_slider": {"ko": "목표 가격 (지금 대비)", "en": "Target (vs. today)"},
    "fact_chance_note": {
        "ko": "양자 회로로 계산하면 {q}예요 (95% 신뢰구간 {lo}–{hi}).",
        "en": "The quantum circuit gets {q} (95% CI {lo}–{hi}).",
    },
    "fact_bt": {"ko": "과거 예측이 맞은 비율", "en": "How often past forecasts held"},
    "fact_bt_note": {
        "ko": "과거 {n}번의 예측에서 실제 가격이 90% 범위 안에 들어간 비율이에요. 90%에 가까울수록 믿을 만해요.",
        "en": "Share of {n} past forecasts where the real price landed in the 90% range. Closer to 90% is better.",
    },
    "fact_swing": {"ko": "보통 움직이는 폭", "en": "Typical swing"},
    "fact_swing_note": {
        "ko": "{date}까지 이 정도(±1 표준편차)는 흔하게 움직여요. 연간 변동성은 {vol}이에요.",
        "en": "A move this size (±1 standard deviation) by {date} is common. Annual volatility is {vol}.",
    },
    # ------------------------------------------------------------------ tabs
    "tab_detail": {"ko": "예측 자세히", "en": "Forecast detail"},
    "tab_options": {"ko": "옵션 가격", "en": "Option prices"},
    "tab_track": {"ko": "얼마나 정확했나", "en": "Track record"},
    "tab_quantum": {"ko": "양자 계산", "en": "Quantum circuit"},
    "col_range": {"ko": "범위", "en": "Range"},
    "col_implied": {"ko": "옵션 시장 기준", "en": "From option prices"},
    "col_hist": {"ko": "지난 1년 기준", "en": "From the past year"},
    "row_range": {"ko": "{lv} 범위", "en": "{lv} range"},
    "how_title": {"ko": "이 숫자들은 어떻게 계산했나요?", "en": "How are these numbers calculated?"},
    "how_body": {"ko": "**과거 주가 움직임 (Filtered Historical Simulation)**\n\n1. 최근 변동성을 지수가중 이동평균(EWMA, λ=0.94)으로 재요. 최근 움직임에 더 큰 비중을 둬요.\n2. 과거의 하루 수익률을 그때의 변동성으로 나눠서 '모양'만 남겨요. 급락·급등이 얼마나 자주 있었는지가 여기 담겨요.\n3. 이 모양들을 무작위로 다시 뽑아 오늘의 변동성으로 키우고, 예측 날짜까지 이어 붙여요. 변동성은 시간이 지나면서 장기 평균으로 돌아가게 했어요(반감기 30거래일). 이렇게 가격 경로 2만 개를 만들어요.\n\n**종 모양 (지난 1년)**: 로그 수익률이 정규분포를 따른다고 가정하고, 지난 252거래일의 변동성 하나만 써요.\n\n두 방식 모두 오를지 내릴지에 대한 방향은 가정하지 않아요.", "en": "**Past price moves (Filtered Historical Simulation)**\n\n1. Today's volatility comes from an exponentially weighted average of squared returns (EWMA, λ = 0.94), which weights recent days more.\n2. Each past daily return is divided by the volatility at the time, leaving only its shape. How often crashes and spikes happened lives here.\n3. Those shapes are redrawn at random, scaled by today's volatility, and chained to the forecast date, with volatility drifting back to its long-run level (half-life 30 trading days). That makes 20,000 price paths.\n\n**Bell curve (past year)**: assumes normally distributed log returns with the last 252 trading days' volatility.\n\nNeither assumes the price will go up or down."},
    "opt_title": {"ko": "시장 옵션 가격과 모델 비교 ({date} 만기)", "en": "Market option prices vs. the model (expiring {date})"},
    "opt_none": {
        "ko": "이 종목은 Yahoo Finance에 쓸 수 있는 옵션 가격이 없어요. 미국 대형주(AAPL, NVDA, SPY 등)로 바꿔 보세요.",
        "en": "No usable option prices on Yahoo Finance for this stock. Try a large US stock such as AAPL, NVDA or SPY.",
    },
    "opt_cols": {"ko": "행사가|매수 호가|매도 호가|시장 가격|시장 변동성|모델 (시장 변동성)|모델 (지난 1년)|거래량",
                 "en": "Strike|Bid|Ask|Market price|Market vol|Model (market vol)|Model (past year)|Volume"},
    "opt_explain": {
        "ko": "콜옵션은 만기일에 정해진 가격(행사가)으로 주식을 살 권리예요. 시장 가격은 매수·매도 호가의 중간값이에요. "
        "모델(시장 변동성)이 행사가에서 멀어질수록 시장과 어긋나는 건, 시장이 행사가마다 다른 변동성(변동성 스마일)을 쓰기 때문이에요.",
        "en": "A call option is the right to buy the stock at the strike price on expiry. Market price is the bid/ask midpoint. "
        "The model drifts from the market away from today's price because the market uses a different volatility for each strike "
        "(the volatility smile).",
    },
    "opt_q_title": {"ko": "한 행사가를 양자 회로로 계산하기", "en": "Price one strike on the quantum circuit"},
    "opt_strike": {"ko": "행사가", "en": "Strike"},
    "m_market": {"ko": "시장 가격", "en": "Market price"},
    "m_formula": {"ko": "공식 (Black–Scholes)", "en": "Formula (Black–Scholes)"},
    "m_quantum": {"ko": "양자 계산", "en": "Quantum circuit"},
    "m_calls": {"ko": "회로 실행 횟수", "en": "Circuit runs"},
    "m_calls_help": {
        "ko": "상태 준비 회로 A를 몇 번 실행했는지예요. 몬테카를로의 샘플 수에 해당해요.",
        "en": "How many times the state-preparation circuit A ran. It plays the role of Monte Carlo's sample count.",
    },
    # track record
    "bt_title": {"ko": "과거에 이 모델을 썼다면 얼마나 맞았을까요?", "en": "How would this model have done in the past?"},
    "bt_body": {"ko": '과거 {n}개의 날짜마다 그 날까지의 주가만 써서 {h}거래일 뒤를 예측하고, 실제 가격이 어디에 떨어졌는지 확인했어요. 두 방식 모두 같은 방법으로 검사했어요. 기간이 겹쳐서 서로 독립적인 검사는 약 {ind}번이에요.', "en": 'On each of {n} past dates, each method forecast {h} trading days ahead using only prices known on that date; then we checked where the price really landed. Overlapping windows make this about {ind} independent checks.'},
    "bt_cols": {"ko": '예측 범위|판정|정상으로 볼 수 있는 범위', "en": 'Forecast range|Verdict|Expected if well calibrated'},
    "v_ok": {"ko": "믿을 만해요", "en": "Holds up"},
    "v_narrow": {"ko": "범위가 너무 좁았어요", "en": "Too narrow"},
    "v_wide": {"ko": "범위가 너무 넓었어요", "en": "Too wide"},
    "bt_tail": {"ko": "아주 큰 움직임 (3σ 넘음)", "en": "Very large moves (beyond 3σ)"},
    "bt_tail_note": {
        "ko": "정규분포 모델은 0.27%를 예상해요. 이보다 많으면 모델이 급락·급등을 과소평가한다는 뜻이에요.",
        "en": "A normal model expects 0.27%. More than that means the model underestimates crashes and spikes.",
    },
    # quantum
    "q_title": {"ko": "양자 컴퓨터에서 무슨 일이 일어나나요?", "en": "What runs on the quantum computer?"},
    "q_steps": {
        "ko": "1. **분포 올리기**: 큐비트 {n}개로 가격을 {cells}칸으로 나누고, 각 칸의 확률을 양자 상태의 진폭에 담아요.\n"
        "2. **결과 표시하기**: 큐비트 하나를 더 써서, 각 칸에서 조건(예: 목표 가격 이상)이 맞는 정도만큼 회전시켜요. "
        "그러면 이 큐비트가 1로 측정될 확률이 우리가 구하려는 확률과 같아져요.\n"
        "3. **진폭 추정(IAE)**: 그로버 반복으로 그 확률을 증폭해서 정밀하게 읽어요. 오차가 몬테카를로(1/√N)보다 빠르게(약 1/N) 줄어요.",
        "en": "1. **Load the distribution**: {n} qubits split the price into {cells} cells, and each cell's probability "
        "goes into a quantum amplitude.\n"
        "2. **Mark the outcome**: one extra qubit is rotated in each cell by how much the condition (e.g. above target) holds, "
        "so the chance of measuring it as 1 equals the probability we want.\n"
        "3. **Amplitude estimation (IAE)**: Grover iterations amplify that probability and read it precisely. The error falls "
        "about like 1/N instead of Monte Carlo's 1/√N.",
    },
    "q_qubits": {"ko": "큐비트", "en": "Qubits"},
    "q_cx": {"ko": "CNOT 게이트", "en": "CNOT gates"},
    "q_depth": {"ko": "회로 깊이", "en": "Circuit depth"},
    "q_advanced": {"ko": "고급 설정", "en": "Advanced settings"},
    "q_n": {"ko": "가격 칸을 나눌 큐비트 수", "en": "Qubits for price cells"},
    "q_n_help": {"ko": "n개면 2ⁿ칸이에요. 많을수록 정밀하지만 회로가 커져요.", "en": "n qubits give 2ⁿ cells. More is finer but the circuit grows."},
    "q_eps": {"ko": "목표 정밀도 ε", "en": "Target precision ε"},
    "q_eps_help": {"ko": "작을수록 정확하지만 회로를 더 많이 실행해요.", "en": "Smaller is more accurate but runs the circuit more times."},
    "q_run": {"ko": "양자 vs 몬테카를로 수렴 비교 실행", "en": "Compare convergence: quantum vs Monte Carlo"},
    "q_run_caption": {
        "ko": "두 방법 모두 샘플링 오차만 비교해요. 양자 쪽은 회로가 담은 정확한 값, 몬테카를로 쪽은 공식 값과 비교했어요. "
        "노이즈 없는 시뮬레이션 결과예요.",
        "en": "Both lines show sampling error only: the quantum one against the circuit's exact value, Monte Carlo against "
        "the formula. Noise-free simulation.",
    },
    "spinner_q": {"ko": "양자 진폭 추정 실행 중...", "en": "Running amplitude estimation..."},
    "spinner_data": {"ko": "시장 데이터를 불러오는 중...", "en": "Loading market data..."},
    # ------------------------------------------------------------------ chart hovers
    "c_price": {"ko": "실제 가격", "en": "Actual price"},
    "c_price_hover": {"ko": "<b>실제 종가</b><br>%{x|%Y-%m-%d}<br>%{customdata}", "en": "<b>Actual close</b><br>%{x|%b %d, %Y}<br>%{customdata}"},
    "c_today": {"ko": "오늘", "en": "Today"},
    "c_today_hover": {"ko": "<b>오늘 가격</b><br>{spot}<br>예측은 여기서 시작해요", "en": "<b>Today's price</b><br>{spot}<br>The forecast starts here"},
    "c_band": {"ko": "{lv} 확률 범위", "en": "{lv} range"},
    "c_core_hover": {
        "ko": "<b>가장 가능성이 높은 영역 (50% 범위)</b><br>{date}에 가격이 절반의 확률로<br>{lo} – {hi} 사이에 있어요",
        "en": "<b>Most likely zone (50% range)</b><br>On {date} there's an even chance<br>the price is between {lo} and {hi}",
    },
    "c_band_hover": {
        "ko": "<b>{lv} 확률 범위의 가장자리</b><br>{date}에 가격이 {lv} 확률로<br>{lo} – {hi} 사이에 있어요<br><i>바깥쪽일수록 가능성이 낮아요</i>",
        "en": "<b>Edge of the {lv} range</b><br>On {date} the price has a {lv} chance<br>of being between {lo} and {hi}<br><i>Fainter means less likely</i>",
    },
    "c_median": {"ko": "가운데 값", "en": "Middle path"},
    "c_median_hover": {
        "ko": "<b>가운데 값 (중앙값)</b><br>%{x|%Y-%m-%d}: %{customdata}<br>이보다 높을 확률과 낮을 확률이 반반이에요",
        "en": "<b>Middle path (median)</b><br>%{x|%b %d}: %{customdata}<br>Equally likely to end above or below",
    },
    "c_hist": {"ko": "지난 1년 기준 90% 범위", "en": "90% range from the past year"},
    "c_hist_hover": {
        "ko": "<b>지난 1년 움직임 기준 90% 범위</b><br>%{x|%Y-%m-%d}: %{customdata}<br>옵션 시장 대신 과거 변동성을 쓰면 이 선이 경계예요",
        "en": "<b>90% range from the past year's moves</b><br>%{x|%b %d}: %{customdata}<br>The boundary if we use past volatility instead of options",
    },
    "c_target": {"ko": "목표 가격", "en": "Target"},
    "c_target_hover": {
        "ko": "<b>목표 가격 {target}</b><br>{date}에 이보다 {side} 끝날 확률: {p}",
        "en": "<b>Target {target}</b><br>Chance it ends {side} this on {date}: {p}",
    },
    "c_above": {"ko": "높게", "en": "above"},
    "c_below": {"ko": "낮게", "en": "below"},
    "c_smile_title": {"ko": "행사가별 시장 변동성 (변동성 스마일)", "en": "Market volatility by strike (volatility smile)"},
    "c_smile_hover": {
        "ko": "<b>행사가 %{x}</b><br>시장이 쓰는 변동성: %{y:.1%}<br>이 행사가의 옵션 가격에서 거꾸로 계산했어요",
        "en": "<b>Strike %{x}</b><br>Volatility the market uses: %{y:.1%}<br>Backed out of this strike's option price",
    },
    "c_smile_hist_hover": {
        "ko": "<b>지난 1년 실제 변동성</b><br>{vol}<br>시장 변동성이 이보다 높으면 시장이 더 큰 움직임을 예상한다는 뜻",
        "en": "<b>Past-year volatility</b><br>{vol}<br>Market vol above this means the market expects bigger moves",
    },
    "c_spot_hover": {"ko": "<b>현재 가격</b><br>{spot}", "en": "<b>Current price</b><br>{spot}"},
    "c_vol": {"ko": "변동성", "en": "Volatility"},
    "c_strike": {"ko": "행사가", "en": "Strike"},
    "c_cal_title": {"ko": "예측 범위 vs 실제 적중률", "en": "Forecast range vs. actual hit rate"},
    "c_cal_x": {"ko": "예측 범위의 크기", "en": "Forecast range size"},
    "c_cal_y": {"ko": "실제 가격이 범위 안에 든 비율", "en": "Share of prices inside the range"},
    "c_cal_ideal": {"ko": "완벽한 예측", "en": "Perfect forecast"},
    "c_cal_ideal_hover": {"ko": "<b>완벽한 예측선</b><br>점이 이 선 위에 있으면<br>범위 크기만큼 정확히 맞혔다는 뜻", "en": "<b>Perfect forecast</b><br>Points on this line hit<br>exactly as often as promised"},
    "c_cal_hover": {
        "ko": "<b>%{x:.0%} 범위</b><br>실제로 맞은 비율: %{y:.1%}<br>선 아래면 범위가 좁았고, 위면 넓었어요",
        "en": "<b>%{x:.0%} range</b><br>Actual hit rate: %{y:.1%}<br>Below the line means too narrow; above, too wide",
    },
    "c_pit_title": {"ko": "실제 가격이 예측 분포의 어디에 떨어졌나", "en": "Where real prices fell in the forecast"},
    "c_pit_x": {"ko": "예측 분포에서의 위치 (0 = 가장 낮은 쪽, 1 = 가장 높은 쪽)", "en": "Position in the forecast (0 = lowest, 1 = highest)"},
    "c_pit_y": {"ko": "비율", "en": "Share"},
    "c_pit_hover": {
        "ko": "<b>예측 분포의 %{customdata}</b><br>이 구간에 떨어진 비율: %{y:.1%}<br>모든 막대가 10% 높이면 이상적이에요",
        "en": "<b>Forecast slice %{customdata}</b><br>Share of outcomes here: %{y:.1%}<br>Ideally every bar is at 10%",
    },
    "c_pit_ideal": {"ko": "이상적인 높이", "en": "Ideal height"},
    "c_conv_title": {"ko": "계산량에 따른 오차", "en": "Error vs. effort"},
    "c_conv_x": {"ko": "샘플 수 / 회로 실행 횟수", "en": "Samples / circuit runs"},
    "c_conv_y": {"ko": "평균 오차 (RMS)", "en": "Average error (RMS)"},
    "c_mc": {"ko": "몬테카를로 (기울기 {s})", "en": "Monte Carlo (slope {s})"},
    "c_q": {"ko": "양자 IAE (기울기 {s})", "en": "Quantum IAE (slope {s})"},
    "c_mc_hover": {
        "ko": "<b>몬테카를로</b><br>샘플 %{x:,.0f}개 → 오차 %{y:.2e}<br>샘플을 100배 늘려야 오차가 10배 줄어요",
        "en": "<b>Monte Carlo</b><br>%{x:,.0f} samples → error %{y:.2e}<br>100× the samples for 10× less error",
    },
    "c_q_hover": {
        "ko": "<b>양자 진폭 추정</b><br>회로 %{x:,.0f}번 실행 → 오차 %{y:.2e}<br>실행을 10배 늘리면 오차도 약 10배 줄어요",
        "en": "<b>Quantum amplitude estimation</b><br>%{x:,.0f} circuit runs → error %{y:.2e}<br>10× the runs for about 10× less error",
    },

    # ------------------------------------------------------------------ market distribution
    "method_label": {"ko": "예측 방식", "en": "Forecast method"},
    "m_market": {"ko": "시장 분포 (옵션 전체)", "en": "Market distribution (all options)"},
    "m_iv": {"ko": "종 모양 (시장 변동성)", "en": "Bell curve (market volatility)"},
    "m_hv": {"ko": "종 모양 (지난 1년)", "en": "Bell curve (past year)"},
    "method_help": {"ko": '과거 주가 움직임: 이 종목이 실제로 움직였던 하루 수익률들을 최근 변동성에 맞춰 다시 섞어서 미래 경로 2만 개를 만들어요. 종 모양을 가정하지 않아서 급락·급등이 실제 빈도대로 들어가요.\n\n종 모양 (지난 1년): 지난 1년의 평균 변동성 하나로 그린 대칭 분포예요. 비교용이에요.', "en": "Past price moves: re-mixes the stock's real daily returns, scaled to today's volatility, into 20,000 future paths. No bell-curve assumption, so crashes and spikes appear as often as they really did.\n\nBell curve (past year): a symmetric shape from one average volatility number, for comparison."},
    "note_expiry": {
        "ko": "시장 분포는 옵션 만기일에 묶여 있어서 날짜를 가장 가까운 만기일인 {date}로 맞췄어요.",
        "en": "The market distribution is tied to an option expiry, so the date is set to the nearest one, {date}.",
    },
    "note_fallback": {
        "ko": "옵션 가격이 부족하거나 고르지 않아서 시장 분포를 만들 수 없었어요. 대신 종 모양 (시장 변동성)으로 계산했어요.",
        "en": "Not enough clean option prices to build the market distribution, so this uses the bell curve with market volatility.",
    },
    "sub_market": {
        "ko": "지금 가격은 {spot}이에요. 이 범위는 행사가별 옵션 가격 전체에서 꺼낸, 시장이 직접 매긴 분포로 계산했어요.",
        "en": "It trades at {spot} now. This range comes from the full set of option prices: the distribution the market itself is pricing.",
    },
    "sub_hist": {
        "ko": "지금 가격은 {spot}이에요. 이 범위는 지난 1년 동안 실제로 움직인 폭으로 계산했어요.",
        "en": "It trades at {spot} now. This range uses how much the stock actually moved over the past year.",
    },
    "dens_title": {"ko": "예측 방식별 가격 분포 비교", "en": "Price distribution by method"},
    "dens_body": {"ko": '같은 날짜의 가격 분포를 방식별로 겹쳐 봤어요. 과거 주가 움직임 모델은 종 모양을 가정하지 않기 때문에 이 종목이 실제로 겪었던 급락·급등의 빈도가 그대로 들어가요. 최근에 시장이 출렁였다면 분포도 넓어져요.', "en": "The price distribution on the same date, one curve per method. The past-price-moves model assumes no bell curve, so the stock's real frequency of crashes and spikes carries through, and a recently stormy market widens it."},
    "tail_cols": {"ko": "이런 일이 생길 확률|시장 분포|종 모양 (시장 변동성)|종 모양 (지난 1년)",
                  "en": "Chance of|Market distribution|Bell curve (market vol)|Bell curve (past year)"},
    "tail_drop": {"ko": "{p} 이상 하락", "en": "Falling {p} or more"},
    "tail_rise": {"ko": "{p} 이상 상승", "en": "Rising {p} or more"},
    "clip_note": {
        "ko": "옵션 호가끼리의 작은 불일치 때문에 확률 {p}를 보정했어요.",
        "en": "Small inconsistencies between quotes required adjusting {p} of probability.",
    },
    "c_dens_title": {"ko": "예측 날짜의 가격 분포", "en": "Price distribution on the forecast date"},
    "c_dens_x": {"ko": "가격", "en": "Price"},
    "c_dens_y": {"ko": "가능성", "en": "Likelihood"},
    "c_dens_hover": {"ko": "가격 %{customdata} 근처<br>곡선이 높을수록 그 가격이 나올 가능성이 커요",
                     "en": "Around %{customdata}<br>The higher the curve, the more likely that price"},
    "c_dens_spot": {"ko": "<b>지금 가격</b><br>{v}", "en": "<b>Price today</b><br>{v}"},
    "c_dens_target": {"ko": "<b>목표 가격</b><br>{v}", "en": "<b>Target</b><br>{v}"},
    "dn_market": {"ko": "옵션 가격 전체에서 계산했어요. 급락 쪽 꼬리가 두꺼운 게 보통이에요",
                  "en": "Built from all option prices; the downside tail is usually fatter"},
    "dn_iv": {"ko": "옵션 시장의 변동성 하나로 그린 종 모양", "en": "Bell curve from the options market's single volatility"},
    "dn_hv": {"ko": "지난 1년 움직임으로 그린 종 모양", "en": "Bell curve from the past year's moves"},
    # ------------------------------------------------------------------ IBM hardware
    "hw_title": {"ko": "실제 IBM 양자 컴퓨터에서 실행하기", "en": "Run on a real IBM quantum computer"},
    "hw_intro": {"ko": "위에서 고른 '{target} 이상으로 끝날 확률'을 **진짜 양자 컴퓨터에게 풀게 하고**, 일반 컴퓨터가 계산한 정답과 비교해요.\n\n회로는 두 가지를 보내요. 확률을 바로 재는 **짧은 회로**, 그리고 더 정밀한 계산에 필요한 '확대' 단계를 넣은 **긴 회로**예요. 두 결과를 비교하면 지금 양자 컴퓨터가 어디까지 정확한지 볼 수 있어요.\n\n먼저 **무료 미리보기**로 IBM 기계의 오류를 흉내 낸 시뮬레이션을 해 보고, 괜찮으면 실제 기계로 보내세요.", "en": "Have a **real quantum computer** work out 'the chance it ends above {target}' (chosen above) and compare it with the exact answer from a regular computer.\n\nTwo circuits go out: a **short circuit** that measures the probability directly, and a **longer circuit** with the 'amplify' step that more precise answers need. Comparing them shows how accurate today's quantum computers are.\n\nTry the **free preview** first, which simulates an IBM machine's errors, then send it to the real machine."},
    "hw_n": {"ko": "가격 칸 큐비트 수", "en": "Qubits for price cells"},
    "hw_n_help": {"ko": "실제 하드웨어에서는 회로가 짧을수록 정확해요. 3을 추천해요.", "en": "On hardware, shorter circuits are more accurate. 3 is recommended."},
    "hw_shots": {"ko": "측정 횟수 (shots)", "en": "Shots"},
    "hw_grover": {"ko": '긴 회로(확대 단계)도 같이 보내기', "en": 'Also send the longer (amplify-step) circuit'},
    "hw_preview_btn": {"ko": "무료 미리보기 (IBM 노이즈로 시뮬레이션)", "en": "Free preview (simulate with IBM noise)"},
    "hw_preview_help": {"ko": "IBM 기계(ibm_fez)의 오류 데이터로 내 컴퓨터에서 시뮬레이션해요. 사용 시간을 쓰지 않아요.",
                        "en": "Simulates on your computer with an IBM machine's (ibm_fez) error data. Uses none of your IBM time."},
    "hw_preview_spinner": {"ko": "IBM 노이즈 모델로 시뮬레이션하는 중이에요 (10~30초)...", "en": "Simulating with IBM's noise model (10–30 s)..."},
    "hw_preview_title": {"ko": "미리보기 결과 (ibm_fez 노이즈 시뮬레이션)", "en": "Preview (simulated ibm_fez noise)"},
    "hw_submit_btn": {"ko": "실제 양자 컴퓨터로 보내기", "en": "Send to the quantum computer"},
    "hw_submit_help": {"ko": "사용 시간 몇 초를 써요. 대기열 때문에 결과가 나오기까지 몇 분에서 몇 시간 걸릴 수 있어요.",
                       "en": "Uses a few seconds of your time. Queues can make results take minutes to hours."},
    "hw_submitting": {"ko": "가장 한가한 기계를 찾아서 보내는 중이에요...", "en": "Finding the least busy machine and sending..."},
    "hw_no_account": {
        "ko": "IBM 계정이 아직 연결되지 않았어요. 터미널에서 계정 저장 명령(save_account)을 먼저 실행해 주세요. 미리보기는 계정 없이도 돼요.",
        "en": "No IBM account is connected yet. Run the save_account command in a terminal first. The preview works without an account.",
    },
    "hw_connected": {"ko": "IBM 계정 연결됨. 남은 사용 시간: 약 {m}분", "en": "IBM account connected. About {m} minutes left"},
    "hw_connected_plain": {"ko": "IBM 계정 연결됨.", "en": "IBM account connected."},
    "hw_submitted": {"ko": "{backend}로 보냈어요. 아래 '보낸 작업'에서 결과를 확인하세요.",
                     "en": "Sent to {backend}. Check the result under 'Sent jobs' below."},
    "hw_submit_fail": {"ko": "보내지 못했어요: {err}", "en": "Couldn't send the job: {err}"},
    "hw_jobs": {"ko": "보낸 작업", "en": "Sent jobs"},
    "hw_jobs_empty": {"ko": "아직 보낸 작업이 없어요.", "en": "No jobs sent yet."},
    "hw_job_label": {"ko": "{when} · {backend} · {ticker} {target} 이상", "en": "{when} · {backend} · {ticker} above {target}"},
    "hw_check": {"ko": "결과 확인", "en": "Check result"},
    "hw_status": {"ko": "상태: {s}", "en": "Status: {s}"},
    "st_QUEUED": {"ko": "대기 중이에요. 잠시 뒤 다시 확인해 주세요.", "en": "Queued. Check again in a while."},
    "st_RUNNING": {"ko": "실행 중이에요.", "en": "Running."},
    "st_INITIALIZING": {"ko": "준비 중이에요.", "en": "Initializing."},
    "st_VALIDATING": {"ko": "검사 중이에요.", "en": "Validating."},
    "st_ERROR": {"ko": "오류로 끝났어요. IBM 대시보드에서 자세한 내용을 볼 수 있어요.", "en": "Ended with an error. See the IBM dashboard for details."},
    "st_CANCELLED": {"ko": "취소됐어요.", "en": "Cancelled."},
    "hw_k0": {"ko": '간단한 회로', "en": 'Short circuit'},
    "hw_k1": {"ko": '확대 단계를 넣은 회로', "en": 'Circuit with an amplify step'},
    "hw_exact": {"ko": '정답 (일반 컴퓨터로 계산)', "en": 'Correct answer (regular computer)'},
    "hw_raw": {"ko": '양자 컴퓨터 결과', "en": 'Quantum computer'},
    "hw_fixed": {"ko": '읽기 오류를 바로잡은 결과', "en": 'Quantum computer, read errors fixed'},
    "hw_exact_hover": {"ko": '<b>정답</b><br>일반 컴퓨터로 정확히 계산한 값: %{y:.1%}<br>양자 컴퓨터가 완벽하다면 이 값이 나와야 해요', "en": '<b>Correct answer</b><br>Computed exactly on a regular computer: %{y:.1%}<br>A perfect quantum computer would give this'},
    "hw_raw_hover": {"ko": '<b>양자 컴퓨터가 낸 결과</b><br>%{y:.1%}<br>회색 막대와 가까울수록 정확해요<br>막대 위 짧은 선은 측정을 반복해서 생기는 흔들림 범위예요', "en": '<b>What the quantum computer gave</b><br>%{y:.1%}<br>The closer to the grey bar, the more accurate<br>The small line on top is the random wobble from repeated measurement'},
    "hw_fixed_hover": {"ko": '<b>읽기 오류를 바로잡은 결과</b><br>%{y:.1%}<br>양자 컴퓨터는 가끔 0을 1로, 1을 0으로 잘못 읽어요.<br>그 비율을 따로 재서 바로잡은 값이에요', "en": '<b>Read errors fixed</b><br>%{y:.1%}<br>Quantum computers sometimes read 0 as 1 or 1 as 0.<br>This corrects for that, measured separately'},
    "hw_chart_title": {"ko": '정답과 양자 컴퓨터 결과 비교', "en": 'Correct answer vs. the quantum computer'},
    "hw_y": {"ko": '결과 값', "en": 'Result'},
    "hw_result": {"ko": '**{target} 이상일 확률**: 정답 {exact}, 양자 컴퓨터 {est}.', "en": '**Chance above {target}**: correct {exact}, quantum computer {est}.'},
    "hw_stats": {
        "ko": "회로 크기: 2큐비트 게이트 {g0}개 (k=0){g1}. 0↔1 측정 오류: {e0} / {e1}. 물리 큐비트 {q}번에서 읽었어요.",
        "en": "Circuit size: {g0} two-qubit gates (k=0){g1}. Readout error 0→1 / 1→0: {e0} / {e1}. Read on physical qubit {q}.",
    },
    "hw_g1": {"ko": ", {g}개 (k=1)", "en": ", {g} (k=1)"},
    "hw_k1_note": {"ko": '오른쪽 회로는 더 정밀한 계산에 필요한 단계를 한 번 더 넣은 거예요. 회로가 길어지면 오류가 쌓여서 정답과 더 크게 어긋나요.', "en": 'The right-hand circuit adds the step needed for more precise answers. Longer circuits pile up errors, so it misses by more.'},
    "hw_used": {"ko": "이 작업에 쓴 양자 컴퓨터 시간: {s}초", "en": "Quantum time used by this job: {s} s"},

    # ------------------------------------------------------------------ stock-price model
    "m_fhs": {"ko": "과거 주가 움직임 (추천)", "en": "Past price moves (recommended)"},
    "method_help_opts": {
        "ko": "과거 주가 움직임: 실제 하루 수익률을 최근 변동성에 맞춰 다시 섞어 미래 경로 2만 개를 만들어요.\n\n"
        "종 모양 (지난 1년): 지난 1년 변동성 하나로 그린 분포예요.\n\n"
        "시장 분포 (옵션 전체): 행사가별 옵션 가격에서 시장이 매긴 분포를 꺼내요.\n\n"
        "종 모양 (시장 변동성): 옵션 시장의 변동성 하나로 그린 분포예요.",
        "en": "Past price moves: real daily returns re-mixed at today's volatility into 20,000 paths.\n\n"
        "Bell curve (past year): one average volatility number.\n\n"
        "Market distribution: the distribution priced into all option strikes.\n\n"
        "Bell curve (market volatility): one volatility number from options.",
    },
    "sub_fhs": {
        "ko": "지금 가격은 {spot}이에요. 이 범위는 이 종목이 실제로 움직였던 하루 {n}개를 지금의 변동성(연 {vol}, 장기 평균 {long})에 맞춰 다시 섞은 가격 경로 2만 개로 만들었어요.",
        "en": "It trades at {spot} now. This range comes from 20,000 price paths that re-mix the stock's {n} real daily moves at today's volatility ({vol} a year; long-run {long}).",
    },
    "opt_toggle": {"ko": "옵션 시장 데이터도 쓰기 (고급)", "en": "Also use options-market data (advanced)"},
    "opt_toggle_help": {
        "ko": "켜면 '옵션 가격' 탭과, 옵션 가격으로 만든 예측 방식 두 가지가 추가돼요. 옵션이 거래되는 미국 종목에서만 돼요.",
        "en": "Adds the Option prices tab and two option-based forecast methods. Works only for US stocks with listed options.",
    },
    "dn_fhs": {"ko": "실제 과거 하루 움직임을 다시 섞어 만든 분포", "en": "Built by re-mixing the stock's real daily moves"},
    "dens_body_opts": {
        "ko": "같은 날짜의 가격 분포를 방식별로 겹쳐 봤어요. 시장 분포는 행사가별 옵션 가격을 두 번 미분해서(Breeden–Litzenberger) 얻어요. 옵션 가격에는 위험 프리미엄이 들어 있어서, 실제보다 급락 쪽을 조금 더 크게 보는 경향이 있어요.",
        "en": "The price distribution on the same date, one curve per method. The market distribution comes from differentiating option prices twice (Breeden–Litzenberger). Option prices carry a risk premium, so it tends to overstate the odds of a crash a little.",
    },
    "hw_head_real": {"ko": '실제 IBM 양자 컴퓨터({backend})로 계산했어요. {ticker}가 {target} 이상으로 끝날 확률을 양자 컴퓨터는 <b>{est}</b>, 일반 컴퓨터는 <b>{exact}</b>로 계산했어요.', "en": 'Run on a real IBM quantum computer ({backend}). For the chance {ticker} ends above {target}, the quantum computer answered <b>{est}</b>; the exact answer is <b>{exact}</b>.'},
    "hw_head_prev": {"ko": 'IBM 양자 컴퓨터의 오류를 흉내 낸 시뮬레이션이에요. {target} 이상으로 끝날 확률을 <b>{est}</b>로 계산했어요. 정답은 <b>{exact}</b>예요.', "en": "A simulation that mimics an IBM quantum computer's errors. It put the chance of ending above {target} at <b>{est}</b>; the exact answer is <b>{exact}</b>."},
    "hw_f_q": {"ko": '양자 컴퓨터의 답', "en": "Quantum computer's answer"},
    "hw_f_q_note": {"ko": '읽기 오류를 바로잡은 값이에요.', "en": 'After fixing read errors.'},
    "hw_f_exact": {"ko": '정답', "en": 'Correct answer'},
    "hw_f_exact_note": {"ko": '같은 문제를 일반 컴퓨터로 정확히 푼 값이에요.', "en": 'The same problem solved exactly on a regular computer.'},
    "hw_f_gap": {"ko": '차이', "en": 'Difference'},
    "hw_f_gap_note": {"ko": '1~3%p 안이면 지금 양자 컴퓨터로는 잘 맞힌 편이에요.', "en": "Within 1–3 points is a good result for today's machines."},
    "hw_f_time": {"ko": '사용한 시간', "en": 'Time used'},
    "hw_f_time_note": {"ko": '무료 플랜은 28일마다 10분이에요.', "en": 'The free plan gives 10 minutes every 28 days.'},
    "hw_f_time_prev": {"ko": '0초', "en": '0 s'},
    "hw_f_time_prev_note": {"ko": '내 컴퓨터에서 시뮬레이션해서 IBM 사용 시간을 쓰지 않았어요.', "en": 'Simulated on your computer, so no IBM time was used.'},
    "hw_how_title": {"ko": '이 그래프 읽는 법', "en": 'How to read this chart'},
    "hw_how_body": {"ko": "- **회색 막대**는 정답이에요. 다른 막대가 회색과 높이가 비슷할수록 양자 컴퓨터가 정확한 거예요.\n- **왼쪽 묶음 (간단한 회로)**: 확률을 바로 재는 짧은 회로예요. 위의 '양자 컴퓨터의 답'이 여기서 나와요.\n- **오른쪽 묶음 (확대 단계를 넣은 회로)**: 더 정밀하게 재려면 결과를 '확대'하는 단계(그로버 반복)를 여러 번 거쳐야 해요. 여기서는 그 단계를 한 번만 넣었어요. 확대된 중간값이라 막대 높이가 위의 확률과 달라요.\n- 오른쪽이 정답과 더 많이 어긋나는 건, 회로가 길어질수록 양자 컴퓨터의 작은 오류가 쌓이기 때문이에요. 이 오류가 줄어들어야 양자 컴퓨터가 일반 컴퓨터보다 빨라질 수 있어요.", "en": "- The **grey bar** is the correct answer. The closer the other bars are to it, the more accurate the quantum computer.\n- **Left group (short circuit)**: measures the probability directly. The 'quantum computer's answer' above comes from here.\n- **Right group (circuit with an amplify step)**: precise answers need the result 'amplified' several times (Grover iterations); this adds one. It shows an amplified intermediate value, so its height differs from the probability above.\n- The right group misses by more because small errors pile up as circuits get longer. Until those errors shrink, quantum computers can't beat regular ones at this."},
    "hw_tech_title": {"ko": '기술 정보', "en": 'Technical details'},
    "hw_tech_body": {"ko": '- 2큐비트 게이트 수: 간단한 회로 {g0}개{g1}\n- 0을 1로 잘못 읽는 비율 {e0}, 1을 0으로 잘못 읽는 비율 {e1}\n- 결과를 읽은 물리 큐비트: {q}번\n- 측정 횟수(shots): {shots}', "en": '- Two-qubit gates: short circuit {g0}{g1}\n- Read 0 as 1: {e0}; read 1 as 0: {e1}\n- Physical qubit read out: {q}\n- Shots: {shots}'},
    "hw_g1b": {"ko": ', 확대 단계를 넣은 회로 {g}개', "en": ', amplify-step circuit {g}'},
    "hw_pp": {"ko": '%p', "en": ' points'},
}
