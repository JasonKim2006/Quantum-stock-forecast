"""Quantum Stock Forecast: web dashboard.

Start with:  python run_dashboard.py   (not `streamlit run app.py`; see run_dashboard.py)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Gate

from qstock.backtest import coverage_band, run_backtest
from qstock.forecast import Forecast, cell_model, quantum_call_price, quantum_prob_above
from qstock.i18n import STR, date, money, pct, t
from qstock import ibm
from qstock.implied import market_forecast
from qstock.stockdist import stock_forecast
from qstock.market import (
    atm_implied_vol,
    bs_call_price,
    load_calls,
    load_puts,
    load_snapshot,
    nearest_expiry,
    year_fraction,
)
from qstock.quantum import build_exact_problem
from qstock.web_charts import (
    calibration_chart,
    convergence_chart,
    density_chart,
    fan_chart,
    hardware_chart,
    pit_chart,
    smile_chart,
)

st.set_page_config(page_title="Quantum Stock Forecast", page_icon="◌", layout="wide",
                   initial_sidebar_state="collapsed")

POPULAR = ["AAPL", "NVDA", "TSLA", "MSFT", "SPY", "005930.KS"]
HORIZONS = [7, 30, 91, 182, 365]
PLOT_CFG = {"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"]}

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+KR:wght@400;500;600;700&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');
:root { --ink:#14213D; --muted:#5A6781; --line:#CBD3CF; --cloud:#5B4BDB; --good:#2E7D5B; --warn:#C2410C; }
.stApp, .stApp p, .stApp li, .stApp label, .stApp input, .stApp button, .stApp td, .stApp th,
.stApp h1, .stApp h2, .stApp h3, .stApp h4 {
  font-family: 'IBM Plex Sans', 'IBM Plex Sans KR', system-ui, sans-serif;
}
.block-container { max-width: 1120px; padding-top: 1.6rem; padding-bottom: 4rem; }
[data-testid="stHeader"] { background: transparent; }
.stApp p.brand { font-weight: 600; font-size: 1.05rem; color: var(--ink); letter-spacing: -0.01em; margin: 0; }
.stApp p.brand-sub { color: var(--muted); font-size: 0.95rem; margin: 0.15rem 0 0; max-width: 62ch; }
.stApp p.headline { font-size: clamp(1.8rem, 3.4vw, 2.75rem); font-weight: 600; line-height: 1.18;
  letter-spacing: -0.022em; color: var(--ink); margin: 1.6rem 0 0.6rem; max-width: 30ch;
  font-variant-numeric: tabular-nums; }
.stApp p.lede { font-size: 1.12rem; color: #33415C; margin: 0 0 0.35rem; max-width: 68ch; }
.stApp p.fine { font-size: 0.88rem; color: var(--muted); margin: 0 0 0.4rem; max-width: 78ch; }
.fact { border-top: 1px solid var(--line); padding-top: 0.9rem; margin-bottom: 0.9rem; }
.fact .label { font-size: 0.95rem; color: var(--muted); font-weight: 500; margin: 0; }
.fact .value { font-size: 2.3rem; font-weight: 600; color: var(--ink); margin: 0.1rem 0 0.2rem;
  font-variant-numeric: tabular-nums; letter-spacing: -0.02em; }
.fact .note { font-size: 0.88rem; color: var(--muted); margin: 0; line-height: 1.45; }
.verdict-ok { color: var(--good); font-weight: 600; }
.verdict-bad { color: var(--warn); font-weight: 600; }
.stTabs [data-baseweb="tab-list"] { gap: 1.6rem; }
.stTabs [data-baseweb="tab"] { font-size: 1rem; padding-left: 0; padding-right: 0; }
@media (max-width: 640px) { .fact .value { font-size: 1.9rem; } }
</style>
""",
    unsafe_allow_html=True,
)


# ------------------------------------------------------------------ cached work
@st.cache_data(ttl=300, show_spinner=False)
def get_snapshot(ticker: str, demo: bool):
    return load_snapshot(ticker, demo=demo)


@st.cache_data(ttl=300, show_spinner=False)
def get_calls(ticker: str, demo: bool, expiry: str):
    return load_calls(get_snapshot(ticker, demo), expiry)


@st.cache_data(ttl=300, show_spinner=False)
def get_market(ticker: str, demo: bool, expiry: str):
    """Option-implied distribution for one expiry, or (None, reason) when it can't be built reliably."""
    snap = get_snapshot(ticker, demo)
    try:
        calls, puts = get_calls(ticker, demo, expiry), load_puts(snap, expiry)
        return market_forecast(calls, puts, snap.spot, snap.rate, year_fraction(expiry, snap.asof)), None
    except Exception as exc:  # sparse quotes, bad fit, network
        return None, str(exc)


@st.cache_data(show_spinner=False)
def q_prob(key, _fc, x, n, eps):
    """Quantum P(S_T > x). ``key`` identifies the forecast for caching; ``_fc`` is not hashed."""
    return quantum_prob_above(_fc, x, n, eps, seed=7)


@st.cache_data(show_spinner=False)
def hw_preview(key, pmf, cond, shots, grover):
    plan, res = ibm.preview(pmf, cond, "FakeFez", shots, grover)
    return ibm.plan_summary(plan), res


@st.cache_resource(show_spinner=False)
def ibm_service():
    try:
        return ibm.connect(), None
    except Exception as exc:
        return None, str(exc)


@st.cache_data(show_spinner=False)
def q_call(spot, vol, rate, T, k, n, eps):
    return quantum_call_price(Forecast(spot, vol, rate, T), k, n, eps, seed=11)


@st.cache_data(show_spinner=False)
def get_backtest(ticker: str, demo: bool, horizon: int, model: str):
    return run_backtest(get_snapshot(ticker, demo).history, horizon=horizon, model=model)


@st.cache_data(ttl=300, show_spinner=False)
def get_stock_forecast(ticker: str, demo: bool, T: float):
    snap = get_snapshot(ticker, demo)
    return stock_forecast(snap.history, T, snap.rate)


def fact(label: str, value: str, note: str) -> None:
    st.markdown(
        f'<div class="fact"><p class="label">{label}</p><p class="value">{value}</p>'
        f'<p class="note">{note}</p></div>',
        unsafe_allow_html=True,
    )


def show_hardware(lang, cur, res, summary, cov, target, ticker="", backend=None, seconds=None, shots=None):
    """Plain-language readout of one hardware run (``backend`` set) or noisy preview (``backend`` None)."""
    est, exact = cov * res["estimate_corrected"], cov * res["estimate_exact"]
    gap = (est - exact) * 100
    if backend:
        head = t(lang, "hw_head_real", backend=backend, ticker=ticker, target=money(cur, target),
                 est=pct(est), exact=pct(exact))
    else:
        head = t(lang, "hw_head_prev", target=money(cur, target), est=pct(est), exact=pct(exact))
    st.markdown(f'<p class="lede">{head}</p>', unsafe_allow_html=True)
    c = st.columns(4, gap="medium")
    with c[0]:
        fact(t(lang, "hw_f_q"), pct(est), t(lang, "hw_f_q_note"))
    with c[1]:
        fact(t(lang, "hw_f_exact"), pct(exact), t(lang, "hw_f_exact_note"))
    with c[2]:
        fact(t(lang, "hw_f_gap"), f"{gap:+.1f}{t(lang, 'hw_pp')}", t(lang, "hw_f_gap_note"))
    with c[3]:
        if backend:
            secs = f"{seconds:.0f}{'초' if lang == 'ko' else ' s'}" if seconds is not None else "–"
            fact(t(lang, "hw_f_time"), secs, t(lang, "hw_f_time_note"))
        else:
            fact(t(lang, "hw_f_time"), t(lang, "hw_f_time_prev"), t(lang, "hw_f_time_prev_note"))
    st.plotly_chart(hardware_chart(lang, res["rows"]), width="stretch", config=PLOT_CFG)
    with st.expander(t(lang, "hw_how_title"), expanded=True):
        st.markdown(t(lang, "hw_how_body"))
    g = summary["two_qubit_gates"]
    with st.expander(t(lang, "hw_tech_title")):
        st.markdown(t(lang, "hw_tech_body", g0=g["k=0"], g1=t(lang, "hw_g1b", g=g["k=1"]) if "k=1" in g else "",
                      e0=pct(res["readout_error_0to1"]), e1=pct(res["readout_error_1to0"]),
                      q=summary["physical_qubit"], shots=f"{shots:,}" if shots else "–"))


# ------------------------------------------------------------------ header
ss = st.session_state
ss.setdefault("ticker_box", "AAPL")
ss.setdefault("qubits", 5)
ss.setdefault("eps", 0.005)

head_l, head_r = st.columns([5, 2], vertical_alignment="center")
with head_r:
    lang_choice = st.segmented_control("Language", ["한국어", "English"], default="한국어",
                                       key="lang_pick", label_visibility="collapsed")
    lang = "en" if lang_choice == "English" else "ko"
    with st.popover(t(lang, "settings")):
        demo = st.toggle(t(lang, "demo_toggle"), key="demo", help=t(lang, "demo_help"))
        use_opts = st.toggle(t(lang, "opt_toggle"), key="use_opts", help=t(lang, "opt_toggle_help"))
with head_l:
    st.markdown(f'<p class="brand">◌ {t(lang, "brand")}</p><p class="brand-sub">{t(lang, "tagline")}</p>',
                unsafe_allow_html=True)


# ------------------------------------------------------------------ search panel
def _pick():
    if ss.get("pick"):
        ss.ticker_box = ss.pick


def _typed():
    ss.pick = None


with st.container(border=True):
    c1, c2 = st.columns([2, 3])
    with c1:
        st.text_input(t(lang, "ticker_label"), key="ticker_box", placeholder=t(lang, "ticker_ph"), on_change=_typed)
    with c2:
        st.pills(t(lang, "popular"), POPULAR, key="pick", on_change=_pick)
    h_labels = {d: t(lang, f"h{d}") for d in HORIZONS}
    hc, mc = st.columns([3, 2])
    with hc:
        horizon = st.segmented_control(t(lang, "horizon_label"), HORIZONS, format_func=h_labels.get,
                                       default=30, key="horizon") or 30
    with mc:
        m_labels = {"fhs": t(lang, "m_fhs"), "hv": t(lang, "m_hv")}
        if use_opts:
            m_labels |= {"market": t(lang, "m_market"), "iv": t(lang, "m_iv")}
        if ss.get("method") not in m_labels:
            ss.method = "fhs"
        method = st.selectbox(t(lang, "method_label"), list(m_labels), format_func=m_labels.get, key="method",
                              help=t(lang, "method_help_opts" if use_opts else "method_help"))

ticker = (ss.ticker_box or "AAPL").strip().upper()

with st.spinner(t(lang, "spinner_data")):
    try:
        snap = get_snapshot(ticker, demo)
    except Exception:
        st.error(t(lang, "load_error", ticker=ticker))
        st.stop()

cur = snap.currency
end_req = snap.asof.normalize() + pd.Timedelta(days=horizon)
hv = snap.hist_vol

# Options are only touched when the advanced switch is on.
expiry = nearest_expiry(snap, end_req) if use_opts else None
calls, iv = None, None
if expiry is not None:
    try:
        calls = get_calls(ticker, demo, expiry)
        iv = atm_implied_vol(calls, snap.spot)
    except Exception:
        calls, iv = None, None
has_iv = iv is not None and np.isfinite(iv)
mf, mf_err = get_market(ticker, demo, expiry) if expiry is not None else (None, "no options")

notes = []
T = horizon / 365
end = end_req
try:
    sf = get_stock_forecast(ticker, demo, T)
except ValueError:
    sf = None
if method == "fhs" and sf is not None:
    fc = sf
    sub_key = "sub_fhs"
elif method == "market" and mf is not None:
    fc, T, end = mf, mf.T, pd.Timestamp(expiry)
    if abs((end - end_req).days) > 2:
        notes.append(t(lang, "note_expiry", date=date(lang, end)))
    sub_key = "sub_market"
elif method == "iv" or (method == "market" and has_iv):
    if method == "market":
        notes.append(t(lang, "note_fallback"))
    fc = Forecast(snap.spot, iv if has_iv else hv, snap.rate, T)
    sub_key = "sub_noopt" if not has_iv else ("sub_more" if iv > hv else "sub_less")
else:
    fc = Forecast(snap.spot, hv, snap.rate, T)
    sub_key = "sub_hist"
fc_hist = Forecast(snap.spot, hv, snap.rate, T)
fc_key = (ticker, demo, method, round(T, 6), round(float(fc.vol), 6), expiry)
vol = float(fc.vol)

# ------------------------------------------------------------------ headline
if snap.source == "demo":
    st.warning(t(lang, "demo_banner"))
lo90, hi90 = fc.interval(0.9)
when = date(lang, end)
st.markdown(
    f'<p class="headline">{t(lang, "headline", date=when, ticker=ticker, lo=money(cur, lo90), hi=money(cur, hi90))}</p>',
    unsafe_allow_html=True,
)
sub_args = {"spot": money(cur, snap.spot)}
if sub_key == "sub_fhs":
    sub_args |= {"n": f"{sf.n_shocks:,}", "vol": pct(sf.vol), "long": pct(sf.long_vol)}
st.markdown(f'<p class="lede">{t(lang, sub_key, **sub_args)}</p>', unsafe_allow_html=True)
cap = t(lang, "live_caption", date=date(lang, snap.asof)) if snap.source != "demo" else ""
st.markdown(f'<p class="fine">{" ".join(notes)} {t(lang, "disclaimer")} {cap}</p>', unsafe_allow_html=True)

# ------------------------------------------------------------------ facts (target slider drives the chart too)
f1, f2, f3 = st.columns(3, gap="large")
move = ss.get("move", 5)  # the slider is drawn under the first fact so the three facts line up
target = snap.spot * (1 + move / 100)
above = move >= 0
p_above = fc.prob_above(target)
p_side = p_above if above else 1 - p_above

with st.spinner(t(lang, "spinner_q")):
    q = q_prob(fc_key, fc, target, ss.qubits, ss.eps)
qv, qlo, qhi = (q.value, q.ci_low, q.ci_high) if above else (1 - q.value, 1 - q.ci_high, 1 - q.ci_low)

with f1:
    fact(t(lang, "fact_above" if above else "fact_below", target=money(cur, target)), pct(p_side),
         t(lang, "fact_chance_note", q=pct(qv), lo=pct(qlo), hi=pct(qhi)))
    st.slider(t(lang, "target_slider"), -30, 30, 5, step=1, format="%+d%%", key="move")

bt_horizon = max(5, round(T * 252))
bts = {}
for model in ("fhs", "normal"):
    try:
        bts[model] = get_backtest(ticker, demo, bt_horizon, model)
    except ValueError:
        pass
# The headline fact shows the backtest of the model in use; option-based forecasts can't be
# backtested (no historical option data), so they show the bell curve's record.
bt = bts.get("fhs" if method == "fhs" else "normal")
with f2:
    if bt is not None:
        obs90 = float(bt.coverage.set_index("level").loc[0.9, "observed"])
        fact(t(lang, "fact_bt"), pct(obs90, 0), t(lang, "fact_bt_note", n=f"{len(bt.records):,}"))
with f3:
    swing = snap.spot * (np.exp(vol * np.sqrt(T)) - 1)
    fact(t(lang, "fact_swing"), f"±{money(cur, swing)}", t(lang, "fact_swing_note", date=when, vol=pct(vol)))

# ------------------------------------------------------------------ the cloud
st.plotly_chart(fan_chart(lang, snap, fc, fc_hist if method != "hv" else None, end, target, p_side),
                width="stretch", config=PLOT_CFG)
st.markdown(f'<p class="fine">{t(lang, "chart_hint")}</p>', unsafe_allow_html=True)

# ------------------------------------------------------------------ tabs
tab_names = [t(lang, "tab_detail")] + ([t(lang, "tab_options")] if use_opts else []) + [t(lang, "tab_track"),
                                                                                         t(lang, "tab_quantum")]
tabs = st.tabs(tab_names)
tab_d, tab_t, tab_q = tabs[0], tabs[-2], tabs[-1]
tab_o = tabs[1] if use_opts else None

with tab_d:
    # Every model on the same date as the headline, so the columns compare like with like.
    models = []
    if sf is not None:
        models.append((t(lang, "m_fhs"), sf, "#5B4BDB", "solid", t(lang, "dn_fhs")))
    if mf is not None:
        models.append((t(lang, "m_market"), mf, "#2E7D5B", "dashdot", t(lang, "dn_market")))
    if has_iv:
        models.append((t(lang, "m_iv"), Forecast(snap.spot, iv, snap.rate, T), "#14213D", "dash", t(lang, "dn_iv")))
    models.append((t(lang, "m_hv"), Forecast(snap.spot, hv, snap.rate, T), "#D98A1C", "dot", t(lang, "dn_hv")))

    rows = []
    for lv in (0.5, 0.7, 0.9, 0.98):
        row = {t(lang, "col_range"): t(lang, "row_range", lv=f"{lv:.0%}")}
        for name, m, *_ in models:
            a, b = m.interval(lv)
            row[name] = f"{money(cur, a)} – {money(cur, b)}"
        rows.append(row)
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

    st.subheader(t(lang, "dens_title"))
    st.markdown(f'<p class="lede">{t(lang, "dens_body_opts" if use_opts else "dens_body")}</p>', unsafe_allow_html=True)
    lo_x, hi_x = models[0][1].interval(0.995)
    xs = np.linspace(lo_x, hi_x, 400)
    curves = []
    for name, m, colour, dash, note in models:
        pdf = m.dist.pdf(xs)
        curves.append((name, xs, pdf, colour, dash, note))
    st.plotly_chart(density_chart(lang, cur, curves, snap.spot, target), width="stretch", config=PLOT_CFG)

    tail_rows = []
    for key, p in (("tail_drop", 0.2), ("tail_drop", 0.1), ("tail_rise", 0.1), ("tail_rise", 0.2)):
        x = snap.spot * (1 - p if key == "tail_drop" else 1 + p)
        row = {"_": t(lang, key, p=f"{p:.0%}")}
        for name, m, *_ in models:
            row[name] = pct(1 - m.prob_above(x) if key == "tail_drop" else m.prob_above(x))
        tail_rows.append(row)
    tail = pd.DataFrame(tail_rows).rename(columns={"_": t(lang, "tail_cols").split("|")[0]})
    st.dataframe(tail, hide_index=True, width="stretch")
    if mf is not None and mf.dist.clipped_mass > 0.001:
        st.markdown(f'<p class="fine">{t(lang, "clip_note", p=pct(mf.dist.clipped_mass))}</p>', unsafe_allow_html=True)
    with st.expander(t(lang, "how_title")):
        st.markdown(t(lang, "how_body"))

if tab_o is not None:
  with tab_o:
    if calls is None or not has_iv:
        st.info(t(lang, "opt_none"))
    else:
        T_exp = year_fraction(expiry, snap.asof)
        st.subheader(t(lang, "opt_title", date=date(lang, expiry)))
        st.markdown(f'<p class="fine">{t(lang, "opt_explain")}</p>', unsafe_allow_html=True)
        near = calls[(calls["strike"] > snap.spot * 0.85) & (calls["strike"] < snap.spot * 1.15)].dropna(subset=["mid"]).copy()
        near["m_iv"] = bs_call_price(snap.spot, near["strike"], snap.rate, iv, T_exp)
        near["m_hv"] = bs_call_price(snap.spot, near["strike"], snap.rate, hv, T_exp)
        cols = t(lang, "opt_cols").split("|")
        table = near[["strike", "bid", "ask", "mid", "iv", "m_iv", "m_hv", "volume"]].copy()
        table.columns = cols
        st.dataframe(table.style.format({cols[4]: "{:.1%}", cols[7]: "{:,.0f}"}, precision=2),
                     hide_index=True, width="stretch")
        st.plotly_chart(smile_chart(lang, snap, calls, hv), width="stretch", config=PLOT_CFG)

        strikes = near["strike"].tolist()
        if strikes:
            st.subheader(t(lang, "opt_q_title"))
            k = st.select_slider(t(lang, "opt_strike"), strikes, value=min(strikes, key=lambda s: abs(s - snap.spot)),
                                 format_func=lambda s: money(cur, s))
            with st.spinner(t(lang, "spinner_q")):
                qc = q_call(snap.spot, iv, snap.rate, T_exp, float(k), ss.qubits, ss.eps)
            m = st.columns(4)
            m[0].metric(t(lang, "m_market"), money(cur, float(near.loc[near["strike"] == k, "mid"].iloc[0])))
            m[1].metric(t(lang, "m_formula"), money(cur, float(bs_call_price(snap.spot, k, snap.rate, iv, T_exp))))
            m[2].metric(t(lang, "m_quantum"), money(cur, qc.value),
                        f"95% CI {money(cur, qc.ci_low)}–{money(cur, qc.ci_high)}", delta_color="off")
            m[3].metric(t(lang, "m_calls"), f"{qc.a_calls:,}", help=t(lang, "m_calls_help"))

with tab_t:
    if not bts:
        st.info(t(lang, "load_error", ticker=ticker))
    else:
        ref = next(iter(bts.values()))
        st.subheader(t(lang, "bt_title"))
        st.markdown(t(lang, "bt_body", n=f"{len(ref.records):,}", h=ref.horizon, ind=ref.independent_n))
        names = {"fhs": t(lang, "m_fhs"), "normal": t(lang, "m_hv")}
        cols = t(lang, "bt_cols").split("|")
        rows = []
        for lv in ref.coverage["level"]:
            lo, hi = coverage_band(lv, ref.independent_n)
            row = {cols[0]: f"{lv:.0%}"}
            for key, b in bts.items():
                obs = float(b.coverage.set_index("level").loc[lv, "observed"])
                ok = lo <= obs <= hi
                mark = "✓" if ok else ("! " + t(lang, "v_narrow" if obs < lo else "v_wide"))
                row[names[key]] = f"{pct(obs)}  {mark}"
            row[cols[2]] = f"{lo:.0%}–{hi:.0%}"
            rows.append(row)
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        c1, c2 = st.columns(2, gap="large")
        series = [(names[k], b.coverage, "#5B4BDB" if k == "fhs" else "#D98A1C") for k, b in bts.items()]
        c1.plotly_chart(calibration_chart(lang, series), width="stretch", config=PLOT_CFG)
        c2.plotly_chart(pit_chart(lang, bts.get("fhs", ref).records["pit"]), width="stretch", config=PLOT_CFG)
        f_cols = st.columns(len(bts), gap="large")
        for col, (key, b) in zip(f_cols, bts.items()):
            with col:
                fact(f'{t(lang, "bt_tail")}: {names[key]}', pct(float((b.records["z"].abs() > 3).mean()), 2),
                     t(lang, "bt_tail_note"))

with tab_q:
    st.subheader(t(lang, "q_title"))
    st.markdown(t(lang, "q_steps", n=ss.qubits, cells=2 ** ss.qubits))
    _, pmf, cond, _ = cell_model(fc, lambda s: (s > target).astype(float), ss.qubits, (target,))
    A = build_exact_problem(pmf, cond).state_preparation
    tc = transpile(A, basis_gates=["cx", "u"], optimization_level=1)
    m = st.columns(3)
    m[0].metric(t(lang, "q_qubits"), A.num_qubits)
    m[1].metric(t(lang, "q_cx"), tc.count_ops().get("cx", 0))
    m[2].metric(t(lang, "q_depth"), tc.depth())
    shown = QuantumCircuit(A.num_qubits)
    shown.append(Gate("Load P(S_T)", ss.qubits, []), range(ss.qubits))
    shown.append(Gate("Payoff RY", ss.qubits + 1, []), range(ss.qubits + 1))
    st.code(str(shown.draw(output="text")), language=None)

    with st.expander(t(lang, "q_advanced")):
        st.slider(t(lang, "q_n"), 3, 7, key="qubits", help=t(lang, "q_n_help"))
        st.select_slider(t(lang, "q_eps"), [0.02, 0.01, 0.005, 0.002, 0.001], key="eps", help=t(lang, "q_eps_help"))

    if st.button(t(lang, "q_run")):
        exact = fc.prob_above(target)
        rng = np.random.default_rng(0)
        q_calls, q_err, mc_calls, mc_err = [], [], [], []
        prog = st.progress(0.0)
        eps_list = [0.02, 0.01, 0.005, 0.002, 0.001]
        for i, eps in enumerate(eps_list):
            ests = [quantum_prob_above(fc, target, ss.qubits, eps, seed=s) for s in range(5)]
            q_calls.append(np.mean([e.a_calls for e in ests]))
            q_err.append(np.sqrt(np.mean([(e.value - e.exact) ** 2 for e in ests])))
            prog.progress((i + 1) / len(eps_list))
        prog.empty()
        for n in np.logspace(2, 5.5, 8).astype(int):
            z = rng.standard_normal((100, n))
            s_T = fc.spot * np.exp((fc.rate - 0.5 * fc.vol**2) * fc.T + fc.vol * np.sqrt(fc.T) * z)
            mc_calls.append(n)
            mc_err.append(np.sqrt(np.mean(((s_T > target).mean(axis=1) - exact) ** 2)))
        st.plotly_chart(convergence_chart(lang, q_calls, q_err, mc_calls, mc_err), width="stretch", config=PLOT_CFG)
        st.markdown(f'<p class="fine">{t(lang, "q_run_caption")}</p>', unsafe_allow_html=True)

    # -------------------------------------------------------------- IBM hardware
    st.divider()
    st.subheader(t(lang, "hw_title"))
    st.markdown(t(lang, "hw_intro", target=money(cur, target)))
    service, _svc_err = ibm_service()
    if service is None:
        st.info(t(lang, "hw_no_account"))
    else:
        mins = ibm.remaining_minutes(service)
        st.caption(t(lang, "hw_connected", m=f"{mins:.1f}") if mins is not None else t(lang, "hw_connected_plain"))

    h1, h2, h3 = st.columns(3, vertical_alignment="bottom")
    hw_n = h1.select_slider(t(lang, "hw_n"), [2, 3, 4], value=3, key="hw_n", help=t(lang, "hw_n_help"))
    hw_shots = h2.select_slider(t(lang, "hw_shots"), [1000, 2000, 4000, 8000], value=4000, key="hw_shots")
    hw_grover = h3.toggle(t(lang, "hw_grover"), value=True, key="hw_grover")
    _, hpmf, hcond, hcov = cell_model(fc, lambda s_: (s_ > target).astype(float), hw_n, (target,))

    b1, b2 = st.columns(2)
    if b1.button(t(lang, "hw_preview_btn"), help=t(lang, "hw_preview_help"), width="stretch"):
        with st.spinner(t(lang, "hw_preview_spinner")):
            summary, res = hw_preview((fc_key, round(target, 6), hw_n), hpmf, hcond, hw_shots, hw_grover)
        ss.hw_prev = (summary, res, hcov, target)
    if b2.button(t(lang, "hw_submit_btn"), help=t(lang, "hw_submit_help"), type="primary",
                 disabled=service is None, width="stretch"):
        with st.spinner(t(lang, "hw_submitting")):
            try:
                rec = ibm.submit(service, hpmf, hcond, hw_shots, hw_grover,
                                 meta={"ticker": ticker, "target": target, "cov": hcov, "currency": cur,
                                       "method": method, "forecast_date": str(end.date())})
                st.success(t(lang, "hw_submitted", backend=rec["backend"]))
            except Exception as exc:
                st.error(t(lang, "hw_submit_fail", err=exc))

    if "hw_prev" in ss:
        summary, res, cov, tgt = ss.hw_prev
        st.markdown(f"**{t(lang, 'hw_preview_title')}**")
        show_hardware(lang, cur, res, summary, cov, tgt, shots=res["rows"][0]["shots"])

    st.markdown(f"**{t(lang, 'hw_jobs')}**")
    jobs = ibm.load_jobs()
    if not jobs:
        st.markdown(f'<p class="fine">{t(lang, "hw_jobs_empty")}</p>', unsafe_allow_html=True)
    for rec in jobs[:8]:
        label = t(lang, "hw_job_label", when=rec["submitted"], backend=rec["backend"], ticker=rec.get("ticker", ""),
                  target=money(rec.get("currency", cur), rec.get("target", 0)))
        with st.expander(label, expanded=rec is jobs[0]):
            if "result" not in rec and st.button(t(lang, "hw_check"), key=f"chk_{rec['job_id']}",
                                                 disabled=service is None):
                try:
                    status, res, used = ibm.fetch(service, rec)
                except Exception as exc:
                    status, res, used = f"ERROR ({exc})", None, None
                if res is not None:
                    rec["result"], rec["quantum_seconds"] = res, used
                    ibm.save_jobs(jobs)
                    st.rerun()  # redraw without the button, now that the result is stored
                else:
                    st.info(t(lang, f"st_{status}") if f"st_{status}" in STR else t(lang, "hw_status", s=status))
            if "result" in rec:
                summary = {"two_qubit_gates": rec["two_qubit_gates"], "physical_qubit": rec["physical_qubit"]}
                show_hardware(lang, rec.get("currency", cur), rec["result"], summary, rec.get("cov", 1.0),
                              rec.get("target", target), ticker=rec.get("ticker", ""), backend=rec["backend"],
                              seconds=rec.get("quantum_seconds"), shots=rec.get("shots"))
            st.caption(f"job id: {rec['job_id']}")
