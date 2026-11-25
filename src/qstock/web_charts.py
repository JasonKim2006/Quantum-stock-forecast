"""Interactive Plotly charts for the dashboard. Every trace has a plain-language hover label."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .forecast import Forecast
from .i18n import _SYMBOLS, date, money, pct, t

INK = "#14213D"
MUTED = "#5A6781"
GRID = "#E3E8E6"
CLOUD = "#5B4BDB"
AMBER = "#D98A1C"
GOOD = "#2E7D5B"
FONT = "IBM Plex Sans, IBM Plex Sans KR, system-ui, sans-serif"

# The cloud is drawn as non-overlapping rings so each point has exactly one hover label:
# the 50% core, then the strips between 50-70%, 70-90% and 90-98%. Opacity falls outward.
BANDS = [(0.50, 0.50), (0.70, 0.34), (0.90, 0.22), (0.98, 0.11)]


def _layout(fig: go.Figure, height: int = 420, **kw) -> go.Figure:
    titled = "title" in kw
    if titled:  # title on its own line at the top, legend on the line below it
        kw["title"] = dict(kw["title"], x=0, xanchor="left", xref="paper", y=0.985, yanchor="top")
    fig.update_layout(
        height=height + (34 if titled else 0),
        margin=dict(l=8, r=8, t=78 if titled else 36, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, color=INK, size=13),
        hoverlabel=dict(bgcolor="white", bordercolor=GRID, font=dict(family=FONT, color=INK, size=13), align="left"),
        hovermode="closest",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0, font=dict(color=MUTED)),
        **kw,
    )
    fig.update_xaxes(gridcolor=GRID, zeroline=False, linecolor=GRID, tickfont=dict(color=MUTED))
    fig.update_yaxes(gridcolor=GRID, zeroline=False, linecolor=GRID, tickfont=dict(color=MUTED))
    return fig


def _rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def fan_chart(lang, snap, fc, fc_hist: Forecast | None, end, target, p_target, show_hist=True):
    """Six months of history, then a probability cloud out to ``end``."""
    cur = snap.currency
    fig = go.Figure()

    span = int(min(252, max(63, 3 * fc.T * 252)))  # history shown scales with the forecast horizon
    hist = snap.history.iloc[-span:]
    fig.add_trace(go.Scatter(
        x=hist.index, y=hist.values, mode="lines", name=t(lang, "c_price"),
        line=dict(color=INK, width=1.8), customdata=[money(cur, v) for v in hist.values],
        hovertemplate=t(lang, "c_price_hover") + "<extra></extra>",
    ))

    start = pd.Timestamp(snap.asof)
    end = pd.Timestamp(end)
    days = pd.date_range(start, end, periods=60)
    frac = (days - start) / (end - start)

    def bounds(f, level: float):
        pairs = [f.interval_at(level, x) for x in frac]
        return np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs])

    when = date(lang, end)
    x_fwd, x_back = list(days), list(days[::-1])
    inner_lo = inner_hi = None
    for level, alpha in BANDS:
        lo, hi = bounds(fc, level)
        lv = f"{level:.0%}"
        if inner_lo is None:  # the core: one closed shape
            xs, ys = x_fwd + x_back, list(hi) + list(lo[::-1])
            key = "c_core_hover"
        else:  # a ring: upper strip and lower strip, separated by a gap (None)
            xs = x_fwd + x_back + [None] + x_fwd + x_back
            ys = list(hi) + list(inner_hi[::-1]) + [None] + list(inner_lo) + list(lo[::-1])
            key = "c_band_hover"
        fig.add_trace(go.Scatter(
            x=xs, y=ys, fill="toself", fillcolor=_rgba(CLOUD, alpha), line=dict(width=0), mode="lines",
            name=t(lang, "c_band", lv=lv), hoveron="fills", hoverinfo="text",
            text=t(lang, key, lv=lv, date=when, lo=money(cur, lo[-1]), hi=money(cur, hi[-1])),
            showlegend=level in (0.9, 0.5),
        ))
        inner_lo, inner_hi = lo, hi

    med = np.array([fc.median_at(x) for x in frac])
    fig.add_trace(go.Scatter(
        x=days, y=med, mode="lines", name=t(lang, "c_median"), line=dict(color=CLOUD, width=1.6, dash="dot"),
        customdata=[money(cur, v) for v in med], hovertemplate=t(lang, "c_median_hover") + "<extra></extra>",
        showlegend=False,
    ))

    if fc_hist is not None and show_hist:
        hlo, hhi = bounds(fc_hist, 0.9)
        for i, arr in enumerate((hhi, hlo)):
            fig.add_trace(go.Scatter(
                x=days, y=arr, mode="lines", name=t(lang, "c_hist"), legendgroup="hist", showlegend=i == 0,
                line=dict(color=AMBER, width=1.6, dash="dash"), customdata=[money(cur, v) for v in arr],
                hovertemplate=t(lang, "c_hist_hover") + "<extra></extra>",
            ))

    side = t(lang, "c_above") if target >= snap.spot else t(lang, "c_below")
    tx = pd.date_range(hist.index[0], end, periods=40)
    fig.add_trace(go.Scatter(
        x=tx, y=[target] * len(tx), mode="lines", name=t(lang, "c_target"),
        line=dict(color=GOOD, width=1.4, dash="dashdot"), hoverinfo="text",
        text=t(lang, "c_target_hover", target=money(cur, target), date=when, side=side, p=pct(p_target)),
    ))

    fig.add_trace(go.Scatter(
        x=[start], y=[snap.spot], mode="markers", name=t(lang, "c_today"), showlegend=False,
        marker=dict(color=INK, size=9, line=dict(color="white", width=2)), hoverinfo="text",
        text=t(lang, "c_today_hover", spot=money(cur, snap.spot)),
    ))
    _layout(fig, height=440)
    fig.update_yaxes(tickprefix=_SYMBOLS.get(cur, ""))
    return fig


def smile_chart(lang, snap, calls: pd.DataFrame, hist_vol: float):
    ok = calls.dropna(subset=["iv"])
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ok["strike"], y=ok["iv"], mode="lines+markers", name=t(lang, "c_vol"),
        line=dict(color=CLOUD, width=2), marker=dict(size=7),
        hovertemplate=t(lang, "c_smile_hover") + "<extra></extra>",
    ))
    xs = [ok["strike"].min(), ok["strike"].max()] if len(ok) else [snap.spot * 0.8, snap.spot * 1.2]
    fig.add_trace(go.Scatter(
        x=np.linspace(*xs, 30), y=[hist_vol] * 30, mode="lines", name=t(lang, "col_hist"),
        line=dict(color=AMBER, dash="dash", width=1.6), hoverinfo="text",
        text=t(lang, "c_smile_hist_hover", vol=pct(hist_vol)),
    ))
    ylo = min([hist_vol] + list(ok["iv"])) if len(ok) else hist_vol
    yhi = max([hist_vol] + list(ok["iv"])) if len(ok) else hist_vol
    fig.add_trace(go.Scatter(
        x=[snap.spot] * 20, y=np.linspace(ylo * 0.95, yhi * 1.05, 20), mode="lines", showlegend=False,
        line=dict(color=MUTED, width=1), hoverinfo="text", text=t(lang, "c_spot_hover", spot=money(snap.currency, snap.spot)),
    ))
    _layout(fig, height=340, title=dict(text=t(lang, "c_smile_title"), font=dict(size=15)))
    fig.update_layout(legend=dict(y=-0.25, yanchor="top"))
    fig.update_xaxes(title=t(lang, "c_strike"))
    fig.update_yaxes(tickformat=".0%")
    return fig


def calibration_chart(lang, series):
    """Forecast range size vs. how often the outcome landed inside, one line per model.

    ``series``: list of (name, coverage DataFrame, colour).
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[0.4, 1.0], y=[0.4, 1.0], mode="lines", name=t(lang, "c_cal_ideal"),
        line=dict(color=MUTED, dash="dash", width=1.2), hoverinfo="text", text=t(lang, "c_cal_ideal_hover"),
    ))
    for i, (name, coverage, colour) in enumerate(series):
        fig.add_trace(go.Scatter(
            x=coverage["expected"], y=coverage["observed"], mode="lines+markers", name=name,
            line=dict(color=colour, width=2.4), marker=dict(size=10, symbol=["circle", "square"][i % 2]),
            hovertemplate=f"<b>{name}</b><br>" + t(lang, "c_cal_hover") + "<extra></extra>",
        ))
    _layout(fig, height=340, title=dict(text=t(lang, "c_cal_title"), font=dict(size=15)))
    fig.update_xaxes(title=t(lang, "c_cal_x"), tickformat=".0%", range=[0.4, 1.0])
    fig.update_yaxes(title=t(lang, "c_cal_y"), tickformat=".0%", range=[0.4, 1.0])
    return fig


def pit_chart(lang, pit: pd.Series):
    counts, edges = np.histogram(pit, bins=10, range=(0, 1))
    share = counts / max(counts.sum(), 1)
    labels = [f"{a:.0%}–{b:.0%}" for a, b in zip(edges[:-1], edges[1:])]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=(edges[:-1] + edges[1:]) / 2, y=share, width=0.09, marker_color=_rgba(CLOUD, 0.75),
        customdata=labels, hovertemplate=t(lang, "c_pit_hover") + "<extra></extra>", showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=np.linspace(0, 1, 30), y=[0.1] * 30, mode="lines", name=t(lang, "c_pit_ideal"),
        line=dict(color=MUTED, dash="dash", width=1.2), hoverinfo="skip",
    ))
    _layout(fig, height=340, title=dict(text=t(lang, "c_pit_title"), font=dict(size=15)))
    fig.update_xaxes(title=t(lang, "c_pit_x"), range=[0, 1])
    fig.update_yaxes(title=t(lang, "c_pit_y"), tickformat=".0%")
    return fig


def convergence_chart(lang, q_calls, q_err, mc_calls, mc_err):
    def slope(x, y):
        x, y = np.asarray(x, float), np.asarray(y, float)
        ok = y > 0
        return np.polyfit(np.log10(x[ok]), np.log10(y[ok]), 1)[0] if ok.sum() > 1 else float("nan")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=mc_calls, y=mc_err, mode="lines+markers", name=t(lang, "c_mc", s=f"{slope(mc_calls, mc_err):.2f}"),
        line=dict(color=AMBER, width=2), marker=dict(size=8), hovertemplate=t(lang, "c_mc_hover") + "<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=q_calls, y=q_err, mode="lines+markers", name=t(lang, "c_q", s=f"{slope(q_calls, q_err):.2f}"),
        line=dict(color=CLOUD, width=2), marker=dict(size=8, symbol="square"),
        hovertemplate=t(lang, "c_q_hover") + "<extra></extra>",
    ))
    _layout(fig, height=380, title=dict(text=t(lang, "c_conv_title"), font=dict(size=15)))
    fig.update_xaxes(type="log", title=t(lang, "c_conv_x"))
    fig.update_yaxes(type="log", title=t(lang, "c_conv_y"), exponentformat="power")
    return fig


def density_chart(lang, cur, curves, spot, target):
    """Probability density of the price on the forecast date, one curve per model.

    ``curves``: list of (name, x, pdf, colour, dash, hover_note).
    """
    fig = go.Figure()
    for name, x, y, colour, dash, note in curves:
        fig.add_trace(go.Scatter(
            x=x, y=y, mode="lines", name=name, line=dict(color=colour, width=2.2, dash=dash),
            fill="tozeroy" if dash == "solid" else None,
            fillcolor=_rgba(colour, 0.12) if dash == "solid" else None,
            customdata=[money(cur, v) for v in x],
            hovertemplate=f"<b>{name}</b><br>" + t(lang, "c_dens_hover") + f"<br><i>{note}</i><extra></extra>",
        ))
    ymax = max(float(np.max(c[2])) for c in curves)
    for xv, key, colour in ((spot, "c_dens_spot", INK), (target, "c_dens_target", GOOD)):
        fig.add_trace(go.Scatter(
            x=[xv, xv], y=[0, ymax * 1.05], mode="lines", showlegend=False,
            line=dict(color=colour, width=1.2, dash="dot"), hoverinfo="text",
            text=t(lang, key, v=money(cur, xv)),
        ))
    _layout(fig, height=360, title=dict(text=t(lang, "c_dens_title"), font=dict(size=15)))
    fig.update_xaxes(title=t(lang, "c_dens_x"), tickprefix=_SYMBOLS.get(cur, ""))
    fig.update_yaxes(title=t(lang, "c_dens_y"), showticklabels=False)
    return fig


def hardware_chart(lang, rows):
    """Exact vs measured P(objective = 1) for each circuit, with error bars."""
    labels = [t(lang, "hw_k0") if r["circuit"] == "k=0" else t(lang, "hw_k1") for r in rows]
    fig = go.Figure()
    for key, colour, field in (("hw_exact", MUTED, "exact"), ("hw_raw", AMBER, "raw"), ("hw_fixed", CLOUD, "corrected")):
        fig.add_trace(go.Bar(
            x=labels, y=[r[field] for r in rows], name=t(lang, key), marker_color=_rgba(colour, 0.85),
            error_y=dict(type="data", array=[r["stderr"] for r in rows], visible=field != "exact", color=INK),
            hovertemplate=t(lang, f"{key}_hover") + "<extra></extra>",
        ))
    _layout(fig, height=360, barmode="group", title=dict(text=t(lang, "hw_chart_title"), font=dict(size=15)))
    fig.update_yaxes(title=t(lang, "hw_y"), tickformat=".0%", range=[0, 1])
    return fig
