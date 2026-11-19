import html

import altair as alt
import streamlit as st

from ui import api

PROVIDER_COLORS = {"AWS": "#2f6db5", "Azure": "#12877f", "GCP": "#7a5cc7"}
INK, MUTED, GRID = "#1e293b", "#64748b", "#e8ebf0"
ACCENT, TEAL = "#1f5fa8", "#12877f"
GOOD, WARNING, CRITICAL = "#15803d", "#b45309", "#b91c1c"
MONEY_AXIS = ("datum.value >= 100000 ? '₹' + format(datum.value / 100000, '.1f') + 'L' : "
              "(datum.value >= 10000 ? '₹' + format(datum.value / 1000, '.0f') + 'K' : "
              "(datum.value >= 1000 ? '₹' + format(datum.value / 1000, '.1f') + 'K' : '₹' + format(datum.value, '.0f')))")
PILL_KINDS = {
    "critical": "critical", "over_budget": "critical", "high": "warning", "warning": "warning", "at_risk": "warning",
    "medium": "neutral", "on_track": "good", "good": "good", "STOP_REVIEW": "critical", "DOWNSIZE": "warning",
    "RETAIN": "neutral", "open": "warning", "acknowledged": "info", "investigated": "neutral", "dismissed": "muted",
    "in_review": "info", "approved": "good", "rejected": "muted", "snoozed": "muted", "applied": "good",
    "not_required": "muted", "running": "good", "scheduled": "info", "stopped": "muted",
}

CSS = """
<style>
.block-container {padding-top: 2.6rem; padding-bottom: 2.5rem; max-width: 1500px;}
h1, h2, h3 {color: #1e293b; letter-spacing: -0.01em;}
div[data-testid="stMetric"] {background: #ffffff; border: 1px solid #e3e7ee; border-radius: 8px; padding: 10px 14px;}
div[data-testid="stMetricLabel"] p {font-size: 0.78rem; color: #64748b; font-weight: 600;}
div[data-testid="stMetricValue"] {font-size: 1.45rem; color: #1e293b; font-weight: 600;}
[class*="st-key-panel"] {background: #ffffff; border-radius: 8px;}
.fo-title {font-size: 1.45rem; font-weight: 650; color: #1e293b; margin: 0;}
.fo-sub {font-size: 0.88rem; color: #64748b; margin: 2px 0 14px;}
.fo-section {font-size: 1.0rem; font-weight: 650; color: #1e293b; margin: 4px 0 2px;}
.fo-caption {font-size: 0.8rem; color: #64748b; margin-bottom: 6px;}
.fo-pill {display: inline-block; padding: 2px 9px; border-radius: 999px; font-size: 0.74rem; font-weight: 600;
  border: 1px solid transparent; margin-right: 4px; white-space: nowrap;}
.fo-pill-good {background: #e7f5ec; color: #166534; border-color: #c6e7d2;}
.fo-pill-warning {background: #fdf3e2; color: #92400e; border-color: #f4dcb0;}
.fo-pill-critical {background: #fbe9e9; color: #991b1b; border-color: #f2c8c8;}
.fo-pill-info {background: #e8f0fa; color: #1f4f8a; border-color: #cbdcf2;}
.fo-pill-neutral {background: #eef1f5; color: #334155; border-color: #dde2ea;}
.fo-pill-muted {background: #f5f6f8; color: #6b7280; border-color: #e8eaee;}
.fo-demo {background: #e8f0fa; color: #1f4f8a; border: 1px solid #cbdcf2; border-radius: 6px; padding: 6px 10px;
  font-size: 0.8rem; font-weight: 600;}
.fo-item {border-bottom: 1px solid #eef0f4; padding: 8px 0;}
.fo-item:last-child {border-bottom: none;}
.fo-item-title {font-weight: 600; color: #1e293b; font-size: 0.9rem;}
.fo-item-detail {color: #475569; font-size: 0.82rem; margin-top: 2px;}
.fo-kv {font-size: 0.86rem; color: #334155; line-height: 1.7;}
.fo-kv b {color: #1e293b;}
</style>
"""


def apply_style():
    st.markdown(CSS, unsafe_allow_html=True)


def pill(text: str, kind: str = None) -> str:
    kind = kind or PILL_KINDS.get(text, "neutral")
    label = text.replace("_", " ")
    if text.islower():
        label = label.capitalize()
    return f'<span class="fo-pill fo-pill-{kind}">{html.escape(label)}</span>'


def page_header(title: str, subtitle: str):
    st.markdown(f'<div class="fo-title">{html.escape(title)}</div><div class="fo-sub">{html.escape(subtitle)}</div>',
                unsafe_allow_html=True)


def section(title: str, caption: str = None):
    caption_html = f'<div class="fo-caption">{html.escape(caption)}</div>' if caption else ""
    st.markdown(f'<div class="fo-section">{html.escape(title)}</div>{caption_html}', unsafe_allow_html=True)


def kpi(column, label: str, value: str, note: str = None, help_text: str = None, delta_color: str = "off"):
    arrow = "off" if delta_color == "off" else "auto"
    column.metric(label, value, note, delta_color=delta_color, delta_arrow=arrow, help=help_text)


def flash_messages():
    message = st.session_state.pop("flash", None)
    if message:
        kind, text = message
        {"success": st.success, "error": st.error, "info": st.info}[kind](text)


def run_action(method: str, path: str, payload: dict = None, success_message=None):
    try:
        with st.spinner("Updating the simulation..."):
            result = api.request(method, path, json=payload)
    except api.ApiError as exc:
        st.session_state["flash"] = ("error", f"Not done: {exc.detail}")
        st.rerun()
    message = success_message(result) if callable(success_message) else (success_message or "Done")
    st.session_state["flash"] = ("success", message)
    st.rerun()


def show_chart(chart):
    chart = (chart.configure_axis(labelColor=MUTED, titleColor=MUTED, gridColor=GRID, domainColor="#cbd2dc",
                                  tickColor="#cbd2dc", labelFontSize=12, titleFontSize=12, titleFontWeight="normal")
             .configure_legend(labelColor=INK, titleColor=MUTED, labelFontSize=12, orient="top", title=None)
             .configure_view(strokeWidth=0))
    st.altair_chart(chart, width="stretch")


def money_column(label: str):
    return st.column_config.NumberColumn(label, format="localized")


def provider_scale():
    return alt.Scale(domain=list(PROVIDER_COLORS), range=list(PROVIDER_COLORS.values()))


def money_axis(title: str = None):
    return alt.Axis(labelExpr=MONEY_AXIS, title=title, tickCount=5)


def stacked_daily_chart(records: list, height: int = 300):
    data = alt.Data(values=records)
    return alt.Chart(data).mark_bar(size=12, cornerRadiusTopLeft=2, cornerRadiusTopRight=2).encode(
        x=alt.X("date:T", title=None, axis=alt.Axis(format="%d %b", labelAngle=0, tickCount=8, grid=False)),
        y=alt.Y("sum(cost):Q", axis=money_axis()),
        color=alt.Color("provider:N", scale=provider_scale(), legend=alt.Legend(title=None)),
        tooltip=[alt.Tooltip("date:T", format="%d %b %Y", title="Date"), alt.Tooltip("provider:N", title="Provider"),
                 alt.Tooltip("cost:Q", format=",.0f", title="Cost (₹)")],
    ).properties(height=height)


def bar_list(records: list, category: str, value: str = "cost", color: str = ACCENT, height: int = None,
             color_field: str = None):
    height = height or max(120, 30 * len(records) + 20)
    color_encoding = alt.Color(f"{color_field}:N", scale=provider_scale(), legend=None) if color_field else alt.value(color)
    base = alt.Chart(alt.Data(values=records)).encode(
        y=alt.Y(f"{category}:N", sort="-x", title=None, axis=alt.Axis(labelLimit=200, ticks=False, domain=False)),
        x=alt.X(f"{value}:Q", axis=money_axis()),
    )
    bars = base.mark_bar(size=16, cornerRadiusEnd=3).encode(
        color=color_encoding, tooltip=[alt.Tooltip(f"{category}:N"), alt.Tooltip(f"{value}:Q", format=",.0f")])
    return bars.properties(height=height)
