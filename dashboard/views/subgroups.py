import plotly.graph_objects as go
import streamlit as st
import theme as t

import data

t.hero("Subgroups & Fairness", "Performance for sex, age band and site with sample sizes and bootstrap CIs. Groups with fewer "
       "than 50 patients are suppressed rather than over-interpreted.", badge="Equal-opportunity view", icon="⚖️")

sg = data.subgroups()
d = data.summary()["disparities"]
t.kpis([(f"{g} — sensitivity gap", f"{v['sensitivity_gap']:.2f}", f"AUROC gap {v['auroc_gap']:.2f} · max calib. gap "
         f"{v['max_abs_calibration_gap']:.2f}", "📏", c) for (g, v), c in zip(d.items(), [t.PRIMARY, t.ACCENT, t.SECONDARY])])

grouping = st.segmented_control("Grouping", list(sg.grouping.unique()), default="age_band") or "age_band"
g = sg[(sg.grouping == grouping)]
ok = g[g.suppressed == 0]
c1, c2 = st.columns(2)
with c1:
    t.section("AUROC with 95% CI")
    fig = go.Figure(go.Scatter(x=ok.auroc, y=ok.group + "  (n=" + ok.n.astype(str) + ")", mode="markers",
                               marker=dict(size=14, color=t.PRIMARY),
                               error_x=dict(type="data", symmetric=False, array=ok.auroc_hi - ok.auroc, arrayminus=ok.auroc - ok.auroc_lo)))
    fig.add_vline(x=0.5, line_dash="dot", line_color=t.MUTED)
    fig.update_layout(xaxis_range=[0.45, 0.85], xaxis_title="AUROC")
    st.plotly_chart(t.style_fig(fig, 320), use_container_width=True)
with c2:
    t.section("Sensitivity & flag rate at the shared threshold")
    fig = go.Figure()
    fig.add_trace(go.Bar(x=ok.group, y=ok.sensitivity, name="sensitivity", marker_color=t.SUCCESS))
    fig.add_trace(go.Bar(x=ok.group, y=ok.flag_rate, name="flag rate", marker_color=t.SECONDARY))
    fig.update_layout(barmode="group", yaxis_range=[0, 1.05], yaxis_tickformat=".0%")
    st.plotly_chart(t.style_fig(fig, 320), use_container_width=True)

t.section("Calibration-in-the-large by group", "mean predicted vs observed rate")
fig = go.Figure()
fig.add_trace(go.Bar(x=ok.group, y=ok.observed_rate, name="observed", marker_color=t.MUTED))
fig.add_trace(go.Bar(x=ok.group, y=ok.mean_predicted, name="mean predicted", marker_color=t.ACCENT))
fig.update_layout(barmode="group", yaxis_tickformat=".0%")
st.plotly_chart(t.style_fig(fig, 300), use_container_width=True)
st.dataframe(g, use_container_width=True, hide_index=True)

st.markdown('<div class="disclaimer">Subgroup attributes are used for <b>evaluation</b>. A sensitivity gap at one shared '
            'threshold largely reflects different base rates and risk distributions; whether group-specific thresholds are '
            'appropriate is an ethical and clinical-governance decision, not a purely statistical one. Wide CIs mean '
            'differences may not be reliable.</div>', unsafe_allow_html=True)
