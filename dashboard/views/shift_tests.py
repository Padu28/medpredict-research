import plotly.graph_objects as go
import streamlit as st
import theme as t

import data

t.hero("Shift Tests", "What happens when data collection changes? Injected lab missingness, older / higher-glucose "
       "populations, a hospital site never seen in training, and a later time period.", badge="Robustness", icon="🔀")

sh = data.shifts()
metric = st.segmented_control("Metric", ["auroc", "ece", "brier", "mean_predicted"], default="ece") or "ece"

c1, c2 = st.columns(2)
with c1:
    t.section("Lab missingness (MCAR)", "HbA1c, glucose, creatinine, haemoglobin")
    m = sh[sh.test == "missingness"]
    fig = go.Figure(go.Scatter(x=m.level, y=m[metric], mode="lines+markers", line=dict(color=t.PRIMARY, width=3, shape="spline"),
                               marker=dict(size=10), fill="tozeroy", fillcolor="rgba(56,189,248,.08)"))
    fig.update_layout(yaxis_title=metric)
    st.plotly_chart(t.style_fig(fig, 320), use_container_width=True)
with c2:
    t.section("Population, site & time shifts")
    o = sh[sh.test != "missingness"]
    colors = {"covariate": t.SECONDARY, "site": t.DANGER, "temporal": t.ACCENT}
    fig = go.Figure(go.Bar(x=o[metric], y=o.test + " · " + o.level, orientation="h", marker_color=[colors[x] for x in o.test],
                           text=o[metric].round(3), textposition="outside"))
    fig.update_layout(xaxis_title=metric, yaxis=dict(autorange="reversed"))
    st.plotly_chart(t.style_fig(fig, 320), use_container_width=True)

site = sh[sh.test == "site"].set_index("level")
held = [x for x in site.index if "held-out" in x][0]
t.section("Site hold-out: predicted vs observed")
c1, c2, c3 = st.columns(3)
with c1:
    t.ring(100 * site.loc[held, "observed_rate"], "Observed readmission (held-out site) %", t.DANGER)
with c2:
    t.ring(100 * site.loc[held, "mean_predicted"], "Mean predicted risk %", t.PRIMARY)
with c3:
    t.ring(100 * site.loc[held, "ece"], "Calibration error ×100", t.WARNING)
st.dataframe(sh, use_container_width=True, hide_index=True)
st.markdown('<div class="disclaimer">The validation-period row is in-sample for the calibrator (ECE ≈ 0 by construction). '
            'Discrimination (AUROC) is fairly stable under these shifts while calibration is not — a model can keep ranking '
            'patients reasonably while its absolute risks become wrong. Local recalibration would be required before any '
            'transfer to a new site.</div>', unsafe_allow_html=True)
