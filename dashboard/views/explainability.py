import numpy as np
import plotly.graph_objects as go
import streamlit as st
import theme as t

import data

t.hero("Explainability", "Standardised logistic-regression coefficients and permutation importance of the selected model. "
       "These are associations learned from data — not causal effects and not treatment guidance.", badge="Associations only", icon="🧠")

e = data.explanations()
c1, c2 = st.columns(2)
with c1:
    t.section("Logistic regression", "log-odds per 1 SD (or category vs reference)")
    lr = e[e.kind == "lr_coefficient"].head(18).iloc[::-1]
    fig = go.Figure(go.Bar(x=lr.coefficient, y=lr.feature, orientation="h",
                           marker_color=[t.ACCENT if v > 0 else t.PRIMARY for v in lr.coefficient],
                           customdata=np.exp(lr.coefficient), hovertemplate="%{y}<br>β=%{x:.3f} · OR=%{customdata:.2f}<extra></extra>"))
    fig.add_vline(x=0, line_color=t.MUTED)
    st.plotly_chart(t.style_fig(fig, 520), use_container_width=True)
with c2:
    t.section("Permutation importance", "AUROC drop on the test period")
    pi = e[e.kind == "permutation"].head(14).iloc[::-1]
    fig = go.Figure(go.Bar(x=pi.coefficient, y=pi.feature, orientation="h",
                           marker=dict(color=pi.coefficient, colorscale=[[0, t.SECONDARY], [1, t.PRIMARY]])))
    st.plotly_chart(t.style_fig(fig, 520), use_container_width=True)

st.markdown('<div class="disclaimer">Missing-value indicators (e.g. <code>missingindicator_hba1c</code>) can carry signal '
            'because <i>whether</i> a test was ordered reflects care processes and site practices — a classic source of '
            'spurious, non-transferable associations.</div>', unsafe_allow_html=True)
