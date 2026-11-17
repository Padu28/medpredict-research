import plotly.graph_objects as go
import streamlit as st
import theme as t
from plotly.subplots import make_subplots

import data

t.hero("Calibration", "Does a predicted 30 % risk mean 30 of 100 similar patients are readmitted? Reliability diagrams, "
       "ECE, Brier and calibration slope — before and after Platt (sigmoid) and isotonic calibration on validation data.",
       badge="Reliability", icon="⚖️")

c = data.curves()
r = data.results()
model = st.segmented_control("Model", list(c.model.unique()), default=data.summary()["primary"]["model"]) or c.model.iloc[0]
sub = c[c.model == model]

fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.75, 0.25], vertical_spacing=0.05)
fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color=t.MUTED, dash="dash"), name="perfect"), 1, 1)
for row in sub.itertuples():
    fig.add_trace(go.Scatter(x=row.predicted, y=row.observed, mode="lines+markers", name=row.calibration,
                             line=dict(color=t.CAL_COLORS[row.calibration], width=3, shape="spline"), marker=dict(size=9)), 1, 1)
    fig.add_trace(go.Bar(x=[i / 20 + 0.025 for i in range(20)], y=row.hist, marker_color=t.CAL_COLORS[row.calibration],
                         opacity=0.5, showlegend=False), 2, 1)
fig.update_xaxes(range=[0, 0.85], title_text="mean predicted risk", row=2, col=1)
fig.update_yaxes(range=[0, 0.85], title_text="observed rate", row=1, col=1)
fig.update_yaxes(title_text="count", row=2, col=1)
fig.update_layout(barmode="overlay")
st.plotly_chart(t.style_fig(fig, 520), use_container_width=True)

m = r[r.model == model].set_index("calibration")
cols = st.columns(3)
for col, cal in zip(cols, ["uncalibrated", "sigmoid", "isotonic"]):
    with col:
        t.kpis([(f"{cal}", f"ECE {m.loc[cal, 'ece']:.3f}",
                 f"Brier {m.loc[cal, 'brier']:.4f} · slope {m.loc[cal, 'cal_slope']:.2f}", "⚖️", t.CAL_COLORS[cal])])

st.markdown('<div class="disclaimer">Calibration is fitted on the validation period only, then applied unchanged to the later '
            'test period. Slope &lt; 1 indicates over-confident (too extreme) predictions; &gt; 1 indicates under-confident. '
            'Isotonic calibration can over-fit small validation sets (step-shaped curves).</div>', unsafe_allow_html=True)
