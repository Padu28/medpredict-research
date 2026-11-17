import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import theme as t

import data

t.hero("Model Comparison", "Logistic regression baseline vs random forest vs gradient boosting, each uncalibrated and with "
       "sigmoid / isotonic calibration. Selection uses validation log loss — never test data.", badge="Test period", icon="📊")

r = data.results()
s = data.summary()
prim = s["primary"]
show = r[["model", "calibration", "auroc", "pr_auc", "brier", "log_loss", "ece", "cal_slope", "accuracy_at_0.5", "val_log_loss"]].copy()
show["selected"] = (show.model == prim["model"]) & (show.calibration == prim["calibration"])
st.dataframe(show, use_container_width=True, hide_index=True, column_config={
    "auroc": st.column_config.ProgressColumn("AUROC", min_value=0.5, max_value=0.8, format="%.3f"),
    "pr_auc": st.column_config.NumberColumn("PR-AUC", format="%.3f"),
    "brier": st.column_config.NumberColumn("Brier", format="%.4f"),
    "log_loss": st.column_config.NumberColumn("Log loss", format="%.4f"),
    "ece": st.column_config.NumberColumn("ECE", format="%.3f"),
    "cal_slope": st.column_config.NumberColumn("Cal. slope", format="%.2f"),
    "accuracy_at_0.5": st.column_config.NumberColumn("Acc@0.5", format="%.3f"),
    "val_log_loss": st.column_config.NumberColumn("Val log loss", format="%.4f"),
    "selected": st.column_config.CheckboxColumn("Selected")})

c1, c2 = st.columns(2)
with c1:
    t.section("AUROC with 95% bootstrap CI")
    u = r[r.calibration == "uncalibrated"]
    fig = go.Figure(go.Scatter(x=u.auroc, y=u.model, mode="markers", marker=dict(size=14, color=[t.MODEL_COLORS[m] for m in u.model]),
                               error_x=dict(type="data", symmetric=False,
                                            array=[ci[1] - a for ci, a in zip(u.auroc_ci, u.auroc)],
                                            arrayminus=[a - ci[0] for ci, a in zip(u.auroc_ci, u.auroc)])))
    fig.update_layout(xaxis_title="AUROC")
    st.plotly_chart(t.style_fig(fig, 300), use_container_width=True)
    st.caption("Overlapping intervals: the tree models' AUROC advantage is not clearly distinguishable from the baseline.")
with c2:
    t.section("Brier score by calibration")
    fig = go.Figure()
    for cal, g in r.groupby("calibration"):
        fig.add_trace(go.Bar(x=g.model, y=g.brier, name=cal, marker_color=t.CAL_COLORS[cal]))
    fig.update_layout(barmode="group", yaxis_range=[r.brier.min() * 0.97, r.brier.max() * 1.02])
    st.plotly_chart(t.style_fig(fig, 300), use_container_width=True)

t.section("Threshold analysis", f"frozen on validation for sensitivity ≥ 0.80 → {prim['threshold']:.3f}")
p = data.predictions()
grid = np.linspace(0.02, 0.9, 89)
y, risk = p.outcome.to_numpy(), p.risk.to_numpy()
sens = [((risk >= g) & (y == 1)).sum() / max((y == 1).sum(), 1) for g in grid]
spec = [((risk < g) & (y == 0)).sum() / max((y == 0).sum(), 1) for g in grid]
ppv = [((risk >= g) & (y == 1)).sum() / max((risk >= g).sum(), 1) for g in grid]
fig = go.Figure()
for name, vals, col in (("sensitivity", sens, t.SUCCESS), ("specificity", spec, t.PRIMARY), ("PPV", ppv, t.ACCENT)):
    fig.add_trace(go.Scatter(x=grid, y=vals, name=name, line=dict(color=col, width=3)))
fig.add_vline(x=prim["threshold"], line_dash="dot", line_color=t.TEXT, annotation_text="frozen")
fig.add_vline(x=0.5, line_dash="dash", line_color=t.MUTED, annotation_text="0.5 (naïve)")
fig.update_layout(xaxis_title="risk threshold", yaxis_range=[0, 1.02])
st.plotly_chart(t.style_fig(fig, 340), use_container_width=True)
st.dataframe(pd.DataFrame([s["primary_metrics"]["at_threshold"]]), use_container_width=True, hide_index=True)
