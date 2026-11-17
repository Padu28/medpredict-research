import plotly.graph_objects as go
import streamlit as st
import theme as t

import data

s = data.summary()
pm = s["primary_metrics"]
at = pm["at_threshold"]
t.hero("MedPredict Research", "How do calibration, subgroup evaluation and dataset-shift tests change our assessment of a "
       "clinical risk model compared with accuracy alone? A reliability study on a synthetic 30-day readmission cohort.",
       badge="Educational · non-clinical", icon="🩺")

t.kpis([
    ("AUROC (test)", f"{pm['auroc']:.3f}", f"95% CI {pm['auroc_ci'][0]:.3f}–{pm['auroc_ci'][1]:.3f}", "📈", t.PRIMARY),
    ("Brier score", f"{pm['brier']:.4f}", "lower is better", "🎯", t.SECONDARY),
    ("Calibration error", f"{pm['ece']:.3f}", f"slope {pm['cal_slope']:.2f}", "⚖️", t.ACCENT),
    ("Sensitivity @ thr.", f"{at['sensitivity']:.1%}", f"specificity {at['specificity']:.1%}", "🧲", t.SUCCESS),
    ("Accuracy @ 0.5", f"{pm['accuracy_at_0.5']:.3f}", "the misleading number", "🙈", t.WARNING),
])

t.section("Study design")
t.flow([("Synthetic cohort", f"{s['audit']['n_patients']:,} patients"), ("Leakage checks", "post-outcome cols excluded"),
        ("Temporal + patient split", "no patient in 2 splits"), ("3 models × 3 calibrations", "LR · RF · HGB"),
        ("Select on validation", "log loss"), ("Subgroups · shift · uncertainty", "with CIs")])

c1, c2 = st.columns([1.2, 1])
with c1:
    t.section("Accuracy vs. calibration", "accuracy barely moves; calibration does")
    r = data.results()
    r["label"] = r.model.str.replace("_", " ") + " · " + r.calibration
    fig = go.Figure()
    fig.add_trace(go.Bar(x=r.label, y=r["accuracy_at_0.5"], name="accuracy@0.5", marker_color=t.MUTED))
    fig.add_trace(go.Scatter(x=r.label, y=r.ece * 10, name="ECE × 10", mode="markers+lines",
                             marker=dict(size=11, color=t.ACCENT), yaxis="y"))
    fig.update_layout(xaxis_tickangle=-35, yaxis_range=[0, 0.85])
    st.plotly_chart(t.style_fig(fig, 380), use_container_width=True)
with c2:
    t.section("Key findings", "computed from this run")
    d = s["disparities"]
    sh = data.shifts()
    site = sh[(sh.test == "site")]
    t.card("Site shift breaks calibration",
           f"On held-out site C, ECE rises to <b>{site.ece.max():.3f}</b> (vs {site.ece.min():.3f} in-distribution) — "
           "risk is under-estimated where prevalence is higher.", t.DANGER)
    t.card("One threshold ≠ equal opportunity",
           f"Sensitivity varies by <b>{d.get('age_band', {}).get('sensitivity_gap', 0):.2f}</b> across age bands at the same "
           "threshold — younger patients are flagged far less often.", t.WARNING)
    t.card("Calibration is cheap and matters",
           "Post-hoc calibration on validation data reduces calibration error for every model family without "
           "changing ranking (AUROC).", t.PRIMARY)
