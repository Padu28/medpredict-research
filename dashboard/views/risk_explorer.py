import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import theme as t

import data
from medpredict.explainability.explain import local_contributions

t.hero("Risk Explorer (research)", "Explore how the model responds to a hypothetical synthetic profile: calibrated "
       "probability, a bootstrap uncertainty band and the factors that moved it. No diagnosis or recommendation is shown.",
       badge="Hypothetical profiles only", icon="🧪")

b = data.bundle()
bg = b["background"]
c1, c2, c3 = st.columns(3)
rec = {
    "age": c1.slider("Age", 18, 95, 68), "sex": c1.selectbox("Sex", ["F", "M"]),
    "bmi": c1.slider("BMI", 15.0, 50.0, 29.0, 0.5), "systolic_bp": c1.slider("Systolic BP", 90, 200, 135),
    "glucose": c2.slider("Glucose (mg/dL)", 60, 300, 120), "hba1c": c2.slider("HbA1c (%)", 4.5, 13.0, 6.2, 0.1),
    "creatinine": c2.slider("Creatinine (mg/dL)", 0.4, 5.0, 1.0, 0.05), "hemoglobin": c2.slider("Haemoglobin (g/dL)", 7.0, 18.0, 13.0, 0.1),
    "length_of_stay": c3.slider("Length of stay (days)", 1, 30, 5), "prior_admissions_12m": c3.slider("Prior admissions (12 m)", 0, 8, 1),
    "medication_count": c3.slider("Medication count", 0, 20, 6),
    "admission_type": c3.selectbox("Admission type", ["emergency", "elective", "urgent"]),
    "smoking_status": c1.selectbox("Smoking", ["never", "former", "current"]),
    "discharge_disposition": c2.selectbox("Discharge disposition", ["home", "home_with_services", "skilled_nursing"]),
}
row = pd.DataFrame([rec])[b["features"]]
p = float(b["model"].predict_proba(row)[:, 1][0])
lo, hi = (float(x[0]) for x in b["ensemble"].band_around([p], row))

c1, c2 = st.columns([1, 1.4])
with c1:
    fig = go.Figure(go.Indicator(mode="gauge+number", value=100 * p, number=dict(suffix=" %", valueformat=".1f"),
                                 gauge=dict(axis=dict(range=[0, 100]), bar=dict(color=t.PRIMARY),
                                            steps=[dict(range=[100 * lo, 100 * hi], color="rgba(167,139,250,.35)")],
                                            threshold=dict(line=dict(color=t.ACCENT, width=3), value=100 * b["threshold"]))))
    st.plotly_chart(t.style_fig(fig, 300, "Calibrated model probability"), use_container_width=True)
    st.markdown(f"<div class='card'><p>Estimated 30-day readmission probability <b>{p:.1%}</b> for this hypothetical profile. "
                f"Bootstrap model-uncertainty band: <b>{lo:.1%} – {hi:.1%}</b>. Research threshold (pink line): "
                f"{b['threshold']:.1%}.<br><br>This is a statistical estimate from a synthetic dataset — <b>not</b> a diagnosis, "
                f"prognosis or care recommendation.</p></div>", unsafe_allow_html=True)
with c2:
    lc = local_contributions(b["lr"], row, bg).head(10).iloc[::-1]
    fig = go.Figure(go.Bar(x=lc.contribution, y=lc.feature, orientation="h",
                           marker_color=[t.ACCENT if v > 0 else t.PRIMARY for v in lc.contribution]))
    fig.add_vline(x=0, line_color=t.MUTED)
    st.plotly_chart(t.style_fig(fig, 380, "What moved the (LR) estimate vs. an average patient — log-odds"), use_container_width=True)
