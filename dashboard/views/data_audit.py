import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import theme as t

import data
from medpredict.data.cohort import generate_cohort

t.hero("Data Audit", "Data card essentials: provenance, target and prediction time, missingness, ranges, prevalence over "
       "time and sites, and the leakage columns that are deliberately excluded.", badge="Synthetic cohort", icon="🧾")

s = data.summary()
a = s["audit"]
t.kpis([
    ("Encounters", f"{a['n_rows']:,}", None, "🏥", t.PRIMARY),
    ("Patients", f"{a['n_patients']:,}", "patient-level split", "👥", t.SECONDARY),
    ("Prevalence", f"{a['prevalence']:.1%}", "30-day readmission", "📊", t.ACCENT),
    ("Leakage columns", f"{len(a['leakage_columns_present'])}", "present → excluded", "🚫", t.DANGER),
])
st.markdown(f"<div class='card'><h4>Target & prediction time</h4><p>{s['target_definition']}</p>"
            f"<p style='margin-top:8px'>Excluded post-outcome columns: <code>{', '.join(a['leakage_columns_present'])}</code></p></div>",
            unsafe_allow_html=True)


@st.cache_data
def cohort() -> pd.DataFrame:
    return generate_cohort(data.settings.dataset["synthetic"])


df = cohort()
c1, c2 = st.columns(2)
with c1:
    t.section("Missingness by feature")
    miss = pd.Series(a["missingness"]).sort_values()
    fig = go.Figure(go.Bar(x=miss.values, y=miss.index, orientation="h", marker_color=t.ACCENT,
                           text=[f"{v:.1%}" for v in miss.values], textposition="outside"))
    fig.update_layout(xaxis_tickformat=".0%")
    st.plotly_chart(t.style_fig(fig, 300), use_container_width=True)
    mby = df.groupby("site").hba1c.apply(lambda x: x.isna().mean())
    st.caption(f"HbA1c missingness by site — A {mby['A']:.0%} · B {mby['B']:.0%} · C {mby['C']:.0%} (data-collection shift)")
with c2:
    t.section("Prevalence over time and by site")
    q = df.assign(quarter=df.discharge_date.dt.to_period("Q").astype(str)).groupby(["quarter", "site"]).readmit_30d.mean().reset_index()
    fig = px.line(q, x="quarter", y="readmit_30d", color="site", markers=True,
                  color_discrete_sequence=[t.PRIMARY, t.SECONDARY, t.ACCENT])
    fig.update_layout(yaxis_tickformat=".0%", yaxis_title="readmission rate")
    st.plotly_chart(t.style_fig(fig, 300), use_container_width=True)

t.section("Temporal + patient-level split")
sp = pd.DataFrame(s["splits"]).T.reset_index(names="split")
st.dataframe(sp, use_container_width=True, hide_index=True,
             column_config={"prevalence": st.column_config.NumberColumn(format="%.3f")})

t.section("Feature distributions", "by outcome")
feat = st.selectbox("Feature", data.settings.numeric)
fig = px.histogram(df, x=feat, color=df.readmit_30d.map({0: "no readmission", 1: "readmitted"}), barmode="overlay",
                   nbins=50, opacity=0.7, histnorm="probability density", color_discrete_sequence=[t.PRIMARY, t.ACCENT])
st.plotly_chart(t.style_fig(fig, 320), use_container_width=True)
