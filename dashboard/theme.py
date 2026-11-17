"""MedPredict Research design system — "Clinical Calm" (light).

Soft pastel glassmorphism, serif editorial headings, floating gradient blobs,
and an animated hero with a live ECG trace and a breathing orb. Calm motion
only: nothing flashes, as befits a sensitive health context.
"""
from __future__ import annotations

import html
from collections.abc import Iterable

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st
import streamlit.components.v1 as components

BRAND = "MedPredict Research"
TAGLINE = "Calibrated · fair · shift-tested risk models"
PRIMARY = "#4f6df5"     # calm blue
SECONDARY = "#14b8a6"   # teal
ACCENT = "#ec4899"      # rose
SUCCESS, WARNING, DANGER, MUTED = "#10b981", "#f59e0b", "#ef4444", "#6b6f86"
BG, SURFACE, BORDER, TEXT = "#f6f7fb", "rgba(255,255,255,0.72)", "#e3e6f2", "#1e1b3a"
COLORWAY = [PRIMARY, SECONDARY, ACCENT, "#8b5cf6", WARNING, "#0ea5e9", "#84cc16", "#64748b"]
MODEL_COLORS = {"logistic_regression": PRIMARY, "random_forest": SUCCESS, "hist_gradient_boosting": WARNING}
CAL_COLORS = {"uncalibrated": "#a1a5b8", "sigmoid": PRIMARY, "isotonic": ACCENT}
SEVERITY_COLORS = {"critical": DANGER, "high": DANGER, "medium": WARNING, "low": SECONDARY, "info": PRIMARY}

pio.templates["medpredict"] = go.layout.Template(layout=dict(
    font=dict(family="Inter, system-ui, sans-serif", color=TEXT, size=12.5),
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,0.6)", colorway=COLORWAY,
    xaxis=dict(gridcolor="#eceef6", zeroline=False, linecolor=BORDER),
    yaxis=dict(gridcolor="#eceef6", zeroline=False, linecolor=BORDER),
    legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", y=1.1, x=0),
    margin=dict(l=10, r=10, t=44, b=10),
    hoverlabel=dict(bgcolor="#ffffff", bordercolor=PRIMARY, font=dict(family="Inter", color=TEXT)),
    transition=dict(duration=700, easing="cubic-in-out"),
))
pio.templates.default = "plotly_white+medpredict"

_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600;9..144,700&family=Inter:wght@400;500;600;700&display=swap');
@property --ring { syntax: '<number>'; inherits: false; initial-value: 0; }
html, body, [class*="css"], .stMarkdown { font-family: 'Inter', system-ui, sans-serif; color: __TEXT__; }
.serif, h1, h2, h3 { font-family: 'Fraunces', Georgia, serif !important; }
.stApp { background: __BG__; }
/* floating pastel blobs */
.stApp::before, .stApp::after { content: ""; position: fixed; z-index: 0; border-radius: 50%; filter: blur(70px); opacity: .55; pointer-events: none; }
.stApp::before { width: 520px; height: 520px; background: #c7d2fe; left: -120px; top: -120px; animation: float1 22s ease-in-out infinite alternate; }
.stApp::after { width: 460px; height: 460px; background: #fbcfe8; right: -140px; bottom: -120px; animation: float2 26s ease-in-out infinite alternate; }
@keyframes float1 { to { transform: translate(180px, 140px) scale(1.15); } }
@keyframes float2 { to { transform: translate(-160px, -120px) scale(1.1); } }
[data-testid="stAppViewContainer"] > * { position: relative; z-index: 1; }
[data-testid="stSidebar"] { background: rgba(255,255,255,.7); backdrop-filter: blur(18px); border-right: 1px solid __BORDER__; }
[data-testid="stSidebarNavLink"] { border-radius: 12px !important; transition: background .25s; }
[data-testid="stSidebarNavLink"][aria-current="page"] { background: #eef1ff !important; }
[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 1.6rem; max-width: 1380px; }

@keyframes soft { from { opacity: 0; transform: translateY(12px); } to { opacity: 1; transform: none; } }
@keyframes breathe { 0%,100% { box-shadow: 0 10px 30px -18px color-mix(in srgb, var(--c1) 60%, transparent); } 50% { box-shadow: 0 16px 40px -14px color-mix(in srgb, var(--c1) 75%, transparent); } }
@keyframes ringFill { from { --ring: 0; } }
@keyframes dash { to { stroke-dashoffset: -20; } }
.element-container, .stPlotlyChart { animation: soft .8s cubic-bezier(.22,1,.36,1) both; }
.stPlotlyChart { background: rgba(255,255,255,.65); border: 1px solid __BORDER__; border-radius: 20px; padding: 8px; backdrop-filter: blur(10px); }

.kpi-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 14px; margin: 10px 0 22px; }
.kpi { position: relative; padding: 18px 20px; border-radius: 22px; background: __SURFACE__; backdrop-filter: blur(14px);
  border: 1px solid #ffffffcc; outline: 1px solid __BORDER__; overflow: hidden;
  animation: soft .8s cubic-bezier(.22,1,.36,1) both, breathe 6s ease-in-out infinite; }
.kpi::before { content: ""; position: absolute; width: 140px; height: 140px; border-radius: 50%; right: -50px; top: -60px;
  background: radial-gradient(circle, color-mix(in srgb, var(--c1) 30%, transparent), transparent 70%); }
.kpi .label { color: __MUTED__; font-size: .78rem; font-weight: 600; }
.kpi .value { font-family: 'Fraunces', Georgia, serif; font-size: 2.05rem; font-weight: 600; color: __TEXT__; margin-top: 4px; letter-spacing: -.02em; }
.kpi .delta { font-size: .78rem; margin-top: 2px; font-weight: 500; }
.kpi .icon { position: absolute; right: 16px; bottom: 14px; font-size: 1.3rem; opacity: .8; }

.section { margin: 32px 0 12px; animation: soft .6s ease both; }
.section .eyebrow { display: inline-block; font-size: .7rem; font-weight: 700; letter-spacing: .12em; text-transform: uppercase; color: __PRIMARY__;
  background: #eef1ff; padding: 3px 10px; border-radius: 999px; }
.section h3 { margin: 6px 0 0; font-size: 1.45rem; font-weight: 600; color: __TEXT__; letter-spacing: -.01em; }
.section .sub { color: __MUTED__; font-size: .88rem; }

.card { background: __SURFACE__; backdrop-filter: blur(14px); border: 1px solid __BORDER__; border-radius: 22px; padding: 20px 22px;
  margin-bottom: 12px; animation: soft .7s ease both; transition: transform .35s cubic-bezier(.22,1,.36,1), box-shadow .35s; }
.card:hover { transform: translateY(-4px); box-shadow: 0 24px 40px -28px #4f6df580; }
.card h4 { margin: 0 0 6px; font-family: 'Fraunces', Georgia, serif; font-weight: 600; font-size: 1.12rem; color: __TEXT__; }
.card p { margin: 0; color: __MUTED__; font-size: .92rem; line-height: 1.6; }

.tl-item { background: __SURFACE__; border: 1px solid __BORDER__; border-radius: 16px; padding: 10px 14px 10px 16px; margin-bottom: 8px;
  box-shadow: inset 4px 0 0 var(--sev); animation: soft .5s ease both; }
.tl-item .meta { font-size: .74rem; color: __MUTED__; }
.tl-item .msg { font-size: .9rem; margin-top: 3px; }
.chip { display: inline-block; padding: 3px 11px; border-radius: 999px; font-size: .72rem; font-weight: 600;
  color: var(--c); background: color-mix(in srgb, var(--c) 12%, white); }

.ring { width: 144px; height: 144px; margin: auto; border-radius: 50%; display: grid; place-items: center; --ring: var(--val);
  background: conic-gradient(from -90deg, var(--rc) calc(var(--ring) * 1%), #e9ecf6 0); animation: ringFill 2.2s cubic-bezier(.22,1,.36,1) both; }
.ring > div { width: 118px; height: 118px; border-radius: 50%; background: #fff; display: grid; place-items: center;
  font: 600 1.8rem 'Fraunces', Georgia, serif; color: __TEXT__; box-shadow: 0 8px 20px -12px #1e1b3a55; }

.flow { display: flex; align-items: center; flex-wrap: wrap; margin: 8px 0 14px; }
.flow .node { background: __SURFACE__; border: 1px solid __BORDER__; border-radius: 999px; padding: 9px 16px; font-weight: 600; font-size: .85rem;
  animation: soft .6s ease both; }
.flow .node small { display: block; color: __MUTED__; font-weight: 500; font-size: .7rem; }
.flow svg { width: 44px; height: 12px; }
.flow svg line { stroke: #a5b4fc; stroke-width: 2.5; stroke-linecap: round; stroke-dasharray: 1 7; animation: dash 1.2s linear infinite; }

.banner { display: flex; gap: 14px; align-items: center; padding: 12px 18px; border-radius: 18px; margin-bottom: 14px;
  background: linear-gradient(90deg, #fff7ed, #fffbeb); border: 1px solid #fed7aa; animation: soft .6s ease both; }
.banner .shield { width: 34px; height: 34px; border-radius: 50%; display: grid; place-items: center; background: #ffedd5; font-size: 1.1rem;
  animation: breathe 4s ease-in-out infinite; --c1: #f59e0b; }
.banner p { margin: 0; font-size: .86rem; color: #7c4a03; }

.stButton > button, .stDownloadButton > button { border-radius: 999px; border: 0; color: #fff; font-weight: 600;
  background: linear-gradient(135deg, __PRIMARY__, #7c8cff); box-shadow: 0 10px 24px -14px __PRIMARY__; transition: transform .2s; }
.stButton > button:hover { transform: translateY(-1px); color: #fff; }
[data-testid="stDataFrame"] { border-radius: 16px; overflow: hidden; border: 1px solid __BORDER__; }
.disclaimer { font-size: .82rem; color: __MUTED__; background: rgba(255,255,255,.7); border: 1px solid __BORDER__; border-radius: 16px;
  padding: 12px 16px; margin-top: 18px; }
.brand { font: 600 1.25rem 'Fraunces', Georgia, serif; color: __TEXT__; }
.brand small { display: block; font: 500 .74rem 'Inter', sans-serif; color: __MUTED__; margin-top: 2px; }
.footer { text-align: center; color: __MUTED__; font-size: .78rem; margin-top: 40px; }
@media (prefers-reduced-motion: reduce) { *, *::before, *::after { animation: none !important; } }
</style>
"""
_TOK = {"__BG__": BG, "__SURFACE__": SURFACE, "__BORDER__": BORDER, "__TEXT__": TEXT, "__MUTED__": MUTED, "__PRIMARY__": PRIMARY}


def _fill(s: str) -> str:
    for k, v in _TOK.items():
        s = s.replace(k, v)
    return s


def _e(s: object) -> str:
    return html.escape(str(s))


def apply_theme() -> None:
    st.markdown(_fill(_CSS), unsafe_allow_html=True)


_HERO = """<!DOCTYPE html><html><head><style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,600&family=Inter:wght@400;500;600&display=swap');
html,body{margin:0;background:transparent;overflow:hidden;font-family:Inter,system-ui,sans-serif;color:#1e1b3a}
.w{position:relative;height:__H__px;border-radius:28px;overflow:hidden;border:1px solid #e3e6f2;
 background:linear-gradient(120deg,#ffffffee,#f5f3ffee 55%,#fdf2f8ee);display:grid;grid-template-columns:1.2fr 1fr}
.blob{position:absolute;border-radius:50%;filter:blur(40px);opacity:.6}
.b1{width:260px;height:260px;background:#c7d2fe;left:-60px;top:-80px;animation:m1 14s ease-in-out infinite alternate}
.b2{width:220px;height:220px;background:#99f6e4;left:38%;bottom:-120px;animation:m2 17s ease-in-out infinite alternate}
@keyframes m1{to{transform:translate(80px,40px)}}@keyframes m2{to{transform:translate(-60px,-30px)}}
.l{position:relative;padding:30px 34px}
.eb{display:inline-flex;align-items:center;gap:8px;font-size:12px;font-weight:600;color:#4f6df5;background:#eef1ffcc;padding:5px 12px;border-radius:999px}
.eb i{width:8px;height:8px;border-radius:50%;background:#10b981;animation:br 2.4s ease-in-out infinite}
@keyframes br{50%{transform:scale(1.6);opacity:.5}}
h1{font-family:Fraunces,Georgia,serif;font-weight:600;font-size:40px;letter-spacing:-.02em;margin:14px 0 8px;opacity:0;transform:translateY(10px);animation:in 1s cubic-bezier(.22,1,.36,1) .1s forwards}
h1 em{font-style:italic;color:#4f6df5}
@keyframes in{to{opacity:1;transform:none}}
p{margin:0;color:#6b6f86;font-size:14.5px;line-height:1.65;max-width:560px;opacity:0;animation:in 1s ease .4s forwards}
.r{position:relative;display:grid;place-items:center}
.mon{position:relative;width:88%;height:72%;border-radius:20px;background:#0f172a;box-shadow:0 30px 50px -30px #1e1b3a99;overflow:hidden}
.mon .grid{position:absolute;inset:0;background:linear-gradient(#22c55e14 1px,transparent 1px) 0 0/100% 18px,linear-gradient(90deg,#22c55e14 1px,transparent 1px) 0 0/18px 100%}
.mon svg{position:absolute;inset:0;width:100%;height:100%}
.trace{fill:none;stroke:#34d399;stroke-width:2.2;stroke-linejoin:round;filter:drop-shadow(0 0 6px #34d399);stroke-dasharray:1050;stroke-dashoffset:1050;animation:draw 3s linear infinite}
@keyframes draw{to{stroke-dashoffset:0}}
.mask{position:absolute;top:0;bottom:0;width:22%;background:linear-gradient(90deg,#0f172a,#0f172a00);animation:sw 3s linear infinite}
@keyframes sw{from{left:-22%}to{left:100%}}
.bpm{position:absolute;right:14px;top:10px;text-align:right;color:#34d399;font:600 11px Inter}
.bpm b{display:block;font:600 28px Fraunces,Georgia,serif;color:#a7f3d0}
.heart{display:inline-block;color:#fb7185;animation:hb 1s ease-in-out infinite}
@keyframes hb{0%,100%{transform:scale(1)}15%{transform:scale(1.35)}30%{transform:scale(1)}45%{transform:scale(1.2)}}
.orb{position:absolute;left:14px;bottom:12px;width:16px;height:16px;border-radius:50%;background:radial-gradient(circle,#c7d2fe,#4f6df5);animation:breath 5s ease-in-out infinite}
@keyframes breath{50%{transform:scale(1.7);opacity:.55}}
.lab{position:absolute;left:38px;bottom:11px;font:500 10px Inter;color:#94a3b8;letter-spacing:.1em}
@media(max-width:820px){.w{grid-template-columns:1fr}.r{display:none}h1{font-size:30px}}
</style></head><body><div class="w"><div class="blob b1"></div><div class="blob b2"></div>
<div class="l"><span class="eb"><i></i>__BADGE__</span><h1>__TITLE_HTML__</h1><p>__SUB__</p></div>
<div class="r"><div class="mon"><div class="grid"></div>
<svg viewBox="0 0 600 200" preserveAspectRatio="none"><path class="trace" d="M0 110 L60 110 L72 104 L84 110 L110 110 L118 122 L128 40 L140 150 L150 110 L190 110 L206 96 L226 110 L300 110 L312 104 L324 110 L350 110 L358 122 L368 40 L380 150 L390 110 L430 110 L446 96 L466 110 L540 110 L552 104 L564 110 L600 110"/></svg>
<div class="mask"></div><div class="bpm"><span class="heart">♥</span> SIMULATED<b>72</b>bpm</div>
<div class="orb"></div><div class="lab">SYNTHETIC SIGNAL · NOT CLINICAL DATA</div></div></div></div></body></html>"""


def hero(title: str, subtitle: str, badge: str = "Live", icon: str = "🩺", height: int = 250) -> None:
    words = _e(title).split(" ")
    title_html = " ".join(words[:-1] + [f"<em>{words[-1]}</em>"]) if len(words) > 1 else _e(title)
    doc = (_HERO.replace("__H__", str(height - 8)).replace("__BADGE__", _e(badge)).replace("__TITLE_HTML__", title_html)
           .replace("__SUB__", _e(subtitle)))
    components.html(doc, height=height)


def safety_banner() -> None:
    st.markdown('<div class="banner"><div class="shield">🛡️</div><p><b>Educational research prototype — not a diagnostic device '
                'or clinical decision system.</b> Synthetic, de-identified data. Do not use for real patient care.</p></div>',
                unsafe_allow_html=True)


def sidebar_brand() -> None:
    st.markdown(f'<div class="brand">🩺 {BRAND}<small>{TAGLINE}</small></div>', unsafe_allow_html=True)


def kpis(items: Iterable[tuple]) -> None:
    cards = []
    for i, it in enumerate(items):
        label, value, delta, icon, color = (list(it) + [None] * 5)[:5]
        color = color or PRIMARY
        d = ""
        if delta:
            dc = SUCCESS if str(delta).startswith(("+", "▲")) else DANGER if str(delta).startswith(("-", "−", "▼")) else MUTED
            d = f'<div class="delta" style="color:{dc}">{_e(delta)}</div>'
        cards.append(f'<div class="kpi" style="--c1:{color};animation-delay:{i * .1:.2f}s,{i * .4:.1f}s"><div class="icon">{icon or ""}</div>'
                     f'<div class="label">{_e(label)}</div><div class="value">{_e(value)}</div>{d}</div>')
    st.markdown(f'<div class="kpi-grid">{"".join(cards)}</div>', unsafe_allow_html=True)


_N = {"i": 0}


def section(title: str, sub: str = "") -> None:
    _N["i"] = _N["i"] % 99 + 1
    st.markdown(f'<div class="section"><span class="eyebrow">Part {_N["i"]:02d}</span><h3>{_e(title)}</h3>'
                f'{f"<div class=sub>{_e(sub)}</div>" if sub else ""}</div>', unsafe_allow_html=True)


def card(title: str, body: str, accent: str = PRIMARY) -> None:
    st.markdown(f'<div class="card" style="box-shadow: inset 0 3px 0 {accent}"><h4>{_e(title)}</h4><p>{body}</p></div>',
                unsafe_allow_html=True)


def chip(text: str, color: str) -> str:
    return f'<span class="chip" style="--c:{color}">{_e(text)}</span>'


def timeline(events: list[dict], limit: int = 8) -> None:
    out = [f'<div class="tl-item" style="--sev:{SEVERITY_COLORS.get(ev.get("severity", "info"), PRIMARY)};animation-delay:{i * .06:.2f}s">'
           f'<div class="meta">{chip(ev.get("event_type", ""), SEVERITY_COLORS.get(ev.get("severity", "info"), PRIMARY))} '
           f'{_e(ev.get("ts", ""))}</div><div class="msg">{_e(ev.get("message", ""))}</div></div>' for i, ev in enumerate(events[:limit])]
    st.markdown("".join(out) or '<div class="card"><p>Nothing yet.</p></div>', unsafe_allow_html=True)


def ring(value: float, label: str, color: str | None = None) -> None:
    color = color or (SUCCESS if value >= 80 else WARNING if value >= 60 else DANGER)
    st.markdown(f'<div style="text-align:center"><div class="ring" style="--val:{min(max(value, 0), 100):.0f};--rc:{color}"><div>{value:.0f}</div></div>'
                f'<div style="color:{MUTED};font-size:.82rem;font-weight:600;margin-top:10px">{_e(label)}</div></div>', unsafe_allow_html=True)


def flow(nodes: list[tuple[str, str]]) -> None:
    arrow = '<svg viewBox="0 0 44 12"><line x1="3" y1="6" x2="41" y2="6"/></svg>'
    parts = [f'<div class="node" style="animation-delay:{i * .12:.2f}s">{_e(n)}<small>{_e(s)}</small></div>' for i, (n, s) in enumerate(nodes)]
    st.markdown(f'<div class="flow">{arrow.join(parts)}</div>', unsafe_allow_html=True)


def footer() -> None:
    st.markdown(f'<div class="footer">{BRAND} · educational, non-clinical research prototype · synthetic data</div>', unsafe_allow_html=True)


def style_fig(fig: go.Figure, height: int = 360, title: str | None = None) -> go.Figure:
    fig.update_layout(height=height, title=dict(text=title, font=dict(family="Fraunces, Georgia, serif", size=16, color=TEXT)) if title else None)
    return fig
