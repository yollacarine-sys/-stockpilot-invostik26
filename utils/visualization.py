"""StockPilot - komponen UI & grafik Plotly. Palet: biru tua, biru, putih, aksen hijau/teal."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

NAVY, BLUE, TEAL, GREEN = "#0B2A4A", "#1F6FB2", "#14B8A6", "#22A06B"
RED, AMBER, GREY, LIGHT = "#D1495B", "#E0A030", "#8A99A8", "#EEF4FA"
RISK_COLORS = {"Tinggi": RED, "Sedang": AMBER, "Rendah": GREEN}
STATUS_STYLES = {
    "ACTUAL": ("DATA AKTUAL", BLUE), "MODEL": ("PREDIKSI MODEL", NAVY),
    "RULE": ("BERBASIS ATURAN", TEAL), "SIM": ("SIMULASI", AMBER), "PLAN": ("RENCANA", GREY),
}

CSS = f"""
<style>
.block-container {{padding-top: 1.4rem; max-width: 1280px;}}
h1, h2, h3 {{color: {NAVY};}}
section[data-testid="stSidebar"] {{background: {NAVY};}}
section[data-testid="stSidebar"] * {{color: #E8F0F8 !important;}}
.sp-kpi {{background:#fff;border:1px solid #D5E2EF;border-left:5px solid {BLUE};border-radius:10px;padding:14px 16px;
         box-shadow:0 1px 3px rgba(11,42,74,.06);}}
.sp-kpi .lbl {{font-size:.78rem;color:#5B6F84;text-transform:uppercase;letter-spacing:.04em;}}
.sp-kpi .val {{font-size:1.75rem;font-weight:700;color:{NAVY};line-height:1.2;}}
.sp-kpi .sub {{font-size:.78rem;color:#5B6F84;}}
.sp-badge {{display:inline-block;padding:2px 10px;border-radius:12px;color:#fff;font-size:.74rem;font-weight:600;letter-spacing:.03em;}}
.sp-panel {{background:{LIGHT};border:1px solid #D5E2EF;border-radius:10px;padding:12px 16px;margin:6px 0 12px 0;}}
.sp-note {{font-size:.85rem;color:#3C5068;}}
</style>
"""


def badge(text: str, color: str) -> str:
    return f'<span class="sp-badge" style="background:{color}">{text}</span>'


def status_badge(kind: str) -> str:
    t, c = STATUS_STYLES[kind]
    return badge(t, c)


def risk_badge(cat: str) -> str:
    return badge(cat, RISK_COLORS.get(cat, GREY))


def kpi_card(label: str, value: str, sub: str = "", color: str = BLUE) -> str:
    return (f'<div class="sp-kpi" style="border-left-color:{color}"><div class="lbl">{label}</div>'
            f'<div class="val">{value}</div><div class="sub">{sub}</div></div>')


def _layout(fig, title=None, h=360):
    fig.update_layout(title=title, height=h, margin=dict(l=10, r=10, t=50 if title else 20, b=10),
                      paper_bgcolor="white", plot_bgcolor="white", font=dict(color=NAVY),
                      legend=dict(orientation="h", y=-0.2))
    fig.update_xaxes(gridcolor="#E6EEF6"); fig.update_yaxes(gridcolor="#E6EEF6")
    return fig


def fig_risk_distribution(cats: pd.Series):
    order = ["Rendah", "Sedang", "Tinggi"]
    vc = cats.value_counts().reindex(order).fillna(0).reset_index()
    vc.columns = ["Kategori", "Jumlah"]
    fig = px.bar(vc, x="Kategori", y="Jumlah", color="Kategori", color_discrete_map=RISK_COLORS, text="Jumlah")
    fig.update_layout(showlegend=False)
    return _layout(fig, "Distribusi kategori risiko", 330)


def fig_prob_hist(prob, high, med):
    fig = px.histogram(x=prob, nbins=40, color_discrete_sequence=[BLUE], labels={"x": "Probabilitas risiko (tidak terkalibrasi)"})
    fig.add_vline(x=high, line_dash="dot", line_color=RED, annotation_text="Tinggi")
    fig.add_vline(x=med, line_dash="dot", line_color=AMBER, annotation_text="Sedang")
    fig.update_yaxes(type="log", title="Jumlah SKU (log)")
    return _layout(fig, "Distribusi probabilitas risiko", 330)


def fig_hist(series: pd.Series, title: str, log_x: bool = False, color=BLUE):
    s = pd.to_numeric(series, errors="coerce").dropna()
    if log_x:
        s = np.sign(s) * np.log1p(np.abs(s))
    fig = px.histogram(x=s, nbins=50, color_discrete_sequence=[color],
                       labels={"x": title + (" (signed log1p)" if log_x else "")})
    return _layout(fig, title, 320)


def fig_scatter(df, x, y, color=None, hover=None, title="", log=True):
    d = df.copy()
    kw = dict(x=x, y=y, hover_data=hover, opacity=.65)
    if color:
        kw["color"] = color
        if color == "risk_category":
            kw["color_discrete_map"] = RISK_COLORS
    fig = px.scatter(d, **kw)
    if log:
        fig.update_xaxes(type="log"); fig.update_yaxes(type="log")
    return _layout(fig, title, 380)


def fig_bar_compare(labels, base, scen, title, ylabel=""):
    fig = go.Figure()
    fig.add_bar(x=labels, y=base, name="Baseline", marker_color=GREY)
    fig.add_bar(x=labels, y=scen, name="Skenario", marker_color=TEAL)
    fig.update_layout(barmode="group", yaxis_title=ylabel)
    return _layout(fig, title, 340)


def fig_confusion(tn, fp, fn, tp):
    z = [[tn, fp], [fn, tp]]
    txt = [[f"{tn:,}", f"{fp:,}"], [f"{fn:,}", f"{tp:,}"]]
    fig = go.Figure(go.Heatmap(z=z, x=["Prediksi: aman", "Prediksi: berisiko"], y=["Aktual: tidak BO", "Aktual: BO"],
                               text=txt, texttemplate="%{text}", colorscale="Blues", showscale=False))
    fig.update_yaxes(autorange="reversed")
    return _layout(fig, "Confusion matrix (holdout Colab)", 330)


def fig_pr(pr_df, prevalence, op_point=None):
    fig = go.Figure()
    fig.add_scatter(x=pr_df["recall"], y=pr_df["precision"], mode="lines", line=dict(color=BLUE, width=3), name="Model")
    fig.add_hline(y=prevalence, line_dash="dash", line_color=GREY, annotation_text="prevalensi (acak)")
    if op_point:
        fig.add_scatter(x=[op_point[0]], y=[op_point[1]], mode="markers", marker=dict(size=12, color=RED), name="Threshold operasi")
    fig.update_layout(xaxis_title="Recall", yaxis_title="Precision")
    return _layout(fig, "Kurva Precision-Recall (holdout)", 340)


def fig_threshold(thr_df, op_thr):
    fig = go.Figure()
    for c, col in (("precision", BLUE), ("recall", RED), ("f1", TEAL)):
        fig.add_scatter(x=thr_df["threshold"], y=thr_df[c], mode="lines+markers", name=c, line=dict(color=col))
    fig.add_scatter(x=thr_df["threshold"], y=thr_df["flagged_%"] / 100, mode="lines", name="proporsi ditandai", line=dict(color=GREY, dash="dash"))
    fig.add_vline(x=op_thr, line_dash="dot", line_color=NAVY, annotation_text="threshold operasi")
    fig.update_layout(xaxis_title="Threshold")
    return _layout(fig, "Analisis threshold (holdout)", 340)


def fig_importance(imp_df, n=12):
    d = imp_df.head(n).iloc[::-1]
    fig = go.Figure(go.Bar(x=d["importance_mean"], y=d["feature"], orientation="h",
                           error_x=dict(type="data", array=d["importance_std"]), marker_color=BLUE))
    fig.update_layout(xaxis_title="Penurunan PR-AUC saat fitur diacak")
    return _layout(fig, "Permutation importance (bukan sebab-akibat)", 380)


def fig_capture(cap_df):
    fig = go.Figure()
    fig.add_scatter(x=cap_df["review_%"], y=cap_df["captured_%"], mode="lines", name="Diperiksa menurut skor risiko", line=dict(color=BLUE, width=3))
    fig.add_scatter(x=cap_df["review_%"], y=cap_df["random_baseline_%"], mode="lines", name="Diperiksa acak", line=dict(color=GREY, dash="dash"))
    fig.update_layout(xaxis_title="% SKU diperiksa", yaxis_title="% backorder aktual tertangkap")
    return _layout(fig, "Kapasitas pemeriksaan vs backorder tertangkap", 340)


def fig_slotting_layout(tbl: pd.DataFrame, loc_df: pd.DataFrame, which="recommended"):
    loc = loc_df.copy()
    if not {"x", "y"} <= set(loc.columns):
        return None
    col = "current_location" if which == "current" else "recommended_location"
    m = tbl[["pick_frequency", col]].rename(columns={col: "location_id"})
    d = loc.merge(m, on="location_id", how="left")
    d["pick_frequency"] = d["pick_frequency"].fillna(0)
    d["terisi"] = np.where(d["pick_frequency"] > 0, "terisi", "kosong")
    fig = px.scatter(d[d["pick_frequency"] > 0], x="x", y="y", size="pick_frequency", color="pick_frequency",
                     color_continuous_scale="Blues", size_max=22, hover_data=["location_id", "pick_frequency", "level"] if "level" in d else None)
    fig.add_scatter(x=[0], y=[0], mode="markers+text", text=["DOCK"], textposition="top right",
                    marker=dict(size=16, color=RED, symbol="square"), name="Dock", showlegend=False)
    fig.update_layout(coloraxis_showscale=False)
    return _layout(fig, "Peta frekuensi picking — " + ("kondisi awal" if which == "current" else "rekomendasi") + " (SIMULASI)", 360)
