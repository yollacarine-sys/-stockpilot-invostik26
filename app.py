"""
StockPilot — Smart Inventory & Material Flow Optimization
Prototipe sistem pendukung keputusan gudang (INVOSTIK 2026).

Jalankan:  streamlit run app.py
Status modul:  DATA AKTUAL | PREDIKSI MODEL | BERBASIS ATURAN | SIMULASI  (lihat badge di tiap halaman)
"""
from __future__ import annotations

import io
import os
import sys

import numpy as np
import pandas as pd
import streamlit as st

APP_DIR = os.path.dirname(os.path.abspath(__file__))
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

from stockpilot_features import TARGET  # noqa: E402
from utils import visualization as viz  # noqa: E402
from utils.analytics import (  # noqa: E402
    REPLEN_FORMULA, ArtifactError, LEAD_TIME_FACTORS, ReplenParams, assign_risk_category, capture_curve,
    inventory_metrics, load_artifacts, predict_risk, replenishment_plan)
from utils.data_loader import (  # noqa: E402
    DataLoadError, SUPPORTED_EXT, classify_dataset, data_quality_report, excel_sheet_names, module_availability,
    prepare_dataset, quality_report_to_csv, read_table, validate_for_model)
from utils.simulation import (  # noqa: E402
    SLOT_LOC_COLS, SLOT_SKU_COLS, SLOTTING_ASSUMPTIONS, generate_synthetic_slotting, params_diff,
    recommend_slotting, validate_slotting_inputs, whatif_replenishment)

MODELS_DIR = os.path.join(APP_DIR, "models")
OUTPUTS_DIR = os.path.join(APP_DIR, "outputs")
SAMPLE_DIR = os.path.join(APP_DIR, "sample_data")
RISK_EMOJI = {"Tinggi": "🔴 Tinggi", "Sedang": "🟠 Sedang", "Rendah": "🟢 Rendah"}
PAGES = ["Home / Executive Dashboard", "Data Upload & Data Quality", "Inventory Overview", "Backorder Risk Monitor",
         "Replenishment Recommendation", "Smart Slotting Simulator", "What-if & Financial Simulator",
         "Model Performance & Explainability", "Reports & Export"]
REPLEN_NEEDS = ["national_inv", "in_transit_qty", "local_bo_qty", "forecast_3_month", "sales_3_month", "lead_time", "min_bank"]

st.set_page_config(page_title="StockPilot", page_icon="📦", layout="wide")
st.markdown(viz.CSS, unsafe_allow_html=True)


# =============================================================== util umum
def csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


@st.cache_resource(show_spinner=False)
def get_artifacts():
    try:
        return load_artifacts(MODELS_DIR), None
    except ArtifactError as exc:
        return None, str(exc)


@st.cache_data(show_spinner=False)
def read_output_csv(name: str):
    p = os.path.join(OUTPUTS_DIR, name)
    return pd.read_csv(p) if os.path.exists(p) else None


def header(title: str, kinds: list[str], blurb: str = ""):
    st.title(title)
    st.markdown(" ".join(viz.status_badge(k) for k in kinds), unsafe_allow_html=True)
    if blurb:
        st.markdown(f'<div class="sp-note">{blurb}</div>', unsafe_allow_html=True)
    st.write("")


def dataset():
    return st.session_state.get("data")


def need_data(min_cols: list[str] | None = None):
    d = dataset()
    if d is None:
        st.info("Belum ada dataset. Buka **Data Upload & Data Quality** untuk mengunggah CSV/XLSX atau memuat contoh.")
        return None
    df = d["df"]
    if min_cols:
        miss = [c for c in min_cols if c not in df.columns]
        if miss:
            st.warning(f"Modul ini membutuhkan kolom: {', '.join(min_cols)}. Kolom yang tidak ada: **{', '.join(miss)}**.")
            return None
    return df


def set_dataset(df_raw: pd.DataFrame, name: str, source: str):
    df, notes = prepare_dataset(df_raw)
    kind = classify_dataset(df)
    if kind == "submission_format":
        raise DataLoadError("File ini hanya berisi 'sku' dan 'went_on_backorder' — itu format contoh submission, "
                            "bukan data persediaan. File tersebut tidak dipakai untuk analisis atau pelatihan.")
    sig = int(pd.util.hash_pandas_object(df.head(200), index=False).sum() % 10**9)
    st.session_state["data"] = {"df": df, "name": name, "kind": kind, "notes": notes, "source": source,
                                "quality": data_quality_report(df), "key": f"{name}-{len(df)}-{df.shape[1]}-{sig}"}
    for k in ("pred_cache", "slot_result", "replen_plan", "whatif_result"):
        st.session_state.pop(k, None)


def thresholds():
    arts, _ = get_artifacts()
    tiers = arts["meta"]["risk_tiers"] if arts else {"high_threshold": 0.5, "medium_threshold": 0.25}
    return (st.session_state.get("high_thr", tiers["high_threshold"]),
            st.session_state.get("med_thr", tiers["medium_threshold"]))


def get_scored(df: pd.DataFrame | None = None):
    """Prediksi (probabilitas di-cache per dataset) + kategori sesuai threshold saat ini. Raise ArtifactError bila gagal."""
    arts, err = get_artifacts()
    if arts is None:
        raise ArtifactError(err)
    d = dataset()
    df = d["df"] if df is None else df
    cache = st.session_state.get("pred_cache")
    if not cache or cache["key"] != d["key"]:
        scored = predict_risk(arts, df)
        st.session_state["pred_cache"] = {"key": d["key"], "prob": scored["risk_probability"].values,
                                          "cls": scored["predicted_class"].values}
        cache = st.session_state["pred_cache"]
    out = df.copy()
    out["risk_probability"] = cache["prob"]
    out["predicted_class"] = cache["cls"]
    out["risk_score"] = pd.Series(cache["prob"], index=out.index).rank(pct=True) * 100
    hi, me = thresholds()
    out["risk_category"] = assign_risk_category(cache["prob"], hi, me)
    return out


def try_scored(df):
    """(df_berisiko, None) bila model & kolom tersedia; selain itu (df, pesan)."""
    d = dataset()
    if d["kind"] not in ("labeled", "unlabeled"):
        return df, "Kolom model tidak lengkap — analisis risiko tidak tersedia."
    try:
        return get_scored(df), None
    except ArtifactError as exc:
        return df, str(exc)


def replen_params_from_state() -> ReplenParams:
    return st.session_state.get("replen_params", ReplenParams())


def to_excel_bytes(sheets: dict) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        for name, d in sheets.items():
            d.to_excel(xw, sheet_name=name[:31], index=False)
    return buf.getvalue()


def show_table(df: pd.DataFrame, height: int = 420):
    d = df.copy()
    cfg = {}
    if "risk_category" in d.columns:
        d["risk_category"] = d["risk_category"].map(RISK_EMOJI).fillna(d["risk_category"])
    if "risk_probability" in d.columns:
        cfg["risk_probability"] = st.column_config.ProgressColumn("risk_probability", min_value=0.0, max_value=1.0, format="%.3f")
    if "risk_score" in d.columns:
        cfg["risk_score"] = st.column_config.NumberColumn("risk_score (persentil)", format="%.1f")
    st.dataframe(d, width="stretch", height=height, column_config=cfg, hide_index=True)


# =============================================================== sidebar
def sidebar() -> str:
    st.sidebar.markdown("## 📦 StockPilot")
    st.sidebar.caption("Smart Inventory & Material Flow Optimization · INVOSTIK 2026")
    page = st.sidebar.radio("Navigasi", PAGES, key="nav", label_visibility="collapsed")
    st.sidebar.divider()
    d = dataset()
    if d:
        st.sidebar.markdown(f"**Data:** {d['name']}  \n{len(d['df']):,} baris · {d['df'].shape[1]} kolom  \nJenis: `{d['kind']}`")
    else:
        st.sidebar.markdown("**Data:** belum dimuat")
    arts, _ = get_artifacts()
    if arts:
        m = arts["meta"]
        st.sidebar.markdown(f"**Model:** {m['model_name']}  \nLatih: {m['trained_at_utc'][:10]}")
    else:
        st.sidebar.markdown("**Model:** ⚠ belum tersedia")
    st.sidebar.divider()
    st.sidebar.caption("Prototipe berbasis file CSV/XLSX. Belum terintegrasi real-time dengan IoT/RFID/ERP/WMS "
                       "dan belum divalidasi di gudang nyata.")
    return page


# =============================================================== 1. HOME
def page_home():
    header("StockPilot — Executive Dashboard", ["ACTUAL", "MODEL", "RULE"],
           "Ringkasan risiko persediaan dari dataset yang sedang dimuat. Tidak ada angka demo: semua nilai dihitung dari data & artefak model.")
    arts, err = get_artifacts()
    d = dataset()
    if d is None:
        st.info("**Mulai:** buka *Data Upload & Data Quality* → unggah CSV/XLSX persediaan (atau muat dataset contoh). Dashboard ini akan terisi otomatis.")
        c = st.columns(3)
        c[0].markdown(viz.kpi_card("Dataset", "—", "belum dimuat", viz.GREY), unsafe_allow_html=True)
        c[1].markdown(viz.kpi_card("Model", arts["meta"]["model_name"] if arts else "tidak ditemukan",
                                   f"dilatih {arts['meta']['trained_at_utc'][:10]}" if arts else (err or "")[:80],
                                   viz.TEAL if arts else viz.RED), unsafe_allow_html=True)
        if arts:
            h = arts["meta"]["metrics"]["holdout"]
            c[2].markdown(viz.kpi_card("PR-AUC holdout (Colab)", f"{h['average_precision']:.3f}", f"recall {h['recall']:.2f} · precision {h['precision']:.2f}", viz.NAVY), unsafe_allow_html=True)
            _home_model_block(arts)
        else:
            c[2].markdown(viz.kpi_card("Metrik holdout", "—", "model belum tersedia", viz.GREY), unsafe_allow_html=True)
        return
    df, q = d["df"], d["quality"]
    scored, msg = try_scored(df)
    has_risk = "risk_category" in scored.columns
    c = st.columns(5)
    c[0].markdown(viz.kpi_card("Record dianalisis", f"{len(df):,}", d["name"]), unsafe_allow_html=True)
    if has_risk:
        n_high = int((scored["risk_category"] == "Tinggi").sum())
        c[1].markdown(viz.kpi_card("SKU risiko tinggi", f"{n_high:,}", f"{100 * n_high / len(df):.1f}% dari data", viz.RED), unsafe_allow_html=True)
    else:
        c[1].markdown(viz.kpi_card("SKU risiko tinggi", "n/a", (msg or "")[:60], viz.GREY), unsafe_allow_html=True)
    n_warn = int((q["issues"]["tingkat"] == "warning").sum())
    c[2].markdown(viz.kpi_card("Kualitas data", f"{q['missing_pct']}% kosong", f"{q['n_duplicates']} duplikat · {n_warn} peringatan",
                               viz.AMBER if n_warn else viz.GREEN), unsafe_allow_html=True)
    if arts:
        h = arts["meta"]["metrics"]["holdout"]
        c[3].markdown(viz.kpi_card("Model", arts["meta"]["model_name"], f"PR-AUC holdout {h['average_precision']:.3f}", viz.TEAL), unsafe_allow_html=True)
        c[4].markdown(viz.kpi_card("Recall / Precision", f"{h['recall']:.2f} / {h['precision']:.2f}", f"@ threshold {h['threshold']:.2f} (holdout Colab)", viz.NAVY), unsafe_allow_html=True)
    else:
        c[3].markdown(viz.kpi_card("Model", "tidak ada", "artefak belum ditempatkan", viz.RED), unsafe_allow_html=True)
        c[4].markdown(viz.kpi_card("Metrik", "n/a", "", viz.GREY), unsafe_allow_html=True)
    st.write("")
    if has_risk:
        a, b = st.columns(2)
        a.plotly_chart(viz.fig_risk_distribution(scored["risk_category"]), width="stretch")
        hi, me = thresholds()
        b.plotly_chart(viz.fig_prob_hist(scored["risk_probability"], hi, me), width="stretch")
        st.caption("Probabilitas bersifat skor peringkat (tidak terkalibrasi), bukan peluang stockout sebenarnya.")
        st.subheader("Ringkasan rekomendasi", anchor=False)
        st.markdown(viz.status_badge("RULE") + " Prioritas dari kombinasi risiko model & aturan reorder point (asumsi di halaman Replenishment).", unsafe_allow_html=True)
        if all(x in df.columns for x in REPLEN_NEEDS):
            plan = replenishment_plan(scored, replen_params_from_state())
            st.dataframe(plan["action"].value_counts().rename_axis("Tindakan").reset_index(name="Jumlah SKU"), hide_index=True, width="stretch")
        else:
            st.caption("Kolom untuk replenishment tidak lengkap pada dataset ini.")
        st.subheader("10 SKU dengan skor risiko tertinggi", anchor=False)
        cols = [c_ for c_ in ["sku", "risk_probability", "risk_score", "risk_category", "national_inv", "forecast_3_month", "lead_time", "in_transit_qty"] if c_ in scored]
        show_table(scored.sort_values("risk_probability", ascending=False)[cols].head(10), 380)
    else:
        st.warning(msg or "Analisis risiko tidak tersedia untuk dataset ini.")
    if arts:
        _home_model_block(arts)


def _home_model_block(arts):
    for w in arts["warnings"]:
        st.warning(w)
    with st.expander("Status modul & batasan (baca sebelum demo)"):
        st.markdown(
            "| Modul | Status |\n|---|---|\n"
            "| Kualitas data, Inventory Overview | DATA AKTUAL (dihitung dari file) |\n"
            "| Backorder Risk Monitor | PREDIKSI MODEL (artefak Colab; probabilitas tidak terkalibrasi) |\n"
            "| Replenishment | BERBASIS ATURAN (asumsi parameter eksplisit; bukan order optimal) |\n"
            "| Smart Slotting | SIMULASI pada data sintetis / data lokasi unggahan; heuristik, bukan optimasi global |\n"
            "| What-if & finansial | SIMULASI (biaya = input pengguna/ilustrasi) |\n"
            "| Integrasi IoT/RFID/ERP/WMS | RENCANA — belum diimplementasikan |")


# =============================================================== 2. UPLOAD & QUALITY
def page_upload():
    header("Data Upload & Data Quality", ["ACTUAL"], "Unggah CSV/XLSX/XLS. Urutan kolom bebas; nama kolom dinormalisasi otomatis. "
                                                     "`sample_Submission.csv` ditolak karena hanya format contoh.")
    up = st.file_uploader("Pilih file dataset", type=[e.strip(".") for e in SUPPORTED_EXT])
    sheet = 0
    if up is not None and up.name.lower().endswith((".xlsx", ".xls")):
        try:
            names = excel_sheet_names(up)
            if len(names) > 1:
                sheet = st.selectbox("Pilih sheet", names)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Tidak dapat membaca daftar sheet: {exc}")
    if up is not None and st.button("Muat file yang diunggah", type="primary"):
        try:
            set_dataset(read_table(up, up.name, sheet_name=sheet), up.name, "upload")
            st.success(f"Berhasil memuat {up.name}.")
        except DataLoadError as exc:
            st.error(str(exc))
        except Exception as exc:  # noqa: BLE001
            st.error(f"Terjadi kesalahan tak terduga saat memuat data: {exc}")
    st.markdown("**Atau gunakan dataset contoh (berasal dari file yang Anda berikan):**")
    c1, c2 = st.columns(2)
    for col, fname, label in ((c1, "test_unlabeled.csv", "Muat contoh: test.csv (tanpa label)"),
                              (c2, "holdout_labeled.csv", "Muat contoh: holdout berlabel")):
        p = os.path.join(SAMPLE_DIR, fname)
        if col.button(label, disabled=not os.path.exists(p), key="btn_" + fname):
            try:
                set_dataset(read_table(p), fname, "sample")
                st.rerun()
            except DataLoadError as exc:
                st.error(str(exc))
    d = dataset()
    if d is None:
        return
    df, q = d["df"], d["quality"]
    st.divider()
    st.subheader(f"Dataset aktif: {d['name']}", anchor=False)
    for n in d["notes"]:
        st.caption("ℹ " + n)
    chk = validate_for_model(df)
    if chk["ok"]:
        st.success(f"Struktur kolom cocok dengan model ({'berlabel' if chk['has_target'] else 'tanpa target'}).")
    else:
        st.warning(f"Kolom wajib model tidak lengkap: {', '.join(chk['missing_columns'])}. Modul risiko & replenishment dapat terbatas.")
    if chk["unused_columns"]:
        st.caption("Kolom tambahan yang tidak dipakai model: " + ", ".join(chk["unused_columns"]))
    m = st.columns(4)
    m[0].metric("Baris", f"{q['n_rows']:,}")
    m[1].metric("Kolom", q["n_cols"])
    m[2].metric("Sel kosong", f"{q['missing_cells']:,} ({q['missing_pct']}%)")
    m[3].metric("Baris duplikat", q["n_duplicates"])
    t1, t2, t3, t4, t5 = st.tabs(["Preview", "Statistik", "Pemeriksaan kualitas", "Tabel kolom", "Kesiapan modul"])
    with t1:
        st.dataframe(df.head(200), width="stretch", hide_index=True)
    with t2:
        st.dataframe(df.describe(include="all").T, width="stretch")
    with t3:
        iss = q["issues"].copy()
        iss["tingkat"] = iss["tingkat"].map({"ok": "✅ ok", "info": "ℹ info", "warning": "⚠ warning"})
        st.dataframe(iss, width="stretch", hide_index=True)
    with t4:
        st.dataframe(q["column_table"], width="stretch", hide_index=True)
    with t5:
        st.dataframe(module_availability(df), width="stretch", hide_index=True)
    st.download_button("⬇ Unduh laporan kualitas data (CSV)", quality_report_to_csv(q), "stockpilot_data_quality.csv", "text/csv")


# =============================================================== 3. INVENTORY OVERVIEW
def page_inventory():
    header("Inventory Overview", ["ACTUAL"], "Gambaran stok, forecast, lead time, dan in-transit. Grafik yang membutuhkan kolom tidak tersedia akan dilewati dengan penjelasan.")
    df = need_data()
    if df is None:
        return
    df, risk_msg = try_scored(df)
    work = inventory_metrics(df)
    has = lambda *c: all(x in work.columns for x in c)  # noqa: E731
    with st.expander("Filter & pencarian", expanded=True):
        f = st.columns(4)
        q = f[0].text_input("Cari SKU (mengandung)", "") if "sku" in work else ""
        mask = pd.Series(True, index=work.index)
        if q:
            mask &= work["sku"].astype(str).str.contains(q, case=False, na=False)
        if "national_inv" in work:
            lo, hi = float(work["national_inv"].min()), float(work["national_inv"].max())
            rng = f[1].slider("Rentang national_inv", lo, hi, (lo, hi)) if lo < hi else (lo, hi)
            mask &= work["national_inv"].between(*rng)
        if "lead_time" in work and work["lead_time"].notna().any():
            lo, hi = float(work["lead_time"].min()), float(work["lead_time"].max())
            rng = f[2].slider("Rentang lead_time", lo, hi, (lo, hi)) if lo < hi else (lo, hi)
            mask &= work["lead_time"].between(*rng) | work["lead_time"].isna()
        if "risk_category" in work:
            cats = f[3].multiselect("Kategori risiko", ["Tinggi", "Sedang", "Rendah"], default=["Tinggi", "Sedang", "Rendah"])
            mask &= work["risk_category"].isin(cats)
        g = st.columns(3)
        if "in_transit_qty" in work and g[0].checkbox("Hanya yang ada in-transit"):
            mask &= work["in_transit_qty"] > 0
        if "below_min_bank" in work and g[1].checkbox("Hanya stok < min_bank"):
            mask &= work["below_min_bank"]
        if "national_inv" in work and g[2].checkbox("Hanya stok negatif"):
            mask &= work["national_inv"] < 0
    w = work[mask]
    st.caption(f"Menampilkan {len(w):,} dari {len(work):,} baris.")
    if w.empty:
        st.warning("Tidak ada baris yang cocok dengan filter.")
        return
    k = st.columns(4)
    if "national_inv" in w:
        k[0].metric("Median stok nasional", f"{w['national_inv'].median():,.0f}")
    if "days_of_cover" in w:
        k[1].metric("Median days of cover", f"{w['days_of_cover'].median():,.0f}" if w["days_of_cover"].notna().any() else "n/a")
    if "in_transit_qty" in w:
        k[2].metric("SKU dengan in-transit", f"{(w['in_transit_qty'] > 0).sum():,}")
    if "below_min_bank" in w:
        k[3].metric("SKU stok < min_bank", f"{int(w['below_min_bank'].sum()):,}")
    a, b = st.columns(2)
    if "national_inv" in w:
        a.plotly_chart(viz.fig_hist(w["national_inv"], "national_inv", log_x=True), width="stretch")
    else:
        a.info("Kolom national_inv tidak tersedia.")
    if "lead_time" in w:
        b.plotly_chart(viz.fig_hist(w["lead_time"], "lead_time", color=viz.TEAL), width="stretch")
    else:
        b.info("Kolom lead_time tidak tersedia.")
    st.caption("Skala signed-log dipakai karena stok sangat berekor panjang. Satuan lead_time tidak terdokumentasi pada dataset.")
    c, d_ = st.columns(2)
    if has("national_inv", "forecast_3_month"):
        sc = w[(w["national_inv"] > 0) & (w["forecast_3_month"] > 0)]
        if len(sc):
            c.plotly_chart(viz.fig_scatter(sc.head(3000), "national_inv", "forecast_3_month", "risk_category" if "risk_category" in sc else None,
                                           ["sku"] if "sku" in sc else None, "Stok vs forecast 3 bulan (log, nilai positif)"), width="stretch")
        else:
            c.info("Tidak ada SKU dengan stok dan forecast positif pada filter ini.")
    else:
        c.info("Scatter stok vs forecast membutuhkan national_inv & forecast_3_month.")
    if has("national_inv", "in_transit_qty"):
        sc2 = w[(w["in_transit_qty"] > 0) & (w["national_inv"] > 0)]
        if len(sc2):
            d_.plotly_chart(viz.fig_scatter(sc2.head(3000), "national_inv", "in_transit_qty", "risk_category" if "risk_category" in sc2 else None,
                                            ["sku"] if "sku" in sc2 else None, "Stok vs in-transit (log)"), width="stretch")
        else:
            d_.info("Tidak ada SKU dengan in-transit dan stok positif pada filter ini.")
    st.subheader("Tabel detail", anchor=False)
    cols_default = [c_ for c_ in ["sku", "national_inv", "in_transit_qty", "forecast_3_month", "sales_1_month", "lead_time", "min_bank",
                                  "days_of_cover", "inv_position", "risk_category", "risk_probability"] if c_ in w.columns]
    shown = st.multiselect("Kolom ditampilkan", list(w.columns), default=cols_default)
    s1, s2 = st.columns([3, 1])
    sort_col = s1.selectbox("Urutkan menurut", shown or list(w.columns))
    asc = s2.checkbox("Menaik", False)
    show_table(w[shown or list(w.columns)].sort_values(sort_col, ascending=asc, na_position="last"), 420)
    if risk_msg:
        st.caption("ℹ " + risk_msg)


# =============================================================== 4. RISK MONITOR
def page_risk():
    header("Backorder Risk Monitor", ["MODEL"], "Prediksi memakai pipeline hasil Colab (tanpa melatih ulang). "
           "Bedakan: **probabilitas** (keluaran model, tidak terkalibrasi), **kelas prediksi** (≥ threshold operasi), "
           "**skor risiko** (persentil relatif dalam dataset ini), dan **kategori**.")
    df = need_data()
    if df is None:
        return
    arts, err = get_artifacts()
    if arts is None:
        st.error(err)
        return
    for w in arts["warnings"]:
        st.warning(w)
    chk = validate_for_model(df)
    if not chk["ok"]:
        st.error(f"Dataset tidak memiliki kolom yang dibutuhkan model: {', '.join(chk['missing_columns'])}. Prediksi tidak dijalankan "
                 "(StockPilot tidak menampilkan hasil pengganti).")
        return
    tiers = arts["meta"]["risk_tiers"]
    with st.expander("Pengaturan kategori risiko", expanded=False):
        st.caption(tiers["definition"])
        c = st.columns(3)
        hi = c[0].slider("Batas 'Tinggi' (p ≥)", 0.05, 0.99, float(st.session_state.get("high_thr", tiers["high_threshold"])), 0.01)
        me_default = float(min(st.session_state.get("med_thr", tiers["medium_threshold"]), hi - 0.01))
        me = c[1].slider("Batas 'Sedang' (p ≥)", 0.01, float(max(hi - 0.01, 0.02)), max(me_default, 0.01), 0.01)
        st.session_state["high_thr"], st.session_state["med_thr"] = hi, me
        if c[2].button("Kembalikan default dari Colab"):
            st.session_state.pop("high_thr", None)
            st.session_state.pop("med_thr", None)
            st.rerun()
        st.caption(f"Default Colab: Tinggi ≥ {tiers['high_threshold']:.3f}, Sedang ≥ {tiers['medium_threshold']:.3f}.")
    try:
        with st.spinner("Menghitung prediksi..."):
            scored = get_scored(df)
    except ArtifactError as exc:
        st.error(str(exc))
        return
    n = len(scored)
    vc = scored["risk_category"].value_counts()
    k = st.columns(4)
    k[0].markdown(viz.kpi_card("SKU dianalisis", f"{n:,}"), unsafe_allow_html=True)
    k[1].markdown(viz.kpi_card("Risiko Tinggi", f"{vc.get('Tinggi', 0):,}", f"{100 * vc.get('Tinggi', 0) / n:.1f}%", viz.RED), unsafe_allow_html=True)
    k[2].markdown(viz.kpi_card("Risiko Sedang", f"{vc.get('Sedang', 0):,}", f"{100 * vc.get('Sedang', 0) / n:.1f}%", viz.AMBER), unsafe_allow_html=True)
    k[3].markdown(viz.kpi_card("Kelas prediksi = 1", f"{int(scored['predicted_class'].sum()):,}", f"p ≥ {arts['meta']['operating_threshold']:.3f}", viz.NAVY), unsafe_allow_html=True)
    st.write("")
    a, b = st.columns(2)
    a.plotly_chart(viz.fig_risk_distribution(scored["risk_category"]), width="stretch")
    b.plotly_chart(viz.fig_prob_hist(scored["risk_probability"], hi, me), width="stretch")
    if TARGET in scored.columns and scored[TARGET].isin([0, 1]).all():
        st.markdown(viz.status_badge("ACTUAL") + " Dataset ini **berlabel**: perbandingan dengan kejadian aktual.", unsafe_allow_html=True)
        t = (scored.groupby("risk_category")[TARGET].agg(n_SKU="count", backorder_aktual="sum", backorder_rate="mean")
             .reindex(["Tinggi", "Sedang", "Rendah"]).fillna(0))
        st.dataframe(t.style.format({"backorder_rate": "{:.1%}"}), width="stretch")
        st.caption("Catatan: bila dataset ini termasuk data yang dipakai melatih model, angka ini TIDAK independen. "
                   "Metrik resmi ada di halaman Model Performance (holdout Colab).")
        st.plotly_chart(viz.fig_capture(capture_curve(scored[TARGET].values, scored["risk_probability"].values)), width="stretch")
    else:
        st.caption("Dataset tanpa label target: tidak ada metrik evaluasi yang dihitung (sesuai prinsip tidak mengevaluasi data tanpa label).")
    st.subheader("Ranking & tabel hasil", anchor=False)
    f = st.columns(3)
    cats = f[0].multiselect("Filter kategori", ["Tinggi", "Sedang", "Rendah"], default=["Tinggi", "Sedang"])
    topn = f[1].number_input("Tampilkan N teratas", 10, max(n, 10), min(200, max(n, 10)), 10)
    qsku = f[2].text_input("Cari SKU", "") if "sku" in scored else ""
    view = scored[scored["risk_category"].isin(cats)]
    if qsku:
        view = view[view["sku"].astype(str).str.contains(qsku, na=False)]
    view = view.sort_values("risk_probability", ascending=False).head(int(topn))
    front = [c_ for c_ in ["sku", "risk_probability", "risk_score", "risk_category", "predicted_class"] if c_ in view.columns]
    show_table(view[front + [c_ for c_ in view.columns if c_ not in front]], 440)
    st.download_button("⬇ Unduh seluruh prediksi (CSV)", csv_bytes(scored.sort_values("risk_probability", ascending=False)),
                       "stockpilot_risk_predictions.csv", "text/csv", type="primary")


# =============================================================== 5. REPLENISHMENT
def page_replen():
    header("Replenishment Recommendation", ["RULE", "MODEL"], "Rekomendasi pemeriksaan/replenishment **berbasis aturan** (bukan kuantitas order optimal). "
           "Parameter di bawah adalah ASUMSI yang dapat Anda ubah.")
    df = need_data(REPLEN_NEEDS)
    if df is None:
        return
    base = replen_params_from_state()
    with st.expander("Parameter replenishment (asumsi)", expanded=True):
        c = st.columns(4)
        unit = c[0].selectbox("Satuan lead_time", list(LEAD_TIME_FACTORS), index=list(LEAD_TIME_FACTORS).index(base.lead_time_unit),
                              help="Satuan tidak terdokumentasi pada dataset; default diasumsikan minggu.")
        fb = c[1].number_input("Lead time pengganti bila kosong", 0.0, 520.0, float(base.fallback_lead_time), 1.0)
        sp = c[2].slider("Safety stock (% dari demand selama lead time)", 0, 200, int(base.safety_pct * 100), 5) / 100
        hz = c[3].number_input("Horizon cakupan tambahan (bulan)", 0.0, 12.0, float(base.review_horizon_months), 0.5)
        c = st.columns(4)
        ub = c[0].checkbox("Safety stock minimal = min_bank", base.use_min_bank, help="Makna min_bank tidak terdokumentasi; dianggap level stok minimum.")
        moq = c[1].number_input("MOQ (0 = tidak dipakai)", 0.0, 1e7, float(base.moq), 1.0)
        rt = c[2].number_input("Kelipatan pembulatan", 1.0, 1e6, float(base.round_to), 1.0)
        params = ReplenParams(lead_time_unit=unit, fallback_lead_time=fb, safety_pct=sp, use_min_bank=ub, review_horizon_months=hz, moq=moq, round_to=rt)
        st.session_state["replen_params"] = params
        st.code(REPLEN_FORMULA, language="text")
    scored, msg = try_scored(df)
    if msg:
        st.info("Risiko model tidak dipakai: " + msg + " Rekomendasi hanya mengikuti aturan stok.")
    plan = replenishment_plan(scored, params)
    st.session_state["replen_plan"] = plan
    k = st.columns(4)
    k[0].markdown(viz.kpi_card("SKU di bawah ROP", f"{int(plan['below_reorder_point'].sum()):,}", f"dari {len(plan):,}", viz.AMBER), unsafe_allow_html=True)
    k[1].markdown(viz.kpi_card("P1 – periksa & pesan", f"{int(plan['action'].str.startswith('P1').sum()):,}", "risiko tinggi & < ROP", viz.RED), unsafe_allow_html=True)
    k[2].markdown(viz.kpi_card("Total qty indikatif", f"{plan['suggested_order_qty'].sum():,.0f}", "unit (indikatif, bukan optimal)", viz.NAVY), unsafe_allow_html=True)
    k[3].markdown(viz.kpi_card("Lead time diasumsikan", f"{int(plan['lead_time_assumed'].sum()):,}", "SKU tanpa lead_time", viz.GREY), unsafe_allow_html=True)
    st.write("")
    f = st.columns(3)
    acts = sorted(plan["action"].unique())
    sel = f[0].multiselect("Filter tindakan", acts, default=[a for a in acts if a != "Tidak ada tindakan"])
    qsku = f[1].text_input("Cari SKU", "") if "sku" in plan else ""
    topn = f[2].number_input("Maks. baris", 10, max(len(plan), 10), min(300, max(len(plan), 10)), 10)
    view = plan[plan["action"].isin(sel)]
    if qsku:
        view = view[view["sku"].astype(str).str.contains(qsku, na=False)]
    if "risk_probability" in view:
        view = view.assign(_k=view["action"].str[:2]).sort_values(["_k", "risk_probability"], ascending=[True, False]).drop(columns="_k")
    cols = [c_ for c_ in ["sku", "action", "risk_category", "risk_probability", "national_inv", "inventory_position", "reorder_point",
                          "shortfall_to_rop", "suggested_order_qty", "lead_time_used", "reason"] if c_ in view.columns]
    show_table(view[cols].head(int(topn)), 460)
    st.markdown('<div class="sp-panel"><b>Cara membaca:</b> P1 = model menilai risiko tinggi <i>dan</i> posisi stok di bawah reorder point; '
                'P2 = risiko tinggi tetapi stok tampak cukup → periksa pemasok/akurasi data; P3 = hanya aturan stok yang memicu; '
                'P4 = risiko sedang → pantau. Kuantitas adalah <b>indikatif</b>: dataset tidak memuat biaya pemesanan, kebijakan pemasok, atau variabilitas demand.</div>',
                unsafe_allow_html=True)
    st.download_button("⬇ Unduh rekomendasi replenishment (CSV)", csv_bytes(plan), "stockpilot_replenishment.csv", "text/csv", type="primary")


# =============================================================== 6. SLOTTING
def _template_csv(kind: str) -> bytes:
    if kind == "sku":
        return csv_bytes(pd.DataFrame({"sku": ["SKU-A", "SKU-B"], "current_location": ["A01-B01-L1", "A02-B05-L2"], "required_volume": [0.3, 0.8],
                                       "pick_frequency": [120, 8], "storage_class": ["STANDARD", "HEAVY"]}))
    return csv_bytes(pd.DataFrame({"location_id": ["A01-B01-L1", "A02-B05-L2"], "distance_to_dock": [5.0, 21.0], "capacity_volume": [0.5, 1.2],
                                   "allowed_storage_class": ["ANY", "STANDARD"], "level": [1, 2]}))


def page_slotting():
    header("Smart Slotting Simulator", ["SIM"], "Dataset backorder tidak memuat lokasi rak, kapasitas, atau frekuensi picking. Gunakan **file lokasi terpisah** "
           "atau demo **data sintetis (SIMULASI)**. Metode: heuristik COI yang dapat dijelaskan — bukan optimasi matematis global.")
    mode = st.radio("Sumber data slotting", ["Demo data sintetis (SIMULASI)", "Unggah data gudang saya"], horizontal=True)
    synthetic = False
    if mode.startswith("Demo"):
        d0 = dataset()
        can_real = d0 is not None and "sku" in d0["df"].columns
        c = st.columns(4)
        n = c[0].slider("Jumlah SKU", 50, 300, 200, 10)
        seed = c[1].number_input("Seed acak", 0, 9999, 42)
        aisles = c[2].slider("Jumlah lorong", 8, 14, 8)
        use_real = c[3].checkbox("Pakai ID SKU dari dataset (atribut lain tetap sintetis)", False, disabled=not can_real)
        ids = d0["df"]["sku"].astype(str).head(n).tolist() if (use_real and can_real) else None
        sku_df, loc_df = generate_synthetic_slotting(n, aisles, 15, 3, int(seed), ids)
        synthetic = True
        st.warning("🧪 **SIMULASI** — seluruh lokasi, volume, frekuensi picking, dan kondisi awal (penempatan acak) adalah SINTETIS.")
    else:
        c = st.columns(2)
        c[0].download_button("⬇ Template file SKU", _template_csv("sku"), "template_slotting_sku.csv", "text/csv")
        c[1].download_button("⬇ Template file lokasi", _template_csv("loc"), "template_slotting_locations.csv", "text/csv")
        st.caption(f"File SKU wajib: {', '.join(SLOT_SKU_COLS)} (opsional: storage_class). "
                   f"File lokasi wajib: {', '.join(SLOT_LOC_COLS)} + distance_to_dock ATAU x,y (opsional: allowed_storage_class, level).")
        u1 = st.file_uploader("File SKU (CSV/XLSX)", type=["csv", "xlsx", "xls"], key="slot_sku")
        u2 = st.file_uploader("File lokasi (CSV/XLSX)", type=["csv", "xlsx", "xls"], key="slot_loc")
        if u1 is None or u2 is None:
            st.info("Unggah kedua file untuk menjalankan rekomendasi, atau gunakan demo sintetis.")
            return
        try:
            sku_df, loc_df = read_table(u1, u1.name), read_table(u2, u2.name)
        except DataLoadError as exc:
            st.error(str(exc))
            return
    chk = validate_slotting_inputs(sku_df, loc_df)
    if not chk["ok"]:
        for e in chk["errors"]:
            st.error(e)
        return
    for w in chk["warnings"]:
        st.warning(w)
    try:
        res = recommend_slotting(sku_df, loc_df)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Gagal menjalankan slotting: {exc}")
        return
    st.session_state["slot_result"] = {"res": res, "synthetic": synthetic}
    S, T = res["summary"], res["table"]
    k = st.columns(4)
    k[0].markdown(viz.kpi_card("Jarak berbobot awal", f"{S['baseline_weighted_distance']:,.0f}", "frekuensi × jarak ke dock", viz.GREY), unsafe_allow_html=True)
    k[1].markdown(viz.kpi_card("Jarak berbobot rekomendasi", f"{S['recommended_weighted_distance']:,.0f}", "heuristik COI", viz.TEAL), unsafe_allow_html=True)
    k[2].markdown(viz.kpi_card("Perubahan", f"{S['pct_change']:.1f}%", f"Δ {S['abs_change']:,.0f}", viz.NAVY), unsafe_allow_html=True)
    k[3].markdown(viz.kpi_card("SKU dipindahkan", f"{S['n_moves']:,}", f"tidak tertempatkan: {S['n_unassigned']}", viz.AMBER), unsafe_allow_html=True)
    if S["baseline_overcapacity_skus"] or S["baseline_shared_locations"]:
        st.warning(f"Kondisi awal melanggar batasan: {S['baseline_overcapacity_skus']} SKU melebihi kapasitas, {S['baseline_shared_locations']} lokasi dipakai >1 SKU.")
    if res["unassigned"]:
        st.error(f"{len(res['unassigned'])} SKU tidak dapat ditempatkan (kapasitas/kompatibilitas): {', '.join(map(str, res['unassigned'][:15]))}")
    a, b = st.columns(2)
    a.plotly_chart(viz.fig_bar_compare(["Jarak berbobot picking"], [S["baseline_weighted_distance"]], [S["recommended_weighted_distance"]],
                                       "Kondisi awal vs rekomendasi" + (" (SIMULASI)" if synthetic else "")), width="stretch")
    cls = T.groupby("abc_class")[["weighted_distance_current", "weighted_distance_recommended"]].sum().reset_index()
    b.plotly_chart(viz.fig_bar_compare(cls["abc_class"].tolist(), cls["weighted_distance_current"].tolist(), cls["weighted_distance_recommended"].tolist(),
                                       "Jarak berbobot per kelas ABC (A = 80% picking)"), width="stretch")
    if {"x", "y"} <= set(loc_df.columns):
        m1, m2 = st.columns(2)
        f1, f2 = viz.fig_slotting_layout(T, loc_df, "current"), viz.fig_slotting_layout(T, loc_df, "recommended")
        if f1:
            m1.plotly_chart(f1, width="stretch")
        if f2:
            m2.plotly_chart(f2, width="stretch")
    st.subheader("Daftar pemindahan", anchor=False)
    cols = ["sku", "abc_class", "pick_frequency", "required_volume", "current_location", "recommended_location", "current_distance",
            "recommended_distance", "distance_saved_per_pick", "moved"]
    moves = T[T["moved"]][[c_ for c_ in cols if c_ in T.columns]].sort_values("distance_saved_per_pick", ascending=False)
    st.dataframe(moves, width="stretch", hide_index=True, height=360)
    with st.expander("Metode, asumsi, dan batasan"):
        st.markdown("**Metode:** " + S["method"] + ". SKU diurutkan menurut *Cube-per-Order Index* (volume ÷ frekuensi picking); "
                    "setiap SKU ditempatkan di lokasi terdekat dock yang kapasitasnya cukup dan kompatibel (SKU HEAVY hanya di lokasi `ANY`/lantai).")
        for a_ in SLOTTING_ASSUMPTIONS:
            st.markdown("- " + a_)
    out = T.copy()
    if synthetic:
        out["data_source"] = "SYNTHETIC"
    st.download_button("⬇ Unduh rekomendasi slotting (CSV)", csv_bytes(out), "stockpilot_slotting" + ("_SYNTHETIC" if synthetic else "") + ".csv", "text/csv", type="primary")


# =============================================================== 7. WHAT-IF
def page_whatif():
    header("What-if & Financial Simulator", ["SIM", "RULE"], "Bandingkan baseline vs skenario pada aturan replenishment yang sama. Biaya adalah **input Anda** "
           "(default hanya ILUSTRASI, bukan biaya aktual).")
    df = need_data(REPLEN_NEEDS)
    if df is None:
        return
    scored, msg = try_scored(df)
    base = replen_params_from_state()
    st.markdown("**Baseline** memakai parameter dari halaman Replenishment (atau default). Atur skenario:")
    c = st.columns(4)
    dm = c[0].slider("Perubahan demand (%)", -50, 100, 0, 5)
    lm = c[1].slider("Perubahan lead time (%)", -50, 100, 0, 5)
    sp = c[2].slider("Safety stock (% LTD)", 0, 200, int(base.safety_pct * 100), 5)
    hz = c[3].slider("Horizon (bulan)", 0.0, 6.0, float(base.review_horizon_months), 0.5)
    scen = ReplenParams(**{**base.__dict__, "demand_multiplier": 1 + dm / 100, "lead_time_multiplier": 1 + lm / 100,
                           "safety_pct": sp / 100, "review_horizon_months": hz})
    st.markdown("**Asumsi biaya** " + viz.badge("ILUSTRATIF — ganti dengan biaya Anda", viz.AMBER), unsafe_allow_html=True)
    c = st.columns(4)
    costs = {"unit_cost": c[0].number_input("Biaya per unit", 0.0, 1e9, 10.0), "order_cost": c[1].number_input("Biaya per pemesanan", 0.0, 1e9, 25.0),
             "holding_cost_per_unit_month": c[2].number_input("Biaya simpan /unit/bulan", 0.0, 1e9, 0.2),
             "backorder_event_cost": c[3].number_input("Biaya per kejadian backorder (risiko tinggi)", 0.0, 1e9, 100.0), "horizon_months": hz}
    diff = params_diff(base, scen)
    if diff.empty:
        st.info("Skenario identik dengan baseline — ubah salah satu parameter.")
        return
    wi = whatif_replenishment(scored, base, scen, costs)
    st.session_state["whatif_result"] = {"table": wi, "diff": diff, "costs": costs}
    st.subheader("Parameter yang diubah", anchor=False)
    st.dataframe(diff, hide_index=True, width="stretch")
    st.subheader("Hasil perbandingan", anchor=False)
    st.dataframe(wi.style.format({"Baseline": "{:,.1f}", "Skenario": "{:,.1f}", "Perubahan absolut": "{:+,.1f}", "Perubahan %": "{:+.1f}%"}, na_rep="—"),
                 hide_index=True, width="stretch")
    sub = wi[wi["Indikator"].isin(["SKU di bawah ROP", "Total kuantitas order indikatif (unit)", "Jumlah baris order"])]
    st.plotly_chart(viz.fig_bar_compare(sub["Indikator"].tolist(), sub["Baseline"].tolist(), sub["Skenario"].tolist(),
                                        "Baseline vs skenario (hitungan dari data)"), width="stretch")
    if "risk_category" not in scored.columns:
        st.caption("ℹ Indikator berbasis risiko model bernilai 0 karena risiko tidak tersedia (" + (msg or "") + ").")
    with st.expander("Asumsi & keterbatasan", expanded=True):
        st.markdown("- Perhitungan bersifat mekanistis atas aturan ROP; **bukan** prediksi dampak nyata pada backorder.\n"
                    "- Baris bertanda *(ILUSTRATIF)* memakai biaya asumsi — jangan disajikan sebagai penghematan aktual.\n"
                    "- Tidak memodelkan dinamika demand, perilaku pemasok, MOQ per pemasok, kapasitas gudang, atau efek skala.\n"
                    "- Perubahan demand/lead time diterapkan seragam ke seluruh SKU.")
    st.download_button("⬇ Unduh ringkasan simulasi (CSV)", csv_bytes(wi), "stockpilot_whatif_summary.csv", "text/csv", type="primary")


# =============================================================== 8. MODEL
def page_model():
    header("Model Performance & Explainability", ["MODEL", "ACTUAL"], "Semua metrik dibaca dari artefak hasil Colab (holdout berlabel). "
           "Aplikasi ini tidak menghitung metrik pada data tanpa label.")
    arts, err = get_artifacts()
    if arts is None:
        st.error(err)
        return
    for w in arts["warnings"]:
        st.warning(w)
    m = arts["meta"]
    h, cv = m["metrics"]["holdout"], m["metrics"]["cv"]
    st.markdown(f"**Model terpilih:** {m['model_name']} · dilatih {m['trained_at_utc']} · random_state {m['random_state']}")
    st.caption(m["selection_rule"])
    k = st.columns(5)
    for col, (lbl, val, sub) in zip(k, [("PR-AUC (holdout)", f"{h['average_precision']:.3f}", f"CV {cv['average_precision_mean']:.3f}±{cv['average_precision_std']:.3f}"),
                                         ("ROC-AUC (holdout)", f"{h['roc_auc']:.3f}", f"CV {cv['roc_auc_mean']:.3f}"),
                                         ("Precision", f"{h['precision']:.3f}", f"@ {h['threshold']:.3f}"), ("Recall", f"{h['recall']:.3f}", ""),
                                         ("F1", f"{h['f1']:.3f}", f"prevalensi {h['prevalence']:.1%}")]):
        col.markdown(viz.kpi_card(lbl, val, sub, viz.TEAL), unsafe_allow_html=True)
    st.write("")
    cm = m["metrics"]["confusion_matrix_holdout"]
    a, b = st.columns(2)
    a.plotly_chart(viz.fig_confusion(cm["TN"], cm["FP"], cm["FN"], cm["TP"]), width="stretch")
    pr = read_output_csv("pr_curve.csv")
    if pr is not None:
        b.plotly_chart(viz.fig_pr(pr, h["prevalence"], (h["recall"], h["precision"])), width="stretch")
    st.markdown(f"**Implikasi:** FP = {cm['FP']:,} SKU diperiksa tanpa perlu; FN = {cm['FN']:,} backorder terlewat. "
                "Turunkan threshold untuk menangkap lebih banyak backorder dengan konsekuensi lebih banyak pemeriksaan.")
    ev = read_output_csv("model_evaluation.csv")
    if ev is not None:
        st.subheader("Perbandingan model (CV dev & holdout)", anchor=False)
        cols = [c_ for c_ in ["model", "CV_AP_mean", "CV_AP_std", "CV_ROC_AUC_mean", "threshold_OOF", "HOLD_AP", "HOLD_ROC_AUC", "HOLD_precision", "HOLD_recall", "HOLD_f1"] if c_ in ev]
        st.dataframe(ev[cols].style.format({c_: "{:.3f}" for c_ in cols if c_ != "model"}), hide_index=True, width="stretch")
        st.caption("Pemilihan model berdasarkan CV PR-AUC pada data dev; holdout hanya untuk pelaporan.")
    th = read_output_csv("threshold_analysis.csv")
    if th is not None:
        c1, c2 = st.columns(2)
        c1.plotly_chart(viz.fig_threshold(th, h["threshold"]), width="stretch")
        capd = read_output_csv("capture_curve.csv")
        if capd is not None:
            c2.plotly_chart(viz.fig_capture(capd), width="stretch")
    imp = read_output_csv("feature_importance.csv")
    if imp is not None:
        st.plotly_chart(viz.fig_importance(imp), width="stretch")
        st.caption("Permutation importance mengukur kontribusi terhadap performa model, **bukan** bukti sebab-akibat; fitur berkorelasi saling membagi kontribusi.")
    ab = read_output_csv("ablation.csv")
    if ab is not None:
        with st.expander("Uji sensitivitas leakage (ablasi)"):
            st.dataframe(ab, hide_index=True, width="stretch")
    tv = read_output_csv("risk_tier_validation_holdout.csv")
    if tv is not None:
        with st.expander("Validasi kategori risiko pada holdout"):
            st.dataframe(tv, hide_index=True, width="stretch")
    with st.expander("Definisi & batasan model"):
        for k_, v in m["definitions"].items():
            st.markdown(f"- **{k_}**: {v}")
        for lim in m["limitations"]:
            st.markdown(f"- ⚠ {lim}")
        st.json(m["environment"])


# =============================================================== 9. REPORTS
def page_reports():
    header("Reports & Export", ["ACTUAL", "MODEL", "RULE", "SIM"], "Unduh hasil analisis. Setiap file mempertahankan label jenis hasilnya.")
    d = dataset()
    sheets = {}
    st.subheader("Unduhan individual", anchor=False)
    if d:
        q = d["quality"]
        st.download_button("⬇ Laporan kualitas data (CSV)", quality_report_to_csv(q), "stockpilot_data_quality.csv", "text/csv")
        sheets["Data_Quality_Issues"] = q["issues"]
        scored, msg = try_scored(d["df"])
        if "risk_category" in scored.columns:
            pred = scored.sort_values("risk_probability", ascending=False)
            st.download_button("⬇ Prediksi risiko (CSV)", csv_bytes(pred), "stockpilot_risk_predictions.csv", "text/csv")
            sheets["Risk_Predictions"] = pred
        else:
            st.caption("Prediksi risiko tidak tersedia: " + (msg or ""))
        if all(x in scored.columns for x in REPLEN_NEEDS):
            plan = replenishment_plan(scored, replen_params_from_state())
            st.download_button("⬇ Rekomendasi replenishment (CSV, berbasis aturan)", csv_bytes(plan), "stockpilot_replenishment.csv", "text/csv")
            sheets["Replenishment_RULE"] = plan
        else:
            st.caption("Replenishment tidak tersedia: kolom tidak lengkap.")
    else:
        st.info("Muat dataset untuk mengaktifkan laporan berbasis data.")
    sr = st.session_state.get("slot_result")
    if sr:
        T = sr["res"]["table"].copy()
        if sr["synthetic"]:
            T["data_source"] = "SYNTHETIC"
        st.download_button("⬇ Rekomendasi slotting" + (" (SIMULASI sintetis)" if sr["synthetic"] else ""), csv_bytes(T), "stockpilot_slotting.csv", "text/csv")
        sheets["Slotting_SIM" if sr["synthetic"] else "Slotting"] = T
    else:
        st.caption("Rekomendasi slotting belum dibuat (jalankan halaman Smart Slotting Simulator).")
    wr = st.session_state.get("whatif_result")
    if wr:
        st.download_button("⬇ Ringkasan simulasi what-if (CSV)", csv_bytes(wr["table"]), "stockpilot_whatif_summary.csv", "text/csv")
        sheets["WhatIf_SIM"] = wr["table"]
    else:
        st.caption("Ringkasan what-if belum dibuat (jalankan halaman What-if).")
    ev = read_output_csv("model_evaluation.csv")
    if ev is not None:
        st.download_button("⬇ Laporan evaluasi model (CSV, dari Colab)", csv_bytes(ev), "stockpilot_model_evaluation.csv", "text/csv")
        sheets["Model_Evaluation"] = ev
    else:
        st.caption("Laporan evaluasi model tidak ditemukan di folder outputs/.")
    st.divider()
    if sheets:
        st.download_button("⬇ Semua laporan dalam satu Excel (.xlsx)", to_excel_bytes(sheets), "stockpilot_report.xlsx",
                           "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", type="primary")
    st.caption("Catatan: sheet bertanda _SIM berasal dari simulasi; _RULE berasal dari aturan dengan asumsi parameter.")


# =============================================================== main
def main():
    page = sidebar()
    {PAGES[0]: page_home, PAGES[1]: page_upload, PAGES[2]: page_inventory, PAGES[3]: page_risk, PAGES[4]: page_replen,
     PAGES[5]: page_slotting, PAGES[6]: page_whatif, PAGES[7]: page_model, PAGES[8]: page_reports}[page]()


main()
