# StockPilot — Smart Inventory & Material Flow Optimization (INVOSTIK 2026)

Prototipe sistem pendukung keputusan gudang: kualitas data → prediksi risiko backorder (ML) → prioritas pemeriksaan →
replenishment berbasis aturan → simulasi smart slotting & what-if. Masukan data: **CSV/XLSX** (belum ada integrasi real-time IoT/RFID/ERP/WMS).

## Menjalankan
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt                         # butuh Python >= 3.11 untuk scikit-learn 1.8.0
streamlit run app.py
```
Model bawaan dilatih dengan **scikit-learn 1.8.0**. Jika Anda menjalankan ulang notebook di Colab, ganti `models/` dan `outputs/` dengan isi ZIP dari Colab,
lalu samakan versi: `pip install scikit-learn==<versi di models/model_metadata.json -> environment>`. Aplikasi menampilkan peringatan bila versi berbeda.

## Struktur
```
app.py                    # 9 halaman
stockpilot_features.py    # feature engineering bersama (WAJIB di root; direferensikan oleh model.pkl)
utils/ data_loader.py analytics.py simulation.py visualization.py
models/                   # model.pkl, feature_columns.pkl, model_metadata.json  (dari Colab)
outputs/                  # laporan evaluasi dari Colab
sample_data/              # test_unlabeled.csv, holdout_labeled.csv, data slotting SINTETIS
```

## Demo cepat (±5 menit)
1. **Data Upload & Data Quality** → klik *Muat contoh: test.csv* (atau unggah file Anda; urutan kolom bebas).
2. **Home** → KPI, distribusi risiko, ringkasan rekomendasi.
3. **Backorder Risk Monitor** → ubah batas kategori, filter, unduh CSV.
4. **Replenishment** → ubah asumsi (satuan lead time, safety stock, MOQ) dan lihat alasan tiap rekomendasi.
5. **Smart Slotting** → demo sintetis (berlabel SIMULASI) atau unggah file lokasi sendiri (ada template).
6. **What-if** → ubah demand/lead time; biaya adalah input Anda.
7. **Model Performance** → metrik holdout dari Colab, threshold, feature importance.
8. **Reports & Export** → CSV per modul atau satu Excel.

## Status modul (jujur untuk juri)
| Modul | Status |
|---|---|
| Data quality, Inventory Overview | Berfungsi — DATA AKTUAL |
| Risk Monitor, Model Performance | Berfungsi — PREDIKSI MODEL (artefak Colab; probabilitas tidak terkalibrasi) |
| Replenishment | Berbasis ATURAN + asumsi eksplisit; bukan kuantitas optimal |
| Smart Slotting | SIMULASI (heuristik COI; data sintetis atau file lokasi unggahan) |
| What-if & finansial | SIMULASI (biaya = input pengguna; default hanya ilustrasi) |
| IoT / RFID / barcode / ERP / WMS | RENCANA — belum diimplementasikan |

## Batasan
- Dataset tidak mendokumentasikan satuan `lead_time`, makna `min_bank`, dan jendela waktu forecast/sales → diasumsikan eksplisit (dapat diubah di UI).
- Waktu pencatatan fitur terhadap label tidak diketahui; risiko *leakage* temporal belum dapat dikesampingkan.
- Validasi hanya pada satu dataset historis; belum diuji di gudang nyata.
- Tidak melatih ulang di aplikasi; tidak menghitung metrik pada data tanpa label.

## Pengujian yang telah dijalankan
Streamlit `AppTest` pada 9 halaman × (tanpa data, data tanpa label, data berlabel) tanpa exception; CSV, XLSX dengan kolom diacak, header berbeda,
kolom kurang, file kosong/format salah, missing values, dan model tidak tersedia (pesan error, tanpa prediksi pengganti).
Belum diuji: peramban sungguhan/unggah via UI manual, dataset > 100 ribu baris (replenishment memakai loop per baris), Python selain 3.12.
