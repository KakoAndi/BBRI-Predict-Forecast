# BBRI.JK Stock Price Forecasting — Enterprise ML Model

Prediksi harga saham PT Bank Rakyat Indonesia (BBRI.JK) 12 bulan ke depan menggunakan pendekatan ensemble machine learning.

## Model yang Digunakan
- Facebook Prophet (trend + seasonality + Indonesian holidays)
- SARIMA(2,1,2)(1,1,1,12) — classical statistical model
- XGBoost — gradient boosting dengan 15 engineered features (iterative forecast)
- LSTM — deep learning 3-layer dengan BatchNormalization & EarlyStopping
- Ensemble Weighted Average — bobot proporsional terhadap 1/MAPE validation

## Dataset
- Sumber: Yahoo Finance (BBRI.JK)
- Periode: Agustus 2020 — Agustus 2025 (1.201 hari trading)
- Frekuensi input: harian → di-resample bulanan
- Target: prediksi harga penutupan bulanan Sep 2025 — Aug 2026

## Fitur Utama
- Ensemble model dengan confidence interval 68% dan 90%
- 15 feature engineering: MA3/6/12, RSI, volatility rolling, lag, volume ratio, dll
- Validasi time-series (walk-forward, bukan random split)
- 4 visualisasi enterprise: EDA, forecast utama, model dashboard, heatmap kalender
- Interactive chart via Plotly (HTML)
- Auto-export hasil ke CSV

## Output
- BBRI_EDA.png — exploratory data analysis
- BBRI_Forecast_Main.png — chart prediksi utama
- BBRI_Model_Dashboard.png — perbandingan semua model
- BBRI_Forecast_Heatmap.png — heatmap kalender 12 bulan
- BBRI_Interactive_Forecast.html — chart interaktif
- BBRI_Forecast_Results.csv — tabel hasil numerik

## Cara Menjalankan
1. Buka BBRI_Enterprise_Forecast.ipynb di Google Colab
2. Jalankan sel install (sel pertama)
3. Upload file BBRI_JK_5y_1d.csv saat diminta
4. Run All — selesai dalam ~5-10 menit
5. Output otomatis ter-download

## Estimasi Akurasi (MAPE, Validation 12 Bulan)
- Prophet  : ~8–14%
- SARIMA   : ~10–16%
- XGBoost  : ~7–12%
- LSTM     : ~9–15%
- Ensemble : ~6–11%

## Requirements
prophet, xgboost, scikit-learn, tensorflow, statsmodels, plotly, pandas, numpy, matplotlib, seaborn, python-dateutil

## Disclaimer
Model ini dibuat untuk keperluan edukasi dan riset. Bukan merupakan rekomendasi investasi. Selalu konsultasikan keputusan investasi dengan penasihat keuangan berlisensi.
