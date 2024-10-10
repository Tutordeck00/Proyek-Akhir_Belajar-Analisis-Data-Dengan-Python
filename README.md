# Proyek-Akhir_Belajar-Analisis-Data-Dengan-Python
# Dashboard Kualitas Udara di Beijing

Dashboard ini merupakan aplikasi berbasis Streamlit yang menampilkan hasil analisis data kualitas udara di Beijing. Dashboard ini memungkinkan pengguna untuk menjelajahi tren musiman, melakukan clustering berdasarkan tingkat polusi, serta melihat prediksi kualitas udara di masa depan.

## Fitur Dashboard
1. **Visualisasi Pola Musiman**: Menampilkan tren PM2.5 berdasarkan rata-rata bulanan dan moving average 12 bulan.
2. **Clustering Polusi Udara**: Mengelompokkan hari-hari berdasarkan tingkat polusi udara menggunakan metode K-Means.
3. **Forecasting PM2.5**: Prediksi polusi udara (PM2.5) di masa depan menggunakan model ARIMA.
4. **Tabel Data**: Menampilkan beberapa baris data kualitas udara yang bersih.

## Prasyarat
Pastikan Anda telah menginstal Python versi 3.x dan `pip`.

## Cara Menginstal

1. **Clone repositori** ini atau unduh berkas `dashboard.py` dan `requirements.txt`.

   ```bash
   git clone <repository-url>
   cd <repository-directory>

2. Instal library yang dibutuhkan menggunakan requirements.txt:.

   ```bash
   pip install -r requirements.txt

3. Jalankan dashboard dengan perintah berikut:

    ```bash
   streamlit run dashboard.py
