﻿# Proyek-Akhir_Belajar-Analisis-Data-Dengan-Python
# Dashboard Analisis Polusi Udara di Beijing

Proyek ini adalah sebuah **dashboard interaktif** yang dibuat menggunakan **Streamlit** untuk menganalisis data polusi udara di Beijing. Dashboard ini memberikan wawasan tentang pola polusi berdasarkan waktu, hubungan antara polusi dan variabel meteorologi, serta pengelompokan tingkat polusi.

## 📁 Dataset
Dataset yang digunakan adalah **PRSA Data** dengan informasi kualitas udara, termasuk PM2.5, PM10, dan beberapa variabel meteorologi (suhu, kelembapan, tekanan, dll.) dari tahun **2013 hingga 2017**.

---

## 🎯 Fitur Analisis

1. **Kapan Polusi Udara Paling Tinggi?**
   - Visualisasi menggunakan **boxplot** untuk menunjukkan distribusi **PM2.5** berdasarkan musim dan jam.
   
2. **Hubungan Variabel Meteorologi dengan Polusi**
   - Korelasi antara variabel cuaca dan tingkat polusi menggunakan **heatmap** dan scatter plot suhu vs PM2.5.
   
3. **Clustering Berdasarkan Tingkat Polusi**
   - **Clustering manual** berdasarkan rentang nilai PM2.5 dan visualisasi distribusi tingkat polusi menggunakan pie chart dan scatter plot.

---

## 📊 Visualisasi Dashboard

- **Boxplot**: Distribusi polusi berdasarkan musim dan jam dalam sehari.
- **Heatmap**: Korelasi variabel meteorologi dengan polusi udara.
- **Scatter Plot**: Hubungan antara suhu dan konsentrasi PM2.5.
- **Pie Chart**: Distribusi tingkat polusi berdasarkan rentang PM2.5.

---

## 🚀 Cara Menjalankan Aplikasi

Pastikan Anda sudah memiliki **Python** dan **Streamlit** terinstall. Jika belum, Anda bisa menginstalnya dengan menjalankan perintah berikut:

```bash
pip install streamlit pandas seaborn matplotlib

Jalankan dashboard dengan perintah berikut:

    ```bash
   streamlit run dashboard.py
