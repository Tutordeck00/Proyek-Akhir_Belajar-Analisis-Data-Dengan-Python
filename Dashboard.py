import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
DATA_PATH = "D:\Bangkit-Machine Learning 2024\Proyek Akhir_Belajar Analisis Data Dengan Python\PRSA_Data_Shunyi_20130301-20170228.csv"
data = pd.read_csv(DATA_PATH, na_values='NA', infer_datetime_format=True)

# Combine the year, month, day, and hour columns into a single datetime column
data['datetime'] = pd.to_datetime(
    data[['year', 'month', 'day', 'hour']]
)

# Add useful columns for analysis
data['month'] = data['datetime'].dt.month
data['hour'] = data['datetime'].dt.hour
data['season'] = data['month'].apply(lambda x: 
                                     'Winter' if x in [12, 1, 2] 
                                     else 'Spring' if x in [3, 4, 5] 
                                     else 'Summer' if x in [6, 7, 8] 
                                     else 'Autumn')

# Clean data: Drop NaN values in PM2.5 column
data_cleaned = data.dropna(subset=['PM2.5'])

# Streamlit dashboard title
st.title("Dashboard Analisis Polusi Udara di Beijing")

# Sidebar for navigation
st.sidebar.title("Navigasi")
option = st.sidebar.selectbox(
    "Pilih Visualisasi", 
    ["Kapan Polusi Udara Paling Tinggi?", 
     "Hubungan Variabel Meteorologi dengan Polusi", 
     "Clustering Berdasarkan Tingkat Polusi"]
)

# Visualisasi 1: Kapan Polusi Udara Paling Tinggi?
if option == "Kapan Polusi Udara Paling Tinggi?":
    st.subheader("Distribusi PM2.5 Berdasarkan Musim")

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(x='season', y='PM2.5', data=data_cleaned, palette='Set2', ax=ax)
    ax.set_title('Distribusi PM2.5 Berdasarkan Musim')
    ax.set_xlabel('Musim')
    ax.set_ylabel('Konsentrasi PM2.5 (µg/m³)')
    st.pyplot(fig)

    st.subheader("Distribusi PM2.5 Berdasarkan Jam dalam Sehari")

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(x='hour', y='PM2.5', data=data_cleaned, palette='coolwarm', ax=ax)
    ax.set_title('Distribusi PM2.5 Berdasarkan Jam dalam Sehari')
    ax.set_xlabel('Jam (24-hour)')
    ax.set_ylabel('Konsentrasi PM2.5 (µg/m³)')
    st.pyplot(fig)

# Visualisasi 2: Hubungan Variabel Meteorologi dengan Polusi
if option == "Hubungan Variabel Meteorologi dengan Polusi":
    st.subheader("Heatmap Korelasi antara Variabel Cuaca dan Polusi Udara")

    correlation_matrix = data_cleaned[['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3', 'TEMP', 'PRES', 'WSPM']].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5, ax=ax)
    ax.set_title('Heatmap Korelasi antara Variabel Cuaca dan Polusi Udara')
    st.pyplot(fig)

    st.subheader("Scatter Plot Hubungan Suhu dan PM2.5")

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(x='TEMP', y='PM2.5', data=data_cleaned, hue='season', palette='viridis', ax=ax)
    ax.set_title('Hubungan antara Suhu dan PM2.5')
    ax.set_xlabel('Suhu (°C)')
    ax.set_ylabel('PM2.5 (µg/m³)')
    st.pyplot(fig)

# Visualisasi 3: Clustering Berdasarkan Tingkat Polusi
if option == "Clustering Berdasarkan Tingkat Polusi":
    st.subheader("Clustering Berdasarkan Tingkat Polusi PM2.5")

    data_cleaned['Polusi Level'] = pd.cut(
        data_cleaned['PM2.5'], 
        bins=[0, 50, 100, 150, 300, 500], 
        labels=['Baik', 'Sedang', 'Tidak Sehat', 'Sangat Tidak Sehat', 'Berbahaya']
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    data_cleaned['Polusi Level'].value_counts().plot.pie(
        autopct='%1.1f%%', colors=sns.color_palette('Set3'), ax=ax
    )
    ax.set_title('Distribusi Tingkat Polusi Berdasarkan PM2.5')
    st.pyplot(fig)

    st.subheader("Scatter Plot PM2.5 vs PM10 Berdasarkan Tingkat Polusi")

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(x='PM2.5', y='PM10', data=data_cleaned, hue='Polusi Level', palette='Set1', ax=ax)
    ax.set_title('PM2.5 vs PM10 Berdasarkan Tingkat Polusi')
    ax.set_xlabel('PM2.5 (µg/m³)')
    ax.set_ylabel('PM10 (µg/m³)')
    st.pyplot(fig)
