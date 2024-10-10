import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima_model import ARIMA

# Load dataset
data = pd.read_csv('PRSA_Data_Shunyi_20130301-20170228.csv')
data['datetime'] = pd.to_datetime(data[['year', 'month', 'day', 'hour']])
data.set_index('datetime', inplace=True)

# Data Cleaning
data_cleaned = data[['PM2.5', 'TEMP', 'DEWP', 'PRES', 'RAIN', 'WSPM']].dropna()

# Title of the dashboard
st.title("Dashboard Kualitas Udara di Beijing")

# Section 1: Analisis Pola Musiman
st.header("Analisis Pola Musiman PM2.5")
monthly_avg_pm25 = data_cleaned.resample('M')['PM2.5'].mean()
moving_avg_pm25 = monthly_avg_pm25.rolling(window=12).mean()

plt.figure(figsize=(10, 6))
plt.plot(monthly_avg_pm25, label='Rata-rata Bulanan PM2.5', color='blue')
plt.plot(moving_avg_pm25, label='Moving Average (12 Bulan)', color='red', linestyle='--')
plt.title('Tren Musiman PM2.5 dengan Moving Average')
plt.xlabel('Tanggal')
plt.ylabel('Konsentrasi PM2.5 (µg/m³)')
plt.legend()
st.pyplot(plt)

# Section 2: Clustering
st.header("Clustering Berdasarkan Tingkat Polusi Udara")
features = data_cleaned[['PM2.5', 'TEMP', 'WSPM']].dropna()
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

kmeans = KMeans(n_clusters=3, random_state=42)
features['Cluster'] = kmeans.fit_predict(scaled_features)

plt.figure(figsize=(8, 6))
sns.scatterplot(x='PM2.5', y='TEMP', hue='Cluster', data=features, palette='Set1')
plt.title('Clustering Berdasarkan PM2.5 dan Suhu')
plt.xlabel('PM2.5 (µg/m³)')
plt.ylabel('Suhu (°C)')
st.pyplot(plt)

# Section 3: Forecasting Polusi Udara
st.header("Forecasting PM2.5 Menggunakan ARIMA")
pm25_series = data_cleaned['PM2.5'].dropna()

# Split data into training and testing
train_size = int(len(pm25_series) * 0.8)
train, test = pm25_series[:train_size], pm25_series[train_size:]

# Build ARIMA model
model = ARIMA(train, order=(5, 1, 2))
model_fit = model.fit()
forecast = model_fit.forecast(steps=len(test))[0]

plt.figure(figsize=(10, 6))
plt.plot(train.index, train, label='Data Training')
plt.plot(test.index, test, label='Data Testing', color='green')
plt.plot(test.index, forecast, label='Prediksi ARIMA', color='red', linestyle='--')
plt.title('Forecasting PM2.5 Menggunakan ARIMA')
plt.xlabel('Tanggal')
plt.ylabel('Konsentrasi PM2.5 (µg/m³)')
plt.legend()
st.pyplot(plt)

# Section 4: Menampilkan Data Tabel
st.header("Tabel Data Kualitas Udara")
st.dataframe(data_cleaned.head(10))

# Footer
st.write("Dashboard ini memberikan informasi mengenai kualitas udara di Beijing berdasarkan analisis data.")