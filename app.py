import pandas as pd
import numpy as np
import streamlit as st
import requests
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
import asyncio
import aiohttp
import time

# ========== Этап 1: Загрузка и обработка данных ==========
def load_data(file):
    """Загрузка исторических данных."""
    return pd.read_csv(file)

def process_city_data(city_data):
    """Обработка данных для одного города."""
    city_data['rolling_mean'] = city_data['temperature'].rolling(window=30).mean()
    city_data['rolling_std'] = city_data['temperature'].rolling(window=30).std()
    city_data['upper_bound'] = city_data['rolling_mean'] + 2 * city_data['rolling_std']
    city_data['lower_bound'] = city_data['rolling_mean'] - 2 * city_data['rolling_std']
    city_data['anomaly'] = (city_data['temperature'] > city_data['upper_bound']) | (city_data['temperature'] < city_data['lower_bound'])
    return city_data

def analyze_with_parallelism(data):
    """Параллельный анализ данных."""
    grouped_data = [group for _, group in data.groupby('city')]
    with ThreadPoolExecutor() as executor:
        processed_data = list(executor.map(process_city_data, grouped_data))
    return pd.concat(processed_data)

def analyze_without_parallelism(data):
    """Последовательный анализ данных."""
    processed_data = []
    for _, group in data.groupby('city'):
        processed_data.append(process_city_data(group))
    return pd.concat(processed_data)

def seasonal_statistics(data):
    """Средняя температура и стандартное отклонение по сезонам."""
    return data.groupby(['city', 'season'])['temperature'].agg(['mean', 'std']).reset_index()

# ========== Этап 2: Получение текущей температуры ==========
async def fetch_temperature_async(session, api_key, city):
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    async with session.get(url) as response:
        if response.status == 200:
            return await response.json()
        else:
            return {'error': f"Error fetching data for {city}: {response.status}"}

async def monitor_with_async(api_key, cities):
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_temperature_async(session, api_key, city) for city in cities]
        results = await asyncio.gather(*tasks)
    return results

def fetch_temperature_sync(api_key, city):
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        return {'error': f"Error fetching data for {city}: {response.status_code}"}

def monitor_without_async(api_key, cities):
    results = [fetch_temperature_sync(api_key, city) for city in cities]
    return results

# ========== Этап 3: Построение приложения ==========
st.title("Анализ температурных данных")

# Загрузка данных
uploaded_file = st.file_uploader("Загрузите файл с историческими данными (CSV):", type=["csv"])
if uploaded_file:
    data = load_data(uploaded_file)
    st.write("Исторические данные:", data.head())

    # Выбор города
    city = st.selectbox("Выберите город:", data['city'].unique())
    city_data = data[data['city'] == city]

    # Анализ данных
    st.subheader("Анализ данных")
    start_time = time.time()
    processed_data_seq = analyze_without_parallelism(data)
    time_seq = time.time() - start_time

    start_time = time.time()
    processed_data_par = analyze_with_parallelism(data)
    time_par = time.time() - start_time

    st.write(f"Время последовательного анализа: {time_seq:.2f} сек")
    st.write(f"Время параллельного анализа: {time_par:.2f} сек")

    # Описательная статистика
    seasonal_stats = seasonal_statistics(city_data)
    st.subheader("Описательная статистика")
    st.write(seasonal_stats)

    # Построение графика временного ряда
    st.subheader("Временной ряд температур")
    fig, ax = plt.subplots()
    ax.plot(processed_data_par['timestamp'], processed_data_par['temperature'], label='Температура')
    ax.plot(processed_data_par['timestamp'], processed_data_par['rolling_mean'], label='Скользящее среднее', color='orange')
    ax.fill_between(processed_data_par['timestamp'], processed_data_par['lower_bound'], processed_data_par['upper_bound'], color='gray', alpha=0.2, label='Диапазон ±2σ')
    ax.scatter(processed_data_par['timestamp'][processed_data_par['anomaly']], 
               processed_data_par['temperature'][processed_data_par['anomaly']], 
               color='red', label='Аномалии')
    ax.legend()
    plt.xticks(rotation=45)
    st.pyplot(fig)

    # Мониторинг текущей температуры
    st.subheader("Мониторинг текущей температуры")
    api_key = st.text_input("Введите ваш OpenWeatherMap API Key:")
    cities = data['city'].unique()

    if api_key:
        st.write("Синхронный мониторинг температуры...")
        start_time = time.time()
        results_sync = monitor_without_async(api_key, cities)
        time_sync = time.time() - start_time
        st.write(f"Время выполнения: {time_sync:.2f} сек")

        st.write("Асинхронный мониторинг температуры...")
        start_time = time.time()
        results_async = asyncio.run(monitor_with_async(api_key, cities))
        time_async = time.time() - start_time
        st.write(f"Время выполнения: {time_async:.2f} сек")

        st.write("Результаты синхронного мониторинга:")
        st.write(results_sync)
        st.write("Результаты асинхронного мониторинга:")
        st.write(results_async)
