import pandas as pd
import numpy as np
import streamlit as st
import requests
import matplotlib.pyplot as plt

# Этап 1: Загрузка и обработка данных
def load_data(file):
    """Загрузка исторических данных."""
    return pd.read_csv(file)

def calculate_statistics(data):
    """Вычисление скользящего среднего, стандартного отклонения и аномалий."""
    data['rolling_mean'] = data['temperature'].rolling(window=30).mean()
    data['rolling_std'] = data['temperature'].rolling(window=30).std()
    data['upper_bound'] = data['rolling_mean'] + 2 * data['rolling_std']
    data['lower_bound'] = data['rolling_mean'] - 2 * data['rolling_std']
    data['anomaly'] = (data['temperature'] > data['upper_bound']) | (data['temperature'] < data['lower_bound'])
    return data

def seasonal_statistics(data):
    """Средняя температура и стандартное отклонение по сезонам."""
    return data.groupby(['city', 'season'])['temperature'].agg(['mean', 'std']).reset_index()

# Этап 2: Получение текущей температуры
def get_current_temperature(api_key, city):
    """Получение текущей температуры через OpenWeatherMap API."""
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()['main']['temp']
    else:
        st.error(f"Error: {response.json().get('message', 'Invalid API key. Please see https://openweathermap.org/faq#error401 for more info.')}")
        return None

# Этап 3: Построение приложения
st.title("Анализ температурных данных")

# Загрузка данных
uploaded_file = st.file_uploader("Загрузите файл с историческими данными (CSV):", type=["csv"])
if uploaded_file:
    data = load_data(uploaded_file)
    st.write("Исторические данные:", data.head())

    # Выбор города
    city = st.selectbox("Выберите город:", data['city'].unique())
    city_data = data[data['city'] == city]

    # Обработка данных
    processed_data = calculate_statistics(city_data)
    seasonal_stats = seasonal_statistics(city_data)

    # Отображение описательной статистики
    st.subheader("Описательная статистика")
    st.write(seasonal_stats)

    # Построение графика временного ряда
    st.subheader("Временной ряд температур")
    fig, ax = plt.subplots()
    ax.plot(processed_data['timestamp'], processed_data['temperature'], label='Температура')
    ax.plot(processed_data['timestamp'], processed_data['rolling_mean'], label='Скользящее среднее', color='orange')
    ax.fill_between(processed_data['timestamp'], processed_data['lower_bound'], processed_data['upper_bound'], color='gray', alpha=0.2, label='Диапазон ±2σ')
    ax.scatter(processed_data['timestamp'][processed_data['anomaly']],
               processed_data['temperature'][processed_data['anomaly']],
               color='red', label='Аномалии')
    ax.legend()
    plt.xticks(rotation=45)
    st.pyplot(fig)

    # Получение текущей температуры
    st.subheader("Мониторинг текущей температуры")
    api_key = st.text_input("Введите ваш OpenWeatherMap API Key:")
    if api_key:
        current_temp = get_current_temperature(api_key, city)
        if current_temp is not None:
            st.write(f"Текущая температура в городе {city}: {current_temp}°C")

            # Определение нормальности температуры
            current_season = city_data['season'].iloc[-1]
            season_mean = seasonal_stats[(seasonal_stats['city'] == city) & (seasonal_stats['season'] == current_season)]['mean'].values[0]
            season_std = seasonal_stats[(seasonal_stats['city'] == city) & (seasonal_stats['season'] == current_season)]['std'].values[0]

            lower_bound = season_mean - 2 * season_std
            upper_bound = season_mean + 2 * season_std

            if lower_bound <= current_temp <= upper_bound:
                st.write("Температура в норме для текущего сезона.")
                st.write(f"Верхняя граница: {round(upper_bound, 2}")
                st.write(f"Нижняя граница: {round(lower_bound, 2)}")
            else:
                st.write("Температура аномальна для текущего сезона.")
                st.write(f"Верхняя граница: {round(upper_bound, 2}")
                st.write(f"Нижняя граница: {round(lower_bound, 2)}")
