import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# Импортируем функции из back части проекта
from back import (load_data, roll_analysis, seasonal_analysis, long_term_trend, current_temperature_api, check_anomaly)

# Основной заголовок приложения
st.title("Анализ температурных данных и выявление аномалий")

# Описание функциональности приложения
st.write(
    """
    Приложение позволяет:
    - Анализировать исторические температурные данные (при загрузке CSV файла)
    - Выявлять аномалии в исторических данных
    - Получать текущую температуру через OpenWeatherMap API и определять ее на аномальность
    """)

# Подзаголовок приложения №1
st.header("Загрузка исторических данных")

# Создаем элемент для загрузки csv файла пользователем
upload_file = st.file_uploader("Загрузите csv файл", type=["csv"])

# Если файл не загружен, то приложение не будет работать
if upload_file is None:
    st.info("Загрузите csv файл")
    st.stop()

# Загрузка данных
df = load_data(upload_file)

# Проводим анализ исторических данных: скользящее среднее и СКО, сезонные статистики, долгосрочный тренд
df = roll_analysis(df)
df = seasonal_analysis(df)
df = long_term_trend(df)

# Вывод "надписи-успеха" в случае успешной обработки csv файла
st.success("Исторические данные успешно загружены")

# Подзаголовок приложения №2
st.header("Выбор города")

# Выпадающий список с городами, которые были указаны в загружаемом файле + сортировка в алфавитном порядке
city = st.selectbox("Выберите город", sorted(df["city"].unique()))

# Фильтрация датасета по выбранному городу
city_df = df[df["city"] == city]

# Подзаголовок приложения №2.1
st.subheader("Описательная статистика")

# Выводим базовую описательную статистику для числовых признаков
st.dataframe(city_df[["temperature", "roll_mean_30", "roll_std_30"]].describe())

# Подзаголовок приложения №2.2
st.subheader("Временной ряд температуры")

# Задаем "полотно" для графика
fig = go.Figure()

# Добавляем на ось Х временной ряд, на ось Y - историческую температуру
fig.add_trace(go.Scatter(
    x=city_df["timestamp"],
    y=city_df["temperature"],
    mode="lines",
    name="Температура"))

# Добавляем прямую - скользящее среднее
fig.add_trace(go.Scatter(
    x=city_df["timestamp"],
    y=city_df["roll_mean_30"],
    mode="lines",
    name="Скользящее среднее"))

# Добавляем маркеры - аномальные значения
anom = city_df[city_df["is_roll_anomaly"]]
fig.add_trace(go.Scatter(
    x=anom["timestamp"],
    y=anom["temperature"],
    mode="markers",
    name="Аномалии",
    marker=dict(color="red", size=3)))

# Отображаем интерактивный график
st.plotly_chart(fig)

# Подзаголовок приложения №2.3
st.subheader("Сезонные профили")

# Рассчитываем средниие сезонные параметры для выбранного города
season_profile = (city_df.groupby("season")[["season_mean", "season_std"]].mean())
st.dataframe(season_profile)

# Подзаголовок приложения №3
st.header("Мониторинг текущей температуры")

# Создаем поле для ввода API ключа
api_k = st.text_input("Введите API ключ к OpenWeatherMap", type="password")

# Если пользователь ввёл ключ, то выполняем запрос к API
if api_k:
    try:
        # Получаем текущую температуру для выбранного города
        curr_temp = current_temperature_api(city, api_k)

        # Отображаем значение температуры в выбранном городе
        st.metric(label=f"Текущая температура в городе {city}", value=f"{curr_temp} С")

        # Проверяем является ли температура аномальной
        anomaly_res = check_anomaly(city, curr_temp, df)

        # Выводим результат проверки на аномальность
        if anomaly_res["is_anomaly"] == "normal":
            st.success("Текущая температура является номарльной")
        else:
            st.error("Текущая температура является аномальной")

    # Обработка ошибки неверно введенного ключа
    except ValueError as e:
        st.error(f"Ошибка API: {e}")

# Если ключ не введён, то просим пользователя его ввести
else:
    st.info("Введите API ключ к OpenWeatherMap, чтобы получить текущую температуру")