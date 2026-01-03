import time
import asyncio

import numpy as np
import pandas as pd
import requests
import httpx
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime

def load_data(path: str):
    """
    Функция для загрузки и базовой предподготовки загружаемых данных.
    """
    df = pd.read_csv(path) # Загружаем данные из csv-файла
    df["timestamp"] = pd.to_datetime(df["timestamp"]) # Приводим столбец с датой к формату datetime
    df = df.sort_values(["city", "timestamp"]).reset_index(drop=True) # Сортировка данных по городу и времени
    return df

def roll_analysis(df: pd.DataFrame, window: int = 30):
    """
    Функция проводит анализ временного ряда температуры.

    Для каждого города рассчитываются:
    - средняя температура с окном 30 дней
    - стандартное отклонение температуры с окном 30 дней
    - аномалии, если значение температуры выходит за пределы скользящего среднего +- 2 стандартных отклонения
    """
    df = df.copy()

    df["roll_mean_30"] = (
        df.groupby("city")["temperature"] # Разделяем данные на группы по городам и берем признак температуры
        .rolling(window=window, min_periods=window) # Задаем параметр "Скользящего окна". Берем "окно" в 30 дней
        .mean() # Считаем среднее внутри окна
        .reset_index(level=0, drop=True) # Убираем лишний уровень индекса, появившийся после groupby + rolling
        )

    df["roll_std_30"] = (
        df.groupby("city")["temperature"]
        .rolling(window=window, min_periods=window)
        .std() # Считаем стандартное отклонение внутри окна
        .reset_index(level=0, drop=True)
        )

    # Поиск аномалий относительно сколзящего среднего
    df["is_roll_anomaly"] = (
        (df["temperature"] > df["roll_mean_30"] + 2 * df["roll_std_30"]) |
        (df["temperature"] < df["roll_mean_30"] - 2 * df["roll_std_30"])
        )

    return df

def seasonal_analysis(df: pd.DataFrame):
    """
    Функция рассчитывает сезонные статистики температуры для каждого города.

    Для каждого города и сезона (пара) рассчитываются:
    - средняя температура,
    - стандартное отклонение температуры,
    - флаг аномалии, если значение температуры выходит за пределы среднего ± 2 стандартных отклонения.
    """
    df = df.copy()

    season_stats = (
        df.groupby(["city", "season"])["temperature"] # Группировка по городу и сезону, далее берем признак температуры
        .agg(season_mean="mean", season_std="std") # Считаем для каждой группы среднее и стандартное отклонение
        .reset_index()
        )

    # Присоединяем сезонные расчеты к основному датафрейму
    df = df.merge(
        season_stats,
        on=["city", "season"],
        how="left"
        )

    # Расчет сезонных аномалий
    df["is_season_anomaly"] = (
        (df["temperature"] > df["season_mean"] + 2 * df["season_std"]) |
        (df["temperature"] < df["season_mean"] - 2 * df["season_std"])
        )

    return df

def long_term_trend(df: pd.DataFrame):
    """
    Функция рассчитывает долгосрочный тренд изменения температуры.
    По тренду можно предположить, будет ли в долгосрочной перспективе расти / снижаться / не меняться.
    """

    df = df.copy()
    df["trend"] = np.nan # Создаем новый столбец и заполняем его Nan

    for city in df["city"].unique(): # Проходимся циклом по уникальным городам
        city_df = df[df["city"] == city]
        t = np.arange(len(city_df)) # Используем порядковый номер наблюдения как временную ось
        coeff = np.polyfit(t, city_df["temperature"], 1) # Расчитываем коэффициенты прямой
        trend = coeff[0] * t + coeff[1] # Считаем значение тренда

        df.loc[city_df.index, "trend"] = trend # Записываем значение в таблицу

    return df

def analyze_city(df: pd.DataFrame, city: str):
    """
    Функция выполняет полный анализ температурных данных для одного города (объединение предыдущих функций).
    """
    city_df = df[df["city"] == city] # Отбираем данные одного города
    city_df = roll_analysis(city_df) # Анализ временного ряда (скользящее среднее + стандартное отклонение)
    city_df = seasonal_analysis(city_df) # Сезонный анализ температуры
    city_df = long_term_trend(city_df) # Долгосрочный тренд изменения температуры

    return city_df

def sequential_analysis(df: pd.DataFrame):
    """
    Функция выполняет последовательный температурный анализ.
    """
    start = time.time() # Фиксируем время старта работы кода
    res = {} # Создаем словарь результатов

    for city in df["city"].unique(): # Проходимся циклом по городам и проводим анализ температур
        res[city] = analyze_city(df, city)

    exec_time = time.time() - start # Считаем время работы кода
    return {"result": res, "time": exec_time}

def city_wrapper(args):
    """
    Вспомогательная функция для проведения параллельного анализа.
    """
    df, city = args
    return city, analyze_city(df, city)

def parallel_analysis(df: pd.DataFrame):
    """
    Функция выполняет параллельный температурный анализ.
    """
    start = time.time() # Фиксируем время старта работы кода
    res = {}

    cities = df["city"].unique() # Создаем список уникальных городов

    with ProcessPoolExecutor() as executor: # Создангие пула процессов
        out = executor.map(
            city_wrapper,
            [(df, city) for city in cities]
            )

    for city, city_df in out: # Сбор результатов в словарь
        res[city] = city_df

    exec_time = time.time() - start # Считаем время работы кода
    return {"result": res, "time": exec_time}

def current_temperature_api(city: str, api_key: str):
    """
    Получает текущую темературу в градусах Цельсия для указанного города через OpenWeatherMap.
    Документация - https://openweathermap.org/current
    """
    url = "https://api.openweathermap.org/data/2.5/weather" # Эндпоинт
    params = { # Параметры запроса
        "q": city, # Поиск по названию города
        "appid": api_key, # API-ключ
        "units": "metric" # Температура в градусах Цельсия
        }

    response = requests.get(url, params=params) # Отправляем HTTP запрос
    data = response.json() # Обрабатываем ответ в формате JSON

    if response.status_code != 200: # Вывод ошибок при запросе
        raise ValueError(data)

    return data["main"]["temp"] # Возвращаем температуру

async def current_temperature_api_async(city: str, api_key: str):
    """
    Асинхронно получает текущую температуру в градусах Цельсия для указанного города через OpenWeatherMap.
    """
    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {
        "q": city,
        "appid": api_key,
        "units": "metric"
        }

    async with httpx.AsyncClient() as cl: # Создание ассинхронного HTTP-клиента

        response = await cl.get(url, params=params)
        data = response.json()

    if response.status_code != 200:
        raise ValueError(data)

    return data["main"]["temp"]

async def current_temperature_api_async_multiple(cities, api_key):
    """
    Асинхронно получает текущую температуру в градусах Цельсия для списка городов через OpenWeatherMap.
    """
    task_list = [ # Создаем список задач
        current_temperature_api_async(city, api_key)
        for city in cities
        ]
    temp = await asyncio.gather(*task_list) # Одновременный запуск задач
    return dict(zip(cities, temp)) # Собираем в словарь пару Город - температура

def get_season(date: datetime):
    """
    Определяется сезон года на основе номера месяца указанной даты.
    """
    month = date.month # Получаем номер месяца
    
    # Если номер месяца равен Х, Y, Z, то это сезон A
    if month in (3, 4, 5): 
        return "spring"
    elif month in (6, 7, 8):
        return "summer"
    elif month in (9, 10, 11):
        return "autumn"
    else:
        return "winter"

def check_anomaly(city: str, current_temp: float, df: pd.DataFrame):
    """
    Функция проверяет, является ли текущая температура аномальной.
    """
    season = get_season(datetime.now()) # Получаем сезон на текущее время

    stats = (df[(df["city"] == city) & (df["season"] == season)].iloc[0]) # Фильтруем таблицу по городу и сезону + берем 1 строку

    lower_lim = stats["season_mean"] - 2 * stats["season_std"] # Рассчитываем нижнюю границу "нормальных" значений
    upper_lim = stats["season_mean"] + 2 * stats["season_std"] # Рассчитываем верхнюю границу "нормальных" значений

    is_anomaly = "normal" if lower_lim <= current_temp <= upper_lim else "anomaly" # Условие аномальности

    return {
        "city": city,
        "season": season,
        "temperature": current_temp,
        "is_anomaly": is_anomaly
        }
