import streamlit as st
from datetime import date
from config import MIEJSCOWOSC, LAT, LON, API_KEY
import pandas as pd
import requests
import time
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
import joblib

import matplotlib.pyplot as plt


st.set_page_config(page_title="Przewidywanie bólu", layout="wide")

st.title("Przewidywanie bólu")

# =========================
#        PARAMETRY
# =========================
st.header("Parametry")

# --- Lokalizacja ---
with st.expander("Lokalizacja", expanded=True):
    col1, col2, col3 = st.columns(3)

    with col1:
        city = st.text_input("Bydgoszcz:", MIEJSCOWOSC)

    with col2:
        lat = st.text_input("Szerokość geogr. (lat):", LAT)

        lon = st.text_input("Długość geogr. (lon):", LON)

    with col3:
        data = st.date_input("Data:", value=date.today())

# --- Biometryka ---
with st.expander("Biometryka", expanded=True):
    col1, col2, col3 = st.columns(3)

    with col1:
        hgh = st.text_input("HgH:", "140")
    with col2:
        hgl = st.text_input("HgL:", "95")
    with col3:
        sen = st.number_input("Sen (h: 0-24):", min_value=0, max_value=24, step=1, value=7)

    stres = st.number_input("Stres (1-10):", min_value=1, max_value=10, step=1,value=2)

# --- Dieta ---
with st.expander("Dieta", expanded=True):
    # st.subheader("Linia 1")
    cols = st.columns(5)
    diet_labels1 = ["Mięso białe (%)", "Mięso czerwone (%)", "Nabiał (%)",
                    "Słodycze (g)", "Alkohol (ml)"]
    diet_values1 = []
    for col, label in zip(cols, diet_labels1):
        diet_values1.append(col.number_input(label, min_value=0, step=1))

    # st.subheader("Linia 2")
    cols = st.columns(5)
    diet_labels2 = ["Warzywa i owoce (%)", "Przetworzone (%)",
                    "Gluten (%)", "Orzechy (garść)", "Kawa (szt.)"]
    diet_values2 = []
    for col, label in zip(cols, diet_labels2):
        diet_values2.append(col.number_input(label, min_value=0, step=1))

# --- Odczucia ---
with st.expander("Odczucia", expanded=True):
    samopoczucie = st.number_input("Samopoczucie (1-10):", min_value=1, max_value=10, step=1,value=10)

# =========================
#      ZAPIS DANYCH
# =========================
if st.button("Zapisz dane"):
    # Tworzymy jeden wiersz zgodny z tabelą daily_df
    new_row = {
        "date": str(data),
        "lat": float(lat) if lat else None,
        "lon": float(lon) if lon else None,
        "max_temp": None,
        "min_temp": None,
        "max_pressure": None,
        "min_pressure": None,
        "delta_temp": None,
        "delta_pressure": None,
        "avg_wind_speed": None,
        "sen": int(sen) if sen else None,
        "stres": int(stres),
        "HgL": int(hgl) if hgl else None,
        "HgH": int(hgh) if hgh else None,
        "miejscowosc": city,
        "bol": 0,
        "samopoczucie": int(samopoczucie),
        "slodycze": int(diet_values1[3]),
        "nabial": int(diet_values1[2]),
        "mieso_biale": int(diet_values1[0]),
        "mieso_czerwone": int(diet_values1[1]),
        "alkohol": int(diet_values1[4]),
        "kawa": int(diet_values2[4]),
        "przetworzone": int(diet_values2[1]),
        "warzywa_owoce": int(diet_values2[0]),
        "gluten": int(diet_values2[2]),
        "orzechy": int(diet_values2[3]),
        "uwagi": "",
        "godzina_bolu": 0,
    }

    # Jeśli nie istnieje df_daily → tworzymy
    if "df_daily" not in st.session_state:
        st.session_state.df_daily = pd.DataFrame(columns=new_row.keys())

    # Usuwamy istniejący wiersz dla tej daty (PRIMARY KEY)
    st.session_state.df_daily = (
        st.session_state.df_daily[st.session_state.df_daily["date"] != new_row["date"]]
    )

    # Dopisujemy nowy rekord
    st.session_state.df_daily = pd.concat(
        [st.session_state.df_daily, pd.DataFrame([new_row])],
        ignore_index=True
    )

    st.success("Dane zapisane do DataFrame df_daily!")
    st.dataframe(st.session_state.df_daily)

###############################
#       DANE OpenWeather
###############################
# ---------------------------------
# Stałe i kolumny DataFrame
# ---------------------------------
COLUMNS = [
    "date", "hour", "lat", "lon",
    "temp", "feels_like", "pressure", "humidity", "dew_point",
    "uvi", "clouds", "visibility",
    "wind_speed", "wind_deg", "wind_gust",
    "rain", "snow",
    "weather_id", "weather_main", "weather_description", "weather_icon",
    "czy_bol", "poziom_bolu", "illum"
]
def moon_phase(year, month, day):
    # prosta metoda Conwaya
    r = year % 100
    r %= 19
    r = ((r * 11) % 30) + month + day
    if month < 3:
        r += 2
    r -= (0 if year < 2000 else 4)
    r = r % 30
    return r  # 0=new, 15=full
# ---------------------------------
# Funkcja zapisująca godzinę do DataFrame
# ---------------------------------
def save_hourly_weather(date_str, hour, lat, lon, ow_hour):
    weather = ow_hour.get("weather", [])
    weather0 = weather[0] if weather else {}

    year, month, day = map(int, date_str.split("-"))
    illum = moon_phase(year, month, day)
    print("Faza księżyca:", illum)
    illum = moon_phase(year, month, day)
    # print("Faza księżyca:", illum)
    # illum = round(m.phase)  # faza księżyca w %

    row = {
        "date": date_str,
        "hour": hour,
        "lat": float(lat) if lat is not None else None,
        "lon": float(lon) if lon is not None else None,
        "temp": ow_hour.get("temp"),
        "feels_like": ow_hour.get("feels_like"),
        "pressure": ow_hour.get("pressure"),
        "humidity": ow_hour.get("humidity"),
        "dew_point": ow_hour.get("dew_point"),
        "uvi": ow_hour.get("uvi"),
        "clouds": ow_hour.get("clouds"),
        "visibility": ow_hour.get("visibility"),
        "wind_speed": ow_hour.get("wind_speed"),
        "wind_deg": ow_hour.get("wind_deg"),
        "wind_gust": ow_hour.get("wind_gust"),
        "rain": (ow_hour.get("rain") or {}).get("1h"),
        "snow": (ow_hour.get("snow") or {}).get("1h"),
        "weather_id": weather0.get("id"),
        "weather_main": weather0.get("main"),
        "weather_description": weather0.get("description"),
        "weather_icon": weather0.get("icon"),
        "czy_bol": None,
        "poziom_bolu": None,
        "illum": illum
    }

    if "df_weather" not in st.session_state:
        st.session_state.df_weather = pd.DataFrame(columns=COLUMNS)

    df = st.session_state.df_weather
    # Usuń wiersz o tym samym date+hour
    df = df[~((df["date"] == date_str) & (df["hour"] == hour))]
    # Dopisz nowy
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df = df.sort_values(["date", "hour"]).reset_index(drop=True)
    st.session_state.df_weather = df

# ---------------------------------
# Funkcja pobierająca godziny z TimeMachine
# ---------------------------------
def get_hourly_weather(lat, lon, api_key, timestamp, max_retries=3, retry_delay=2):
    url = "https://api.openweathermap.org/data/3.0/onecall/timemachine"
    params = {"lat": lat, "lon": lon, "dt": timestamp, "appid": api_key, "units": "metric"}

    for attempt in range(max_retries):
        try:
            response = requests.get(url, params=params)
            if response.status_code == 200:
                data = response.json()
                # Obsługa różnych formatów
                if "hourly" in data and isinstance(data["hourly"], list):
                    return data["hourly"]
                elif "data" in data and isinstance(data["data"], list):
                    return data["data"]
                else:
                    print(f"Brak danych godzinowych dla timestamp {timestamp}")
                    return None
            else:
                print(f"Błąd API {response.status_code}: {response.text}")
        except requests.RequestException as e:
            print(f"Błąd połączenia: {e}")

        time.sleep(retry_delay)
    return None

# ---------------------------------
# Funkcja pobierająca 24 godziny danego dnia
# ---------------------------------
def get_daily_stats(lat, lon, api_key, day, tz_name="Europe/Warsaw"):
    tz = ZoneInfo(tz_name)
    date_str = day.strftime("%Y-%m-%d")

    for hour in range(24):
        local_dt = datetime.combine(day, datetime.min.time()) + timedelta(hours=hour)
        timestamp = int(local_dt.replace(tzinfo=tz).astimezone(timezone.utc).timestamp())
        hourly_data = get_hourly_weather(lat, lon, api_key, timestamp)
        if not hourly_data:
            print(f"Brak danych godzinowych dla {date_str} {hour}:00")
            continue

        for hour_data in hourly_data:
            dt_local = datetime.fromtimestamp(hour_data["dt"], tz=timezone.utc).astimezone(tz)
            if dt_local.date() == day:
                save_hourly_weather(date_str, dt_local.hour, lat, lon, hour_data)

def prepare_data():
    """
    Przygotowuje dane do trenowania modelu, wczytując je z bazy danych,
    łącząc tabele i tworząc nowe cechy.

    Args:
        db_path (str): Ścieżka do pliku bazy danych SQLite.

    Returns:
        tuple: Zawiera przygotowane ramki danych (X, Y) oraz listę użytych cech.
    """
    # Wczytanie danych

    df_weather = st.session_state.df_weather
    df_daily = st.session_state.df_daily

    # Pobieramy dane pogodowe dla daty predykcji i 4 poprzednich dni

    # Sprawdzenie NaN
    print('Kolumny z wartościami NaN:')
    for columns, sum in df_weather.isna().sum().items():
        if sum != 0:
            print(f'- df_weather-> \t{columns}: {sum}')
    for columns, sum in df_daily.isna().sum().items():
        if sum != 0:
            print(f'- df_daily-> \t{columns}: {sum}')

    print('\nSprawdzam Typ domyślny snow:')
    print(df_weather['snow'].dtype)  # W przypadku gdy wszytskie wartości są puste, DF przetrzymuje typ danych object
    df_weather['snow'] = pd.to_numeric(df_weather['snow'], errors='coerce').fillna(0)  # Zmień jawnie typ na float
    print('\nTyp snow po jawnej deklaracji:')
    print(df_weather['snow'].dtype)

    df_weather["date"] = pd.to_datetime(df_weather["date"])
    df_daily["date"] = pd.to_datetime(df_daily["date"])

    # ============ Łączenie danych ===================================
    df = pd.merge(df_weather, df_daily, on="date", how="left", suffixes=["_hourly", ""])
    # Sortowanie
    df.sort_values(by=["date", "hour"], inplace=True)

    # Definicja zmiennej docelowej i cech
    Y_col = "poziom_bolu"

    # Wybór cech Bio
    # initial_bio_diet_cols = [
    #     "sen", "stres", "HgH", "HgL", "slodycze", "nabial", "mieso_biale",
    #     "mieso_czerwone", "alkohol", "kawa", "przetworzone", "warzywa_owoce",
    #     "gluten", "orzechy"
    # ]
    initial_bio_diet_cols = [
        "HgH", "HgL", "slodycze", "nabial",
        "alkohol", "przetworzone", "warzywa_owoce",
        "gluten", "orzechy"]
    # rekomendacja
    #
    # ZOSTAWIĆ(rdzeń modelu):
    # UVI
    # interakcje: uvi_dew_temp, uvi_pressure_temp
    # Meteo: temp, humidity, wind_speed, pressure
    # Hematologia: HgL_sr_2dni, HgH_wczoraj
    # Dieta: nabial_sr_2dni, przetworzone_sr_2dni, orzechy_sr_2dni, słodycze_sr_2dni

    # Różnicowe: temp_diff_mean
    # USUNĄĆ(redundantne / szum):
    # Pozostałe
    # UVI(uvi_dew_pressure, uvi_humidity_temp, uvi_humidity_pressure)
    # sen_ *, stres_ *
    # mięso_białe_ *, mięso_czerwone_ *, kawa_ *
    # dew_point, pressure_diff_mean
    # Nadmiarowe
    # HgL / HgH(zostawić tylko 2)
    # gluten_ *, alkohol_ *, warzywa_owoce_ * (jeśli nie chcesz przeładowywać modelu)


    # bio_diet_cols_present wszystkie pola określone w initial_bio_diet_cols, ale podwarunkiem że znajdują się w df.columns
    bio_diet_cols_present = [col for col in initial_bio_diet_cols if col in df.columns]
    # print (bio_diet_cols_present)

    # Dodaj do df dane cechy bio z wczorajszego dnia i średnie z dwóch dni
    for col in bio_diet_cols_present:
        df[f"{col}_wczoraj"] = df[f"{col}"].shift(24).fillna(0)
        df[f"{col}_pwczoraj"] = df[f"{col}_wczoraj"].shift(24).fillna(0)
        df[f"{col}_śr_2_dni"] = (df[f"{col}_wczoraj"] + df[f"{col}_pwczoraj"]) / 2

    # print(df[["date", "hour", "nabial_wczoraj", "slodycze_wczoraj", "nabial_śr_2_dni", "slodycze_śr_2_dni"]].iloc[1020:1030])
    # mask = (df['date'] == '2025-08-08')
    # mask = (df['date'] >= '2025-08-11') & (df['date'] <= '2025-08-12')
    # print (f'Podgląd:')
    # print(df.loc[mask,["date", "hour", "nabial", "nabial_wczoraj", "nabial_śr_2_dni"]])

    # Definicja zmiennej docelowej i cech


    # Wybór cech pogodowych
    okres_LiczbGodzin = 2
    print(f'Okres [h]: {okres_LiczbGodzin}')
    df["temp_mean_prev"] = df["temp"].shift(1).rolling(window=okres_LiczbGodzin).mean()
    df["temp_diff_mean"] = df["temp"] - df["temp_mean_prev"]
    df["temp_diff_mean"] = df["temp_diff_mean"].fillna(0)

    df["pressure_mean_prev"] = df["pressure"].shift(1).rolling(window=okres_LiczbGodzin).mean()
    df["pressure_diff_mean"] = df["pressure"] - df["pressure_mean_prev"]
    df["pressure_diff_mean"] = df["pressure_diff_mean"].fillna(0)
    # print(df[["date", "hour", "pressure", "pressure_diff_mean", "temp_diff_mean"]].iloc[1020:1030])
    # df["uvi_dew_temp"] = df["uvi"].fillna(0) * df["dew_point"].fillna(0) * df["temp"].fillna(0)
    df["uvi_pressure_temp"] = df["uvi"].fillna(0) * df["pressure"].fillna(0) * df["temp"].fillna(0)
    # df["uvi_temp"] = df["uvi"].fillna(0) * df["temp"].fillna(0)
    # df["uvi_pressure"] = df["uvi"].fillna(0) * df["pressure"].fillna(0)
    # df["uvi_humidity"] = df["uvi"].fillna(0) * df["humidity"].fillna(0)

    ################
    # df["delta_temp_1h"] = df["temp"].diff()
    # df["delta_press_1h"] = df["pressure"].diff()
    #
    # # wartości bezwzględne
    # df["abs_delta_temp_1h"] = df["delta_temp_1h"].abs()
    # df["abs_delta_press_1h"] = df["delta_press_1h"].abs()
    #
    # # kierunek zmian (trend: -1 = spadek, 0 = brak, 1 = wzrost)
    # df["trend_temp"] = np.sign(df["delta_temp_1h"]).fillna(0)
    # df["trend_press"] = np.sign(df["delta_press_1h"]).fillna(0)
    #
    # # relacje między temp i ciśnieniem
    # df["temp_press_product"] = df["delta_temp_1h"] * df["delta_press_1h"]
    # df["temp_press_ratio"] = df["delta_temp_1h"] / (df["delta_press_1h"].replace(0, np.nan))
    #
    # # --- zmiany w dłuższym oknie (3h i 6h) ---
    # for h in [3, 6]:
    #     df[f"delta_temp_{h}h"] = df["temp"].diff(h)
    #     df[f"delta_press_{h}h"] = df["pressure"].diff(h)

    weather_features = ["temp", "pressure", "humidity", "illum", "dew_point", "temp_diff_mean",
                        "pressure_diff_mean",'uvi_pressure_temp']



    # Przepisz wybrane cechy Pogodowe i Bio do features_columns
    # Przepisz wybrane weather_features, tylko jeśli są w df
    feature_columns = []
    for col in weather_features:
        if col in df.columns:
            feature_columns.append(col)
    # Bio wczorajsze
    for col_base in bio_diet_cols_present:
        if f"{col_base}_wczoraj" in df.columns:
            feature_columns.append(f"{col_base}_wczoraj")
        if f"{col_base}_śr_2_dni" in df.columns:
            feature_columns.append(f"{col_base}_śr_2_dni")

    feature_columns = list(dict.fromkeys(feature_columns))
    print(f'feature_columns {feature_columns}')
    cols_to_check_for_na = [Y_col] + feature_columns
    cols_to_check_for_na = [col for col in cols_to_check_for_na if col in df.columns]

    # df.dropna(subset=cols_to_check_for_na, inplace=True) #Usuwanie rekordów z pustymi wartościami (NaN)

    df[cols_to_check_for_na] = df[cols_to_check_for_na].fillna(0)  # Zerowanie pustych wartości

    if len(df) < 20:
        print(f"Zbyt mało danych ({len(df)} próbek) do trenowania modelu. Dalsze kroki nie mają sensu")

    Y = df[Y_col]
    X = df[feature_columns]

    return X, Y, feature_columns
def predict_for_date(model, db_path, feature_columns, prediction_date_str):
    """
    Przewiduje ból głowy dla określonej daty, używając wytrenowanego modelu.

    Args:
        model: Wytrenowany model do predykcji.
        db_path (str): Ścieżka do pliku bazy danych SQLite.
        feature_columns (list): Lista cech użytych do trenowania modelu.
        prediction_date_str (str): Data w formacie 'YYYY-MM-DD', dla której ma być wykonana predykcja.

    Returns:
        pd.DataFrame: Ramka danych z predykcjami dla określonej daty.
    """
    try:
        prediction_date = pd.to_datetime(prediction_date_str)
    except ValueError:
        print(f"Błąd: Nieprawidłowy format daty '{prediction_date_str}'. Oczekiwano 'YYYY-MM-DD'.")
        return None

    try:
        df_weather = st.session_state.df_weather
        df_daily = st.session_state.df_daily

    except Exception as e:
        print(f"Błąd podczas wczytywania danych do predykcji: {e}")
        return None

    if df_weather.empty or df_daily.empty:
        print(f"Brak danych w bazie dla daty {prediction_date_str} lub dni poprzedzających.")
        return None

    df_weather["date"] = pd.to_datetime(df_weather["date"])
    df_daily["date"] = pd.to_datetime(df_daily["date"])

    # Łączenie ramek danych
    df_predict = pd.merge(df_weather, df_daily, on="date", how="left", suffixes=["_hourly", ""])

    # Inżynieria cech dla danych do predykcji
    df_predict.sort_values(by=["date", "hour"], inplace=True)

    # initial_bio_diet_cols = [
    #     "sen", "stres", "HgH", "HgL", "slodycze", "nabial", "mieso_biale",
    #     "mieso_czerwone", "alkohol", "kawa", "przetworzone", "warzywa_owoce",
    #     "gluten", "orzechy"]
    initial_bio_diet_cols = [
        "HgH", "HgL", "slodycze", "nabial",
        "alkohol", "przetworzone", "warzywa_owoce",
        "gluten", "orzechy"]

    bio_diet_cols_present = [col for col in initial_bio_diet_cols if col in df_predict.columns]

    for col in bio_diet_cols_present:
        df_predict[f"{col}_wczoraj"] = df_predict[f"{col}"].shift(24).fillna(0)
        df_predict[f"{col}_pwczoraj"] = df_predict[f"{col}_wczoraj"].shift(24).fillna(0)
        df_predict[f"{col}_śr_2_dni"] = (df_predict[f"{col}_wczoraj"] + df_predict[f"{col}_pwczoraj"]) / 2

    # Inżynieria cech pogodowych
    okres_LiczbGodzin = 2
    print(f'Okres [h]: {okres_LiczbGodzin}')
    df_predict["temp_mean_prev"] = df_predict["temp"].shift(1).rolling(window=okres_LiczbGodzin).mean()
    df_predict["temp_diff_mean"] = df_predict["temp"] - df_predict["temp_mean_prev"]
    df_predict["temp_diff_mean"] = df_predict["temp_diff_mean"].fillna(0)

    df_predict["pressure_mean_prev"] = df_predict["pressure"].shift(1).rolling(window=okres_LiczbGodzin).mean()
    df_predict["pressure_diff_mean"] = df_predict["pressure"] - df_predict["pressure_mean_prev"]
    df_predict["pressure_diff_mean"] = df_predict["pressure_diff_mean"].fillna(0)

    # df_predict["uvi_dew_temp"] = df_predict["uvi"].fillna(0) * df_predict["dew_point"].fillna(0) * df_predict["temp"].fillna(0)
    df_predict["uvi_pressure_temp"] = df_predict["uvi"].fillna(0) * df_predict["pressure"].fillna(0) * df_predict["temp"].fillna(0)
    # df_predict["uvi_temp"] = df_predict["uvi"].fillna(0) * df_predict["temp"].fillna(0)
    # df_predict["uvi_pressure"] = df_predict["uvi"].fillna(0) * df_predict["pressure"].fillna(0)
    # df_predict["uvi_humidity"] = df_predict["uvi"].fillna(0) * df_predict["humidity"].fillna(0)

    ################
    # df_predict["delta_temp_1h"] = df_predict["temp"].diff()
    # df_predict["delta_press_1h"] = df_predict["pressure"].diff()
    #
    # # wartości bezwzględne
    # df_predict["abs_delta_temp_1h"] = df_predict["delta_temp_1h"].abs()
    # df_predict["abs_delta_press_1h"] = df_predict["delta_press_1h"].abs()
    #
    # # kierunek zmian (trend: -1 = spadek, 0 = brak, 1 = wzrost)
    # df_predict["trend_temp"] = np.sign(df_predict["delta_temp_1h"]).fillna(0)
    # df_predict["trend_press"] = np.sign(df_predict["delta_press_1h"]).fillna(0)
    #
    # # relacje między temp i ciśnieniem
    # df_predict["temp_press_product"] = df_predict["delta_temp_1h"] * df_predict["delta_press_1h"]
    # df_predict["temp_press_ratio"] = df_predict["delta_temp_1h"] / (df_predict["delta_press_1h"].replace(0, np.nan))
    #
    # # --- zmiany w dłuższym oknie (3h i 6h) ---
    # for h in [3, 6]:
    #     df_predict[f"delta_temp_{h}h"] = df_predict["temp"].diff(h)
    #     df_predict[f"delta_press_{h}h"] = df_predict["pressure"].diff(h)

    # weather_features = ["temp", "pressure", "humidity", "wind_speed", "dew_point", "uvi_dew_temp", "temp_diff_mean",
    #                     "pressure_diff_mean","delta_temp_1h","abs_delta_press_1h","trend_temp","trend_press","temp_press_product","temp_press_ratio","delta_temp_3h","delta_press_6h"]
    # Filtrujemy dane tylko dla prediction_date
    df_predict_target_date = df_predict[df_predict["date"] == prediction_date].copy()

    if df_predict_target_date.empty:
        print(f"Brak danych do predykcji dla daty {prediction_date_str} po inżynierii cech.")
        return None

    # Upewnienie się, że wszystkie wymagane kolumny istnieją w df_predict_target_date
    for col in feature_columns:
        if col not in df_predict_target_date.columns:
            df_predict_target_date[col] = 0  # Uzupełnienie brakujących kolumn zerami

    # Diagnostyka różnic w cechach
    missing_features = [col for col in feature_columns if col not in df_predict_target_date.columns]
    extra_features = [col for col in df_predict_target_date.columns if
                      col not in feature_columns and col != "date" and col != "hour" and col != "predicted_pain"]
    if missing_features or extra_features:
        print(f"Brakujące cechy w danych predykcyjnych: {missing_features}")
        print(f"Dodatkowe cechy w danych predykcyjnych: {extra_features}")

    X_predict = df_predict_target_date[feature_columns]

    # Predykcja
    predictions = model.predict(X_predict)
    df_predict_target_date["predicted_pain"] = predictions

    return df_predict_target_date[["date", "hour", "predicted_pain"]]

def przewidywanie_bez_uczenia(pred_for_date):
    loaded_model = joblib.load('model.joblib')

    if not pred_for_date is None:
        print(pred_for_date)
        prediction_date_input = pred_for_date
        # Krok 1: Przygotowanie danych
        X, Y, features = prepare_data()

        if X is not None and Y is not None:
            # loaded_model (X,Y)
            print(f'data pred: {prediction_date_input}')
            predictions_output = predict_for_date(loaded_model, "", features, prediction_date_input)
            if predictions_output is not None:
                print(f"\n--- Prognoza bólu głowy na {prediction_date_input} ---")
                # print(predictions_output)
                plot_pain_prediction(predictions_output, prediction_date_input, 0, 0, loaded_model)
    else:
        st.warning("X Błąd daty")


def plot_pain_prediction(predictions_output, prediction_date_str, best_r2, mae_pain, best_model):
    """
    Wyświetla wykres przewidywanego poziomu bólu głowy w poszczególnych godzinach w Streamlit,
    z uwzględnieniem zakresu błędu MAE.

    Args:
        predictions_output (pd.DataFrame): Ramka danych z predykcjami (kolumny 'hour', 'predicted_pain').
        prediction_date_str (str): Data predykcji w formacie 'YYYY-MM-DD'.
        best_r2 (float): Wartość R^2 najlepszego modelu.
        mae_pain (float): Średni błąd bezwzględny (MAE) modelu.
        best_model (str): Nazwa najlepszego modelu.
    """
    if predictions_output is None or predictions_output.empty:
        st.info("Brak danych do wygenerowania wykresu.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    # Wykres przewidywanego bólu
    ax.plot(predictions_output["hour"], predictions_output["predicted_pain"], marker="o", linestyle="-", label="Przewidywany ból")

    # Dodanie linii błędu MAE
    ax.plot(predictions_output["hour"], predictions_output["predicted_pain"] + mae_pain, linestyle=":", color="gray", label=f"+MAE ({mae_pain:.2f})")
    ax.plot(predictions_output["hour"], predictions_output["predicted_pain"] - mae_pain, linestyle=":", color="gray", label=f"-MAE ({mae_pain:.2f})")

    ax.set_title(f"Przewidywany poziom bólu głowy na {prediction_date_str} ({best_model})")
    ax.set_xlabel("Godzina")
    ax.set_ylabel("Przewidywany poziom bólu")
    ax.set_xticks(range(0, 24))
    ax.grid(True)
    ax.set_ylim(bottom=0)
    ax.legend()
    plt.tight_layout()

    # Wyświetlenie w Streamlit
    st.pyplot(fig)

    # Opcjonalnie: zapis do pliku
    plot_filename = f"headache_prediction_{prediction_date_str}.png"
    fig.savefig(plot_filename)
    st.success(f"Wykres zapisano jako: {plot_filename}")

# ---------------------------------
# Streamlit – demo
# ---------------------------------
st.title(f"Dane z OpenWeather 0-23h dla wybranego dnia {data}")
selected_date = data

if st.button("Pobierz z OpenWeather"):
    get_daily_stats(LAT, LON, API_KEY, selected_date)
    st.success("✔ Dane zapisane do daily_df")

# Wyświetlenie DataFrame
st.subheader("Dane godzinowe")
if "df_weather" in st.session_state:
    st.dataframe(st.session_state.df_weather.sort_values(["date", "hour"]), use_container_width=True)
    prepare_data()
else:
    st.info("Brak danych — pobierz dane z OpenWeather")


if st.button("Prognoza bólu"):
    przewidywanie_bez_uczenia(data)
