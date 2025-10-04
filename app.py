import streamlit as st
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import pickle
import requests
from io import BytesIO

# --- Pobierz bazę danych z Dysku Google ---
GDRIVE_FILE_ID = "1Y5mfNI-KvJ7GdhXlQcevhvP_JxhS1Qkr"
url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"

@st.cache_data
def load_database():
    r = requests.get(url)
    open("weather_data.db", "wb").write(r.content)
    return "weather_data.db"

db_path = load_database()
conn = sqlite3.connect(db_path)

# --- Interfejs Streamlit ---
st.title("Aplikacja predykcji danych biometrycznych")

# Formularz wejściowy
with st.form("form_input"):
    temp = st.number_input("Temperatura", -20.0, 40.0, 20.0)
    pressure = st.number_input("Ciśnienie", 950.0, 1050.0, 1013.0)
    uvi = st.number_input("UVI", 0.0, 11.0, 2.0)
    submitted = st.form_submit_button("Oblicz predykcję")

# --- Predykcja ---
if submitted:
    model = pickle.load(open("model.pkl", "rb"))
    X_new = pd.DataFrame([[temp, pressure, uvi]], columns=["temp", "pressure", "uvi"])
    y_pred = model.predict(X_new)[0]
    st.success(f"Przewidywany poziom bólu: {y_pred:.2f}")

# --- Wykres z bazy ---
df = pd.read_sql_query("SELECT date, bol FROM weather", conn)
st.dataframe(df.tail(10))

fig, ax = plt.subplots()
ax.plot(df["date"], df["bol"])
ax.set_title("Ostatnie wyniki bólu")
st.pyplot(fig)
