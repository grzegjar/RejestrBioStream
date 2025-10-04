import streamlit as st
import pandas as pd
from datetime import date
from your_module import get_weather_data_by_date  # <- Twój moduł do pobierania danych

st.set_page_config(page_title="Rejestr BIO", layout="wide")

st.title("📊 Rejestr BIO - dane biometryczne, pogodowe i dietetyczne")

# --- Sekcja Lokalizacja ---
st.header("🌍 Lokalizacja")
col1, col2, col3, col4 = st.columns(4)
with col1:
    city = st.text_input("Miejscowość", "Warszawa")
with col2:
    lat = st.number_input("Szerokość geogr. (lat)", value=52.23)
with col3:
    lon = st.number_input("Długość geogr. (lon)", value=21.01)
with col4:
    selected_date = st.date_input("Data", date.today())

# --- Sekcja Biometryka ---
st.header("💓 Biometryka")
col1, col2, col3, col4 = st.columns(4)
with col1:
    hgh = st.number_input("HgH", min_value=0, max_value=200, value=120)
with col2:
    hgl = st.number_input("HgL", min_value=0, max_value=200, value=80)
with col3:
    sen = st.number_input("Sen (h)", min_value=0.0, max_value=24.0, value=7.0)
with col4:
    stres = st.slider("Stres (1-10)", 1, 10, 5)

# --- Sekcja Dieta ---
st.header("🍎 Dieta")
cols = st.columns(5)
dieta_labels = ["Mięso białe (%)", "Mięso czerwone (%)", "Nabiał (%)", "Słodycze (g)", "Alkohol (ml)"]
dieta_values = [cols[i].number_input(dieta_labels[i], min_value=0, max_value=100, value=0) for i in range(5)]

cols2 = st.columns(5)
dieta_labels2 = ["Warzywa i owoce (%)", "Przetworzone (%)", "Gluten (%)", "Orzechy (garść)", "Kawa (szt.)"]
dieta_values2 = [cols2[i].number_input(dieta_labels2[i], min_value=0, max_value=100, value=0) for i in range(5)]

# --- Sekcja Odczucia ---
st.header("🧠 Odczucia")
col1, col2, col3 = st.columns(3)
with col1:
    bol = st.slider("Ból (1-10)", 0, 10, 0)
with col2:
    godzina_bolu = st.number_input("Godzina bólu (0-23)", min_value=0, max_value=23, value=0)
with col3:
    samopoczucie = st.slider("Samopoczucie (1-10)", 1, 10, 5)

uwagi = st.text_area("📝 Uwagi", "")

# --- Przyciski ---
col1, col2 = st.columns(2)
with col1:
    if st.button("💾 Zapisz dane"):
        st.success("Dane zapisane (przykładowo – tu dodamy zapis do bazy SQLite / Google Drive)")
with col2:
    if st.button("🧹 Wyczyść dane"):
        st.info("Dane wyczyszczone (tymczasowo – jeszcze bez logiki)")

# --- Sekcja Tabeli pogodowej ---
st.header("🌦️ Dane pogodowe")
columns, rows = get_weather_data_by_date(selected_date)
if rows:
    df = pd.DataFrame(rows, columns=columns)
    st.dataframe(df, use_container_width=True)
else:
    st.warning("Brak danych pogodowych dla wybranej daty.")
