import streamlit as st
from datetime import date
from config import MIEJSCOWOSC, LAT, LON

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
    dane = {
        "city": city,
        "lat": lat,
        "lon": lon,
        "data": str(data),
        "hgh": hgh,
        "hgl": hgl,
        "sen": sen,
        "stres": stres,
        "dieta1": diet_values1,
        "dieta2": diet_values2,
        "godzina_bolu": godzina_bolu,
        "samopoczucie": samopoczucie,
    }

    st.success("Dane zostały zapisane!")
    st.json(dane)
