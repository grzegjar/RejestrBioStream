import streamlit as st
import pandas as pd
from datetime import date
import requests
import sqlite3
import os
from database import get_stored_data

# TO MUSI BYĆ PIERWSZA KOMENDA STREAMLIT
st.set_page_config(page_title="Rejestr BIO", layout="wide")

# Dopiero teraz importujemy pozostałe moduły
from database import get_weather_data_by_date
from config import API_KEY, LAT, LON, MIEJSCOWOSC, GDRIVE_FILE_ID

url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"


def load_database():
    """Pobiera bazę danych z Google Drive jeśli nie istnieje lub się zmieniła"""
    try:
        db_filename = "weather_data.db"

        # Sprawdź rozmiar pliku na Google Drive używając GET z ograniczeniem pobierania
        #st.info("🔍 Sprawdzanie aktualności bazy danych...")

        # Pobierz tylko nagłówki aby sprawdzić rozmiar
        with requests.get(url, stream=True) as response:
            if response.status_code != 200:
                st.error(f"❌ Błąd sprawdzania pliku: Status {response.status_code}")
                return None

            remote_size = int(response.headers.get('content-length', 0))
           # st.write(f"📦 Rozmiar pliku na Google Drive: {remote_size} bajtów")

        # Sprawdź czy plik lokalny istnieje i ma ten sam rozmiar
        local_exists = os.path.exists(db_filename)

        if local_exists:
            local_size = os.path.getsize(db_filename)
            #st.write(f"💾 Rozmiar pliku lokalnego: {local_size} bajtów")

            if local_size == remote_size and remote_size > 0:
             #   st.success("✅ Używam istniejącej bazy danych (pliki identyczne)")

                # Sprawdź czy baza jest poprawna
                try:
                    conn = sqlite3.connect(db_filename)
                    cursor = conn.cursor()
                    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                    tables = cursor.fetchall()
                    conn.close()

                    if tables:
                        #st.success(f"✅ Baza danych poprawna! Tabele: {tables}")
                        return db_filename
                    else:
                        st.warning("⚠️ Baza istnieje ale nie zawiera tabel - pobieram ponownie")
                except sqlite3.Error:
                    st.warning("⚠️ Baza istnieje ale jest uszkodzona - pobieram ponownie")
            else:
                st.info("🔄 Plik się zmienił lub rozmiary różne - pobieram nową wersję")
        else:
            st.info("📥 Brak lokalnej bazy - pobieram z Google Drive")

        # Jeśli potrzebujemy pobrać, kontynuujemy z tym samym połączeniem
        st.info("📥 Pobieranie bazy danych z Google Drive...")
        r = requests.get(url)

        if r.status_code != 200:
            st.error(f"❌ Błąd pobierania: Status {r.status_code}")
            return None

        downloaded_size = len(r.content)
        st.write(f"📦 Rozmiar pobranego pliku: {downloaded_size} bajtów")

        if downloaded_size == 0:
            st.error("❌ Pobrano pusty plik!")
            return None

        # Zapisz plik
        with open(db_filename, "wb") as f:
            f.write(r.content)

        # Sprawdź czy plik został zapisany
        if os.path.exists(db_filename):
            final_size = os.path.getsize(db_filename)
            st.write(f"💾 Zapisany plik: {final_size} bajtów")

            # Sprawdź czy to prawidłowa baza SQLite
            try:
                conn = sqlite3.connect(db_filename)
                cursor = conn.cursor()
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                tables = cursor.fetchall()
                conn.close()

                st.success(f"✅ Baza danych załadowana! Tabele: {tables}")
                return db_filename

            except sqlite3.Error as e:
                st.error(f"❌ Błąd SQLite: {e}")
                return None
        else:
            st.error("❌ Plik nie został zapisany!")
            return None

    except Exception as e:
        st.error(f"❌ Błąd podczas ładowania bazy: {e}")
        return None
def check_database_structure():
    """Sprawdza strukturę bazy danych"""
    try:
        if not os.path.exists("weather_data.db"):
            return "Brak pliku bazy danych", None, None, None

        conn = sqlite3.connect("weather_data.db")
        cursor = conn.cursor()

        # Sprawdź jakie tabele istnieją
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()

        table_info = {}
        sample_data = None
        available_dates = None

        if tables:
            for table in tables:
                cursor.execute(f"PRAGMA table_info({table[0]})")
                columns = cursor.fetchall()
                table_info[table[0]] = columns

            # Sprawdź przykładowe dane z tabeli weather
            weather_tables = [t[0] for t in tables]
            if 'weather' in weather_tables:
                cursor.execute("SELECT * FROM weather LIMIT 5")
                sample_data = cursor.fetchall()
                cursor.execute("SELECT DISTINCT date FROM weather LIMIT 10")
                available_dates = cursor.fetchall()

        conn.close()
        return tables, table_info, sample_data, available_dates

    except Exception as e:
        return f"Błąd: {e}", None, None, None

def diagnostyka():
    # WCZYTAJ I SPRAWDŹ BAZĘ DANYCH
    st.title("🔧 Diagnostyka systemu")

    with st.expander("📊 Status bazy danych", expanded=True):
        db_path = load_database()

        if db_path and False:
            tables, table_info, sample_data, available_dates = check_database_structure()

            st.subheader("Struktura bazy danych:")
            st.write("**Tabele w bazie:**", tables)

            if table_info:
                st.write("**Struktura tabel:**")
                for table_name, columns in table_info.items():
                    st.write(f"📋 **Tabela {table_name}:**")
                    for col in columns:
                        st.write(f"   - {col[1]} ({col[2]})")

            if sample_data:
                st.write("**Przykładowe dane z tabeli weather:**")
                for row in sample_data:
                    st.write(f"   {row}")

            if available_dates:
                st.write("**Dostępne daty w bazie:**")
                for date_row in available_dates:
                    st.write(f"   - {date_row[0]}")
        else:
            st.error("❌ Nie udało się załadować bazy danych!")

if False:
    '''
        NA POCZĄTKU APLIKACJI, po st.title():
        tab1, tab2, tab3 = st.tabs(["📊 Dane główne", "🌦️ Pogoda", "📈 Statystyki"])
        
        with tab1:
            # Tutaj przenieś główne sekcje (Biometryka, Dieta, Odczucia)
            
        with tab2:
            # Tutaj tylko dane pogodowe
            
        with tab3:
            # Ewentualne statystyki
    '''
db_path = load_database()
tables, table_info, sample_data, available_dates = check_database_structure()
data = date.today()
city, lat,lon, selected_date, hgh, hgl, sen, stres, dieta_values1, dieta_values2, bol, godzina_bolu, samopoczucie, uwagi, data_str = (
    MIEJSCOWOSC,LAT,LON,date.today(),0,0,0,1,(0,0,0,0,0),(0,0,0,0,0),0,'',10,'',date.today().strftime('%Y-%m-%d'))

tab1, tab2, tab3, tab4, tab5 = st.tabs(["🌍 Lokalizacja", "💓 Biometryka", "🍎 Dieta","🧠 Odczucia","💾 Zapis"])

def entry_from_base(data,tab1, tab2, tab3, tab4):
    data_str = data.today().strftime('%Y-%m-%d')
    row_bio_diet = get_stored_data(data_str)

    columns = ("date","max_temp","min_temp","max_pressure","min_pressure","avg_wind_speed","lat","lon","delta_temp","delta_pressure",
                "sen","stres","HgL","HgH","bol","samopoczucie","slodycze","nabial","mieso_biale",
                "mieso_czerwone","alkohol","kawa","przetworzone","warzywa_owoce","gluten","orzechy","uwagi","miejscowosc","godzina_bolu")
    # columns i row_diet są już zdefiniowane
    data_dict = dict(zip(columns, row_bio_diet))

    # GŁÓWNA APLIKACJA
    st.title("📊 Rejestr BIO")


    # --- Sekcja Lokalizacja ---
    with tab1:
        st.header("🌍 Lokalizacja")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            city = st.text_input("Miejscowość", f"{data_dict["miejscowosc"]}")
        with col2:
            lat = st.number_input(f"Szerokość geogr. (lat)", value=data_dict["lat"], format="%.8f")
        with col3:
            lon = st.number_input("Długość geogr. (lon)", value=data_dict["lon"], format="%.8f")
        with col4:
            selected_date = st.date_input("Data", value=data_dict["date"])
    with tab2:
        # --- Sekcja Biometryka ---
        st.header("💓 Biometryka")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            hgh = st.number_input("HgH", min_value=0, max_value=220, value=data_dict["HgH"])
        with col2:
            hgl = st.number_input("HgL", min_value=0, max_value=180, value=data_dict["HgL"])
        with col3:
            sen = st.number_input("Sen (h)", min_value=0, max_value=24, value=int(data_dict["sen"]))
        with col4:
            stres = st.slider("Stres (1-10)", 1, 10, data_dict["stres"])
    with tab3:
        st.header("🍎 Dieta")
        cols1 = st.columns(5)
        dieta_labels1 = ["Mięso białe (%)", "Mięso czerwone (%)", "Nabiał (%)", "Słodycze (g)", "Alkohol (ml)"]
        dieta_val1 = [data_dict["mieso_biale"],data_dict["mieso_czerwone"],data_dict["nabial"],data_dict["slodycze"],data_dict["alkohol"]]
        dieta_values1 = [cols1[i].number_input(dieta_labels1[i], min_value=0, max_value=100, value=dieta_val1[i]) for i in range(5)]

        cols2 = st.columns(5)
        dieta_labels2 = ["Warzywa i owoce (%)", "Przetworzone (%)", "Gluten (%)", "Orzechy (garść)", "Kawa (szt.)"]
        dieta_val2 = [data_dict["warzywa_owoce"], data_dict["przetworzone"], data_dict["gluten"], data_dict["orzechy"],
                      data_dict["kawa"]]

        dieta_values2 = [cols2[i].number_input(dieta_labels2[i], min_value=0, max_value=100, value=dieta_val2[i]) for i in range(5)]

    with tab4:
        # --- Sekcja Odczucia ---
        st.header("🧠 Odczucia")
        col1, col2, col3 = st.columns(3)
        with col1:
            bol = st.slider("Ból (1-10)", 0, 10, data_dict["bol"])
        with col2:
            godzina_bolu = st.number_input("Godzina bólu (0-23)", min_value=0, max_value=23, value=data_dict["godzina_bolu"])
        with col3:
            samopoczucie = st.slider("Samopoczucie (1-10)", 1, 10, data_dict["samopoczucie"])

        uwagi = st.text_area("📝 Uwagi", data_dict["uwagi"])
        data_str = date.today().strftime('%Y-%m-%d')

    return city, lat,lon, selected_date, hgh, hgl, sen, stres, dieta_values1, dieta_values2, bol, godzina_bolu, samopoczucie, uwagi, data_str

entry_from_base(data)
if False:
    """
# GŁÓWNA APLIKACJA
st.title("📊 Rejestr BIO - dane biometryczne, pogodowe i dietetyczne")

# --- Sekcja Lokalizacja ---
st.header("🌍 Lokalizacja")
col1, col2, col3, col4 = st.columns(4)
with col1:
    city = st.text_input("Miejscowość", f"{MIEJSCOWOSC}")
with col2:
    lat = st.number_input(f"Szerokość geogr. (lat)", value=LAT, format="%.8f")
with col3:
    lon = st.number_input("Długość geogr. (lon)", value=LON, format="%.8f")
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
    sen = st.number_input("Sen (h)", min_value=0, max_value=24, value=7)
with col4:
    stres = st.slider("Stres (1-10)", 1, 10, 5)

# db_path = load_database()
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
data_str =  date.today().strftime('%Y-%m-%d')
"""

# --- Przyciski ---
with tab5:
    col1, col2 = st.columns(2)
    with col1:
        if st.button("💾 Zapisz dane"):
            st.success("Dane zapisane (przykładowo – tu dodamy zapis do bazy SQLite / Google Drive)")
    with col2:
        if st.button("🧹 Wyczyść dane"):
            st.info("Dane wyczyszczone (tymczasowo – jeszcze bez logiki)")

# --- Sekcja Tabeli pogodowej ---
st.header("🌦️ Dane pogodowe")

if db_path and tables and any('weather' in str(t) for t in tables):
    try:
        columns, rows = get_weather_data_by_date(selected_date)

        if rows:
            df = pd.DataFrame(rows, columns=columns)
            st.dataframe(df, use_container_width=True)
            st.success(f"✅ Znaleziono {len(rows)} rekordów pogodowych")
        else:
            st.warning(f"⚠️ Brak danych pogodowych dla daty: {selected_date}")

            # Pokaż dostępne daty
            if available_dates:
                st.info("📅 Dostępne daty w bazie:")
                for date_row in available_dates:
                    st.write(f"   - {date_row[0]}")

    except Exception as e:
        st.error(f"❌ Błąd podczas ładowania danych pogodowych: {str(e)}")
else:
    st.error("❌ Baza danych nie jest dostępna lub nie zawiera tabeli 'weather'")

def test_połączenia_z_bazą_danych():
    # Dodatkowy test bezpośredniego połączenia z bazą
    with st.expander("🔍 Bezpośredni test bazy danych"):
        if st.button("Przetestuj połączenie z bazą"):
            try:
                conn = sqlite3.connect("weather_data.db")
                cursor = conn.cursor()

                # Sprawdź tabele
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                tables = cursor.fetchall()
                st.write("Tabele:", tables)

                # Sprawdź dane z weather
                if tables and any('weather' in str(t) for t in tables):
                    cursor.execute("SELECT COUNT(*) FROM weather")
                    count = cursor.fetchone()[0]
                    st.write(f"Liczba rekordów w weather: {count}")

                    cursor.execute("SELECT * FROM weather LIMIT 3")
                    sample = cursor.fetchall()
                    st.write("Przykładowe rekordy:", sample)

                conn.close()
                st.success("✅ Test zakończony pomyślnie")

            except Exception as e:
                st.error(f"❌ Błąd testu: {e}")