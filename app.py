import sys
import os

print(f"=== DEBUG INTERPRETER ===")
print(f"Python executable: {sys.executable}")
print(f"Virtual env: {os.getenv('VIRTUAL_ENV')}")
print(f"Python path: {sys.path[:3]}")
print(f"=== KONIEC DEBUG ===")

import streamlit as st
import pandas as pd
from datetime import date
import requests
import sqlite3
from database import get_stored_data
import predict_headeche_stremalit
from datetime import timedelta
from GoogleDriveSync import DriveAutoSync
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import date, timedelta
import warnings

from predict_headeche_stremalit import (
    prepare_data, train_and_evaluate_models, predict_for_date,
    show_feature_importances_streamlit, plot_pain_prediction_streamlit,
    plot_learning_curves_streamlit, plot_shap_summary_streamlit,
    plot_correlation_streamlit, plot_korelacja_z_bolem_streamlit,
    analiza_poziomu_predykcji_streamlit
)

warnings.filterwarnings("ignore")

# TO MUSI BYĆ PIERWSZA KOMENDA STREAMLIT
st.set_page_config(page_title="Rejestr BIO", layout="wide")

# Inicjalizacja stanu sesji - TO JEST KLUCZOWE!
if 'database_initialized' not in st.session_state:
    st.session_state.database_initialized = False
if 'drive_sync' not in st.session_state:
    st.session_state.drive_sync = None


def initialize_database_once():
    """Inicjalizuje bazę danych TYLKO RAZ na sesję"""
    if not st.session_state.database_initialized:
        print('========== Inicjalizacja bazy danych ================')

        # Utwórz instancję DriveAutoSync tylko raz
        if st.session_state.drive_sync is None:
            st.session_state.drive_sync = DriveAutoSync(
                "4/1AVGzR1BKGcDm_KZYbu4JACAtwD0Nkfjz8kmCQrU60TQs9CNWBp3z_2hCIHo")

        # Pobierz bazę tylko jeśli potrzebne
        success = st.session_state.drive_sync.download_from_drive_2()

        if success:
            print('======= Baza danych gotowa do użycia ===============')
            st.session_state.database_initialized = True
        else:
            print('❌ Problem z bazą danych - używam lokalnej wersji jeśli istnieje')

    return st.session_state.database_initialized


# Inicjalizuj bazę danych (tylko raz)
database_ready = initialize_database_once()

# Sprawdź czy baza w ogóle istnieje
if not os.path.exists("weather_data.db"):
    st.error("❌ Brak bazy danych!")
    st.stop()

print('======= System gotowy ===============')

from database import get_weather_data_by_date
from config import API_KEY, LAT, LON, MIEJSCOWOSC, GDRIVE_FILE_ID

url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"

import sqlite3

conn = sqlite3.connect("weather_data.db")


def main_prediction_section():
    """Główna sekcja predykcji - oddzielona od reszty aplikacji"""
    st.title("🤖 Zaawansowane Predykcje Bólu Głowy")

    # Sidebar z opcjami
    st.sidebar.header("⚙️ Opcje Predykcji")

    # Data predykcji - UNIKALNY KLUCZ
    pred_date = st.sidebar.date_input(
        "Data predykcji",
        value=date.today() + timedelta(days=1),
        key="prediction_date_sidebar"
    )

    # Zaawansowane opcje
    with st.sidebar.expander("🔧 Opcje zaawansowane"):
        show_importance = st.checkbox("Importance cech", value=False, key="show_importance")
        show_learning = st.checkbox("Krzywe uczenia", value=False, key="show_learning")
        show_shap = st.checkbox("Analiza SHAP", value=False, key="show_shap")
        show_correlation = st.checkbox("Macierz korelacji", value=False, key="show_correlation")
        show_analysis = st.checkbox("Analiza klasyfikacji", value=False, key="show_analysis")

    # Przyciski akcji
    col1, col2 = st.sidebar.columns(2)
    with col1:
        predict_btn = st.button("🎯 Wykonaj predykcję", type="primary", use_container_width=True, key="predict_btn")
    with col2:
        clear_btn = st.button("🧹 Wyczyść", use_container_width=True, key="clear_btn")

    # Główna zawartość
    if predict_btn:
        run_prediction(str(pred_date), show_importance, show_learning, show_shap, show_correlation, show_analysis)

    # Informacje o modelu
    with st.sidebar.expander("ℹ️ O modelu"):
        st.write("""
        **Model wykorzystuje:**
        - Random Forest
        - Gradient Boosting  
        - XGBoost
        - LightGBM

        **Dane:** pogodowe, biometryczne, dietetyczne
        **Cel:** predykcja bólu głowy z wyprzedzeniem 24h
        """)


def debug_prepare_data():
    """Funkcja diagnostyczna do sprawdzenia prepare_data"""
    try:
        print("🔍 DEBUG: Sprawdzanie funkcji prepare_data...")
        X, Y, features = prepare_data("weather_data.db")

        print(f"X type: {type(X)}, shape: {X.shape if hasattr(X, 'shape') else 'No shape'}")
        print(f"Y type: {type(Y)}, shape: {Y.shape if hasattr(Y, 'shape') else 'No shape'}")
        print(f"features type: {type(features)}, value: {features}")

        if X is not None:
            print(f"X columns: {X.columns.tolist() if hasattr(X, 'columns') else 'No columns'}")
        if Y is not None:
            print(f"Y sample values: {Y[:5] if hasattr(Y, '__getitem__') else 'No values'}")

        return X, Y, features
    except Exception as e:
        print(f"❌ Błąd w prepare_data: {e}")
        return None, None, None
def run_prediction(prediction_date, show_importance, show_learning, show_shap, show_correlation, show_analysis):
    """Uruchamia predykcję - baza jest już załadowana w session_state"""

    # """Uruchamia predykcję z diagnostyką"""
    # if not st.session_state.database_initialized:
    #     st.error("❌ Baza danych nie jest gotowa!")
    #     return
    #
    # # Uruchom diagnostykę
    # with st.spinner("🔍 Diagnozowanie problemu z danymi..."):
    #     X, Y, features = debug_prepare_data()
    #
    # if X is None or Y is None:
    #     st.error("❌ Funkcja prepare_data zwróciła None - sprawdź konsolę dla szczegółów")
    #     return

    # Sprawdź czy baza jest gotowa
    if not st.session_state.database_initialized:
        st.error("❌ Baza danych nie jest gotowa!")
        return

    try:
        with st.spinner("🔄 Przygotowywanie danych..."):
            # Krok 1: Przygotowanie danych - używamy istniejącej bazy
            X, Y, features = prepare_data("weather_data.db")

            # DODAJEMY SPRAWDZENIE ZWROCONYCH WARTOŚCI
            if X is None or Y is None or features is None:
                st.error("❌ Nie udało się przygotować danych do predykcji - funkcja prepare_data zwróciła None")
                return

            # DODATKOWE SPRAWDZENIE CZY DANE SĄ POPRAWNE
            if len(X) == 0 or len(Y) == 0:
                st.error("❌ Brak danych do trenowania modelu")
                return

            st.success(f"✅ Przygotowano dane: {X.shape[0]} próbek, {len(features)} cech")

        with st.spinner("🤖 Trenowanie modeli..."):
            # Krok 2: Trenowanie modeli
            trained_models, best_model_name, best_r2, best_mae_pain, data_models = train_and_evaluate_models(X, Y)

            if not trained_models:
                st.error("❌ Nie udało się wytrenować modeli")
                return

            best_model = trained_models.get(best_model_name)

            if best_model is None:
                st.error("❌ Nie znaleziono najlepszego modelu")
                return

        # Wyświetl metryki modelu
        st.success(f"✅ Model wytrenowany pomyślnie!")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Najlepszy model", best_model_name)
        with col2:
            st.metric("Jakość (R²)", f"{best_r2:.4f}")
        with col3:
            st.metric("Błąd (MAE)", f"{best_mae_pain:.4f}")
        with col4:
            st.metric("Liczba cech", len(features))

        with st.spinner("📊 Generowanie predykcji..."):
            # Krok 3: Predykcja
            predictions_output = predict_for_date(best_model, "weather_data.db", features, prediction_date)

            if predictions_output is not None:
                display_prediction_results(predictions_output, prediction_date, best_r2, best_mae_pain, best_model_name)
            else:
                st.error("❌ Nie udało się wygenerować predykcji")

        # Dodatkowe wizualizacje
        if show_importance and best_model:
            with st.expander("📈 Importance cech", expanded=True):
                show_feature_importances_streamlit(best_model, features, top_n=15)

        if show_learning:
            with st.expander("🔄 Krzywe uczenia", expanded=True):
                plot_learning_curves_streamlit(data_models)

        if show_shap and best_model:
            with st.expander("🔍 Analiza SHAP", expanded=True):
                data_models_list = data_models.to_dict(orient='records')
                X_train = next(res["X_train"] for res in data_models_list if res["model_name"] == best_model_name)
                plot_shap_summary_streamlit(best_model, X_train, best_model_name)

        if show_correlation:
            with st.expander("📊 Macierz korelacji", expanded=True):
                data_models_list = data_models.to_dict(orient='records')
                X_train = next(res["X_train"] for res in data_models_list if res["model_name"] == best_model_name)
                plot_correlation_streamlit(X_train, features)
                plot_korelacja_z_bolem_streamlit(X, Y)

        if show_analysis and best_model:
            with st.expander("🎯 Analiza klasyfikacji", expanded=True):
                data_models_list = data_models.to_dict(orient='records')
                X_test = next(res["X_test"] for res in data_models_list if res["model_name"] == best_model_name)
                y_train = next(res["y_train"] for res in data_models_list if res["model_name"] == best_model_name)
                y_test = next(res["y_test"] for res in data_models_list if res["model_name"] == best_model_name)
                analiza_poziomu_predykcji_streamlit(best_model, X_test, y_train, y_test)

    except Exception as e:
        st.error(f"❌ Błąd podczas predykcji: {str(e)}")
        # DODAJEMY WIĘCEJ INFORMACJI O BŁĘDZIE
        st.error("💡 Sprawdź czy baza danych zawiera wymagane tabele i kolumny")


def display_prediction_results(predictions_df, prediction_date, r2_score, mae_score, model_name):
    """Wyświetla wyniki predykcji"""
    st.header(f"📊 Predykcja na {prediction_date}")

    # Tabela z wynikami
    st.subheader("Przewidywany poziom bólu w godzinach")

    # Formatowanie tabeli
    display_df = predictions_df.copy()
    display_df["predicted_pain"] = display_df["predicted_pain"].round(3)
    display_df["godzina"] = display_df["hour"].astype(str) + ":00"

    # Pokoloruj wiersze w zależności od poziomu bólu
    def color_pain(val):
        if val > 7:
            color = '#ff6b6b'
        elif val > 4:
            color = '#ffd166'
        else:
            color = '#06d6a0'
        return f'background-color: {color}'

    styled_df = display_df[["godzina", "predicted_pain"]].style.map(
        color_pain, subset=['predicted_pain']
    ).format({'predicted_pain': '{:.3f}'})

    st.dataframe(styled_df, width='stretch')

    # Wykres predykcji
    st.subheader("Wykres predykcji")
    fig = plot_pain_prediction_streamlit(predictions_df, prediction_date, r2_score, mae_score, model_name)
    st.pyplot(fig)

    # Statystyki
    col1, col2, col3 = st.columns(3)
    with col1:
        max_pain = predictions_df["predicted_pain"].max()
        st.metric("Maksymalny przewidywany ból", f"{max_pain:.2f}")
    with col2:
        avg_pain = predictions_df["predicted_pain"].mean()
        st.metric("Średni przewidywany ból", f"{avg_pain:.2f}")
    with col3:
        pain_hours = len(predictions_df[predictions_df["predicted_pain"] > 5])
        st.metric("Godziny z bólem >5", f"{pain_hours}/24")


def load_database():
    """Zwraca ścieżkę do bazy danych"""
    return "weather_data.db"


def check_database_structure():
    """Sprawdza strukturę bazy danych"""
    try:
        if not os.path.exists("weather_data.db"):
            return "Brak pliku bazy danych", None, None, None

        conn = sqlite3.connect("weather_data.db")
        cursor = conn.cursor()
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


def entry_from_base(data):
    data_str = data.strftime('%Y-%m-%d')
    row_bio_diet = get_stored_data(data_str)

    columns = (
        "date", "max_temp", "min_temp", "max_pressure", "min_pressure", "avg_wind_speed", "lat", "lon", "delta_temp",
        "delta_pressure",
        "sen", "stres", "HgL", "HgH", "bol", "samopoczucie", "slodycze", "nabial", "mieso_biale",
        "mieso_czerwone", "alkohol", "kawa", "przetworzone", "warzywa_owoce", "gluten", "orzechy", "uwagi",
        "miejscowosc",
        "godzina_bolu")

    if row_bio_diet is None:
        st.warning(f"⚠️ Brak danych dla daty {data_str} - używam wartości domyślnych")
        return MIEJSCOWOSC, LAT, LON, 0, 0, 0, 1, (0, 0, 0, 0, 0), (0, 0, 0, 0, 0), 0, '', 10, '', data_str

    data_dict = dict(zip(columns, row_bio_diet))

    st.title("📊 Rejestr BIO")

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🌍 Lokalizacja", "💓 Biometryka", "🍎 Dieta", "🧠 Odczucia",
        "💾 Zapis", "🤖 Predykcja"
    ])

    with tab1:
        st.header("🌍 Lokalizacja")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            city = st.text_input("Miejscowość", f"{data_dict['miejscowosc']}", key="city_input")
        with col2:
            lat = st.number_input(f"Szerokość geogr. (lat)", value=data_dict["lat"], format="%.8f", key="lat_input")
        with col3:
            lon = st.number_input("Długość geogr. (lon)", value=data_dict["lon"], format="%.8f", key="lon_input")
        with col4:
            selected_date = st.date_input("Data", value=data_dict["date"], key="main_date_input")

    with tab2:
        st.header("💓 Biometryka")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            hgh = st.number_input("HgH", min_value=0, max_value=220, value=data_dict["HgH"], key="hgh_input")
        with col2:
            hgl = st.number_input("HgL", min_value=0, max_value=180, value=data_dict["HgL"], key="hgl_input")
        with col3:
            sen = st.number_input("Sen (h)", min_value=0, max_value=24, value=int(data_dict["sen"]), key="sen_input")
        with col4:
            stres = st.slider("Stres (1-10)", 1, 10, data_dict["stres"], key="stres_input")

    with tab3:
        st.header("🍎 Dieta")
        cols1 = st.columns(5)
        dieta_labels1 = ["Mięso białe (%)", "Mięso czerwone (%)", "Nabiał (%)", "Słodycze (g)", "Alkohol (ml)"]
        dieta_val1 = [data_dict["mieso_biale"], data_dict["mieso_czerwone"], data_dict["nabial"], data_dict["slodycze"],
                      data_dict["alkohol"]]
        dieta_values1 = [cols1[i].number_input(dieta_labels1[i], min_value=0, max_value=100, value=dieta_val1[i],
                                               key=f"dieta1_{i}") for i in range(5)]

        cols2 = st.columns(5)
        dieta_labels2 = ["Warzywa i owoce (%)", "Przetworzone (%)", "Gluten (%)", "Orzechy (garść)", "Kawa (szt.)"]
        dieta_val2 = [data_dict["warzywa_owoce"], data_dict["przetworzone"], data_dict["gluten"], data_dict["orzechy"],
                      data_dict["kawa"]]
        dieta_values2 = [cols2[i].number_input(dieta_labels2[i], min_value=0, max_value=100, value=dieta_val2[i],
                                               key=f"dieta2_{i}") for i in range(5)]

    with tab4:
        st.header("🧠 Odczucia")
        col1, col2, col3 = st.columns(3)
        with col1:
            bol = st.slider("Ból (1-10)", 0, 10, data_dict["bol"], key="bol_input")
        with col2:
            godzina_bolu = st.number_input("Godzina bólu (0-23)", min_value=0, max_value=23,
                                           value=data_dict["godzina_bolu"], key="godzina_bolu_input")
        with col3:
            samopoczucie = st.slider("Samopoczucie (1-10)", 1, 10, data_dict["samopoczucie"], key="samopoczucie_input")

        uwagi = st.text_area("📝 Uwagi", data_dict["uwagi"], key="uwagi_input")
        data_str = date.today().strftime('%Y-%m-%d')

    with tab5:
        col1, col2 = st.columns(2)
        with col1:
            if st.button("💾 Zapisz dane", key="zapisz_btn"):
                st.success("Dane zapisane (przykładowo – tu dodamy zapis do bazy SQLite / Google Drive)")
        with col2:
            if st.button("🧹 Wyczyść dane", key="czysc_btn"):
                st.info("Dane wyczyszczone (tymczasowo – jeszcze bez logiki)")

        st.session_state.selected_date = selected_date

    with tab6:
        main_prediction_section()

    return city, lat, lon, hgh, hgl, sen, stres, dieta_values1, dieta_values2, bol, godzina_bolu, samopoczucie, uwagi, data_str


def main_app():
    """Główna aplikacja z wszystkimi funkcjami"""
    db_path = load_database()

    if not db_path or not os.path.exists("weather_data.db"):
        st.error("❌ Baza danych nie jest dostępna!")
        return

    tables, table_info, sample_data, available_dates = check_database_structure()

    if 'selected_date' not in st.session_state:
        st.session_state.selected_date = date.today()

    data = date.today()
    entry_from_base(data)

    st.header("🌦️ Dane pogodowe")

    if db_path and tables and any('weather' in str(t) for t in tables):
        try:
            columns, rows = get_weather_data_by_date(st.session_state.selected_date)

            if rows:
                df = pd.DataFrame(rows, columns=columns)
                st.dataframe(df, width='stretch')
                st.success(f"✅ Znaleziono {len(rows)} rekordów pogodowych")
            else:
                st.warning(f"⚠️ Brak danych pogodowych dla daty: {st.session_state.selected_date}")

                if available_dates:
                    st.info("📅 Dostępne daty w bazie:")
                    for date_row in available_dates:
                        st.write(f"   - {date_row[0]}")

        except Exception as e:
            st.error(f"❌ Błąd podczas ładowania danych pogodowych: {str(e)}")
    else:
        st.error("❌ Baza danych nie jest dostępna lub nie zawiera tabeli 'weather'")


# Uruchom główną aplikację
if __name__ == "__main__":
    main_app()