import pandas as pd
import numpy as np
import sqlite3
from sklearn.model_selection import learning_curve, train_test_split, KFold
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from xgboost import XGBRegressor
import lightgbm as lgb
import warnings
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import shap
import seaborn as sns
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pickle
import joblib


warnings.filterwarnings("ignore", category=UserWarning)
prediction_date_input = "2025-09-29"  # Domyślna data predykcji
DB_PATH = "weather_data.db"
print_importances = False
print_predict = True
print_krzywe_uczenia =False
print_shap = True
print_macierz_korelacji = False


def prepare_data(db_path):
    """
    Przygotowuje dane do trenowania modelu, wczytując je z bazy danych,
    łącząc tabele i tworząc nowe cechy.

    Args:
        db_path (str): Ścieżka do pliku bazy danych SQLite.

    Returns:
        tuple: Zawiera przygotowane ramki danych (X, Y) oraz listę użytych cech.
    """
    # Wczytanie danych
    conn = sqlite3.connect(db_path)
    max_to_train_data =  datetime.strptime(prediction_date_input, "%Y-%m-%d")
    end_to_train_data = (max_to_train_data - timedelta(days=1)).strftime('%Y-%m-%d')

    pd.set_option("display.max_rows", None)

    df_weather = pd.read_sql(f"SELECT * FROM weather WHERE date<='{end_to_train_data}'", conn)
    df_daily = pd.read_sql(f"SELECT * FROM daily_weather WHERE date<='{end_to_train_data}'", conn)
    print(f"SELECT * FROM weather WHERE date<='{end_to_train_data}'")
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


def tuning(X_train, y_train, param_grid, model_name="Random Forest"):
    print(f"DEBUG: TUNING MODELU {model_name}")
    print("Typy danych w X_train:\n", X_train.dtypes)
    train_sizes = np.linspace(0.3, 1.0, 5)
    print("train_sizes:", train_sizes)  # Powinno być np. [ 0.1, 0.325, 0.55, 0.775, 1.0 ]

    from sklearn.model_selection import GridSearchCV
    search_rf = GridSearchCV(
        estimator=RandomForestRegressor(random_state=42),
        param_grid=param_grid,
        cv=5,
        scoring='neg_mean_absolute_error',
        n_jobs=-1
    )
    search_rf.fit(X_train, y_train)

    print("Najlepsze parametry RF:", search_rf.best_params_)
    return search_rf.best_params_
    # Najlepsze parametry
    # RF: {'max_depth': 11, 'max_features': 0.9, 'min_samples_leaf': 1, 'n_estimators': 178}



def train_and_evaluate_models(X, Y):
    from sklearn.model_selection import StratifiedKFold, train_test_split
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from xgboost import XGBRegressor
    import lightgbm as lgb
    from sklearn.metrics import mean_absolute_error, r2_score
    from imblearn.over_sampling import RandomOverSampler
    import pandas as pd
    import numpy as np

    if X is not None and Y is not None:
        # 1️⃣ Podział na zbiór treningowy i testowy
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

        print("Zbiór treningowy:", X_train.shape, y_train.shape)
        print("Zbiór testowy   :", X_test.shape, y_test.shape, "\n")

        # 2️⃣ Oversampling dni z bólem – tworzymy binarną klasę 0/1 tylko do oversamplingu
        ros = RandomOverSampler(sampling_strategy=1.0, random_state=42)

        # 3️⃣ Definicja modeli
        # "Random Forest": RandomForestRegressor(n_estimators=178, max_depth=11, max_features=0.8,
        #                                        min_samples_leaf=1, random_state=42, n_jobs=-1)
        # param_grid_rf = {
        #     'n_estimators': [200, 210, 220],
        #     'max_depth': [None, 15, 20],  # Sprawdź czy None to optimum
        #     'min_samples_leaf': [4, 5, 6],
        #     'max_features': [0.65, 0.7, 0.75]
        # }
        # rf_params = tuning(X_train, y_train, param_grid_rf)
        # print (rf_params)

        models = {
            "Random Forest":
                # RandomForestRegressor(**rf_params, random_state=42),
                RandomForestRegressor(n_estimators=210, max_depth=None, max_features=0.75,
                                                   min_samples_leaf=4, random_state=42, n_jobs=-1),
            "Gradient Boosting": GradientBoostingRegressor(n_estimators=62, max_depth=5, learning_rate=0.1,
                                                           min_samples_leaf=10, subsample=0.8, random_state=42),
            "XGBoost": XGBRegressor(n_estimators=32, learning_rate=0.1, max_depth=5, subsample=0.8,
                                    colsample_bytree=0.8, random_state=42),
            "LightGBM": lgb.LGBMRegressor(n_estimators=42, learning_rate=0.1, num_leaves=20, max_depth=5,
                                          min_child_samples=10, subsample=0.8, colsample_bytree=0.8,
                                          random_state=42, verbose=-1)
        }


        trained_models = {}
        best_mae = float("inf")
        best_mae_pain = float("inf")
        best_model_name = None
        best_model_name_pain = None
        best_r2 = None

        y_test_arr = np.array(y_test)
        # Na potrzeby wykresu krzywych uczenia
        results = []

        # 4️⃣ Pętla trenowania modeli
        for name, model in models.items():
            print(f"--- Trenowanie modelu: {name} ---")

            # Oversampling na zbiorze treningowym
            y_train_binary = (y_train > 0).astype(int)
            X_res, y_res_bin = ros.fit_resample(X_train, y_train_binary)
            y_res = y_train.iloc[ros.sample_indices_].reset_index(drop=True)
            """
            y_train = [0, 0, 0, 2]     # 3x brak bólu, 1x ból=2
            y_train_binary = [0, 0, 0, 1]

            # Po oversamplingu (powiększaniu zbioru danych)
            ros.sample_indices_ = [0, 1, 2, 3, 3, 3]  
            # zawiera tylko indeksy do zbioru z wartością bólu
            y_res = y_train.iloc[ros.sample_indices_].reset_index(drop=True)
            Dzięki temu y_res zawiera wartości, a nie same indeksy  
            """

            # Fit
            model.fit(X_res, y_res)
            y_pred = model.predict(X_test)

            # Zapisanie modelu
            joblib.dump(models["Random Forest"], 'model.joblib')
            # Wczytaj
            loaded_model = joblib.load('model.joblib')

            # MAE i R² na zbiorze testowym
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            # MAE osobno dla bólu i bez bólu
            mask_pain = y_test_arr > 0
            mask_nopain = y_test_arr == 0

            mae_pain = mean_absolute_error(
                y_test_arr[mask_pain], y_pred[mask_pain]
            ) if mask_pain.sum() > 0 else np.nan
            mae_nopain = mean_absolute_error(
                y_test_arr[mask_nopain], y_pred[mask_nopain]
            ) if mask_nopain.sum() > 0 else np.nan
            ''' lub
            if mask_pain.sum() > 0:
              mae_pain = mean_absolute_error(y_test_arr[mask_pain], y_pred[mask_pain])
            else:
              mae_pain = np.nan
            '''
            # 5️⃣ Cross-validation z oversamplingiem w każdym foldzie
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            cv_scores = []

            for train_idx, val_idx in skf.split(X_train, y_train_binary):
                X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
                # print("\t\t\t\tZbiór treningowy:", X_tr.shape, y_tr.shape)
                # print("\t\t\t\tZbiór testowy   :", X_val.shape, y_val.shape, "\n")

                # oversampling tylko na foldzie treningowym
                y_tr_bin = (y_tr > 0).astype(int)
                X_tr_res, y_tr_res_bin = ros.fit_resample(X_tr, y_tr_bin)
                y_tr_res = y_tr.iloc[ros.sample_indices_].reset_index(drop=True)
                # print("\t\t\t\tZbiór treningowy over:", X_tr_res.shape, y_tr_res_bin.shape, y_tr_res)
                # print('')

                # model_cv = model.__class__(**model.get_params())
                # Tworzy nową instancję modelu z zerowymi wagami
                # w przeciwnym wypadku model po każdym treningu ma zmieniane wagi
                # natomiast każdy cross validation fold wymaga czystego startu (zerowe wagi cech)
                # model_cv=model, to tylko nowa referencja do obiektu (zajmującego miejsce w pamieci [ref. to adres w pam.])
                model_cv = model.__class__(**model.get_params())
                model_cv.fit(X_tr_res, y_tr_res)
                y_val_pred = model_cv.predict(X_val)

                cv_scores.append(r2_score(y_val, y_val_pred))

            print("Cross Średni R²:", np.mean(cv_scores))

            # Wyniki
            print(f"Wyniki dla {name}:")
            print(f"  MAE: {mae:.4f} | MAE_pain: {mae_pain:.4f} | MAE_nopain: {mae_nopain:.4f}")
            print(f"  R^2: {r2:.4f}\n")

            trained_models[name] = model
            print(f'name --> {name}')
            results.append({
                "model_name": name,
                "trained_model": model,
                "MAE": mae,
                "MAE_pain": mae_pain,
                "MAE_nopain": mae_nopain,
                "R2": r2,
                "CV_R2_mean": np.mean(cv_scores),
                "X_train": X_train,
                "y_train": y_train,
                "X_test": X_test,
                "y_test": y_test})

            # wybór najlepszego po MAE globalnym
            if mae < best_mae:
                best_mae = mae
                best_r2 = r2
                best_model_name = name

            # wybór najlepszego po MAE dla bólu
            if not np.isnan(mae_pain) and mae_pain < best_mae_pain:
                best_mae_pain = mae_pain
                best_model_name_pain = name

        print(f"Najlepszy model (MAE globalny): {best_model_name} z MAE = {best_mae:.4f}")
        print(f"Najlepszy model (MAE pain): {best_model_name_pain} z MAE_pain = {best_mae_pain:.4f}")

        # Na potrzeby wykresu krzywych uczenia
        results_df = pd.DataFrame(results)
        print(len(results_df))
        analiza_poziomu_predykcji(trained_models[best_model_name], X_test, y_train, y_test)

    return trained_models, best_model_name, best_r2, best_mae_pain, results_df

def show_feature_importances(model, feature_columns, top_n=10):
    """
    Wyświetla cechy o największym wpływie na predykcję.

    Args:
        model: Wytrenowany model drzewiasty (Random Forest, Gradient Boosting, XGBoost, LightGBM)
        feature_columns: Lista nazw cech użytych w modelu
        top_n: Ile top cech pokazać
    """
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        feat_imp_df = pd.DataFrame({
            'feature': feature_columns,
            'importance': importances
        })
        feat_imp_df.sort_values(by='importance', ascending=False, inplace=True)
        print(f"Top {top_n} cech według wpływu na przewidywany ból:")
        print(feat_imp_df.head(top_n))
        # Wykres słupkowy
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(10, 6))
        fig.canvas.manager.set_window_title('Wagi modelu Random Forest')
        plt.barh(feat_imp_df['feature'].head(top_n)[::-1], feat_imp_df['importance'].head(top_n)[::-1])
        plt.xlabel("Importance")
        plt.title("Top cechy według wpływu na przewidywany ból głowy")
        plt.tight_layout()
        plt.show()
    else:
        print("Model nie ma atrybutu feature_importances_ – nie jest modelem drzewiastym.")
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
        conn = sqlite3.connect(db_path)
        # Pobieramy dane pogodowe dla daty predykcji i 4 poprzednich dni
        start_date_weather = (prediction_date - timedelta(days=4)).strftime('%Y-%m-%d')
        end_date_weather = prediction_date.strftime('%Y-%m-%d')
        df_weather = pd.read_sql(
            f"SELECT * FROM weather WHERE date BETWEEN '{start_date_weather}' AND '{end_date_weather}' ORDER BY date ASC, hour ASC",
            conn)

        # Pobieramy dane dzienne dla daty predykcji i dwóch poprzednich dni
        start_date_daily = (prediction_date - timedelta(days=4)).strftime('%Y-%m-%d')
        end_date_daily = prediction_date.strftime('%Y-%m-%d')
        df_daily = pd.read_sql(
            f"SELECT * FROM daily_weather WHERE date BETWEEN '{start_date_daily}' AND '{end_date_daily}' ORDER BY date ASC",
            conn)

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
def plot_shap_summary(model, X_train, model_name="Model"):
    """
    Rysuje wykres SHAP summary (ważność cech) dla podanego modelu i danych treningowych.

    Parameters
    ----------
    model : trained model (np. RandomForestRegressor, LGBMRegressor, XGBRegressor)
    X_train : pandas.DataFrame
        Dane treningowe (bez y).
    model_name : str
        Nazwa modelu do podpisania wykresu.
    """
    # Tworzymy explainer
    try:
        X_train = model.get("X_train")   # jeśli nie ma w DF, może być None lub musisz podać ręcznie
        explainer = shap.Explainer(model, X_train)  # ogólny explainer
    except Exception:
        explainer = shap.TreeExplainer(model)       # fallback dla drzewiastych modeli

    # Obliczamy wartości SHAP
    shap_values = explainer(X_train)

    # Rysujemy podsumowanie
    fig = plt.figure(figsize=(10, 6))
    fig.canvas.manager.set_window_title('SHAP')
    plt.title(f"SHAP - {model_name}")
    fig.canvas.manager.set_window_title('Krzywe uczenia')
    shap.summary_plot(shap_values, X_train, show=True)

def plot_corelation(X_train, features):
    plt.figure(figsize=(16, 12))
    try:
        weather_features = [
            "temp", "pressure", "humidity", "wind_speed", "dew_point",
            "uvi_dew_temp", "temp_diff_mean", "pressure_diff_mean",
            "delta_temp_1h", "abs_delta_press_1h", "trend_temp", "trend_press",
            "temp_press_product", "temp_press_ratio", "delta_temp_3h", "delta_press_6h"
        ]

        corr_matrix = X_train[features].corr()

        sns.heatmap(
            corr_matrix,
            annot=True,
            cmap='coolwarm',
            center=0,
            ax=plt.gca()
        )

        plt.title("Macierz korelacji")
        plt.tight_layout()
        plt.show()

    except Exception as e:
        plt.text(0.5, 0.5, f"Błąd heatmap: {str(e)}", ha='center', color='red')
        plt.show()
def plot_shap_all_models(trained_models, data_models, max_display=10):
    """
    Rysuje SHAP barplot + summary_plot (dot) dla wszystkich modeli.

    trained_models : dict
        {"NazwaModelu": trained_model, ...}
    data_models : list of dict
        lista wyników z train_and_evaluate_models
    max_display : int
        Ile cech maksymalnie pokazać na wykresach
    """
    n_models = len(data_models)
    ncols = 2  # barplot i dot obok siebie
    nrows = n_models

    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 5 * nrows))
    fig.canvas.manager.set_window_title('SHAP wszystkich modeli')
    if n_models == 1:  # gdy tylko jeden model
        axes = [axes]

    for i, res in enumerate(data_models):
        name = res["model_name"]  # zmiana klucza
        model = trained_models[name]  # obiekt modelu
        X_train = res.get("X_train")  # jeśli nie ma w DF, może być None lub musisz podać ręcznie

        try:
            explainer = shap.Explainer(model, X_train)
        except Exception:
            explainer = shap.TreeExplainer(model)

        shap_values = explainer(X_train)

        # Barplot
        # plt.sca(axes[i][0] if n_models > 1 else axes[0])
        # shap.summary_plot(
        #     shap_values,
        #     X_train,
        #     plot_type="bar",
        #     show=False,
        #     max_display=max_display
        # )
        # axes[i][0].set_title(f"{name} - SHAP Barplot")

        # Dot summary
        plt.sca(axes[i][1] if n_models > 1 else axes[1])
        shap.summary_plot(
            shap_values,
            X_train,
            plot_type="dot",
            show=False,
            max_display=max_display
        )
        axes[i][1].set_title(f"{name} - SHAP Summary")

    plt.tight_layout()
    plt.show()
def plot_pain_prediction(predictions_output, prediction_date_str, best_r2, mae_pain, best_model):
    """
    Wyświetla wykres przewidywanego poziomu bólu głowy w poszczególnych godzinach,
    z uwzględnieniem zakresu błędu MAE.

    Args:
        predictions_output (pd.DataFrame): Ramka danych z predykcjami (kolumny 'hour', 'predicted_pain').
        prediction_date_input (str): Data predykcji w formacie 'YYYY-MM-DD'.
        best_mae (float): Wartość błędu średniego bezwzględnego (MAE) modelu.
    """
    if predictions_output is None or predictions_output.empty:
        print("Brak danych do wygenerowania wykresu.")
        return

    fig = plt.figure(figsize=(10, 6))
    fig.canvas.manager.set_window_title(f'Predykcja {best_model}')
    plt.plot(predictions_output["hour"], predictions_output["predicted_pain"], marker="o", linestyle="-",
             label="Przewidywany ból")

    # Dodanie linii błędu MAE
    plt.plot(predictions_output["hour"], predictions_output["predicted_pain"] - mae_pain, linestyle=":", color="gray",
             label=f"MAE_PAIN: {mae_pain:.4f}, R2: {best_r2:.4f}")
    plt.plot(predictions_output["hour"], predictions_output["predicted_pain"] + mae_pain, linestyle=":", color="gray",
             label="Przewidywany ból + MAE_PAIN")
    plt.plot(predictions_output["hour"], predictions_output["predicted_pain"] - mae_pain, linestyle=":", color="gray",
             label="Przewidywany ból - MAE_PAIN")

    plt.title(f"Przewidywany poziom bólu głowy na {prediction_date_str}")
    plt.xlabel("Godzina")
    plt.ylabel("Przewidywany poziom bólu")
    plt.xticks(range(0, 24))
    plt.grid(True)
    plt.ylim(bottom=0)
    plt.legend()
    plt.tight_layout()
    plot_filename = f"headache_prediction_{prediction_date_str}.png"
    plt.savefig(plot_filename)
    plt.show()
    print(f"Wykres zapisano jako: {plot_filename}")
def plot_learning_curves_all(models_results_df, cv=5):
    """
    Rysuje krzywe uczenia wszystkich modeli na jednym wykresie.

    Args:
        models_results_df (pd.DataFrame): DataFrame zwrócony przez train_and_evaluate_models
        cv (int): Liczba foldów w cross-validation
    """
    fig = plt.figure(figsize=(12, 7))
    fig.canvas.manager.set_window_title('Krzywe uczenia')
    # Konwersja DataFrame na listę słowników
    models_results = models_results_df.to_dict('records')

    for res in models_results:
        model = res["trained_model"]
        X_train = res["X_train"]
        y_train = res["y_train"]
        name = res["model_name"]

        print(f'name {name}')
        # print(len(data_models))

        train_sizes, train_scores, val_scores = learning_curve(
            model, X_train, y_train,
            cv=cv, scoring='r2', n_jobs=-1,
            train_sizes=np.linspace(0.1, 1.0, 10)
        )

        train_mean = -np.mean(train_scores, axis=1)
        val_mean = -np.mean(val_scores, axis=1)

        plt.plot(train_sizes, train_mean, '--', label=f"{name} Train")
        plt.plot(train_sizes, val_mean, '-', label=f"{name} Validation")

    plt.title("Learning Curves for All Models")
    plt.xlabel("Training Samples")
    plt.ylabel("R² score")
    plt.grid(True)
    plt.legend(loc="best")
    plt.show()

def analiza_poziomu_predykcji(model, X_test, y_train, y_test):
    # Zakładając, że masz już podział na zbiory:
    # X_train, X_test, y_train, y_test

    # 1. Sprawdź rozkład w zbiorze testowym (powinien być niezbilansowany jak rzeczywistość)
    print("Rozkład bólu w zbiorze TESTowym:")
    print(y_test.value_counts().sort_index())
    print(f"Proporcja dni z bólem: {((y_test > 0).sum() / len(y_test)):.2%}")

    # 2. Sprawdź rozkład w zbiorze treningowym (powinien być zbalansowany)
    print("\nRozkład bólu w zbiorze TRENINGowym:")
    print(y_train.value_counts().sort_index())
    print(f"Proporcja dni z bólem: {((y_train > 0).sum() / len(y_train)):.2%}")

    # 3. Wykonaj predykcje na zbiorze testowym
    y_pred = model.predict(X_test)

    # 4. POPRAWKA: Zdefiniuj próg dla klasyfikacji binarnej
    # Ponieważ ból jest w skali 1-10, musimy zdefiniować co uważamy za "ból"
    pain_threshold_true = 5  # lub wyższe wartości: 3, 5 - dostosuj do potrzeb
    y_true_binary = (y_test >= pain_threshold_true).astype(int)

    print(f"\nKlasyfikacja binarna - próg rzeczywisty: ból >= {pain_threshold_true}")
    print(f"Dni z bólem: {y_true_binary.sum()}/{len(y_true_binary)} ({y_true_binary.mean():.2%})")

    # 5. Analiza dla różnych progów PREDYKCJI (teraz w poprawnej skali)
    thresholds = [0.5, 1.0, 2.0, 3.0, 4.0]  # wartości w skali 1-10, tak jak y_pred
    results = []

    print("\n" + "=" * 60)
    print("ANALIZA DLA RÓŻNYCH PROGÓW PREDYKCJI")
    print("=" * 60)

    for threshold in thresholds:
        # POPRAWKA: y_pred są w skali 1-10, więc porównujemy z threshold w tej samej skali
        y_pred_binary = (y_pred >= threshold).astype(int)

        precision = precision_score(y_true_binary, y_pred_binary, zero_division=0)
        recall = recall_score(y_true_binary, y_pred_binary, zero_division=0)
        f1 = f1_score(y_true_binary, y_pred_binary, zero_division=0)

        # Macierz pomyłek
        cm = confusion_matrix(y_true_binary, y_pred_binary)

        results.append({
            'threshold': threshold,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'TP': cm[1, 1],  # Prawdziwe pozytywy
            'FP': cm[0, 1],  # Fałszywe pozytywy
            'FN': cm[1, 0],  # Fałszywe negatywy
            'TN': cm[0, 0]  # Prawdziwe negatywy
        })

        print(f"\nThreshold {threshold}:")
        print(f"  Precision: {precision:.3f} | Recall: {recall:.3f} | F1: {f1:.3f}")
        print(f"  Macierz pomyłek: TP={cm[1, 1]}, FP={cm[0, 1]}, FN={cm[1, 0]}, TN={cm[0, 0]}")

    # 6. Znajdź najlepszy próg (najwyższy F1)
    best_result = max(results, key=lambda x: x['f1'])
    print(f"\n🎯 NAJLEPSZY PRÓG: {best_result['threshold']}")
    print(f"   F1-score: {best_result['f1']:.3f}")
    print(f"   Precision: {best_result['precision']:.3f}")
    print(f"   Recall: {best_result['recall']:.3f}")

    # 7. Wizualizacja
    results_df = pd.DataFrame(results)
    plt.figure(figsize=(10, 6))
    plt.plot(results_df['threshold'], results_df['precision'], marker='o', label='Precision')
    plt.plot(results_df['threshold'], results_df['recall'], marker='o', label='Recall')
    plt.plot(results_df['threshold'], results_df['f1'], marker='o', label='F1-score')
    plt.axvline(best_result['threshold'], color='red', linestyle='--', alpha=0.7,
                label=f'Najlepszy próg ({best_result["threshold"]})')
    plt.xlabel('Próg klasyfikacji (skala 1-10)')
    plt.ylabel('Wartość metryki')
    plt.title(f'Metryki klasyfikacji - ból rzeczywisty >= {pain_threshold_true}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    # 8. Szczegółowy raport dla najlepszego progu
    print("\n" + "=" * 60)
    print("SZCZEGÓŁOWY RAPORT DLA NAJLEPSZEGO PROGU")
    print("=" * 60)

    best_threshold = best_result['threshold']
    y_pred_best = (y_pred >= best_threshold).astype(int)

    print(classification_report(y_true_binary, y_pred_best,
                                target_names=['Brak bólu', 'Ból głowy']))

    # 9. Macierz pomyłek dla najlepszego progu
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_true_binary, y_pred_best)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Przewidziano brak', 'Przewidziano ból'],
                yticklabels=['Rzeczywisty brak', 'Rzeczywisty ból'])
    plt.title(f'Macierz pomyłek - próg predykcji {best_threshold}')
    plt.show()

def plot_korelacja_z_bolem (X,Y):
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    # policz korelację Y z każdą cechą X
    corr_y_x = pd.DataFrame({
        "Feature": X.columns,
        "Correlation_with_Y": [Y.corr(X[col]) for col in X.columns]
    })

    # posortuj wg wartości bezwzględnej korelacji
    corr_y_x = corr_y_x.reindex(corr_y_x.Correlation_with_Y.abs().sort_values(ascending=False).index)

    print(corr_y_x)

    # --- wykres słupkowy ---
    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=corr_y_x,
        x="Correlation_with_Y",
        y="Feature",
        palette="coolwarm",
        hue="Correlation_with_Y",  # kolor zależny od wartości
        dodge=False
    )
    plt.axvline(0, color="black", linewidth=1)
    plt.title("Korelacja cech X z Y")
    plt.xlabel("Współczynnik korelacji")
    plt.ylabel("Cecha")
    plt.legend([], [], frameon=False)  # usuń legendę (niepotrzebna duplikacja)
    plt.tight_layout()
    plt.show()

def przewidywanie(data_pred,
                    print_importances = False,
                    print_predict = True,
                    print_krzywe_uczenia = False,
                    print_shap = False,
                    print_macierz_korelacji = False):

    warnings.filterwarnings("ignore", category=UserWarning)
    #prediction_date_input = "2025-09-29"  # Domyślna data predykcji
    DB_PATH = "weather_data.db"
    print_importances = print_importances
    print_predict = print_predict
    print_krzywe_uczenia = print_krzywe_uczenia
    print_shap = print_shap
    print_macierz_korelacji = print_macierz_korelacji

    if not data_pred is None:
        print (data_pred)
        prediction_date_input = data_pred

        # Krok 1: Przygotowanie danych
        X, Y, features = prepare_data(DB_PATH)

        if X is not None and Y is not None:
            # Krok 2: Trenowanie i ocena modeli
            trained_models, best_model, best_r2, best_mae_pain, data_models = train_and_evaluate_models(X, Y)

            # Krok 3: Wybór najlepszego modelu (np. na podstawie najniższego MAE)
            best_model = trained_models.get("Random Forest")

            if best_model:
                # Krok 4: Predykcja dla określonej daty (np. jutro)
                if print_importances:
                    show_feature_importances(best_model, features, top_n=15)

                if print_predict:
                    print(f'data pred: {prediction_date_input}')
                    predictions_output = predict_for_date(best_model, DB_PATH, features, prediction_date_input)
                    if predictions_output is not None:
                        print(f"\n--- Prognoza bólu głowy na {prediction_date_input} ---")
                        # print(predictions_output)
                        plot_pain_prediction(predictions_output, prediction_date_input, best_r2, best_mae_pain, best_model)

                if print_krzywe_uczenia:
                    print('Krzywe uczenia')
                    plot_learning_curves_all(data_models, cv=5)

                if print_shap:
                    print('SHAP')
                    data_models_list = data_models.to_dict(orient='records')
                    X_train = next(res["X_train"] for res in data_models_list if res["model_name"] == "Random Forest")
                    plot_shap_summary(best_model, X_train, model_name="Forest Regres")
                    # Przykład predykcji dla innej daty (np. 2025-09-10)
                    # prediction_date_input_past = "2025-09-10"
                    # predictions_output_past = predict_for_date(best_model, DB_PATH, features, pred

                if print_macierz_korelacji:
                    print('Korelacja')
                    data_models_list = data_models.to_dict(orient='records')
                    X_train = next(res["X_train"] for res in data_models_list if res["model_name"] == "Random Forest")

                    # policz korelacje
                    corr_matrix = X_train[features].corr()

                    # wybierz tylko wartości większe niż 0.6 (z wyłączeniem diagonalnych == 1.0)
                    high_corr = corr_matrix[(corr_matrix.abs() > 0.6) & (corr_matrix.abs() < 1.0)]

                    # „stopnij” macierz do listy par (feature1, feature2, korelacja)
                    high_corr_pairs = (
                        high_corr.stack()  # spłaszczenie macierzy
                        .reset_index()
                        .rename(columns={0: "correlation", "level_0": "feature1", "level_1": "feature2"})
                    )

                    print(high_corr_pairs)

                    # Wykres
                    plot_corelation(X_train, features)
                    plot_korelacja_z_bolem(X, Y)

def przewidywanie_bez_uczenia(data_pred):
    loaded_model = joblib.load('model.joblib')
    DB_PATH = "weather_data.db"
    if not data_pred is None:
        print(data_pred)
        prediction_date_input = data_pred
        # Krok 1: Przygotowanie danych
        X, Y, features = prepare_data(DB_PATH)

        if X is not None and Y is not None:
            # loaded_model (X,Y)
            print(f'data pred: {prediction_date_input}')
            predictions_output = predict_for_date(loaded_model, DB_PATH, features, prediction_date_input)
            if predictions_output is not None:
                print(f"\n--- Prognoza bólu głowy na {prediction_date_input} ---")
                # print(predictions_output)
                plot_pain_prediction(predictions_output, prediction_date_input, 0, 0, loaded_model)
if __name__ == "__main__":
    przewidywanie(None)

