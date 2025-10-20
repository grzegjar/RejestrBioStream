import sqlite3
from config import DB_FILE


def init_db():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    # Tabela z zagregowanymi danymi dziennymi i bio
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS daily_weather (
            date TEXT PRIMARY KEY,
            lat REAL,
            lon REAL,
            max_temp REAL,
            min_temp REAL,
            max_pressure REAL,
            min_pressure REAL,
            delta_temp REAL,
            delta_pressure REAL,
            avg_wind_speed REAL,
            sen REAL,
            stres INTEGER,
            HgL INTEGER,
            HgH INTEGER,
            miejscowosc TEXT, 
            bol INTEGER,
            samopoczucie INTEGER,
            slodycze INTEGER, 
            nabial INTEGER, 
            mieso_biale INTEGER, 
            mieso_czerwone INTEGER, 
            alkohol INTEGER, 
            kawa INTEGER, 
            przetworzone INTEGER, 
            warzywa_owoce INTEGER, 
            gluten INTEGER, 
            orzechy INTEGER, 
            uwagi TEXT,
            godzina_bolu INTEGER
        )
    ''')
    try:
        print(' ')
        # cursor.execute("ALTER TABLE daily_weather ADD COLUMN godzina_bolu INTEGER")
    except sqlite3.OperationalError:
        pass  # kolumna już istnieje

    cursor.execute("PRAGMA table_info(weather);")
    columns = [row[1] for row in cursor.fetchall()]
    # print(columns)
    # Tabela ze szczegółowymi danymi godzinowymi
    # Dodaj poziom_bolu i inne
    cursor.execute('''
            CREATE TABLE IF NOT EXISTS weather (
                date TEXT,
                hour INTEGER,
                lat REAL,
                lon REAL,
                temp REAL,
                feels_like REAL,
                pressure INTEGER,
                humidity INTEGER,
                dew_point REAL,
                uvi REAL,
                clouds INTEGER,
                visibility INTEGER,
                wind_speed REAL,
                wind_deg INTEGER,
                wind_gust REAL,
                rain REAL,
                snow REAL,
                weather_id INTEGER,
                weather_main TEXT,
                weather_description TEXT,
                weather_icon TEXT,
                czy_bol INTEGER,
                poziom_bolu,
                PRIMARY KEY (date, hour)
            )
        ''')
    try:
        cursor.execute("ALTER TABLE weather ADD COLUMN poziom_bolu INTEGER;")
        print("ALTER TABLE weather ADD COLUMN poziom_bolu INTEGER ")
    except sqlite3.OperationalError:
        pass  # Kolumna już istnieje

    # Czyszczenie starych, nieprawidłowych danych
    # cursor.execute("UPDATE daily_weather SET godzina_bolu = NULL WHERE godzina_bolu = 'None'")
    # cursor.execute("UPDATE daily_weather SET uwagi = NULL WHERE uwagi = 'None'")

    conn.commit()
    conn.close()


def get_weather_data_by_date(date_filter):
    conn = sqlite3.connect(DB_FILE)
    print (f"PLIK BAZY {DB_FILE}")
    query = '''
        SELECT * FROM weather WHERE date = ? ORDER BY hour
    '''
    cursor = conn.cursor()
    cursor.execute(query, (date_filter,))
    rows = cursor.fetchall()
    columns = [description[0] for description in cursor.description]
    conn.close()
    return columns, rows


def get_all_data():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute('''
        SELECT date, miejscowosc, max_temp, min_temp, max_pressure, min_pressure, avg_wind_speed,
               sen, stres, HgH, HgL, bol, samopoczucie,
               slodycze, nabial, mieso_biale, mieso_czerwone, alkohol, kawa, przetworzone, warzywa_owoce, gluten, orzechy, uwagi, lat, lon, godzina_bolu 
        FROM daily_weather ORDER BY date DESC
    ''')
    rows = cursor.fetchall()
    conn.close()
    return rows


def get_stored_data(date_str):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM daily_weather WHERE date = ?", (date_str,))
    row = cursor.fetchone()
    conn.close()
    return row


def save_data_to_db(date_str, miejscowosc, lat, lon, stats, sen=None, stres=None, HgH=None, HgL=None,
                    bol=None, samopoczucie=None, godzina_bolu=None,
                    diet_values=None, uwagi=None):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    delta_temp = stats["max_temp"] - stats["min_temp"]
    delta_pressure = stats["max_pressure"] - stats["min_pressure"]

    if diet_values is None:
        diet_values = [None] * 10

    cursor.execute('''
        INSERT OR REPLACE INTO daily_weather 
        (date, miejscowosc, lat, lon, max_temp, min_temp, max_pressure, min_pressure, avg_wind_speed,
         delta_temp, delta_pressure,
         sen, stres, HgH, HgL, bol, samopoczucie, godzina_bolu,
         mieso_biale, mieso_czerwone, nabial, slodycze, alkohol, 
         warzywa_owoce, przetworzone, gluten, orzechy, kawa, uwagi)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
    date_str, miejscowosc, lat, lon, stats["max_temp"], stats["min_temp"], stats["max_pressure"], stats["min_pressure"],
    stats["avg_wind_speed"],
    delta_temp, delta_pressure,
    sen, stres, HgH, HgL, bol, samopoczucie, godzina_bolu,
    diet_values[0], diet_values[1], diet_values[2], diet_values[3], diet_values[4],
    diet_values[5], diet_values[6], diet_values[7], diet_values[8], diet_values[9], uwagi))
    conn.commit()
    conn.close()


def update_user_data(date_str, miejscowosc, lat, lon, sen, stres, HgH, HgL, bol, samopoczucie, godzina_bolu,
                     diet_values, uwagi):
    print('[update_user_data]')
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute('''
        UPDATE daily_weather 
        SET miejscowosc=?, lat=?, lon=?, sen=?, stres=?, HgH=?, HgL=?, bol=?, samopoczucie=?, godzina_bolu=?,
            mieso_biale=?, mieso_czerwone=?, nabial=?, slodycze=?, alkohol=?, 
            warzywa_owoce=?, przetworzone=?, gluten=?, orzechy=?, kawa=?, uwagi=?
        WHERE date=?
    ''', (miejscowosc, lat, lon, sen, stres, HgH, HgL, bol, samopoczucie, godzina_bolu,
          diet_values[0], diet_values[1], diet_values[2], diet_values[3], diet_values[4],
          diet_values[5], diet_values[6], diet_values[7], diet_values[8], diet_values[9], uwagi, date_str))
    conn.commit()
    conn.close()


def delete_data(date_str):
    print('[delete_data]')
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM daily_weather WHERE date = ?", (date_str,))
    conn.commit()
    conn.close()


def save_hourly_weather(date_str, hour, lat, lon, hour_data):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    weather = hour_data.get("weather", [{}])[0]
    print('[save_hourly_weather]')
    cursor.execute('''
        INSERT OR REPLACE INTO weather (
            date, hour, lat, lon, temp, feels_like, pressure, humidity, dew_point,
            uvi, clouds, visibility, wind_speed, wind_deg, wind_gust,
            rain, snow, weather_id, weather_main, weather_description, weather_icon
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        date_str, hour, lat, lon,
        hour_data.get("temp"), hour_data.get("feels_like"), hour_data.get("pressure"),
        hour_data.get("humidity"), hour_data.get("dew_point"), hour_data.get("uvi"),
        hour_data.get("clouds"), hour_data.get("visibility"), hour_data.get("wind_speed"),
        hour_data.get("wind_deg"), hour_data.get("wind_gust"),
        hour_data.get("rain", {}).get("1h"), hour_data.get("snow", {}).get("1h"),
        weather.get("id"), weather.get("main"), weather.get("description"), weather.get("icon")
    ))
    cursor.execute("UPDATE weather SET poziom_bolu = 0 WHERE poziom_bolu is Null")
    cursor.execute("UPDATE weather SET czy_bol = 0 WHERE czy_bol is Null")
    conn.commit()
    conn.close()


def update_pain_level_for_hours(date_str, hours_list, pain_level, zero_out_other_hours=False):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    print('[update_pain_level_for_hours]')
    cursor.execute("UPDATE weather SET poziom_bolu = 0 WHERE poziom_bolu is Null")
    cursor.execute("UPDATE weather SET czy_bol = 0 WHERE czy_bol is Null")
    if zero_out_other_hours:
        # Zeruj poziom bólu dla wszystkich godzin w danym dniu
        cursor.execute("UPDATE weather SET czy_bol=0, poziom_bolu = 0 WHERE date = ?", (date_str,))
        print(f'"UPDATE weather SET czy_bol=0, poziom_bolu = 0 WHERE date = ?", ({date_str},)')
    # Ustaw poziom bólu dla określonych godzin
    for hour in hours_list:
        if pain_level != 0:
            cursor.execute("UPDATE weather SET czy_bol=1, poziom_bolu = ? WHERE date = ? AND hour = ?",
                           (pain_level, date_str, hour))

    conn.commit()
    conn.close()