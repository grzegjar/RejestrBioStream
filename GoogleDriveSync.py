import os
import json
from datetime import datetime
from tkinter import simpledialog, messagebox
import webbrowser
import requests
import base64
from dotenv import load_dotenv
'''
Zrób tylko to:
GoogleCloud->Menu główne->Bibloteka
📚 Biblioteka
Wyszukaj "Google Drive API"
Kliknij "WŁĄCZ" - jeśli status nie jest włączony
W googleDriveAPI karta Dane uwierzytelniające - > Utwórz Dane logowania
🔑 Dane logowania
Kliknij "+ UTWÓRZ DANE LOGOWANIA"
zapisz plik json - poniższe dane możeżesz odczytać bezpośrdenio z pliku
Wybierz "Identyfikator klienta OAuth"
Typ aplikacji: "Aplikacja komputerowa"
Nazwa: "Weather-App"

self.get_autorization_token_z_google()                      # Otwiera przeglądarkę
auth_code = input("Wklej kod: ")                            # Użytkownik wpisuje kod
refresh_token = self.get_refresh_token_z_google(auth_code)  # Zamienia na refresh
self.safe_token('logowanie.json', refresh_token)            # Zapisuje

# 2. UŻYCIE (robi się ZAWSZE)
self.get_data_json('logowanie.json')                        # Wczytuje refresh_token
self.get_access_token()                                     # 👈 ZDOBYWA access_token z refresh_token
self.upload_to_drive()                                      # 👈 UŻYWA access_token do uploadu
'''


class DriveAutoSync:
    def __init__(self, root):
        self.root = root
        load_dotenv()
        print (f"[DEBUG] {os.getenv('GOOGLE_REFRESH_TOKEN')}")
        self.refresh_token = os.getenv('GOOGLE_REFRESH_TOKEN')
        self.client_id =  os.getenv('GOOGLE_CLIENT_ID')
        self.client_secret = os.getenv('GOOGLE_CLIENT_SECRET')
        self.access_token = os.getenv('GOOGLE_REFRESH_TOKEN')
        self.local_db = "weather_data.db"
        self.get_dane_logowania()

    def check_token_scopes(self):
        """Sprawdź zakres uprawnień tokenu"""
        if self.access_token:
            try:
                # Dekoduj token JWT aby zobaczyć zakres
                parts = self.access_token.split('.')
                if len(parts) > 1:
                    payload = parts[1]
                    # Dodaj padding jeśli potrzeba
                    padding = 4 - len(payload) % 4
                    if padding != 4:
                        payload += '=' * padding
                    decoded = json.loads(base64.b64decode(payload))
                    scopes = decoded.get('scope', 'Nieznany')
                    print(f"🔐 Zakresy tokenu: {scopes}")

                    # Sprawdź czy ma uprawnienia do Drive
                    if 'drive' in scopes or 'https://www.googleapis.com/auth/drive' in scopes:
                        print("✅ Token ma uprawnienia do Google Drive")
                    else:
                        print("❌ Token NIE ma uprawnień do Google Drive ")

            except Exception as e:
                print(f"ℹ️ Nie można zdekodować tokenu: {e}")

    def debug_folders(self):
        """Diagnostyka folderów - sprawdź co naprawdę widzi API"""
        headers = {'Authorization': f'Bearer {self.access_token}'}

        print("🔍 DIAGNOSTYKA FOLDERÓW:")

        # 1. Sprawdź wszystkie foldery
        response = requests.get(
            "https://www.googleapis.com/drive/v3/files?q=mimeType='application/vnd.google-apps.folder'&fields=files(id,name,parents)",
            headers=headers
        )

        if response.status_code == 200:
            all_folders = response.json().get('files', [])
            print(f"📂 Znaleziono {len(all_folders)} folderów w sumie:")
            for folder in all_folders:
                print(f"   - '{folder['name']}' (ID: {folder['id']})")
        else:
            print(f"❌ Błąd pobierania folderów: {response.status_code}")
            return

        # 2. Sprawdź konkretnie folder 'Dane'
        print(f"\n🔍 Szukam konkretnie folderu 'Dane':")
        query = f"name='Dane' and mimeType='application/vnd.google-apps.folder' and trashed=false"
        response = requests.get(
            f"https://www.googleapis.com/drive/v3/files?q={requests.utils.quote(query)}&fields=files(id,name,parents)",
            headers=headers
        )

        if response.status_code == 200:
            dane_folders = response.json().get('files', [])
            print(f"📋 Zapytanie: '{query}'")
            print(f"📋 Znaleziono {len(dane_folders)} folderów 'Dane':")
            for folder in dane_folders:
                print(f"   - ID: {folder['id']}, Parents: {folder.get('parents', [])}")
        else:
            print(f"❌ Błąd wyszukiwania folderu 'Dane': {response.status_code}")
    def get_access_token(self):
        """Uzyskaj access token używając refresh token"""
        try:
            url = "https://oauth2.googleapis.com/token"
            data = {
                'client_id': self.client_id,
                'client_secret': self.client_secret,
                'refresh_token': self.refresh_token,
                'grant_type': 'refresh_token'
            }

            response = requests.post(url, data=data)
            print(f"Status: {response.status_code}")
            print(f"Odpowiedź: {response.text}")  # 👈 DODAJ TE LINIE

            if response.status_code == 200:
                self.access_token = response.json()['access_token']
                print("✅ Token dostępu uzyskany")
                return True
            else:
                print(f"❌ Błąd uzyskiwania tokena: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Błąd: {e}")
            return False
    def find_file_on_drive(self, filename="weather_data.db"):
        """Znajdź plik na Google Drive - ZACHOWANE DO PÓŹNIEJSZEGO WYKORZYSTANIA"""
        if not self.access_token:
            if not self.get_access_token():
                return None

        url = "https://www.googleapis.com/drive/v3/files"
        headers = {'Authorization': f'Bearer {self.access_token}'}
        params = {
            'q': f"name='{filename}' and trashed=false",
            'fields': 'files(id, name, modifiedTime)'
        }

        response = requests.get(url, headers=headers, params=params)
        if response.status_code == 200:
            files = response.json().get('files', [])
            if files:
                print(f"✅ Znaleziono plik na Drive: {files[0]['name']}")
                return files[0]
            else:
                print("ℹ️ Plik nie istnieje na Drive")
                return None
        else:
            print(f"❌ Błąd wyszukiwania: {response.status_code}")
            return None

    def upload_to_drive(self, folder_name="Dane"):
        """UPLOAD NA ŻYCZENIE - zapisuje w folderze 'Dane'"""
        print("🔄 Rozpoczynam upload bazy danych na Google Drive...")

        if not os.path.exists(self.local_db):
            print(f"❌ Błąd: Plik {self.local_db} nie istnieje!")
            return False

        # Sprawdź rozmiar pliku
        file_size = os.path.getsize(self.local_db)
        print(f"📁 Rozmiar pliku: {file_size / 1024:.2f} KB")

        if not self.access_token:
            if not self.get_access_token():
                return False

        # 1. Znajdź lub utwórz folder "Dane"
        folder_id = self.find_or_create_folder(folder_name)
        print(f"📂 Pracuję z folderem ID: {folder_id}")
        if not folder_id:
            print(f"❌ Nie można znaleźć lub utworzyć folderu '{folder_name}'")
            return False

        # 2. Znajdź plik w folderze "Dane"
        file_info = self.find_file_on_drive_in_folder(folder_id)

        try:
            # Wczytaj plik
            with open(self.local_db, 'rb') as f:
                file_content = f.read()

            if file_info:
                # Aktualizuj istniejący plik
                print("📤 Aktualizuję istniejący plik na Drive...")
                url = f"https://www.googleapis.com/upload/drive/v3/files/{file_info['id']}"
                params = {'uploadType': 'media'}
                headers = {
                    'Authorization': f'Bearer {self.access_token}',
                    'Content-Type': 'application/octet-stream'
                }

                response = requests.patch(url, headers=headers, params=params, data=file_content)
            else:
                # Utwórz nowy plik w folderze "Dane"
                print("📤 Tworzę nowy plik w folderze 'Dane'...")
                url = "https://www.googleapis.com/upload/drive/v3/files?uploadType=multipart"

                metadata = {
                    'name': 'weather_data.db',
                    'mimeType': 'application/x-sqlite3',
                    'parents': [folder_id]
                }

                # ✅ Ręczne zbudowanie multipart/related
                boundary = "===============7330845974216740156=="
                metadata_str = json.dumps(metadata)

                body = (
                           f"--{boundary}\r\n"
                           "Content-Type: application/json; charset=UTF-8\r\n\r\n"
                           f"{metadata_str}\r\n"
                           f"--{boundary}\r\n"
                           "Content-Type: application/octet-stream\r\n\r\n"
                       ).encode("utf-8") + file_content + f"\r\n--{boundary}--".encode("utf-8")

                headers = {
                    'Authorization': f'Bearer {self.access_token}',
                    'Content-Type': f'multipart/related; boundary={boundary}'
                }

                response = requests.post(url, headers=headers, data=body)

            if response.status_code in [200, 201]:
                print(f"✅ Baza danych została pomyślnie przesłana do folderu '{folder_name}' na Google Drive!")
                print(f"🕒 Czas: {datetime.now().strftime('%H:%M:%S')}")
                return True
            else:
                print(f"❌ Błąd uploadu: {response.status_code} - {response.text}")
                return False

        except Exception as e:
            print(f"❌ Błąd podczas uploadu: {e}")
            return False

    def find_folder(self, folder_name="Dane"):
        """Tylko szuka folderu, nie tworzy nowego - dla pobierania"""
        headers = {'Authorization': f'Bearer {self.access_token}'}

        query = f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder' and trashed=false"

        print(f"🔍 Szukam folderu '{folder_name}'...")

        response = requests.get(
            f"https://www.googleapis.com/drive/v3/files?q={requests.utils.quote(query)}&fields=files(id,name)",
            headers=headers
        )

        if response.status_code == 200:
            files = response.json().get('files', [])
            if files:
                folder_id = files[0]['id']
                print(f"✅ Znaleziono folder '{folder_name}' (ID: {folder_id})")
                return folder_id
            else:
                print(f"❌ Folder '{folder_name}' nie znaleziony")
                return None
        else:
            print(f"❌ Błąd wyszukiwania folderu: {response.status_code}")
            return None
    def find_or_create_folder(self, folder_name="Dane"):
        """Znajduje lub tworzy folder o podanej nazwie w katalogu głównym Drive."""
        headers = {'Authorization': f'Bearer {self.access_token}'}

        # Szukamy folderu o danej nazwie
        query = f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder' and trashed=false"

        print(f"🔍 Szukam folderu '{folder_name}'...")

        response = requests.get(
            f"https://www.googleapis.com/drive/v3/files?q={requests.utils.quote(query)}&fields=files(id, name)",
            headers=headers
        )

        if response.status_code == 200:
            files = response.json().get('files', [])
            print(f"📋 Znaleziono foldery: {len(files)}")

            if files:
                # Używamy pierwszego znalezionego folderu (nawet jeśli nie w root)
                folder_id = files[0]['id']
                print(f"✅ Używam istniejącego folderu '{folder_name}' (ID: {folder_id})")
                return folder_id
            else:
                # Tworzymy nowy folder w root
                print(f"📁 Tworzę nowy folder '{folder_name}' w root...")
                metadata = {
                    'name': folder_name,
                    'mimeType': 'application/vnd.google-apps.folder',
                    'parents': ['root']
                }
                create_resp = requests.post(
                    "https://www.googleapis.com/drive/v3/files",
                    headers={**headers, 'Content-Type': 'application/json'},
                    data=json.dumps(metadata)
                )
                if create_resp.status_code in [200, 201]:
                    folder_id = create_resp.json()['id']
                    print(f"✅ Utworzono folder '{folder_name}' (ID: {folder_id})")
                    return folder_id
                else:
                    print(f"❌ Błąd tworzenia folderu: {create_resp.text}")
                    return None
        else:
            print(f"❌ Błąd wyszukiwania folderu: {response.text}")
            return None
    def find_or_create_folder_p1(self, folder_name="Dane"):
        """Znajduje lub tworzy folder o podanej nazwie w katalogu głównym Drive."""
        headers = {'Authorization': f'Bearer {self.access_token}'}

        # Szukamy folderu tylko w root
        query = f"name = '{folder_name}' and mimeType = 'application/vnd.google-apps.folder' and 'root' in parents and trashed = false"

        response = requests.get(
            f"https://www.googleapis.com/drive/v3/files?q={requests.utils.quote(query)}&fields=files(id, name)",
            headers=headers
        )

        if response.status_code == 200:
            files = response.json().get('files', [])
            if files:
                # ✅ Zwracamy pierwszy znaleziony folder
                folder_id = files[0]['id']
                print(f"✅ Znaleziono folder '{folder_name}' (ID: {folder_id})")
                return folder_id
            else:
                # Tworzymy nowy folder w root
                metadata = {
                    'name': folder_name,
                    'mimeType': 'application/vnd.google-apps.folder',
                    'parents': ['root']
                }
                create_resp = requests.post(
                    "https://www.googleapis.com/drive/v3/files",
                    headers={**headers, 'Content-Type': 'application/json'},
                    data=json.dumps(metadata)
                )
                if create_resp.status_code in [200, 201]:
                    folder_id = create_resp.json()['id']
                    print(f"✅ Utworzono folder '{folder_name}' (ID: {folder_id})")
                    return folder_id
                else:
                    print(f"❌ Błąd tworzenia folderu: {create_resp.text}")
                    return None
        else:
            print(f"❌ Błąd wyszukiwania folderu: {response.text}")
            return None

    def find_file_on_drive_in_folder(self, folder_id, filename="weather_data.db"):
        """Znajdź plik w konkretnym folderze na Google Drive"""
        if not self.access_token:
            if not self.get_access_token():
                return None

        url = "https://www.googleapis.com/drive/v3/files"
        headers = {'Authorization': f'Bearer {self.access_token}'}

        # Szukaj pliku w konkretnym folderze
        query = f"name='{filename}' and '{folder_id}' in parents and trashed=false"
        params = {
            'q': query,
            'fields': 'files(id, name, modifiedTime)'
        }

        response = requests.get(url, headers=headers, params=params)
        if response.status_code == 200:
            files = response.json().get('files', [])
            if files:
                print(f"✅ Znaleziono plik w folderze: {files[0]['name']}")
                return files[0]
            else:
                print(f"ℹ️ Plik nie istnieje w folderze")
                return None
        else:
            print(f"❌ Błąd wyszukiwania: {response.status_code}")
            return None

    def download_from_drive_2(self, folder_name="Dane"):
        # SPRAWDŹ CZY BAZA JUŻ ISTNIEJE I JEST POPRAWNA
        if os.path.exists(self.local_db):
            try:
                conn = sqlite3.connect(self.local_db)
                cursor = conn.cursor()
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                tables = cursor.fetchall()
                conn.close()

                if tables:
                    print("💾 Używam istniejącej bazy danych")
                    return True
            except:
                pass  # Baza uszkodzona - kontynuuj pobieranie

        # TYLKO jeśli baza nie istnieje lub jest uszkodzona, pobierz ją
        print("📥 Pobieranie bazy danych z Google Drive...")
        """POBIERANIE NA ŻĄDANIE - z folderu 'Dane'"""
        print("📥 Rozpoczynam pobieranie bazy danych z Google Drive...")

        if not self.access_token:
            if not self.get_access_token():
                return False

        folder_id = self.find_folder(folder_name)
        if not folder_id:
            print(f"❌ Folder '{folder_name}' nie znaleziony na Drive")
            return False

        file_info = self.find_file_on_drive_in_folder(folder_id)
        if not file_info:
            print(f"❌ Plik nie znaleziony w folderze '{folder_name}' na Drive")
            return False

        try:
            print(f"💾 Pobieram plik: {file_info['name']}")
            print(f"💾 Zostanie zapisany jako: {self.local_db}")

            url = f"https://www.googleapis.com/drive/v3/files/{file_info['id']}?alt=media"
            headers = {'Authorization': f'Bearer {self.access_token}'}

            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                # BACKUP BEZ shutil - prostsza wersja
                if os.path.exists(self.local_db):
                    backup_name = f"{self.local_db}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    # Proste kopiowanie pliku
                    with open(self.local_db, 'rb') as source, open(backup_name, 'wb') as target:
                        target.write(source.read())
                    print(f"📦 Stworzono backup: {backup_name}")

                # Zapisz pobrany plik
                with open(self.local_db, 'wb') as f:
                    f.write(response.content)

                print(f"✅ Baza danych pobrana pomyślnie z folderu '{folder_name}' na Google Drive!")
                print(f"💾 Zapisano jako: {self.local_db}")
                print(f"🕒 Ostatnia modyfikacja na Drive: {file_info.get('modifiedTime', 'unknown')}")
                return True
            else:
                print(f"❌ Błąd pobierania: {response.status_code}")
                return False

        except Exception as e:
            print(f"❌ Błąd podczas pobierania: {e}")
            return False

    def download_from_drive(self):
        """Inteligentne pobieranie - tylko gdy baza jest nowsza/nie istnieje"""

        # Sprawdź czy plik już istnieje i jest poprawny
        if os.path.exists(self.local_db):
            try:
                conn = sqlite3.connect(self.local_db)
                cursor = conn.cursor()
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                tables = cursor.fetchall()
                conn.close()

                if tables:
                    print("✅ Używam istniejącej bazy danych")
                    return True
            except:
                pass  # Baza uszkodzona - pobierz nową

        # Jeśli dotarliśmy tutaj, trzeba pobrać bazę
        print("📥 Pobieranie bazy danych z Google Drive...")

        # Tutaj reszta Twojej istniejącej logiki pobierania...
        if not self.access_token:
            if not self.get_access_token():
                return False

        folder_id = self.find_folder("Dane")
        if not folder_id:
            return False

        file_info = self.find_file_on_drive_in_folder(folder_id)
        if not file_info:
            return False

        try:
            url = f"https://www.googleapis.com/drive/v3/files/{file_info['id']}?alt=media"
            headers = {'Authorization': f'Bearer {self.access_token}'}

            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                with open(self.local_db, 'wb') as f:
                    f.write(response.content)

                print("✅ Baza danych pobrana pomyślnie!")
                return True
            else:
                print(f"❌ Błąd pobierania: {response.status_code}")
                return False

        except Exception as e:
            print(f"❌ Błąd podczas pobierania: {e}")
            return False
    def check_drive_status(self, folder_name="Dane"):
        """Sprawdź status pliku na Drive - w folderze 'Dane'"""
        print("🔍 Sprawdzam status pliku na Google Drive...")

        if not self.access_token:
            if not self.get_access_token():
                return False

        # Znajdź folder
        folder_id = self.find_or_create_folder(folder_name)
        if not folder_id:
            print(f"❌ Folder '{folder_name}' nie istnieje na Google Drive")
            return False

        file_info = self.find_file_on_drive_in_folder(folder_id)
        if file_info:
            print(f"📊 Informacje o pliku w folderze '{folder_name}':")
            print(f"   📂 Folder: {folder_name}")
            print(f"   📄 Nazwa: {file_info['name']}")
            print(f"   🆔 ID: {file_info['id']}")
            print(f"   🕒 Ostatnia modyfikacja: {file_info.get('modifiedTime', 'unknown')}")
            return True
        else:
            print(f"ℹ️ Plik nie istnieje w folderze '{folder_name}' na Google Drive")
            return False

    ###################################################
    #                AUTORYZACJA GOOGLE
    #   1. POBIERANIE KODU AUTORYZACYJNEGO - TYMCZASOWEGO ZE STRONY GOOGLE PRZEZ LOGOWANIE
    #   2. GENEROWANIE TOKENA DŁUGOTRWAŁEGO
    #   3. ZACHOWANIE TOKENA
    # Funkcja do uzyskania refresh_token (JEDNORAZOWO)
    def get_autorization_token_z_google(self):
        redirect_uri = "urn:ietf:wg:oauth:2.0:oob"

        # Krok 1: Otwórz przeglądarkę i zdobądź kod
        auth_url = (
            f"https://accounts.google.com/o/oauth2/auth?"
            f"client_id={self.client_id}&"
            f"redirect_uri={redirect_uri}&"
            f"scope=https://www.googleapis.com/auth/drive.file&"
            f"response_type=code&"
            f"access_type=offline&"
            f"prompt=consent"
        )

        print("=== KONFIGURACJA JEDNORAZOWA ===")
        print("1. Otwórz ten URL w przeglądarce:")
        print(auth_url)
        webbrowser.open(auth_url)
    def get_refresh_token_z_google(self,authorization_code):
        redirect_uri = "urn:ietf:wg:oauth:2.0:oob"  # Dla aplikacji desktop

        # Zamień kod na refresh_token
        token_url = "https://oauth2.googleapis.com/token"
        data = {
            'client_id': self.client_id,
            'client_secret': self.client_secret,  # UŻYJ SWOJEGO client_secret!
            'code': authorization_code,  # Kod z przeglądarki (tymczasowy)
            'grant_type': 'authorization_code',  # Typ wymiany
            'redirect_uri': redirect_uri  # Adres zwrotny
        }

        response = requests.post(token_url, data=data)
        if response.status_code == 200:
            # 👇 Otrzymujesz:
            # - access_token (ważny 1 godzinę)
            # - refresh_token (ważny latami) ← TEN NAS INTERESUJE!
            tokens = response.json()
            refresh_token = tokens['refresh_token']

            # Zapisz token do pliku
            print(f"TOKEN: {refresh_token}")
            return refresh_token
        else:
            print(f"❌ Błąd: {response.text}")
            return None
    def safe_token(self, filename, refresh_token):
        # Aktualizuje refresh_token w pliku JSON, Tworzy strukturę jeśli nie istnieje
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            file_path = os.path.join(script_dir, filename)

            # Sprawdzenie czy plik istnieje i odczyt danych
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as file:
                    data = json.load(file)  # POPRAWIONE: json.load(file) nie json.load(file_path)
            else:
                data = {}

            # Utwórz strukturę jeśli nie istnieje
            if 'installed' not in data:
                data['installed'] = {}

            # POPRAWIONE: zawsze aktualizuj refresh_token, nie sprawdzaj czy jest None
            data['installed']['refresh_token'] = refresh_token  # POPRAWIONE: użyj parametru funkcji

            # Zapis danych
            with open(file_path, 'w', encoding='utf-8') as file:
                json.dump(data, file, indent=4, ensure_ascii=False)

            print("🎉 KONFIGURACJA ZAKOŃCZONA SUKCESEM!")
            print("=" * 50)
            print(f"🔑 Twój refresh_token: {refresh_token}")
            print("=" * 50)
            return True

        except Exception as e:
            print(f"✗ Błąd: {e}")
            return False
    def get_data_json(self, filename):
        # Pobieranie danych z pliku autoryzacji
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))  # script_dir = os.getcwd()
            file_path = os.path.join(script_dir, filename)
            file_path = rf'{file_path}'  # file_path = r"C:\Users\Administrator\PycharmProjects\RejestrBio\logowanie.json"

            if not os.path.exists(file_path):
                # print("❌ Plik nie istnieje!")
                return

            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)

            # print("Zawartość pliku JSON:")
            # print(json.dumps(data, indent=2))

            installed = data.get('installed', {})
            self.client_id = installed.get('client_id')
            self.client_secret = installed.get('client_secret')
            self.refresh_token = installed.get('refresh_token')

            if not self.client_id or not self.client_secret:
                return

        except Exception as e:
            print(f"✗ Błąd: {e}")
            return

    def get_dane_logowania(self):
        # 1️⃣ Wczytaj dane z logowanie.json
        self.get_data_json('logowanie.json')



        # Sprawdź czy plik zawiera client_id i client_secret
        if not self.client_id or not self.client_secret:
            print("❌ Brak client_id lub client_secret w pliku logowanie.json")
            return False

        # 2️⃣ Jeśli mamy refresh_token → uzyskaj access_token i zakończ
        if self.refresh_token:
            print(f"✅ Wykorzystuję zapisany refresh_token: {self.refresh_token}")
            self.get_access_token()  # odśwież access_token
            return True

        # 3️⃣ Brak refresh_token → uruchom proces pierwszej autoryzacji
        print("🚀 Pierwsza konfiguracja — wymagane logowanie do Google")
        self.get_autorization_token_z_google()

        authorization_code = simpledialog.askstring(
            "Kod autoryzacyjny",
            "Wprowadź kod autoryzacyjny ze strony Google (ważny 60 sekund):",
            parent=self.root
        )

        if not authorization_code:
            print("❌ Anulowano proces autoryzacji")
            return False

        # 4️⃣ Wymień kod autoryzacyjny na refresh_token
        refresh_token = self.get_refresh_token_z_google(authorization_code)

        if not refresh_token:
            print("❌ Nie udało się uzyskać refresh_token")
            return False

        # 5️⃣ Zapisz refresh_token do logowanie.json
        confirm = messagebox.askyesno("Potwierdzenie", "Czy chcesz zapisać token w pliku?")
        if confirm:
            self.safe_token('logowanie.json', refresh_token)
            self.refresh_token = refresh_token
            print("✅ Refresh token zapisany i gotowy do użycia")
            return True
        else:
            print("❌ Token nie został zapisany")
            return False
