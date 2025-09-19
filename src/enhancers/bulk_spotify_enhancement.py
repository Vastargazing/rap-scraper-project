#!/usr/bin/env python3
"""
🎵 Улучшенное массовое обогащение базы данных метаданными из Spotify API

НАЗНАЧЕНИЕ:
- Массовое обогащение песен и артистов метаданными и аудио-фичами из Spotify
- Поддержка retry/backoff, фильтрации, fallback стратегий, статистики ошибок
- Возможность продолжения с чекпоинтов

ИСПОЛЬЗОВАНИЕ:
python src/enhancers/bulk_spotify_enhancement.py

ЗАВИСИМОСТИ:
- Python 3.8+
- requests, dotenv
- src/enhancers/spotify_enhancer.py
- .env с SPOTIFY_CLIENT_ID/SECRET

РЕЗУЛЬТАТ:
- Массовое обновление базы Spotify-метаданными
- Логи, статистика, поддержка восстановления

АВТОР: AI Assistant
ДАТА: Сентябрь 2025
"""
import os
import sys
import time
import json
import sqlite3
import requests
import ssl
from datetime import datetime
from dotenv import load_dotenv
from typing import Optional, List, Dict, Any, Tuple
import logging
import re
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter

try:
    from src.enhancers.spotify_enhancer import SpotifyEnhancer
except Exception:
    from spotify_enhancer import SpotifyEnhancer

load_dotenv()

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/spotify_enhancement_improved.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ImprovedSpotifyEnhancer(SpotifyEnhancer):
    """Улучшенная версия SpotifyEnhancer с robust retry-механизмом"""
    
    def __init__(self, client_id: str = None, client_secret: str = None, db_path: str = None):
        super().__init__(client_id, client_secret, db_path)
        
        # Настройка robust HTTP session с SSL конфигурацией
        self.session = requests.Session()
        
        # Retry стратегия с экспоненциальным backoff
        retry_strategy = Retry(
            total=5,  # Общее количество попыток
            backoff_factor=2,  # Экспоненциальный backoff (1, 2, 4, 8, 16 сек)
            status_forcelist=[429, 500, 502, 503, 504],  # HTTP коды для retry
            allowed_methods=["GET"],  # Только GET запросы
            raise_on_status=False
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        # SSL и connection pool конфигурация
        import ssl
        import urllib3
        
        # Отключаем warnings для SSL если есть проблемы
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        
        # Увеличиваем timeout
        self.default_timeout = 30
        
        # Увеличиваем паузы между запросами
        self.requests_per_second = 3  # Еще более консервативный лимит
        
        # Добавляем статистику стратегий поиска
        self.stats = {
            'strategies_used': {
                'exact': 0,
                'basic': 0,
                'no_feat': 0,
                'keywords': 0,
                'simple': 0,
                'track_only': 0,
                'alt_artist': 0,
                'not_found': 0
            }
        }
        
        print("🔧 Улучшенный SpotifyEnhancer инициализирован с robust retry-механизмом")
    
    def _recreate_session(self):
        """Пересоздание HTTP session для решения SSL проблем"""
        try:
            self.session.close()
        except:
            pass
        
        # Создаем новую session
        self.session = requests.Session()
        
        # Retry стратегия
        retry_strategy = Retry(
            total=5,
            backoff_factor=2,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"],
            raise_on_status=False
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
    
    def _make_request(self, endpoint: str, params: Dict = None) -> Optional[Dict]:
        """Улучшенный метод запросов с robust retry-механизмом"""
        if not self.get_access_token():
            return None
        
        # Rate limiting с увеличенными паузами
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        min_interval = 1.0 / self.requests_per_second
        
        if time_since_last < min_interval:
            sleep_time = min_interval - time_since_last
            time.sleep(sleep_time)
        
        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json"
        }
        
        url = f"{self.base_url}/{endpoint}"
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                response = self.session.get(
                    url, 
                    headers=headers, 
                    params=params, 
                    timeout=self.default_timeout
                )
                
                self.last_request_time = time.time()
                self.api_calls_count += 1
                
                if response.status_code == 200:
                    return response.json()
                    
                elif response.status_code == 429:
                    # Rate limit exceeded
                    retry_after = int(response.headers.get('Retry-After', 60))
                    logger.warning(f"Rate limit exceeded, ждем {retry_after} секунд")
                    time.sleep(retry_after + 5)  # +5 сек буфера
                    continue  # Попробуем еще раз
                    
                elif response.status_code == 401:
                    # Token expired
                    logger.info("Token истек, обновляем...")
                    self.access_token = None
                    if self.get_access_token():
                        headers["Authorization"] = f"Bearer {self.access_token}"
                        continue  # Попробуем еще раз
                    return None
                    
                else:
                    logger.error(f"API ошибка: {response.status_code}")
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt)  # Экспоненциальный backoff
                        continue
                    return None
                    
            except (requests.exceptions.SSLError, 
                    requests.exceptions.ConnectionError,
                    requests.exceptions.Timeout,
                    requests.exceptions.ProxyError,
                    ConnectionResetError,
                    ssl.SSLEOFError) as e:
                
                logger.warning(f"Сетевая ошибка (попытка {attempt + 1}/{max_retries}): {type(e).__name__}: {str(e)[:100]}")
                
                if attempt < max_retries - 1:
                    # Экспоненциальный backoff для сетевых ошибок
                    backoff_time = (2 ** attempt) + 5  # +5 сек базовая пауза
                    logger.info(f"Ждем {backoff_time} секунд перед повтором...")
                    time.sleep(backoff_time)
                    
                    # Для SSL ошибок пересоздаем session
                    if isinstance(e, (requests.exceptions.SSLError, ssl.SSLEOFError)):
                        logger.info("Пересоздаем HTTP session из-за SSL ошибки...")
                        self._recreate_session()
                    
                    continue
                else:
                    logger.error(f"Все попытки исчерпаны для запроса: {url}")
                    return None
                    
            except Exception as e:
                logger.error(f"Неожиданная ошибка: {e}")
                return None
        
        return None
    
    def search_track_improved(self, track_name: str, artist_name: str) -> Optional[Any]:
        """Улучшенный поиск трека с множественными fallback стратегиями"""
        
        print(f"      🎯 Начинаем поиск: {track_name} - {artist_name}")
        
        # Проверяем доступность API
        if not self.get_access_token():
            print(f"      ❌ Не удалось получить access token")
            return None
        
        # Очищаем название трека от лишних символов
        clean_track = self._clean_track_name(track_name)
        clean_artist = self._clean_artist_name(artist_name)
        
        print(f"      🧹 После очистки: {clean_track} - {clean_artist}")
        
        # Стратегия 1: точный поиск
        print(f"      1️⃣ Пробуем стратегию 'exact'...")
        result = self._try_search_strategy(clean_track, clean_artist, "exact")
        if result:
            logger.debug(f"Найдено стратегией 'exact': {track_name} - {artist_name}")
            self.stats['strategies_used']['exact'] += 1
            print(f"      ✅ Найдено стратегией 'exact'!")
            return result
        
        print(f"      🔄 Exact не сработал, пробуем другие стратегии...")
        
        # Стратегия 2: базовое название (без скобок, версий)
        basic_track = self._get_basic_track_name(clean_track)
        if basic_track != clean_track:
            print(f"      2️⃣ Пробуем стратегию 'basic': {basic_track}")
            result = self._try_search_strategy(basic_track, clean_artist, "basic")
            if result:
                logger.debug(f"Найдено стратегией 'basic': {track_name} - {artist_name}")
                self.stats['strategies_used']['basic'] += 1
                print(f"      ✅ Найдено стратегией 'basic'!")
                return result
        
        # Стратегия 3: убираем feat/ft из названия
        no_feat_track = re.sub(r'\s+(feat\.|ft\.|featuring).*$', '', basic_track, flags=re.IGNORECASE)
        if no_feat_track != basic_track:
            print(f"      3️⃣ Пробуем стратегию 'no_feat': {no_feat_track}")
            result = self._try_search_strategy(no_feat_track, clean_artist, "no_feat")
            if result:
                logger.debug(f"Найдено стратегией 'no_feat': {track_name} - {artist_name}")
                self.stats['strategies_used']['no_feat'] += 1
                print(f"      ✅ Найдено стратегией 'no_feat'!")
                return result
        
        # Стратегия 4: поиск только по артисту + ключевые слова
        keywords = self._extract_keywords(no_feat_track)
        if keywords:
            print(f"      4️⃣ Пробуем стратегию 'keywords': {keywords}")
            result = self._try_search_strategy(keywords, clean_artist, "keywords")
            if result:
                logger.debug(f"Найдено стратегией 'keywords': {track_name} - {artist_name}")
                self.stats['strategies_used']['keywords'] += 1
                print(f"      ✅ Найдено стратегией 'keywords'!")
                return result
        
        # Стратегия 5: простой поиск без кавычек
        print(f"      5️⃣ Пробуем стратегию 'simple'...")
        result = self._try_search_strategy(no_feat_track, clean_artist, "simple")
        if result:
            logger.debug(f"Найдено стратегией 'simple': {track_name} - {artist_name}")
            self.stats['strategies_used']['simple'] += 1
            print(f"      ✅ Найдено стратегией 'simple'!")
            return result
        
        # Ничего не найдено
        print(f"      ❌ Трек не найден ни одной стратегией")
        self.stats['strategies_used']['not_found'] += 1
        
        return None
    
    def _try_search_strategy(self, track_name: str, artist_name: str, strategy: str) -> Optional[Any]:
        """Попытка поиска с определенной стратегией"""
        try:
            if strategy == "exact":
                query = f'track:"{track_name}" artist:"{artist_name}"'
            elif strategy == "basic":
                query = f'track:"{track_name}" artist:"{artist_name}"'
            elif strategy == "no_feat":
                query = f'track:"{track_name}" artist:"{artist_name}"'
            elif strategy == "keywords":
                query = f'"{track_name}" artist:"{artist_name}"'
            elif strategy == "simple":
                query = f'{track_name} {artist_name}'
            elif strategy == "track_only":
                query = f'track:"{track_name}"'
            elif strategy == "alt_artist":
                query = f'track:"{track_name}" artist:"{artist_name}"'
            else:
                query = f'{track_name} {artist_name}'
            
            params = {
                "q": query,
                "type": "track",
                "limit": 10  # Увеличиваем лимит для лучшего matching
            }
            
            print(f"        📡 Делаем запрос: {query[:100]}...")
            
            data = self._make_request("search", params)
            if not data:
                print(f"        ❌ Нет ответа от API")
                return None
                
            if "tracks" not in data:
                print(f"        ❌ Нет треков в ответе API")
                return None
            
            tracks = data["tracks"]["items"]
            if not tracks:
                print(f"        ❌ Пустой список треков")
                return None
            
            print(f"        📦 Получили {len(tracks)} треков, ищем лучший match...")
            
            # Ищем лучший match по названию и артисту
            best_match = self._find_best_match(tracks, track_name, artist_name, strategy)
            if best_match:
                print(f"        ✅ Найден match: {best_match.get('name', 'Unknown')} by {best_match.get('artists', [{}])[0].get('name', 'Unknown')}")
                return self._create_spotify_track(best_match)
            else:
                print(f"        ❌ Не найден подходящий match среди {len(tracks)} треков")
                
        except Exception as e:
            print(f"        ❌ Ошибка в стратегии {strategy}: {e}")
            logger.debug(f"Ошибка в стратегии {strategy}: {e}")
            
        return None
    
    def _clean_track_name(self, track_name: str) -> str:
        """Очистка названия трека с агрессивной нормализацией"""
        if not track_name:
            return ""
        
        # Убираем лишние пробелы
        clean = " ".join(track_name.split())
        
        # Заменяем специальные символы
        replacements = {
            ''': "'",
            ''': "'", 
            '"': '"',
            '"': '"',
            '–': '-',
            '—': '-',
            '…': '...',
            '&': 'and',
            '@': 'at',
            '$': 's',
            '4': 'for',
            '2': 'to',
            'u': 'you',
            'ur': 'your',
            'w/': 'with',
            'w': 'with',
            'n': 'and',
            'b4': 'before',
            'luv': 'love',
            'thru': 'through'
        }
        
        for old, new in replacements.items():
            clean = clean.replace(old, new)
        
        # Удаляем распространенные префиксы/суффиксы
        clean = re.sub(r'^(official\s+)?(music\s+)?(video\s+)?', '', clean, flags=re.IGNORECASE)
        clean = re.sub(r'\s+(official|music|video|audio|lyric|lyrics)(\s+video|\s+audio)?$', '', clean, flags=re.IGNORECASE)
        
        # Удаляем теги качества/источника
        clean = re.sub(r'\s*\((hq|hd|4k|official|clean|explicit|radio\s+edit)\)\s*', ' ', clean, flags=re.IGNORECASE)
        clean = re.sub(r'\s*\[(hq|hd|4k|official|clean|explicit|radio\s+edit)\]\s*', ' ', clean, flags=re.IGNORECASE)
        
        return " ".join(clean.split()).strip()
    
    def _clean_artist_name(self, artist_name: str) -> str:
        """Очистка имени артиста"""
        if not artist_name:
            return ""
        
        clean = self._clean_track_name(artist_name)
        
        # Стандартизируем некоторые имена
        standardizations = {
            "A$AP": "ASAP",
            "A$AP Rocky": "ASAP Rocky",
            "A$AP Ferg": "ASAP Ferg",
            "XXXTentacion": "XXXTENTACION"
        }
        
        for old, new in standardizations.items():
            if old in clean:
                clean = clean.replace(old, new)
        
        return clean
    
    def _get_basic_track_name(self, track_name: str) -> str:
        """Получение базового названия трека с агрессивной очисткой"""
        
        # Более агрессивные паттерны для удаления
        patterns_to_remove = [
            # Версии и ремиксы
            r'\(.*?demo.*?\)',  # (Demo), (Demo 1)
            r'\(.*?remix.*?\)', # (Remix)
            r'\(.*?mix.*?\)',   # (Mixed)
            r'\(.*?original.*?\)', # (Original)
            r'\(.*?version.*?\)', # (Version)
            r'\(.*?edit.*?\)',  # (Edit)
            r'\(.*?remaster.*?\)', # (Remastered)
            r'\[.*?v\d+.*?\]',  # [V1], [V2]
            r'\[.*?original.*?\]', # [Original]
            
            # Качество и источники
            r'\(.*?(hq|hd|4k|320|mp3|flac|wav).*?\)',
            r'\[.*?(hq|hd|4k|320|mp3|flac|wav).*?\]',
            r'\(.*?(official|clean|explicit|radio).*?\)',
            r'\[.*?(official|clean|explicit|radio).*?\]',
            
            # Фичеринги в скобках (оставляем в основном тексте)
            r'\(.*?feat.*?\)',
            r'\(.*?ft\..*?\)',
            r'\[.*?feat.*?\]',
            r'\[.*?ft\..*?\]',
            
            # Теги и метки
            r'\*+\s*$',        # Звездочки в конце
            r'\s*\(\s*\)\s*',  # Пустые скобки
            r'\s*\[\s*\]\s*',  # Пустые квадратные скобки
            r'\s*-\s*$',       # Дефис в конце
            r'^\s*-\s*',       # Дефис в начале
            
            # Годы
            r'\(19\d{2}\)',    # (1990-1999)
            r'\(20\d{2}\)',    # (2000-2099)
            r'\[19\d{2}\]',    # [1990-1999]
            r'\[20\d{2}\]',    # [2000-2099]
        ]
        
        basic = track_name
        for pattern in patterns_to_remove:
            basic = re.sub(pattern, '', basic, flags=re.IGNORECASE)
        
        # Очистка feat/ft в основном тексте (но сохраняем имена)
        basic = re.sub(r'\s+(feat\.|ft\.|featuring)\s+', ' feat ', basic, flags=re.IGNORECASE)
        
        # Убираем лишние пробелы и знаки препинания
        basic = re.sub(r'\s*[,;]\s*$', '', basic)  # Запятые и точки с запятой в конце
        basic = " ".join(basic.split())
        
        return basic.strip()
    
    def _extract_keywords(self, track_name: str) -> str:
        """Извлечение ключевых слов из названия трека"""
        
        # Убираем стоп-слова
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
            'of', 'with', 'by', 'feat', 'ft', 'featuring'
        }
        
        # Разбиваем на слова
        words = re.findall(r'\b\w+\b', track_name.lower())
        
        # Фильтруем стоп-слова и короткие слова
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        
        # Берем первые 3 ключевых слова
        return " ".join(keywords[:3])
    
    def _get_alternative_artist_name(self, artist_name: str) -> str:
        """Получение альтернативного имени артиста для поиска"""
        
        # Убираем общие суффиксы/префиксы
        alt_name = artist_name
        
        # Убираем "Lil", "Big", "Young", "DJ" и т.д.
        prefixes_to_remove = [
            r'^(lil\s+|big\s+|young\s+|dj\s+|mc\s+|the\s+)',
            r'^(lil|big|young|dj|mc)\s*',
        ]
        
        for prefix in prefixes_to_remove:
            alt_name = re.sub(prefix, '', alt_name, flags=re.IGNORECASE).strip()
        
        # Заменяем $ на S в именах типа A$AP
        alt_name = alt_name.replace('$', 'S')
        
        # Убираем скобки с дополнительной информацией
        alt_name = re.sub(r'\s*\([^)]*\)\s*', ' ', alt_name)
        
        # Стандартизируем пробелы
        alt_name = " ".join(alt_name.split())
        
        return alt_name if alt_name != artist_name else artist_name
    
    def _find_best_match(self, tracks: List[Dict], target_track: str, target_artist: str, strategy: str = "exact") -> Optional[Dict]:
        """Поиск лучшего совпадения среди найденных треков"""
        
        def similarity_score(track_data: Dict) -> float:
            """Оценка схожести трека"""
            score = 0
            
            track_name = track_data.get("name", "").lower()
            track_artists = [artist.get("name", "").lower() for artist in track_data.get("artists", [])]
            
            target_track_lower = target_track.lower()
            target_artist_lower = target_artist.lower()
            
            # Проверяем совпадение артиста
            artist_match = any(target_artist_lower in artist or artist in target_artist_lower 
                             for artist in track_artists)
            if artist_match:
                score += 50
            
            # Проверяем совпадение названия трека
            if target_track_lower in track_name or track_name in target_track_lower:
                score += 30
            
            # Бонус за популярность
            popularity = track_data.get("popularity", 0)
            score += popularity * 0.2
            
            return score
        
        if not tracks:
            return None
        
        # Находим трек с лучшим score
        best_track = max(tracks, key=similarity_score)
        best_score = similarity_score(best_track)
        
        # Устанавливаем минимальный порог
        if best_score >= 30:  # Минимальный приемлемый score
            return best_track
        
        return None
    
    def _create_spotify_track(self, track_data: Dict) -> Any:
        """Создание объекта SpotifyTrack из данных API"""
        try:
            from src.models.models import SpotifyTrack
        except ImportError:
            # Fallback для альтернативных путей импорта
            try:
                from ..models.models import SpotifyTrack
            except ImportError:
                # Создаем простой объект если модель недоступна
                class SimpleSpotifyTrack:
                    def __init__(self, **kwargs):
                        for key, value in kwargs.items():
                            setattr(self, key, value)
                SpotifyTrack = SimpleSpotifyTrack
        
        return SpotifyTrack(
            spotify_id=track_data["id"],
            name=track_data["name"],
            artist_id=track_data["artists"][0]["id"],
            album_name=track_data.get("album", {}).get("name"),
            release_date=track_data.get("album", {}).get("release_date"),
            duration_ms=track_data.get("duration_ms"),
            popularity=track_data.get("popularity", 0),
            explicit=track_data.get("explicit", False),
            spotify_url=track_data["external_urls"]["spotify"],
            preview_url=track_data.get("preview_url")
        )


class ImprovedBulkSpotifyEnhancement:
    """Улучшенный класс для массового обогащения"""
    
    def __init__(self, enhancer: ImprovedSpotifyEnhancer):
        self.enhancer = enhancer
        self.stats = {
            'tracks_processed': 0,
            'tracks_success': 0,
            'tracks_failed': 0,
            'total_api_calls': 0,
            'start_time': None,
            'errors': {},
            'strategies_used': {
                'exact': 0,
                'basic': 0,
                'no_feat': 0,
                'keywords': 0,
                'simple': 0,
                'track_only': 0,
                'alt_artist': 0,
                'not_found': 0
            }
        }
        # Сохраняем checkpoint в папку results 
        self.checkpoint_file = "results/spotify_enhancement_checkpoint.json"
    
    def enhance_all_tracks_improved(self, start_from: int = 0, batch_size: int = 1000):
        """Улучшенное массовое обогащение с checkpoint'ами"""
        print("🎵 УЛУЧШЕННОЕ МАССОВОЕ ОБОГАЩЕНИЕ ТРЕКОВ")
        print("=" * 60)
        
        # Загружаем checkpoint если есть
        if start_from == 0:
            start_from = self._load_checkpoint()
        
        conn = sqlite3.connect(self.enhancer.db_path)
        cursor = conn.cursor()
        
        # Получаем треки начиная с checkpoint
        cursor.execute("""
            SELECT s.id, s.title, s.artist 
            FROM songs s
            LEFT JOIN spotify_tracks st ON s.id = st.song_id
            WHERE st.song_id IS NULL AND s.id >= ?
            ORDER BY s.id
            LIMIT ?
        """, (start_from, batch_size))
        
        tracks = cursor.fetchall()
        conn.close()
        
        if not tracks:
            print(f"🎉 Все треки уже обработаны!")
            return
        
        print(f"📋 Обрабатываем треки с ID {tracks[0][0]} до {tracks[-1][0]}")
        print(f"📊 Всего треков в батче: {len(tracks)}")
        
        self.stats['start_time'] = datetime.now()
        
        try:
            for i, (song_id, title, artist) in enumerate(tracks, 1):
                
                # Проверяем прерывание пользователем
                try:
                    # Показываем прогресс каждые 10 треков или первые 5
                    if i <= 5 or i % 10 == 0:
                        self._show_progress(i, len(tracks), song_id)
                        self._save_checkpoint(song_id)
                    
                    # Показываем что обрабатываем (для диагностики)
                    print(f"[{i}/{len(tracks)}] 🔍 Обрабатываем: {artist} - {title}")
                    
                    # Обрабатываем трек
                    result = self._process_track(song_id, title, artist)
                    
                    if result['success']:
                        self.stats['tracks_success'] += 1
                        print(f"[{song_id}] ✅ Найдено на Spotify")
                    else:
                        self.stats['tracks_failed'] += 1
                        error_type = result.get('error_type', 'unknown')
                        print(f"[{song_id}] ❌ Не найдено ({error_type})")
                        self.stats['errors'][error_type] = self.stats['errors'].get(error_type, 0) + 1
                        
                        if i % 100 == 0:  # Показываем ошибки еще реже
                            print(f"[{song_id}] ❌ {artist} - {title} ({error_type})")
                    
                    self.stats['tracks_processed'] += 1
                    self.stats['total_api_calls'] += result.get('api_calls', 0)
                    
                    # Адаптивная пауза на основе успешности
                    success_rate = self.stats['tracks_success'] / max(self.stats['tracks_processed'], 1)
                    if success_rate < 0.1:  # Если успешность < 10%
                        time.sleep(0.3)  # Увеличиваем паузу
                    else:
                        time.sleep(0.15)  # Обычная пауза
                        
                except KeyboardInterrupt:
                    print(f"\n⏹️ Получен сигнал прерывания при обработке трека {i}")
                    current_id = song_id
                    self._save_checkpoint(current_id)
                    raise  # Пробрасываем KeyboardInterrupt выше
                
        except KeyboardInterrupt:
            print(f"\n⏹️ Обработка остановлена пользователем")
            if 'i' in locals() and i > 0:
                current_id = tracks[i-1][0] if i > 0 else start_from
                self._save_checkpoint(current_id)
            else:
                print(f"⚠️ Прерывание произошло до начала обработки")
        except Exception as e:
            print(f"\n❌ Неожиданная ошибка: {e}")
            logger.error(f"Неожиданная ошибка в основном цикле: {e}")
            if 'i' in locals() and i > 0:
                current_id = tracks[i-1][0] if i > 0 else start_from
                self._save_checkpoint(current_id)
        
        self._show_final_stats()
        self._clear_checkpoint()
    
    def _process_track(self, song_id: int, title: str, artist: str) -> Dict[str, Any]:
        """Обработка одного трека с улучшенной логикой"""
        
        start_time = time.time()
        api_calls_start = self.enhancer.api_calls_count
        
        try:
            # Используем стандартный поиск
            track_data = self.enhancer.search_track(title, artist)
            
            api_calls_used = self.enhancer.api_calls_count - api_calls_start
            
            if track_data:
                # Сохраняем в базу
                self._save_track_to_db(song_id, track_data)
                
                return {
                    'success': True,
                    'api_calls': api_calls_used,
                    'processing_time': time.time() - start_time
                }
            else:
                return {
                    'success': False,
                    'error_type': 'not_found',
                    'api_calls': api_calls_used,
                    'processing_time': time.time() - start_time
                }
                
        except Exception as e:
            error_type = 'network_error' if 'connection' in str(e).lower() else 'unknown_error'
            
            return {
                'success': False,
                'error_type': error_type,
                'error_message': str(e),
                'api_calls': self.enhancer.api_calls_count - api_calls_start,
                'processing_time': time.time() - start_time
            }
    
    def _save_track_to_db(self, song_id: int, track):
        """Сохранение трека в базу данных"""
        try:
            conn = sqlite3.connect(self.enhancer.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT OR REPLACE INTO spotify_tracks 
                (song_id, spotify_id, artist_spotify_id, album_name, 
                 release_date, duration_ms, popularity, explicit, 
                 spotify_url, preview_url)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                song_id,
                track.spotify_id,
                track.artist_id,
                track.album_name,
                track.release_date,
                track.duration_ms,
                track.popularity,
                track.explicit,
                track.spotify_url,
                track.preview_url
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Ошибка сохранения трека {song_id}: {e}")
    
    def _save_checkpoint(self, current_id: int):
        """Сохранение checkpoint для возможности продолжения"""
        try:
            # Копируем stats и конвертируем datetime в строку
            stats_copy = self.stats.copy()
            if 'start_time' in stats_copy and stats_copy['start_time']:
                if isinstance(stats_copy['start_time'], datetime):
                    stats_copy['start_time'] = stats_copy['start_time'].isoformat()
            
            checkpoint_data = {
                'last_processed_id': current_id,
                'timestamp': datetime.now().isoformat(),
                'stats': stats_copy
            }
            
            # Убедимся, что директория существует
            os.makedirs(os.path.dirname(self.checkpoint_file), exist_ok=True)
            
            with open(self.checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump(checkpoint_data, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            logger.error(f"Ошибка сохранения checkpoint: {e}")
            # Выводим подробности ошибки для отладки
            import traceback
            logger.debug(f"Traceback: {traceback.format_exc()}")
    
    def _load_checkpoint(self) -> int:
        """Загрузка checkpoint для продолжения работы"""
        try:
            if os.path.exists(self.checkpoint_file):
                with open(self.checkpoint_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                last_id = data.get('last_processed_id', 0)
                if last_id > 0:
                    print(f"📁 Найден checkpoint: продолжаем с ID {last_id}")
                    return last_id
                    
        except Exception as e:
            logger.error(f"Ошибка загрузки checkpoint: {e}")
        
        return 0
    
    def _clear_checkpoint(self):
        """Очистка checkpoint после завершения"""
        try:
            if os.path.exists(self.checkpoint_file):
                os.remove(self.checkpoint_file)
                print("📁 Checkpoint очищен")
        except Exception as e:
            logger.error(f"Ошибка очистки checkpoint: {e}")
    
    def _show_progress(self, current: int, total: int, song_id: int):
        """Показать прогресс обработки"""
        print(f"\n🔄 ПРОГРЕСС: {current}/{total} треков в батче (ID: {song_id})")
        print(f"✅ Успешно: {self.stats['tracks_success']}")
        print(f"❌ Ошибок: {self.stats['tracks_failed']}")
        print(f"🌐 API вызовов: {self.stats['total_api_calls']}")
        
        if self.stats['tracks_processed'] > 0:
            success_rate = self.stats['tracks_success'] / self.stats['tracks_processed'] * 100
            print(f"📊 Успешность: {success_rate:.1f}%")
    
    def _show_final_stats(self):
        """Показать финальную статистику"""
        print("\n" + "="*60)
        print("📈 ФИНАЛЬНАЯ СТАТИСТИКА")
        print("="*60)
        
        if self.stats['start_time']:
            elapsed = datetime.now() - self.stats['start_time']
            print(f"⏱️ Время работы: {elapsed}")
        
        print(f"🎵 Треки:")
        print(f"   • Обработано: {self.stats['tracks_processed']}")
        print(f"   • Успешно: {self.stats['tracks_success']}")
        print(f"   • Ошибки: {self.stats['tracks_failed']}")
        
        if self.stats['tracks_processed'] > 0:
            success_rate = self.stats['tracks_success'] / self.stats['tracks_processed'] * 100
            print(f"   • Успешность: {success_rate:.1f}%")
        
        print(f"📡 Всего API вызовов: {self.stats['total_api_calls']}")
        
        # Статистика ошибок
        if self.stats['errors']:
            print(f"\n❌ Типы ошибок:")
            for error_type, count in self.stats['errors'].items():
                print(f"   • {error_type}: {count}")
        
        # Обновленная статистика базы
        db_stats = self.enhancer.get_stats()
        print(f"\n📊 СТАТИСТИКА БАЗЫ:")
        for key, value in db_stats.items():
            print(f"   • {key}: {value}")


def main():
    print("🚀 УЛУЧШЕННЫЙ Bulk Spotify Enhancement")
    print("=" * 60)
    
    # Проверяем credentials
    client_id = os.getenv('SPOTIFY_CLIENT_ID')
    client_secret = os.getenv('SPOTIFY_CLIENT_SECRET')
    
    if not client_id or not client_secret or client_id == 'your_client_id_here':
        print("❌ Spotify credentials не найдены!")
        print("Сначала настройте credentials в .env")
        return
    
    # Создаем улучшенный enhancer
    enhancer = ImprovedSpotifyEnhancer(client_id, client_secret)
    bulk_enhancer = ImprovedBulkSpotifyEnhancement(enhancer)
    
    print("\nВыберите режим:")
    print("1. 🔄 Продолжить с checkpoint'а (автоматически)")
    print("2. 🚀 ОБОГАЩЕНИЕ БАТЧА ТРЕКОВ (1000 штук)")
    print("3. 🎯 ПОЛНОЕ ОБОГАЩЕНИЕ (все треки)")
    print("4. 📊 Показать текущую статистику")
    
    choice = input("\nВаш выбор (1-4): ").strip()
    
    if choice == "1":
        print("\n🔄 ПРОДОЛЖЕНИЕ С CHECKPOINT'А")
        bulk_enhancer.enhance_all_tracks_improved()
    
    elif choice == "2":
        batch_size = input("Размер батча (по умолчанию 1000): ").strip()
        batch_size = int(batch_size) if batch_size.isdigit() else 1000
        
        print(f"\n🚀 ОБОГАЩЕНИЕ БАТЧА ({batch_size} треков)")
        bulk_enhancer.enhance_all_tracks_improved(batch_size=batch_size)
    
    elif choice == "3":
        confirm = input("\n⚠️ Это обработает ВСЕ треки. Продолжить? (y/N): ")
        if confirm.lower() == 'y':
            print("\n🎯 ПОЛНОЕ ОБОГАЩЕНИЕ")
            bulk_enhancer.enhance_all_tracks_improved(batch_size=50000)  # Большой батч
        else:
            print("Отменено")
    
    elif choice == "4":
        print("\n📊 ТЕКУЩАЯ СТАТИСТИКА")
        stats = enhancer.get_stats()
        for key, value in stats.items():
            print(f"  • {key}: {value}")
    
    else:
        print("❌ Неверный выбор")


if __name__ == "__main__":
    main()
