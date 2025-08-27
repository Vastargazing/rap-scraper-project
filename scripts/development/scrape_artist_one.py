#!/usr/bin/env python3
"""
Скрипт для тестирования скрапинга одного артиста.
Решает проблемы с прокси и показывает детальную отладку.
"""

import sys
import os
import logging
from pathlib import Path

# Добавляем корневую папку в путь
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.scrapers.rap_scraper_optimized import OptimizedGeniusScraper
from src.utils.config import GENIUS_TOKEN, LOG_FORMAT, LOG_FILE

# Настройка логирования с детальной отладкой
logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    handlers=[
        logging.FileHandler(LOG_FILE, encoding='utf-8'),
        logging.StreamHandler()
    ]
)

class SingleArtistScraper(OptimizedGeniusScraper):
    """Расширенный скрапер для тестирования одного артиста с обходом прокси проблем"""
    
    def __init__(self, token: str, db_name: str = None):
        # Перед инициализацией убираем проблемные прокси переменные
        self._clear_proxy_env()
        super().__init__(token, db_name, memory_limit_mb=1024)
        
        # Дополнительная настройка для обхода прокси проблем
        self.genius.timeout = 30  # Увеличиваем timeout
        self.genius.retries = 2   # Уменьшаем количество попыток чтобы быстрее переключаться
        
    def _clear_proxy_env(self):
        """Убираем все проблемные прокси переменные"""
        proxy_vars = [
            'HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy',
            'FTP_PROXY', 'ftp_proxy', 'ALL_PROXY', 'all_proxy',
            'NO_PROXY', 'no_proxy'
        ]
        
        self.cleared_proxies = {}
        for var in proxy_vars:
            if var in os.environ:
                self.cleared_proxies[var] = os.environ.pop(var)
                logging.info(f"🚫 Убрали прокси переменную: {var}")
                
    def _restore_proxy_env(self):
        """Восстанавливаем прокси переменные если нужно"""
        for var, value in self.cleared_proxies.items():
            os.environ[var] = value
            
    def safe_search_artist(self, artist_name: str, max_songs: int = 20):
        """Безопасный поиск артиста с обработкой прокси ошибок"""
        logging.info(f"🎯 Попытка поиска артиста: {artist_name}")
        
        try:
            # Первая попытка поиска
            artist = self.genius.search_artist(
                artist_name, 
                max_songs=max_songs, 
                sort="popularity",
                get_full_info=False
            )
            logging.info(f"✅ Артист найден: {artist.name if artist else 'None'}")
            return artist
            
        except Exception as e:
            error_msg = str(e).lower()
            logging.error(f"❌ Ошибка поиска артиста: {e}")
            
            # Проверяем является ли это прокси ошибкой
            if any(keyword in error_msg for keyword in ['proxy', 'connection', 'timeout', 'retries exceeded']):
                logging.warning("🔄 Детектирована проблема с соединением, пробуем альтернативные методы...")
                
                # Пробуем различные обходы
                return self._try_alternative_connections(artist_name, max_songs)
            else:
                logging.error(f"💥 Неизвестная ошибка: {e}")
                return None
                
    def _try_alternative_connections(self, artist_name: str, max_songs: int):
        """Пробуем альтернативные способы подключения"""
        
        # Метод 1: Изменяем User-Agent и headers
        try:
            logging.info("🔧 Метод 1: Изменяем User-Agent...")
            original_headers = getattr(self.genius._session, 'headers', {})
            
            self.genius._session.headers.update({
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'application/json, text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1'
            })
            
            artist = self.genius.search_artist(artist_name, max_songs=max_songs, sort="popularity")
            if artist:
                logging.info("✅ Метод 1 сработал!")
                return artist
                
        except Exception as e:
            logging.warning(f"⚠️ Метод 1 не сработал: {e}")
            
        # Метод 2: Уменьшаем количество песен и пробуем снова
        try:
            logging.info("🔧 Метод 2: Уменьшаем количество песен...")
            artist = self.genius.search_artist(artist_name, max_songs=5, sort="popularity")
            if artist:
                logging.info("✅ Метод 2 сработал!")
                return artist
                
        except Exception as e:
            logging.warning(f"⚠️ Метод 2 не сработал: {e}")
            
        # Метод 3: Пробуем совсем простой поиск
        try:
            logging.info("🔧 Метод 3: Простой поиск песни...")
            # Ищем одну песню вместо артиста
            songs = self.genius.search_songs(f"{artist_name} song", per_page=5)
            if songs and songs['hits']:
                # Находим песню этого артиста
                for hit in songs['hits']:
                    if hit['result']['primary_artist']['name'].lower() == artist_name.lower():
                        logging.info(f"✅ Метод 3 сработал! Найдена песня: {hit['result']['title']}")
                        # Создаем минимальный объект артиста
                        return self._create_minimal_artist_from_song(hit['result'])
                        
        except Exception as e:
            logging.warning(f"⚠️ Метод 3 не сработал: {e}")
            
        logging.error("❌ Все методы исчерпаны, артист не найден")
        return None
        
    def _create_minimal_artist_from_song(self, song_data):
        """Создаем минимальный объект артиста из данных песни"""
        class MinimalArtist:
            def __init__(self, song_data):
                self.name = song_data['primary_artist']['name']
                self.id = song_data['primary_artist']['id']
                self.songs = []
                
                # Добавляем найденную песню
                class MinimalSong:
                    def __init__(self, song_data):
                        self.title = song_data['title']
                        self.id = song_data['id']
                        self.url = song_data['url']
                        self.lyrics = ""  # Нужно будет загрузить отдельно
                        
                self.songs.append(MinimalSong(song_data))
                
        return MinimalArtist(song_data)
    
    def test_single_artist(self, artist_name: str, max_songs: int = 20):
        """Тестирование скрапинга одного артиста"""
        logging.info(f"🎤 Начинаем тестирование артиста: {artist_name}")
        
        # Мониторинг ресурсов
        self.monitor.log_resources()
        
        try:
            # Безопасный поиск артиста
            artist = self.safe_search_artist(artist_name, max_songs)
            
            if not artist:
                logging.error(f"❌ Артист {artist_name} не найден")
                return 0
                
            if not hasattr(artist, 'songs') or not artist.songs:
                logging.warning(f"⚠️ У артиста {artist_name} нет песен")
                return 0
                
            total_songs = len(artist.songs)
            logging.info(f"📀 Найдено {total_songs} песен для {artist_name}")
            
            # Показываем список песен
            logging.info("🎵 Список найденных песен:")
            for i, song in enumerate(artist.songs[:10], 1):  # Показываем первые 10
                logging.info(f"  {i:2d}. {song.title}")
                
            if total_songs > 10:
                logging.info(f"  ... и еще {total_songs - 10} песен")
                
            # Скрапим песни
            added_count = 0
            for i, song in enumerate(artist.songs, 1):
                if self.shutdown_requested:
                    break
                    
                try:
                    logging.info(f"🎵 Обрабатываем песню {i}/{total_songs}: {song.title}")
                    
                    # Проверяем дубликат
                    if self.db.song_exists(url=song.url):
                        logging.info(f"⏩ Песня уже в базе: {song.title}")
                        continue
                        
                    # Получаем тексты если их нет
                    if not hasattr(song, 'lyrics') or not song.lyrics:
                        try:
                            song = self.genius.lyrics(song.url)
                        except Exception as e:
                            logging.error(f"❌ Ошибка получения текста для {song.title}: {e}")
                            continue
                    
                    # Очищаем и проверяем тексты
                    lyrics = self.clean_lyrics(song.lyrics)
                    if not self._is_valid_lyrics(lyrics):
                        logging.warning(f"⚠️ Некачественный текст: {song.title}")
                        continue
                        
                    # Сохраняем в БД
                    metadata = self.extract_metadata(song)
                    if self.db.add_song(
                        artist_name, song.title, lyrics, song.url,
                        getattr(song, 'id', None), metadata
                    ):
                        added_count += 1
                        word_count = len(lyrics.split())
                        quality = self.db._calculate_lyrics_quality(lyrics)
                        
                        logging.info(f"✅ Добавлено: {song.title} ({word_count} слов, Q:{quality:.2f})")
                    
                    # Небольшая пауза между песнями
                    self.safe_delay()
                    
                except Exception as e:
                    logging.error(f"❌ Ошибка с песней {song.title}: {e}")
                    continue
                    
            logging.info(f"✅ Завершено! Добавлено {added_count} песен для {artist_name}")
            return added_count
            
        except Exception as e:
            logging.error(f"💥 Критическая ошибка при обработке {artist_name}: {e}")
            return 0
        finally:
            self.db.conn.commit()
            self.show_current_results()
            
    def close(self):
        """Закрытие с восстановлением прокси настроек"""
        super().close()
        self._restore_proxy_env()

def main():
    """Главная функция"""
    if not GENIUS_TOKEN:
        print("❌ GENIUS_TOKEN не найден в .env файле!")
        return
        
    # Артист для тестирования (можно изменить)
    test_artist = "Drama"  # Или любой другой артист
    
    if len(sys.argv) > 1:
        test_artist = sys.argv[1]
        
    logging.info(f"🎯 Тестируем артиста: {test_artist}")
    
    # Создаем скрапер
    scraper = SingleArtistScraper(GENIUS_TOKEN)
    
    try:
        # Тестируем скрапинг
        added_count = scraper.test_single_artist(test_artist, max_songs=20)
        print(f"Added for {test_artist}: {added_count}")
        
    except KeyboardInterrupt:
        logging.info("⌨️ Остановлено пользователем")
    except Exception as e:
        logging.error(f"💥 Ошибка: {e}")
    finally:
        scraper.close()

if __name__ == "__main__":
    main()
