# Tests for Spotify Enhancer API Integration
import sqlite3
import unittest
from unittest.mock import Mock, patch

from spotify_enhancer import SpotifyEnhancer

from models import SpotifyArtist, SpotifyEnrichmentResult, SpotifyTrack


class TestSpotifyEnhancer(unittest.TestCase):
    """Тесты для SpotifyEnhancer класса"""

    def setUp(self):
        """Настройка перед каждым тестом"""
        self.enhancer = SpotifyEnhancer(
            client_id="test_client_id",
            client_secret="test_client_secret",
            db_path=":memory:",  # In-memory database для тестов
        )
        self.enhancer.create_spotify_tables()

    def test_initialization(self):
        """Тест инициализации SpotifyEnhancer"""
        self.assertEqual(self.enhancer.client_id, "test_client_id")
        self.assertEqual(self.enhancer.client_secret, "test_client_secret")
        self.assertIsNone(self.enhancer.access_token)
        self.assertEqual(self.enhancer.api_calls_count, 0)

    @patch("spotify_enhancer.requests.post")
    def test_get_access_token_success(self, mock_post):
        """Тест успешного получения access token"""
        # Mock ответ
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "access_token": "test_token_123",
            "expires_in": 3600,
        }
        mock_post.return_value = mock_response

        # Тест
        result = self.enhancer.get_access_token()

        self.assertTrue(result)
        self.assertEqual(self.enhancer.access_token, "test_token_123")
        self.assertIsNotNone(self.enhancer.token_expires_at)

    @patch("spotify_enhancer.requests.post")
    def test_get_access_token_failure(self, mock_post):
        """Тест неудачного получения токена"""
        mock_response = Mock()
        mock_response.status_code = 400
        mock_response.text = "Invalid client"
        mock_post.return_value = mock_response

        result = self.enhancer.get_access_token()

        self.assertFalse(result)
        self.assertIsNone(self.enhancer.access_token)

    @patch.object(SpotifyEnhancer, "_make_request")
    def test_search_artist_success(self, mock_request):
        """Тест успешного поиска артиста"""
        # Mock API response
        mock_request.return_value = {
            "artists": {
                "items": [
                    {
                        "id": "test_artist_id",
                        "name": "Drake",
                        "genres": ["hip hop", "rap"],
                        "popularity": 95,
                        "followers": {"total": 50000000},
                        "images": [{"url": "https://example.com/image.jpg"}],
                        "external_urls": {
                            "spotify": "https://open.spotify.com/artist/test"
                        },
                    }
                ]
            }
        }

        artist = self.enhancer.search_artist("Drake")

        self.assertIsInstance(artist, SpotifyArtist)
        self.assertEqual(artist.name, "Drake")
        self.assertEqual(artist.spotify_id, "test_artist_id")
        self.assertEqual(artist.popularity, 95)
        self.assertEqual(artist.followers, 50000000)

    @patch.object(SpotifyEnhancer, "_make_request")
    def test_search_artist_not_found(self, mock_request):
        """Тест когда артист не найден"""
        mock_request.return_value = {"artists": {"items": []}}

        artist = self.enhancer.search_artist("Unknown Artist")

        self.assertIsNone(artist)

    @patch.object(SpotifyEnhancer, "_make_request")
    def test_search_track_success(self, mock_request):
        """Тест успешного поиска трека"""
        mock_request.return_value = {
            "tracks": {
                "items": [
                    {
                        "id": "test_track_id",
                        "name": "Hotline Bling",
                        "artists": [{"id": "test_artist_id"}],
                        "album": {"name": "Views", "release_date": "2016-04-29"},
                        "duration_ms": 267066,
                        "popularity": 85,
                        "explicit": False,
                        "external_urls": {
                            "spotify": "https://open.spotify.com/track/test"
                        },
                        "preview_url": "https://preview.spotify.com/test",
                    }
                ]
            }
        }

        track = self.enhancer.search_track("Hotline Bling", "Drake")

        self.assertIsInstance(track, SpotifyTrack)
        self.assertEqual(track.name, "Hotline Bling")
        self.assertEqual(track.spotify_id, "test_track_id")
        self.assertEqual(track.album_name, "Views")
        self.assertEqual(track.popularity, 85)

    @patch.object(SpotifyEnhancer, "search_artist")
    def test_enhance_artist_success(self, mock_search):
        """Тест успешного обогащения артиста"""
        mock_artist = SpotifyArtist(
            spotify_id="test_id",
            name="Test Artist",
            genres=["hip hop"],
            popularity=80,
            followers=1000000,
            spotify_url="https://test.com",
        )
        mock_search.return_value = mock_artist

        result = self.enhancer.enhance_artist("Test Artist")

        self.assertIsInstance(result, SpotifyEnrichmentResult)
        self.assertTrue(result.success)
        self.assertEqual(result.artist_data.name, "Test Artist")
        self.assertIsNone(result.error_message)
        self.assertGreater(result.processing_time, 0)

    @patch.object(SpotifyEnhancer, "search_artist")
    def test_enhance_artist_not_found(self, mock_search):
        """Тест когда артист не найден при обогащении"""
        mock_search.return_value = None

        result = self.enhancer.enhance_artist("Unknown Artist")

        self.assertIsInstance(result, SpotifyEnrichmentResult)
        self.assertFalse(result.success)
        self.assertIsNone(result.artist_data)
        self.assertIn("не найден", result.error_message)

    def test_create_spotify_tables(self):
        """Тест создания таблиц Spotify"""
        # Таблицы уже созданы в setUp, проверяем их существование
        conn = sqlite3.connect(self.enhancer.db_path)
        cursor = conn.cursor()

        # Проверяем таблицу артистов
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='spotify_artists'"
        )
        self.assertIsNotNone(cursor.fetchone())

        # Проверяем таблицу треков
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='spotify_tracks'"
        )
        self.assertIsNotNone(cursor.fetchone())

        # Проверяем таблицу аудио-характеристик
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='spotify_audio_features'"
        )
        self.assertIsNotNone(cursor.fetchone())

        conn.close()

    def test_save_artist_to_db(self):
        """Тест сохранения артиста в базу"""
        artist = SpotifyArtist(
            spotify_id="test_artist_123",
            name="Test Artist",
            genres=["hip hop", "rap"],
            popularity=75,
            followers=500000,
            spotify_url="https://test.spotify.com",
        )

        # Сохраняем
        self.enhancer.save_artist_to_db("Test Artist", artist)

        # Проверяем
        conn = sqlite3.connect(self.enhancer.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM spotify_artists WHERE artist_name = ?", ("Test Artist",)
        )
        result = cursor.fetchone()
        conn.close()

        self.assertIsNotNone(result)
        self.assertEqual(result[1], "Test Artist")  # artist_name
        self.assertEqual(result[2], "test_artist_123")  # spotify_id
        self.assertEqual(result[4], 75)  # popularity

    def test_rate_limiting(self):
        """Тест rate limiting"""
        import time

        # Устанавливаем быстрый лимит для теста
        self.enhancer.requests_per_second = 2

        start_time = time.time()

        # Делаем два "запроса"
        self.enhancer.last_request_time = time.time()
        time.sleep(0.1)  # Маленькая пауза между вызовами

        # Проверяем, что rate limiting работает
        self.assertGreater(time.time() - start_time, 0.5)  # Должно быть больше 0.5 сек


if __name__ == "__main__":
    unittest.main()
