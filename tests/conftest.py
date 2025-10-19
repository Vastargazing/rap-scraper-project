# Pytest Configuration and Fixtures
import os
import sqlite3
import sys
import tempfile
from pathlib import Path

import pytest

# Добавляем src в path для импорта модулей
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from enhancers.spotify_enhancer import SpotifyEnhancer
from models.models import SpotifyArtist, SpotifyAudioFeatures, SpotifyTrack


@pytest.fixture
def temp_db():
    """Временная база данных для тестов"""
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
    temp_file.close()

    # Создаем базовую структуру таблиц
    conn = sqlite3.connect(temp_file.name)
    cursor = conn.cursor()

    # Основная таблица songs
    cursor.execute("""
        CREATE TABLE songs (
            id INTEGER PRIMARY KEY,
            title TEXT,
            artist TEXT,
            lyrics TEXT,
            url TEXT
        )
    """)

    # Тестовые данные
    test_songs = [
        (
            1,
            "Hotline Bling",
            "Drake",
            "You used to call me on my...",
            "https://genius.com/1",
        ),
        (
            2,
            "HUMBLE.",
            "Kendrick Lamar",
            "Sit down, be humble...",
            "https://genius.com/2",
        ),
        (3, "God's Plan", "Drake", "I only love my bed...", "https://genius.com/3"),
    ]

    cursor.executemany("INSERT INTO songs VALUES (?, ?, ?, ?, ?)", test_songs)
    conn.commit()
    conn.close()

    yield temp_file.name

    # Cleanup
    os.unlink(temp_file.name)


@pytest.fixture
def mock_spotify_enhancer(temp_db):
    """Mock Spotify enhancer с временной базой"""
    enhancer = SpotifyEnhancer(
        client_id="test_client_id", client_secret="test_client_secret", db_path=temp_db
    )
    enhancer.create_spotify_tables()
    return enhancer


@pytest.fixture
def sample_spotify_artist():
    """Пример данных Spotify артиста"""
    return SpotifyArtist(
        spotify_id="3TVXtAsR1Inumwj472S9r4",
        name="Drake",
        genres=["canadian hip hop", "canadian pop", "hip hop", "pop rap", "rap"],
        popularity=100,
        followers=88500000,
        image_url="https://i.scdn.co/image/abc123",
        spotify_url="https://open.spotify.com/artist/3TVXtAsR1Inumwj472S9r4",
    )


@pytest.fixture
def sample_spotify_track():
    """Пример данных Spotify трека"""
    return SpotifyTrack(
        spotify_id="0wwPcA6wtMf6HUMpIRdeP7",
        name="Hotline Bling",
        artist_id="3TVXtAsR1Inumwj472S9r4",
        album_name="Views",
        release_date="2016-04-29",
        duration_ms=267066,
        popularity=77,
        explicit=False,
        spotify_url="https://open.spotify.com/track/0wwPcA6wtMf6HUMpIRdeP7",
        preview_url="https://p.scdn.co/mp3-preview/xyz789",
    )


@pytest.fixture
def sample_audio_features():
    """Пример аудио-характеристик"""
    return SpotifyAudioFeatures(
        danceability=0.715,
        energy=0.526,
        valence=0.357,
        tempo=135.051,
        acousticness=0.00663,
        instrumentalness=0.000586,
        speechiness=0.332,
        liveness=0.123,
        loudness=-7.305,
    )


@pytest.fixture
def mock_spotify_api_response():
    """Mock ответов Spotify API"""
    return {
        "search_artist": {
            "artists": {
                "items": [
                    {
                        "id": "3TVXtAsR1Inumwj472S9r4",
                        "name": "Drake",
                        "genres": ["canadian hip hop", "canadian pop"],
                        "popularity": 100,
                        "followers": {"total": 88500000},
                        "images": [{"url": "https://i.scdn.co/image/abc123"}],
                        "external_urls": {
                            "spotify": "https://open.spotify.com/artist/3TVXtAsR1Inumwj472S9r4"
                        },
                    }
                ]
            }
        },
        "search_track": {
            "tracks": {
                "items": [
                    {
                        "id": "0wwPcA6wtMf6HUMpIRdeP7",
                        "name": "Hotline Bling",
                        "artists": [{"id": "3TVXtAsR1Inumwj472S9r4"}],
                        "album": {"name": "Views", "release_date": "2016-04-29"},
                        "duration_ms": 267066,
                        "popularity": 77,
                        "explicit": False,
                        "external_urls": {
                            "spotify": "https://open.spotify.com/track/0wwPcA6wtMf6HUMpIRdeP7"
                        },
                        "preview_url": "https://p.scdn.co/mp3-preview/xyz789",
                    }
                ]
            }
        },
        "audio_features": {
            "danceability": 0.715,
            "energy": 0.526,
            "valence": 0.357,
            "tempo": 135.051,
            "acousticness": 0.00663,
            "instrumentalness": 0.000586,
            "speechiness": 0.332,
            "liveness": 0.123,
            "loudness": -7.305,
        },
    }
