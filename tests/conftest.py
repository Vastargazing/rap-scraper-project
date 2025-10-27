# Pytest Configuration and Fixtures
# SQLite fixtures removed - project migrated to PostgreSQL
# See src/database/postgres_adapter.py for current database implementation

import sys
from pathlib import Path

import pytest

# Добавляем src в path для импорта модулей
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from models.models import SpotifyArtist, SpotifyAudioFeatures, SpotifyTrack


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
