# Tests for Pydantic Models
import unittest
from pydantic import ValidationError
from models import SpotifyArtist, SpotifyTrack, SpotifyAudioFeatures, SpotifyEnrichmentResult

class TestSpotifyModels(unittest.TestCase):
    """Тесты для Pydantic моделей Spotify"""
    
    def test_spotify_artist_valid(self):
        """Тест валидного SpotifyArtist"""
        artist_data = {
            "spotify_id": "3TVXtAsR1Inumwj472S9r4",
            "name": "Drake",
            "genres": ["hip hop", "canadian hip hop"],
            "popularity": 95,
            "followers": 85000000,
            "spotify_url": "https://open.spotify.com/artist/3TVXtAsR1Inumwj472S9r4"
        }
        
        artist = SpotifyArtist(**artist_data)
        
        self.assertEqual(artist.name, "Drake")
        self.assertEqual(artist.popularity, 95)
        self.assertEqual(len(artist.genres), 2)
        self.assertIn("hip hop", artist.genres)
    
    def test_spotify_artist_required_fields(self):
        """Тест обязательных полей SpotifyArtist"""
        # Пропускаем обязательное поле name
        with self.assertRaises(ValidationError):
            SpotifyArtist(
                spotify_id="test_id",
                genres=["hip hop"],
                popularity=80,
                followers=1000000,
                spotify_url="https://test.com"
            )
    
    def test_spotify_artist_popularity_validation(self):
        """Тест валидации popularity (должно быть 0-100)"""
        # Popularity больше 100
        with self.assertRaises(ValidationError):
            SpotifyArtist(
                spotify_id="test_id",
                name="Test Artist",
                popularity=150,
                followers=1000,
                spotify_url="https://test.com"
            )
        
        # Negative popularity
        with self.assertRaises(ValidationError):
            SpotifyArtist(
                spotify_id="test_id", 
                name="Test Artist",
                popularity=-10,
                followers=1000,
                spotify_url="https://test.com"
            )
    
    def test_spotify_track_valid(self):
        """Тест валидного SpotifyTrack"""
        track_data = {
            "spotify_id": "0wwPcA6wtMf6HUMpIRdeP7",
            "name": "Hotline Bling",
            "artist_id": "3TVXtAsR1Inumwj472S9r4",
            "album_name": "Views", 
            "release_date": "2016-04-29",
            "duration_ms": 267066,
            "popularity": 77,
            "explicit": False,
            "spotify_url": "https://open.spotify.com/track/0wwPcA6wtMf6HUMpIRdeP7"
        }
        
        track = SpotifyTrack(**track_data)
        
        self.assertEqual(track.name, "Hotline Bling")
        self.assertEqual(track.duration_ms, 267066)
        self.assertFalse(track.explicit)
        self.assertEqual(track.release_date, "2016-04-29")
    
    def test_spotify_audio_features_valid(self):
        """Тест валидных SpotifyAudioFeatures"""
        features_data = {
            "danceability": 0.715,
            "energy": 0.526,
            "valence": 0.357,
            "tempo": 135.051,
            "acousticness": 0.00663,
            "instrumentalness": 0.000586,
            "speechiness": 0.332,
            "liveness": 0.123,
            "loudness": -7.305
        }
        
        features = SpotifyAudioFeatures(**features_data)
        
        self.assertAlmostEqual(features.danceability, 0.715, places=3)
        self.assertAlmostEqual(features.tempo, 135.051, places=3)
        self.assertAlmostEqual(features.loudness, -7.305, places=3)
    
    def test_spotify_audio_features_range_validation(self):
        """Тест валидации диапазонов аудио-характеристик"""
        # Danceability должно быть 0.0-1.0
        with self.assertRaises(ValidationError):
            SpotifyAudioFeatures(
                danceability=1.5,  # Больше 1.0
                energy=0.5,
                valence=0.3,
                tempo=120.0
            )
        
        # Energy должно быть 0.0-1.0
        with self.assertRaises(ValidationError):
            SpotifyAudioFeatures(
                danceability=0.7,
                energy=-0.1,  # Меньше 0.0
                valence=0.3,
                tempo=120.0
            )
    
    def test_spotify_enrichment_result_success(self):
        """Тест успешного SpotifyEnrichmentResult"""
        artist = SpotifyArtist(
            spotify_id="test_id",
            name="Test Artist",
            genres=["hip hop"],
            popularity=80,
            followers=1000000,
            spotify_url="https://test.com"
        )
        
        result = SpotifyEnrichmentResult(
            success=True,
            artist_data=artist,
            processing_time=1.25,
            api_calls_used=2
        )
        
        self.assertTrue(result.success)
        self.assertEqual(result.artist_data.name, "Test Artist")
        self.assertIsNone(result.error_message)
        self.assertEqual(result.api_calls_used, 2)
    
    def test_spotify_enrichment_result_failure(self):
        """Тест неудачного SpotifyEnrichmentResult"""
        result = SpotifyEnrichmentResult(
            success=False,
            error_message="Artist not found",
            processing_time=0.5,
            api_calls_used=1
        )
        
        self.assertFalse(result.success)
        self.assertEqual(result.error_message, "Artist not found")
        self.assertIsNone(result.artist_data)
        self.assertIsNone(result.track_data)
    
    def test_spotify_track_with_audio_features(self):
        """Тест SpotifyTrack с аудио-характеристиками"""
        audio_features = SpotifyAudioFeatures(
            danceability=0.8,
            energy=0.7,
            valence=0.6,
            tempo=128.0,
            acousticness=0.1,
            instrumentalness=0.0,
            speechiness=0.2,
            liveness=0.1,
            loudness=-5.0
        )
        
        track = SpotifyTrack(
            spotify_id="test_track_id",
            name="Test Track",
            artist_id="test_artist_id",
            popularity=75,
            spotify_url="https://test.com",
            audio_features=audio_features
        )
        
        self.assertEqual(track.name, "Test Track")
        self.assertIsNotNone(track.audio_features)
        self.assertEqual(track.audio_features.danceability, 0.8)
        self.assertEqual(track.audio_features.tempo, 128.0)

if __name__ == '__main__':
    unittest.main()
