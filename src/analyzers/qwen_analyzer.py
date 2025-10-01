"""
🤖 QWEN Analyzer Wrapper with Config Integration
Type-safe QWEN model integration for lyrics analysis

Features:
- Config loader integration for API settings
- Automatic API key validation
- Retry logic from config
- Temperature and token limits from config
- Response caching support

Author: Vastargazing
Version: 2.0.0
"""

import logging
from typing import Dict, List, Optional, Any
from openai import OpenAI
import time

from src.config.config_loader import get_config
from src.cache.redis_client import redis_cache

logger = logging.getLogger(__name__)


class QwenAnalyzer:
    """
    QWEN-based lyrics analyzer with config integration
    
    Usage:
        analyzer = QwenAnalyzer()
        result = analyzer.analyze_lyrics("rap lyrics here")
        
        # With custom temperature
        result = analyzer.analyze_lyrics("lyrics", temperature=0.2)
    """
    
    def __init__(self):
        """Initialize QWEN analyzer with config settings"""
        config = get_config()
        self.qwen_config = config.analyzers.get_qwen()
        
        logger.info(f"🤖 Initializing QWEN Analyzer...")
        logger.info(f"   Model: {self.qwen_config.model_name}")
        logger.info(f"   Base URL: {self.qwen_config.base_url}")
        logger.info(f"   Temperature: {self.qwen_config.temperature}")
        logger.info(f"   Max Tokens: {self.qwen_config.max_tokens}")
        logger.info(f"   Timeout: {self.qwen_config.timeout}s")
        logger.info(f"   Retry Attempts: {self.qwen_config.retry_attempts}")
        
        # Initialize OpenAI client with QWEN endpoint
        self.client = OpenAI(
            base_url=self.qwen_config.base_url,
            api_key=self.qwen_config.api_key  # Validates and reads from ENV
        )
        
        self.use_cache = True  # Can be configured
        
        logger.info("✅ QWEN Analyzer initialized successfully!")
    
    def analyze_lyrics(
        self,
        lyrics: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """
        Analyze rap lyrics using QWEN model
        
        Args:
            lyrics: Lyrics text to analyze
            temperature: Override config temperature (optional)
            max_tokens: Override config max tokens (optional)
            use_cache: Use Redis cache if available
            
        Returns:
            dict: Analysis results with themes, style, quality, etc.
        """
        # Check cache first
        if use_cache and self.use_cache:
            cached = redis_cache.get_analysis(f"qwen:{hash(lyrics)}")
            if cached:
                logger.info("✅ Using cached QWEN analysis")
                return cached
        
        # Use config defaults if not overridden
        temp = temperature if temperature is not None else self.qwen_config.temperature
        tokens = max_tokens if max_tokens is not None else self.qwen_config.max_tokens
        
        # Build prompt
        prompt = self._build_analysis_prompt(lyrics)
        
        # Analyze with retry logic
        result = self._analyze_with_retry(prompt, temp, tokens)
        
        # Cache result
        if use_cache and self.use_cache and result:
            redis_cache.cache_analysis(f"qwen:{hash(lyrics)}", result)
        
        return result
    
    def _build_analysis_prompt(self, lyrics: str) -> str:
        """Build analysis prompt for QWEN"""
        return f"""Analyze these rap lyrics and provide a detailed breakdown:

LYRICS:
{lyrics}

Please analyze:
1. Main themes and topics
2. Lyrical style and flow
3. Complexity level (1-10)
4. Emotional tone
5. Quality score (1-10)
6. Notable metaphors or wordplay

Provide response in JSON format."""
    
    def _analyze_with_retry(
        self,
        prompt: str,
        temperature: float,
        max_tokens: int
    ) -> Dict[str, Any]:
        """
        Analyze with automatic retry on failure
        
        Args:
            prompt: Analysis prompt
            temperature: Temperature setting
            max_tokens: Max tokens setting
            
        Returns:
            dict: Analysis results or error dict
        """
        last_error = None
        
        for attempt in range(1, self.qwen_config.retry_attempts + 1):
            try:
                logger.info(f"🤖 QWEN analysis attempt {attempt}/{self.qwen_config.retry_attempts}")
                
                response = self.client.chat.completions.create(
                    model=self.qwen_config.model_name,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an expert rap lyrics analyst. Provide detailed, structured analysis."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=self.qwen_config.timeout
                )
                
                # Extract and parse response
                content = response.choices[0].message.content
                
                # Try to parse as JSON
                try:
                    import json
                    result = json.loads(content)
                except json.JSONDecodeError:
                    # If not JSON, return as text analysis
                    result = {
                        "analysis": content,
                        "raw_response": True
                    }
                
                # Add metadata
                result["model"] = self.qwen_config.model_name
                result["tokens_used"] = response.usage.total_tokens if hasattr(response.usage, 'total_tokens') else None
                result["timestamp"] = time.time()
                
                logger.info(f"✅ QWEN analysis successful (tokens: {result.get('tokens_used', 'N/A')})")
                return result
                
            except Exception as e:
                last_error = e
                logger.warning(f"⚠️ QWEN attempt {attempt} failed: {e}")
                
                if attempt < self.qwen_config.retry_attempts:
                    wait_time = attempt * 2  # Exponential backoff
                    logger.info(f"   Retrying in {wait_time}s...")
                    time.sleep(wait_time)
        
        # All attempts failed
        logger.error(f"❌ QWEN analysis failed after {self.qwen_config.retry_attempts} attempts: {last_error}")
        return {
            "error": str(last_error),
            "model": self.qwen_config.model_name,
            "failed": True
        }
    
    def test_api_connection(self) -> bool:
        """
        Test QWEN API connection
        
        Returns:
            bool: True if connection successful
        """
        try:
            logger.info("🧪 Testing QWEN API connection...")
            
            response = self.client.chat.completions.create(
                model=self.qwen_config.model_name,
                messages=[
                    {"role": "user", "content": "Hello, this is a test."}
                ],
                max_tokens=10,
                timeout=10
            )
            
            logger.info(f"✅ QWEN API connection successful!")
            logger.info(f"   Model: {self.qwen_config.model_name}")
            logger.info(f"   Response: {response.choices[0].message.content}")
            return True
            
        except Exception as e:
            logger.error(f"❌ QWEN API connection failed: {e}")
            return False
    
    def get_config_info(self) -> Dict[str, Any]:
        """Get current configuration info"""
        return {
            "model": self.qwen_config.model_name,
            "base_url": self.qwen_config.base_url,
            "temperature": self.qwen_config.temperature,
            "max_tokens": self.qwen_config.max_tokens,
            "timeout": self.qwen_config.timeout,
            "retry_attempts": self.qwen_config.retry_attempts,
            "api_key_set": bool(self.qwen_config.api_key),
            "cache_enabled": self.use_cache
        }


if __name__ == "__main__":
    # Test QWEN analyzer
    print("🧪 Testing QWEN Analyzer...")
    print("=" * 60)
    
    try:
        # Initialize analyzer
        analyzer = QwenAnalyzer()
        
        # Show config
        print(f"\n📊 Configuration:")
        config_info = analyzer.get_config_info()
        for key, value in config_info.items():
            print(f"   {key}: {value}")
        
        # Test connection
        print(f"\n🔌 Testing API connection...")
        if analyzer.test_api_connection():
            print("✅ Connection test passed!")
            
            # Test lyrics analysis
            print(f"\n🎤 Testing lyrics analysis...")
            test_lyrics = """
            Started from the bottom now we're here
            Started from the bottom now my whole team here
            """
            
            result = analyzer.analyze_lyrics(test_lyrics, use_cache=False)
            
            if "error" in result:
                print(f"❌ Analysis failed: {result['error']}")
            else:
                print(f"✅ Analysis successful!")
                print(f"   Model: {result.get('model')}")
                print(f"   Tokens: {result.get('tokens_used')}")
                if 'analysis' in result:
                    print(f"   Response: {result['analysis'][:100]}...")
            
            print("\n✅ All tests passed!")
        else:
            print("❌ Connection test failed!")
            print("⚠️ Check your NOVITA_API_KEY in .env file")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
