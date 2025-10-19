"""
🧪 Quick test for ML API Service
"""

import asyncio
from datetime import datetime

import aiohttp


async def test_api():
    """Test ML API endpoints"""
    base_url = "http://127.0.0.1:8001"

    # Test endpoints
    tests = [
        {"name": "Health Check", "url": f"{base_url}/health", "method": "GET"},
        {"name": "Root Endpoint", "url": f"{base_url}/", "method": "GET"},
        {"name": "Models Info", "url": f"{base_url}/models/info", "method": "GET"},
    ]

    async with aiohttp.ClientSession() as session:
        for test in tests:
            try:
                print(f"🧪 Testing {test['name']}...")

                if test["method"] == "GET":
                    async with session.get(test["url"]) as response:
                        if response.status == 200:
                            data = await response.json()
                            print(f"✅ {test['name']}: SUCCESS")
                            if test["name"] == "Health Check":
                                print(f"   Models status: {data.get('models', {})}")
                        else:
                            print(f"❌ {test['name']}: HTTP {response.status}")

            except Exception as e:
                print(f"❌ {test['name']}: {e}")

    # Test generation endpoint
    try:
        print("\n🎤 Testing Text Generation...")
        generation_data = {
            "prompt": "I'm on the mic with the flow",
            "artist_style": "kendrick",
            "mood": "confident",
            "theme": "success",
            "max_length": 100,
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{base_url}/generate", json=generation_data
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    print("✅ Generation: SUCCESS")
                    print(f"   Generated: {data.get('generated_text', '')[:100]}...")
                else:
                    print(f"❌ Generation: HTTP {response.status}")
                    text = await response.text()
                    print(f"   Error: {text[:200]}...")

    except Exception as e:
        print(f"❌ Generation test: {e}")


if __name__ == "__main__":
    print("🚀 ML API TEST SUITE")
    print("=" * 50)
    print(f"Starting tests at {datetime.now()}")
    print("=" * 50)

    try:
        asyncio.run(test_api())
        print("\n✅ Test suite completed!")
    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")
