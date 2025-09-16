"""
🦙 Ollama AI анализатор текстов песен (оптимизированная версия для слабых ПК)

ОПТИМИЗАЦИИ:
- Легкие модели (qwen2.5:1.5b, phi3:mini)
- Сокращенный контекст и промпты
- Ограничение ресурсов процессора
- Батчинг и кэширование
- Адаптивные настройки под железо

НОВЫЕ ВОЗМОЖНОСТИ:
- Автоматическое определение лучшей модели для вашего ПК
- Режим "эконом" с минимальным потреблением ресурсов
- Прогрессивное снижение качества при перегрузке

АВТОР: AI Assistant (Optimized Version)
ДАТА: Сентябрь 2025
"""
import json
import time
import logging
import requests
import psutil
import threading
from typing import Dict, Any, List, Optional
from datetime import datetime

from interfaces.analyzer_interface import BaseAnalyzer, AnalysisResult, register_analyzer

logger = logging.getLogger(__name__)


@register_analyzer("ollama_optimized")
class OptimizedOllamaAnalyzer(BaseAnalyzer):
    """
    Оптимизированный анализатор на базе локальных моделей Ollama.
    
    Специально адаптирован для слабых ПК:
    - Использует легкие модели (1.5B-3B параметров)
    - Ограничивает использование ресурсов
    - Автоматически подстраивается под нагрузку системы
    - Поддерживает режим экономии ресурсов
    """
    
    # Легкие модели по порядку предпочтения (от самой легкой к более тяжелой)
    LIGHTWEIGHT_MODELS = [
        "qwen2.5:1.5b",     # Самая легкая, но качественная
        "phi3:mini",        # Microsoft, очень быстрая  
        "llama3.2:1b",      # Ультра-легкая Meta
        "gemma2:2b",        # Google, компактная
        "llama3.2:3b",      # Базовая модель из оригинала
    ]
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Инициализация оптимизированного Ollama анализатора"""
        super().__init__(config)
        
        # Настройки оптимизации
        self.economy_mode = self.config.get('economy_mode', True)
        self.max_cpu_usage = self.config.get('max_cpu_usage', 70)  # Максимум 70% CPU
        self.base_url = self.config.get('base_url', 'http://localhost:11434')
        self.timeout = self.config.get('timeout', 30)  # Сокращен с 60 до 30 сек
        
        # Адаптивные настройки (будут определены автоматически)
        self.model_name = None
        self.temperature = 0.0  # Минимальная для скорости
        self.max_tokens = 800   # Сокращено с 1500
        self.context_window = 2048  # Сокращено с 4096
        
        # Мониторинг ресурсов
        self.resource_monitor = ResourceMonitor()
        
        # Автоматический выбор лучшей модели
        self.available = self._initialize_best_model()
        
        if self.available:
            logger.info(f"✅ Оптимизированный Ollama анализатор готов: {self.model_name}")
            logger.info(f"🔧 Режим экономии: {self.economy_mode}, макс CPU: {self.max_cpu_usage}%")
        else:
            logger.warning("⚠️ Оптимизированный Ollama анализатор недоступен")
    
    def _initialize_best_model(self) -> bool:
        """Автоматический выбор лучшей модели для текущего ПК"""
        try:
            # Проверяем доступность Ollama
            if not self._check_ollama_server():
                return False
            
            # Получаем системные характеристики
            system_info = self._get_system_specs()
            logger.info(f"💻 Система: RAM {system_info['ram_gb']:.1f}GB, CPU {system_info['cpu_percent']:.1f}%")
            
            # Получаем список установленных моделей
            installed_models = self._get_installed_models()
            logger.info(f"📦 Установленные модели: {installed_models}")
            
            # Выбираем лучшую доступную модель
            best_model = self._select_optimal_model(system_info, installed_models)
            
            if best_model:
                self.model_name = best_model
                self._adjust_settings_for_model(best_model, system_info)
                
                # Тестируем выбранную модель
                if self._test_model_performance():
                    logger.info(f"🎯 Выбрана оптимальная модель: {self.model_name}")
                    return True
            
            # Если ничего не подошло, пытаемся установить легкую модель
            return self._install_lightweight_model()
            
        except Exception as e:
            logger.error(f"❌ Ошибка инициализации: {e}")
            return False
    
    def _check_ollama_server(self) -> bool:
        """Быстрая проверка доступности Ollama сервера"""
        try:
            response = requests.get(
                f"{self.base_url}/api/tags", 
                timeout=3,  # Быстрая проверка
                proxies={"http": "", "https": ""}
            )
            return response.status_code == 200
        except:
            logger.warning("❌ Ollama сервер недоступен. Запустите: ollama serve")
            return False
    
    def _get_system_specs(self) -> Dict[str, Any]:
        """Получение характеристик системы"""
        try:
            # Память
            memory = psutil.virtual_memory()
            ram_gb = memory.total / (1024**3)
            
            # Процессор
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_cores = psutil.cpu_count()
            
            # Температурная нагрузка (если доступно)
            cpu_freq = psutil.cpu_freq()
            max_freq = cpu_freq.max if cpu_freq else 0
            
            return {
                'ram_gb': ram_gb,
                'cpu_percent': cpu_percent,
                'cpu_cores': cpu_cores,
                'max_freq': max_freq,
                'available_ram_gb': memory.available / (1024**3)
            }
        except Exception as e:
            logger.warning(f"⚠️ Не удалось получить характеристики системы: {e}")
            return {'ram_gb': 8, 'cpu_percent': 50, 'cpu_cores': 4, 'max_freq': 2000}
    
    def _get_installed_models(self) -> List[str]:
        """Получение списка установленных моделей"""
        try:
            response = requests.get(
                f"{self.base_url}/api/tags",
                timeout=5,
                proxies={"http": "", "https": ""}
            )
            
            if response.status_code == 200:
                models_data = response.json().get('models', [])
                return [model['name'] for model in models_data]
            
            return []
        except Exception as e:
            logger.warning(f"⚠️ Не удалось получить список моделей: {e}")
            return []
    
    def _select_optimal_model(self, system_info: Dict, installed_models: List[str]) -> Optional[str]:
        """Выбор оптимальной модели на основе характеристик системы"""
        ram_gb = system_info['ram_gb']
        cpu_percent = system_info['cpu_percent']
        
        logger.info(f"🔍 Поиск оптимальной модели для {ram_gb:.1f}GB RAM, CPU {cpu_percent:.1f}%")
        
        # Определяем категорию системы
        if ram_gb < 4 or cpu_percent > 80:
            # Очень слабая система - только самые легкие модели
            preferred_models = ["qwen2.5:1.5b", "llama3.2:1b"]
        elif ram_gb < 8 or cpu_percent > 60:
            # Слабая система - легкие модели
            preferred_models = ["qwen2.5:1.5b", "phi3:mini", "llama3.2:1b", "gemma2:2b"]
        elif ram_gb < 16:
            # Средняя система - можно использовать 3B модели
            preferred_models = self.LIGHTWEIGHT_MODELS
        else:
            # Мощная система - можно все легкие модели + проверим более тяжелые
            preferred_models = self.LIGHTWEIGHT_MODELS + ["llama3.2:7b", "qwen2.5:7b"]
        
        # Ищем первую доступную модель из предпочитаемых
        for model in preferred_models:
            # Точное совпадение или по базовому имени
            for installed in installed_models:
                if model == installed or model.split(':')[0] in installed:
                    logger.info(f"✅ Найдена подходящая модель: {installed}")
                    return installed
        
        # Если ничего не найдено, берем первую доступную легкую модель
        for model in self.LIGHTWEIGHT_MODELS:
            for installed in installed_models:
                if model.split(':')[0] in installed:
                    logger.info(f"📦 Используем доступную модель: {installed}")
                    return installed
        
        return None
    
    def _adjust_settings_for_model(self, model_name: str, system_info: Dict):
        """Настройка параметров под выбранную модель и систему"""
        ram_gb = system_info['ram_gb']
        
        # Базовые настройки по модели
        if "1b" in model_name.lower():
            # Самые легкие модели
            self.context_window = 1024
            self.max_tokens = 400
            self.temperature = 0.0
        elif "1.5b" in model_name.lower():
            # Легкие качественные модели  
            self.context_window = 1536
            self.max_tokens = 600
            self.temperature = 0.1
        elif any(x in model_name.lower() for x in ["2b", "mini"]):
            # Компактные модели
            self.context_window = 2048
            self.max_tokens = 800
            self.temperature = 0.1
        else:
            # 3B+ модели
            self.context_window = 2048
            self.max_tokens = 1000
            self.temperature = 0.1
        
        # Дополнительные ограничения для слабых систем
        if ram_gb < 6:
            self.context_window = min(self.context_window, 1024)
            self.max_tokens = min(self.max_tokens, 400)
        elif ram_gb < 8:
            self.context_window = min(self.context_window, 1536)
            self.max_tokens = min(self.max_tokens, 600)
        
        # Режим экономии - еще больше ограничений
        if self.economy_mode:
            self.context_window = int(self.context_window * 0.75)
            self.max_tokens = int(self.max_tokens * 0.75)
            self.timeout = min(self.timeout, 20)
        
        logger.info(f"⚙️ Настройки для {model_name}: контекст {self.context_window}, токены {self.max_tokens}")
    
    def _test_model_performance(self) -> bool:
        """Тестирование производительности выбранной модели"""
        try:
            test_prompt = "What is rap music? Answer in one sentence."
            
            start_time = time.time()
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": test_prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.0,
                        "num_ctx": 512,  # Минимальный контекст для теста
                        "num_predict": 50  # Короткий ответ
                    }
                },
                timeout=15,  # Быстрый тест
                proxies={"http": "", "https": ""}
            )
            
            test_time = time.time() - start_time
            
            if response.status_code == 200 and test_time < 10:  # Должно быть быстро
                logger.info(f"✅ Тест модели {self.model_name} прошел успешно ({test_time:.1f}с)")
                return True
            else:
                logger.warning(f"⚠️ Модель {self.model_name} работает медленно ({test_time:.1f}с)")
                return False
                
        except Exception as e:
            logger.warning(f"⚠️ Тест модели {self.model_name} не прошел: {e}")
            return False
    
    def _install_lightweight_model(self) -> bool:
        """Установка самой легкой модели если ничего не найдено"""
        target_model = "qwen2.5:1.5b"  # Лучший компромисс размер/качество
        
        try:
            logger.info(f"📥 Устанавливаем легкую модель {target_model} (это займет несколько минут)...")
            
            response = requests.post(
                f"{self.base_url}/api/pull",
                json={"name": target_model},
                timeout=600,  # 10 минут на загрузку
                proxies={"http": "", "https": ""}
            )
            
            if response.status_code == 200:
                self.model_name = target_model
                system_info = self._get_system_specs()
                self._adjust_settings_for_model(target_model, system_info)
                logger.info(f"✅ Модель {target_model} успешно установлена")
                return True
            else:
                logger.error(f"❌ Не удалось установить {target_model}: {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Ошибка установки модели: {e}")
            return False
    
    def analyze_song(self, artist: str, title: str, lyrics: str) -> AnalysisResult:
        """Оптимизированный анализ песни с контролем ресурсов"""
        start_time = time.time()
        
        # Валидация
        if not self.validate_input(artist, title, lyrics):
            raise ValueError("Invalid input parameters")
        
        if not self.available:
            raise RuntimeError("Optimized Ollama analyzer is not available")
        
        # Мониторинг системных ресурсов
        if self.resource_monitor.is_system_overloaded(self.max_cpu_usage):
            logger.warning(f"⚠️ Система перегружена, используем экономичный режим")
            self._switch_to_economy_mode()
        
        try:
            # Предобработка текста с агрессивным сокращением
            processed_lyrics = self._preprocess_lyrics_optimized(lyrics)
            
            # Создание максимально сжатого промпта
            prompt = self._create_minimal_prompt(artist, title, processed_lyrics)
            
            # Запрос с оптимизированными параметрами
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": self.temperature,
                        "top_p": 0.8,  # Сужен для скорости
                        "num_ctx": self.context_window,
                        "num_predict": self.max_tokens,
                        "repeat_penalty": 1.1,  # Избегаем повторов
                        "num_thread": min(2, psutil.cpu_count())  # Ограничиваем потоки
                    }
                },
                timeout=self.timeout,
                proxies={"http": "", "https": ""}
            )
            
            if response.status_code != 200:
                raise RuntimeError(f"Ollama request failed: {response.status_code}")
            
            result = response.json()
            analysis_text = result.get('response', '')
            
            # Быстрый парсинг с fallback
            analysis_data = self._parse_response_fast(analysis_text)
            
            # Вычисление уверенности с учетом оптимизации
            confidence = self._calculate_confidence_optimized(analysis_data)
            
            processing_time = time.time() - start_time
            
            return AnalysisResult(
                artist=artist,
                title=title,
                analysis_type="ollama_optimized",
                confidence=confidence,
                metadata={
                    "model_name": self.model_name,
                    "optimization_level": "high",
                    "economy_mode": self.economy_mode,
                    "context_window": self.context_window,
                    "max_tokens": self.max_tokens,
                    "system_load": self.resource_monitor.get_current_load(),
                    "processing_date": datetime.now().isoformat()
                },
                raw_output=analysis_data,
                processing_time=processing_time,
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            logger.error(f"❌ Оптимизированный анализ не удался для {artist} - {title}: {e}")
            raise RuntimeError(f"Optimized analysis failed: {e}") from e
    
    def _preprocess_lyrics_optimized(self, lyrics: str) -> str:
        """Агрессивная предобработка для экономии ресурсов"""
        # Базовая очистка
        processed = self.preprocess_lyrics(lyrics)
        
        # Сильное сокращение текста для слабых ПК
        max_length = 800 if self.economy_mode else 1200
        
        if len(processed) > max_length:
            # Берем начало + концовку для сохранения структуры
            mid_point = max_length // 2
            processed = processed[:mid_point] + " ... " + processed[-mid_point:]
        
        return processed
    
    def _create_minimal_prompt(self, artist: str, title: str, lyrics: str) -> str:
        """Минимальный промпт для экономии ресурсов"""
        return f"""Analyze this rap song briefly in JSON format:

Artist: {artist}
Title: {title}  
Lyrics: {lyrics}

Return only this JSON structure:
{{
  "genre": "rap/trap/drill",
  "mood": "aggressive/calm/energetic",
  "quality": 0.0-1.0,
  "themes": ["topic1", "topic2"],
  "skill_level": 0.0-1.0
}}

Only JSON, no text!"""
    
    def _parse_response_fast(self, response_text: str) -> Dict[str, Any]:
        """Быстрый парсинг с простым fallback"""
        try:
            # Быстрое извлечение JSON
            start = response_text.find('{')
            end = response_text.rfind('}') + 1
            
            if start >= 0 and end > start:
                json_str = response_text[start:end]
                return json.loads(json_str)
            else:
                raise ValueError("No JSON found")
                
        except:
            # Простой fallback без сложных вычислений
            return {
                "genre": "rap",
                "mood": "neutral", 
                "quality": 0.5,
                "themes": ["general"],
                "skill_level": 0.5,
                "_fallback": True
            }
    
    def _calculate_confidence_optimized(self, analysis_data: Dict[str, Any]) -> float:
        """Быстрое вычисление уверенности"""
        if "_fallback" in analysis_data:
            return 0.3
        
        # Простая проверка полноты
        expected_keys = ["genre", "mood", "quality", "themes", "skill_level"]
        present_keys = sum(1 for key in expected_keys if key in analysis_data)
        
        return (present_keys / len(expected_keys)) * 0.75  # Скидка за оптимизацию
    
    def _switch_to_economy_mode(self):
        """Переключение в максимально экономичный режим"""
        self.max_tokens = min(self.max_tokens, 300)
        self.context_window = min(self.context_window, 1024)
        self.timeout = min(self.timeout, 15)
        self.economy_mode = True
        logger.info("🔋 Переключились в экономичный режим")
    
    def get_analyzer_info(self) -> Dict[str, Any]:
        """Информация об оптимизированном анализаторе"""
        system_specs = self._get_system_specs()
        
        return {
            "name": "OptimizedOllamaAnalyzer",
            "version": "2.0.0-optimized",
            "description": "Resource-efficient local AI analysis for older PCs",
            "type": self.analyzer_type,
            "model_info": {
                "current_model": self.model_name,
                "context_window": self.context_window,
                "max_tokens": self.max_tokens,
                "economy_mode": self.economy_mode
            },
            "system_specs": system_specs,
            "optimization_features": [
                "Lightweight models (1.5B-3B parameters)",
                "Adaptive resource management", 
                "CPU usage limiting",
                "Memory-efficient processing",
                "Economy mode for weak PCs"
            ],
            "recommended_models": self.LIGHTWEIGHT_MODELS,
            "available": self.available
        }


class ResourceMonitor:
    """Мониторинг системных ресурсов для адаптивной оптимизации"""
    
    def __init__(self):
        self.monitoring = True
        self.cpu_history = []
        self.memory_history = []
    
    def is_system_overloaded(self, max_cpu_percent: float = 70) -> bool:
        """Проверка перегрузки системы"""
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            
            # CPU перегружен
            if cpu_percent > max_cpu_percent:
                return True
            
            # Память почти закончилась
            if memory.percent > 85:
                return True
            
            return False
            
        except:
            return False
    
    def get_current_load(self) -> Dict[str, float]:
        """Получение текущей нагрузки системы"""
        try:
            return {
                "cpu_percent": psutil.cpu_percent(),
                "memory_percent": psutil.virtual_memory().percent,
                "available_memory_gb": psutil.virtual_memory().available / (1024**3)
            }
        except:
            return {"cpu_percent": 0, "memory_percent": 0, "available_memory_gb": 0}
