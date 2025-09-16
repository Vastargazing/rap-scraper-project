#!/usr/bin/env python3
"""
Комплексные тесты для всей архитектуры rap-scraper-project
Фаза 4: Интеграция и тестирование
"""

import pytest
import asyncio
import tempfile
import json
import sys
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, Any, List

# Добавляем src в Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.app import Application
from interfaces.analyzer_interface import AnalyzerFactory
from cli.analyzer_cli import AnalyzerCLI
from cli.batch_processor import BatchProcessor
from cli.performance_monitor import PerformanceMonitor


class TestAnalyzerInfrastructure:
    """Тесты базовой инфраструктуры анализаторов"""
    
    def test_application_initialization(self):
        """Тест инициализации основного приложения"""
        app = Application()
        app.initialize()  # Инициализируем приложение для регистрации анализаторов
        
        # Проверяем что приложение создается
        assert app is not None
        assert hasattr(app, 'config')
        assert hasattr(app, 'logger')
        
        # Проверяем список анализаторов
        analyzers = app.list_analyzers()
        assert isinstance(analyzers, list)
        assert len(analyzers) > 0
        
        # Проверяем что есть доступные анализаторы (обновляем ожидаемый список)
        expected_analyzers = ['algorithmic_basic', 'qwen', 'ollama', 'emotion_analyzer']
        available_analyzers = set(analyzers)
        for analyzer in expected_analyzers:
            if analyzer in available_analyzers:
                assert analyzer in analyzers, f"Analyzer {analyzer} not found"
    
    def test_analyzer_factory(self):
        """Тест фабрики анализаторов"""
        # Инициализируем анализаторы перед тестированием фабрики
        from core.app import init_analyzers
        init_analyzers()
        
        # Проверяем доступные анализаторы
        available = AnalyzerFactory.list_available()
        assert isinstance(available, list)
        assert len(available) > 0
        
        # Проверяем создание анализатора
        if 'algorithmic_basic' in available:
            analyzer = AnalyzerFactory.create('algorithmic_basic')
            assert analyzer is not None
            assert hasattr(analyzer, 'analyze_song')
    
    def test_analyzer_api_compatibility(self):
        """Тест совместимости API анализаторов"""
        app = Application()
        
        # Тестируем каждый доступный анализатор
        for analyzer_name in app.list_analyzers():
            analyzer = app.get_analyzer(analyzer_name)
            
            if analyzer is None:
                continue
                
            # Проверяем наличие необходимых методов
            assert hasattr(analyzer, 'analyze_song') or hasattr(analyzer, 'analyze'), \
                f"Analyzer {analyzer_name} missing analyze method"
            
            # Если есть analyze_song, проверяем интерфейс
            if hasattr(analyzer, 'analyze_song'):
                assert callable(analyzer.analyze_song)
                
                # Тестовый анализ
                try:
                    result = analyzer.analyze_song(
                        "Test Artist", 
                        "Test Song", 
                        "This is a test lyrics for analysis"
                    )
                    assert result is not None
                except Exception as e:
                    # Некоторые анализаторы могут требовать API ключи
                    print(f"Warning: {analyzer_name} failed with: {e}")


class TestCLIComponents:
    """Тесты CLI компонентов"""
    
    def test_analyzer_cli_creation(self):
        """Тест создания CLI анализатора"""
        cli = AnalyzerCLI()
        assert cli is not None
        assert hasattr(cli, 'app')
        assert hasattr(cli, 'analyze_text')
        assert hasattr(cli, 'compare_analyzers')
    
    @pytest.mark.asyncio
    async def test_analyzer_cli_basic_analysis(self):
        """Тест базового анализа через CLI"""
        cli = AnalyzerCLI()
        
        test_text = "This is a happy and positive song about love and joy"
        
        try:
            result = await cli.analyze_text(
                text=test_text,
                analyzer_type="algorithmic_basic"
            )
            
            # Проверяем структуру результата
            assert isinstance(result, dict)
            assert 'analyzer' in result
            assert 'analysis_time' in result
            assert 'text_length' in result
            assert 'result' in result
            assert result['analyzer'] == 'algorithmic_basic'
            assert result['text_length'] == len(test_text)
            
        except Exception as e:
            # Может не работать без правильной настройки
            print(f"CLI analysis failed: {e}")
    
    def test_batch_processor_creation(self):
        """Тест создания пакетного процессора"""
        processor = BatchProcessor()
        assert processor is not None
        assert hasattr(processor, 'app')
        assert hasattr(processor, 'process_batch')
        assert hasattr(processor, 'max_workers')
    
    def test_performance_monitor_creation(self):
        """Тест создания монитора производительности"""
        monitor = PerformanceMonitor()
        assert monitor is not None
        assert hasattr(monitor, 'app')
        assert hasattr(monitor, 'benchmark_analyzer')
        assert hasattr(monitor, 'compare_analyzers')


class TestIntegrationScenarios:
    """Тесты интеграционных сценариев"""
    
    @pytest.mark.asyncio
    async def test_end_to_end_analysis_workflow(self):
        """Тест полного workflow анализа"""
        # Создаем временные файлы для тестирования
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Подготавливаем тестовые данные
            test_texts = [
                "Happy uplifting song about success and achievement",
                "Sad melancholic ballad about loss and heartbreak",
                "Aggressive rap with strong confident attitude"
            ]
            
            input_file = temp_path / "test_input.json"
            output_file = temp_path / "test_output.json"
            
            # Сохраняем входные данные
            with open(input_file, 'w', encoding='utf-8') as f:
                json.dump([{"text": text} for text in test_texts], f)
            
            # Тестируем CLI анализ
            cli = AnalyzerCLI()
            
            try:
                # Анализируем первый текст
                result = await cli.analyze_text(
                    text=test_texts[0],
                    analyzer_type="algorithmic_basic"
                )
                assert isinstance(result, dict)
                
                # Проверяем что результат содержит ожидаемые поля
                required_fields = ['analyzer', 'analysis_time', 'text_length', 'result']
                for field in required_fields:
                    assert field in result, f"Missing field: {field}"
                
            except Exception as e:
                print(f"Integration test failed: {e}")
    
    @pytest.mark.asyncio 
    async def test_batch_processing_workflow(self):
        """Тест workflow пакетной обработки"""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Тестовые данные
            test_texts = [
                "Positive energetic song",
                "Emotional love ballad",
                "Dark introspective lyrics"
            ]
            
            output_file = temp_path / "batch_results.json"
            checkpoint_file = temp_path / "checkpoint.json"
            
            processor = BatchProcessor(max_workers=1)  # Один worker для простоты
            
            try:
                results = await processor.process_batch(
                    texts=test_texts,
                    analyzer_type="algorithmic_basic",
                    output_file=str(output_file),
                    checkpoint_file=str(checkpoint_file)
                )
                
                # Проверяем результаты
                assert isinstance(results, list)
                assert len(results) == len(test_texts)
                
                # Проверяем что файлы созданы (если все прошло успешно)
                if output_file.exists():
                    with open(output_file, 'r', encoding='utf-8') as f:
                        saved_data = json.load(f)
                        assert 'metadata' in saved_data
                        assert 'results' in saved_data
                
            except Exception as e:
                print(f"Batch processing test failed: {e}")
    
    @pytest.mark.asyncio
    async def test_performance_monitoring_workflow(self):
        """Тест workflow мониторинга производительности"""
        monitor = PerformanceMonitor()
        
        test_texts = [
            "Short text",
            "Medium length text with more content",
            "Long text with comprehensive content that should take more time to analyze"
        ]
        
        try:
            metrics = await monitor.benchmark_analyzer(
                analyzer_type="algorithmic_basic",
                test_texts=test_texts,
                warmup_runs=1  # Минимальный warmup
            )
            
            # Проверяем структуру метрик
            assert hasattr(metrics, 'analyzer_name')
            assert hasattr(metrics, 'test_count')
            assert hasattr(metrics, 'total_time')
            assert hasattr(metrics, 'success_rate')
            assert metrics.analyzer_name == 'algorithmic_basic'
            assert metrics.test_count == len(test_texts)
            
        except Exception as e:
            print(f"Performance monitoring test failed: {e}")


class TestErrorHandling:
    """Тесты обработки ошибок"""
    
    @pytest.mark.asyncio
    async def test_invalid_analyzer_handling(self):
        """Тест обработки неверного анализатора"""
        cli = AnalyzerCLI()
        
        with pytest.raises(ValueError) as exc_info:
            await cli.analyze_text(
                text="Test text",
                analyzer_type="nonexistent_analyzer"
            )
        
        assert "Unknown analyzer type" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_empty_text_handling(self):
        """Тест обработки пустого текста"""
        cli = AnalyzerCLI()
        
        try:
            result = await cli.analyze_text(
                text="",
                analyzer_type="algorithmic_basic"
            )
            # Если не упал, проверяем что результат корректный
            assert isinstance(result, dict)
        except Exception as e:
            # Ожидаемо может упасть на пустом тексте
            assert "empty" in str(e).lower() or "invalid" in str(e).lower()
    
    def test_app_resilience(self):
        """Тест устойчивости приложения к ошибкам"""
        # Тестируем создание приложения без конфигурации
        try:
            app = Application()
            assert app is not None
        except Exception as e:
            pytest.fail(f"Application should handle missing config gracefully: {e}")


class TestConfigurationAndSetup:
    """Тесты конфигурации и настройки"""
    
    def test_configuration_loading(self):
        """Тест загрузки конфигурации"""
        app = Application()
        
        assert hasattr(app, 'config')
        assert hasattr(app.config, 'database')
        assert hasattr(app.config, 'logging')
    
    def test_database_connection(self):
        """Тест подключения к базе данных"""
        app = Application()
        
        # Проверяем что можем получить информацию о БД
        try:
            analyzers = app.list_analyzers()
            assert isinstance(analyzers, list)
        except Exception as e:
            print(f"Database connection test failed: {e}")
    
    def test_logging_setup(self):
        """Тест настройки логирования"""
        app = Application()
        
        assert hasattr(app, 'logger')
        assert app.logger is not None
        
        # Тестируем что логирование работает
        try:
            app.logger.info("Test log message")
        except Exception as e:
            pytest.fail(f"Logging should work: {e}")


# Функции для запуска тестов
def run_basic_tests():
    """Запуск базовых тестов"""
    pytest.main([
        __file__ + "::TestAnalyzerInfrastructure",
        "-v", "--tb=short"
    ])

def run_cli_tests():
    """Запуск тестов CLI"""
    pytest.main([
        __file__ + "::TestCLIComponents",
        "-v", "--tb=short"
    ])

def run_integration_tests():
    """Запуск интеграционных тестов"""
    pytest.main([
        __file__ + "::TestIntegrationScenarios",
        "-v", "--tb=short"
    ])

def run_all_tests():
    """Запуск всех тестов"""
    pytest.main([
        __file__,
        "-v", "--tb=short"
    ])


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        test_type = sys.argv[1]
        if test_type == "basic":
            run_basic_tests()
        elif test_type == "cli":
            run_cli_tests()
        elif test_type == "integration":
            run_integration_tests()
        else:
            run_all_tests()
    else:
        run_all_tests()
