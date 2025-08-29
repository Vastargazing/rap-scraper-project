#!/usr/bin/env python3
"""
Утилита для пакетной обработки больших объемов данных
Оптимизирована для обработки тысяч текстов с мониторингом прогресса
"""

import asyncio
import json
import time
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
import logging

# Добавляем src в Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.app import Application
from interfaces.analyzer_interface import AnalyzerType


@dataclass
class BatchProgress:
    """Прогресс пакетной обработки"""
    total: int
    completed: int
    failed: int
    start_time: float
    current_time: float
    
    @property
    def elapsed_time(self) -> float:
        return self.current_time - self.start_time
    
    @property
    def progress_percent(self) -> float:
        return (self.completed / self.total * 100) if self.total > 0 else 0
    
    @property
    def eta_seconds(self) -> float:
        if self.completed == 0:
            return 0
        avg_time_per_item = self.elapsed_time / self.completed
        remaining_items = self.total - self.completed
        return avg_time_per_item * remaining_items
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            **asdict(self),
            'progress_percent': self.progress_percent,
            'eta_seconds': self.eta_seconds,
            'elapsed_time': self.elapsed_time
        }


class BatchProcessor:
    """Класс для пакетной обработки текстов"""
    
    def __init__(self, max_workers: int = 4, progress_interval: float = 5.0):
        self.app = Application()
        self.max_workers = max_workers
        self.progress_interval = progress_interval
        self.logger = logging.getLogger(__name__)
    
    async def process_batch(
        self,
        texts: List[str],
        analyzer_type: str,
        output_file: Optional[str] = None,
        checkpoint_file: Optional[str] = None,
        resume_from_checkpoint: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Обработка пакета текстов с поддержкой checkpoint'ов
        
        Args:
            texts: Список текстов для анализа
            analyzer_type: Тип анализатора
            output_file: Файл для сохранения результатов
            checkpoint_file: Файл для сохранения промежуточных результатов
            resume_from_checkpoint: Продолжить с последнего checkpoint'а
        """
        
        # Проверяем checkpoint если нужно
        start_index = 0
        existing_results = []
        
        if resume_from_checkpoint and checkpoint_file and Path(checkpoint_file).exists():
            self.logger.info(f"Resuming from checkpoint: {checkpoint_file}")
            with open(checkpoint_file, 'r', encoding='utf-8') as f:
                checkpoint_data = json.load(f)
                existing_results = checkpoint_data.get('results', [])
                start_index = len(existing_results)
                self.logger.info(f"Resuming from index {start_index}")
        
        # Инициализация прогресса
        progress = BatchProgress(
            total=len(texts),
            completed=start_index,
            failed=0,
            start_time=time.time(),
            current_time=time.time()
        )
        
        # Получаем анализатор напрямую по строке
        analyzer = self.app.get_analyzer(analyzer_type)
        if not analyzer:
            available = self.app.list_analyzers()
            raise ValueError(f"Unknown analyzer type: {analyzer_type}. Available: {available}")
        
        self.logger.info(f"🚀 Starting batch processing: {len(texts)} texts with {analyzer_type}")
        self.logger.info(f"Max workers: {self.max_workers}")
        
        results = existing_results.copy()
        
        # Планировщик для сохранения прогресса
        last_progress_time = time.time()
        
        # Обработка в пакетах
        batch_size = self.max_workers * 2  # Оптимальный размер пакета
        
        for batch_start in range(start_index, len(texts), batch_size):
            batch_end = min(batch_start + batch_size, len(texts))
            batch_texts = texts[batch_start:batch_end]
            
            # Асинхронная обработка пакета
            batch_results = await self._process_batch_async(
                batch_texts, analyzer, batch_start
            )
            
            # Обновляем результаты и прогресс
            results.extend(batch_results)
            
            # Подсчитываем неудачные
            batch_failed = sum(1 for r in batch_results if 'error' in r)
            progress.failed += batch_failed
            progress.completed = len(results)
            progress.current_time = time.time()
            
            # Сохраняем checkpoint
            if checkpoint_file:
                await self._save_checkpoint(checkpoint_file, results, progress)
            
            # Выводим прогресс
            current_time = time.time()
            if current_time - last_progress_time >= self.progress_interval:
                self._print_progress(progress)
                last_progress_time = current_time
        
        # Финальный прогресс
        progress.current_time = time.time()
        self._print_progress(progress, final=True)
        
        # Сохраняем итоговые результаты
        if output_file:
            await self._save_results(output_file, results, progress)
        
        return results
    
    async def _process_batch_async(
        self,
        texts: List[str],
        analyzer,
        start_index: int
    ) -> List[Dict[str, Any]]:
        """Асинхронная обработка пакета текстов"""
        
        tasks = []
        for i, text in enumerate(texts):
            task = self._analyze_single_text(
                text, analyzer, start_index + i
            )
            tasks.append(task)
        
        # Выполняем все задачи параллельно
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Обрабатываем исключения
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append({
                    'index': start_index + i,
                    'error': str(result),
                    'text_preview': texts[i][:100] + "..." if len(texts[i]) > 100 else texts[i]
                })
            else:
                processed_results.append(result)
        
        return processed_results
    
    async def _analyze_single_text(
        self,
        text: str,
        analyzer,
        index: int
    ) -> Dict[str, Any]:
        """Анализ одного текста"""
        try:
            start_time = time.time()
            
            # Адаптируем к разным API анализаторов
            if hasattr(analyzer, 'analyze'):
                result = await analyzer.analyze(text)
            elif hasattr(analyzer, 'analyze_song'):
                # Старый API - передаем фиктивные данные
                analysis_result = analyzer.analyze_song("Unknown", f"Text_{index}", text)
                # Конвертируем в нужный формат
                result = {
                    'sentiment': getattr(analysis_result, 'metadata', {}).get('sentiment_analysis', {}).get('overall_sentiment', 'neutral'),
                    'confidence': getattr(analysis_result, 'confidence', 0.8),
                    'analysis_type': 'batch_analysis',
                    'metadata': getattr(analysis_result, 'metadata', {}),
                    'raw_output': getattr(analysis_result, 'raw_output', {})
                }
            else:
                raise RuntimeError(f"Analyzer has no analyze method")
            
            analysis_time = time.time() - start_time
            
            return {
                'index': index,
                'analysis_time': round(analysis_time, 3),
                'text_length': len(text),
                'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
                'result': result
            }
        except Exception as e:
            return {
                'index': index,
                'error': str(e),
                'text_preview': text[:100] + "..." if len(text) > 100 else text
            }
    
    async def _save_checkpoint(
        self,
        checkpoint_file: str,
        results: List[Dict[str, Any]],
        progress: BatchProgress
    ) -> None:
        """Сохранение checkpoint'а"""
        try:
            checkpoint_data = {
                'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
                'progress': progress.to_dict(),
                'results': results
            }
            
            checkpoint_path = Path(checkpoint_file)
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(checkpoint_path, 'w', encoding='utf-8') as f:
                json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            self.logger.warning(f"Failed to save checkpoint: {e}")
    
    async def _save_results(
        self,
        output_file: str,
        results: List[Dict[str, Any]],
        progress: BatchProgress
    ) -> None:
        """Сохранение итоговых результатов"""
        output_data = {
            'metadata': {
                'total_processed': len(results),
                'successful': len([r for r in results if 'error' not in r]),
                'failed': progress.failed,
                'total_time': progress.elapsed_time,
                'avg_time_per_item': progress.elapsed_time / len(results) if results else 0,
                'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
            },
            'results': results
        }
        
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"✅ Results saved to: {output_file}")
    
    def _print_progress(self, progress: BatchProgress, final: bool = False) -> None:
        """Вывод прогресса обработки"""
        
        if final:
            print(f"\n🎉 Batch processing completed!")
        else:
            print(f"📊 Progress: {progress.completed}/{progress.total} "
                  f"({progress.progress_percent:.1f}%)")
        
        print(f"   ✅ Successful: {progress.completed - progress.failed}")
        print(f"   ❌ Failed: {progress.failed}")
        print(f"   ⏱️  Elapsed: {progress.elapsed_time:.1f}s")
        
        if not final and progress.completed > 0:
            avg_time = progress.elapsed_time / progress.completed
            print(f"   📈 Avg time/item: {avg_time:.2f}s")
            print(f"   🕒 ETA: {progress.eta_seconds:.0f}s")
        
        print()


async def main():
    """Главная функция для демонстрации"""
    
    # Настраиваем логирование
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Создаем тестовые данные
    test_texts = [
        "This is a positive and uplifting song about hope and dreams",
        "Dark and angry lyrics expressing frustration and pain",
        "Neutral descriptive text about everyday activities",
        "Love song with romantic and emotional content",
        "Aggressive rap with explicit language and strong opinions"
    ] * 3  # 15 текстов для теста
    
    processor = BatchProcessor(max_workers=2)
    
    try:
        results = await processor.process_batch(
            texts=test_texts,
            analyzer_type="algorithmic",
            output_file="batch_results.json",
            checkpoint_file="batch_checkpoint.json",
            resume_from_checkpoint=False
        )
        
        print(f"📋 Processing completed: {len(results)} results")
        
        # Статистика
        successful = len([r for r in results if 'error' not in r])
        failed = len(results) - successful
        
        print(f"   ✅ Successful: {successful}")
        print(f"   ❌ Failed: {failed}")
        
        if successful > 0:
            avg_time = sum(r.get('analysis_time', 0) for r in results if 'analysis_time' in r) / successful
            print(f"   ⏱️  Average analysis time: {avg_time:.3f}s")
        
    except Exception as e:
        print(f"❌ Batch processing failed: {e}")


if __name__ == "__main__":
    asyncio.run(main())
