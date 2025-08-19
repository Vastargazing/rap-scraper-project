"""
Гибридная стратегия анализа: комбинация Gemini + Ollama для оптимального результата
"""
import logging
import time
from typing import Dict, Any, List, Optional
from datetime import datetime

from langchain_analyzer import GeminiLyricsAnalyzer
from ollama_analyzer import OllamaAnalyzer
from enhanced_scraper import EnhancedLyricsDatabase

logger = logging.getLogger(__name__)

class HybridAnalysisStrategy:
    """
    Умная стратегия анализа песен:
    
    1. TOP TIER (Gemini) - 1000 лучших песен
       - Популярные артисты
       - Высокое качество текстов
       - Максимальная точность анализа
    
    2. BULK PROCESSING (Ollama) - остальные 15,000+ песен
       - Локальная обработка без лимитов
       - Быстро и бесплатно
       - Хорошее качество для массовых данных
    
    3. QUALITY CONTROL - проверка и коррекция
       - Сравнение результатов на пересекающихся данных
       - Калибровка Ollama под Gemini
    """
    
    def __init__(self, db_path: str = "rap_lyrics.db"):
        self.db = EnhancedLyricsDatabase(db_path)
        
        # Пытаемся инициализировать анализаторы
        self.gemini_analyzer = None
        self.ollama_analyzer = None
        
        try:
            self.gemini_analyzer = GeminiLyricsAnalyzer()
            logger.info("✅ Gemini analyzer available")
        except Exception as e:
            logger.warning(f"⚠️ Gemini analyzer unavailable: {e}")
        
        try:
            self.ollama_analyzer = OllamaAnalyzer()
            logger.info("✅ Ollama analyzer available")
        except Exception as e:
            logger.warning(f"⚠️ Ollama analyzer unavailable: {e}")
    
    def analyze_with_hybrid_strategy(self):
        """Главная функция гибридного анализа"""
        
        logger.info("🚀 Starting hybrid analysis strategy")
        
        # Этап 1: Определяем приоритетные песни для Gemini
        priority_songs = self._get_priority_songs_for_gemini()
        logger.info(f"📊 Selected {len(priority_songs)} priority songs for Gemini")
        
        # Этап 2: Анализируем приоритетные песни через Gemini
        if self.gemini_analyzer and priority_songs:
            self._analyze_with_gemini(priority_songs)
        
        # Этап 3: Анализируем остальные песни через Ollama
        if self.ollama_analyzer:
            remaining_songs = self._get_remaining_songs()
            logger.info(f"📊 {len(remaining_songs)} songs remaining for Ollama")
            self._analyze_with_ollama(remaining_songs)
        
        # Этап 4: Калибровка и качественный контроль
        self._calibrate_and_validate()
        
        logger.info("🎉 Hybrid analysis completed!")
    
    def _get_priority_songs_for_gemini(self, limit: int = 1000) -> List[Dict[str, Any]]:
        """
        Выбираем топ песни для высококачественного анализа через Gemini
        
        Критерии приоритета:
        1. Популярные артисты (>5 песен в базе)
        2. Оптимальная длина (200-800 слов)
        3. Отсутствие анализа
        """
        
        query = """
        SELECT s.id, s.artist, s.title, s.lyrics, s.url, s.genius_id, 
               s.scraped_date, s.word_count,
               COUNT(*) OVER (PARTITION BY s.artist) as artist_song_count
        FROM songs s
        LEFT JOIN ai_analysis a ON s.id = a.song_id
        WHERE a.song_id IS NULL  -- Еще не анализировались
          AND s.word_count BETWEEN 200 AND 800  -- Оптимальная длина
          AND s.lyrics NOT LIKE '%[Instrumental]%'  -- Не инструментальные
        ORDER BY 
          artist_song_count DESC,  -- Сначала популярные артисты
          s.word_count DESC        -- Потом более содержательные
        LIMIT ?
        """
        
        cursor = self.db.conn.execute(query, (limit,))
        return [dict(row) for row in cursor.fetchall()]
    
    def _get_remaining_songs(self, batch_size: int = 1000) -> List[Dict[str, Any]]:
        """Получаем оставшиеся песни для Ollama"""
        
        query = """
        SELECT s.id, s.artist, s.title, s.lyrics, s.url, s.genius_id, 
               s.scraped_date, s.word_count
        FROM songs s
        LEFT JOIN ai_analysis a ON s.id = a.song_id
        WHERE a.song_id IS NULL  -- Еще не анализировались
          AND s.word_count >= 50   -- Минимальная длина
        ORDER BY s.word_count DESC
        LIMIT ?
        """
        
        cursor = self.db.conn.execute(query, (batch_size,))
        return [dict(row) for row in cursor.fetchall()]
    
    def _analyze_with_gemini(self, songs: List[Dict[str, Any]]):
        """Анализируем приоритетные песни через Gemini"""
        
        logger.info(f"🎯 Starting Gemini analysis for {len(songs)} priority songs")
        
        # Проверяем лимиты
        remaining_requests = self.gemini_analyzer.get_remaining_requests()
        if remaining_requests <= 0:
            logger.warning("⚠️ No Gemini requests remaining today")
            return
        
        # Batch обработка для экономии лимитов
        batch_size = 5
        batches_possible = min(remaining_requests // 1, len(songs) // batch_size)
        
        logger.info(f"📊 Can process {batches_possible} batches today")
        
        for i in range(0, min(batches_possible * batch_size, len(songs)), batch_size):
            batch = songs[i:i+batch_size]
            
            try:
                # Используем оптимизированный анализатор
                from optimized_analyzer import OptimizedGeminiAnalyzer
                optimizer = OptimizedGeminiAnalyzer()
                
                results = optimizer.analyze_song_batch(batch)
                
                # Сохраняем результаты
                for result in results:
                    song_id = next(s['id'] for s in batch if s['artist'] == result.artist and s['title'] == result.title)
                    self.db.save_ai_analysis(song_id, result)
                
                logger.info(f"✅ Gemini: processed batch {i//batch_size + 1}")
                
                # Пауза между батчами
                time.sleep(10)
                
            except Exception as e:
                logger.error(f"❌ Gemini batch failed: {e}")
                break
    
    def _analyze_with_ollama(self, songs: List[Dict[str, Any]]):
        """Анализируем остальные песни через Ollama"""
        
        logger.info(f"🔥 Starting Ollama analysis for {len(songs)} songs")
        
        batch_size = 10  # Ollama может обрабатывать больше за раз
        
        for i in range(0, len(songs), batch_size):
            batch = songs[i:i+batch_size]
            
            try:
                results = self.ollama_analyzer.analyze_song_batch(batch)
                
                # Сохраняем результаты
                for result in results:
                    song_id = next(s['id'] for s in batch if s['artist'] == result.artist and s['title'] == result.title)
                    self.db.save_ai_analysis(song_id, result)
                
                if (i // batch_size + 1) % 10 == 0:  # Логируем каждые 10 батчей
                    logger.info(f"✅ Ollama: processed {i + len(batch)} songs")
                
            except Exception as e:
                logger.error(f"❌ Ollama batch {i//batch_size + 1} failed: {e}")
                continue
    
    def _calibrate_and_validate(self):
        """Калибруем Ollama результаты относительно Gemini"""
        
        logger.info("🔬 Starting calibration and validation")
        
        # Получаем песни, проанализированные обеими моделями (если есть)
        query = """
        SELECT 
            a1.authenticity_score as gemini_auth,
            a2.authenticity_score as ollama_auth,
            a1.lyrical_creativity as gemini_creativity,
            a2.lyrical_creativity as ollama_creativity
        FROM ai_analysis a1
        JOIN ai_analysis a2 ON a1.song_id = a2.song_id
        WHERE a1.model_version LIKE '%gemini%'
          AND a2.model_version LIKE '%llama%'
        """
        
        cursor = self.db.conn.execute(query)
        comparisons = cursor.fetchall()
        
        if comparisons:
            # Рассчитываем корреляцию и корректирующие коэффициенты
            gemini_scores = [row[0] for row in comparisons]
            ollama_scores = [row[1] for row in comparisons]
            
            import numpy as np
            correlation = np.corrcoef(gemini_scores, ollama_scores)[0,1]
            
            logger.info(f"📊 Gemini-Ollama correlation: {correlation:.3f}")
            
            # Если корреляция низкая, логируем предупреждение
            if correlation < 0.7:
                logger.warning("⚠️ Low correlation between models - consider manual review")
        
        # Генерируем отчет по качеству
        self._generate_quality_report()
    
    def _generate_quality_report(self):
        """Генерируем отчет по качеству анализа"""
        
        stats = self.db.get_analysis_stats()
        
        report = f"""
        
🎵 HYBRID ANALYSIS QUALITY REPORT 🎵
{'='*50}

📊 COVERAGE:
   Total songs: {stats['total_songs']:,}
   Analyzed songs: {stats['analyzed_songs']:,}
   Coverage: {(stats['analyzed_songs']/stats['total_songs']*100):.1f}%

⭐ QUALITY METRICS:
   Avg Authenticity: {stats.get('avg_authenticity', 0):.3f}
   Avg Commercial Appeal: {stats.get('avg_commercial_appeal', 0):.3f}
   Unique Genres: {stats.get('unique_genres', 0)}

🔥 MODEL BREAKDOWN:
"""
        
        # Статистика по моделям
        cursor = self.db.conn.execute("""
            SELECT model_version, COUNT(*) as count
            FROM ai_analysis
            GROUP BY model_version
        """)
        
        for row in cursor.fetchall():
            model, count = row
            report += f"   {model}: {count:,} songs\n"
        
        logger.info(report)
        
        # Сохраняем в файл
        with open(f"hybrid_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt", "w", encoding="utf-8") as f:
            f.write(report)
    
    def get_analysis_strategy_summary(self) -> Dict[str, Any]:
        """Возвращает резюме стратегии анализа"""
        
        return {
            "strategy": "Hybrid (Gemini + Ollama)",
            "gemini_available": self.gemini_analyzer is not None,
            "ollama_available": self.ollama_analyzer is not None,
            "priority_songs_target": 1000,
            "bulk_processing": "Unlimited (Ollama)",
            "estimated_time": "1-3 days",
            "cost": "FREE (mostly local)",
            "quality": "High for priority, Good for bulk"
        }


def main():
    """Демо гибридной стратегии"""
    
    strategy = HybridAnalysisStrategy()
    
    # Показываем план
    summary = strategy.get_analysis_strategy_summary()
    
    print("🎯 HYBRID ANALYSIS STRATEGY")
    print("="*40)
    for key, value in summary.items():
        print(f"{key}: {value}")
    
    # Запускаем анализ (осторожно - это долгий процесс!)
    # strategy.analyze_with_hybrid_strategy()

if __name__ == "__main__":
    main()
