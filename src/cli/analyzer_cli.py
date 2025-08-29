#!/usr/bin/env python3
"""
CLI утилита для работы с анализаторами текстов
Поддерживает все типы анализаторов: gemma, ollama, algorithmic, hybrid
"""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional

# Добавляем src в Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.app import Application
from interfaces.analyzer_interface import AnalyzerType


class AnalyzerCLI:
    """CLI интерфейс для анализаторов"""
    
    def __init__(self):
        self.app = Application()
    
    async def analyze_text(
        self, 
        text: str, 
        analyzer_type: str,
        output_format: str = "json",
        save_to_db: bool = False
    ) -> Dict[str, Any]:
        """Анализ текста указанным анализатором"""
        
        # Получаем анализатор напрямую по строке
        analyzer = self.app.get_analyzer(analyzer_type)
        if not analyzer:
            available = self.app.list_analyzers()
            raise ValueError(f"Unknown analyzer type: {analyzer_type}. Available: {available}")
        
        print(f"📊 Analyzing with {analyzer_type} analyzer...")
        start_time = time.time()
        
        # Выполняем анализ - адаптируем к старому API
        if hasattr(analyzer, 'analyze'):
            result = await analyzer.analyze(text)
        elif hasattr(analyzer, 'analyze_song'):
            # Старый API - передаем фиктивные данные
            analysis_result = analyzer.analyze_song("Unknown", "Text Analysis", text)
            # Конвертируем в нужный формат
            result = {
                'sentiment': getattr(analysis_result, 'metadata', {}).get('sentiment_analysis', {}).get('overall_sentiment', 'neutral'),
                'confidence': getattr(analysis_result, 'confidence', 0.8),
                'analysis_type': analyzer_type,
                'metadata': getattr(analysis_result, 'metadata', {}),
                'raw_output': getattr(analysis_result, 'raw_output', {})
            }
        else:
            raise RuntimeError(f"Analyzer {analyzer_type} has no analyze method")
        
        analysis_time = time.time() - start_time
        
        # Добавляем метаданные
        result_with_meta = {
            "analyzer": analyzer_type,
            "analysis_time": round(analysis_time, 3),
            "text_length": len(text),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "result": result
        }
        
        # Сохраняем в БД если требуется
        if save_to_db:
            await self._save_to_database(text, result_with_meta)
        
        return result_with_meta
    
    async def batch_analyze(
        self,
        input_file: str,
        analyzer_type: str,
        output_file: Optional[str] = None,
        save_to_db: bool = False
    ) -> None:
        """Пакетный анализ текстов из файла"""
        
        input_path = Path(input_file)
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")
        
        # Читаем входные данные
        with open(input_path, 'r', encoding='utf-8') as f:
            if input_path.suffix == '.json':
                data = json.load(f)
                if isinstance(data, list):
                    texts = [item.get('text', str(item)) for item in data]
                else:
                    texts = [data.get('text', str(data))]
            else:
                # Текстовый файл - каждая строка отдельный текст
                texts = [line.strip() for line in f if line.strip()]
        
        print(f"🔄 Batch analyzing {len(texts)} texts with {analyzer_type}...")
        
        results = []
        for i, text in enumerate(texts, 1):
            print(f"Processing {i}/{len(texts)}...")
            try:
                result = await self.analyze_text(
                    text, analyzer_type, save_to_db=save_to_db
                )
                results.append(result)
            except Exception as e:
                print(f"Error processing text {i}: {e}")
                results.append({
                    "error": str(e),
                    "text": text[:100] + "..." if len(text) > 100 else text
                })
        
        # Сохраняем результаты
        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            print(f"✅ Results saved to: {output_file}")
        
        return results
    
    async def compare_analyzers(
        self,
        text: str,
        analyzers: Optional[list] = None,
        output_file: Optional[str] = None
    ) -> Dict[str, Any]:
        """Сравнение результатов разных анализаторов"""
        
        if analyzers is None:
            analyzers = ["algorithmic", "gemma", "ollama", "hybrid"]
        
        print(f"🔍 Comparing {len(analyzers)} analyzers...")
        
        comparison_results = {
            "text": text,
            "text_length": len(text),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "analyzers": {}
        }
        
        for analyzer_type in analyzers:
            print(f"  Running {analyzer_type} analyzer...")
            try:
                result = await self.analyze_text(text, analyzer_type)
                comparison_results["analyzers"][analyzer_type] = result
            except Exception as e:
                print(f"  Error with {analyzer_type}: {e}")
                comparison_results["analyzers"][analyzer_type] = {
                    "error": str(e)
                }
        
        # Анализ различий
        if len([r for r in comparison_results["analyzers"].values() if "error" not in r]) > 1:
            comparison_results["summary"] = self._analyze_differences(comparison_results)
        
        # Сохраняем результаты
        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(comparison_results, f, indent=2, ensure_ascii=False)
            
            print(f"✅ Comparison saved to: {output_file}")
        
        return comparison_results
    
    async def _save_to_database(self, text: str, analysis_result: Dict[str, Any]) -> None:
        """Сохранение результата анализа в базу данных"""
        try:
            # Здесь можно добавить логику сохранения в БД
            # Пока просто логируем
            print(f"💾 Saving to database: {analysis_result['analyzer']} analysis")
        except Exception as e:
            print(f"Warning: Failed to save to database: {e}")
    
    def _analyze_differences(self, comparison_results: Dict[str, Any]) -> Dict[str, Any]:
        """Анализ различий между анализаторами"""
        summary = {
            "performance": {},
            "consensus": {},
            "differences": []
        }
        
        # Анализ производительности
        for analyzer, result in comparison_results["analyzers"].items():
            if "error" not in result:
                summary["performance"][analyzer] = {
                    "analysis_time": result.get("analysis_time", 0),
                    "success": True
                }
            else:
                summary["performance"][analyzer] = {
                    "analysis_time": None,
                    "success": False,
                    "error": result["error"]
                }
        
        # TODO: Добавить анализ консенсуса и различий
        
        return summary


def main():
    """Главная функция CLI"""
    parser = argparse.ArgumentParser(
        description="CLI утилита для анализа текстов",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:

  # Анализ одного текста
  python analyzer_cli.py analyze "Your text here" --analyzer gemma

  # Пакетный анализ из файла
  python analyzer_cli.py batch input.json --analyzer hybrid --output results.json

  # Сравнение анализаторов
  python analyzer_cli.py compare "Your text here" --analyzers gemma ollama hybrid

  # Анализ с сохранением в БД
  python analyzer_cli.py analyze "Your text here" --analyzer algorithmic --save-db
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Команда analyze
    analyze_parser = subparsers.add_parser('analyze', help='Analyze single text')
    analyze_parser.add_argument('text', help='Text to analyze')
    analyze_parser.add_argument('--analyzer', '-a', required=True,
                                choices=['gemma', 'ollama', 'algorithmic', 'hybrid'],
                                help='Analyzer to use')
    analyze_parser.add_argument('--output', '-o', help='Output file (JSON)')
    analyze_parser.add_argument('--save-db', action='store_true',
                                help='Save results to database')
    analyze_parser.add_argument('--format', choices=['json', 'yaml', 'table'],
                                default='json', help='Output format')
    
    # Команда batch
    batch_parser = subparsers.add_parser('batch', help='Batch analyze texts from file')
    batch_parser.add_argument('input', help='Input file (JSON or text)')
    batch_parser.add_argument('--analyzer', '-a', required=True,
                              choices=['gemma', 'ollama', 'algorithmic', 'hybrid'],
                              help='Analyzer to use')
    batch_parser.add_argument('--output', '-o', help='Output file (JSON)')
    batch_parser.add_argument('--save-db', action='store_true',
                              help='Save results to database')
    
    # Команда compare
    compare_parser = subparsers.add_parser('compare', help='Compare multiple analyzers')
    compare_parser.add_argument('text', help='Text to analyze')
    compare_parser.add_argument('--analyzers', nargs='+',
                                choices=['gemma', 'ollama', 'algorithmic', 'hybrid'],
                                default=['algorithmic', 'hybrid'],
                                help='Analyzers to compare')
    compare_parser.add_argument('--output', '-o', help='Output file (JSON)')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Создаем CLI и выполняем команду
    cli = AnalyzerCLI()
    
    try:
        if args.command == 'analyze':
            result = asyncio.run(cli.analyze_text(
                args.text,
                args.analyzer,
                args.format,
                args.save_db
            ))
            
            if args.output:
                with open(args.output, 'w', encoding='utf-8') as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)
                print(f"✅ Result saved to: {args.output}")
            else:
                print(json.dumps(result, indent=2, ensure_ascii=False))
        
        elif args.command == 'batch':
            asyncio.run(cli.batch_analyze(
                args.input,
                args.analyzer,
                args.output,
                args.save_db
            ))
        
        elif args.command == 'compare':
            result = asyncio.run(cli.compare_analyzers(
                args.text,
                args.analyzers,
                args.output
            ))
            
            if not args.output:
                print(json.dumps(result, indent=2, ensure_ascii=False))
    
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
