#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Precision Lyrics Cleaner - точечная очистка специфических проблем

Исправляет:
1. Обрезанные названия песен (s Theory -> Wesley's Theory)
2. Описания треков между названием и текстом песни
3. Любые остатки мусора
"""

import psycopg2
import re
import yaml
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config():
    with open("config.yaml", 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def connect_db():
    config = load_config()
    db_config = config['database']
    
    return psycopg2.connect(
        host=db_config['host'],
        port=db_config['port'],
        database=db_config['name'],
        user=db_config['username'],
        password=db_config['password']
    )

def precision_clean_lyrics(lyrics, song_title=None):
    """
    Точная очистка текста с учетом названия песни
    """
    if not lyrics:
        return lyrics
    
    original = lyrics
    
    # Паттерн 1: Ищем любой текст + "Lyrics" + описание + реальный текст песни
    pattern = r'^(.*?Lyrics)(.*)$'
    match = re.match(pattern, lyrics, re.DOTALL)
    
    if match:
        title_part = match.group(1).strip()
        content_part = match.group(2).strip()
        
        # Если у нас есть правильное название песни, используем его
        if song_title and not title_part.startswith(song_title):
            title_part = f"{song_title} Lyrics"
        
        # Ищем конец описания и начало текста песни
        # Обычно описание заканчивается на "Read More" или многоточие
        content_patterns = [
            r'(?:Read More|…)\s*\n+(.*)$',  # После "Read More"
            r'"[^"]*?"\s*(?:Read More|…)?\s*\n+(.*)$',  # После описания в кавычках
            r'(?:album|single|track|song|version|features?).*?\.\s*\n+(.*)$',  # После описания альбома/трека
            r'(?:establishes|describes|pictures|features).*?\.\s*\n+(.*)$',  # После описательного текста
            r'(?:Many|This|The).*?(?:differences|version|snippets).*?\.\s*\n+(.*)$',  # Специфические описания
            r'\.\s*\n+([A-Z].*?)$',  # После точки + перенос + заглавная буква
            r'\n\s*\n\s*([A-Z].*?)$',  # Два переноса + заглавная буква
        ]
        
        clean_content = ""
        for pattern in content_patterns:
            content_match = re.search(pattern, content_part, re.DOTALL | re.IGNORECASE)
            if content_match:
                clean_content = content_match.group(1).strip()
                if len(clean_content) > 50:  # Убеждаемся что это содержательный текст
                    break
        
        # Если не нашли четкого начала, но есть длинный контент
        if not clean_content and len(content_part) > 300:
            # Ищем первую строку что выглядит как начало песни
            lines = content_part.split('\n')
            for i, line in enumerate(lines):
                line = line.strip()
                # Пропускаем пустые строки и описания
                if not line or len(line) < 10:
                    continue
                # Если строка не выглядит как описание, берем её как начало
                if not re.search(r'\b(?:version|album|single|track|song|features?|establishes|describes|pictures|Many|This|The)\b', line, re.IGNORECASE):
                    clean_content = '\n'.join(lines[i:])
                    break
        
        if clean_content:
            lyrics = f"{title_part}\n{clean_content}"
        else:
            # Если ничего не нашли, оставляем как есть но убираем лишние переносы
            lyrics = f"{title_part}\n{content_part}"
    
    # Финальная очистка
    lyrics = lyrics.strip()
    lyrics = re.sub(r'\n{3,}', '\n\n', lyrics)
    
    # Проверяем что не потеряли слишком много контента
    if len(lyrics) < 100 and len(original) > 300:
        return original
    
    return lyrics

def find_tracks_with_descriptions():
    """Найти треки с описаниями между названием и текстом"""
    conn = connect_db()
    cursor = conn.cursor()
    
    # Ищем треки где после "Lyrics" идет описание
    query = """
        SELECT id, title, lyrics 
        FROM tracks 
        WHERE lyrics IS NOT NULL 
        AND (
            lyrics ~ 'Lyrics.*album.*\\.' OR
            lyrics ~ 'Lyrics.*single.*\\.' OR
            lyrics ~ 'Lyrics.*track.*\\.' OR
            lyrics ~ 'Lyrics.*version.*\\.' OR
            lyrics ~ 'Lyrics.*features.*\\.' OR
            lyrics ~ 'Lyrics.*Read More' OR
            lyrics ~ 'Lyrics.*establishes.*theme' OR
            lyrics ~ 'Lyrics.*describes.*how' OR
            lyrics ~ '^[a-z].*Lyrics'
        )
    """
    
    cursor.execute(query)
    results = cursor.fetchall()
    
    conn.close()
    return results

def preview_precision_cleaning(limit=10):
    """Предварительный просмотр точной очистки"""
    problematic = find_tracks_with_descriptions()[:limit]
    
    logger.info(f"=== ПРЕДВАРИТЕЛЬНЫЙ ПРОСМОТР ТОЧНОЙ ОЧИСТКИ ({len(problematic)} треков) ===")
    
    for i, (track_id, title, lyrics) in enumerate(problematic, 1):
        print(f"\n--- Трек {i} ---")
        print(f"ID: {track_id}")
        print(f"Название: {title}")
        
        print(f"Оригинал (первые 200 символов):")
        print(f"'{lyrics[:200]}...'")
        
        cleaned = precision_clean_lyrics(lyrics, title)
        print(f"После точной очистки (первые 200 символов):")
        print(f"'{cleaned[:200]}...'")
        
        if lyrics != cleaned:
            print("✅ БУДЕТ ИЗМЕНЕН")
        else:
            print("❌ БЕЗ ИЗМЕНЕНИЙ")

def execute_precision_cleaning():
    """Выполнить точную очистку"""
    problematic = find_tracks_with_descriptions()
    logger.info(f"Найдено {len(problematic)} треков с описаниями")
    
    if not problematic:
        logger.info("Треков с описаниями не найдено!")
        return
    
    conn = connect_db()
    cursor = conn.cursor()
    
    updated_count = 0
    
    try:
        for i, (track_id, title, lyrics) in enumerate(problematic, 1):
            cleaned = precision_clean_lyrics(lyrics, title)
            
            if cleaned != lyrics:
                logger.info(f"Очищаем трек {i}/{len(problematic)}: {title}")
                
                cursor.execute(
                    "UPDATE tracks SET lyrics = %s WHERE id = %s",
                    (cleaned, track_id)
                )
                updated_count += 1
            
            if i % 25 == 0:
                logger.info(f"Обработано {i}/{len(problematic)} треков")
        
        conn.commit()
        logger.info(f"✅ Успешно обновлено {updated_count} треков")
        
    except Exception as e:
        logger.error(f"Ошибка: {e}")
        conn.rollback()
    finally:
        conn.close()

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--preview':
        preview_precision_cleaning()
    elif len(sys.argv) > 1 and sys.argv[1] == '--execute':
        response = input("Выполнить точную очистку треков с описаниями? (yes/no): ")
        if response.lower().strip() in ['yes', 'y']:
            execute_precision_cleaning()
        else:
            logger.info("Очистка отменена")
    else:
        print("Использование:")
        print("  python precision_cleaner.py --preview   # Просмотр")
        print("  python precision_cleaner.py --execute   # Выполнить очистку")