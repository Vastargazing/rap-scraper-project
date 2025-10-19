#!/usr/bin/env python3
"""
Precision Lyrics Cleaner - точечная очистка специфических проблем

Исправляет:
1. Обрезанные названия песен (s Theory -> Wesley's Theory)
2. Описания треков между названием и текстом песни (включая "XX Contributors...Read More")
3. Мусор вроде "XX Contributor<SongTitle> Lyrics"
4. Повторяющиеся названия вроде "i n t e r l u d e Lyrics" или дубли "Song Lyrics"
5. Описания треков до "Read More" (например, "The 10th track off...")
6. Мусор вроде "X Contributor<SongTitle> Lyrics"
7. Повторы названий вроде "(Nas is; Nas is; ...)" после описания
8. Мусор вроде "<SongTitle> Lyrics" в начале без дополнительных описаний
9. Новый фикс: Повторяющиеся строки в начале текста (например, "Cop another bag and smoke today")
10. Любые остатки мусора
"""

import logging
import re

import psycopg2
import yaml

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_config():
    with open("config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def connect_db():
    config = load_config()
    db_config = config["database"]

    return psycopg2.connect(
        host=db_config["host"],
        port=db_config["port"],
        database=db_config["name"],
        user=db_config["username"],
        password=db_config["password"],
    )


def precision_clean_lyrics(lyrics, song_title=None):
    """
    Точная очистка текста с учетом названия песни
    """
    if not lyrics:
        return lyrics

    original = lyrics

    # Паттерн 1: Удаляем мусор вроде "XX Contributors...", "XX Contributor<SongTitle> Lyrics", описания до "Read More" и повторы
    patterns = [
        r"^\d+\s*Contributor(?:s)?(?:\s*\w+)?\s*Lyrics\s*\n*(.*)$",  # "XX Contributor(s)<SongTitle> Lyrics" или "XX Contributors...Read More"
        r"^(?:[a-zA-Z\s]+?\s*Lyrics\s*){1,2}\n*(.*)$",  # "i n t e r l u d e Lyrics" или дубли "Song Lyrics"
        r"^.*?(?:Read More|…)\s*\n*(?:\(.*?\)\s*\n*)*(.*)$",  # Всё до "Read More" или "…" + повторы вроде "(Nas is; Nas is; ...)"
        r"^(?:[a-zA-Z\s\-]+?)\s*Lyrics\s*\n*(.*)$",  # "<SongTitle> Lyrics" в начале
        r"^((?:.*?\n){2,})\1{2,}(.*)$",  # Новый паттерн: повторяющиеся строки (3+ раза) в начале
        r"^.*?Lyrics\s*(?:\[.*?\])?\s*(?:Read More|…)\s*\n*(.*)$",  # После "Lyrics" и "Read More"
        r"^.*?Lyrics\s*(?:album|single|track|song|version|features?).*?\.\s*\n+(.*)$",  # После описания альбома/трека
        r"^.*?Lyrics\s*(?:establishes|describes|pictures|features).*?\.\s*\n+(.*)$",  # После описательного текста
        r"^.*?Lyrics\s*(?:Many|This|The).*?(?:differences|version|snippets).*?\.\s*\n+(.*)$",  # Специфические описания
        r"^.*?Lyrics\s*\.\s*\n+([A-Z].*?)$",  # После точки + перенос + заглавная буква
        r"^.*?Lyrics\s*\n\s*\n\s*([A-Z].*?)$",  # Два переноса + заглавная буква
    ]

    clean_content = lyrics
    for pattern in patterns:
        match = re.match(pattern, lyrics, re.DOTALL | re.IGNORECASE)
        if match:
            clean_content = (
                match.group(1).strip()
                if pattern != patterns[4]
                else match.group(2).strip()
            )
            if len(clean_content) > 50:  # Убеждаемся, что это содержательный текст
                break

    # Если ничего не нашли, но текст длинный, ищем первую "песенную" строку
    if clean_content == lyrics and len(lyrics) > 300:
        lines = lyrics.split("\n")
        seen_lines = set()
        repeat_count = 0
        last_line = None
        for i, line in enumerate(lines):
            line = line.strip()
            if not line or len(line) < 10:
                continue
            # Проверяем на повторы
            if line == last_line:
                repeat_count += 1
            else:
                repeat_count = 1
                last_line = line
            seen_lines.add(line)
            # Если строка не выглядит как описание, повтор или мусор, берем её как начало
            if repeat_count < 3 and not re.search(
                r"\b(?:Contributors?|version|album|single|track|song|features?|establishes|describes|pictures|Many|This|The|\(.*?;.*?\))\b",
                line,
                re.IGNORECASE,
            ):
                clean_content = "\n".join(lines[i:])
                break

    # Восстанавливаем название песни, если есть
    if song_title:
        clean_content = f"{song_title} Lyrics\n{clean_content}"

    # Финальная очистка
    clean_content = clean_content.strip()
    clean_content = re.sub(r"\n{3,}", "\n\n", clean_content)

    # Проверяем, что не потеряли слишком много контента
    if len(clean_content) < 100 and len(original) > 300:
        return original

    return clean_content


def find_tracks_with_descriptions():
    """Найти треки с описаниями между названием и текстом"""
    conn = connect_db()
    cursor = conn.cursor()

    # Обновленный запрос для поиска треков с мусором
    query = """
        SELECT id, title, lyrics 
        FROM tracks 
        WHERE lyrics IS NOT NULL 
        AND (
            lyrics ~ 'Contributor(?:s)?\\s*(?:\\w+\\s*)*Lyrics' OR
            lyrics ~ '^[a-zA-Z\\s]+?\\s*Lyrics\\s*.*\n.*Lyrics' OR
            lyrics ~ '.*(Read More|…)' OR
            lyrics ~ '\\(.*?\\;.*?\\)' OR
            lyrics ~ '^[a-zA-Z\\s\\-]+?\\s*Lyrics\\s*\n' OR
            lyrics ~ '^([^\n]*\n)\1{2,}' OR  -- Новый паттерн для повторяющихся строк (3+ раза)
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

    logger.info(
        f"=== ПРЕДВАРИТЕЛЬНЫЙ ПРОСМОТР ТОЧНОЙ ОЧИСТКИ ({len(problematic)} треков) ==="
    )

    for i, (track_id, title, lyrics) in enumerate(problematic, 1):
        print(f"\n--- Трек {i} ---")
        print(f"ID: {track_id}")
        print(f"Название: {title}")

        print("Оригинал (первые 200 символов):")
        print(f"'{lyrics[:200]}...'")

        cleaned = precision_clean_lyrics(lyrics, title)
        print("После точной очистки (первые 200 символов):")
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
                    "UPDATE tracks SET lyrics = %s WHERE id = %s", (cleaned, track_id)
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

    if len(sys.argv) > 1 and sys.argv[1] == "--preview":
        preview_precision_cleaning()
    elif len(sys.argv) > 1 and sys.argv[1] == "--execute":
        response = input("Выполнить точную очистку треков с описаниями? (yes/no): ")
        if response.lower().strip() in ["yes", "y"]:
            execute_precision_cleaning()
        else:
            logger.info("Очистка отменена")
    else:
        print("Использование:")
        print("  python precision_cleaner.py --preview   # Просмотр")
        print("  python precision_cleaner.py --execute   # Выполнить очистку")
