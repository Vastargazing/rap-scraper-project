#!/usr/bin/env python3
"""
Скрипт для очистки уже сохраненных текстов в базе данных
"""
import sqlite3
import re

def clean_lyrics_advanced(lyrics: str) -> str:
    """Расширенная функция очистки текстов"""
    if not lyrics:
        return ""
    
    # Удаляем информацию о контрибьюторах (например, "81 Contributors")
    lyrics = re.sub(r"^\d+\s+Contributors.*?Lyrics", "", lyrics, flags=re.MULTILINE | re.DOTALL)
    
    # Удаляем блок с переводами (например, "TranslationsEnglishEspañol") 
    lyrics = re.sub(r"Translations[A-Za-z]+", "", lyrics, flags=re.MULTILINE)
    
    # Удаляем информацию о исполнителе и описание песни в начале
    lyrics = re.sub(r"Lyrics[A-Z].*?Read More\s*", "", lyrics, flags=re.DOTALL)
    
    # Удаляем стандартные блоки от Genius
    lyrics = re.sub(r"(?i)(Embed|Submitted by [^\n]*|Written by [^\n]*|You might also like).*$", "", lyrics, flags=re.DOTALL)
    
    # Удаляем ссылки и URL
    lyrics = re.sub(r"https?://[^\s]+", "", lyrics)
    
    # Удаляем блоки в квадратных скобках (описания или переходы)
    lyrics = re.sub(r"\[.*?\]", "", lyrics)
    
    # Удаляем множественные переносы строк
    lyrics = re.sub(r"\n{3,}", "\n\n", lyrics)
    lyrics = re.sub(r"\n{2,}", "\n", lyrics.strip())
    
    return lyrics.strip()

def test_cleaning():
    """Тест функции очистки на примере"""
    test_text = """81 ContributorsBe Free LyricsAs many other people did, including Killer Mike, J. Cole took to the internet to express his pain for the loss of Michael Brown. Cole also visited Ferguson in the aftermath of the grand jury's decision not to… Read More 
And I'm in denial, uh
And it don't take no x-ray to see right through my smile
I know"""
    
    print("ОРИГИНАЛ:")
    print(test_text[:200] + "...")
    print("\nПОСЛЕ ОЧИСТКИ:")
    cleaned = clean_lyrics_advanced(test_text)
    print(cleaned[:200] + "...")
    print(f"\nДлина до: {len(test_text)}, после: {len(cleaned)}")

def clean_database():
    """Очищает все тексты в базе данных"""
    conn = sqlite3.connect("rap_lyrics.db")
    cursor = conn.cursor()
    
    # Получаем все песни
    cursor.execute("SELECT id, artist, title, lyrics FROM songs")
    songs = cursor.fetchall()
    
    print(f"Найдено {len(songs)} песен для очистки...")
    
    updated_count = 0
    for song_id, artist, title, lyrics in songs:
        cleaned_lyrics = clean_lyrics_advanced(lyrics)
        
        # Обновляем только если текст изменился
        if cleaned_lyrics != lyrics:
            new_word_count = len(cleaned_lyrics.split()) if cleaned_lyrics else 0
            cursor.execute("""
                UPDATE songs 
                SET lyrics = ?, word_count = ? 
                WHERE id = ?
            """, (cleaned_lyrics, new_word_count, song_id))
            updated_count += 1
            print(f"✅ Обновлено: {artist} - {title}")
    
    conn.commit()
    conn.close()
    
    print(f"\n🎉 Очистка завершена! Обновлено {updated_count} из {len(songs)} песен")

if __name__ == "__main__":
    print("🧹 ТЕСТ ФУНКЦИИ ОЧИСТКИ:")
    print("=" * 50)
    test_cleaning()
    
    print("\n\n🧹 ОЧИСТКА БАЗЫ ДАННЫХ:")
    print("=" * 50)
    
    choice = input("\nОчистить все тексты в базе? (y/N): ").lower()
    if choice == 'y':
        clean_database()
    else:
        print("Очистка отменена.")
