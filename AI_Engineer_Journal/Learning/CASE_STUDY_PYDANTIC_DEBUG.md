# 🔥 Case Study: AI Pipeline Debug

## Проблема
После рефакторинга кода анализатор рэп-песен перестал работать - 100% ошибок валидации Pydantic.

## Симптомы
```bash
❌ Ошибка создания анализа: 3 validation errors for LyricsAnalysis
emotional_tone: Field required [type=missing]
storytelling_type: Field required [type=missing]  
wordplay_quality: Field required [type=missing]
```

## Root Cause
**Schema Evolution Problem**: Обновили Pydantic модель с новыми обязательными полями, но забыли обновить LLM промпты.

```python
# models.py (NEW)
class LyricsAnalysis(BaseModel):
    structure: str
    emotional_tone: str      # ❌ NEW FIELD
    storytelling_type: str   # ❌ NEW FIELD
    wordplay_quality: str    # ❌ NEW FIELD

# multi_model_analyzer.py (OLD PROMPT)
"lyrics_analysis": {
    "structure": "verse-chorus",
    # ❌ Missing new fields in prompt
}
```

## Solution
### 1. Immediate Fix - Defensive Programming
```python
def _parse_analysis(self, analysis_text: str):
    lyrics_data = data.get('lyrics_analysis', {})
    
    # ✅ Fallback values
    if 'emotional_tone' not in lyrics_data:
        lyrics_data['emotional_tone'] = 'neutral'
        logger.warning("Using default for emotional_tone")
```

### 2. Long-term Fix - Updated Prompts
```python
def _create_analysis_prompt(self, artist: str, title: str, lyrics: str):
    return f"""
Return ONLY valid JSON:
{{
    "lyrics_analysis": {{
        "structure": "verse-chorus-verse",
        "emotional_tone": "mixed",        // ✅ ADDED
        "storytelling_type": "narrative", // ✅ ADDED
        "wordplay_quality": "excellent"   // ✅ ADDED
    }}
}}

REQUIRED FIELDS:
- emotional_tone: positive/negative/neutral/mixed
- storytelling_type: narrative/abstract/conversational
- wordplay_quality: basic/good/excellent
"""
```

### 3. Prevention - Contract Testing
```python
def test_prompt_model_compatibility():
    """Ensure prompt output matches Pydantic model"""
    model_fields = set(LyricsAnalysis.__fields__.keys())
    prompt_fields = extract_fields_from_prompt(create_prompt())
    
    missing = model_fields - prompt_fields
    assert not missing, f"Missing fields: {missing}"
```

## Results
**Before Fix:**
- ✅ Успешно: 0 песен
- ❌ Ошибок: 3 песни  
- 📊 Success rate: 0%

**After Fix:**
- ✅ Успешно: 19 песен  
- ❌ Ошибок: 0 песен
- 📊 Success rate: 100%
- 🎯 Authenticity: 0.766/1.0
- 🤖 AI Likelihood: 0.137/1.0 (низкая)

**Current Status:** Pipeline полностью восстановлен и продолжает анализировать новые песни

## Key Learnings
1. **Always test contracts** between components after schema changes
2. **Graceful degradation** - system should work with incomplete data
3. **Schema versioning** - use inheritance for backward compatibility

## Interview Sound Bite
*"Столкнулся с классической проблемой schema evolution в ML pipeline. LLM генерировал JSON по старому формату после обновления Pydantic модели. Решил через defensive programming + обновленные промпты + контрактные тесты для предотвращения в будущем."*

---

**Time to fix:** 30 минут  
**Impact:** Pipeline восстановлен, добавлена устойчивость к schema changes  
**Prevention:** Автоматические тесты совместимости prompt ↔ model
