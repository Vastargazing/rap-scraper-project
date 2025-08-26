#!/usr/bin/env python3
"""
Production-Ready AI Lyrics Analyzer with Monitoring & Safety Validation
Enterprise-grade система для анализа текстов с comprehensive monitoring
"""

import time
import asyncio
import logging
from typing import Dict, Optional, Any
from datetime import datetime
import json
from dataclasses import dataclass

from multi_model_analyzer import MultiModelAnalyzer, SafetyValidator, InterpretableAnalyzer
from models import SongMetadata, LyricsAnalysis, QualityMetrics

# Production logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('production_analysis.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class ProductionMetrics:
    """Production анализ метрики"""
    processing_time: float
    model_used: str
    safety_score: float
    reliability: bool
    explanation_confidence: float
    deception_risk: float
    timestamp: str

class ProductionLyricsAnalyzer:
    """Enterprise-grade анализатор с полным мониторингом"""
    
    def __init__(self):
        """Инициализация production компонентов"""
        logger.info("🚀 Initializing Production Lyrics Analyzer...")
        
        # Core components
        self.multi_analyzer = MultiModelAnalyzer()
        self.safety_validator = SafetyValidator() 
        self.interpretability = InterpretableAnalyzer(self.multi_analyzer)
        
        # Production monitoring
        self.metrics_store = []
        self.error_count = 0
        self.total_analyses = 0
        
        # Performance thresholds
        self.max_processing_time = 30.0  # seconds
        self.min_safety_score = 0.6
        self.max_deception_risk = 0.4
        
        logger.info("✅ Production Analyzer initialized successfully")
    
    async def analyze_with_monitoring(self, artist: str, title: str, lyrics: str) -> Dict[str, Any]:
        """Production-ready анализ с comprehensive мониторингом"""
        
        start_time = time.time()
        analysis_id = f"{artist}_{title}_{int(start_time)}"
        
        logger.info(f"🎵 Starting production analysis: {analysis_id}")
        
        try:
            # 1. Primary analysis с объяснениями
            logger.info("🔍 Phase 1: AI Analysis with Explanations")
            result = await self._safe_analyze_with_explanation(artist, title, lyrics)
            
            if not result:
                return await self._fallback_analysis(artist, title, lyrics, "Primary analysis failed")
            
            # 2. Safety validation
            logger.info("🛡️ Phase 2: Safety Validation")
            safety_check = await self._comprehensive_safety_check(lyrics, result)
            
            # 3. Deception detection
            logger.info("🕵️ Phase 3: Deception Detection")
            deception_risk = await self._detect_explanation_deception(lyrics, result)
            
            # 4. Performance monitoring
            processing_time = time.time() - start_time
            
            # 5. Production metrics
            metrics = ProductionMetrics(
                processing_time=processing_time,
                model_used=result.analysis.model_used,
                safety_score=safety_check["reliability_score"],
                reliability=safety_check["is_reliable"],
                explanation_confidence=result.confidence,
                deception_risk=deception_risk,
                timestamp=datetime.now().isoformat()
            )
            
            # 6. Quality gates
            quality_check = self._validate_quality_gates(metrics)
            
            # 7. Logging для мониторинга
            await self._log_analysis_metrics(analysis_id, metrics, quality_check)
            
            # 8. Build production response
            production_result = {
                "analysis_id": analysis_id,
                "analysis": result.analysis.__dict__,
                "explanation": result.explanation,
                "confidence": result.confidence,
                "decision_factors": result.decision_factors,
                "influential_phrases": result.influential_phrases,
                "safety_validation": safety_check,
                "deception_risk": deception_risk,
                "processing_time": processing_time,
                "production_ready": quality_check["passed"],
                "quality_gates": quality_check,
                "metrics": metrics.__dict__,
                "timestamp": metrics.timestamp
            }
            
            self.total_analyses += 1
            self.metrics_store.append(metrics)
            
            logger.info(f"✅ Production analysis completed: {analysis_id} in {processing_time:.2f}s")
            return production_result
            
        except Exception as e:
            self.error_count += 1
            logger.error(f"❌ Production analysis failed: {analysis_id} - {str(e)}")
            return await self._fallback_analysis(artist, title, lyrics, str(e))
    
    async def _safe_analyze_with_explanation(self, artist: str, title: str, lyrics: str):
        """Safe wrapper для анализа с объяснениями"""
        try:
            result = self.interpretability.analyze_with_explanation(artist, title, lyrics)
            return result
        except Exception as e:
            logger.warning(f"⚠️ Explanation analysis failed: {e}")
            return None
    
    async def _comprehensive_safety_check(self, lyrics: str, result) -> Dict:
        """Comprehensive safety validation"""
        try:
            # Convert analysis to dict for safety validator
            analysis_dict = {
                "genre": result.analysis.metadata.genre,
                "mood": result.analysis.metadata.mood,
                "energy_level": result.analysis.metadata.energy_level,
                "explicit_content": result.analysis.metadata.explicit_content,
                "structure": result.analysis.lyrics_analysis.structure,
                "rhyme_scheme": result.analysis.lyrics_analysis.rhyme_scheme,
                "complexity_level": result.analysis.lyrics_analysis.complexity_level,
                "main_themes": result.analysis.lyrics_analysis.main_themes,
                "authenticity_score": result.analysis.quality_metrics.authenticity_score,
                "lyrical_creativity": result.analysis.quality_metrics.lyrical_creativity,
                "commercial_appeal": result.analysis.quality_metrics.commercial_appeal,
                "uniqueness": result.analysis.quality_metrics.uniqueness,
                "overall_quality": result.analysis.quality_metrics.overall_quality,
                "ai_likelihood": result.analysis.quality_metrics.ai_likelihood
            }
            
            return self.safety_validator.validate_analysis(lyrics, analysis_dict)
            
        except Exception as e:
            logger.error(f"❌ Safety validation failed: {e}")
            return {
                "is_reliable": False,
                "reliability_score": 0.0,
                "validation_summary": f"Safety check failed: {e}"
            }
    
    async def _detect_explanation_deception(self, lyrics: str, result) -> float:
        """Detects если AI 'лжет' о своих объяснениях"""
        try:
            deception_score = 0.0
            
            # Check 1: Explanation consistency with analysis
            explained_themes = []
            for category, explanations in result.explanation.items():
                for explanation in explanations:
                    if "определен по словам" in explanation.lower():
                        # Extract claimed keywords
                        claimed_words = explanation.split("по словам:")[-1].strip()
                        explained_themes.extend(claimed_words.split(", "))
            
            # Verify if claimed words actually exist in lyrics
            lyrics_lower = lyrics.lower()
            missing_words = 0
            for word in explained_themes:
                clean_word = word.strip("'\"., ")
                if clean_word and clean_word not in lyrics_lower:
                    missing_words += 1
                    deception_score += 0.2
            
            # Check 2: Confidence vs actual evidence alignment
            claimed_confidence = result.confidence
            actual_themes = result.analysis.lyrics_analysis.main_themes
            
            # If high confidence but few actual theme indicators
            if claimed_confidence > 0.8 and len(actual_themes) < 2:
                deception_score += 0.3
            
            # Check 3: Influential phrases validation
            phrase_score = 0
            for category, phrases in result.influential_phrases.items():
                for phrase in phrases:
                    if phrase.strip() and phrase.lower() not in lyrics_lower:
                        phrase_score += 0.1
            
            deception_score += phrase_score
            
            return min(deception_score, 1.0)
            
        except Exception as e:
            logger.warning(f"⚠️ Deception detection failed: {e}")
            return 0.5  # Medium risk if detection fails
    
    def _validate_quality_gates(self, metrics: ProductionMetrics) -> Dict:
        """Production quality gates validation"""
        gates = {
            "processing_time_ok": metrics.processing_time <= self.max_processing_time,
            "safety_score_ok": metrics.safety_score >= self.min_safety_score,
            "reliability_ok": metrics.reliability,
            "deception_risk_ok": metrics.deception_risk <= self.max_deception_risk,
            "explanation_confidence_ok": metrics.explanation_confidence >= 0.5
        }
        
        gates["passed"] = all(gates.values())
        gates["score"] = sum(gates.values()) / len(gates)
        
        return gates
    
    async def _log_analysis_metrics(self, analysis_id: str, metrics: ProductionMetrics, quality_check: Dict):
        """Production metrics logging"""
        log_entry = {
            "analysis_id": analysis_id,
            "timestamp": metrics.timestamp,
            "processing_time": metrics.processing_time,
            "model_used": metrics.model_used,
            "safety_score": metrics.safety_score,
            "reliability": metrics.reliability,
            "deception_risk": metrics.deception_risk,
            "quality_gates_passed": quality_check["passed"],
            "quality_score": quality_check["score"]
        }
        
        # Log to file for monitoring systems
        logger.info(f"📊 METRICS: {json.dumps(log_entry)}")
        
        # Alert on quality issues
        if not quality_check["passed"]:
            logger.warning(f"🚨 QUALITY GATE FAILURE: {analysis_id}")
        
        if metrics.deception_risk > 0.7:
            logger.warning(f"🕵️ HIGH DECEPTION RISK: {analysis_id} - {metrics.deception_risk:.2f}")
        
        if metrics.processing_time > self.max_processing_time:
            logger.warning(f"⏰ SLOW PROCESSING: {analysis_id} - {metrics.processing_time:.2f}s")
    
    async def _fallback_analysis(self, artist: str, title: str, lyrics: str, error_reason: str) -> Dict:
        """Graceful degradation fallback"""
        logger.info(f"🔄 Executing fallback analysis for {artist} - {title}")
        
        try:
            # Simple analysis без объяснений
            simple_result = self.multi_analyzer.analyze_song(artist, title, lyrics)
            
            if simple_result:
                return {
                    "analysis_id": f"fallback_{int(time.time())}",
                    "analysis": simple_result.__dict__,
                    "explanation": {"fallback": ["Analysis completed with reduced functionality"]},
                    "confidence": 0.3,  # Lower confidence for fallback
                    "safety_validation": {"is_reliable": False, "reliability_score": 0.3},
                    "deception_risk": 0.8,  # High risk due to degraded mode
                    "production_ready": False,
                    "fallback_mode": True,
                    "error_reason": error_reason,
                    "timestamp": datetime.now().isoformat()
                }
            else:
                # Ultimate fallback - mock analysis
                return {
                    "analysis_id": f"mock_fallback_{int(time.time())}",
                    "analysis": {"error": "Complete analysis failure"},
                    "explanation": {"error": ["Analysis system unavailable"]},
                    "confidence": 0.0,
                    "production_ready": False,
                    "fallback_mode": True,
                    "error_reason": error_reason,
                    "timestamp": datetime.now().isoformat()
                }
                
        except Exception as e:
            logger.error(f"❌ Fallback analysis also failed: {e}")
            return {
                "analysis_id": f"error_{int(time.time())}",
                "error": "Complete system failure",
                "fallback_mode": True,
                "error_reason": f"{error_reason} | Fallback failed: {e}",
                "timestamp": datetime.now().isoformat()
            }
    
    def get_production_health(self) -> Dict:
        """Production health metrics"""
        if not self.metrics_store:
            return {"status": "no_data", "message": "No analyses completed yet"}
        
        recent_metrics = self.metrics_store[-10:]  # Last 10 analyses
        
        avg_processing_time = sum(m.processing_time for m in recent_metrics) / len(recent_metrics)
        avg_safety_score = sum(m.safety_score for m in recent_metrics) / len(recent_metrics)
        avg_deception_risk = sum(m.deception_risk for m in recent_metrics) / len(recent_metrics)
        reliability_rate = sum(1 for m in recent_metrics if m.reliability) / len(recent_metrics)
        
        health_score = (
            (1.0 if avg_processing_time <= self.max_processing_time else 0.5) +
            (avg_safety_score) +
            (1.0 - avg_deception_risk) +
            reliability_rate
        ) / 4
        
        return {
            "status": "healthy" if health_score > 0.7 else "degraded" if health_score > 0.4 else "unhealthy",
            "health_score": health_score,
            "total_analyses": self.total_analyses,
            "error_rate": self.error_count / max(self.total_analyses, 1),
            "avg_processing_time": avg_processing_time,
            "avg_safety_score": avg_safety_score,
            "avg_deception_risk": avg_deception_risk,
            "reliability_rate": reliability_rate,
            "last_updated": datetime.now().isoformat()
        }

# Production usage example
async def main():
    """Production analyzer demonstration"""
    
    print("🚀 Production Lyrics Analyzer Demo")
    print("=" * 50)
    
    analyzer = ProductionLyricsAnalyzer()
    
    # Example analysis
    test_lyrics = """
    Started from the bottom now we here
    All my people with me when we pull up to the spot
    Money on my mind, success in my vision
    Never gonna stop till I reach the top
    """
    
    result = await analyzer.analyze_with_monitoring(
        artist="Demo Artist",
        title="Test Track", 
        lyrics=test_lyrics
    )
    
    print(f"\n📊 PRODUCTION RESULT:")
    print(f"Analysis ID: {result.get('analysis_id')}")
    print(f"Production Ready: {result.get('production_ready')}")
    print(f"Processing Time: {result.get('processing_time', 0):.2f}s")
    print(f"Safety Score: {result.get('safety_validation', {}).get('reliability_score', 0):.2f}")
    print(f"Deception Risk: {result.get('deception_risk', 0):.2f}")
    
    # Health check
    health = analyzer.get_production_health()
    print(f"\n🏥 SYSTEM HEALTH:")
    print(f"Status: {health['status']}")
    print(f"Health Score: {health['health_score']:.2f}")
    print(f"Total Analyses: {health['total_analyses']}")

if __name__ == "__main__":
    asyncio.run(main())
