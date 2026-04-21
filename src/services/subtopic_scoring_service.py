"""
Subtopic scoring service.
Scores editorial, SEO, and GEO readiness independently and returns validation state.
"""

from typing import Any, Dict, List


class SubtopicScoringService:
    """Score subtopics using editorial and keyword evidence."""

    def _score_editorial_value(self, subtopic: Dict[str, Any]) -> float:
        decision_type = (subtopic.get("decision_type") or "decision").lower()
        type_bonus = {
            "comparison": 0.22,
            "framework": 0.2,
            "checklist": 0.18,
            "audit": 0.2,
            "calculator": 0.2,
            "scenario": 0.16,
            "decision": 0.15,
            "problem": 0.14,
        }.get(decision_type, 0.12)
        title = subtopic.get("title") or ""
        problem = subtopic.get("user_problem") or ""
        summary = subtopic.get("summary") or ""
        richness = 0.2 if len(title.split()) >= 4 else 0.1
        richness += 0.15 if problem else 0.0
        richness += 0.1 if summary else 0.0
        return min(1.0, round(0.35 + type_bonus + richness, 4))

    def _score_seo_support(self, keywords: List[Dict[str, Any]]) -> float:
        if not keywords:
            return 0.0
        top = keywords[0]
        vol = int(top.get("search_volume") or 0)
        kd = int(top.get("keyword_difficulty") or 0)
        comp = (top.get("competition") or "UNKNOWN").upper()

        vol_score = min(1.0, vol / 1500.0)
        kd_score = 1.0 - min(1.0, kd / 100.0) if kd > 0 else 0.5
        comp_score = 1.0 if comp == "LOW" else 0.65 if comp == "MEDIUM" else 0.35 if comp == "HIGH" else 0.45
        return round((0.5 * vol_score) + (0.25 * kd_score) + (0.25 * comp_score), 4)

    def _score_geo_readiness(self, subtopic: Dict[str, Any], keywords: List[Dict[str, Any]]) -> float:
        hints = subtopic.get("geo_entity_hints") or []
        title = (subtopic.get("title") or "").lower()
        keyword_text = " ".join([(k.get("keyword") or "").lower() for k in keywords[:5]])
        entity_score = min(1.0, len(hints) / 4.0) if hints else 0.0
        comparison_score = 0.35 if (" vs " in title or "comparison" in title) else 0.15
        geo_presence = 0.35 if any(token in keyword_text for token in ["tax", "jurisdiction", "country", "visa", "state", "city"]) else 0.12
        return round(min(1.0, entity_score + comparison_score + geo_presence), 4)

    def determine_state(self, seo_support_score: float, editorial_value_score: float) -> str:
        if seo_support_score >= 0.45 and editorial_value_score >= 0.55:
            return "validated"
        if editorial_value_score >= 0.55:
            return "weak_seo_support"
        return "editorial_only"

    def score(self, subtopic: Dict[str, Any], keywords: List[Dict[str, Any]]) -> Dict[str, Any]:
        editorial = self._score_editorial_value(subtopic)
        seo = self._score_seo_support(keywords)
        geo = self._score_geo_readiness(subtopic, keywords)
        state = self.determine_state(seo, editorial)
        final = round((0.45 * editorial) + (0.35 * seo) + (0.20 * geo), 4)
        return {
            "editorial_value_score": editorial,
            "seo_support_score": seo,
            "geo_readiness_score": geo,
            "final_subtopic_score": final,
            "validation_state": state,
        }


subtopic_scoring_service = SubtopicScoringService()

