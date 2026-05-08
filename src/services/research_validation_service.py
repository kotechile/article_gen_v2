"""
Validation service for the research rebuild.
"""

from __future__ import annotations

from datetime import datetime, timezone
import math
from typing import Any, Dict, List, Optional
from uuid import UUID

from src.integrations.dataforseo import dataforseo_api

from .research_rebuild_base_service import ResearchRebuildBaseService
from .supabase_service import SupabaseService


class ResearchValidationService(ResearchRebuildBaseService):
    """Validate candidates, track freshness, and persist SERP-backed evidence."""

    table_name = "research_validation_runs"
    serp_snapshot_table = "research_serp_snapshots"

    def __init__(self, supabase_service: Optional[SupabaseService] = None):
        super().__init__(supabase_service=supabase_service)

    async def validate_candidate(
        self,
        *,
        candidate: Dict[str, Any],
        website_context: Dict[str, Any],
        force_refresh: bool = False,
    ) -> Dict[str, Any]:
        """
        Validate a single candidate.

        First-pass implementation backed by DataForSEO metrics and SERP evidence.
        """
        candidate_type = str(candidate.get("candidate_type") or "").strip().lower()
        query = str(candidate.get("candidate_text") or "").strip()
        if not query:
            raise ValueError("candidate_text is required for validation")

        source_keywords = []
        for raw in candidate.get("source_keywords_json") or []:
            cleaned = str(raw or "").strip()
            if cleaned and cleaned.lower() not in {s.lower() for s in source_keywords}:
                source_keywords.append(cleaned)
        metric_keywords = [query] + source_keywords[:4]

        metrics_rows = await dataforseo_api.get_bulk_metrics_standard(metric_keywords)
        kd_rows = await dataforseo_api.get_keyword_difficulty(metric_keywords[:20])
        serp_rows = await dataforseo_api.get_serp_standard(query, depth=10)

        metrics_map = {str(row.get("keyword") or "").strip().lower(): row for row in metrics_rows or []}
        kd_map = {str(row.get("keyword") or "").strip().lower(): row for row in kd_rows or []}

        primary_metrics = metrics_map.get(query.lower(), {})
        primary_kd = kd_map.get(query.lower(), {})
        search_volume = int(primary_metrics.get("search_volume") or primary_kd.get("search_volume") or 0)
        cpc = float(primary_metrics.get("cpc") or primary_kd.get("cpc") or 0.0)
        kd = float(primary_metrics.get("keyword_difficulty") or primary_kd.get("keyword_difficulty") or 0.0)

        serp_weakness_score = self._compute_serp_weakness_score(serp_rows)
        intent_match_score = self._compute_intent_match_score(candidate_type, query, serp_rows)
        software_pattern_score = self._compute_software_pattern_score(query)
        feasibility_score = self._compute_feasibility_score(candidate)
        monetization_fit_score = min(1.0, cpc / 8.0) if cpc > 0 else 0.0
        volume_score = math.log1p(max(0, search_volume)) / math.log1p(5000) if search_volume > 0 else 0.0
        kd_ease_score = max(0.0, min(1.0, 1.0 - (kd / 100.0)))
        niche_drift_score = self._compute_niche_drift_score(query, website_context)
        serp_gap_score = self._compute_serp_gap_score(query, serp_rows, software_pattern_score)

        if candidate_type == "software":
            achievability_score = (
                0.25 * intent_match_score
                + 0.20 * serp_gap_score
                + 0.20 * software_pattern_score
                + 0.20 * feasibility_score
                + 0.10 * monetization_fit_score
                + 0.05 * volume_score
            )
            eligibility_passed = (
                intent_match_score >= 0.70
                and software_pattern_score >= 0.60
                and serp_gap_score >= 0.30
                and feasibility_score >= 0.60
            )
        elif candidate_type == "editorial":
            achievability_score = max(0.0, min(1.0, 0.60 * intent_match_score + 0.40 * (1.0 - niche_drift_score)))
            eligibility_passed = achievability_score >= 0.60
        else:
            achievability_score = (
                0.40 * serp_weakness_score
                + 0.25 * intent_match_score
                + 0.15 * kd_ease_score
                + 0.10 * monetization_fit_score
                + 0.10 * volume_score
                - 0.20 * niche_drift_score
            )
            achievability_score = max(0.0, min(1.0, achievability_score))
            eligibility_passed = (
                intent_match_score >= 0.65
                and serp_weakness_score >= 0.35
            )

        validation_result = {
            "validated_at": datetime.now(timezone.utc).isoformat(),
            "freshness_state": "fresh",
            "eligibility_passed": eligibility_passed,
            "intent_match_score": round(intent_match_score, 4),
            "serp_weakness_score": round(serp_weakness_score, 4),
            "serp_gap_score": round(serp_gap_score, 4),
            "software_pattern_score": round(software_pattern_score, 4),
            "feasibility_score": round(feasibility_score, 4),
            "monetization_fit_score": round(monetization_fit_score, 4),
            "volume_score": round(volume_score, 4),
            "kd_ease_score": round(kd_ease_score, 4),
            "niche_drift_score": round(niche_drift_score, 4),
            "achievability_score": round(achievability_score, 4),
            "validation_reason_codes": self._build_reason_codes(
                candidate_type=candidate_type,
                eligibility_passed=eligibility_passed,
                intent_match_score=intent_match_score,
                serp_weakness_score=serp_weakness_score,
                serp_gap_score=serp_gap_score,
                feasibility_score=feasibility_score,
                niche_drift_score=niche_drift_score,
            ),
            "validation_metadata": {
                "query": query,
                "search_volume": search_volume,
                "cpc": cpc,
                "keyword_difficulty": kd,
                "raw_serp_count": len(serp_rows or []),
                "serp_rows": serp_rows or [],
                "source_keywords": source_keywords,
                "metrics_rows": metrics_rows or [],
                "kd_rows": kd_rows or [],
            },
        }
        return validation_result

    async def validate_candidates(
        self,
        *,
        candidates: List[Dict[str, Any]],
        website_context: Dict[str, Any],
        max_candidates: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Validate multiple candidates with future batching and cost controls."""
        selected = candidates[:max_candidates] if max_candidates else candidates
        results: List[Dict[str, Any]] = []
        for candidate in selected:
            results.append(
                await self.validate_candidate(
                    candidate=candidate,
                    website_context=website_context,
                    force_refresh=False,
                )
            )
        return results

    async def list_validation_runs(
        self,
        *,
        user_id: UUID,
        project_id: UUID,
        candidate_id: Optional[UUID] = None,
        freshness_state: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """List validation runs for a project/candidate scope."""
        filters: Dict[str, Any] = {"project_id": str(project_id)}
        if candidate_id:
            filters["candidate_id"] = str(candidate_id)
        if freshness_state:
            filters["freshness_state"] = freshness_state
        return await self.list_records(
            user_id=user_id,
            filters=filters,
            order_by={"validated_at": "desc"},
        )

    async def save_validation_run(
        self,
        *,
        user_id: UUID,
        project_id: UUID,
        candidate_id: UUID,
        validation_version: str,
        payload: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Persist a validation run."""
        data = dict(payload)
        data["project_id"] = str(project_id)
        data["candidate_id"] = str(candidate_id)
        data["validation_version"] = validation_version
        return await self.create_record(user_id=user_id, data=data)

    async def create_manual_validation_run(
        self,
        *,
        user_id: UUID,
        project_id: UUID,
        candidate_id: UUID,
        validation_version: str,
        ttl_days: int = 14,
        payload: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Create a manual validation run with sane defaults."""
        data = dict(payload or {})
        validated_at = data.get("validated_at") or datetime.now(timezone.utc).isoformat()
        expires_at = data.get("expires_at") or self.compute_validation_expiry(ttl_days=ttl_days)
        data.setdefault("validated_at", validated_at)
        data.setdefault("expires_at", expires_at)
        data.setdefault("freshness_state", "fresh")
        data.setdefault("eligibility_passed", False)
        data.setdefault("validation_reason_codes", [])
        data.setdefault("validation_metadata", {})
        return await self.save_validation_run(
            user_id=user_id,
            project_id=project_id,
            candidate_id=candidate_id,
            validation_version=validation_version,
            payload=data,
        )

    async def refresh_validation_run(
        self,
        *,
        validation_run_id: UUID,
        user_id: UUID,
        ttl_days: Optional[int] = None,
        freshness_state: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Refresh a validation run's freshness metadata."""
        data: Dict[str, Any] = {
            "validated_at": datetime.now(timezone.utc).isoformat(),
        }
        if ttl_days is not None:
            data["expires_at"] = self.compute_validation_expiry(ttl_days=ttl_days)
        if freshness_state is not None:
            data["freshness_state"] = freshness_state
        return await self.update_record(record_id=validation_run_id, user_id=user_id, data=data)

    async def list_serp_snapshots(
        self,
        *,
        user_id: UUID,
        project_id: UUID,
        candidate_id: Optional[UUID] = None,
        validation_run_id: Optional[UUID] = None,
    ) -> List[Dict[str, Any]]:
        """List persisted SERP evidence."""
        filters: Dict[str, Any] = {"project_id": str(project_id)}
        if candidate_id:
            filters["candidate_id"] = str(candidate_id)
        if validation_run_id:
            filters["validation_run_id"] = str(validation_run_id)
        return await self.supabase_service.get_by_filters(
            self.serp_snapshot_table,
            filters=filters,
            user_id=user_id,
            order_by={"validated_at": "desc"},
        )

    async def save_serp_snapshot(
        self,
        *,
        user_id: UUID,
        project_id: UUID,
        candidate_id: UUID,
        validation_run_id: UUID,
        payload: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Persist Top 10 SERP evidence for traceability."""
        data = dict(payload)
        data["project_id"] = str(project_id)
        data["candidate_id"] = str(candidate_id)
        data["validation_run_id"] = str(validation_run_id)
        return await self.supabase_service.create(self.serp_snapshot_table, data=data, user_id=user_id)

    def _compute_serp_weakness_score(self, serp_rows: List[Dict[str, Any]]) -> float:
        if not serp_rows:
            return 0.5
        weak_domains = ("reddit.", "quora.", "medium.", "pinterest.", "forum", "community", "stackoverflow.")
        weak = 0
        for row in serp_rows[:10]:
            haystack = f"{row.get('title') or ''} {row.get('url') or ''}".lower()
            if any(token in haystack for token in weak_domains):
                weak += 1
        return max(0.0, min(1.0, weak / max(1, min(len(serp_rows), 10))))

    def _compute_intent_match_score(self, candidate_type: str, query: str, serp_rows: List[Dict[str, Any]]) -> float:
        q = query.lower()
        titles = " ".join(str(row.get("title") or "").lower() for row in serp_rows[:10])
        tool_words = ("calculator", "tool", "template", "checker", "generator", "estimator", "planner", "compare")
        article_words = ("guide", "how to", "best", "what is", "tips", "examples")
        if candidate_type == "software":
            query_intent = 1.0 if any(word in q for word in tool_words) else 0.6
            serp_support = 0.8 if any(word in titles for word in tool_words) else 0.55
            return max(0.0, min(1.0, (query_intent + serp_support) / 2.0))
        if candidate_type == "editorial":
            return 0.75
        query_intent = 0.9 if any(word in q for word in article_words) else 0.7
        serp_support = 0.9 if any(word in titles for word in article_words) else 0.65
        return max(0.0, min(1.0, (query_intent + serp_support) / 2.0))

    def _compute_software_pattern_score(self, query: str) -> float:
        q = query.lower()
        strong = ("calculate", "calculator", "compare", "comparison", "estimate", "estimator", "convert", "converter", "track", "planner", "checker", "generator", "template")
        medium = ("cost", "roi", "eligibility", "audit", "score", "pricing")
        if any(word in q for word in strong):
            return 0.9
        if any(word in q for word in medium):
            return 0.65
        return 0.25

    def _compute_feasibility_score(self, candidate: Dict[str, Any]) -> float:
        metadata = candidate.get("candidate_metadata") or {}
        complexity = str(metadata.get("build_complexity") or "").lower()
        text = str(candidate.get("candidate_text") or "").lower()
        if complexity == "low":
            return 0.9
        if complexity == "medium":
            return 0.65
        if complexity == "high":
            return 0.35
        if any(word in text for word in ("dashboard", "workflow", "tracker")):
            return 0.55
        if any(word in text for word in ("calculator", "template", "checker", "converter", "estimator")):
            return 0.85
        return 0.6

    def _compute_niche_drift_score(self, query: str, website_context: Dict[str, Any]) -> float:
        q_tokens = {tok for tok in query.lower().split() if len(tok) > 2}
        if not q_tokens:
            return 0.5
        context_blob = " ".join(
            str(
                website_context.get(key) or ""
            )
            for key in (
                "website_description",
                "primary_category_name",
                "primary_category_description",
                "secondary_category_name",
                "secondary_category_description",
            )
        ).lower()
        context_tokens = {tok for tok in context_blob.split() if len(tok) > 2}
        if not context_tokens:
            return 0.3
        overlap = len(q_tokens & context_tokens) / max(1, len(q_tokens))
        return max(0.0, min(1.0, 1.0 - overlap))

    def _compute_serp_gap_score(self, query: str, serp_rows: List[Dict[str, Any]], software_pattern_score: float) -> float:
        if software_pattern_score < 0.5:
            return 0.25
        titles = " ".join(str(row.get("title") or "").lower() for row in serp_rows[:10])
        tool_words = ("calculator", "tool", "template", "checker", "generator", "estimator", "planner")
        if any(word in titles for word in tool_words):
            return 0.35
        return 0.8

    def _build_reason_codes(
        self,
        *,
        candidate_type: str,
        eligibility_passed: bool,
        intent_match_score: float,
        serp_weakness_score: float,
        serp_gap_score: float,
        feasibility_score: float,
        niche_drift_score: float,
    ) -> List[str]:
        reasons: List[str] = []
        if eligibility_passed:
            reasons.append("eligible")
        else:
            reasons.append("not_eligible")
        if candidate_type == "software" and feasibility_score < 0.45:
            reasons.append("low_feasibility")
        if intent_match_score < 0.65:
            reasons.append("weak_intent_match")
        if candidate_type != "software" and serp_weakness_score < 0.35:
            reasons.append("strong_serp")
        if candidate_type == "software" and serp_gap_score < 0.30:
            reasons.append("low_serp_gap")
        if niche_drift_score > 0.55:
            reasons.append("high_niche_drift")
        return reasons
