"""
Strategic competitive SERP mining workflow for research rebuild.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta, timezone
import math
import re
from statistics import median
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse
from uuid import UUID

from supabase_client import (
    LLM_ROLE_RESEARCH_IDEA_GENERATION,
    LLM_ROLE_RESEARCH_TOPIC_GENERATION,
)

from .llm.llm_service import llm_service
from .research_candidate_service import ResearchCandidateService
from .research_dataforseo_search_service import ResearchDataforseoSearchService
from .research_generation_service import ResearchGenerationService
from .research_job_service import ResearchJobService
from .research_keyword_pack_service import ResearchKeywordPackService
from .research_rebuild_base_service import ResearchRebuildBaseService
from .research_routing_service import ResearchRoutingService
from .research_validation_service import ResearchValidationService


STOPWORDS = {
    "a", "an", "and", "or", "the", "to", "for", "of", "in", "on", "with",
    "before", "after", "best", "does", "do", "is", "are", "should", "can",
    "value", "values", "guide", "tips", "impact",
}

EXCLUDED_LARGE_DOMAINS = {
    "tomshardware.com",
    "www.tomshardware.com",
    "forbes.com",
    "www.forbes.com",
    "wikipedia.org",
    "www.wikipedia.org",
    "investopedia.com",
    "www.investopedia.com",
    "nerdwallet.com",
    "www.nerdwallet.com",
    "bankrate.com",
    "www.bankrate.com",
    "pcmag.com",
    "www.pcmag.com",
    "zdnet.com",
    "www.zdnet.com",
}


class ResearchStrategyService(ResearchRebuildBaseService):
    """Orchestrate the competitive SERP mining workflow."""

    table_name = "research_strategy_runs"

    topic_bets_table = "research_topic_bets"
    probe_queries_table = "research_probe_queries"
    competitor_pages_table = "research_competitor_pages"
    keyword_clusters_table = "research_keyword_clusters"

    DEFAULT_LIMITS = {
        "max_bets": 6,
        "max_trend_batches": 2,
        "max_probe_queries_per_bet": 2,
        "max_surviving_bets": 3,
        "max_competitor_urls_per_bet": 2,
        "max_ranked_keywords_calls": 15,
        "max_keyword_overview_keywords": 40,
        "max_keyword_opportunities_per_bet": 10,
    }

    def __init__(
        self,
        *,
        job_service: Optional[ResearchJobService] = None,
        dataforseo_search_service: Optional[ResearchDataforseoSearchService] = None,
        candidate_service: Optional[ResearchCandidateService] = None,
        validation_service: Optional[ResearchValidationService] = None,
        routing_service: Optional[ResearchRoutingService] = None,
        keyword_pack_service: Optional[ResearchKeywordPackService] = None,
        generation_service: Optional[ResearchGenerationService] = None,
    ):
        super().__init__()
        self.job_service = job_service or ResearchJobService()
        self.dataforseo_search_service = dataforseo_search_service or ResearchDataforseoSearchService()
        self.candidate_service = candidate_service or ResearchCandidateService()
        self.validation_service = validation_service or ResearchValidationService()
        self.routing_service = routing_service or ResearchRoutingService()
        self.keyword_pack_service = keyword_pack_service or ResearchKeywordPackService()
        self.generation_service = generation_service or ResearchGenerationService()

    async def list_runs(
        self,
        *,
        user_id: UUID,
        project_id: UUID,
        topic_id: Optional[UUID] = None,
        primary_category_id: Optional[UUID] = None,
        secondary_category_id: Optional[UUID] = None,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        filters: Dict[str, Any] = {"project_id": str(project_id)}
        if topic_id:
            filters["topic_id"] = str(topic_id)
        if primary_category_id:
            filters["primary_category_id"] = str(primary_category_id)
        if secondary_category_id:
            filters["secondary_category_id"] = str(secondary_category_id)
        return await self.list_records(
            user_id=user_id,
            filters=filters,
            order_by={"created_at": "desc"},
            limit=limit,
        )

    async def get_run_detail(self, *, user_id: UUID, run_id: UUID) -> Optional[Dict[str, Any]]:
        run = await self.get_record(record_id=run_id, user_id=user_id)
        if not run:
            return None
        return await self._assemble_run_detail(user_id=user_id, run=run)

    async def list_feasible_keywords(
        self,
        *,
        user_id: UUID,
        project_id: UUID,
        topic_id: Optional[UUID] = None,
        primary_category_id: Optional[UUID] = None,
        secondary_category_id: Optional[UUID] = None,
        include_used: bool = False,
        limit: int = 200,
    ) -> List[Dict[str, Any]]:
        runs = await self.list_runs(
            user_id=user_id,
            project_id=project_id,
            topic_id=topic_id,
            primary_category_id=primary_category_id,
            secondary_category_id=secondary_category_id,
            limit=max(limit, 200),
        )
        if not runs:
            return []

        run_map = {str(run["id"]): run for run in runs}
        topic_map = {
            str(topic["id"]): topic
            for topic in await self.job_service.list_jobs(
                user_id=user_id,
                project_id=project_id,
                primary_category_id=primary_category_id,
                secondary_category_id=secondary_category_id,
                include_archived=True,
                active_only=False,
            )
        }
        clusters = await self.supabase_service.get_by_filters(
            self.keyword_clusters_table,
            filters={"project_id": str(project_id), "cluster_type": "keyword_opportunity"},
            user_id=user_id,
            order_by={"opportunity_score": "desc"},
            limit=max(limit * 4, 200),
        )
        candidates = await self.candidate_service.list_candidates(user_id=user_id, project_id=project_id)
        used_cluster_ids = {
            str((candidate.get("candidate_metadata") or {}).get("cluster_id") or "").strip()
            for candidate in candidates
            if str((candidate.get("candidate_metadata") or {}).get("cluster_id") or "").strip()
        }

        items: List[Dict[str, Any]] = []
        for cluster in clusters:
            run = run_map.get(str(cluster.get("run_id") or ""))
            if not run:
                continue
            topic = topic_map.get(str(run.get("topic_id") or "")) or {}
            metadata = dict(cluster.get("cluster_metadata") or {})
            cluster_id = str(cluster.get("id") or "")
            used_in_article = cluster_id in used_cluster_ids
            if used_in_article and not include_used:
                continue
            items.append(
                {
                    "id": cluster_id,
                    "run_id": str(cluster.get("run_id") or ""),
                    "topic_id": str(run.get("topic_id") or ""),
                    "topic_text": str(topic.get("job_text") or ""),
                    "topic_status": topic.get("status"),
                    "primary_category_id": run.get("primary_category_id"),
                    "secondary_category_id": run.get("secondary_category_id"),
                    "route": run.get("winning_route"),
                    "keyword": str(cluster.get("primary_keyword_candidate") or cluster.get("cluster_name") or ""),
                    "search_volume": metadata.get("search_volume"),
                    "keyword_difficulty": metadata.get("keyword_difficulty"),
                    "intent": metadata.get("intent"),
                    "competitor_rank": metadata.get("median_rank") or cluster.get("median_rank"),
                    "opportunity_score": cluster.get("opportunity_score"),
                    "source_domain": metadata.get("source_domain"),
                    "source_url": metadata.get("source_url"),
                    "supporting_competitor_urls": cluster.get("supporting_competitor_urls_json") or [],
                    "used_in_article": used_in_article,
                    "created_at": cluster.get("created_at"),
                }
            )

        items.sort(key=lambda item: (bool(item.get("used_in_article")), -(float(item.get("opportunity_score") or 0))))
        return items[:limit]

    async def start_strategy_run(
        self,
        *,
        user_id: UUID,
        project_id: UUID,
        primary_category_id: Optional[UUID] = None,
        secondary_category_id: Optional[UUID] = None,
        topic_id: Optional[UUID] = None,
        topic_text: Optional[str] = None,
        force_refresh: bool = False,
    ) -> Dict[str, Any]:
        topic = await self._resolve_topic(
            user_id=user_id,
            project_id=project_id,
            primary_category_id=primary_category_id,
            secondary_category_id=secondary_category_id,
            topic_id=topic_id,
            topic_text=topic_text,
        )
        if not topic:
            raise ValueError("A topic is required to start a strategy run")

        website_context = await self._build_website_context(
            user_id=user_id,
            project_id=project_id,
            primary_category_id=primary_category_id,
            secondary_category_id=secondary_category_id,
            topic=topic,
        )
        now = datetime.now(timezone.utc)
        run = await self.create_record(
            user_id=user_id,
            data={
                "project_id": str(project_id),
                "topic_id": str(topic["id"]),
                "primary_category_id": str(primary_category_id) if primary_category_id else None,
                "secondary_category_id": str(secondary_category_id) if secondary_category_id else None,
                "status": "running",
                "current_stage": "bet_generation",
                "limits_json": self.DEFAULT_LIMITS,
                "run_metadata": {
                    "topic_text": topic.get("job_text"),
                    "force_refresh": force_refresh,
                },
                "validated_at": now.isoformat(),
                "expires_at": (now + timedelta(days=14)).isoformat(),
            },
        )
        if not run:
            raise ValueError("Failed to create strategy run")

        bets = await self._generate_bets(
            user_id=user_id,
            project_id=project_id,
            run_id=UUID(run["id"]),
            topic=topic,
            website_context=website_context,
        )
        probes = await self._generate_probe_queries(
            user_id=user_id,
            project_id=project_id,
            run_id=UUID(run["id"]),
            topic_id=UUID(topic["id"]),
            bets=bets,
        )
        await self._run_trends_stage(
            user_id=user_id,
            project_id=project_id,
            probes=probes,
            force_refresh=force_refresh,
        )
        screening = await self._run_serp_screen(
            user_id=user_id,
            project_id=project_id,
            bets=bets,
            probes=probes,
            force_refresh=force_refresh,
        )
        article_bets = screening["article_bets"]
        software_bets = screening["software_bets"]
        editorial_bets = screening["editorial_bets"]
        competitor_pages = await self._mine_competitor_pages(
            user_id=user_id,
            project_id=project_id,
            run_id=UUID(run["id"]),
            article_bets=article_bets,
            probes=probes,
            force_refresh=force_refresh,
        )
        clusters = await self._build_clusters(
            user_id=user_id,
            project_id=project_id,
            run_id=UUID(run["id"]),
            topic=topic,
            article_bets=article_bets,
            competitor_pages=competitor_pages,
            force_refresh=force_refresh,
        )

        winning_route = "rejected_low_achievability"
        confidence_score = 0.0
        selected_bet_id = None
        selected_cluster_id = None
        current_stage = "screened"
        status = "screened"

        if clusters:
            best_cluster = max(clusters, key=lambda item: float(item.get("opportunity_score") or 0.0))
            winning_route = "article_ready"
            confidence_score = float(best_cluster.get("opportunity_score") or 0.0)
            selected_bet_id = best_cluster.get("bet_id")
            selected_cluster_id = best_cluster.get("id")
            current_stage = "cluster_review"
            status = "clustered"
        elif software_bets and not article_bets:
            best_bet = max(software_bets, key=lambda item: float(item.get("serp_articleability_score") or 0.0))
            winning_route = "software_ready"
            confidence_score = float(best_bet.get("serp_articleability_score") or 0.55)
            selected_bet_id = best_bet.get("id")
            current_stage = "cluster_review"
            status = "clustered"
        elif editorial_bets and not article_bets:
            best_bet = max(editorial_bets, key=lambda item: float(item.get("trend_score") or 0.0))
            winning_route = "editorial_only"
            confidence_score = float(best_bet.get("trend_score") or 0.55)
            selected_bet_id = best_bet.get("id")
            current_stage = "cluster_review"
            status = "clustered"
        else:
            status = "rejected"
            current_stage = "screened"

        await self.update_record(
            record_id=UUID(run["id"]),
            user_id=user_id,
            data={
                "status": status,
                "current_stage": current_stage,
                "selected_bet_id": selected_bet_id,
                "selected_cluster_id": selected_cluster_id,
                "winning_route": winning_route,
                "confidence_score": confidence_score,
                "run_metadata": {
                    **dict(run.get("run_metadata") or {}),
                    "article_bet_count": len(article_bets),
                    "software_bet_count": len(software_bets),
                    "editorial_bet_count": len(editorial_bets),
                    "cluster_count": len(clusters),
                },
            },
        )
        return await self.get_run_detail(user_id=user_id, run_id=UUID(run["id"])) or run

    async def rerun_stage(
        self,
        *,
        user_id: UUID,
        run_id: UUID,
        stage: str,
        force_refresh: bool = False,
    ) -> Dict[str, Any]:
        run = await self.get_record(record_id=run_id, user_id=user_id)
        if not run:
            raise ValueError("Run not found")
        topic_id = run.get("topic_id")
        project_id = UUID(str(run.get("project_id")))
        primary_category_id = UUID(str(run["primary_category_id"])) if run.get("primary_category_id") else None
        secondary_category_id = UUID(str(run["secondary_category_id"])) if run.get("secondary_category_id") else None

        if stage not in {"trends", "serp", "competitor_mining"}:
            raise ValueError("stage must be one of trends, serp, competitor_mining")

        if stage == "trends":
            await self._delete_run_artifacts(user_id=user_id, run_id=run_id, tables=[self.keyword_clusters_table, self.competitor_pages_table, self.probe_queries_table, self.topic_bets_table])
        elif stage == "serp":
            await self._delete_run_artifacts(user_id=user_id, run_id=run_id, tables=[self.keyword_clusters_table, self.competitor_pages_table, self.probe_queries_table])
        else:
            await self._delete_run_artifacts(user_id=user_id, run_id=run_id, tables=[self.keyword_clusters_table, self.competitor_pages_table])

        return await self.start_strategy_run(
            user_id=user_id,
            project_id=project_id,
            primary_category_id=primary_category_id,
            secondary_category_id=secondary_category_id,
            topic_id=UUID(str(topic_id)),
            force_refresh=force_refresh,
        )

    async def dismiss_run(
        self,
        *,
        user_id: UUID,
        run_id: UUID,
        reason: Optional[str] = None,
    ) -> Dict[str, Any]:
        run = await self.get_record(record_id=run_id, user_id=user_id)
        if not run:
            raise ValueError("Run not found")

        metadata = dict(run.get("run_metadata") or {})
        metadata["user_decision"] = "not_pursuing"
        metadata["dismissed_at"] = datetime.now(timezone.utc).isoformat()
        if reason:
            metadata["dismissal_reason"] = reason

        updated_run = await self.update_record(
            record_id=run_id,
            user_id=user_id,
            data={
                "status": "dismissed",
                "current_stage": "dismissed",
                "run_metadata": metadata,
            },
        )
        if not updated_run:
            raise ValueError("Failed to dismiss strategy run")
        return await self._assemble_run_detail(user_id=user_id, run=updated_run)

    async def select_cluster(
        self,
        *,
        user_id: UUID,
        run_id: UUID,
        cluster_id: Optional[UUID] = None,
        force_refresh: bool = False,
    ) -> Dict[str, Any]:
        run = await self.get_record(record_id=run_id, user_id=user_id)
        if not run:
            raise ValueError("Run not found")
        topic = await self.job_service.get_record(record_id=UUID(str(run["topic_id"])), user_id=user_id)
        if not topic:
            raise ValueError("Topic not found")
        website_context = await self._build_website_context(
            user_id=user_id,
            project_id=UUID(str(run["project_id"])),
            primary_category_id=UUID(str(run["primary_category_id"])) if run.get("primary_category_id") else None,
            secondary_category_id=UUID(str(run["secondary_category_id"])) if run.get("secondary_category_id") else None,
            topic=topic,
        )
        route = str(run.get("winning_route") or "")
        selected_cluster = None
        if cluster_id:
            rows = await self.supabase_service.get_by_filters(
                self.keyword_clusters_table,
                filters={"id": str(cluster_id), "run_id": str(run_id)},
                user_id=user_id,
                limit=1,
            )
            selected_cluster = rows[0] if rows else None
        elif run.get("selected_cluster_id"):
            rows = await self.supabase_service.get_by_filters(
                self.keyword_clusters_table,
                filters={"id": str(run["selected_cluster_id"]), "run_id": str(run_id)},
                user_id=user_id,
                limit=1,
            )
            selected_cluster = rows[0] if rows else None

        bet = None
        if run.get("selected_bet_id"):
            bet_rows = await self.supabase_service.get_by_filters(
                self.topic_bets_table,
                filters={"id": str(run["selected_bet_id"])},
                user_id=user_id,
                limit=1,
            )
            bet = bet_rows[0] if bet_rows else None

        cluster_keywords: List[str] = []
        if selected_cluster:
            cluster_keywords = [
                str(selected_cluster.get("primary_keyword_candidate") or "").strip(),
                *[str(item).strip() for item in (selected_cluster.get("secondary_keywords_json") or [])],
            ]
            cluster_keywords = [item for item in cluster_keywords if item]

        overview_search = None
        final_serp_search = None
        if cluster_keywords:
            overview_search = await self.dataforseo_search_service.run_search(
                user_id=user_id,
                project_id=UUID(str(run["project_id"])),
                user_job_id=UUID(str(topic["id"])),
                primary_category_id=str(run.get("primary_category_id") or "") or None,
                secondary_category_id=str(run.get("secondary_category_id") or "") or None,
                search_type="keyword_overview",
                keywords=cluster_keywords[: self.DEFAULT_LIMITS["max_keyword_overview_keywords"]],
                cache_ttl_days=30,
                force_refresh=force_refresh,
            )
            primary_query = str(selected_cluster.get("primary_keyword_candidate") or cluster_keywords[0])
            final_serp_search = await self.dataforseo_search_service.run_search(
                user_id=user_id,
                project_id=UUID(str(run["project_id"])),
                user_job_id=UUID(str(topic["id"])),
                primary_category_id=str(run.get("primary_category_id") or "") or None,
                secondary_category_id=str(run.get("secondary_category_id") or "") or None,
                search_type="serp_probe",
                query_text=primary_query,
                cache_ttl_days=14,
                limit=10,
                force_refresh=force_refresh,
            )

        final_output = await self._generate_final_output(
            topic=topic,
            bet=bet,
            cluster=selected_cluster,
            route=route,
            competitor_urls=selected_cluster.get("supporting_competitor_urls_json") if selected_cluster else [],
        )
        final_output = self._normalize_final_output(
            route=route,
            output=final_output,
            topic=topic,
            bet=bet,
            cluster=selected_cluster,
        )

        candidate_type = "seo_article"
        if route == "software_ready":
            candidate_type = "software"
        elif route == "editorial_only":
            candidate_type = "editorial"

        candidate_text = str(final_output.get("title") or bet.get("bet_text") if bet else topic.get("job_text") or "").strip()
        source_keywords_json = cluster_keywords[:15] if cluster_keywords else [str(topic.get("job_text") or "").strip()]
        candidate = await self.candidate_service.create_candidate(
            user_id=user_id,
            project_id=UUID(str(run["project_id"])),
            user_job_id=UUID(str(topic["id"])),
            candidate_type=candidate_type,
            candidate_text=candidate_text,
            normalized_candidate_text=candidate_text.lower(),
            status="validated",
            candidate_metadata={
                "creation_source": "strategy_run",
                "strategy_run_id": str(run_id),
                "topic_id": str(topic["id"]),
                "bet_id": str(bet["id"]) if bet else None,
                "cluster_id": str(selected_cluster["id"]) if selected_cluster else None,
                "route_hint": route,
                "title": final_output.get("title"),
                "slug": final_output.get("slug"),
                "outline": final_output.get("outline"),
                "competitor_urls": selected_cluster.get("supporting_competitor_urls_json") if selected_cluster else [],
            },
            source_keywords_json=source_keywords_json,
        )
        if not candidate:
            raise ValueError("Failed to create candidate for final output")

        validation_payload = await self.validation_service.validate_candidate(
            candidate=candidate,
            website_context=website_context,
            force_refresh=force_refresh,
        )
        validation_row = await self.validation_service.save_validation_run(
            user_id=user_id,
            project_id=UUID(str(run["project_id"])),
            candidate_id=UUID(str(candidate["id"])),
            validation_version="strategy_v1",
            payload=validation_payload,
        )
        if not validation_row:
            raise ValueError("Failed to persist validation run")

        if final_serp_search:
            await self.validation_service.save_serp_snapshot(
                user_id=user_id,
                project_id=UUID(str(run["project_id"])),
                candidate_id=UUID(str(candidate["id"])),
                validation_run_id=UUID(str(validation_row["id"])),
                payload={
                    "query_text": str(selected_cluster.get("primary_keyword_candidate") if selected_cluster else source_keywords_json[0]),
                    "snapshot_source": "dataforseo_serp_probe",
                    "validated_at": validation_payload.get("validated_at"),
                    "top_results_json": (final_serp_search.get("result_summary_json") or {}).get("top_items") or [],
                    "serp_summary_json": {
                        "search_id": final_serp_search.get("id"),
                        "serp_weakness_score": validation_payload.get("serp_weakness_score"),
                    },
                },
            )

        routing_row = await self.routing_service.save_routing_decision(
            user_id=user_id,
            project_id=UUID(str(run["project_id"])),
            candidate_id=UUID(str(candidate["id"])),
            validation_run_id=UUID(str(validation_row["id"])),
            route=route or "article_ready",
            route_reason_codes=validation_payload.get("validation_reason_codes") or [],
            route_metadata={
                "strategy_run_id": str(run_id),
                "cluster_id": str(selected_cluster["id"]) if selected_cluster else None,
            },
        )

        keyword_pack = None
        if cluster_keywords and route != "editorial_only":
            keyword_pack = await self.keyword_pack_service.save_keyword_pack(
                user_id=user_id,
                project_id=UUID(str(run["project_id"])),
                candidate_id=UUID(str(candidate["id"])),
                validation_run_id=UUID(str(validation_row["id"])),
                payload={
                    "primary_keyword": cluster_keywords[0],
                    "secondary_keywords_json": cluster_keywords[1:16],
                    "keyword_metrics_json": {
                        "overview_search_id": overview_search.get("id") if overview_search else None,
                        "overview_summary": overview_search.get("result_summary_json") if overview_search else {},
                    },
                    "keyword_pack_status": "ready",
                    "keyword_pack_reason_codes": [],
                },
            )

        generated_outcome = await self.generation_service.save_generated_outcome(
            user_id=user_id,
            project_id=UUID(str(run["project_id"])),
            candidate_id=UUID(str(candidate["id"])),
            payload={
                "validation_run_id": str(validation_row["id"]),
                "routing_decision_id": str(routing_row["id"]) if routing_row else None,
                "outcome_type": "software" if route == "software_ready" else ("editorial" if route == "editorial_only" else "article"),
                "status": "generated",
                "outcome_metadata": {
                    **final_output,
                    "confidence_score": run.get("confidence_score"),
                    "competitor_urls_used": selected_cluster.get("supporting_competitor_urls_json") if selected_cluster else [],
                    "serp_rationale": self._build_serp_rationale(bet, selected_cluster),
                    "keyword_pack_id": keyword_pack.get("id") if keyword_pack else None,
                },
            },
        )

        await self.update_record(
            record_id=run_id,
            user_id=user_id,
            data={
                "status": "completed",
                "current_stage": "completed",
                "selected_cluster_id": str(selected_cluster["id"]) if selected_cluster else None,
                "winning_route": route,
            },
        )
        detail = await self.get_run_detail(user_id=user_id, run_id=run_id)
        if detail is not None:
            detail["final_selection"] = {
                "candidate": candidate,
                "validation_run": validation_row,
                "routing_decision": routing_row,
                "keyword_pack": keyword_pack,
                "generated_outcome": generated_outcome,
            }
        return detail or {}

    async def _resolve_topic(
        self,
        *,
        user_id: UUID,
        project_id: UUID,
        primary_category_id: Optional[UUID],
        secondary_category_id: Optional[UUID],
        topic_id: Optional[UUID],
        topic_text: Optional[str],
    ) -> Optional[Dict[str, Any]]:
        if topic_id:
            return await self.job_service.get_record(record_id=topic_id, user_id=user_id)
        if not topic_text or not str(topic_text).strip():
            return None
        return await self.job_service.create_job(
            user_id=user_id,
            project_id=project_id,
            primary_category_id=primary_category_id,
            secondary_category_id=secondary_category_id,
            job_text=str(topic_text).strip(),
            job_type_hint="hybrid",
            job_source="manual_topic",
            status="approved",
            website_context_snapshot={},
        )

    async def _build_website_context(
        self,
        *,
        user_id: UUID,
        project_id: UUID,
        primary_category_id: Optional[UUID],
        secondary_category_id: Optional[UUID],
        topic: Dict[str, Any],
    ) -> Dict[str, Any]:
        client = self.supabase_service.get_client()
        project_response = client.table("projects").select("*").eq("id", str(project_id)).eq("user_id", str(user_id)).limit(1).execute()
        project = project_response.data[0] if project_response.data else {}
        categories_response = client.table("project_categories").select("id,name,description,level,parent_category_id").eq("project_id", str(project_id)).eq("user_id", str(user_id)).execute()
        categories = categories_response.data or []
        primary_category = next((row for row in categories if str(row.get("id") or "") == str(primary_category_id or "")), None)
        secondary_category = next((row for row in categories if str(row.get("id") or "") == str(secondary_category_id or "")), None)
        return {
            "project_name": project.get("app_name") or project.get("domain") or "Project",
            "website_description": project.get("site_description") or project.get("websiteDescription") or "",
            "primary_category_name": primary_category.get("name") if primary_category else None,
            "primary_category_description": primary_category.get("description") if primary_category else None,
            "secondary_category_name": secondary_category.get("name") if secondary_category else None,
            "secondary_category_description": secondary_category.get("description") if secondary_category else None,
            "topic_text": topic.get("job_text"),
        }

    async def _generate_bets(
        self,
        *,
        user_id: UUID,
        project_id: UUID,
        run_id: UUID,
        topic: Dict[str, Any],
        website_context: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        prompt = f"""
Return valid JSON:
{{
  "bets": [
    {{
      "bet_text": "string",
      "searcher_problem": "string",
      "article_format": "roi_guide|comparison|decision_guide|how_to|tool_evaluation|trend_analysis",
      "commercial_angle": "string",
      "buyer_or_seller_intent": "buyer|seller|researcher|operator",
      "route_hint": "article|software|editorial|hybrid"
    }}
  ]
}}

Rules:
- Generate 4-6 non-duplicate simple Google-style search seeds.
- Each bet_text should be a literal search phrase, not a headline.
- Use the topic only as a seed into the market; do not overfit tightly to the topic wording.
- Prefer broad but realistic article-search phrasing that can surface strong competitor content.
- Only mark software when the search seed clearly implies a repeated workflow or tool.
- Keep bet_text short, concrete, and searchable.

Website context:
- Project: {website_context.get("project_name")}
- Website description: {website_context.get("website_description")}
- Primary category: {website_context.get("primary_category_name")}
- Secondary category: {website_context.get("secondary_category_name")}

Topic:
- {topic.get("job_text")}
"""
        response = await llm_service.generate_json(prompt, task_role=LLM_ROLE_RESEARCH_TOPIC_GENERATION, max_tokens=2200)
        raw_bets = response.get("bets") if isinstance(response, dict) else []
        seen = set()
        payloads: List[Dict[str, Any]] = []
        for item in raw_bets or []:
            if not isinstance(item, dict):
                continue
            bet_text = " ".join(str(item.get("bet_text") or "").strip().split())
            if not bet_text:
                continue
            lowered = bet_text.lower()
            if lowered in seen:
                continue
            seen.add(lowered)
            payloads.append({
                "project_id": str(project_id),
                "run_id": str(run_id),
                "topic_id": str(topic["id"]),
                "bet_text": bet_text,
                "searcher_problem": item.get("searcher_problem"),
                "article_format": item.get("article_format"),
                "commercial_angle": item.get("commercial_angle"),
                "buyer_or_seller_intent": item.get("buyer_or_seller_intent"),
                "route_hint": item.get("route_hint") or "article",
                "status": "draft",
                "bet_metadata": {
                    "topic_text": topic.get("job_text"),
                },
            })
            if len(payloads) >= self.DEFAULT_LIMITS["max_bets"]:
                break
        return await self.supabase_service.bulk_create(self.topic_bets_table, payloads, user_id)

    async def _generate_probe_queries(
        self,
        *,
        user_id: UUID,
        project_id: UUID,
        run_id: UUID,
        topic_id: UUID,
        bets: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        payloads: List[Dict[str, Any]] = []
        for bet in bets:
            base_query = " ".join(str(bet.get("bet_text") or "").strip().split())
            prompt = f"""
Return valid JSON:
{{
  "probe_queries": [
    {{"query_text": "string", "query_role": "primary_probe|secondary_probe"}}
  ]
}}

Rules:
- The primary query should stay very close to the base seed query.
- You may add one alternate variant if it would surface a meaningfully different SERP.
- Keep both queries literal, short, and article-discovery focused.

Bet:
- base seed query: {base_query}
- format: {bet.get("article_format")}
- commercial angle: {bet.get("commercial_angle")}
"""
            response = await llm_service.generate_json(prompt, task_role=LLM_ROLE_RESEARCH_TOPIC_GENERATION, max_tokens=800)
            raw_queries = response.get("probe_queries") if isinstance(response, dict) else []
            seen = {base_query.lower()} if base_query else set()
            added = 0
            if base_query:
                payloads.append({
                    "project_id": str(project_id),
                    "run_id": str(run_id),
                    "bet_id": str(bet["id"]),
                    "query_text": base_query,
                    "query_role": "primary_probe",
                    "probe_metadata": {"topic_id": str(topic_id), "source": "seed_query"},
                })
                added = 1
            for item in raw_queries or []:
                if not isinstance(item, dict):
                    continue
                query_text = " ".join(str(item.get("query_text") or "").strip().split())
                if not query_text:
                    continue
                key = query_text.lower()
                if key in seen:
                    continue
                seen.add(key)
                payloads.append({
                    "project_id": str(project_id),
                    "run_id": str(run_id),
                    "bet_id": str(bet["id"]),
                    "query_text": query_text,
                    "query_role": item.get("query_role") if item.get("query_role") in {"primary_probe", "secondary_probe"} else ("primary_probe" if added == 0 else "secondary_probe"),
                    "probe_metadata": {"topic_id": str(topic_id), "source": "serp_variant"},
                })
                added += 1
                if added >= self.DEFAULT_LIMITS["max_probe_queries_per_bet"]:
                    break
            if added == 0:
                payloads.append({
                    "project_id": str(project_id),
                    "run_id": str(run_id),
                    "bet_id": str(bet["id"]),
                    "query_text": str(bet.get("bet_text") or ""),
                    "query_role": "primary_probe",
                    "probe_metadata": {"topic_id": str(topic_id)},
                })
        return await self.supabase_service.bulk_create(self.probe_queries_table, payloads, user_id)

    async def _run_trends_stage(
        self,
        *,
        user_id: UUID,
        project_id: UUID,
        probes: List[Dict[str, Any]],
        force_refresh: bool,
    ) -> None:
        grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for probe in probes:
            grouped[str(probe.get("bet_id"))].append(probe)
        bet_representatives = []
        for items in grouped.values():
            probe = items[0]
            bet_representatives.append((probe["bet_id"], str(probe.get("query_text") or "").strip()))
        for batch_start in range(0, min(len(bet_representatives), self.DEFAULT_LIMITS["max_trend_batches"] * 5), 5):
            batch = bet_representatives[batch_start: batch_start + 5]
            keywords = [item[1] for item in batch]
            search = await self.dataforseo_search_service.run_search(
                user_id=user_id,
                project_id=project_id,
                search_type="google_trends",
                keywords=keywords,
                cache_ttl_days=30,
                force_refresh=force_refresh,
            )
            for bet_id, keyword in batch:
                trend_score, trend_metadata = self._score_trend(keyword, search)
                bet_rows = await self.supabase_service.get_by_filters(
                    self.topic_bets_table,
                    filters={"id": str(bet_id)},
                    user_id=user_id,
                    limit=1,
                )
                if not bet_rows:
                    continue
                bet = bet_rows[0]
                bet_metadata = dict(bet.get("bet_metadata") or {})
                bet_metadata["trend_summary"] = trend_metadata
                await self.supabase_service.update(
                    self.topic_bets_table,
                    UUID(str(bet_id)),
                    {"trend_score": trend_score, "bet_metadata": bet_metadata},
                    user_id,
                )
                for probe in grouped[bet_id]:
                    await self.supabase_service.update(
                        self.probe_queries_table,
                        UUID(str(probe["id"])),
                        {"trend_search_id": str(search["id"])},
                        user_id,
                    )

    async def _run_serp_screen(
        self,
        *,
        user_id: UUID,
        project_id: UUID,
        bets: List[Dict[str, Any]],
        probes: List[Dict[str, Any]],
        force_refresh: bool,
    ) -> Dict[str, List[Dict[str, Any]]]:
        probes_by_bet: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for probe in probes:
            probes_by_bet[str(probe.get("bet_id"))].append(probe)
        article_bets: List[Dict[str, Any]] = []
        software_bets: List[Dict[str, Any]] = []
        editorial_bets: List[Dict[str, Any]] = []
        for bet in bets:
            probe_rows = probes_by_bet.get(str(bet["id"]), [])
            best_probe_score = -1.0
            best_classification = {}
            for probe in probe_rows[: self.DEFAULT_LIMITS["max_probe_queries_per_bet"]]:
                search = await self.dataforseo_search_service.run_search(
                    user_id=user_id,
                    project_id=project_id,
                    search_type="serp_probe",
                    query_text=str(probe.get("query_text") or ""),
                    cache_ttl_days=14,
                    limit=10,
                    force_refresh=force_refresh,
                )
                top_items = (search.get("result_summary_json") or {}).get("top_items") or []
                classification = self._classify_serp(
                    query_text=str(probe.get("query_text") or ""),
                    rows=top_items,
                    article_format=str(bet.get("article_format") or ""),
                    route_hint=str(bet.get("route_hint") or "article"),
                )
                await self.supabase_service.update(
                    self.probe_queries_table,
                    UUID(str(probe["id"])),
                    {
                        "serp_search_id": str(search["id"]),
                        "articleability_passed": bool(classification["articleability_passed"]),
                        "serp_classification": classification["classification"],
                        "probe_metadata": {**dict(probe.get("probe_metadata") or {}), **classification},
                    },
                    user_id,
                )
                if float(classification["articleability_score"]) > best_probe_score:
                    best_probe_score = float(classification["articleability_score"])
                    best_classification = classification
            route_hint = str(bet.get("route_hint") or "article")
            bet_status = "killed"
            reason_codes = list(best_classification.get("reason_codes") or [])
            article_score = float(best_classification.get("articleability_score") or 0.0)
            if best_classification.get("articleability_passed") or (
                best_classification.get("classification") == "mixed"
                and article_score >= 0.4
                and float(best_classification.get("serp_weakness_score") or 0.0) >= 0.3
            ):
                bet_status = "survived"
                if not reason_codes:
                    reason_codes = ["article_candidate"]
            elif best_classification.get("classification") == "tool_dominant" or route_hint == "software":
                bet_status = "survived"
                reason_codes = ["software_serp"]
            elif float(bet.get("trend_score") or 0.0) >= 0.55:
                bet_status = "survived"
                reason_codes = ["editorial_candidate"]

            updated = await self.supabase_service.update(
                self.topic_bets_table,
                UUID(str(bet["id"])),
                {
                    "status": bet_status,
                    "serp_articleability_score": best_classification.get("articleability_score"),
                    "serp_weakness_score": best_classification.get("serp_weakness_score"),
                    "intent_fit_score": best_classification.get("intent_fit_score"),
                    "article_fit_score": best_classification.get("article_format_fit"),
                    "reason_codes": reason_codes,
                    "route_hint": best_classification.get("route_hint") or route_hint,
                },
                user_id,
            )
            if not updated:
                continue
            if best_classification.get("articleability_passed") or (
                best_classification.get("classification") == "mixed"
                and article_score >= 0.4
                and float(best_classification.get("serp_weakness_score") or 0.0) >= 0.3
            ):
                article_bets.append(updated)
            elif best_classification.get("classification") == "tool_dominant" or route_hint == "software":
                software_bets.append(updated)
            elif bet_status == "survived":
                editorial_bets.append(updated)
        article_bets = sorted(
            article_bets,
            key=lambda item: float(item.get("serp_articleability_score") or 0.0) + float(item.get("trend_score") or 0.0),
            reverse=True,
        )[: self.DEFAULT_LIMITS["max_surviving_bets"]]
        return {
            "article_bets": article_bets,
            "software_bets": software_bets,
            "editorial_bets": editorial_bets,
        }

    async def _mine_competitor_pages(
        self,
        *,
        user_id: UUID,
        project_id: UUID,
        run_id: UUID,
        article_bets: List[Dict[str, Any]],
        probes: List[Dict[str, Any]],
        force_refresh: bool,
    ) -> List[Dict[str, Any]]:
        probes_by_bet: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for probe in probes:
            probes_by_bet[str(probe.get("bet_id"))].append(probe)
        competitor_pages: List[Dict[str, Any]] = []
        ranked_keyword_calls = 0
        for bet in article_bets:
            bet_probes = probes_by_bet.get(str(bet["id"]), [])
            probe_rows = await self.supabase_service.get_by_filters(
                self.probe_queries_table,
                filters={"bet_id": str(bet["id"])},
                user_id=user_id,
            )
            probe_by_id = {str(item["id"]): item for item in probe_rows}
            probe_candidates = [probe_by_id.get(str(item["id"])) or item for item in bet_probes]
            best_probe = max(
                probe_candidates,
                key=lambda item: float((item.get("probe_metadata") or {}).get("articleability_score") or 0.0),
                default=None,
            )
            if not best_probe:
                continue
            serp_candidates: List[Dict[str, Any]] = []
            for probe in probe_candidates:
                serp_search_id = probe.get("serp_search_id")
                if not serp_search_id:
                    continue
                search_rows = await self.dataforseo_search_service.list_records(
                    user_id=user_id,
                    filters={"id": str(serp_search_id)},
                    order_by={"searched_at": "desc"},
                    limit=1,
                )
                if not search_rows:
                    continue
                serp_rows = (search_rows[0].get("result_summary_json") or {}).get("top_items") or []
                for page in self._extract_article_competitor_urls(serp_rows):
                    serp_candidates.append({**page, "probe_query_id": str(probe["id"])})

            selected_targets = self._select_attractive_competitor_targets(serp_candidates)
            for page in selected_targets[: self.DEFAULT_LIMITS["max_competitor_urls_per_bet"]]:
                page_row = await self.supabase_service.create(
                    self.competitor_pages_table,
                    data={
                        "project_id": str(project_id),
                        "run_id": str(run_id),
                        "bet_id": str(bet["id"]),
                        "probe_query_id": str(page.get("probe_query_id") or best_probe["id"]),
                        "url": page["url"],
                        "title": page.get("title"),
                        "domain": page.get("domain"),
                        "page_type": "article",
                        "rank_group": page.get("rank_group"),
                        "selected_for_mining": True,
                        "page_metadata": {
                            "analysis_target": page.get("analysis_target") or page.get("domain") or page.get("url"),
                            "domain_hits": page.get("domain_hits"),
                            "source_urls": page.get("source_urls") or [page.get("url")],
                        },
                    },
                    user_id=user_id,
                )
                if not page_row:
                    continue
                competitor_pages.append(page_row)
                if ranked_keyword_calls >= self.DEFAULT_LIMITS["max_ranked_keywords_calls"]:
                    continue
                analysis_target = str(
                    (page_row.get("page_metadata") or {}).get("analysis_target")
                    or page.get("analysis_target")
                    or page.get("domain")
                    or page.get("url")
                    or ""
                ).strip()
                ranked_search = await self.dataforseo_search_service.run_search(
                    user_id=user_id,
                    project_id=project_id,
                    search_type="ranked_keywords",
                    target=analysis_target,
                    cache_ttl_days=30,
                    limit=100,
                    force_refresh=force_refresh,
                )
                ranked_keyword_calls += 1
                await self.supabase_service.update(
                    self.competitor_pages_table,
                    UUID(str(page_row["id"])),
                    {
                        "mined_search_id": str(ranked_search["id"]),
                        "page_metadata": {
                            **(page_row.get("page_metadata") or {}),
                            "ranked_keywords_count": (ranked_search.get("result_summary_json") or {}).get("result_count"),
                        },
                    },
                    user_id,
                )
        return competitor_pages

    async def _build_clusters(
        self,
        *,
        user_id: UUID,
        project_id: UUID,
        run_id: UUID,
        topic: Dict[str, Any],
        article_bets: List[Dict[str, Any]],
        competitor_pages: List[Dict[str, Any]],
        force_refresh: bool,
    ) -> List[Dict[str, Any]]:
        pages_by_bet: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for page in competitor_pages:
            pages_by_bet[str(page.get("bet_id"))].append(page)
        created_clusters: List[Dict[str, Any]] = []
        for bet in article_bets:
            page_rows = pages_by_bet.get(str(bet["id"]), [])
            harvested_keywords: List[Dict[str, Any]] = []
            for page in page_rows:
                search_id = page.get("mined_search_id")
                if not search_id:
                    continue
                search_rows = await self.dataforseo_search_service.list_records(
                    user_id=user_id,
                    filters={"id": str(search_id)},
                    order_by={"searched_at": "desc"},
                    limit=1,
                )
                if not search_rows:
                    continue
                for item in (search_rows[0].get("result_summary_json") or {}).get("top_items") or []:
                    if not isinstance(item, dict):
                        continue
                    harvested_keywords.append(
                        {
                            **item,
                            "source_url": page.get("url"),
                            "source_domain": page.get("domain"),
                            "source_title": page.get("title"),
                        }
                    )
            shortlisted_keywords = self._qualify_competitor_keywords(
                topic_text=str(topic.get("job_text") or ""),
                bet_text=str(bet.get("bet_text") or ""),
                rows=harvested_keywords,
            )
            if not shortlisted_keywords:
                continue
            overview_search = await self.dataforseo_search_service.run_search(
                user_id=user_id,
                project_id=project_id,
                user_job_id=UUID(str(topic["id"])),
                primary_category_id=None,
                secondary_category_id=None,
                search_type="keyword_overview",
                keywords=[str(row.get("keyword") or "").strip() for row in shortlisted_keywords[: self.DEFAULT_LIMITS["max_keyword_overview_keywords"]] if str(row.get("keyword") or "").strip()],
                cache_ttl_days=30,
                force_refresh=force_refresh,
            )
            overview_items = (overview_search.get("result_summary_json") or {}).get("top_items") or []
            qualified_rows = self._merge_keyword_overview_metrics(shortlisted_keywords, overview_items)
            qualified_rows = self._filter_really_competitive_keywords(qualified_rows)
            keyword_opportunities = self._materialize_keyword_opportunities(
                rows=qualified_rows,
                bet=bet,
            )
            for cluster in keyword_opportunities:
                row = await self.supabase_service.create(
                    self.keyword_clusters_table,
                    data={
                        "project_id": str(project_id),
                        "run_id": str(run_id),
                        "bet_id": str(bet["id"]),
                        "cluster_name": cluster["cluster_name"],
                        "primary_keyword_candidate": cluster["primary_keyword_candidate"],
                        "secondary_keywords_json": cluster["secondary_keywords"],
                        "supporting_competitor_urls_json": cluster["supporting_urls"],
                        "cluster_type": cluster["cluster_type"],
                        "competitor_support_score": cluster["competitor_support_score"],
                        "kd_median_score": cluster["kd_median_score"],
                        "commercial_value_score": cluster["commercial_value_score"],
                        "trend_score": float(bet.get("trend_score") or 0.5),
                        "articleability_score": float(bet.get("serp_articleability_score") or 0.0),
                        "serp_weakness_score": float(bet.get("serp_weakness_score") or 0.0),
                        "article_fit_score": float(bet.get("article_fit_score") or 0.0),
                        "opportunity_score": self._compute_cluster_opportunity_score(
                            serp_weakness_score=float(bet.get("serp_weakness_score") or 0.0),
                            competitor_support_score=cluster["competitor_support_score"],
                            kd_median_score=cluster["kd_median_score"],
                            commercial_value_score=cluster["commercial_value_score"],
                            trend_score=float(bet.get("trend_score") or 0.5),
                            article_fit_score=float(bet.get("article_fit_score") or 0.0),
                        ),
                        "status": "survived",
                        "cluster_metadata": {
                            "keyword_count": cluster["keyword_count"],
                            "median_rank": cluster["median_rank"],
                            "qualified_keyword_count": len(qualified_rows),
                            "overview_search_id": overview_search.get("id") if overview_search else None,
                            **(cluster.get("cluster_metadata") or {}),
                        },
                    },
                    user_id=user_id,
                )
                if row:
                    created_clusters.append(row)
        return created_clusters

    async def _assemble_run_detail(self, *, user_id: UUID, run: Dict[str, Any]) -> Dict[str, Any]:
        topic = await self.job_service.get_record(record_id=UUID(str(run["topic_id"])), user_id=user_id)
        bets = await self.supabase_service.get_by_filters(self.topic_bets_table, filters={"run_id": str(run["id"])}, user_id=user_id, order_by={"created_at": "asc"})
        probes = await self.supabase_service.get_by_filters(self.probe_queries_table, filters={"run_id": str(run["id"])}, user_id=user_id, order_by={"created_at": "asc"})
        pages = await self.supabase_service.get_by_filters(self.competitor_pages_table, filters={"run_id": str(run["id"])}, user_id=user_id, order_by={"created_at": "asc"})
        clusters = await self.supabase_service.get_by_filters(self.keyword_clusters_table, filters={"run_id": str(run["id"])}, user_id=user_id, order_by={"opportunity_score": "desc"})
        final_selection = await self._load_final_selection(
            user_id=user_id,
            project_id=UUID(str(run["project_id"])),
            topic_id=UUID(str(run["topic_id"])),
            run_id=UUID(str(run["id"])),
        )
        return {
            "run": run,
            "topic": topic,
            "bets": bets,
            "probe_queries": probes,
            "competitor_pages": pages,
            "clusters": clusters,
            "final_selection": final_selection,
        }

    async def _load_final_selection(
        self,
        *,
        user_id: UUID,
        project_id: UUID,
        topic_id: UUID,
        run_id: UUID,
    ) -> Optional[Dict[str, Any]]:
        candidates = await self.candidate_service.list_candidates(
            user_id=user_id,
            project_id=project_id,
            user_job_id=topic_id,
        )
        strategy_candidates = [
            candidate
            for candidate in candidates
            if str((candidate.get("candidate_metadata") or {}).get("strategy_run_id") or "") == str(run_id)
        ]
        if not strategy_candidates:
            return None

        candidate = strategy_candidates[0]
        candidate_id = UUID(str(candidate["id"]))

        validation_runs = await self.validation_service.list_validation_runs(
            user_id=user_id,
            project_id=project_id,
            candidate_id=candidate_id,
        )
        routing_decisions = await self.routing_service.list_routing_decisions(
            user_id=user_id,
            project_id=project_id,
            candidate_id=candidate_id,
        )
        keyword_packs = await self.keyword_pack_service.list_keyword_packs(
            user_id=user_id,
            project_id=project_id,
            candidate_id=candidate_id,
        )
        generated_outcomes = await self.generation_service.list_generated_outcomes(
            user_id=user_id,
            project_id=project_id,
            candidate_id=candidate_id,
        )
        generated_outcome = generated_outcomes[0] if generated_outcomes else None
        routing_decision = routing_decisions[0] if routing_decisions else None
        route = str((routing_decision or {}).get("route") or "")
        if generated_outcome:
            metadata = dict(generated_outcome.get("outcome_metadata") or {})
            strategy_bet_id = str((candidate.get("candidate_metadata") or {}).get("bet_id") or "")
            related_bet = None
            if strategy_bet_id:
                matching_bets = await self.supabase_service.get_by_filters(
                    self.topic_bets_table,
                    filters={"id": strategy_bet_id},
                    user_id=user_id,
                    limit=1,
                )
                related_bet = matching_bets[0] if matching_bets else None
            generated_outcome = {
                **generated_outcome,
                "outcome_metadata": self._normalize_final_output(
                    route=route,
                    output=metadata,
                    topic={},
                    bet=related_bet,
                    cluster=None,
                ),
            }

        return {
            "candidate": candidate,
            "validation_run": validation_runs[0] if validation_runs else None,
            "routing_decision": routing_decision,
            "keyword_pack": keyword_packs[0] if keyword_packs else None,
            "generated_outcome": generated_outcome,
        }

    async def _delete_run_artifacts(self, *, user_id: UUID, run_id: UUID, tables: List[str]) -> None:
        for table in tables:
            await self.supabase_service.execute_query(
                table=table,
                operation="delete",
                filters={"run_id": str(run_id), "user_id": str(user_id)},
            )

    def _score_trend(self, keyword: str, search: Dict[str, Any]) -> tuple[float, Dict[str, Any]]:
        items = (search.get("result_summary_json") or {}).get("top_items") or []
        numeric_values: List[float] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            candidate = item.get("values") or item.get("data") or item.get("interest")
            if isinstance(candidate, list):
                for value in candidate:
                    try:
                        numeric_values.append(float(value))
                    except Exception:
                        continue
        if len(numeric_values) >= 4:
            head = sum(numeric_values[: max(1, len(numeric_values)//3)]) / max(1, len(numeric_values[: max(1, len(numeric_values)//3)]))
            tail = sum(numeric_values[-max(1, len(numeric_values)//3):]) / max(1, len(numeric_values[-max(1, len(numeric_values)//3):]))
            if tail >= head * 1.15:
                return 0.8, {"direction": "rising", "keyword": keyword}
            if tail <= head * 0.85:
                return 0.2, {"direction": "declining", "keyword": keyword}
        return 0.5, {"direction": "stable", "keyword": keyword}

    def _classify_serp(
        self,
        *,
        query_text: str,
        rows: List[Dict[str, Any]],
        article_format: str,
        route_hint: str,
    ) -> Dict[str, Any]:
        article_count = 0
        weak_count = 0
        tool_count = 0
        service_count = 0
        ecommerce_count = 0
        authority_count = 0
        mixed_count = 0
        titles_blob = " ".join(str(row.get("title") or "") for row in rows).lower()
        comparison_style_terms = {
            "vs",
            "comparison",
            "compare",
            "review",
            "reviews",
            "alternatives",
            "alternative",
            "top",
            "best",
        }
        for row in rows[:10]:
            url = str(row.get("url") or "").lower()
            title = str(row.get("title") or "").lower()
            domain = str(row.get("domain") or urlparse(url).netloc).lower()
            article_signal = (
                any(token in title for token in ["how", "guide", "worth", "roi", "should", "when"])
                or any(token in title for token in comparison_style_terms)
                or any(token in url for token in ["/blog/", "/guides/", "/resources/", "/compare", "/vs-"])
            )
            if article_signal:
                article_count += 1
            if any(token in url for token in ["calculator", "tool"]) or any(token in title for token in ["calculator", "tool"]):
                tool_count += 1
            if any(token in url for token in ["/services/", "/service/", "/products/", "/shop/"]) or any(token in title for token in ["near me", "service", "buy now"]):
                service_count += 1
            if any(token in url for token in ["/product/", "/category/", "/collections/"]) or any(token in title for token in ["product", "price", "shop"]):
                ecommerce_count += 1
            if any(token in domain for token in ["reddit.", "quora.", "medium.", "forum", "community"]):
                weak_count += 1
            if any(token in domain for token in ["forbes.", "wikipedia.", "nerdwallet.", "bankrate.", "investopedia.", "gov"]):
                authority_count += 1
            if "vs" in title or "comparison" in title or "worth" in title:
                mixed_count += 1
        query_lower = query_text.lower()
        comparison_query = any(token in query_lower for token in [" vs ", "compare", "comparison", "alternative", "alternatives"])
        comparison_serp = mixed_count >= 2
        article_format_fit = 0.8 if any(token in titles_blob for token in article_format.replace("_", " ").split()) else (0.65 if article_count >= 3 else 0.35)
        if comparison_query and comparison_serp:
            article_format_fit = max(article_format_fit, 0.7)
        intent_fit = min(1.0, (article_count / 5.0) + (mixed_count / 6.0))
        serp_weakness = max(0.0, min(1.0, (weak_count + max(0, 3 - authority_count)) / 6.0))
        articleability = (
            0.45 * min(1.0, article_count / 5.0)
            + 0.25 * article_format_fit
            + 0.15 * serp_weakness
            + 0.15 * intent_fit
        )
        classification = "article_friendly"
        reason_codes: List[str] = []
        articleability_passed = (
            (
                article_count >= 3
                or (article_count >= 2 and mixed_count >= 2)
                or (comparison_query and article_count >= 2)
            )
            and article_format_fit >= 0.55
            and service_count < 5
            and ecommerce_count < 5
            and tool_count < 5
        )
        inferred_route_hint = route_hint
        if tool_count >= 4:
            classification = "tool_dominant"
            inferred_route_hint = "software"
            articleability_passed = False
            reason_codes.append("tool_dominant_serp")
        elif service_count >= 5:
            classification = "service_dominant"
            articleability_passed = False
            reason_codes.append("service_dominant_serp")
        elif ecommerce_count >= 5:
            classification = "ecommerce_dominant"
            articleability_passed = False
            reason_codes.append("ecommerce_dominant_serp")
        elif article_count < 3 and mixed_count < 2:
            classification = "mixed"
            articleability_passed = False
            reason_codes.append("not_enough_articles")
        if not reason_codes and articleability_passed:
            reason_codes.append("article_friendly_serp")
        return {
            "articleability_score": round(articleability, 4),
            "articleability_passed": articleability_passed,
            "classification": classification,
            "serp_weakness_score": round(serp_weakness, 4),
            "intent_fit_score": round(intent_fit, 4),
            "article_format_fit": round(article_format_fit, 4),
            "reason_codes": reason_codes,
            "route_hint": inferred_route_hint,
        }

    def _extract_article_competitor_urls(self, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        filtered = []
        for row in rows[:10]:
            url = str(row.get("url") or "")
            title = str(row.get("title") or "")
            domain = str(row.get("domain") or urlparse(url).netloc)
            haystack = f"{url} {title}".lower()
            if any(token in haystack for token in ["calculator", "/tool", "/tools/", "/product/", "/category/", "near me", "/services/"]):
                continue
            filtered.append({
                "url": url,
                "title": title,
                "domain": domain,
                "rank_group": row.get("rank_group"),
            })
        return filtered

    def _select_attractive_competitor_targets(self, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        grouped: Dict[str, Dict[str, Any]] = {}
        for row in rows:
            domain = str(row.get("domain") or urlparse(str(row.get("url") or "")).netloc).lower()
            if not domain or domain in EXCLUDED_LARGE_DOMAINS:
                continue
            entry = grouped.setdefault(
                domain,
                {
                    "domain": domain,
                    "rows": [],
                    "best_rank": 99.0,
                },
            )
            entry["rows"].append(row)
            try:
                entry["best_rank"] = min(float(row.get("rank_group") or 99.0), entry["best_rank"])
            except Exception:
                pass

        selected: List[Dict[str, Any]] = []
        for domain, payload in grouped.items():
            rows_for_domain = sorted(
                payload["rows"],
                key=lambda item: float(item.get("rank_group") or 99.0),
            )
            representative = rows_for_domain[0]
            selected.append(
                {
                    **representative,
                    "analysis_target": domain,
                    "domain_hits": len(rows_for_domain),
                    "source_urls": [str(item.get("url") or "") for item in rows_for_domain if str(item.get("url") or "")],
                }
            )
        return sorted(
            selected,
            key=lambda item: (
                int(item.get("domain_hits") or 0),
                -float(item.get("rank_group") or 99.0),
            ),
            reverse=True,
        )

    def _cluster_keywords(self, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        buckets: Dict[str, Dict[str, Any]] = {}
        for row in rows:
            keyword = " ".join(str(row.get("keyword") or "").strip().split())
            if not keyword:
                continue
            tokens = [token for token in re.findall(r"[a-z0-9]+", keyword.lower()) if token not in STOPWORDS and len(token) > 2]
            if not tokens:
                continue
            bucket_key = " ".join(tokens[:3])
            bucket = buckets.setdefault(bucket_key, {
                "cluster_name": bucket_key.replace("  ", " ").strip().title(),
                "keywords": [],
                "supporting_urls": set(),
                "kd_values": [],
                "cpc_values": [],
                "competition_values": [],
                "ranks": [],
            })
            bucket["keywords"].append({
                "keyword": keyword,
                "search_volume": row.get("search_volume"),
                "keyword_difficulty": row.get("keyword_difficulty"),
                "rank_group": row.get("rank_group"),
            })
            if row.get("source_url"):
                bucket["supporting_urls"].add(str(row.get("source_url")))
            if row.get("keyword_difficulty") is not None:
                try:
                    bucket["kd_values"].append(float(row.get("keyword_difficulty")))
                except Exception:
                    pass
            if row.get("cpc") is not None:
                try:
                    bucket["cpc_values"].append(float(row.get("cpc")))
                except Exception:
                    pass
            if row.get("competition_index") is not None:
                try:
                    bucket["competition_values"].append(float(row.get("competition_index")))
                except Exception:
                    pass
            if row.get("rank_group") is not None:
                try:
                    bucket["ranks"].append(float(row.get("rank_group")))
                except Exception:
                    pass
        clusters: List[Dict[str, Any]] = []
        for bucket_key, bucket in buckets.items():
            keyword_entries: List[Dict[str, Any]] = []
            seen_keywords = set()
            for item in bucket["keywords"]:
                key = str(item.get("keyword") or "").lower()
                if not key or key in seen_keywords:
                    continue
                seen_keywords.add(key)
                keyword_entries.append(item)
            if len(keyword_entries) < 2:
                continue
            keyword_entries = sorted(
                keyword_entries,
                key=lambda item: (
                    float(item.get("search_volume") or 0.0),
                    -float(item.get("keyword_difficulty") or 100.0),
                    -float(item.get("rank_group") or 99.0),
                    -len(str(item.get("keyword") or "")),
                ),
                reverse=True,
            )
            primary = str(keyword_entries[0].get("keyword") or "")
            keywords = [str(item.get("keyword") or "") for item in keyword_entries if str(item.get("keyword") or "")]
            support_count = len(bucket["supporting_urls"])
            competitor_support_score = max(0.0, min(1.0, (support_count / 3.0) * 0.7 + (min(len(keywords), 10) / 10.0) * 0.3))
            kd_median = median(bucket["kd_values"]) if bucket["kd_values"] else 45.0
            kd_median_score = max(0.0, min(1.0, 1.0 - (kd_median / 100.0)))
            cpc_median = median(bucket["cpc_values"]) if bucket["cpc_values"] else 0.0
            comp_median = median(bucket["competition_values"]) if bucket["competition_values"] else 0.0
            commercial_value_score = max(0.0, min(1.0, min(1.0, cpc_median / 8.0) * 0.6 + min(1.0, comp_median / 1.0) * 0.4))
            clusters.append({
                "cluster_name": bucket["cluster_name"],
                "primary_keyword_candidate": primary,
                "secondary_keywords": [keyword for keyword in keywords if keyword != primary][:15],
                "supporting_urls": sorted(bucket["supporting_urls"]),
                "competitor_support_score": round(competitor_support_score, 4),
                "kd_median_score": round(kd_median_score, 4),
                "commercial_value_score": round(commercial_value_score, 4),
                "keyword_count": len(keywords),
                "median_rank": round(median(bucket["ranks"]), 2) if bucket["ranks"] else None,
            })
        return sorted(clusters, key=lambda item: (item["competitor_support_score"], item["commercial_value_score"]), reverse=True)

    def _materialize_keyword_opportunities(
        self,
        *,
        rows: List[Dict[str, Any]],
        bet: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        opportunities: List[Dict[str, Any]] = []
        serp_weakness = float(bet.get("serp_weakness_score") or 0.0)
        trend_score = float(bet.get("trend_score") or 0.5)
        article_fit_score = float(bet.get("article_fit_score") or 0.0)
        articleability_score = float(bet.get("serp_articleability_score") or 0.0)

        for row in rows[: self.DEFAULT_LIMITS["max_keyword_opportunities_per_bet"]]:
            keyword = str(row.get("keyword") or "").strip()
            if not keyword:
                continue
            source_url = str(row.get("source_url") or "").strip()
            source_domain = str(row.get("source_domain") or urlparse(source_url).netloc).strip()
            search_volume = float(row.get("search_volume") or 0.0)
            kd = float(row.get("keyword_difficulty") or 100.0)
            cpc = float(row.get("cpc") or 0.0)
            competition_index = float(row.get("competition_index") or 0.0)
            competitor_support_score = max(
                0.25,
                min(
                    1.0,
                    (1.0 if source_domain else 0.0) * 0.55
                    + max(0.0, 1.0 - (float(row.get("rank_group") or 99.0) / 25.0)) * 0.45,
                ),
            )
            kd_median_score = max(0.0, min(1.0, 1.0 - (kd / 100.0)))
            commercial_value_score = max(
                0.0,
                min(1.0, min(1.0, cpc / 8.0) * 0.6 + min(1.0, competition_index) * 0.4),
            )
            opportunity_score = self._compute_cluster_opportunity_score(
                serp_weakness_score=serp_weakness,
                competitor_support_score=competitor_support_score,
                kd_median_score=kd_median_score,
                commercial_value_score=commercial_value_score,
                trend_score=trend_score,
                article_fit_score=article_fit_score,
            )
            opportunities.append(
                {
                    "cluster_name": keyword,
                    "primary_keyword_candidate": keyword,
                    "secondary_keywords": [],
                    "supporting_urls": [source_url] if source_url else [],
                    "cluster_type": "keyword_opportunity",
                    "competitor_support_score": round(competitor_support_score, 4),
                    "kd_median_score": round(kd_median_score, 4),
                    "commercial_value_score": round(commercial_value_score, 4),
                    "keyword_count": 1,
                    "median_rank": row.get("rank_group"),
                    "opportunity_score": opportunity_score,
                    "cluster_metadata": {
                        "search_volume": search_volume,
                        "keyword_difficulty": kd,
                        "intent": row.get("intent"),
                        "cpc": cpc,
                        "competition_index": competition_index,
                        "relevance_score": row.get("relevance_score"),
                        "seed_overlap": row.get("seed_overlap"),
                        "qualification_score": row.get("qualification_score"),
                        "source_domain": source_domain,
                        "source_url": source_url,
                        "source_title": row.get("source_title"),
                    },
                }
            )
        return opportunities

    def _qualify_competitor_keywords(
        self,
        *,
        topic_text: str,
        bet_text: str,
        rows: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        scored: List[Dict[str, Any]] = []
        topic_tokens = {token for token in re.findall(r"[a-z0-9]+", topic_text.lower()) if token not in STOPWORDS and len(token) > 2}
        bet_tokens = {token for token in re.findall(r"[a-z0-9]+", bet_text.lower()) if token not in STOPWORDS and len(token) > 2}
        anchor_tokens = topic_tokens | bet_tokens
        for row in rows:
            keyword = str(row.get("keyword") or "").strip()
            if not keyword:
                continue
            keyword_tokens = {token for token in re.findall(r"[a-z0-9]+", keyword.lower()) if token not in STOPWORDS and len(token) > 2}
            if not keyword_tokens:
                continue
            overlap = len(anchor_tokens & keyword_tokens)
            source_title = str(row.get("source_title") or "")
            source_url = str(row.get("source_url") or "")
            source_tokens = {
                token
                for token in re.findall(r"[a-z0-9]+", f"{source_title} {source_url}".lower())
                if token not in STOPWORDS and len(token) > 2
            }
            source_overlap = len(anchor_tokens & source_tokens)
            intent = str(row.get("intent") or "").lower()
            search_volume = float(row.get("search_volume") or 0.0)
            rank_group = float(row.get("rank_group") or 99.0)
            kd = float(row.get("keyword_difficulty") or 100.0)
            if search_volume < 20:
                continue
            if rank_group > 25:
                continue
            if intent in {"navigational"}:
                continue
            if kd > 70:
                continue
            if len(keyword_tokens) <= 1:
                continue
            if overlap == 0 and source_overlap == 0:
                continue
            overlap_score = min(1.0, overlap / max(1, min(len(keyword_tokens), 3)))
            source_overlap_score = min(1.0, source_overlap / max(1, min(len(source_tokens), 4))) if source_tokens else 0.0
            relevance_score = (
                overlap_score * 0.08
                + source_overlap_score * 0.07
                + min(1.0, search_volume / 500.0) * 0.3
                + max(0.0, 1.0 - (rank_group / 25.0)) * 0.3
                + max(0.0, 1.0 - (kd / 100.0)) * 0.25
                + (0.03 if overlap == 0 and source_overlap > 0 else 0.0)
            )
            scored.append({
                **row,
                "relevance_score": round(relevance_score, 4),
                "seed_overlap": overlap,
                "source_overlap": source_overlap,
            })
        scored.sort(key=lambda item: float(item.get("relevance_score") or 0.0), reverse=True)
        return scored[: self.DEFAULT_LIMITS["max_keyword_overview_keywords"]]

    def _merge_keyword_overview_metrics(
        self,
        ranked_rows: List[Dict[str, Any]],
        overview_rows: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        overview_map = {
            str(item.get("keyword") or "").strip().lower(): item
            for item in (overview_rows or [])
            if str(item.get("keyword") or "").strip()
        }
        merged: List[Dict[str, Any]] = []
        for row in ranked_rows:
            keyword = str(row.get("keyword") or "").strip()
            if not keyword:
                continue
            overview = overview_map.get(keyword.lower(), {})
            merged.append({
                **row,
                "search_volume": overview.get("search_volume", row.get("search_volume")),
                "cpc": overview.get("cpc", row.get("cpc")),
                "competition": overview.get("competition", row.get("competition")),
                "competition_index": overview.get("competition_index", row.get("competition_index")),
                "keyword_difficulty": overview.get("keyword_difficulty", row.get("keyword_difficulty")),
                "intent": overview.get("intent", row.get("intent")),
            })
        return merged

    def _filter_really_competitive_keywords(self, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        qualified: List[Dict[str, Any]] = []
        for row in rows:
            search_volume = float(row.get("search_volume") or 0.0)
            kd = float(row.get("keyword_difficulty") or 100.0)
            competition_index = float(row.get("competition_index") or 0.0)
            intent = str(row.get("intent") or "").lower()
            if search_volume < 30:
                continue
            if kd > 55:
                continue
            if intent not in {"informational", "commercial", "transactional", "commercial_investigation", ""}:
                continue
            score = (
                max(0.0, 1.0 - (kd / 100.0)) * 0.45
                + min(1.0, search_volume / 500.0) * 0.3
                + min(1.0, competition_index) * 0.1
                + float(row.get("relevance_score") or 0.0) * 0.15
            )
            qualified.append({**row, "qualification_score": round(score, 4)})
        qualified.sort(key=lambda item: float(item.get("qualification_score") or 0.0), reverse=True)
        return qualified[:25]

    def _compute_cluster_opportunity_score(
        self,
        *,
        serp_weakness_score: float,
        competitor_support_score: float,
        kd_median_score: float,
        commercial_value_score: float,
        trend_score: float,
        article_fit_score: float,
    ) -> float:
        score = (
            0.30 * serp_weakness_score
            + 0.20 * competitor_support_score
            + 0.15 * kd_median_score
            + 0.15 * commercial_value_score
            + 0.10 * trend_score
            + 0.10 * article_fit_score
        )
        return round(max(0.0, min(1.0, score)), 4)

    async def _generate_final_output(
        self,
        *,
        topic: Dict[str, Any],
        bet: Optional[Dict[str, Any]],
        cluster: Optional[Dict[str, Any]],
        route: str,
        competitor_urls: List[str],
    ) -> Dict[str, Any]:
        if route == "software_ready":
            prompt = f"""
Return valid JSON:
{{
  "title": "string",
  "slug": "string",
  "software_concept": "one paragraph describing the software product clearly",
  "target_user": "string",
  "user_problem": "string",
  "core_workflow": ["step 1", "step 2", "step 3", "step 4"],
  "key_features": ["feature 1", "feature 2", "feature 3", "feature 4"],
  "inputs": ["input 1", "input 2"],
  "outputs": ["output 1", "output 2"],
  "mvp_scope": ["mvp item 1", "mvp item 2", "mvp item 3"],
  "build_notes": "short paragraph on implementation direction",
  "primary_keyword": "string",
  "secondary_keywords": ["string"],
  "confidence_score": 0.0,
  "rationale": "string"
}}

Rules:
- This is a software idea, not an article brief.
- Do not return an article outline or section headings.
- Be concrete about what the tool does for the user.
- Make the workflow and features feel buildable as a real MVP.
- Keep key_features to 4-6 items.
- Keep core_workflow to 4-6 steps.

Topic: {topic.get("job_text")}
Bet: {bet or {}}
Cluster: {cluster or {}}
Route: {route}
Competitor URLs: {competitor_urls}
"""
        elif route == "editorial_only":
            prompt = f"""
Return valid JSON:
{{
  "title": "string",
  "slug": "string",
  "editorial_angle": "string",
  "why_now": "string",
  "outline": ["heading 1", "heading 2", "heading 3"],
  "primary_keyword": "string",
  "secondary_keywords": ["string"],
  "confidence_score": 0.0,
  "rationale": "string"
}}

Rules:
- This is an editorial outcome, not a software spec.
- Outline should contain 5-8 sections.

Topic: {topic.get("job_text")}
Bet: {bet or {}}
Cluster: {cluster or {}}
Route: {route}
Competitor URLs: {competitor_urls}
"""
        else:
            prompt = f"""
Return valid JSON:
{{
  "title": "string",
  "slug": "string",
  "outline": ["heading 1", "heading 2", "heading 3"],
  "primary_keyword": "string",
  "secondary_keywords": ["string"],
  "confidence_score": 0.0,
  "rationale": "string"
}}

Rules:
- Generate a strong SEO article title from the winning keyword opportunity.
- Outline should contain 5-8 sections.

Topic: {topic.get("job_text")}
Bet: {bet or {}}
Cluster: {cluster or {}}
Route: {route}
Competitor URLs: {competitor_urls}
"""
        response = await llm_service.generate_json(prompt, task_role=LLM_ROLE_RESEARCH_IDEA_GENERATION, max_tokens=1600)
        if not isinstance(response, dict):
            response = {}
        return response

    def _normalize_final_output(
        self,
        *,
        route: str,
        output: Dict[str, Any],
        topic: Dict[str, Any],
        bet: Optional[Dict[str, Any]],
        cluster: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        normalized = dict(output or {})
        cluster_metadata = dict((cluster or {}).get("cluster_metadata") or {})
        competitor_urls_used = list((cluster or {}).get("supporting_competitor_urls_json") or [])
        if competitor_urls_used:
            normalized["competitor_urls_used"] = competitor_urls_used
        if cluster_metadata.get("source_url"):
            normalized["source_competitor_url"] = cluster_metadata.get("source_url")
        if cluster_metadata.get("source_domain"):
            normalized["source_competitor_domain"] = cluster_metadata.get("source_domain")

        if route != "software_ready":
            return normalized

        product_name = str(normalized.get("product_name") or normalized.get("title") or "").strip()
        if not product_name or self._is_article_like_title(product_name):
            product_name = self._derive_software_product_name(
                bet_text=str((bet or {}).get("bet_text") or ""),
                topic_text=str((topic or {}).get("job_text") or ""),
            )
        if product_name:
            normalized["product_name"] = product_name
            normalized["title"] = product_name

        if not str(normalized.get("software_concept") or "").strip():
            fallback_problem = str((bet or {}).get("searcher_problem") or "").strip()
            fallback_rationale = str(normalized.get("rationale") or "").strip()
            normalized["software_concept"] = fallback_problem or fallback_rationale or (
                "A focused software concept derived from the winning workflow-oriented research bet."
            )

        if not str(normalized.get("target_user") or "").strip():
            normalized["target_user"] = "Operators or solo creators managing repeatable multi-step workflows."

        if not str(normalized.get("user_problem") or "").strip():
            normalized["user_problem"] = str((bet or {}).get("searcher_problem") or "").strip() or str(normalized.get("software_concept") or "").strip()

        if not isinstance(normalized.get("core_workflow"), list):
            normalized["core_workflow"] = []
        if not isinstance(normalized.get("key_features"), list):
            normalized["key_features"] = []
        if not isinstance(normalized.get("inputs"), list):
            normalized["inputs"] = []
        if not isinstance(normalized.get("outputs"), list):
            normalized["outputs"] = []
        if not isinstance(normalized.get("mvp_scope"), list):
            normalized["mvp_scope"] = []

        if not normalized["core_workflow"]:
            normalized["core_workflow"] = [
                "Connect the source content or workflow trigger.",
                "Choose the channels or downstream apps to target.",
                "Generate or transform the outputs with an agentic automation layer.",
                "Review, approve, and publish the results without repetitive manual steps.",
            ]
        if not normalized["key_features"]:
            normalized["key_features"] = [
                "Cross-app workflow orchestration",
                "Template-driven output generation",
                "Human approval before publishing",
                "Reusable automation presets by use case",
            ]
        if not normalized["inputs"]:
            normalized["inputs"] = [
                "Source asset or workflow payload",
                "Distribution targets and user preferences",
            ]
        if not normalized["outputs"]:
            normalized["outputs"] = [
                "Ready-to-publish transformed outputs",
                "A reusable workflow run with status and revision trail",
            ]
        if not normalized["mvp_scope"]:
            normalized["mvp_scope"] = [
                "One core workflow with 2-3 downstream destinations",
                "Simple approval and edit layer",
                "Reusable prompt and transformation templates",
            ]

        if not str(normalized.get("build_notes") or "").strip():
            normalized["build_notes"] = (
                "Start with a narrow MVP that handles one high-frequency workflow end to end, "
                "then expand destination support and automation templates after validating repeat usage."
            )

        if not str(normalized.get("slug") or "").strip() and product_name:
            normalized["slug"] = re.sub(r"[^a-z0-9]+", "-", product_name.lower()).strip("-")

        primary_keyword = str(normalized.get("primary_keyword") or "").strip()
        if not primary_keyword:
            primary_keyword = str((cluster or {}).get("primary_keyword_candidate") or "").strip()
        if not primary_keyword:
            primary_keyword = str((bet or {}).get("bet_text") or "").strip().lower()
        normalized["primary_keyword"] = primary_keyword

        secondary_keywords = normalized.get("secondary_keywords")
        if not isinstance(secondary_keywords, list):
            secondary_keywords = []
        normalized["secondary_keywords"] = [str(item).strip() for item in secondary_keywords if str(item).strip()]
        return normalized

    def _is_article_like_title(self, value: str) -> bool:
        lowered = value.lower()
        return any(
            token in lowered
            for token in [" vs. ", " vs ", "guide", "choosing", "should ", "best ", "how ", ":"]
        )

    def _derive_software_product_name(self, *, bet_text: str, topic_text: str) -> str:
        source = bet_text or topic_text or "Workflow Opportunity"
        lowered = source.lower()
        if "repurpos" in lowered:
            return "RepurposeFlow AI"
        if "crm" in lowered:
            return "FollowUp Agent"
        if "domain" in lowered:
            return "DomainSignal AI"
        tokens = [token.capitalize() for token in re.findall(r"[A-Za-z0-9]+", source) if token.lower() not in STOPWORDS]
        return "".join(tokens[:2]) + (" AI" if tokens else "Workflow AI")

    def _build_serp_rationale(self, bet: Optional[Dict[str, Any]], cluster: Optional[Dict[str, Any]]) -> str:
        if not bet:
            return "Selected from the best surviving strategic bet."
        parts = [
            f"Bet '{bet.get('bet_text')}' survived the SERP screen",
            f"with articleability {bet.get('serp_articleability_score')}",
            f"and weakness {bet.get('serp_weakness_score')}.",
        ]
        if cluster:
            parts.append(
                f"Winning keyword opportunity '{cluster.get('primary_keyword_candidate') or cluster.get('cluster_name')}' "
                f"was supported by {len(cluster.get('supporting_competitor_urls_json') or [])} competitor page(s)."
            )
        return " ".join(parts)
