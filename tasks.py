"""
Celery tasks for Content Generator V2.

This module contains all the asynchronous tasks for the article generation pipeline.
"""

import logging
import os
import re
import html
import sys
import concurrent.futures
import asyncio
from datetime import datetime
from typing import Dict, Any, Optional, List
from celery import current_task
from celery_config import celery
from supabase_client import get_supabase_client, get_llm_api_key, get_linkup_api_key, get_default_llm_provider
from llm_client import create_llm_client
from rag_client import create_rag_client, RAGQuery
from linkup_client import create_linkup_client, SearchQuery
from article_structure_generator import create_article_structure_generator
from content_generator import create_content_generator, get_tone_specific_instructions
from citation_generator import create_citation_generator, CitationStyle
from src.utils.config import get_config

# Setup logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def _ensure_project_root_on_path() -> None:
    """
    Ensure imports like `src.services...` work even if the worker starts
    from a non-project working directory.
    """
    project_root = os.path.dirname(os.path.abspath(__file__))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

# Task status constants
TASK_STATUS = {
    'PENDING': 'PENDING',
    'PROGRESS': 'PROGRESS', 
    'SUCCESS': 'SUCCESS',
    'FAILURE': 'FAILURE',
    'CANCELLED': 'CANCELLED'
}
PIPELINE_STAGES = [
    'INITIALIZED',
    'CLAIM_EXTRACTION',
    'EVIDENCE_COLLECTION',
    'EVIDENCE_RANKING',
    'STRUCTURE_GENERATION',
    'CONTENT_GENERATION',
    'CITATION_GENERATION',
    'REFINEMENT',
    'FINALIZATION',
    'COMPLETED',
]

# Minimum dossier validation thresholds (Phase 1 gates)
DOSSIER_MIN_QUALITY_SCORE = 30
DOSSIER_MIN_SOURCE_COUNT = 2
DOSSIER_MIN_CLAIM_COUNT = 2
DOSSIER_MIN_SUMMARY_CHARS = 200
ARTICLE_MIN_QUALITY_SCORE = 55
ARTICLE_MIN_GROUNDING_SCORE = 45
ARTICLE_MIN_GEO_SCORE = 45
ARTICLE_MIN_AVG_CLAIM_CONFIDENCE = 0.40
ARTICLE_MAX_WEAK_CLAIMS = 3

_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "how",
    "in", "is", "it", "of", "on", "or", "that", "the", "this", "to", "was",
    "were", "what", "when", "where", "which", "who", "why", "with", "will",
}
_CONTRADICTION_MARKERS = (
    "however", "but", "although", "despite", "in contrast", "on the other hand",
    "contrary", "disputed", "mixed", "unclear", "inconclusive", "no consensus",
)
_SOURCE_TYPES = ("expert", "primary", "secondary", "commercial", "community")


def _normalize_claim_text(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", str(text or "").strip())
    cleaned = re.sub(r"\.+$", "", cleaned)
    return cleaned


def _make_claim_id(text: str, idx: int) -> str:
    normalized = _normalize_claim_text(text).lower()
    slug = re.sub(r"[^a-z0-9]+", "-", normalized).strip("-")
    slug = slug[:42] if slug else "claim"
    return f"clm_{idx + 1:02d}_{slug}"


def _create_citation_links(content: str, citations: List[Dict[str, Any]]) -> str:
    """
    Convert in-text markers like [1] into clickable anchors to the References section.
    Keeps original marker text if index is out of range.
    """
    if not content:
        return ""
    if not citations:
        return content

    def _replace(match: re.Match) -> str:
        raw = match.group(0)
        try:
            idx = int(match.group(1))
        except Exception:
            return raw
        if idx < 1 or idx > len(citations):
            return raw
        return f'<a href="#ref-{idx}" class="citation-link">[{idx}]</a>'

    return re.sub(r"\[(\d+)\]", _replace, content)


def _extract_focus_keyword(keywords: str) -> str:
    """Pick a primary keyword from comma-separated keywords."""
    if not keywords:
        return ""
    parts = [p.strip() for p in str(keywords).split(",") if p.strip()]
    if not parts:
        return ""
    # Prefer a moderately descriptive keyword phrase.
    parts.sort(key=lambda p: (abs(len(p) - 28), -len(p)))
    return parts[0]


def _truncate_seo_title(title: str, max_length: int = 60, focus_keyword: str = "") -> str:
    title = (title or "").strip()
    if len(title) <= max_length:
        return title
    keyword = (focus_keyword or "").strip()
    if keyword and len(keyword) < max_length - 6:
        prefix = f"{keyword}: "
        remaining = max_length - len(prefix) - 1
        if remaining > 8:
            return prefix + title[:remaining].rstrip(" -:;,.") + "…"
    return title[: max_length - 1].rstrip(" -:;,.") + "…"


def _ensure_meta_description_length(meta_description: str, max_length: int = 160) -> str:
    text = re.sub(r"\s+", " ", (meta_description or "").strip())
    if not text:
        return ""
    if len(text) <= max_length:
        return text
    clipped = text[:max_length]
    # Try not to cut mid-word.
    if " " in clipped:
        clipped = clipped.rsplit(" ", 1)[0]
    return clipped.rstrip(" -:;,.") + "…"


def _generate_wp_slug(title: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", (title or "").lower()).strip("-")
    return slug[:80] if slug else f"article-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"


def _generate_wp_tag_ids(keywords: str) -> List[int]:
    # Keep generation deterministic without depending on WP lookup at this stage.
    # Downstream publish flow can map names -> IDs if needed.
    return []


def _generate_breadcrumb_title(title: str) -> str:
    t = (title or "").strip()
    if len(t) <= 48:
        return t
    return t[:47].rstrip(" -:;,.") + "…"


def _extract_plain_text(html_content: str) -> str:
    if not html_content:
        return ""
    text = re.sub(r"<[^>]+>", " ", str(html_content))
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def _extract_external_links(citations: List[Dict[str, Any]]) -> List[str]:
    links: List[str] = []
    seen = set()
    for c in citations or []:
        url = (c or {}).get("url") or (c or {}).get("source")
        if not url:
            continue
        u = str(url).strip()
        if not u or u in seen:
            continue
        seen.add(u)
        links.append(u)
    return links


def _calculate_readability_score(text: str) -> float:
    words = re.findall(r"\b\w+\b", text or "")
    if not words:
        return 0.0
    sentences = max(1, len(re.findall(r"[.!?]+", text or "")))
    avg_sentence_len = len(words) / sentences
    # Simple heuristic score in 0-100 range (shorter sentences score higher).
    return float(max(0, min(100, 100 - (avg_sentence_len - 12) * 3)))


def _calculate_seo_score(title: str, meta_description: str, word_count: int, citations_count: int) -> float:
    score = 0.0
    t_len = len((title or "").strip())
    m_len = len((meta_description or "").strip())
    if 35 <= t_len <= 60:
        score += 30
    elif t_len > 0:
        score += 15
    if 120 <= m_len <= 160:
        score += 25
    elif m_len > 0:
        score += 10
    if 900 <= int(word_count or 0) <= 3500:
        score += 25
    elif int(word_count or 0) > 0:
        score += 10
    if int(citations_count or 0) >= 3:
        score += 20
    elif int(citations_count or 0) > 0:
        score += 10
    return float(max(0, min(100, score)))


def _calculate_viral_score(hook: str, excerpt: str, word_count: int) -> float:
    score = 0.0
    if len((hook or "").strip()) >= 40:
        score += 35
    if len((excerpt or "").strip()) >= 80:
        score += 30
    wc = int(word_count or 0)
    if 1000 <= wc <= 2800:
        score += 35
    elif wc > 0:
        score += 15
    return float(max(0, min(100, score)))


def _claim_keywords(claim_text: str) -> List[str]:
    tokens = re.findall(r"\b[a-z0-9]{3,}\b", str(claim_text or "").lower())
    return [t for t in tokens if t not in _STOPWORDS]


def _evidence_text(evidence_item: Dict[str, Any]) -> str:
    parts = [
        evidence_item.get("title", ""),
        evidence_item.get("content", ""),
        evidence_item.get("snippet", ""),
        evidence_item.get("summary", ""),
    ]
    return " ".join([str(p) for p in parts if p]).strip().lower()


def _evidence_source_host(evidence_item: Dict[str, Any]) -> str:
    source = str(evidence_item.get("source", "") or "")
    url = str(evidence_item.get("url", "") or "")
    host_candidate = url if url else source
    if not host_candidate:
        return "unknown"
    host_candidate = host_candidate.replace("https://", "").replace("http://", "")
    return host_candidate.split("/")[0].lower() or "unknown"


def _compute_claim_alignment(
    claim_keywords: List[str], evidence_item: Dict[str, Any]
) -> Dict[str, Any]:
    text = _evidence_text(evidence_item)
    if not claim_keywords:
        return {"matched_terms": [], "match_ratio": 0.0}
    matched_terms = [k for k in claim_keywords if k in text]
    match_ratio = len(set(matched_terms)) / max(len(set(claim_keywords)), 1)
    return {"matched_terms": sorted(set(matched_terms)), "match_ratio": match_ratio}


def _assess_rag_coverage(
    rag_evidence: List[Dict[str, Any]],
    keywords: str = "",
    min_sources: int = 2,
    min_relevance: float = 0.55,
) -> Dict[str, Any]:
    """Assess whether current evidence coverage is sufficient to skip web search."""
    evidence = [e for e in (rag_evidence or []) if isinstance(e, dict)]
    source_count = len(evidence)
    if source_count == 0:
        return {
            "sufficient": False,
            "source_count": 0,
            "avg_relevance": 0.0,
            "keyword_coverage": 0.0,
            "assessment": "no_evidence",
        }

    relevance_scores = []
    for e in evidence:
        score = e.get("relevance_score")
        if score is None:
            score = e.get("similarity_score", 0.0)
        try:
            relevance_scores.append(float(score or 0.0))
        except Exception:
            relevance_scores.append(0.0)
    avg_relevance = sum(relevance_scores) / max(len(relevance_scores), 1)

    kw_tokens = [k.strip().lower() for k in str(keywords or "").split(",") if k.strip()]
    if kw_tokens:
        text_parts: List[str] = []
        for e in evidence:
            text_parts.extend(
                [
                    str(e.get("title", "") or ""),
                    str(e.get("content", "") or ""),
                    str(e.get("snippet", "") or ""),
                ]
            )
        combined_text = " ".join(text_parts).lower()
        hits = sum(1 for kw in kw_tokens if kw in combined_text)
        keyword_coverage = hits / max(len(kw_tokens), 1)
    else:
        keyword_coverage = 1.0

    sufficient = source_count >= int(min_sources) and avg_relevance >= float(min_relevance)
    if sufficient:
        assessment = "sufficient"
    elif source_count < int(min_sources):
        assessment = "insufficient_sources"
    else:
        assessment = "low_relevance"

    return {
        "sufficient": sufficient,
        "source_count": source_count,
        "avg_relevance": round(avg_relevance, 3),
        "keyword_coverage": round(keyword_coverage, 3),
        "assessment": assessment,
    }


def _detect_contradiction_signal(evidence_item: Dict[str, Any], claim_text: str) -> bool:
    text = _evidence_text(evidence_item)
    if not text:
        return False
    if any(marker in text for marker in _CONTRADICTION_MARKERS):
        return True
    return bool(re.search(r"\b(not|no|unlikely|fails to)\b", text)) and bool(
        _claim_keywords(claim_text)
    )


def _classify_source_type(evidence_item: Dict[str, Any]) -> str:
    text = " ".join([
        str(evidence_item.get("source", "") or ""),
        str(evidence_item.get("url", "") or ""),
        str((evidence_item.get("metadata") or {}).get("publisher", "") if isinstance(evidence_item.get("metadata"), dict) else ""),
        str((evidence_item.get("metadata") or {}).get("title", "") if isinstance(evidence_item.get("metadata"), dict) else ""),
    ]).lower()
    if any(k in text for k in ("reddit", "quora", "stackexchange", "forum", "community")):
        return "community"
    if any(k in text for k in (".gov", ".edu", "whitepaper", "dataset", "arxiv", "who.int", "cdc.gov")):
        return "primary"
    if any(k in text for k in ("journal", "research", "study", "expert", "analyst", "institute")):
        return "expert"
    if any(k in text for k in ("wikipedia", "encyclopedia", "news", "report", "guide", "explainer")):
        return "secondary"
    if any(k in text for k in ("pricing", "product", "shop", "affiliate", "sponsored", "agency")):
        return "commercial"
    return "secondary"


def _validate_research_dossier(
    dossier: Dict[str, Any],
    dossier_status: Optional[str],
    dossier_quality_score: Optional[int],
) -> Dict[str, Any]:
    """Validate dossier readiness and return structured gate result."""
    if not isinstance(dossier, dict) or not dossier:
        return {
            "valid": False,
            "reason": "missing_dossier",
            "quality_score": 0,
            "source_count": 0,
            "claim_count": 0,
        }

    source_count = int((dossier.get("source_quality_summary") or {}).get("source_count", 0) or 0)
    claim_count = len(dossier.get("primary_claims") or [])
    summary_len = len(str(dossier.get("summary", "") or ""))
    quality_score = int(dossier_quality_score or dossier.get("dossier_quality_score") or 0)
    status_ok = str(dossier_status or "").lower() == "ready"

    reasons: List[str] = []
    if not status_ok:
        reasons.append("status_not_ready")
    if quality_score < DOSSIER_MIN_QUALITY_SCORE:
        reasons.append("quality_below_threshold")
    if source_count < DOSSIER_MIN_SOURCE_COUNT:
        reasons.append("insufficient_sources")
    if claim_count < DOSSIER_MIN_CLAIM_COUNT:
        reasons.append("insufficient_claims")
    if summary_len < DOSSIER_MIN_SUMMARY_CHARS:
        reasons.append("summary_too_short")

    return {
        "valid": len(reasons) == 0,
        "reason": ",".join(reasons) if reasons else "ok",
        "quality_score": quality_score,
        "source_count": source_count,
        "claim_count": claim_count,
    }


def _build_confidence_map(claim_bundles: List[Dict[str, Any]], html_content: str) -> Dict[str, Any]:
    """Build claim/paragraph confidence map for review and gating."""
    high_conf = []
    mixed_conf = []
    low_conf = []
    for bundle in claim_bundles or []:
        entry = {
            "claim_id": bundle.get("claim_id"),
            "claim": bundle.get("claim"),
            "confidence_score": float(bundle.get("confidence_score", 0) or 0),
            "support_count": int(bundle.get("support_count", 0) or 0),
            "contradiction_count": int(bundle.get("contradiction_count", 0) or 0),
            "mixed_signal": bool(bundle.get("mixed_signal")),
        }
        if entry["mixed_signal"] or entry["contradiction_count"] > 0:
            mixed_conf.append(entry)
        elif entry["confidence_score"] >= 0.70:
            high_conf.append(entry)
        elif entry["confidence_score"] < 0.45:
            low_conf.append(entry)
        else:
            mixed_conf.append(entry)

    uncited_paragraphs: List[Dict[str, Any]] = []
    paragraphs = re.findall(r"<p[^>]*>(.*?)</p>", html_content or "", flags=re.IGNORECASE | re.DOTALL)
    for idx, p in enumerate(paragraphs, start=1):
        clean = re.sub(r"<[^>]+>", " ", p)
        clean = re.sub(r"\s+", " ", clean).strip()
        if len(clean.split()) < 25:
            continue
        has_citation = bool(re.search(r"\[\^?\d+\]", clean))
        if not has_citation:
            uncited_paragraphs.append({
                "paragraph_index": idx,
                "excerpt": clean[:220],
            })

    avg_confidence = round(
        (
            sum(float(b.get("confidence_score", 0) or 0) for b in (claim_bundles or [])) /
            max(len(claim_bundles or []), 1)
        ),
        3,
    ) if claim_bundles else 0.0

    return {
        "high_confidence_claims": high_conf[:20],
        "mixed_evidence_claims": mixed_conf[:20],
        "low_confidence_claims": low_conf[:20],
        "uncited_paragraphs": uncited_paragraphs[:30],
        "summary": {
            "high_count": len(high_conf),
            "mixed_count": len(mixed_conf),
            "low_count": len(low_conf),
            "uncited_paragraph_count": len(uncited_paragraphs),
            "avg_claim_confidence": avg_confidence,
        },
    }


def _is_sensitive_topic(research_data: Dict[str, Any], title: str) -> bool:
    corpus = " ".join([
        str(research_data.get("brief", "") or ""),
        str(research_data.get("keywords", "") or ""),
        str(title or ""),
    ]).lower()
    sensitive_markers = [
        "medical", "health", "clinical", "treatment",
        "legal", "law", "attorney", "regulation",
        "finance", "investment", "tax", "insurance",
        "security", "cybersecurity", "privacy", "compliance",
    ]
    return any(m in corpus for m in sensitive_markers)


def _evaluate_article_quality_gates(
    research_data: Dict[str, Any],
    quality_report: Dict[str, Any],
    claim_bundles: List[Dict[str, Any]],
    confidence_map: Dict[str, Any],
    citations_count: int,
) -> Dict[str, Any]:
    """Evaluate completion gates and return Created vs Needs Review decision."""
    overall = float(quality_report.get("overall_score", 0) or 0)
    grounding = float(quality_report.get("grounding_score", 0) or 0)
    geo = float(quality_report.get("geo_score", 0) or 0)
    avg_claim_conf = float((confidence_map.get("summary") or {}).get("avg_claim_confidence", 0) or 0)
    weak_claims = int((confidence_map.get("summary") or {}).get("low_count", 0) or 0)
    uncited_paragraphs = int((confidence_map.get("summary") or {}).get("uncited_paragraph_count", 0) or 0)

    sensitive = _is_sensitive_topic(research_data, research_data.get("draft_title", ""))
    min_quality = ARTICLE_MIN_QUALITY_SCORE + (8 if sensitive else 0)
    min_grounding = ARTICLE_MIN_GROUNDING_SCORE + (10 if sensitive else 0)
    min_geo = ARTICLE_MIN_GEO_SCORE
    max_uncited = 4 if sensitive else 8
    min_citations = 3 if sensitive else 1

    failures: List[str] = []
    if overall < min_quality:
        failures.append("overall_quality_below_threshold")
    if grounding < min_grounding:
        failures.append("grounding_below_threshold")
    if geo < min_geo:
        failures.append("geo_below_threshold")
    if avg_claim_conf < ARTICLE_MIN_AVG_CLAIM_CONFIDENCE:
        failures.append("avg_claim_confidence_below_threshold")
    if weak_claims > ARTICLE_MAX_WEAK_CLAIMS:
        failures.append("too_many_low_confidence_claims")
    if uncited_paragraphs > max_uncited:
        failures.append("too_many_uncited_paragraphs")
    if citations_count < min_citations:
        failures.append("insufficient_citations")

    decision = "Created" if not failures else "Needs Review"
    return {
        "decision": decision,
        "pass": len(failures) == 0,
        "sensitive_topic": sensitive,
        "thresholds": {
            "min_quality": min_quality,
            "min_grounding": min_grounding,
            "min_geo": min_geo,
            "min_avg_claim_confidence": ARTICLE_MIN_AVG_CLAIM_CONFIDENCE,
            "max_low_confidence_claims": ARTICLE_MAX_WEAK_CLAIMS,
            "max_uncited_paragraphs": max_uncited,
            "min_citations": min_citations,
        },
        "observed": {
            "overall_score": overall,
            "grounding_score": grounding,
            "geo_score": geo,
            "avg_claim_confidence": avg_claim_conf,
            "low_confidence_claim_count": weak_claims,
            "uncited_paragraph_count": uncited_paragraphs,
            "citations_count": citations_count,
        },
        "failures": failures,
    }

# ...

@celery.task(bind=True, name='content_generator_v2.tasks.research.process_research_task')
def process_research_task(self, research_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main research task that orchestrates the entire article generation pipeline.
    
    Args:
        research_data: Dictionary containing research parameters from the API request
        
    Returns:
        Dictionary containing the generated article and metadata
    """
    task_id = self.request.id
    logger.info(f"Starting research task {task_id} with data: {research_data}")
    result: Dict[str, Any] = {
        'task_id': task_id,
        'status': TASK_STATUS['PENDING'],
        'created_at': datetime.utcnow().isoformat(),
        'research_data': research_data,
        'pipeline_stages': PIPELINE_STAGES,
        'current_stage': 'INITIALIZED',
        'progress': 0,
        'article': None,
        'error': None,
    }
    
    # Update DB status to 'Generating' immediately so frontend persists state
    article_id = research_data.get('article_id')
    if article_id:
        try:
            logger.info(f"Attempting to set initial status to 'Generating' for article {article_id}")
            supabase = get_supabase_client()
            if supabase:
                # Update status
                supabase.table('Titles').update({'status': 'Generating'}).eq('id', article_id).execute()
                logger.info(f"Successfully set initial status to 'Generating' for article {article_id}")
                
                # Fetch content_outline and research dossier if available
                logger.info(f"Fetching content_outline for article {article_id}")
                try:
                    response = (
                        supabase.table('Titles')
                        .select('content_outline,research_dossier,dossier_status,dossier_quality_score')
                        .eq('id', article_id)
                        .execute()
                    )
                except Exception as select_error:
                    # Backward compatibility: if dossier columns are not migrated yet,
                    # fall back to content_outline only.
                    if "research_dossier" in str(select_error):
                        logger.warning(
                            "research_dossier column missing; falling back to content_outline only: %s",
                            str(select_error),
                        )
                        response = (
                            supabase.table('Titles')
                            .select('content_outline')
                            .eq('id', article_id)
                            .execute()
                        )
                    else:
                        raise
                if response.data and len(response.data) > 0:
                    content_outline = response.data[0].get('content_outline')
                    research_dossier = response.data[0].get('research_dossier')
                    dossier_status = response.data[0].get('dossier_status')
                    dossier_quality_score = response.data[0].get('dossier_quality_score')
                    if content_outline:
                        logger.info(f"Found content_outline for article {article_id}: {str(content_outline)[:100]}...")
                        research_data['content_outline'] = content_outline
                    else:
                        logger.info(f"No content_outline found in DB for article {article_id}")
                    if research_dossier:
                        gate = _validate_research_dossier(
                            dossier=research_dossier,
                            dossier_status=dossier_status,
                            dossier_quality_score=dossier_quality_score,
                        )
                        research_data['dossier_status'] = dossier_status
                        research_data['dossier_quality_score'] = dossier_quality_score
                        research_data['dossier_gate'] = gate
                        research_data['dossier_validated'] = gate.get("valid", False)
                        if gate.get("valid"):
                            research_data['research_dossier'] = research_dossier
                        else:
                            # Keep raw dossier for observability, but do not use as trusted context.
                            research_data['research_dossier_raw'] = research_dossier
                        logger.info(
                            "Loaded research dossier for article %s (status=%s, quality_score=%s, valid=%s, reason=%s)",
                            article_id,
                            dossier_status,
                            dossier_quality_score,
                            gate.get("valid"),
                            gate.get("reason"),
                        )
                else:
                    logger.warning(f"Failed to fetch article row for {article_id}")
            else:
                logger.warning("Supabase client not available for initial status update")
        except Exception as e:
            logger.error(f"Failed to set initial status or fetch outline: {e}")
    
    try:
        # Update task status to PROGRESS
        self.update_state(
            state=TASK_STATUS['PROGRESS'],
            meta={
                'current_stage': 'INITIALIZED',
                'progress': 5,
                'message': 'Preparing research parameters...'
            }
        )
        
        # Initialize result structure
        result = {
            'task_id': task_id,
            'status': TASK_STATUS['PROGRESS'],
            'created_at': datetime.utcnow().isoformat(),
            'research_data': research_data,
            'pipeline_stages': PIPELINE_STAGES,
            'current_stage': 'INITIALIZED',
            'progress': 0,
            'article': None,
            'error': None
        }
        
        # Stage 1: Claim Extraction
        result = _process_stage(
            self, 
            result, 
            'CLAIM_EXTRACTION', 
            10,
            'Extracting claims from research brief...',
            _extract_claims
        )
        
        # Stage 2: Evidence Collection
        result = _process_stage(
            self,
            result,
            'EVIDENCE_COLLECTION',
            25,
            'Collecting evidence from RAG and web search...',
            _collect_evidence
        )
        
        # Stage 3: Evidence Ranking
        result = _process_stage(
            self,
            result,
            'EVIDENCE_RANKING',
            40,
            'Ranking and assessing evidence quality...',
            _rank_evidence
        )
        
        # Stage 4: Structure Generation
        result = _process_stage(
            self,
            result,
            'STRUCTURE_GENERATION',
            55,
            'Generating article structure and outline...',
            _generate_structure
        )
        
        # Stage 5: Content Generation
        result = _process_stage(
            self,
            result,
            'CONTENT_GENERATION',
            70,
            'Generating article content...',
            lambda r: _generate_content(r, self)
        )
        
        # Stage 6: Citation Generation
        result = _process_stage(
            self,
            result,
            'CITATION_GENERATION',
            80,
            'Generating citations and references...',
            _generate_citations
        )
        
        # Stage 7: Refinement
        result = _process_stage(
            self,
            result,
            'REFINEMENT',
            90,
            'Refining and optimizing article...',
            lambda r: _refine_article(r, task_instance=self)
        )
        
        # Stage 8: Finalization
        result = _process_stage(
            self,
            result,
            'FINALIZATION',
            95,
            'Finalizing article...',
            _finalize_article
        )
        
        # Complete the task
        result.update({
            'status': TASK_STATUS['SUCCESS'],
            'current_stage': 'COMPLETED',
            'progress': 100,
            'completed_at': datetime.utcnow().isoformat(),
            'message': 'Article generation completed successfully!'
        })
        
        # Save to Supabase if article_id is present
        article_id = research_data.get('article_id')
        if article_id:
            try:
                supabase = get_supabase_client()
                if supabase:
                    # extract the final article content
                    final_article = result.get('article') or {}
                    if not final_article and result.get('content'):
                         # Try to construct it if it's not in 'article' key (tasks structure might vary)
                         # Based on _finalize_article, it seems 'article' key is supposed to be set?
                         # Wait, _finalize_article returns a dict with keys. 
                         # Let's check _process_stage calls.
                         # Stage 8: _finalize_article returns result.update(stage_result)
                         # _finalize_article DOES NOT return 'article' key directly in the snippet I saw?
                         # Let's re-read _finalize_article return value.
                         pass

                    # Actually, let's look at what _finalize_article returns. 
                    # It returns a dict that updates 'result'.
                    # I need to know the keys.
                    # Assuming standard Noodl structure or similar.
                    # Let's rely on what `get_research_result` in app.py uses:
                    # final_article = result.get('final_article', {})
                    
                    # So I should check if 'final_article' is in result.
                    final_content = result.get('final_article') or result.get('article') or {}
                    
                    logger.info(f"Preparing Supabase update for article {article_id}")
                    logger.info(f"Final content keys present: {list(final_content.keys())}")
                    
                    article_text = final_content.get('articleText', '')
                    html_article = final_content.get('htmlArticle', '')
                    
                    logger.info(f"Content lengths - Text: {len(article_text)}, HTML: {len(html_article)}")
                    
                    import json
                    
                    # Serialize citations for storage
                    citations_json = json.dumps(final_content.get('citations', []))
                    final_decision = final_content.get('generation_status', 'Created')
                    
                    updates = {
                        'status': final_decision,
                        'articleText': article_text,
                        'htmlArticle': html_article,
                        'seo_optimization_score': int(float(final_content.get('seo_optimization_score', 0))),
                        'readability_score': int(float(final_content.get('readability_score', 0))),
                        'overall_quality_score': int(float(final_content.get('overall_quality_score', 0))),
                        'quality_report': final_content.get('quality_report', {}),
                        'confidence_map': final_content.get('confidence_map', {}),
                        'quality_gate': final_content.get('quality_gate', {}),
                        'citations': citations_json,  # Store citations as JSON for Reference Selector
                        'include_in_text_citations': True,  # Default to showing in-text citations
                        'deck': final_content.get('deck', ''),
                        'hook': final_content.get('hook', ''),
                        'thesis': final_content.get('thesis', ''),
                        'excerpt': final_content.get('excerpt', ''),
                        # Add other fields as needed
                    }

                    try:
                        response = supabase.table('Titles').update(updates).eq('id', article_id).execute()
                    except Exception as update_error:
                        # Backward-compatible fallback for environments that have not
                        # applied latest metadata migrations yet.
                        err = str(update_error)
                        removable_fields = ['quality_report', 'confidence_map', 'quality_gate']
                        retried = False
                        for field in removable_fields:
                            if field in err and field in updates:
                                logger.warning(
                                    "%s column missing, retrying Titles update without %s: %s",
                                    field,
                                    field,
                                    err,
                                )
                                updates = {k: v for k, v in updates.items() if k != field}
                                retried = True
                        if retried:
                            response = supabase.table('Titles').update(updates).eq('id', article_id).execute()
                        else:
                            raise
                    
                    if response.data:
                        logger.info(f"Updated Supabase Titles for article {article_id}. Rows modified: {len(response.data)}")
                    else:
                        logger.warning(f"Supabase update returned success but NO rows were modified for {article_id}. Potential RLS or ID mismatch.")
                        
            except Exception as e:
                logger.error(f"Failed to update Supabase for article {article_id}: {str(e)}")
                # Try to update status to Failed so frontend doesn't hang
                try:
                    supabase.table('Titles').update({'status': 'Failed'}).eq('id', article_id).execute()
                except:
                    pass
                raise e # Re-raise to trigger outer failure handler

        logger.info(f"Research task {task_id} completed successfully")
        return result
        
    except Exception as e:
        logger.error(f"Research task {task_id} failed: {str(e)}", exc_info=True)

        # Preserve explicit failure metadata for polling clients.
        result.update({
            'status': TASK_STATUS['FAILURE'],
            'error': str(e),
            'failed_at': datetime.utcnow().isoformat(),
            'message': f'Article generation failed: {str(e)}'
        })
        # Do NOT call update_state(..., state=FAILURE, meta=<plain dict>).
        # Celery backends expect a serialized exception payload for FAILURE and
        # can corrupt result metadata when given arbitrary dicts here. Re-raising
        # below lets Celery persist a proper terminal failure state.

        # Ensure Supabase status is updated to Failed
        try:
            article_id = (result.get('research_data') or {}).get('article_id')
            if article_id:
                supabase = get_supabase_client()
                if supabase:
                    supabase.table('Titles').update({'status': 'Failed'}).eq('id', article_id).execute()
        except Exception:
            pass

        # Re-raise so Celery result state is terminal FAILURE (not SUCCESS/PROGRESS hang).
        raise

def _process_stage(self, result: Dict[str, Any], stage: str, progress: int, 
                  message: str, stage_function) -> Dict[str, Any]:
    """
    Process a single pipeline stage and update task status.
    
    Args:
        self: Celery task instance
        result: Current result dictionary
        stage: Stage name
        progress: Progress percentage
        message: Status message
        stage_function: Function to execute for this stage
        
    Returns:
        Updated result dictionary
    """
    try:
        logger.info(f"🚀 Starting stage {stage} ({progress}%) - {message} for task {result['task_id']}")
        # Update task status
        self.update_state(
            state=TASK_STATUS['PROGRESS'],
            meta={
                'current_stage': stage,
                'progress': progress,
                'message': message
            }
        )
        
        # Update result
        result.update({
            'current_stage': stage,
            'progress': progress,
            'message': message
        })
        
        # Execute stage function
        try:
            # Attempt to pass task_instance for granular updates
            stage_result = stage_function(result, task_instance=self)
        except TypeError:
            # Fallback for functions that don't accept task_instance yet
            stage_result = stage_function(result)
        result.update(stage_result)
        
        logger.info(f"Completed stage {stage} for task {result['task_id']}")
        return result
        
    except Exception as e:
        logger.error(f"Stage {stage} failed for task {result['task_id']}: {str(e)}")
        raise

# Stage functions with LLM integration
def _extract_claims(result: Dict[str, Any], task_instance: Any = None) -> Dict[str, Any]:
    """Extract claims from research brief using LLM."""
    try:
        # Update sub-progress
        if task_instance:
            task_instance.update_state(
                state=TASK_STATUS['PROGRESS'],
                meta={
                    'current_stage': 'CLAIM_EXTRACTION',
                    'progress': 12,
                    'message': 'Analyzing research brief to identify key claims...'
                }
            )
        research_data = result.get('research_data', {})
        research_dossier = research_data.get('research_dossier') if research_data.get('dossier_validated', True) else {}
        brief = research_data.get('brief', '')
        keywords = research_data.get('keywords', '')

        dossier_claims = research_dossier.get('primary_claims') if isinstance(research_dossier, dict) else None
        if dossier_claims and isinstance(dossier_claims, list):
            normalized_claims = []
            for idx, claim in enumerate(dossier_claims):
                if isinstance(claim, dict):
                    text = _normalize_claim_text(claim.get('claim', ''))
                    if not text:
                        continue
                    normalized_claims.append(
                        {
                            "claim_id": claim.get("claim_id") or _make_claim_id(text, idx),
                            "claim": text,
                            "category": claim.get('category', 'dossier'),
                            "importance": claim.get('importance', 'medium'),
                            "keywords": _claim_keywords(text)[:16],
                        }
                    )
            if normalized_claims:
                logger.info(f"Using {len(normalized_claims)} claims from research dossier")
                return {
                    'claims': normalized_claims,
                    'stage_data': {
                        'extracted_claims': len(normalized_claims),
                        'claim_source': 'research_dossier',
                    }
                }
        
        # Create LLM client
        
        # Verify provider/model and fetch API key
        provider = research_data.get('provider')
        model = research_data.get('model')
        api_key = None
        
        if not provider or not model:
            logger.info("Provider or model not specified in research_data - fetching default from Supabase")
            def_provider, def_model, def_key = get_default_llm_provider()
            if def_provider and def_model and def_key:
                provider = def_provider
                model = def_model
                api_key = def_key
                # Update research_data for consistency
                research_data['provider'] = provider
                research_data['model'] = model
            else:
                 logger.warning("Failed to fetch default LLM provider - falling back to 'openai'/'gpt-4' (and likely missing key)")
                 provider = 'openai'
                 model = 'gpt-4'
        else:
             # Fetch key for specific provider/model
             api_key = get_llm_api_key(provider, model)

        if not api_key:
             logger.warning(f"Could not fetch API key for {provider}/{model}")

        llm_client = create_llm_client(
            provider=provider,
            model=model,
            api_key=api_key,
            temperature=0.3  # Lower temperature for more focused extraction
        )
        
        # Create prompt for claim extraction
        messages = [
            {
                "role": "system",
                "content": "You are an expert researcher. Extract key claims and assertions from the research brief. Return a JSON array of claims, each with 'claim', 'category', and 'importance' fields."
            },
            {
                "role": "user",
                "content": f"Research Brief: {brief}\nKeywords: {keywords}\n\nExtract the main claims and assertions that need to be researched and validated."
            }
        ]
        
        # Generate claims
        response = llm_client.generate(messages)
        
        # Parse response (simplified for now)
        fallback_text = _normalize_claim_text(f"Claim extracted from: {brief[:100]}...")
        claims = [{
            "claim_id": _make_claim_id(fallback_text, 0),
            "claim": fallback_text,
            "category": "general",
            "importance": "high",
            "keywords": _claim_keywords(fallback_text)[:16],
        }]
        
        logger.info(f"Extracted {len(claims)} claims using {response.model}")
        
        return {
            'claims': claims,
            'stage_data': {
                'extracted_claims': len(claims),
                'llm_model': response.model,
                'llm_cost': response.cost,
                'llm_time': response.response_time
            }
        }
        
    except Exception as e:
        logger.error(f"Error in claim extraction: {str(e)}")
        return {'claims': [], 'stage_data': {'extracted_claims': 0, 'error': str(e)}}

def _collect_evidence(result: Dict[str, Any], task_instance: Any = None) -> Dict[str, Any]:
    """Collect evidence from RAG and web search."""
    try:
        # Update sub-progress
        if task_instance:
            task_instance.update_state(
                state=TASK_STATUS['PROGRESS'],
                meta={
                    'current_stage': 'EVIDENCE_COLLECTION',
                    'progress': 26,
                    'message': 'Searching internal knowledge base (RAG)...'
                }
            )
        logger.info("🔍 Starting evidence collection stage...")
        research_data = result.get('research_data', {})
        claims = result.get('claims', [])
        brief = research_data.get('brief', '')
        keywords = research_data.get('keywords', '')
        research_dossier = research_data.get('research_dossier') or {}
        
        logger.info(f"📊 Claims count: {len(claims)}")
        logger.info(f"📝 Brief length: {len(brief)} chars")
        logger.info(f"🏷️ Keywords: {keywords}")
        
        evidence = []
        rag_sources = 0
        web_sources = 0
        
        # Collect evidence from RAG if enabled
        rag_enabled = research_data.get('rag_enabled', False)
        # We check endpoint validity inside the block now to allow fallback to config
        if rag_enabled:
            # Update sub-progress
            if task_instance:
                task_instance.update_state(
                    state=TASK_STATUS['PROGRESS'],
                    meta={
                        'current_stage': 'EVIDENCE_COLLECTION',
                        'progress': 28,
                        'message': 'Searching internal knowledge base (RAG)...'
                    }
                )
            
            # Resolve RAG configuration
            app_config = get_config()
            
            # Determine RAG endpoint: research_data > config > None
            rag_endpoint = research_data.get('rag_endpoint')
            if not rag_endpoint and app_config.RAG_API_URL:
                rag_endpoint = app_config.RAG_API_URL
                logger.info(f"Using RAG endpoint from config: {rag_endpoint}")
            
            # Determine RAG API Key
            rag_api_key = research_data.get('rag_api_key') or app_config.RAG_API_KEY
            
            if rag_enabled and rag_endpoint:
                logger.info(f"🔍 RAG search enabled - query target: {rag_endpoint}")
                
                # Ensure endpoint is full URL to query path
                if not rag_endpoint.endswith('/query_hybrid_enhanced'):
                    base_url = rag_endpoint.rstrip('/')
                    rag_endpoint = f"{base_url}/query_hybrid_enhanced"
                    logger.info(f"Adjusted RAG endpoint to: {rag_endpoint}")
            
            try:
                # Use provided collection - no default, require explicit collection name
                rag_collection = research_data.get('rag_collection') or research_data.get('rag_collection_name')
                if not rag_collection:
                    logger.warning("⚠️ RAG enabled but no collection specified - will proceed with global query if endpoint supports it")
                    # For now, skip RAG if no collection is provided
                    rag_collection = None
                
                if not rag_collection:
                    logger.warning("⚠️ Skipping RAG search - collection name required")
                else:
                    rag_client = create_rag_client(
                        endpoint=rag_endpoint,
                        api_key=rag_api_key,
                        collection=rag_collection,
                        max_results=5,
                        similarity_threshold=0.7
                    )
                    
                    # Create RAG query prioritizing specific keywords over long brief
                    # Extract first sentence of brief for context, then add keywords
                    brief_sentences = brief.split('.')
                    brief_context = brief_sentences[0].strip() if brief_sentences else brief
                    dossier_claims = []
                    dossier_questions = []
                    dossier_summary = ""
                    if isinstance(research_dossier, dict):
                        dossier_summary = str(research_dossier.get('summary', '') or '')[:240]
                        claims_list = research_dossier.get('primary_claims') or []
                        if isinstance(claims_list, list):
                            dossier_claims = [
                                c.get('claim', '').strip() for c in claims_list[:3]
                                if isinstance(c, dict) and c.get('claim')
                            ]
                        questions_list = research_dossier.get('unresolved_questions') or []
                        if isinstance(questions_list, list):
                            dossier_questions = [str(q).strip() for q in questions_list[:2] if str(q).strip()]
                    
                    # Include draft title if provided for better focus
                    draft_title = research_data.get('draft_title', '')
                    dossier_context = " ".join(
                        [part for part in [dossier_summary, " ".join(dossier_claims), " ".join(dossier_questions)] if part]
                    ).strip()
                    if draft_title:
                        rag_query_text = f"{keywords} {draft_title} {brief_context} {dossier_context}"
                    else:
                        rag_query_text = f"{keywords} {brief_context} {dossier_context}"
                    logger.info(f"Creating RAG query:")
                    logger.info(f"  - Brief: '{brief}'")
                    logger.info(f"  - Keywords: '{keywords}'")
                    logger.info(f"  - Draft Title: '{draft_title}'")
                    logger.info(f"  - Brief Context: '{brief_context}'")
                    logger.info(f"  - Combined Query: '{rag_query_text}'")
                    logger.info(f"  - Collection: '{rag_collection}'")
                    logger.info(f"  - Balance Emphasis: '{research_data.get('rag_balance_emphasis', 'auto')}'")
                    
                    rag_query = RAGQuery(
                        query=rag_query_text,
                        collection=rag_collection,
                        max_results=10,  # Increased from 5 to 10 for better global coverage
                        balance_emphasis=research_data.get('rag_balance_emphasis', 'auto')
                    )
                    
                    rag_response = rag_client.query(rag_query)
                    
                    if rag_response.success and rag_response.results:
                        for rag_result in rag_response.results:
                            # Only add evidence if it has actual content
                            if rag_result.content and rag_result.content.strip():
                                evidence.append({
                                    "source": rag_result.source,
                                    "content": rag_result.content,
                                    "title": rag_result.metadata.get('title', '') if rag_result.metadata else '',
                                    "relevance_score": rag_result.relevance_score or 0.7,
                                    "credibility_score": rag_result.credibility_score or 0.7,
                                    "similarity_score": rag_result.similarity_score or 0.7,
                                    "source_type": "rag",
                                    "metadata": rag_result.metadata or {}
                                })
                                rag_sources += 1
                        
                        if rag_sources > 0:
                            logger.info(f"✅ Collected {rag_sources} RAG sources from collection '{rag_collection}'")
                        else:
                            logger.warning(f"⚠️ RAG query returned {len(rag_response.results)} results but none had valid content")
                    else:
                        if not rag_response.success:
                            logger.warning(f"⚠️ RAG query failed: {rag_response.error}")
                        else:
                            logger.warning(f"⚠️ RAG query returned no results")
                    
            except Exception as e:
                logger.error(f"❌ Error in RAG evidence collection: {str(e)}")
                logger.info("Continuing without RAG evidence - will rely on web search or proceed without")
        else:
            if not rag_enabled:
                logger.info("RAG search disabled by flag, skipping RAG search")
            else:
                logger.info("RAG search enabled but no endpoint configured, skipping RAG search")
        
        logger.info(f"✅ RAG collection completed. Total evidence so far: {len(evidence)}")
        
        # Assess RAG coverage to determine if Linkup search is needed
        # Only assess if RAG was actually enabled and used
        rag_enabled = research_data.get('rag_enabled', False)
        # Filter RAG evidence to only include items with actual content
        rag_evidence = [e for e in evidence if e.get('source_type') == 'rag' and e.get('content') and e.get('content').strip()]
        config = get_config()
        optimization_config = config.linkup_optimization
        
        # Assess RAG coverage to determine if web search is needed
        # Update sub-progress
        if task_instance:
            task_instance.update_state(
                state=TASK_STATUS['PROGRESS'],
                meta={
                    'current_stage': 'EVIDENCE_COLLECTION',
                    'progress': 30,
                    'message': 'Analyzing RAG coverage and determining search strategy...'
                }
            )
            
        coverage = _assess_rag_coverage(
            evidence, 
            keywords=keywords,
            min_sources=optimization_config.rag_coverage_min_sources,
            min_relevance=optimization_config.rag_coverage_min_relevance
        )
        
        # Only assess RAG coverage if RAG was enabled and we have evidence with content
        # If RAG is disabled or has no valid evidence, we should always use Linkup (if claims_research_enabled)
        if rag_enabled and len(rag_evidence) > 0:
            rag_coverage = _assess_rag_coverage(
                rag_evidence=rag_evidence,
                keywords=keywords,
                min_sources=optimization_config.rag_coverage_min_sources,
                min_relevance=optimization_config.rag_coverage_min_relevance
            )
            
            logger.info(f"📊 RAG Coverage Assessment:")
            logger.info(f"  - Sources: {rag_coverage['source_count']} (min: {optimization_config.rag_coverage_min_sources})")
            logger.info(f"  - Avg Relevance: {rag_coverage['avg_relevance']:.2f} (min: {optimization_config.rag_coverage_min_relevance})")
            logger.info(f"  - Keyword Coverage: {rag_coverage['keyword_coverage']:.2f}")
            logger.info(f"  - Assessment: {rag_coverage['assessment']}")
            logger.info(f"  - Sufficient: {rag_coverage['sufficient']}")
        else:
            # RAG disabled or no RAG evidence with content - assume insufficient coverage
            rag_coverage = {
                'sufficient': False,
                'source_count': 0,
                'avg_relevance': 0.0,
                'keyword_coverage': 0.0,
                'assessment': 'rag_disabled_or_no_valid_evidence'
            }
            if rag_enabled:
                logger.info(f"📊 RAG Coverage: RAG enabled but no valid evidence with content - Linkup will be used if enabled")
            else:
                logger.info(f"📊 RAG Coverage: RAG disabled - Linkup will be used if enabled")
        
        # Collect evidence from web search if claims research is enabled
        # Default to True (consistent with app.py) - web search should run unless explicitly disabled
        claims_research_enabled = research_data.get('claims_research_enabled', True)
        research_provider_strategy = str(
            research_data.get('research_provider_strategy') or
            os.getenv('RESEARCH_PROVIDER_STRATEGY', 'hybrid')
        ).strip().lower()
        if research_provider_strategy not in {'linkup_only', 'tavily_only', 'hybrid', 'auto'}:
            logger.warning(f"Unknown research_provider_strategy '{research_provider_strategy}', defaulting to hybrid")
            research_provider_strategy = 'hybrid'
        
        # Auto-enable LinkUp in scenarios where RAG doesn't provide sufficient evidence:
        # 1. RAG is disabled at the flag level
        # 2. RAG was enabled but failed to collect evidence (no collection, connection error, etc.)
        # 3. RAG was enabled but coverage is insufficient
        # This ensures we have evidence for content generation
        # Only override if claims_research_enabled was explicitly set to False
        if not rag_enabled:
            if 'claims_research_enabled' not in research_data:
                # Not explicitly set, default to True when RAG is disabled
                claims_research_enabled = True
                logger.info("RAG disabled - enabling Linkup by default (claims_research not explicitly disabled)")
            elif not claims_research_enabled:
                logger.info("Both RAG and Linkup are disabled - proceeding without external evidence sources")
            else:
                # RAG is disabled but claims_research_enabled is explicitly True - use Linkup
                logger.info("RAG disabled but claims_research_enabled is True - will use Linkup for evidence collection")
        elif rag_enabled and len(rag_evidence) == 0:
            # RAG was enabled but failed to collect evidence - auto-enable LinkUp as fallback
            if 'claims_research_enabled' not in research_data:
                # Not explicitly disabled, enable LinkUp as fallback
                claims_research_enabled = True
                logger.info("RAG enabled but no valid evidence collected - enabling Linkup as fallback (claims_research not explicitly disabled)")
            elif not claims_research_enabled:
                logger.info("RAG enabled but no valid evidence collected, and Linkup is explicitly disabled - proceeding without evidence sources")
        elif rag_enabled and not rag_coverage.get('sufficient', False):
            # RAG was enabled but coverage is insufficient - ensure Linkup is used
            if claims_research_enabled:
                logger.info(f"RAG coverage insufficient ({rag_coverage['source_count']} sources, relevance: {rag_coverage['avg_relevance']:.2f}) - Linkup will be used to supplement")
            elif 'claims_research_enabled' not in research_data:
                # Not explicitly disabled, enable LinkUp as fallback when RAG is insufficient
                claims_research_enabled = True
                logger.info("RAG coverage insufficient - enabling Linkup as fallback (claims_research not explicitly disabled)")
        
        web_evidence = []
        if claims_research_enabled:
            # Determine if Linkup search is needed based on RAG coverage
            request_depth = research_data.get('depth', 'standard')
            
            # Skip Linkup entirely if RAG coverage is sufficient and depth is not 'deep'
            # But ONLY if RAG was actually enabled and provided sufficient coverage
            if rag_enabled and rag_coverage['sufficient'] and request_depth != 'deep':
                logger.info(f"⏭️  Skipping Linkup search - RAG coverage is sufficient "
                          f"({rag_coverage['source_count']} sources, relevance: {rag_coverage['avg_relevance']:.2f})")
            else:
                # Update sub-progress
                if task_instance:
                    task_instance.update_state(
                        state=TASK_STATUS['PROGRESS'],
                        meta={
                            'current_stage': 'EVIDENCE_COLLECTION',
                            'progress': 33,
                            'message': 'Searching web for supplemental evidence...'
                        }
                    )
                logger.info(f"🔍 Web search needed - provider strategy: {research_provider_strategy}")
                try:
                    def _add_search_results(resp, target_list, provider_label: str):
                        nonlocal web_sources
                        seen_urls = {ev.get('source') for ev in evidence + target_list if ev.get('source_type') in {'web', 'linkup', 'tavily'}}
                        added = 0
                        for search_item in resp.results:
                            if search_item.url and search_item.url in seen_urls:
                                continue
                            target_list.append({
                                "source": search_item.url,
                                "content": search_item.content or search_item.snippet,
                                "relevance_score": search_item.relevance_score,
                                "credibility_score": search_item.credibility_score,
                                "source_type": "web",
                                "metadata": {**(search_item.metadata or {}), "provider": provider_label}
                            })
                            seen_urls.add(search_item.url)
                            web_sources += 1
                            added += 1
                        return added

                    normalized_query = ' '.join(f"{brief} {keywords}".split())
                    severe_insufficient = (
                        rag_coverage.get('source_count', 0) < optimization_config.deep_trigger_min_sources or
                        rag_coverage.get('avg_relevance', 0.0) < optimization_config.deep_trigger_min_avg_relevance or
                        rag_coverage.get('keyword_coverage', 0.0) < optimization_config.deep_trigger_min_keyword_coverage
                    )

                    # Provider 1: Linkup (default in hybrid/auto/linkup_only)
                    linkup_results_added = 0
                    linkup_standard_result_count = 0
                    linkup_search_succeeded = False
                    if research_provider_strategy in {'linkup_only', 'hybrid', 'auto'}:
                        linkup_api_key = get_linkup_api_key()
                        if not linkup_api_key:
                            logger.warning("Linkup API key not found in Supabase api_keys table")
                        else:
                            logger.info(f"Using Linkup API key: {linkup_api_key[:10]}...")
                            linkup_client = create_linkup_client(
                                api_key=linkup_api_key,
                                cache_enabled=optimization_config.cache_enabled
                            )
                        
                        # Progressive search: start with standard, escalate to deep only if needed
                            initial_depth = 'standard'
                            if request_depth == 'deep' and not rag_coverage.get('sufficient', False) and severe_insufficient:
                                initial_depth = 'deep'

                            logger.info(f"🎯 Linkup strategy: initial_depth='{initial_depth}', severe_insufficient={severe_insufficient}")
                            linkup_response = linkup_client.search(SearchQuery(query=normalized_query, depth=initial_depth))
                            if linkup_response.success:
                                linkup_search_succeeded = True
                                linkup_standard_result_count = len(linkup_response.results)
                                linkup_results_added = _add_search_results(linkup_response, web_evidence, "linkup")
                                logger.info(
                                    f"Linkup ({initial_depth}) returned {len(linkup_response.results)} results, added {linkup_results_added} new (deduped)"
                                )

                                need_deep = (
                                    initial_depth == 'standard' and
                                    not rag_coverage.get('sufficient', False) and
                                    len(linkup_response.results) < optimization_config.deep_min_standard_results_threshold and
                                    severe_insufficient
                                )
                                if need_deep:
                                    logger.info("🚀 Escalating to Linkup deep search: standard results below threshold and RAG insufficient")
                                    deep_resp = linkup_client.search(SearchQuery(query=normalized_query, depth='deep'))
                                    if deep_resp.success:
                                        added_deep = _add_search_results(deep_resp, web_evidence, "linkup")
                                        linkup_results_added += added_deep
                                        logger.info(
                                            f"Linkup (deep) returned {len(deep_resp.results)} results, added {added_deep} new (deduped)"
                                        )
                                    else:
                                        logger.warning(f"Linkup deep search failed: {deep_resp.error}")
                            else:
                                logger.warning(f"Linkup search failed: {linkup_response.error}")

                    # Provider 2: Tavily (for tavily_only / hybrid / auto escalation)
                    should_run_tavily = False
                    if research_provider_strategy == 'tavily_only':
                        should_run_tavily = True
                    elif research_provider_strategy == 'hybrid':
                        should_run_tavily = True
                    elif research_provider_strategy == 'auto':
                        should_run_tavily = (not linkup_search_succeeded) or (
                            linkup_standard_result_count < optimization_config.deep_min_standard_results_threshold
                        )

                    if should_run_tavily:
                        from tavily_client import create_tavily_client
                        from supabase_client import get_api_key

                        tavily_api_key = get_api_key('tavily') or os.getenv('TAVILY_API_KEY')
                        if not tavily_api_key:
                            logger.warning("Tavily API key not found; skipping Tavily search")
                        else:
                            tavily_client = create_tavily_client(
                                api_key=tavily_api_key,
                                timeout=30,
                                max_results=8,
                            )
                            tavily_depth = 'deep' if (request_depth == 'deep' or severe_insufficient) else 'standard'
                            tavily_resp = tavily_client.search(SearchQuery(query=normalized_query, depth=tavily_depth))
                            if tavily_resp.success:
                                added_tavily = _add_search_results(tavily_resp, web_evidence, "tavily")
                                logger.info(
                                    f"Tavily ({tavily_depth}) returned {len(tavily_resp.results)} results, added {added_tavily} new (deduped)"
                                )
                            else:
                                logger.warning(f"Tavily search failed: {tavily_resp.error}")
                        
                except Exception as e:
                    logger.error(f"Error in web search (SIGSEGV protection): {str(e)}")
                    logger.info("Continuing without web search to prevent worker crashes")
        else:
            logger.info("Web search disabled by flag, skipping web search")
        
        # Deduplicate and combine evidence
        # Update sub-progress
        if task_instance:
            task_instance.update_state(
                state=TASK_STATUS['PROGRESS'],
                meta={
                    'current_stage': 'EVIDENCE_COLLECTION',
                    'progress': 35,
                    'message': 'Analyzing search results...'
                }
            )
        
        combined_evidence = evidence + web_evidence
        # Simple content-based deduplication
        seen_content = set()
        unique_evidence = []
        for item in combined_evidence:
            content_hash = hash(item.get('content', '')[:100])
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                unique_evidence.append(item)
        
        evidence = unique_evidence
        logger.info(f"📈 Total combined evidence items: {len(evidence)} (from RAG and Web)")
        
        # If no evidence collected, continue without evidence instead of using mock
        if not evidence:
            logger.info("No evidence collected - proceeding without evidence sources")
        
        logger.info(f"Collected {len(evidence)} total evidence sources")
        
        return {
            'evidence': evidence,
            'stage_data': {
                'rag_sources': rag_sources,
                'web_sources': web_sources,
                'total_sources': len(evidence)
            }
        }
        
    except Exception as e:
        logger.error(f"Error in evidence collection: {str(e)}")
        return {'evidence': [], 'stage_data': {'rag_sources': 0, 'web_sources': 0, 'error': str(e)}}

def _rank_evidence(result: Dict[str, Any], task_instance: Any = None) -> Dict[str, Any]:
    """Rank and assess evidence quality using LLM."""
    try:
        # Update sub-progress
        if task_instance:
            task_instance.update_state(
                state=TASK_STATUS['PROGRESS'],
                meta={
                    'current_stage': 'EVIDENCE_RANKING',
                    'progress': 42,
                    'message': 'Evaluating evidence relevance and credibility...'
                }
            )
        logger.info("🔍 Starting evidence ranking stage...")
        research_data = result.get('research_data', {})
        evidence = result.get('evidence', [])
        claims = result.get('claims', [])
        
        logger.info(f"📊 Evidence count: {len(evidence)}")
        
        if not evidence:
            logger.info("⚠️ No evidence to rank - proceeding without evidence sources")
            return {
                'ranked_evidence': [],
                'claim_bundles': [],
                'stage_data': {'ranked_sources': 0, 'note': 'No evidence available'}
            }
        
        # Limit evidence size to prevent memory issues
        if len(evidence) > 5:
            evidence = evidence[:5]
            logger.warning(f"Limited evidence to 5 sources to prevent memory issues")
        
        logger.info("🔄 Starting evidence ranking...")
        
        # Simple evidence ranking based on existing scores
        ranked_evidence = []
        source_type_distribution = {k: 0 for k in _SOURCE_TYPES}
        for i, ev in enumerate(evidence):
            ranked_ev = ev.copy()
            source_type = _classify_source_type(ranked_ev)
            ranked_ev["source_type_taxonomy"] = source_type
            source_type_distribution[source_type] = source_type_distribution.get(source_type, 0) + 1
            # Use existing scores or create simple ones
            ranked_ev.update({
                'relevance_score': ev.get('relevance_score', 0.8 - (i * 0.05)),
                'credibility_score': ev.get('credibility_score', 0.7 - (i * 0.05)),
                'quality_score': ev.get('quality_score', 0.75 - (i * 0.05)),
                'rank': i + 1
            })
            ranked_evidence.append(ranked_ev)
        
        # Sort by relevance score
        ranked_evidence.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
        
        # Update sub-progress
        if task_instance:
            task_instance.update_state(
                state=TASK_STATUS['PROGRESS'],
                meta={
                    'current_stage': 'EVIDENCE_RANKING',
                    'progress': 45,
                    'message': 'Synthesizing evidence for content generation...'
                }
            )
            
        claim_bundles = []
        claim_supporting_pairs = 0
        contradiction_bundles = 0
        mixed_bundles = 0
        for idx, claim in enumerate(claims):
            claim_text = _normalize_claim_text(claim.get("claim", ""))
            if not claim_text:
                continue
            claim_id = claim.get("claim_id") or _make_claim_id(claim_text, idx)
            keywords = claim.get("keywords") or _claim_keywords(claim_text)
            support_items = []
            contradict_items = []
            neutral_items = []
            matched_hosts = set()

            for ev in ranked_evidence:
                alignment = _compute_claim_alignment(keywords, ev)
                if alignment["match_ratio"] >= 0.15 or len(alignment["matched_terms"]) >= 2:
                    pair = {
                        "rank": ev.get("rank"),
                        "source": ev.get("source"),
                        "url": ev.get("url"),
                        "relevance_score": ev.get("relevance_score"),
                        "credibility_score": ev.get("credibility_score"),
                        "quality_score": ev.get("quality_score"),
                        "match_ratio": round(alignment["match_ratio"], 3),
                        "matched_terms": alignment["matched_terms"][:8],
                    }
                    is_contradiction = _detect_contradiction_signal(ev, claim_text)
                    if is_contradiction:
                        contradict_items.append(pair)
                    elif alignment["match_ratio"] >= 0.25 or len(alignment["matched_terms"]) >= 3:
                        support_items.append(pair)
                    else:
                        neutral_items.append(pair)
                    matched_hosts.add(_evidence_source_host(ev))

            coverage = min(1.0, len(support_items) / 3.0) if support_items else 0.0
            contradiction_ratio = (
                len(contradict_items) / max((len(support_items) + len(contradict_items)), 1)
            )
            mixed_signal = len(support_items) > 0 and len(contradict_items) > 0
            source_type_counts = {k: 0 for k in _SOURCE_TYPES}
            for item in support_items + contradict_items + neutral_items:
                ev_match = next((e for e in ranked_evidence if e.get("rank") == item.get("rank")), None)
                if ev_match:
                    st = ev_match.get("source_type_taxonomy", "secondary")
                    source_type_counts[st] = source_type_counts.get(st, 0) + 1
            confidence = max(0.0, min(
                1.0,
                (coverage * 0.55) +
                (min(len(matched_hosts), 4) / 4.0 * 0.25) +
                ((1.0 - contradiction_ratio) * 0.20),
            ))
            if len(contradict_items) > 0:
                contradiction_bundles += 1
            if mixed_signal:
                mixed_bundles += 1
            claim_supporting_pairs += len(support_items) + len(contradict_items)
            claim_bundles.append({
                "claim_id": claim_id,
                "claim": claim_text,
                "keywords": keywords[:16],
                "supporting_evidence": support_items[:5],
                "contradicting_evidence": contradict_items[:5],
                "neutral_evidence": neutral_items[:3],
                "support_count": len(support_items),
                "contradiction_count": len(contradict_items),
                "neutral_count": len(neutral_items),
                "coverage_score": round(coverage, 3),
                "confidence_score": round(confidence, 3),
                "mixed_signal": mixed_signal,
                "source_diversity": len(matched_hosts),
                "source_type_distribution": source_type_counts,
            })

        logger.info(
            f"✅ Ranked {len(ranked_evidence)} evidence sources; built {len(claim_bundles)} claim bundles"
        )

        return {
            'ranked_evidence': ranked_evidence,
            'claim_bundles': claim_bundles,
            'stage_data': {
                'ranked_sources': len(ranked_evidence),
                'claim_bundles': len(claim_bundles),
                'claim_supporting_pairs': claim_supporting_pairs,
                'contradiction_bundles': contradiction_bundles,
                'mixed_signal_bundles': mixed_bundles,
                'source_type_distribution': source_type_distribution,
                'llm_model': 'fallback',
                'note': 'LLM calls disabled to prevent SIGSEGV'
            }
        }
        
    except Exception as e:
        logger.error(f"Error in evidence ranking: {str(e)}")
        return {
            'ranked_evidence': [],
            'claim_bundles': [],
            'stage_data': {'ranked_sources': 0, 'claim_bundles': 0, 'error': str(e)}
        }

def _generate_structure(result: Dict[str, Any], task_instance: Any = None) -> Dict[str, Any]:
    """Generate article structure using comprehensive structure generator."""
    try:
        # Update sub-progress
        if task_instance:
            task_instance.update_state(
                state=TASK_STATUS['PROGRESS'],
                meta={
                    'current_stage': 'STRUCTURE_GENERATION',
                    'progress': 58,
                    'message': 'Refining article outline and hierarchy...'
                }
            )
        research_data = result.get('research_data', {})
        claims = result.get('claims', [])
        evidence = result.get('evidence', [])
        claim_bundles = result.get('claim_bundles', [])
        
        # Limit evidence size to prevent memory issues
        if len(evidence) > 10:
            evidence = evidence[:10]
            logger.warning(f"Limited evidence to 10 sources for structure generation")
        
        # Use real LLM-powered structure generation
        logger.info("🔄 Starting real LLM-powered structure generation...")
        
        # Get LLM client and config
        # Use strict API key retrieval from Payload or Database
        # api_key = _fetch_api_key_strict(research_data)
        api_key = get_llm_api_key(
             research_data.get('provider', 'gemini'),
             research_data.get('model', 'gemini-2.5-flash')
        )
        llm_client = create_llm_client(
            provider=research_data.get('provider', 'gemini'),
            model=research_data.get('model', 'gemini-2.5-flash'),
            api_key=api_key,
            timeout=180  # Increased timeout for structure generation (3 minutes)
        )
        config = get_config()
        
        # Generate structure using the article structure generator (Verbalized sampling explicitly disabled)
        use_verbalized_sampling = False
        structure_generator = create_article_structure_generator(llm_client, use_verbalized_sampling)
        # Keep call signature aligned with ArticleStructureGenerator.
        structure = structure_generator.generate_structure(
            research_data=research_data,
            claims=claims,
            evidence=evidence,
        )
        
        # Convert ArticleStructure object to dictionary
        structure_dict = {
            'title': structure.title,
            'hook': structure.hook,
            'deck': structure.deck,
            'excerpt': structure.excerpt,
            'thesis': structure.thesis,
            'meta_description': structure.meta_description,
            'call_to_action': structure.call_to_action,
            'keywords': structure.keywords,
            'article_type': structure.article_type,
            'target_audience': structure.target_audience,
            'tone': structure.tone,
            'sections': [{
                'title': section.title,
                'subtitle': section.subtitle,
                'key_points': section.key_points,
                'word_count_target': section.word_count_target,
                'content_type': section.content_type,
                'order': section.order,
                'importance': section.importance
            } for section in structure.sections]
        }
        
        logger.info(f"Generated structure with {len(structure.sections)} sections")
        
        return {
            'structure': structure_dict,
            'stage_data': {
                'generated_sections': len(structure.sections),
                'llm_model': research_data.get('model', 'unknown'),
                'structure_type': 'llm_generated'
            }
        }
        
    except Exception as e:
        logger.error(f"Error in structure generation: {str(e)}", exc_info=True)
        # Fail fast instead of silently returning an empty/fallback structure.
        # Silent fallback hides provider/auth/model issues and later manifests
        # as a generic "Produced only 0 words" error in finalization.
        raise RuntimeError(f"Structure generation failed: {str(e)}") from e

def _collect_section_evidence(section_outline: Dict[str, Any], research_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Collect section-specific evidence from RAG and Linkup (if needed)."""
    try:
        section_title = section_outline.get('title', '')
        key_points = section_outline.get('key_points', [])
        brief = research_data.get('brief', '')
        keywords = research_data.get('keywords', '')
        research_dossier = research_data.get('research_dossier') or {}
        
        # Create a focused section query that prioritizes keywords and specific content
        # Extract the main topic from the brief (first sentence or key phrases)
        brief_sentences = brief.split('.')
        main_topic = brief_sentences[0].strip() if brief_sentences else brief
        
        # Include draft title if provided for better focus
        draft_title = research_data.get('draft_title', '')
        
        dossier_claims = []
        dossier_questions = []
        dossier_summary = ""
        if isinstance(research_dossier, dict):
            dossier_summary = str(research_dossier.get('summary', '') or '')[:180]
            claims_list = research_dossier.get('primary_claims') or []
            if isinstance(claims_list, list):
                dossier_claims = [
                    c.get('claim', '').strip() for c in claims_list[:3]
                    if isinstance(c, dict) and c.get('claim')
                ]
            questions_list = research_dossier.get('unresolved_questions') or []
            if isinstance(questions_list, list):
                dossier_questions = [str(q).strip() for q in questions_list[:2] if str(q).strip()]

        # Create a focused query that prioritizes keywords first, then section content
        if key_points and len(key_points) > 0:
            # Use the most specific key points, avoiding generic terms
            specific_points = [point for point in key_points 
                             if not any(generic in point.lower() for generic in 
                                      ['key takeaways', 'next steps', 'final thoughts', 'overview', 'summary', 'introduction', 'conclusion'])]
            
            if specific_points:
                if draft_title:
                    section_query = f"{keywords} {draft_title} {section_title} {' '.join(specific_points[:2])} {main_topic}"
                else:
                    section_query = f"{keywords} {section_title} {' '.join(specific_points[:2])} {main_topic}"
            else:
                if draft_title:
                    section_query = f"{keywords} {draft_title} {section_title} {main_topic}"
                else:
                    section_query = f"{keywords} {section_title} {main_topic}"
        else:
            if draft_title:
                section_query = f"{keywords} {draft_title} {section_title} {main_topic}"
            else:
                section_query = f"{keywords} {section_title} {main_topic}"

        dossier_context = " ".join(
            [part for part in [dossier_summary, " ".join(dossier_claims), " ".join(dossier_questions)] if part]
        ).strip()
        if dossier_context:
            section_query = f"{section_query} {dossier_context}"
        
        # Clean up extra spaces and ensure it's not too long
        section_query = ' '.join(section_query.split())
        if len(section_query) > 200:
            section_query = section_query[:200] + "..."
        
        logger.info(f"  - Section Query: '{section_query}'")
        logger.info(f"  - Balance Emphasis: '{research_data.get('rag_balance_emphasis', 'auto')}'")
        
        section_evidence = []
        
        # Step 1: Try to collect RAG evidence if RAG is enabled
        rag_enabled = research_data.get('rag_enabled', False)
        if rag_enabled and research_data.get('rag_endpoint'):
            rag_collection = research_data.get('rag_collection') or research_data.get('rag_collection_name')
            if rag_collection:
                try:
                    rag_client = create_rag_client(
                        endpoint=research_data.get('rag_endpoint'),
                        collection=rag_collection,
                        max_results=3,  # Fewer results per section to avoid overwhelming
                        similarity_threshold=0.7
                    )
                    
                    # Create RAG query for this specific section
                    rag_query = RAGQuery(
                        query=section_query,
                        collection=rag_collection,
                        max_results=3,
                        similarity_threshold=0.7,
                        balance_emphasis=research_data.get('rag_balance_emphasis', 'auto')
                    )
                    
                    rag_response = rag_client.query(rag_query)
                    
                    if rag_response.success:
                        for rag_result in rag_response.results:
                            evidence_item = {
                                'content': rag_result.content,
                                'source': rag_result.source,
                                'source_type': 'rag',
                                'similarity_score': rag_result.similarity_score,
                                'metadata': rag_result.metadata,
                                'relevance_score': rag_result.relevance_score,
                                'credibility_score': rag_result.credibility_score
                            }
                            section_evidence.append(evidence_item)
                        
                        logger.info(f"  - Found {len(section_evidence)} section-specific RAG evidence items")
                    else:
                        logger.warning(f"  - Section RAG query failed: {rag_response.error}")
                except Exception as e:
                    logger.warning(f"  - Error collecting RAG evidence for section: {str(e)}")
            else:
                logger.warning(f"  - No RAG collection specified for section, skipping RAG search")
        
        # Step 2: Assess RAG coverage and use Linkup if needed and enabled
        claims_research_enabled = research_data.get('claims_research_enabled', True)
        if claims_research_enabled:
            config = get_config()
            optimization_config = config.linkup_optimization
            research_provider_strategy = str(
                research_data.get('research_provider_strategy') or
                os.getenv('RESEARCH_PROVIDER_STRATEGY', 'hybrid')
            ).strip().lower()
            if research_provider_strategy not in {'linkup_only', 'tavily_only', 'hybrid', 'auto'}:
                research_provider_strategy = 'hybrid'
            
            # Determine if we need Linkup
            need_linkup = False
            
            if rag_enabled:
                # If RAG is enabled, assess coverage to see if Linkup is needed
                # Use lower thresholds for section-specific evidence (sections need less evidence than full article)
                section_min_sources = max(1, optimization_config.rag_coverage_min_sources - 1)  # At least 1 source for section
                section_min_relevance = max(0.5, optimization_config.rag_coverage_min_relevance - 0.1)  # Slightly lower threshold
                
                rag_coverage = _assess_rag_coverage(
                    rag_evidence=section_evidence,
                    keywords=keywords,
                    min_sources=section_min_sources,
                    min_relevance=section_min_relevance
                )
                
                logger.info(f"  - Section RAG Coverage: {rag_coverage['source_count']} sources, "
                           f"relevance: {rag_coverage['avg_relevance']:.2f}, "
                           f"sufficient: {rag_coverage['sufficient']}")
                
                # If RAG evidence is insufficient, use Linkup for this section
                if not rag_coverage['sufficient']:
                    need_linkup = True
                    logger.info(f"  - Section RAG evidence insufficient - using Linkup for additional information")
                else:
                    logger.info(f"  - Section RAG evidence sufficient - skipping Linkup for this section")
            else:
                # If RAG is not enabled, use Linkup directly when claims_research_enabled is true
                need_linkup = True
                logger.info(f"  - RAG not enabled, using Linkup for section-specific evidence")
            
            # Use Linkup if needed
            if need_linkup:
                try:
                    # Deduplicate against existing evidence
                    seen_urls = {ev.get('source') for ev in section_evidence if ev.get('source')}
                    linkup_result_count = 0

                    def _append_section_results(search_response, provider_label: str) -> int:
                        added_count = 0
                        for search_result in search_response.results:
                            if search_result.url and search_result.url not in seen_urls:
                                section_evidence.append({
                                    "source": search_result.url,
                                    "content": search_result.content or search_result.snippet,
                                    "relevance_score": search_result.relevance_score,
                                    "credibility_score": search_result.credibility_score,
                                    "source_type": "web",
                                    "metadata": {**(search_result.metadata or {}), "provider": provider_label},
                                })
                                seen_urls.add(search_result.url)
                                added_count += 1
                        return added_count

                    if research_provider_strategy in {'linkup_only', 'hybrid', 'auto'}:
                        linkup_api_key = get_linkup_api_key()
                        if not linkup_api_key:
                            logger.warning("  - Linkup API key not found in Supabase api_keys table")
                        else:
                            linkup_client = create_linkup_client(
                                api_key=linkup_api_key,
                                cache_enabled=optimization_config.cache_enabled
                            )
                            linkup_response = linkup_client.search(SearchQuery(query=section_query, depth='standard'))
                            if linkup_response.success:
                                linkup_result_count = len(linkup_response.results)
                                added = _append_section_results(linkup_response, "linkup")
                                logger.info(f"  - Linkup added {added} additional evidence items for section")
                            else:
                                logger.warning(f"  - Section Linkup search failed: {linkup_response.error}")

                    should_run_tavily = False
                    if research_provider_strategy == 'tavily_only':
                        should_run_tavily = True
                    elif research_provider_strategy == 'hybrid':
                        should_run_tavily = True
                    elif research_provider_strategy == 'auto':
                        should_run_tavily = linkup_result_count < optimization_config.deep_min_standard_results_threshold

                    if should_run_tavily:
                        from tavily_client import create_tavily_client
                        from supabase_client import get_api_key

                        tavily_api_key = get_api_key('tavily') or os.getenv('TAVILY_API_KEY')
                        if not tavily_api_key:
                            logger.warning("  - Tavily API key not found, skipping section Tavily search")
                        else:
                            tavily_client = create_tavily_client(api_key=tavily_api_key, timeout=30, max_results=6)
                            tavily_response = tavily_client.search(SearchQuery(query=section_query, depth='standard'))
                            if tavily_response.success:
                                added = _append_section_results(tavily_response, "tavily")
                                logger.info(f"  - Tavily added {added} additional evidence items for section")
                            else:
                                logger.warning(f"  - Section Tavily search failed: {tavily_response.error}")
                except Exception as e:
                    logger.error(f"  - Error in section Linkup search: {str(e)}")
                    logger.info("  - Continuing without Linkup evidence for this section")
        else:
            logger.info(f"  - Claims research disabled - using only RAG evidence for section")
        
        logger.info(f"  - Total section evidence: {len(section_evidence)} items")
        return section_evidence
            
    except Exception as e:
        logger.error(f"Error collecting section evidence: {str(e)}")
        return []

def _generate_content(result: Dict[str, Any], task_instance=None) -> Dict[str, Any]:
    """Generate article content using comprehensive content generator."""
    try:
        research_data = result.get('research_data', {})
        structure = result.get('structure', {})
        claims = result.get('claims', [])
        evidence = result.get('evidence', [])
        
        # Verify tone is being passed correctly - research_data is the source of truth (comes from API)
        tone_from_research = research_data.get('tone', 'journalistic')
        tone_from_structure = structure.get('tone', 'journalistic')
        
        # Log tones for debugging
        logger.info(f"📝 Content Generation - Tone from API/research_data: '{tone_from_research}'")
        logger.info(f"📝 Content Generation - Tone from structure: '{tone_from_structure}'")
        
        if tone_from_research != tone_from_structure:
            logger.warning(f"⚠️ Tone mismatch detected: research_data has '{tone_from_research}' but structure has '{tone_from_structure}'. Using research_data tone (source of truth from API).")
            # Override structure tone with research_data tone to ensure consistency
            structure['tone'] = tone_from_research
        
        # Use tone from research_data (source of truth - comes directly from API request)
        final_tone = tone_from_research
        logger.info(f"📝 Generating content with tone: '{final_tone}' (using research_data tone - source of truth from API request)")
        
        # Ensure research_data has the correct tone for downstream stages
        research_data['tone'] = final_tone
        
        # Ensure research_data has the correct tone for downstream stages
        research_data['tone'] = final_tone
        
        # Verify provider/model and fetch API key
        provider = research_data.get('provider')
        model = research_data.get('model')
        api_key = None
        
        if not provider or not model:
            logger.info("Provider or model not specified - fetching default from Supabase")
            def_provider, def_model, def_key = get_default_llm_provider()
            if def_provider and def_model and def_key:
                provider = def_provider
                model = def_model
                api_key = def_key
                # Update research_data
                research_data['provider'] = provider
                research_data['model'] = model
            else:
                 logger.warning("Failed to fetch default LLM provider - falling back to 'openai'/'gpt-4'")
                 provider = 'openai'
                 model = 'gpt-4'
        else:
             api_key = get_llm_api_key(provider, model)

        llm_client = create_llm_client(
            provider=provider,
            model=model,
            api_key=api_key,
            temperature=0.7,
            timeout=180  # Increased timeout for content generation (3 minutes)
        )
        
        # Create content generator
        use_verbalized_sampling = False
        content_generator = create_content_generator(llm_client, use_verbalized_sampling)
        
        sections = structure.get('sections', [])
        total_sections = len(sections)

        def _is_intro_section(title: str) -> bool:
            t = (title or "").strip().lower()
            return any(k in t for k in ["introduction", "overview"])

        def _is_conclusion_section(title: str) -> bool:
            t = (title or "").strip().lower()
            return any(k in t for k in ["conclusion", "final thoughts", "closing"])

        def _section_text(section_dict: Dict[str, Any]) -> str:
            blocks = section_dict.get("content_blocks") or []
            return " ".join([str(b.get("content", "")) for b in blocks if isinstance(b, dict)]).strip().lower()

        def _build_section_memory_context(memory: Dict[str, Any], section_outline: Dict[str, Any]) -> str:
            recent_sections = memory.get("recent_sections", [])[-3:]
            repeated = sorted(
                memory.get("theme_counts", {}).items(),
                key=lambda kv: kv[1],
                reverse=True
            )
            top_repeated = [k for k, v in repeated if v > 1][:5]
            uncovered_promises = sorted(list(memory.get("promised_points_remaining", set())))[:8]
            return (
                "Section continuity memory:\n"
                f"- Recently covered sections: {', '.join(recent_sections) if recent_sections else 'none yet'}\n"
                f"- Covered claim ids: {', '.join(sorted(memory.get('covered_claim_ids', set()))) if memory.get('covered_claim_ids') else 'none yet'}\n"
                f"- Used sources count: {len(memory.get('used_sources', set()))}\n"
                f"- Repeated themes to avoid overusing: {', '.join(top_repeated) if top_repeated else 'none'}\n"
                f"- Promised but not yet fully covered points: {', '.join(uncovered_promises) if uncovered_promises else 'none'}\n"
                f"- Current section key points: {', '.join(section_outline.get('key_points', [])[:6])}\n"
            )

        def _update_memory_from_section(
            memory: Dict[str, Any],
            section_outline: Dict[str, Any],
            section_dict: Dict[str, Any],
            section_evidence: List[Dict[str, Any]],
        ) -> None:
            text = _section_text(section_dict)
            title = section_outline.get("title", "")
            if title:
                memory["recent_sections"].append(title)
                for tok in re.findall(r"\b[a-z0-9]{4,}\b", title.lower()):
                    if tok not in _STOPWORDS:
                        memory["theme_counts"][tok] = memory["theme_counts"].get(tok, 0) + 1

            for kp in section_outline.get("key_points", []) or []:
                kp_norm = str(kp).strip()
                if not kp_norm:
                    continue
                memory["promised_points"].add(kp_norm)
                kp_tokens = [t for t in re.findall(r"\b[a-z0-9]{4,}\b", kp_norm.lower()) if t not in _STOPWORDS]
                covered = any(tok in text for tok in kp_tokens[:4]) if kp_tokens else False
                if not covered:
                    memory["promised_points_remaining"].add(kp_norm)
                elif kp_norm in memory["promised_points_remaining"]:
                    memory["promised_points_remaining"].remove(kp_norm)

            for claim in claims:
                cid = claim.get("claim_id")
                if not cid:
                    continue
                kws = claim.get("keywords") or _claim_keywords(claim.get("claim", ""))
                if kws and sum(1 for k in kws[:8] if k in text) >= 2:
                    memory["covered_claim_ids"].add(cid)

            for ev in section_evidence:
                ev_url = ev.get("source") or ev.get("url")
                if ev_url:
                    memory["used_sources"].add(ev_url)

        # Phase 4 hybrid orchestration:
        # 1) collect section-specific evidence in parallel (fast IO step)
        # 2) generate section prose sequentially with live memory for coherence
        claims_research_enabled = research_data.get('claims_research_enabled', True)
        rag_enabled = research_data.get('rag_enabled', False)
        section_specific_map: Dict[int, List[Dict[str, Any]]] = {}
        all_section_evidence = []
        seen_urls = {ev.get('source') for ev in evidence if ev.get('source')}

        def _prefetch_section_evidence(idx: int, section_outline: Dict[str, Any]) -> tuple:
            section_title = section_outline.get('title', 'Unknown Section')
            section_title_lower = section_title.lower()
            if section_title_lower in ['introduction', 'conclusion', 'overview', 'summary']:
                return idx, []
            if (rag_enabled and research_data.get('rag_endpoint')) or claims_research_enabled:
                return idx, _collect_section_evidence(section_outline, research_data)
            return idx, []

        prefetch_workers = min(max(total_sections, 1), 5)
        logger.info(
            f"🚀 Prefetching section evidence in parallel with {prefetch_workers} workers for {total_sections} sections"
        )
        with concurrent.futures.ThreadPoolExecutor(max_workers=prefetch_workers) as executor:
            futures = {
                executor.submit(_prefetch_section_evidence, i, section): i
                for i, section in enumerate(sections)
            }
            for future in concurrent.futures.as_completed(futures):
                idx = futures[future]
                try:
                    sec_idx, sec_evidence = future.result()
                    section_specific_map[sec_idx] = sec_evidence
                    for ev in sec_evidence:
                        ev_url = ev.get('source') or ev.get('url', '')
                        if ev_url and ev_url not in seen_urls:
                            all_section_evidence.append(ev)
                            seen_urls.add(ev_url)
                except Exception as e:
                    logger.error(f"Error prefetching evidence for section {idx + 1}: {str(e)}")
                    section_specific_map[idx] = []

        # Generate body first, intro second-to-last, conclusion last when present.
        intro_idxs = [i for i, s in enumerate(sections) if _is_intro_section(s.get("title", ""))]
        concl_idxs = [i for i, s in enumerate(sections) if _is_conclusion_section(s.get("title", ""))]
        normal_idxs = [i for i in range(total_sections) if i not in set(intro_idxs + concl_idxs)]
        generation_order = normal_idxs + intro_idxs + concl_idxs

        logger.info(
            f"🧠 Sequential drafting order (Phase 4): {generation_order} | intro={intro_idxs} conclusion={concl_idxs}"
        )
        generated_sections_map: Dict[int, Dict[str, Any]] = {}
        generated_section_objects = []
        section_memory = {
            "covered_claim_ids": set(),
            "used_sources": set(),
            "theme_counts": {},
            "recent_sections": [],
            "promised_points": set(),
            "promised_points_remaining": set(),
            "similarity_alerts": 0,
        }

        completed_count = 0
        for ordinal, idx in enumerate(generation_order):
            section_outline = sections[idx]
            section_title = section_outline.get('title', f'Section {idx + 1}')
            section_evidence = evidence.copy()
            section_evidence.extend(section_specific_map.get(idx, []))

            section_research_data = dict(research_data)
            section_research_data["section_memory_context"] = _build_section_memory_context(
                section_memory, section_outline
            )

            prev_context = generated_section_objects[-3:]
            section_content = content_generator.generate_section_content(
                section_outline, section_research_data, claims, section_evidence, prev_context
            )
            section_dict = {
                'title': section_content.title,
                'subtitle': section_content.subtitle,
                'content_blocks': [
                    {
                        'content': block.content,
                        'content_type': block.content_type,
                        'word_count': block.word_count,
                        'citations': block.citations or [],
                        'metadata': block.metadata or {}
                    }
                    for block in section_content.content_blocks
                ],
                'total_word_count': section_content.total_word_count,
                'key_points_covered': section_content.key_points_covered,
                'citations': section_content.citations,
                'section_order': section_content.section_order
            }
            generated_sections_map[idx] = section_dict
            generated_section_objects.append(section_content)
            _update_memory_from_section(
                section_memory,
                section_outline=section_outline,
                section_dict=section_dict,
                section_evidence=section_evidence,
            )

            # lightweight anti-repetition diagnostic across adjacent sections
            if len(generated_section_objects) >= 2:
                prev_text = " ".join(
                    b.content for b in generated_section_objects[-2].content_blocks if getattr(b, "content", "")
                ).lower()
                curr_text = " ".join(
                    b.content for b in generated_section_objects[-1].content_blocks if getattr(b, "content", "")
                ).lower()
                prev_set = set(re.findall(r"\b[a-z0-9]{4,}\b", prev_text[:1200]))
                curr_set = set(re.findall(r"\b[a-z0-9]{4,}\b", curr_text[:1200]))
                overlap = len(prev_set.intersection(curr_set)) / max(len(prev_set.union(curr_set)), 1)
                if overlap > 0.52:
                    section_memory["similarity_alerts"] += 1
                    logger.warning(
                        f"High adjacent-section similarity detected ({overlap:.2f}) between '{generated_section_objects[-2].title}' and '{generated_section_objects[-1].title}'"
                    )

            completed_count += 1
            if task_instance:
                section_progress = 70 + int((completed_count / max(total_sections, 1)) * 10)
                task_instance.update_state(
                    state=TASK_STATUS['PROGRESS'],
                    meta={
                        'current_stage': 'CONTENT_GENERATION',
                        'progress': section_progress,
                        'message': f'Completed {completed_count}/{total_sections} sections (sequential draft)...'
                    }
                )

        # Restore original section order for downstream steps and finalization
        generated_sections = [generated_sections_map[i] for i in range(total_sections) if i in generated_sections_map]
        
        # Aggregate all evidence
        aggregated_evidence = evidence.copy()
        aggregated_evidence.extend(all_section_evidence)
        
        # Calculate total statistics
        total_words = sum(s.get('total_word_count', 0) for s in generated_sections)
        total_citations = sum(len(s.get('citations', [])) for s in generated_sections)
        
        logger.info(f"Generated content for {len(generated_sections)} sections with {total_words} total words in hybrid mode")
        
        return {
            'content': {
                'sections': generated_sections,
                'word_count': total_words
            },
            'aggregated_evidence': aggregated_evidence,
            'stage_data': {
                'sections_written': len(generated_sections),
                'word_count': total_words,
                'total_citations': total_citations,
                'average_words_per_section': total_words // len(generated_sections) if generated_sections else 0,
                'llm_model': llm_client.config.model,
                'aggregated_evidence_count': len(aggregated_evidence),
                'parallel_execution': False,
                'hybrid_execution': True,
                'evidence_prefetch_parallel': True,
                'evidence_prefetch_workers': prefetch_workers,
                'generation_order': generation_order,
                'section_memory_covered_claims': len(section_memory['covered_claim_ids']),
                'section_memory_used_sources': len(section_memory['used_sources']),
                'section_memory_promised_points_remaining': len(section_memory['promised_points_remaining']),
                'adjacent_similarity_alerts': section_memory['similarity_alerts']
            }
        }
        
    except Exception as e:
        logger.error(f"Error in content generation: {str(e)}", exc_info=True)
        # Fail fast so clients see the real root cause (e.g. auth/model errors),
        # instead of a downstream generic finalization failure.
        raise RuntimeError(f"Content generation stage failed: {str(e)}") from e

def _generate_citations(result: Dict[str, Any]) -> Dict[str, Any]:
    """Generate citations and references using comprehensive citation generator."""
    try:
        research_data = result.get('research_data', {})
        # Use aggregated evidence from content generation if available, otherwise use ranked evidence
        evidence = result.get('aggregated_evidence') or result.get('ranked_evidence', [])
        content = result.get('content', {})
        
        # Debug logging
        logger.info(f"🔍 Citation generation debug - Evidence count: {len(evidence)}")
        logger.info(f"🔍 Citation generation debug - Evidence source: {'aggregated_evidence' if result.get('aggregated_evidence') else 'ranked_evidence'}")
        logger.info(f"🔍 Citation generation debug - Evidence keys: {list(evidence[0].keys()) if evidence else 'No evidence'}")
        logger.info(f"🔍 Citation generation debug - Content sections: {len(content.get('sections', []))}")
        
        # Support both llm_key (legacy) and api_key (normalized)
        api_key = research_data.get('api_key') or research_data.get('llm_key', '')
        llm_client = create_llm_client(
            provider=research_data.get('provider', 'openai'),
            model=research_data.get('model', 'gpt-4'),
            api_key=api_key,
            temperature=0.3
        )
        
        # Create citation generator
        citation_generator = create_citation_generator(llm_client, CitationStyle.APA)
        
        # Pre-process evidence to ensure proper citation data
        processed_evidence = []
        for i, ev in enumerate(evidence):
            processed_ev = ev.copy()
            
            # Ensure proper title and URL for citations
            if ev.get('source_type') == 'rag':
                # Extract title from metadata - now we have proper titles from RAG
                title = ev.get('metadata', {}).get('title', '')
                if not title:
                    # Only create title from actual content - don't use generic fallbacks
                    content_preview = ev.get('content', '')[:100] if ev.get('content') and ev.get('content').strip() else ''
                    if content_preview:
                        # Extract first sentence or meaningful phrase
                        first_sentence = content_preview.split('.')[0]
                        if len(first_sentence) > 150:
                            title = first_sentence[:150] + "..."
                        else:
                            title = first_sentence
                    else:
                        # Skip this evidence if it has no content - don't create generic title
                        logger.warning(f"Skipping evidence {i+1} - no title and no content available")
                        continue  # Skip this evidence entirely rather than creating a generic citation
                
                # Extract URL from metadata
                url = ev.get('metadata', {}).get('url', '')
                if not url:
                    url = ev.get('source', '#')
                
                # Extract author from metadata - now we have proper authors from RAG
                author = ev.get('metadata', {}).get('author', '')
                if not author:
                    author = "Unknown Author"
                
                # Extract publication date from metadata
                publication_date = ev.get('metadata', {}).get('publication_date', '')
                
                # Update the evidence with proper citation data
                processed_ev.update({
                    'title': title,
                    'url': url,
                    'author': author,
                    'source_title': title,
                    'publication_date': publication_date,
                    'publisher': ev.get('metadata', {}).get('publisher', '')
                })
            else:
                # For web sources, ensure we have proper citation data
                # Only include sources with actual content
                if not processed_ev.get('content') or not processed_ev.get('content').strip():
                    logger.warning(f"Skipping web source {i+1} - no content available")
                    continue  # Skip evidence without content
                    
                if not processed_ev.get('title'):
                    # Try to extract title from content if possible
                    content_preview = processed_ev.get('content', '')[:100] if processed_ev.get('content') else ''
                    if content_preview:
                        first_sentence = content_preview.split('.')[0]
                        if len(first_sentence) > 150:
                            processed_ev['title'] = first_sentence[:150] + "..."
                        else:
                            processed_ev['title'] = first_sentence
                    else:
                        # Skip if we can't create a meaningful title from content
                        logger.warning(f"Skipping web source {i+1} - no title or content available")
                        continue
                if not processed_ev.get('url'):
                    processed_ev['url'] = processed_ev.get('source', '#')
            
            processed_evidence.append(processed_ev)
        
        # Only generate citations if we have actual evidence with content
        if not processed_evidence:
            logger.warning("⚠️ No evidence available - skipping citation generation. No citations will be created without real evidence sources.")
            return {
                'citations': [],
                'formatted_citations': [],
                'reference_list': ['References', '', 'This article is based on general knowledge and industry best practices. No specific sources were cited as no evidence sources were available during generation.'],
                'processed_sections': content.get('sections', []),
                'stage_data': {
                    'generated_citations': 0,
                    'citation_style': 'apa',
                    'reference_count': 0,
                    'note': 'No evidence available - no citations generated'
                }
            }
        
        # Filter out evidence with no actual content - don't create citations from empty evidence
        valid_evidence = [ev for ev in processed_evidence if ev.get('content') and ev.get('content').strip()]
        
        if not valid_evidence:
            logger.warning(f"⚠️ No valid evidence with content - skipping citation generation. All {len(processed_evidence)} evidence items have empty content.")
            return {
                'citations': [],
                'formatted_citations': [],
                'reference_list': ['References', '', 'This article is based on general knowledge and industry best practices. No specific sources were cited as no evidence sources were available during generation.'],
                'processed_sections': content.get('sections', []),
                'stage_data': {
                    'generated_citations': 0,
                    'citation_style': 'apa',
                    'reference_count': 0,
                    'note': 'No evidence with content - no citations generated'
                }
            }
        
        # Check if in-text citations should be included
        include_in_text_citations = research_data.get('include_in_text_citations', True)
        logger.info(f"Citation generation - include_in_text_citations: {include_in_text_citations}")
        
        # Generate citations only from valid evidence
        citation_result = citation_generator.generate_citations(
            evidence=valid_evidence,  # Use only evidence with actual content
            content_sections=content.get('sections', []),
            style=CitationStyle.APA,
            include_in_text_citations=include_in_text_citations
        )
        
        logger.info(f"Generated {citation_result['total_citations']} citations from {len(valid_evidence)} valid evidence sources in {citation_result['style']} style")
        
        # Convert Citation objects to dictionaries for JSON serialization
        def convert_to_dict(obj):
            """Convert objects to dictionaries for JSON serialization."""
            if hasattr(obj, '__dict__'):
                return obj.__dict__
            elif isinstance(obj, dict):
                return {k: convert_to_dict(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_dict(item) for item in obj]
            else:
                return obj
        
        citations_dict = convert_to_dict(citation_result['citations'])
        
        return {
            'citations': citations_dict,
            'formatted_citations': convert_to_dict(citation_result['formatted_citations']),
            'reference_list': convert_to_dict(citation_result['reference_list']),
            'processed_sections': convert_to_dict(citation_result['processed_sections']),
            'stage_data': {
                'generated_citations': citation_result['total_citations'],
                'citation_style': citation_result['style'],
                'reference_count': len(citation_result['reference_list']),
                'llm_model': llm_client.config.model
            }
        }
        
    except Exception as e:
        logger.error(f"Error in citation generation: {str(e)}")
        return {
            'citations': [],
            'formatted_citations': [],
            'reference_list': [],
            'processed_sections': [],
            'stage_data': {'generated_citations': 0, 'error': str(e)}
        }

def _build_refinement_user_message(tone: str, original_content: str) -> str:
    """Build user message for refinement with proper tone handling."""
    tone_upper = tone.upper()
    
    # Build tone-specific guidance
    tone_guidance = ""
    if tone.lower() == 'friendly':
        tone_guidance = "\n\nFOR FRIENDLY TONE: Make it personal, use first-person storytelling, include specific examples, and write like you're talking to a friend. Avoid formal words like \"individuals\", \"necessitates\", \"crucial\"."
    elif tone.lower() == 'journalistic':
        tone_guidance = "\n\nFOR JOURNALISTIC TONE: Write in a clear, objective journalistic style with proper attribution and balanced reporting."
    elif tone.lower() == 'professional':
        tone_guidance = "\n\nFOR PROFESSIONAL TONE: Write clearly and professionally, using accessible language while maintaining authority."
    
    return f"""IMPORTANT: The tone for this article is {tone_upper}.

Refine this section to match the {tone} tone perfectly. {tone_guidance}

Return ONLY the refined HTML content - no explanations, no meta-commentary, no "Here's the refined content" text. Start directly with the HTML.

Original content:
{original_content}"""

def _refine_article(result: Dict[str, Any], task_instance=None) -> Dict[str, Any]:
    """Refine and optimize article using LLM."""
    try:
        research_data = result.get('research_data', {})
        content = result.get('content', {})
        tone = research_data.get('tone', 'journalistic')
        include_in_text_citations = research_data.get('include_in_text_citations', True)
        
        # Verify tone is correct - log warning if it seems wrong
        if tone.lower() not in ['friendly', 'professional', 'journalistic', 'casual', 'academic', 'technical', 'persuasive']:
            logger.warning(f"⚠️ Unusual tone value: '{tone}' - proceeding anyway")
        
        # Log tone for debugging
        logger.info(f"🔍 REFINEMENT STAGE - Tone from research_data: '{tone}'")
        if tone.lower() == 'friendly':
            logger.info(f"🔍 REFINEMENT STAGE - Friendly tone detected - should use first-person, personal stories, casual language")
        
        # Create LLM client
        # Support both llm_key (legacy) and api_key (normalized)
        api_key = research_data.get('api_key') or research_data.get('llm_key', '')
        llm_client = create_llm_client(
            provider=research_data.get('provider', 'openai'),
            model=research_data.get('model', 'gpt-4'),
            api_key=api_key,
            temperature=0.5,
            timeout=60,  # Reduced timeout to prevents hangs (60s per section)
            max_retries=1 # Reduce retries to fail fast
        )
        
        # Get tone-specific instructions for refinement
        tone_instructions = get_tone_specific_instructions(tone)
        
        # Log tone for debugging
        logger.info(f"🔍 Refinement - Using tone: '{tone}' (from research_data)")
        logger.info(f"🔍 Refinement - Tone instructions length: {len(tone_instructions)} chars")
        from src.services.humanization_service import analyze_humanization

        def _clean_llm_html(raw_text: str) -> str:
            cleaned = (raw_text or "").strip()
            patterns_to_remove = [
                r'^Here\'s the refined content[^\n]*\n*',
                r'^Here is the refined content[^\n]*\n*',
                r'^Refined content[^\n]*\n*',
                r'^Here\'s the improved version[^\n]*\n*',
                r'^Here is the improved version[^\n]*\n*',
                r'optimized for [^\n]*tone[^\n]*\n*',
                r'with improved clarity[^\n]*\n*',
                r'^[^\<]*?(?=<)',
                r'^.*?optimized for.*?\n',
                r'^.*?refined content.*?\n',
                r'^.*?improved version.*?\n',
            ]
            for pattern in patterns_to_remove:
                cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE | re.MULTILINE)
            if not cleaned.strip().startswith('<'):
                html_match = re.search(r'<[^>]+>', cleaned)
                if html_match:
                    cleaned = cleaned[html_match.start():]
            return cleaned.strip()

        def _run_refinement_pass(pass_name: str, system_content: str, input_html: str) -> Optional[str]:
            try:
                response = llm_client.generate([
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": _build_refinement_user_message(tone, input_html)},
                ])
                return _clean_llm_html(response.content)
            except Exception as run_err:
                logger.error(f"Failed {pass_name} pass: {run_err}")
                return None
        
        # Helper function to remove citation references
        def remove_citations_from_text(text: str) -> str:
            """Remove citation references like [1], [2] from text."""
            if not text or include_in_text_citations:
                return text
            import re
            # Remove citation references like [1], [2], [3], etc.
            citation_pattern = r'\[\d+\]'
            text = re.sub(citation_pattern, '', text)
            # Clean up any extra spaces left behind
            text = re.sub(r'\s+', ' ', text)
            text = re.sub(r'\s+([.,;:!?])', r'\1', text)  # Remove space before punctuation
            return text.strip()
        
        # Refine each section
        refinements = []
        sections = content.get('sections', [])
        total_sections = len(sections)
        failed_sections = 0
        factual_pass_applied = 0
        humanization_pass_applied = 0
        cleanup_pass_applied = 0
        weak_sections_detected = 0
        
        for i, section in enumerate(sections):
            # Update progress if task instance is available
            if task_instance:
                try:
                    current_progress = 90 + int((i / total_sections) * 5)  # 90-95%
                    section_title_display = section.get('title', f'Section {i+1}')
                    task_instance.update_state(
                        state='PROGRESS',
                        meta={
                            'current_stage': 'REFINEMENT',
                            'progress': current_progress,
                            'message': f"Refining section {i+1} of {total_sections}: {section_title_display}..."
                        }
                    )
                except Exception as e:
                    logger.warning(f"Failed to update task state during refinement: {e}")

            # Circuit breaker: Stop refinement if 2 or more sections failed to avoid hanging forever
            if failed_sections >= 2:
                logger.warning(f"⚠️ Refinement circuit breaker triggered! {failed_sections} sections failed. Skipping remaining sections to ensure article completion.")
                logger.warning("Breaking refinement loop - proceeding to cleanup and finalization.")
                break

            # Skip references section - it should not be refined and citations should be preserved there
            section_title = section.get('title', '') or section.get('heading', '')
            if section_title and 'reference' in section_title.lower():
                logger.info(f"Skipping refinement for references section: '{section_title}'")
                continue
            
            # Extract content from section - handle both content_blocks and direct content field
            section_content_blocks = section.get('content_blocks', [])
            if section_content_blocks and isinstance(section_content_blocks, list):
                # Extract content from content_blocks
                original_content = '\n\n'.join([
                    block.get('content', '') 
                    for block in section_content_blocks 
                    if isinstance(block, dict) and block.get('content')
                ])
            else:
                # Use direct content field
                original_content = section.get('content', '') or section.get('text', '') or ''
            
            if not original_content.strip():
                logger.warning(f"Skipping refinement for section '{section_title}' - no content found")
                continue
            
            # Determine citation handling instructions
            citation_instructions = ""
            if not include_in_text_citations:
                citation_instructions = """
                    
                    CRITICAL - CITATION REMOVAL:
                    - Remove ALL in-text citation references like [1], [2], [3], etc. from the content
                    - Do NOT include any citation markers in the refined content
                    - The references section will be preserved separately, so remove all inline citations
                    - Clean up any spaces left after removing citations
                    - Make sure the text flows naturally without citation markers"""
            else:
                citation_instructions = """
                    
                    CITATION HANDLING:
                    - Preserve all in-text citation references like [1], [2], [3], etc. as-is
                    - Do not remove or modify citation markers"""
            
            # Add friendly tone specific checks if needed
            friendly_checks = ""
            if tone.lower() == 'friendly':
                friendly_checks = """
                    
                    FOR FRIENDLY TONE - CRITICAL CHECKS:
                    - Does it use first-person storytelling ("I've found", "Last month I", "My favorite")?
                    - Is it personal and conversational, not formal or professional?
                    - Does it have specific, relatable examples with details?
                    - Is it warm and engaging, not boring or academic?
                    - Does it avoid formal words like "crucial", "paramount", "necessitates", "individuals"?
                    - Does it sound like someone talking to a friend, not writing a report?
                    """
            
            # Log the exact tone being used
            logger.info(f"🔍 Refining section '{section_title}' with tone: '{tone}'")
            if tone.lower() == 'friendly':
                logger.info(f"🔍 Friendly tone - expecting: first-person, personal stories, casual language, warm and engaging")
            
            # Build tone-specific warnings (only warn against other tones, not the requested one)
            tone_warnings = ""
            if tone.lower() != 'journalistic':
                tone_warnings += "\n                    - DO NOT use journalistic tone - this is WRONG for this article"
            if tone.lower() != 'professional':
                tone_warnings += "\n                    - DO NOT use professional tone - this is WRONG for this article (unless tone is professional)"
            if tone.lower() not in ['academic', 'formal']:
                tone_warnings += "\n                    - DO NOT use academic or formal tone - this is WRONG for this article"
            
            # Phase 5 multi-pass refinement:
            # 1) factual integrity pass
            # 2) humanization pass (targeted only when weak)
            # 3) final cleanup/tone consistency pass
            refined_content = original_content

            factual_prompt = f"""You are a factual integrity editor.
Maintain original meaning and factual claims while improving precision and clarity.
Rules:
- Do not invent facts.
- Remove overstatements that are not supported.
- Preserve all HTML structure and any citation markers.
- Keep wording concise and concrete.
- Return only cleaned HTML."""
            factual_result = _run_refinement_pass("factual_integrity", factual_prompt, refined_content)
            if factual_result:
                refined_content = factual_result
                factual_pass_applied += 1

            diagnostics = analyze_humanization(refined_content)
            weak_reasons = diagnostics.get("weak_reasons", [])
            if diagnostics.get("needs_humanization_rewrite"):
                weak_sections_detected += 1
                humanization_prompt = f"""You are a senior human editor.
Rewrite this section so it sounds naturally human, not robotic.
Target fixes: {", ".join(weak_reasons) if weak_reasons else "robotic style"}.
Rules:
- Vary sentence cadence.
- Prefer concrete language over abstract filler.
- Reduce hedging and repetitive transitions.
- Remove banned/generic phrases.
- Keep the same facts and HTML structure.
- Keep tone as {tone}.
- Return only revised HTML."""
                humanized = _run_refinement_pass("humanization", humanization_prompt, refined_content)
                if humanized:
                    refined_content = humanized
                    humanization_pass_applied += 1

            cleanup_prompt = f"""You are an expert editor. Review and refine the content to ensure it matches the {tone} tone perfectly, while improving clarity, flow, and engagement.

            ========================================
            ⚠️ CRITICAL: THE TONE FOR THIS ARTICLE IS {tone.upper()} ⚠️
            ========================================
            YOU MUST USE ONLY THE {tone.upper()} TONE AS SPECIFIED BELOW
            {tone_warnings}
            
            The tone is {tone} - use ONLY this tone, not any other tone.
            
            ========================================
            TONE REQUIREMENTS (HIGHEST PRIORITY)
            ========================================
            {tone_instructions}
            {friendly_checks}
            
            ========================================
            REFINEMENT TASKS
            ========================================
            - Ensure the content consistently follows the {tone} tone throughout EVERY sentence
            - Improve clarity and readability while maintaining the {tone} tone
            - Enhance flow and transitions between ideas
            - Make sure complex concepts are explained simply (especially for friendly tone)
            - Ensure the language matches the {tone} tone perfectly
            - Keep the content engaging and natural
            - Maintain the original meaning and factual accuracy
            - Maintain HTML structure exactly as provided
            {citation_instructions}
            
            ========================================
            OUTPUT REQUIREMENTS
            ========================================
            - Return ONLY the refined HTML content, no meta-commentary."""
            cleanup_result = _run_refinement_pass("cleanup", cleanup_prompt, refined_content)
            if not cleanup_result:
                logger.error(f"Failed to refine section '{section_title}' in final cleanup")
                failed_sections += 1
                continue
            refined_content = cleanup_result
            cleanup_pass_applied += 1
            
            # For friendly tone, do an additional check and fix if needed
            if tone.lower() == 'friendly':
                # Check if content still has formal language that shouldn't be there
                formal_words = ['individuals', 'necessitates', 'crucial', 'paramount', 'cultivate', 'strategic', 'trajectory', 'implement', 'ensure', 'facilitate']
                content_lower = refined_content.lower()
                found_formal = [word for word in formal_words if word in content_lower]
                if found_formal:
                    logger.warning(f"⚠️ Friendly tone content still contains formal words: {found_formal[:3]} - content may need stronger tone enforcement")
            
            # Remove citations from refined content if flag is disabled (double-check in case LLM didn't follow instructions)
            if not include_in_text_citations:
                refined_content = remove_citations_from_text(refined_content)
            
            # Update the section with refined content
            if section_content_blocks and isinstance(section_content_blocks, list):
                # Update the first content block with refined content, or create a new one
                if section_content_blocks:
                    section_content_blocks[0]['content'] = refined_content
                else:
                    section_content_blocks.append({
                        'content': refined_content,
                        'content_type': 'paragraph',
                        'word_count': len(refined_content.split())
                    })
                section['content_blocks'] = section_content_blocks
            else:
                # Update direct content field
                section['content'] = refined_content
            
            section['refined'] = True
            section['refined_at'] = datetime.utcnow().isoformat()
            
            refinements.append({
                'section': section_title,
                'original_word_count': len(original_content.split()),
                'refined_word_count': len(refined_content.split()),
                'improvements': [
                    'Factual integrity pass',
                    'Humanization pass' if diagnostics.get("needs_humanization_rewrite") else 'Humanization check passed',
                    'Final cleanup pass',
                    'Citations handled' if not include_in_text_citations else 'Citations preserved'
                ],
                'humanization': diagnostics,
            })
        
        logger.info(f"Applied {len(refinements)} refinements using {llm_client.config.model}")
        
        # Update the result with refined content
        result['content'] = content
        
        return {
            'refinements': refinements,
            'stage_data': {
                'refinements_applied': len(refinements),
                'factual_pass_applied': factual_pass_applied,
                'humanization_pass_applied': humanization_pass_applied,
                'cleanup_pass_applied': cleanup_pass_applied,
                'weak_sections_detected': weak_sections_detected,
            }
        }
        
    except Exception as e:
        logger.error(f"Error in article refinement: {str(e)}")
        return {'refinements': [], 'stage_data': {'refinements_applied': 0, 'error': str(e)}}

def _finalize_article(result: Dict[str, Any], task_instance: Any = None) -> Dict[str, Any]:
    """Finalize the article."""
    try:
        structure = result.get('structure', {})
        content = result.get('content', {})
        citations = result.get('citations', [])
        claim_bundles = result.get('claim_bundles', [])
        research_data = result.get('research_data', {})
        include_in_text_citations = research_data.get('include_in_text_citations', True)
        
        # Debug logging
        logger.info(f"Finalization debug - Structure keys: {list(structure.keys())}")
        logger.info(f"Finalization debug - Content keys: {list(content.keys())}")
        logger.info(f"Finalization debug - Content sections: {len(content.get('sections', []))}")
        if content.get('sections'):
            first_section = content['sections'][0]
            logger.info(f"Finalization debug - First section keys: {list(first_section.keys())}")
            logger.info(f"Finalization debug - First section title: {first_section.get('title', 'NO_TITLE')}")
            logger.info(f"Finalization debug - First section content_blocks type: {type(first_section.get('content_blocks'))}")
            logger.info(f"Finalization debug - First section content_blocks: {first_section.get('content_blocks', 'NO_CONTENT_BLOCKS')}")
        
        # Combine all content - handle different content structures
        full_content = ""
        geo_auto_applied = False
        
        # Function to remove citation references if needed
        def remove_citations_from_text(text: str) -> str:
            """Remove citation references like [1], [2] from text."""
            if not text or include_in_text_citations:
                return text
            import re
            # Remove citation references like [1], [2], [3], etc.
            citation_pattern = r'\[\d+\]'
            text = re.sub(citation_pattern, '', text)
            # Clean up any extra spaces left behind
            text = re.sub(r'\s+', ' ', text)
            text = re.sub(r'\s+([.,;:!?])', r'\1', text)  # Remove space before punctuation
            return text.strip()
        
        # Try different content structures
        sections = content.get('sections', [])
        if not sections:
            # Fallback: try to get content directly
            full_content = content.get('content', '')
            if not include_in_text_citations:
                full_content = remove_citations_from_text(full_content)
        else:
            for section in sections:
                # Try different field names for heading and content
                heading = section.get('heading') or section.get('title') or section.get('name', '')
                
                # Skip references section - it should preserve all citation markers
                # References section will be added separately at the end
                if heading and 'reference' in heading.lower():
                    logger.info(f"Skipping references section in finalization: '{heading}' - will be added separately")
                    continue
                
                # Try different content field names, including content_blocks
                section_content = (section.get('content') or 
                                 section.get('text') or 
                                 section.get('body') or 
                                 section.get('content_blocks', ''))
                
                # If content_blocks is a list, extract the 'content' field from each block
                if isinstance(section_content, list):
                    content_parts = []
                    for block in section_content:
                        if isinstance(block, dict) and 'content' in block:
                            block_content = block['content']
                            # Remove citations if flag is disabled (but not from references section)
                            if not include_in_text_citations:
                                block_content = remove_citations_from_text(block_content)
                            content_parts.append(block_content)
                        else:
                            content_parts.append(str(block))
                    section_content = '\n\n'.join(content_parts)
                elif not include_in_text_citations:
                    # Remove citations from section content if it's a string (but not from references section)
                    section_content = remove_citations_from_text(str(section_content))
                
                if heading:
                    full_content += f"<h2>{heading}</h2>\n\n"
                if section_content:
                    full_content += f"{section_content}\n\n"
                    logger.info(f"Finalization debug - Added content for section '{heading}': {len(section_content)} chars")
                else:
                    logger.warning(f"Finalization debug - No content found for section '{heading}'")
        
        # If still empty, create a basic structure
        # If still empty, raise an error - do NOT create fake success message
        # If still empty or very short (just headings), raise an error
        current_word_count = content.get('word_count', 0)
        if not full_content.strip() or (current_word_count < 50 and len(sections) > 0):
            logger.error(f"Finalization failed: Content too short ({current_word_count} words)")
            raise ValueError(f"Content generation failed: Produced only {current_word_count} words")

        # Phase 6 GEO enforcement: ensure answer-first block near top when enabled.
        geo_enforce_answer_first = bool(research_data.get('geo_enforce_answer_first', True))
        lower_content = full_content.lower()
        has_answer_first = any(
            marker in lower_content for marker in [
                "<h2>short answer",
                "<h2>quick answer",
                "in short,",
                "the short answer",
            ]
        )
        if geo_enforce_answer_first and not has_answer_first:
            title = structure.get('title', 'this topic')
            thesis = structure.get('thesis', '') or structure.get('excerpt', '')
            answer_block = (
                "<h2>Short Answer</h2>\n\n"
                f"<p>In short: {thesis[:280].strip() if thesis else f'The best approach to {title} is to prioritize clear goals, evidence-based decisions, and practical execution.'}</p>\n\n"
            )
            full_content = answer_block + full_content
            geo_auto_applied = True
        
        # Create clickable citation links only if in-text citations are enabled
        if include_in_text_citations:
            html_content_with_citations = _create_citation_links(full_content, citations)
        else:
            # If in-text citations are disabled, use the content as-is without citation links
            html_content_with_citations = full_content
        
        # Add References section - always generate references from evidence, even if in-text citations are disabled
        # Get evidence from the result to generate references
        evidence_for_references = result.get('ranked_evidence', []) or result.get('evidence', [])
        references_section = ""
        logger.info(f"Finalization debug - Citations count: {len(citations) if citations else 0}")
        logger.info(f"Finalization debug - Evidence count for references: {len(evidence_for_references)}")
        
        # Use citations if available, otherwise generate references from evidence
        if citations and len(citations) > 0:
            references_section = "\n\n<hr>\n\n<h2>References</h2>\n\n"
            for i, citation in enumerate(citations, 1):
                # Extract citation details
                title = citation.get('title', citation.get('source_title', 'Unknown Source'))
                url = citation.get('url', '#')
                author = citation.get('author', '')
                source_type = citation.get('source_type', 'unknown')
                publication_date = citation.get('publication_date', '')
                
                # Format the reference with proper style
                references_section += f'<p id="ref-{i}"><strong>[{i}]</strong> '
                
                if author and author != "Unknown Author":
                    references_section += f"{author}"
                    if publication_date:
                        references_section += f" ({publication_date})"
                    references_section += ". "
                elif source_type == 'rag' and author == "Unknown Author":
                    references_section += "Unknown Author"
                    if publication_date:
                        references_section += f" ({publication_date})"
                    references_section += ". "
                else:
                    # Fallback for any other case
                    if publication_date:
                        references_section += f"({publication_date}) "
                
                if url and url != '#' and url != '':
                    references_section += f'<a href="{url}" target="_blank" rel="noopener noreferrer">{title}</a>'
                else:
                    references_section += f"<em>{title}</em>"
                
                references_section += ".</p>\n"
        elif evidence_for_references and len(evidence_for_references) > 0:
            # No citations but we have evidence - generate references from evidence
            references_section = "\n\n<hr>\n\n<h2>References</h2>\n\n"
            for i, ev in enumerate(evidence_for_references, 1):
                # Only include evidence with actual content
                if not ev.get('content') or not ev.get('content').strip():
                    continue
                    
                title = ev.get('title') or ev.get('source_title', 'Unknown Source')
                url = ev.get('source') or ev.get('url', '#')
                author = ev.get('author', '')
                source_type = ev.get('source_type', 'unknown')
                publication_date = ev.get('publication_date', '')
                
                # Format the reference
                references_section += f'<p id="ref-{i}"><strong>[{i}]</strong> '
                
                if author and author != "Unknown Author":
                    references_section += f"{author}"
                    if publication_date:
                        references_section += f" ({publication_date})"
                    references_section += ". "
                elif source_type == 'rag':
                    references_section += "Unknown Author"
                    if publication_date:
                        references_section += f" ({publication_date})"
                    references_section += ". "
                else:
                    if publication_date:
                        references_section += f"({publication_date}) "
                
                if url and url != '#' and url != '':
                    references_section += f'<a href="{url}" target="_blank" rel="noopener noreferrer">{title}</a>'
                else:
                    references_section += f"<em>{title}</em>"
                
                references_section += ".</p>\n"
            
            if not references_section.endswith("</p>\n"):
                # No valid evidence with content was found
                references_section = "\n\n<hr>\n\n<h2>References</h2>\n\n<p><em>This article is based on general knowledge and industry best practices. No specific sources were cited as no evidence sources were available during generation.</em></p>\n"
        else:
            # No citations and no evidence available - add a note instead
            references_section = "\n\n<hr>\n\n<h2>References</h2>\n\n<p><em>This article is based on general knowledge and industry best practices. No specific sources were cited as no evidence sources were available during generation.</em></p>\n"
        
        # Add references to both content versions
        if references_section:
            logger.info(f"Finalization debug - Adding References section ({len(references_section)} chars)")
            full_content += "\n" + references_section
            html_content_with_citations += "\n" + references_section
        else:
            logger.info("Finalization debug - No References section to add")
        
        # Get SEO fields from structure
        title = structure.get('title', 'Generated Article')
        meta_description = structure.get('meta_description', '')
        hook = structure.get('hook', '')
        excerpt = structure.get('excerpt', '')
        call_to_action = structure.get('call_to_action', '')
        keywords_str = ', '.join(structure.get('keywords', []))
        
        # Get word count and citations count
        word_count = content.get('word_count', 0)
        citations_count = len(citations)
        
        # Extract focus keyword first (needed for keyword-aware truncation)
        focus_keyword = _extract_focus_keyword(keywords_str)
        
        # Create SEO-optimized fields with proper length constraints
        # SEO title (for search engines) should be max 60 characters
        # Preserve focus keyword in truncated title for better SEO
        seo_title_optimized = _truncate_seo_title(title, max_length=60, focus_keyword=focus_keyword)
        metaTitle = seo_title_optimized  # Same as seo_title_optimized
        
        # Meta description should be max 160 characters
        metaDescription = _ensure_meta_description_length(meta_description, max_length=160)
        seo_meta_desc_optimized = metaDescription  # Same as metaDescription
        
        # Generate WordPress fields
        wp_slug = _generate_wp_slug(title)
        wp_tag_ids = _generate_wp_tag_ids(keywords_str)
        wp_excerpt_auto_generated = excerpt
        wp_custom_fields = {
            'article_type': structure.get('article_type', ''),
            'tone': structure.get('tone', ''),
            'target_audience': structure.get('target_audience', ''),
            'generated_at': datetime.utcnow().isoformat()
        }
        
        # Generate other SEO and content fields
        # (focus_keyword already extracted above for SEO title truncation)
        breadcrumb_title = _generate_breadcrumb_title(title)
        articleText = _extract_plain_text(full_content)
        htmlArticle = full_content
        external_links_suggested = _extract_external_links(citations)
        
        # Calculate scores
        seo_optimization_score = _calculate_seo_score(title, meta_description, word_count, citations_count)
        viral_potential_score = _calculate_viral_score(hook, excerpt, word_count)
        readability_score = _calculate_readability_score(articleText)

        # Phase 0 instrumentation: machine-readable quality diagnostics
        # Keep this non-blocking so generation resilience is preserved.
        quality_report: Dict[str, Any] = {}
        try:
            from src.services.article_quality_evaluator import build_article_quality_report

            evidence_for_quality = result.get('aggregated_evidence') or result.get('ranked_evidence') or result.get('evidence') or []
            quality_report = build_article_quality_report(
                title=title,
                html_content=htmlArticle,
                plain_text=articleText,
                citations=citations,
                sections=content.get('sections', []),
                evidence_count=len(evidence_for_quality),
            )
            logger.info(
                "Quality report generated: overall=%s humanization=%s grounding=%s geo=%s",
                quality_report.get('overall_score', 0),
                quality_report.get('humanization_score', 0),
                quality_report.get('grounding_score', 0),
                quality_report.get('geo_score', 0),
            )
        except Exception as quality_error:
            logger.warning(f"Quality report generation failed: {quality_error}")
            quality_report = {
                'version': 'phase0_v1',
                'overall_score': 0.0,
                'humanization_score': 0.0,
                'grounding_score': 0.0,
                'geo_score': 0.0,
                'diagnostics': {},
                'warnings': [f'quality_report_unavailable: {quality_error}'],
            }

        # Create engagement hooks array (include hook and potentially excerpt)
        engagement_hooks = []
        if hook:
            engagement_hooks.append(hook)
        if excerpt and excerpt != hook:
            # Add first sentence of excerpt as additional hook if different
            excerpt_first = excerpt.split('.')[0] + '.' if '.' in excerpt else excerpt
            if excerpt_first != hook and len(excerpt_first) > 20:
                engagement_hooks.append(excerpt_first)
        
        # Create final article with all required fields
        claim_bundle_summary = {
            "bundle_count": len(claim_bundles),
            "mixed_signal_count": len([b for b in claim_bundles if b.get("mixed_signal")]),
            "weak_claims_count": len([b for b in claim_bundles if b.get("confidence_score", 0) < 0.45]),
            "avg_confidence": round(
                (
                    sum(float(b.get("confidence_score", 0) or 0) for b in claim_bundles) /
                    max(len(claim_bundles), 1)
                ),
                3,
            ) if claim_bundles else 0.0,
        }
        confidence_map = _build_confidence_map(claim_bundles, full_content)
        quality_gate = _evaluate_article_quality_gates(
            research_data=research_data,
            quality_report=quality_report,
            claim_bundles=claim_bundles,
            confidence_map=confidence_map,
            citations_count=citations_count,
        )
        final_article = {
            'title': title,
            'hook': hook,
            'deck': structure.get('deck', ''),
            'excerpt': excerpt,
            'thesis': structure.get('thesis', ''),
            'meta_description': meta_description,
            'content': full_content,
            'html_content': full_content,  # For compatibility with Noodl
            'html_content_in_text_citations': html_content_with_citations,  # With clickable citations
            'citations': citations,
            'sections': content.get('sections', []),
            # SEO fields for Titles table
            'seo_title_optimized': seo_title_optimized,
            'metaTitle': metaTitle,
            'metaDescription': metaDescription,
            'seo_meta_desc_optimized': seo_meta_desc_optimized,
            'focus_keyword': focus_keyword,
            'breadcrumb_title': breadcrumb_title,
            # Content fields
            'articleText': articleText,
            'htmlArticle': htmlArticle,
            # WordPress fields
            'wp_slug': wp_slug,
            'wp_tag_ids': wp_tag_ids,
            'wp_excerpt_auto_generated': wp_excerpt_auto_generated,
            'wp_custom_fields': {**wp_custom_fields, 'deck': structure.get('deck', '')},
            # Engagement and scoring fields
            'engagement_hooks': engagement_hooks,
            'call_to_action_text': call_to_action,
            'viral_potential_score': viral_potential_score,
            'seo_optimization_score': seo_optimization_score,
            'readability_score': readability_score,
            'overall_quality_score': quality_report.get('overall_score', 0),
            'quality_report': quality_report,
            'claim_bundles': claim_bundles,
            'claim_bundle_summary': claim_bundle_summary,
            'confidence_map': confidence_map,
            'quality_gate': quality_gate,
            'generation_status': quality_gate.get('decision', 'Created'),
            'geo_optimization': {
                'answer_first_enforced': geo_enforce_answer_first,
                'answer_first_auto_inserted': geo_auto_applied,
                'passage_extractability_score': (
                    (quality_report.get('diagnostics') or {})
                    .get('passage_quality', {})
                    .get('extractability_score', 0)
                ),
                'entity_clarity_score': (
                    (quality_report.get('diagnostics') or {})
                    .get('entity_clarity', {})
                    .get('entity_clarity_score', 0)
                ),
            },
            'external_links_suggested': external_links_suggested,
            'metadata': {
                'deck': structure.get('deck', ''),
                'word_count': word_count,
                'sections': len(content.get('sections', [])),
                'citations_count': citations_count,
                'claim_bundle_count': claim_bundle_summary.get("bundle_count", 0),
                'claim_bundle_avg_confidence': claim_bundle_summary.get("avg_confidence", 0.0),
                'quality_gate_decision': quality_gate.get('decision', 'Created'),
                'geo_answer_first_auto_inserted': geo_auto_applied,
                'quality_report_version': quality_report.get('version', 'phase0_v1'),
                'generated_at': datetime.utcnow().isoformat()
            }
        }
        
        logger.info(f"Finalized article with {final_article['metadata']['word_count']} words")
        
        return {
            'article': final_article,  # Store in 'article' field for API response
            'final_article': final_article,  # Keep both for compatibility
            'stage_data': {
                'finalized': True,
                'quality_overall_score': quality_report.get('overall_score', 0),
                'quality_humanization_score': quality_report.get('humanization_score', 0),
                'quality_grounding_score': quality_report.get('grounding_score', 0),
                'quality_geo_score': quality_report.get('geo_score', 0),
                'quality_gate_decision': quality_gate.get('decision', 'Created'),
            }
        }
        
    except Exception as e:
        logger.error(f"Error in article finalization: {str(e)}")
        # Re-raise the exception to trigger task failure
        raise e

def get_task_status(task_id: str) -> Optional[Dict[str, Any]]:
    """
    Get the status of a research task.
    
    Args:
        task_id: Task ID to check
        
    Returns:
        Task status information or None if not found
    """
    try:
        # Get task result from Celery
        task_result = celery.AsyncResult(task_id)
        
        # Check if task is registered (exists)
        if not task_result.ready() and task_result.state == 'PENDING':
            # Try to refresh to see if task exists
            try:
                task_result.get(timeout=0.1)
            except Exception:
                pass
        
        # Normalize Celery states to API-friendly statuses
        state = task_result.state or TASK_STATUS['PENDING']

        # Treat Celery's STARTED/RETRY as PROGRESS to avoid "unknown" in clients
        if state in ('STARTED', 'RETRY'):
            return {
                'task_id': task_id,
                'status': TASK_STATUS['PROGRESS'],
                'progress': 0,
                'progress_percent': 0,
                'current_stage': 'STARTED',
                'message': 'Task has started',
                'info': {
                    'progress': 0,
                    'message': 'Task has started'
                }
            }

        if state == TASK_STATUS['PENDING']:
            return {
                'task_id': task_id,
                'status': TASK_STATUS['PENDING'],
                'progress': 0,
                'progress_percent': 0,
                'message': 'Task is waiting to be processed...',
                'info': {
                    'progress': 0,
                    'message': 'Task is waiting to be processed...'
                }
            }
        elif state == TASK_STATUS['PROGRESS']:
            try:
                meta = task_result.info
                if meta is None:
                    meta = {}
            except Exception:
                meta = {}
            
            progress = meta.get('progress', 0) if isinstance(meta, dict) else 0
            message = meta.get('message', 'Processing...') if isinstance(meta, dict) else 'Processing...'
            
            return {
                'task_id': task_id,
                'status': TASK_STATUS['PROGRESS'],
                'progress': progress,
                'progress_percent': progress,
                'current_stage': meta.get('current_stage', 'UNKNOWN') if isinstance(meta, dict) else 'UNKNOWN',
                'message': message,
                'info': {
                    'progress': progress,
                    'message': message
                }
            }
        elif state == TASK_STATUS['SUCCESS']:
            try:
                result = task_result.result
            except Exception as e:
                logger.error(f"Error getting task result: {str(e)}")
                result = {}
            return {
                'task_id': task_id,
                'status': TASK_STATUS['SUCCESS'],
                'progress': 100,
                'progress_percent': 100,
                'current_stage': 'COMPLETED',
                'message': 'Task completed successfully!',
                'result': result,
                'info': {
                    'progress': 100,
                    'message': 'Task completed successfully!'
                }
            }
        elif state == TASK_STATUS['FAILURE']:
            try:
                meta = task_result.info
                if meta is None:
                    meta = {}
            except Exception:
                meta = {}
            progress = meta.get('progress', 0) if isinstance(meta, dict) else 0
            message = meta.get('message', 'Task failed') if isinstance(meta, dict) else 'Task failed'
            return {
                'task_id': task_id,
                'status': TASK_STATUS['FAILURE'],
                'progress': progress,
                'progress_percent': progress,
                'current_stage': meta.get('current_stage', 'UNKNOWN') if isinstance(meta, dict) else 'UNKNOWN',
                'message': message,
                'error': meta.get('error', 'Unknown error') if isinstance(meta, dict) else str(task_result.info),
                'info': {
                    'progress': progress,
                    'message': message
                }
            }
        else:
            return {
                'task_id': task_id,
                'status': state,
                'progress': 0,
                'progress_percent': 0,
                'message': f'Unknown task state: {state}',
                'info': {
                    'progress': 0,
                    'message': f'Unknown task state: {state}'
                }
            }
            
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Error getting task status for {task_id}: {error_msg}", exc_info=True)
        # If Celery result metadata is corrupted, surface a failure to the UI instead
        # of masking it as PENDING forever.
        if "exception type" in error_msg.lower():
            return {
                'task_id': task_id,
                'status': TASK_STATUS['FAILURE'],
                'progress': 0,
                'progress_percent': 0,
                'current_stage': 'UNKNOWN',
                'message': 'Task failed while persisting result metadata.',
                'error': error_msg,
                'info': {
                    'progress': 0,
                    'message': 'Task failed while persisting result metadata.'
                }
            }
        # For transient lookup issues, keep PENDING fallback behavior.
        return {
            'task_id': task_id,
            'status': TASK_STATUS['PENDING'],
            'progress': 0,
            'progress_percent': 0,
            'message': 'Task is waiting to be processed...',
            'info': {
                'progress': 0,
                'message': 'Task is waiting to be processed...'
            }
        }

@celery.task(name='content_generator_v2.tasks.research.cancel_task')
def cancel_task(task_id: str) -> bool:
    """
    Cancel a running research task.
    
    Args:
        task_id: Task ID to cancel
        
    Returns:
        True if cancellation was successful, False otherwise
    """
    try:
        # Revoke the task
        celery.control.revoke(task_id, terminate=True)
        logger.info(f"Task {task_id} cancelled successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error cancelling task {task_id}: {str(e)}")
        return False

# -----------------------------------------------------------------------------
# Trend Analysis Task
# -----------------------------------------------------------------------------

@celery.task(bind=True, name='content_generator_v2.tasks.trends.process_trend_task')
def process_trend_task(self, site_id: str, primary_category_id: Optional[str] = None, secondary_category_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Process trend analysis for a specific site.

    Args:
        site_id: ID of the site/project
        primary_category_id: Optional ID of the selected primary category
        secondary_category_id: Optional ID of the selected secondary category

    Returns:
        Dictionary containing the trend report
    """
    logger.info(
        "trend_task: start site_id=%s primary_category_id=%s secondary_category_id=%s task_id=%s",
        site_id,
        primary_category_id,
        secondary_category_id,
        getattr(self.request, "id", None),
    )

    try:
        _ensure_project_root_on_path()

        # Import here to avoid circular dependencies if any
        from src.services.trend_engine import TrendEngine

        # Initialize engine
        engine = TrendEngine()

        # Run async method in synchronous Celery task
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            # Execute the async method
            result = loop.run_until_complete(
                engine.get_whats_trending(
                    site_id,
                    primary_category_id=primary_category_id,
                    secondary_category_id=secondary_category_id,
                )
            )

            try:
                topics = (((result or {}).get("report_content") or {}).get("topics") or [])
                logger.info("trend_task: synthesized_topics_count=%s", len(topics) if isinstance(topics, list) else "non_list")
            except Exception:
                logger.warning("trend_task: unable to extract topics for logging")

            # Fail fast if we didn't actually produce usable topics.
            report_content = (result or {}).get("report_content") or {}
            topics = report_content.get("topics") if isinstance(report_content, dict) else None
            has_titles = isinstance(topics, list) and any(
                isinstance(t, dict) and str(t.get("title") or "").strip()
                for t in topics
            )
            if not has_titles:
                error_msg = "Trend synthesis produced no usable topics"
                logger.error(
                    "trend_task: %s site_id=%s primary_category_id=%s secondary_category_id=%s report_content_keys=%s",
                    error_msg,
                    site_id,
                    primary_category_id,
                    secondary_category_id,
                    list(report_content.keys()) if isinstance(report_content, dict) else None,
                )
                return {
                    'status': 'FAILURE',
                    'site_id': site_id,
                    'error': error_msg,
                    'result': result,
                    'failed_at': datetime.utcnow().isoformat(),
                }

            logger.info(f"Trend analysis completed successfully for site_id: {site_id}")
            return {
                'status': 'SUCCESS',
                'site_id': site_id,
                'result': result,
                'completed_at': datetime.utcnow().isoformat()
            }

        except Exception as async_error:
            logger.error(f"Async execution failed: {async_error}", exc_info=True)
            raise async_error
        finally:
            loop.close()

    except Exception as e:
        logger.error(f"Trend task failed: {str(e)}", exc_info=True)
        return {
            'status': 'FAILURE',
            'site_id': site_id,
            'error': str(e),
            'failed_at': datetime.utcnow().isoformat()
        }
