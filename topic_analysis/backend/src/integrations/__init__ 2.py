"""
External API Integrations Module
Provides integrations with various external services for the TrendTap platform
"""

from .google_trends import (
    google_trends_api,
    get_trend_data,
    get_related_queries,
    get_interest_over_time
)

from .affiliate_networks import (
    affiliate_networks_manager,
    search_affiliate_programs,
    get_program_details,
    get_available_networks,
    get_network_stats
)

from .dataforseo import (
    dataforseo_api,
    get_keyword_ideas,
    get_keyword_metrics,
    get_serp_analysis,
    get_related_keywords,
    get_keyword_difficulty
)

from .social_media import (
    social_media_manager,
    search_social_media,
    get_trending_topics,
    get_available_platforms,
    get_platform_stats
)

from .llm_providers import (
    llm_providers_manager,
    generate_content,
    analyze_trends,
    generate_headlines,
    get_available_providers,
    test_all_providers
)

from .surfer_seo import (
    surfer_seo_api,
    analyze_content,
    get_keyword_suggestions,
    get_content_planner,
    get_serp_analyzer,
    get_audit_report
)

from .frase import (
    frase_api,
    get_topic_research,
    get_content_optimization,
    get_question_research,
    get_content_outline,
    get_competitor_analysis
)

from .coschedule import (
    coschedule_api,
    analyze_headline,
    generate_headlines,
    get_calendar_events,
    create_calendar_event,
    get_content_ideas,
    get_team_performance
)

from .export_platforms import (
    export_platforms_manager,
    export_content,
    export_to_multiple_platforms,
    get_export_status,
    get_available_platforms,
    test_all_platforms
)

__all__ = [
    # Google Trends
    "google_trends_api",
    "get_trend_data",
    "get_related_queries", 
    "get_interest_over_time",
    
    # Affiliate Networks
    "affiliate_networks_manager",
    "search_affiliate_programs",
    "get_program_details",
    "get_available_networks",
    "get_network_stats",
    
    # DataForSEO
    "dataforseo_api",
    "get_keyword_ideas",
    "get_keyword_metrics",
    "get_serp_analysis",
    "get_related_keywords",
    "get_keyword_difficulty",
    
    # Social Media
    "social_media_manager",
    "search_social_media",
    "get_trending_topics",
    "get_available_platforms",
    "get_platform_stats",
    
    # LLM Providers
    "llm_providers_manager",
    "generate_content",
    "analyze_trends",
    "generate_headlines",
    "get_available_providers",
    "test_all_providers",
    
    # SurferSEO
    "surfer_seo_api",
    "analyze_content",
    "get_keyword_suggestions",
    "get_content_planner",
    "get_serp_analyzer",
    "get_audit_report",
    
    # Frase
    "frase_api",
    "get_topic_research",
    "get_content_optimization",
    "get_question_research",
    "get_content_outline",
    "get_competitor_analysis",
    
    # CoSchedule
    "coschedule_api",
    "analyze_headline",
    "generate_headlines",
    "get_calendar_events",
    "create_calendar_event",
    "get_content_ideas",
    "get_team_performance",
    
    # Export Platforms
    "export_platforms_manager",
    "export_content",
    "export_to_multiple_platforms",
    "get_export_status",
    "get_available_platforms",
    "test_all_platforms"
]
