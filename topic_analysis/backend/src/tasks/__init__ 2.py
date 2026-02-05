"""
Background Tasks Module
"""

from .affiliate_tasks import (
    search_affiliate_programs_task,
    update_affiliate_programs,
    analyze_affiliate_program_performance
)

from .trend_tasks import (
    analyze_trends_task,
    update_trend_data,
    compare_trends_task,
    generate_trend_insights_task
)

__all__ = [
    # Affiliate tasks
    "search_affiliate_programs_task",
    "update_affiliate_programs",
    "analyze_affiliate_program_performance",
    
    # Trend tasks
    "analyze_trends_task",
    "update_trend_data",
    "compare_trends_task",
    "generate_trend_insights_task"
]
