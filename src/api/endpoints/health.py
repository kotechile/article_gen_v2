"""
Health check endpoints for Content Generator V2.

This module provides health check and monitoring endpoints
for the system.
"""

import logging
import psutil
from datetime import datetime
from flask import Blueprint, jsonify, current_app

from ...core.models.errors import ErrorResponse
from ...utils.health import HealthChecker


logger = logging.getLogger(__name__)

# Create blueprint
health_bp = Blueprint('health', __name__, url_prefix='/api/v1')


@health_bp.route('/health', methods=['GET'])
def health_check():
    """
    Basic health check endpoint.
    
    Returns:
        System health status
    """
    try:
        return jsonify({
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "version": "2.0.0",
            "service": "content-generator-v2"
        }), 200
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return jsonify(ErrorResponse(
            error="health_check_failed",
            message="Health check failed",
            status=500
        ).dict()), 500


@health_bp.route('/health/detailed', methods=['GET'])
def detailed_health_check():
    """
    Detailed health check endpoint.
    
    Returns:
        Detailed system health information
    """
    try:
        health_checker = HealthChecker()
        health_status = health_checker.get_detailed_status()
        
        # Determine overall status
        overall_status = "healthy"
        if health_status["database"]["status"] != "healthy":
            overall_status = "degraded"
        if health_status["celery"]["status"] != "healthy":
            overall_status = "degraded"
        if health_status["redis"]["status"] != "healthy":
            overall_status = "unhealthy"
        
        return jsonify({
            "status": overall_status,
            "timestamp": datetime.utcnow().isoformat(),
            "version": "2.0.0",
            "service": "content-generator-v2",
            "components": health_status
        }), 200 if overall_status in ["healthy", "degraded"] else 503
        
    except Exception as e:
        logger.error(f"Detailed health check failed: {str(e)}")
        return jsonify(ErrorResponse(
            error="detailed_health_check_failed",
            message="Detailed health check failed",
            status=500
        ).dict()), 500


@health_bp.route('/health/ready', methods=['GET'])
def readiness_check():
    """
    Readiness check endpoint for Kubernetes.
    
    Returns:
        Readiness status
    """
    try:
        health_checker = HealthChecker()
        
        # Check critical components
        redis_status = health_checker.check_redis()
        celery_status = health_checker.check_celery()
        
        if redis_status["status"] == "healthy" and celery_status["status"] == "healthy":
            return jsonify({
                "status": "ready",
                "timestamp": datetime.utcnow().isoformat()
            }), 200
        else:
            return jsonify({
                "status": "not_ready",
                "timestamp": datetime.utcnow().isoformat(),
                "issues": [
                    issue for issue in [redis_status, celery_status]
                    if issue["status"] != "healthy"
                ]
            }), 503
            
    except Exception as e:
        logger.error(f"Readiness check failed: {str(e)}")
        return jsonify(ErrorResponse(
            error="readiness_check_failed",
            message="Readiness check failed",
            status=500
        ).dict()), 500


@health_bp.route('/health/live', methods=['GET'])
def liveness_check():
    """
    Liveness check endpoint for Kubernetes.
    
    Returns:
        Liveness status
    """
    try:
        # Basic liveness check - just verify the service is running
        return jsonify({
            "status": "alive",
            "timestamp": datetime.utcnow().isoformat(),
            "uptime": psutil.Process().create_time()
        }), 200
        
    except Exception as e:
        logger.error(f"Liveness check failed: {str(e)}")
        return jsonify(ErrorResponse(
            error="liveness_check_failed",
            message="Liveness check failed",
            status=500
        ).dict()), 500


@health_bp.route('/metrics', methods=['GET'])
def get_metrics():
    """
    Get system metrics.
    
    Returns:
        System metrics and statistics
    """
    try:
        health_checker = HealthChecker()
        metrics = health_checker.get_metrics()
        
        return jsonify({
            "timestamp": datetime.utcnow().isoformat(),
            "metrics": metrics
        }), 200
        
    except Exception as e:
        logger.error(f"Metrics collection failed: {str(e)}")
        return jsonify(ErrorResponse(
            error="metrics_collection_failed",
            message="Failed to collect metrics",
            status=500
        ).dict()), 500


@health_bp.route('/status', methods=['GET'])
def get_status():
    """
    Get system status information.
    
    Returns:
        System status and configuration
    """
    try:
        return jsonify({
            "service": "content-generator-v2",
            "version": "2.0.0",
            "status": "running",
            "timestamp": datetime.utcnow().isoformat(),
            "environment": current_app.config.get("ENV", "development"),
            "debug": current_app.config.get("DEBUG", False),
            "endpoints": {
                "health": "/api/v1/health",
                "detailed_health": "/api/v1/health/detailed",
                "readiness": "/api/v1/health/ready",
                "liveness": "/api/v1/health/live",
                "metrics": "/api/v1/metrics",
                "research": "/api/v1/research"
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Status check failed: {str(e)}")
        return jsonify(ErrorResponse(
            error="status_check_failed",
            message="Status check failed",
            status=500
        ).dict()), 500
