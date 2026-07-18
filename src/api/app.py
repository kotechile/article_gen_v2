"""
Main Flask application for Content Generator V2.

This module creates and configures the Flask application
with all necessary middleware, blueprints, and error handlers.
"""

import logging
import os
from datetime import datetime
from flask import Flask, jsonify, request
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

from .endpoints import research_bp, health_bp
from .middleware.auth import AuthMiddleware
from .middleware.logging import LoggingMiddleware
from .middleware.error_handler import ErrorHandler
from ..core.models.errors import ErrorResponse
from ..utils.config import get_config
from ..utils.logging import setup_logging


def create_app(config_name: str = None) -> Flask:
    """
    Create and configure Flask application.
    
    Args:
        config_name: Configuration name (development, production, testing)
        
    Returns:
        Configured Flask application
    """
    # Create Flask app
    app = Flask(__name__)
    
    # Load configuration
    config = get_config(config_name)
    app.config.from_object(config)
    
    # Setup logging
    setup_logging(app.config)
    
    # Initialize extensions
    CORS(app, origins=app.config.get('CORS_ORIGINS', ['*']))
    
    # Initialize rate limiter (compatible with Flask-Limiter v3+)
    limiter = Limiter(
        key_func=get_remote_address,
        default_limits=[],  # Use per-endpoint limits to avoid parsing issues
        storage_uri=app.config.get('RATELIMIT_STORAGE_URL', 'memory://')
    )
    limiter.init_app(app)
    
    # Register middleware
    app.before_request(AuthMiddleware.before_request)
    app.after_request(LoggingMiddleware.after_request)
    
    # Register endpoints
    from .endpoints.research import research_bp
    from .endpoints.health import health_bp
    from .endpoints.images import images_bp
    from .endpoints.settings import settings_bp
    from .endpoints.research_topics import research_topics_bp
    from .endpoints.content_ideas import content_ideas_bp
    from .endpoints.research_tools import research_tools_bp
    from .endpoints.research_pipeline import research_pipeline_bp
    from .endpoints.ai import ai_bp
    from .wordpress import wordpress_bp
    from .internal_links import internal_links_bp
    from .trends import trends_bp
    from .screen_capture import screen_capture_bp
    
    # Register blueprints
    app.register_blueprint(research_bp)
    app.register_blueprint(health_bp)
    app.register_blueprint(images_bp)
    app.register_blueprint(settings_bp)
    app.register_blueprint(research_topics_bp)
    app.register_blueprint(content_ideas_bp)
    app.register_blueprint(research_tools_bp)
    app.register_blueprint(research_pipeline_bp)
    app.register_blueprint(wordpress_bp)
    app.register_blueprint(internal_links_bp)
    app.register_blueprint(ai_bp)
    app.register_blueprint(trends_bp)
    app.register_blueprint(screen_capture_bp)
    
    # Register error handlers
    ErrorHandler.register_handlers(app)
    
    # Add custom error handlers
    @app.errorhandler(404)
    def not_found(error):
        return jsonify(ErrorResponse(
            error="not_found",
            message="The requested resource was not found",
            error_code="NOT_FOUND",
            status=404
        ).dict()), 404
    
    @app.errorhandler(405)
    def method_not_allowed(error):
        return jsonify(ErrorResponse(
            error="method_not_allowed",
            message="The method is not allowed for the requested URL",
            error_code="METHOD_NOT_ALLOWED",
            status=405
        ).dict()), 405
    
    @app.errorhandler(500)
    def internal_error(error):
        return jsonify(ErrorResponse(
            error="internal_server_error",
            message="An internal server error occurred",
            error_code="INTERNAL_SERVER_ERROR",
            status=500
        ).dict()), 500
    
    # Add request logging
    @app.before_request
    def log_request():
        if app.config.get('LOG_REQUESTS', True):
            logger = logging.getLogger(__name__)
            logger.info(f"Request: {request.method} {request.path} from {request.remote_addr}")
    
    # Add response logging
    @app.after_request
    def log_response(response):
        if app.config.get('LOG_REQUESTS', True):
            logger = logging.getLogger(__name__)
            logger.info(f"Response: {response.status_code} for {request.method} {request.path}")
        return response
    
    # Health check endpoint
    @app.route('/')
    def root():
        return jsonify({
            "service": "content-generator-v2",
            "version": "2.0.0",
            "status": "running",
            "timestamp": datetime.utcnow().isoformat(),
            "endpoints": {
                "health": "/api/v1/health",
                "research": "/api/v1/research",
                "docs": "/api/v1/docs"
            }
        })

    # Video Generation endpoints
    @app.route('/api/v1/video/blueprint', methods=['POST'])
    def generate_video_blueprint_api():
        """Scrape an article or parse a custom script and generate the video blueprint JSON."""
        import subprocess
        import json
        import os
        import uuid
        temp_script_path = None
        try:
            data = request.get_json() or {}
            url = data.get('url')
            script_text = data.get('script_text')
            
            if not url and not script_text:
                return jsonify({'error': 'missing_parameter', 'message': 'Either url or script_text is required'}), 400
                
            primary = data.get('primary_color')
            secondary = data.get('secondary_color')
            background = data.get('background_color')
            
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # root dir
            script_path = os.path.join(base_dir, "generate_video.py")
            
            cmd = ["python3", script_path]
            
            if script_text:
                temp_filename = f"temp_script_{uuid.uuid4().hex}.txt"
                temp_script_path = os.path.join(base_dir, "_remotion", temp_filename)
                with open(temp_script_path, 'w', encoding='utf-8') as f:
                    f.write(script_text)
                cmd += ["--script-file", temp_script_path]
            else:
                cmd += [url]
                
            cmd += ["--blueprint-only"]
            
            if primary:
                cmd += ["--primary", primary]
            if secondary:
                cmd += ["--secondary", secondary]
            if background:
                cmd += ["--background", background]
                
            logger = logging.getLogger(__name__)
            logger.info(f"Generating video blueprint: {' '.join(cmd)}")
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            
            if result.returncode != 0:
                logger.error(f"Blueprint generation failed.\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}")
                return jsonify({
                    'status': 'error',
                    'message': 'Blueprint generation failed',
                    'stderr': result.stderr,
                    'stdout': result.stdout
                }), 500
                
            # Parse JSON blueprint from script output
            stdout = result.stdout
            start_marker = "=== BLUEPRINT_JSON_START ==="
            end_marker = "=== BLUEPRINT_JSON_END ==="
            if start_marker in stdout and end_marker in stdout:
                json_str = stdout.split(start_marker)[1].split(end_marker)[0].strip()
                blueprint = json.loads(json_str)
                return jsonify({
                    'status': 'success',
                    'blueprint': blueprint
                })
            else:
                return jsonify({
                    'status': 'error',
                    'message': 'Could not parse blueprint from output',
                    'stdout': stdout
                }), 500
                
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.error(f"Error during blueprint generation: {str(e)}", exc_info=True)
            return jsonify({
                'status': 'error',
                'message': str(e)
            }), 500
        finally:
            if temp_script_path and os.path.exists(temp_script_path):
                try:
                    os.remove(temp_script_path)
                except Exception:
                    pass

    def cleanup_old_uploads():
        """Delete uploaded files older than 24 hours to prevent VPS disk clutter."""
        import time
        try:
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            uploads_dir = os.path.join(base_dir, "_remotion", "public", "uploads")
            if not os.path.exists(uploads_dir):
                return
                
            now = time.time()
            cutoff = now - (24 * 3600)  # 24 hours in seconds
            
            for filename in os.listdir(uploads_dir):
                file_path = os.path.join(uploads_dir, filename)
                if os.path.isfile(file_path):
                    # Robust check: parse timestamp from filename: custom_{timestamp}_{uuid}.ext
                    parts = filename.split('_')
                    file_time = None
                    if len(parts) >= 3 and parts[0] == 'custom':
                        try:
                            file_time = int(parts[1])
                        except ValueError:
                            pass
                    
                    if file_time is None:
                        # Fallback to filesystem mtime
                        file_time = os.path.getmtime(file_path)
                        
                    if file_time < cutoff:
                        try:
                            os.remove(file_path)
                            app.logger.info(f"🗑️ Cleaned up old upload file: {filename}")
                        except Exception as e:
                            app.logger.warning(f"Failed to delete old upload file {filename}: {e}")
        except Exception as e:
            app.logger.error(f"Error during upload cleanup: {e}")

    @app.route('/api/v1/video/upload', methods=['POST'])
    def upload_video_asset_api():
        """Upload custom image files for video scenes."""
        import uuid
        import time
        import threading
        from werkzeug.utils import secure_filename
        try:
            if 'file' not in request.files:
                return jsonify({'error': 'no_file', 'message': 'No file part in the request'}), 400
            file = request.files['file']
            if file.filename == '':
                return jsonify({'error': 'empty_file', 'message': 'No file selected for uploading'}), 400
                
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # root dir
            uploads_dir = os.path.join(base_dir, "_remotion", "public", "uploads")
            os.makedirs(uploads_dir, exist_ok=True)
            
            # Generate a secure unique name containing upload timestamp to avoid clock drift issues
            ext = os.path.splitext(secure_filename(file.filename))[1] or ".jpg"
            unique_filename = f"custom_{int(time.time())}_{uuid.uuid4().hex}{ext}"
            file_path = os.path.join(uploads_dir, unique_filename)
            file.save(file_path)
            
            # If it's a video file that is NOT mp4 (e.g. mov, webm, m4v), transcode it to standard h264 mp4
            import subprocess
            if ext.lower() in ['.mov', '.webm', '.m4v', '.qt']:
                mp4_filename = f"custom_{int(time.time())}_{uuid.uuid4().hex}.mp4"
                mp4_path = os.path.join(uploads_dir, mp4_filename)
                
                # Transcode command (H.264 video, AAC audio, YUV420p pixel format for maximum browser compatibility)
                transcode_cmd = [
                    "ffmpeg", "-i", file_path,
                    "-vcodec", "libx264", "-acodec", "aac",
                    "-pix_fmt", "yuv420p", "-y", mp4_path
                ]
                
                transcode_result = subprocess.run(transcode_cmd, capture_output=True, text=True)
                
                if transcode_result.returncode == 0:
                    # Successfully transcoded, delete original non-mp4 file
                    try:
                        os.remove(file_path)
                    except Exception:
                        pass
                    unique_filename = mp4_filename
                else:
                    # Log error if transcoding fails
                    logger = logging.getLogger(__name__)
                    logger.error(f"FFmpeg transcoding failed for {unique_filename}: {transcode_result.stderr}")
            
            # Trigger background cleanup thread asynchronously
            threading.Thread(target=cleanup_old_uploads, daemon=True).start()
            
            # Return relative path for mockPayload.json and absolute static url for Lambda
            relative_url = f"uploads/{unique_filename}"
            
            # Dynamic host url resolution
            host_url = request.host_url.rstrip('/')
            if "localhost" not in host_url and "127.0.0.1" not in host_url:
                host_url = host_url.replace("http://", "https://")
            
            full_url = f"{host_url}/api/v1/video/static/{relative_url}"
            
            return jsonify({
                'status': 'success',
                'relative_path': relative_url,
                'url': full_url
            })
            
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.error(f"Error during asset upload: {str(e)}", exc_info=True)
            return jsonify({
                'status': 'error',
                'message': str(e)
            }), 500

    @app.route('/api/v1/generate-video', methods=['POST'])
    def generate_video_api():
        """Endpoint to trigger video generation from article URL or direct blueprint."""
        import subprocess
        import json
        import uuid
        try:
            data = request.get_json() or {}
            url = data.get('url')
            blueprint_payload = data.get('blueprint_payload')
            
            if not url and not blueprint_payload:
                return jsonify({'error': 'missing_parameter', 'message': 'url or blueprint_payload is required'}), 400
                
            voice = data.get('voice', 'onyx')
            provider = data.get('provider', 'openai')
            caption_position = data.get('caption_position', 'center')
            primary = data.get('primary_color')
            secondary = data.get('secondary_color')
            background = data.get('background_color')
            aspect_ratio = data.get('aspect_ratio', 'vertical')
            music = data.get('music', 'background.mp3')
            concurrency = data.get('concurrency', 4)
            
            # Resolve host_url from request dynamically
            host_url = request.host_url.rstrip('/')
            if "localhost" not in host_url and "127.0.0.1" not in host_url:
                host_url = host_url.replace("http://", "https://")
            
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # root dir
            script_path = os.path.join(base_dir, "generate_video.py")
            
            temp_payload_path = None
            
            # If a custom blueprint is provided, write it to a temporary file
            if blueprint_payload:
                temp_filename = f"temp_blueprint_{uuid.uuid4().hex}.json"
                temp_payload_path = os.path.join(base_dir, "_remotion", temp_filename)
                with open(temp_payload_path, 'w') as f:
                    json.dump(blueprint_payload, f, indent=2)
            
            # Build subprocess command
            cmd = [
                "python3",
                script_path,
                url or "",
                "--voice", voice,
                "--provider", provider,
                "--caption-position", caption_position,
                "--aspect-ratio", aspect_ratio,
                "--host-url", host_url,
                "--render-on-lambda",
                "--concurrency", str(concurrency)
            ]
            
            if temp_payload_path:
                cmd += ["--blueprint-payload", temp_payload_path]
            else:
                if primary:
                    cmd += ["--primary", primary]
                if secondary:
                    cmd += ["--secondary", secondary]
                if background:
                    cmd += ["--background", background]
            
            if music:
                cmd += ["--music", music]
                
            logger = logging.getLogger(__name__)
            logger.info(f"Triggering video generation: {' '.join(cmd)}")
            
            # Run the script synchronously with a timeout of 600s
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            
            # Clean up the temp blueprint file if created
            if temp_payload_path and os.path.exists(temp_payload_path):
                try:
                    os.remove(temp_payload_path)
                except Exception as ex:
                    logger.warning(f"Failed to remove temp blueprint file: {ex}")
            
            if result.returncode != 0:
                logger.error(f"Video generation script failed.\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}")
                return jsonify({
                    'status': 'error',
                    'message': 'Video generation failed',
                    'stderr': result.stderr,
                    'stdout': result.stdout
                }), 500
                
            return jsonify({
                'status': 'success',
                'message': 'Video generated successfully',
                'video_url': '/api/v1/video/download'
            })
            
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.error(f"Error during video generation: {str(e)}", exc_info=True)
            return jsonify({
                'status': 'error',
                'message': str(e)
            }), 500

    @app.route('/api/v1/video/download', methods=['GET'])
    def download_video():
        """Endpoint to download the generated video."""
        from flask import send_file
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # root dir
        video_path = os.path.join(base_dir, "_remotion", "output-generated.mp4")
        if os.path.exists(video_path):
            return send_file(video_path, mimetype='video/mp4', as_attachment=True, download_name='output-generated.mp4')
        return jsonify({'error': 'file_not_found', 'message': 'Generated video file not found'}), 404

    @app.route('/api/v1/video/static/<path:filename>', methods=['GET'])
    def serve_video_static(filename):
        """Endpoint to serve video assets (images/audio) to Remotion Lambda."""
        from flask import send_from_directory
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # root dir
        public_dir = os.path.join(base_dir, "_remotion", "public")
        return send_from_directory(public_dir, filename)
    
    # API documentation endpoint
    @app.route('/api/v1/docs')
    def api_docs():
        return jsonify({
            "title": "Content Generator V2 API",
            "version": "2.0.0",
            "description": "A clean and reliable content generation system",
            "endpoints": {
                "research": {
                    "create": {
                        "method": "POST",
                        "path": "/api/v1/research",
                        "description": "Create a new research task"
                    },
                    "status": {
                        "method": "GET",
                        "path": "/api/v1/research/{task_id}",
                        "description": "Get task status and progress"
                    },
                    "result": {
                        "method": "GET",
                        "path": "/api/v1/research/{task_id}/result",
                        "description": "Get completed task result"
                    },
                    "cancel": {
                        "method": "POST",
                        "path": "/api/v1/research/{task_id}/cancel",
                        "description": "Cancel a running task"
                    }
                },
                "health": {
                    "basic": {
                        "method": "GET",
                        "path": "/api/v1/health",
                        "description": "Basic health check"
                    },
                    "detailed": {
                        "method": "GET",
                        "path": "/api/v1/health/detailed",
                        "description": "Detailed health check"
                    },
                    "ready": {
                        "method": "GET",
                        "path": "/api/v1/health/ready",
                        "description": "Readiness check"
                    },
                    "live": {
                        "method": "GET",
                        "path": "/api/v1/health/live",
                        "description": "Liveness check"
                    }
                }
            },
            "authentication": {
                "type": "API Key",
                "header": "X-API-Key",
                "description": "API key authentication required for all endpoints except health checks"
            },
            "rate_limiting": {
                "default": "1000 requests per hour",
                "research_creation": "10 requests per minute",
                "status_checking": "1000 requests per hour"
            }
        })
    
    logger = logging.getLogger(__name__)
    logger.info(f"Flask application created with config: {config_name}")
    
    # Perform startup cleanup
    cleanup_stuck_tasks()
    
    return app


def cleanup_stuck_tasks():
    """
    Reset any tasks stuck in 'Generating' state to 'Error' on startup.
    This prevents the UI from showing stuck progress bars for killed tasks.
    """
    logger = logging.getLogger(__name__)
    try:
        # Import here to avoid circular dependencies and ensure env is loaded
        import sys
        import os
        
        # Add root directory to path if needed to find supabase_client
        current_dir = os.path.dirname(os.path.abspath(__file__))
        root_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
        if root_dir not in sys.path:
            sys.path.append(root_dir)
            
        from supabase_client import get_supabase_client
        
        supabase = get_supabase_client()
        if not supabase:
            logger.warning("Startup cleanup: Failed to initialize Supabase client")
            return

        # Find stuck tasks
        response = supabase.table('Titles').select('id').eq('status', 'Generating').execute()
        stuck_tasks = response.data
        
        if stuck_tasks:
            count = len(stuck_tasks)
            logger.warning(f"Startup cleanup: Found {count} stuck 'Generating' tasks. Resetting to 'Error'...")
            
            for task in stuck_tasks:
                article_id = task['id']
                supabase.table('Titles').update({
                    'status': 'Error', 
                    'error_message': 'Generation interrupted by server restart'
                }).eq('id', article_id).execute()
                
            logger.info(f"Startup cleanup: Successfully reset {count} tasks.")
        else:
            logger.info("Startup cleanup: No stuck tasks found.")
            
    except Exception as e:
        logger.error(f"Startup cleanup failed: {str(e)}")


def run_app(host: str = '0.0.0.0', port: int = 5001, debug: bool = False):
    """
    Run the Flask application.
    
    Args:
        host: Host to bind to
        port: Port to bind to
        debug: Enable debug mode
    """
    app = create_app()
    
    logger = logging.getLogger(__name__)
    logger.info(f"Starting Content Generator V2 on {host}:{port}")
    
    app.run(host=host, port=port, debug=debug)


if __name__ == '__main__':
    run_app()
