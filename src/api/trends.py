
from flask import Blueprint, jsonify, request
from tasks import process_trend_task, get_task_status

trends_bp = Blueprint('trends', __name__)

@trends_bp.route('/api/v1/trends/<site_id>', methods=['POST'])
def generate_trend_report_endpoint(site_id):
    """
    Trigger trend report generation for a site.
    """
    try:
        # Submit task
        task = process_trend_task.delay(site_id)
        
        return jsonify({
            'message': 'Trend analysis started',
            'task_id': task.id,
            'status': 'pending',
            'site_id': site_id
        }), 202
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'message': 'Failed to start trend analysis'
        }), 500

@trends_bp.route('/api/v1/trends/task/<task_id>', methods=['GET'])
def get_trend_status(task_id):
    """
    Get status of trend analysis task.
    """
    try:
        status = get_task_status(task_id)
        if status:
            return jsonify(status)
        return jsonify({'error': 'Task not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500
