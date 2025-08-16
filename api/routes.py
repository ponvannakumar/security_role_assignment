# api/routes.py
from flask import request, jsonify
from . import api_bp
from .utils import validate_input, format_response
import logging

logger = logging.getLogger(__name__)

@api_bp.route('/predict', methods=['POST'])
def api_predict():
    """API endpoint for role prediction"""
    try:
        data = request.get_json()
        
        if not data or 'finding' not in data:
            return jsonify({'error': 'Finding text is required'}), 400
        
        finding = data['finding'].strip()
        
        if not validate_input(finding):
            return jsonify({'error': 'Invalid input format'}), 400
        
        # Import predictor here to avoid circular imports
        from app import predictor
        result = predictor.predict_role(finding)
        
        if 'error' in result:
            return jsonify(result), 500
        
        return jsonify(format_response(result))
        
    except Exception as e:
        logger.error(f"API prediction error: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@api_bp.route('/retrain', methods=['POST'])
def api_retrain():
    """API endpoint for model retraining"""
    try:
        from app import predictor
        success = predictor.train_model()
        
        if success:
            return jsonify({'message': 'Model retrained successfully'})
        else:
            return jsonify({'error': 'Model retraining failed'}), 500
            
    except Exception as e:
        logger.error(f"API retrain error: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@api_bp.route('/health', methods=['GET'])
def api_health():
    """API health check endpoint"""
    try:
        from app import predictor
        return jsonify({
            'status': 'healthy',
            'model_trained': predictor.is_trained,
            'version': '1.0'
        })
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return jsonify({'status': 'unhealthy', 'error': str(e)}), 500

@api_bp.route('/roles', methods=['GET'])
def api_roles():
    """Get list of supported roles"""
    try:
        from app import predictor
        return jsonify({
            'roles': predictor.roles,
            'count': len(predictor.roles)
        })
    except Exception as e:
        logger.error(f"Roles endpoint error: {e}")
        return jsonify({'error': 'Internal server error'}), 500