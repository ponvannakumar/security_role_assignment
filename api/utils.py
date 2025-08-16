# api/utils.py
import re
from typing import Dict, Any, List
from datetime import datetime

def validate_input(finding: str) -> bool:
    """Validate input finding text"""
    if not finding or not isinstance(finding, str):
        return False
    
    if len(finding.strip()) < 10:
        return False
    
    if len(finding) > 5000:  # Max length check
        return False
    
    # Check for malicious patterns
    malicious_patterns = [
        r'<script',
        r'javascript:',
        r'eval\(',
        r'exec\(',
    ]
    
    for pattern in malicious_patterns:
        if re.search(pattern, finding.lower()):
            return False
    
    return True

def format_response(result: Dict[str, Any]) -> Dict[str, Any]:
    """Format prediction response for API"""
    formatted = {
        'predicted_role': result.get('predicted_role', ''),
        'confidence_scores': result.get('confidence_scores', {}),
        'keywords_extracted': result.get('keywords_extracted', []),
        'explanation': result.get('explanation', ''),
        'timestamp': datetime.now().isoformat(),
        'top_3_roles': result.get('top_3_roles', [])
    }
    
    return formatted

def sanitize_input(text: str) -> str:
    """Sanitize input text"""
    if not text:
        return ""
    
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Trim
    text = text.strip()
    
    return text

def calculate_confidence_level(scores: Dict[str, float]) -> str:
    """Calculate overall confidence level"""
    if not scores:
        return "low"
    
    max_score = max(scores.values())
    
    if max_score >= 0.8:
        return "high"
    elif max_score >= 0.6:
        return "medium"
    else:
        return "low"

def extract_security_indicators(text: str) -> List[str]:
    """Extract security-related indicators from text"""
    indicators = []
    
    # Define security patterns
    patterns = {
        'aws_services': r'\b(s3|ec2|iam|vpc|rds|lambda)\b',
        'vulnerabilities': r'\b(cve-\d{4}-\d+|vulnerability|exploit)\b',
        'access_terms': r'\b(permission|privilege|access|auth)\b',
        'network_terms': r'\b(firewall|port|protocol|ssl|tls)\b'
    }
    
    for category, pattern in patterns.items():
        matches = re.findall(pattern, text.lower())
        if matches:
            indicators.extend([f"{category}:{match}" for match in matches])
    
    return indicators

def log_prediction(finding: str, result: Dict[str, Any]):
    """Log prediction for monitoring"""
    import logging
    
    logger = logging.getLogger(__name__)
    
    log_data = {
        'finding_length': len(finding),
        'predicted_role': result.get('predicted_role', ''),
        'max_confidence': max(result.get('confidence_scores', {}).values()) if result.get('confidence_scores') else 0,
        'timestamp': datetime.now().isoformat()
    }
    
    logger.info(f"Prediction logged: {log_data}")

def rate_limit_key(request) -> str:
    """Generate rate limit key for request"""
    # Use IP address as default key
    return request.remote_addr or 'unknown'