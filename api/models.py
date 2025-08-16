# api/models.py
from dataclasses import dataclass
from typing import List, Dict, Optional
from datetime import datetime

@dataclass
class PredictionRequest:
    """Data model for prediction requests"""
    finding: str
    include_explanation: bool = True
    confidence_threshold: float = 0.0

@dataclass
class PredictionResponse:
    """Data model for prediction responses"""
    predicted_role: str
    confidence_scores: Dict[str, float]
    keywords_extracted: List[tuple]
    explanation: Optional[str] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

@dataclass
class ModelInfo:
    """Data model for model information"""
    version: str
    accuracy: float
    training_samples: int
    last_trained: datetime
    supported_roles: List[str]

@dataclass
class HealthStatus:
    """Data model for health status"""
    status: str
    model_trained: bool
    version: str
    uptime: Optional[str] = None
    last_prediction: Optional[datetime] = None

@dataclass
class TrainingStats:
    """Data model for training statistics"""
    total_samples: int
    accuracy: float
    precision: Dict[str, float]
    recall: Dict[str, float]
    f1_score: Dict[str, float]
    training_time: float
    cross_val_score: float