REST API for Security Findings Role Assignment System.

## Base URL
```
http://localhost:5000
```

## Endpoints

### POST /predict
Predict role for a security finding.

**Request:**
```json
{
  "finding": "AWS S3 bucket has public read access enabled"
}
```

**Response:**
```json
{
  "predicted_role": "Cloud Security Engineer",
  "confidence_scores": {
    "Cloud Security Engineer": 0.85,
    "Cloud Admin": 0.68
  },
  "keywords_extracted": [["aws", 0.9], ["s3", 0.8]],
  "explanation": "Cloud infrastructure security issue..."
}
```

### POST /retrain
Retrain the ML model.

**Response:**
```json
{
  "message": "Model retrained successfully"
}
```

### GET /health
Check system health.

**Response:**
```json
{
  "status": "healthy",
  "model_trained": true
}
```