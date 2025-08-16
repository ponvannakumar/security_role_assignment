# docs/DEPLOYMENT.md
# Deployment Guide

## Local Development
```bash
python app.py
```

## Docker Deployment
```bash
docker build -t security-role-assignment .
docker run -p 5000:5000 security-role-assignment
```

## Docker Compose
```bash
docker-compose up -d
```

## Kubernetes
```bash
kubectl apply -f k8s-deployment.yaml
```

## Production Considerations
- Use Gunicorn for WSGI server
- Set up reverse proxy (Nginx)
- Configure SSL/TLS
- Set up monitoring
- Regular model retraining
- Backup strategy

# docs/TROUBLESHOOTING.md
# Troubleshooting Guide

## Common Issues

### Model Training Fails
- Check data format and quality
- Verify file paths
- Ensure sufficient training data
- Check dependencies

### Low Prediction Accuracy
- Add more training examples
- Verify data quality
- Check role consistency
- Retrain model

### Performance Issues
- Optimize feature extraction
- Use caching
- Scale horizontally
- Monitor resource usage

### API Errors
- Check request format
- Verify model is trained
- Check server logs
- Validate input data

## Logs Location
- Application logs: `logs/app.log`
- Training logs: `logs/training.log`
- Error logs: `logs/error.log`

## Support
For additional support, check the project documentation or create an issue in the repository.