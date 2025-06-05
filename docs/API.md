# Supply Chain Risk Intelligence System - API Documentation

## Overview

The Supply Chain Risk Intelligence System provides real-time risk assessment for supply chain management using multimodal data sources including satellite imagery, weather patterns, economic indicators, news events, and social media sentiment.

## Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd supply-chain-risk-intelligence

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys and configuration

# Run the system
python main.py
```

### Environment Variables

```bash
# API Keys
OPENWEATHER_API_KEY=your_openweather_api_key
NEWS_API_KEY=your_news_api_key
TWITTER_BEARER_TOKEN=your_twitter_bearer_token

# Database
DATABASE_URL=sqlite:///supply_chain_risk.db
ASYNC_DATABASE_URL=sqlite+aiosqlite:///supply_chain_risk.db

# Redis Cache
REDIS_URL=redis://localhost:6379/0

# Authentication
JWT_SECRET_KEY=your_secret_key
API_DEMO_TOKEN=demo_user_token_12345
API_ADMIN_TOKEN=admin_user_token_67890

# Cache Configuration
CACHE_DEFAULT_TTL=3600
CACHE_MAX_MEMORY_ITEMS=1000

# Rate Limiting
RATE_LIMIT_REQUESTS=100
RATE_LIMIT_WINDOW=60
```

## API Endpoints

### Authentication

All API endpoints require authentication using Bearer tokens.

**Headers:**
```
Authorization: Bearer <token>
```

**Available Tokens:**
- Demo Token: `demo_user_token_12345` (read-only access)
- Admin Token: `admin_user_token_67890` (full access)

### Core Endpoints

#### Health Check
```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-01T12:00:00Z",
  "version": "1.0.0",
  "uptime": 3600,
  "checks": {
    "database": "healthy",
    "cache": "healthy",
    "models": "healthy"
  }
}
```

#### Risk Prediction
```http
POST /predict
```

**Request Body:**
```json
{
  "location": "Shanghai Port, China",
  "data_sources": ["satellite", "weather", "economic", "news", "social"],
  "include_recommendations": true,
  "confidence_threshold": 0.7
}
```

**Response:**
```json
{
  "location": "Shanghai Port, China",
  "timestamp": "2024-01-01T12:00:00Z",
  "overall_risk": 0.65,
  "confidence": 0.85,
  "uncertainty": 0.12,
  "individual_predictions": {
    "satellite": 0.7,
    "weather": 0.6,
    "economic": 0.65,
    "news": 0.68,
    "social": 0.58
  },
  "model_agreement": 0.78,
  "processing_time_ms": 245,
  "recommendations": [
    {
      "type": "mitigation",
      "priority": "high",
      "message": "Consider alternative shipping routes",
      "confidence": 0.85
    }
  ],
  "alerts": [
    {
      "severity": "medium",
      "message": "Weather conditions may affect shipping",
      "source": "weather"
    }
  ]
}
```

#### Batch Prediction
```http
POST /predict/batch
```

**Request Body:**
```json
{
  "locations": [
    "Shanghai Port, China",
    "Los Angeles Port, USA",
    "Hamburg Port, Germany"
  ],
  "data_sources": ["satellite", "weather", "economic"]
}
```

#### Model Status
```http
GET /model/status
```

**Response:**
```json
{
  "models": {
    "satellite": {
      "status": "loaded",
      "version": "1.2.0",
      "last_updated": "2024-01-01T10:00:00Z",
      "accuracy": 0.89
    },
    "weather": {
      "status": "loaded",
      "version": "1.1.0",
      "last_updated": "2024-01-01T10:00:00Z",
      "accuracy": 0.92
    }
  },
  "fusion_engine": {
    "status": "ready",
    "version": "2.0.0"
  }
}
```

### Data Quality Endpoints

#### Data Quality Status
```http
GET /data/quality
```

**Response:**
```json
{
  "overall_score": 0.87,
  "sources": {
    "satellite": {
      "score": 0.92,
      "last_updated": "2024-01-01T11:30:00Z",
      "issues": [],
      "recommendations": []
    },
    "weather": {
      "score": 0.88,
      "last_updated": "2024-01-01T11:45:00Z",
      "issues": ["missing data for 2 locations"],
      "recommendations": ["refresh weather data"]
    }
  }
}
```

### Historical Data Endpoints

#### Risk Trends
```http
GET /history/risks/{location}?days=7
```

**Response:**
```json
{
  "location": "Shanghai Port, China",
  "period": {
    "start": "2024-01-01T00:00:00Z",
    "end": "2024-01-08T00:00:00Z"
  },
  "trends": [
    {
      "timestamp": "2024-01-01T12:00:00Z",
      "overall_risk": 0.65,
      "individual_risks": {
        "satellite": 0.7,
        "weather": 0.6
      }
    }
  ],
  "statistics": {
    "avg_risk": 0.68,
    "max_risk": 0.85,
    "min_risk": 0.45,
    "trend_direction": "increasing"
  }
}
```

### Alert Management

#### Get Active Alerts
```http
GET /alerts?severity=high
```

**Response:**
```json
{
  "alerts": [
    {
      "id": 123,
      "timestamp": "2024-01-01T12:00:00Z",
      "severity": "high",
      "type": "risk",
      "title": "High Risk Detected",
      "message": "Significant supply chain disruption risk detected",
      "location": "Shanghai Port, China",
      "status": "active"
    }
  ],
  "total_count": 5
}
```

#### Acknowledge Alert
```http
POST /alerts/{alert_id}/acknowledge
```

#### Resolve Alert
```http
POST /alerts/{alert_id}/resolve
```

### System Monitoring

#### System Metrics
```http
GET /system/metrics
```

**Response:**
```json
{
  "timestamp": "2024-01-01T12:00:00Z",
  "performance": {
    "cpu_usage": 45.2,
    "memory_usage": 68.7,
    "disk_usage": 23.1
  },
  "api": {
    "requests_per_minute": 150,
    "avg_response_time_ms": 245,
    "error_rate": 0.02
  },
  "cache": {
    "hit_rate": 0.85,
    "size_mb": 128.5
  },
  "models": {
    "prediction_latency_ms": 89,
    "accuracy": 0.89
  }
}
```

## Rate Limiting

- **Default Limit:** 100 requests per minute per user
- **Headers Returned:**
  - `X-RateLimit-Limit`: Rate limit window size
  - `X-RateLimit-Remaining`: Requests remaining in window
  - `X-RateLimit-Reset`: Time when window resets

## Error Handling

### Error Response Format
```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid request parameters",
    "details": {
      "field": "location",
      "issue": "Location is required"
    },
    "correlation_id": "req_abc123def456"
  },
  "timestamp": "2024-01-01T12:00:00Z"
}
```

### Common Error Codes

| Code | HTTP Status | Description |
|------|-------------|-------------|
| `VALIDATION_ERROR` | 400 | Invalid request parameters |
| `UNAUTHORIZED` | 401 | Missing or invalid authentication |
| `FORBIDDEN` | 403 | Insufficient permissions |
| `NOT_FOUND` | 404 | Resource not found |
| `RATE_LIMITED` | 429 | Rate limit exceeded |
| `INTERNAL_ERROR` | 500 | Internal server error |
| `SERVICE_UNAVAILABLE` | 503 | Service temporarily unavailable |

## Data Sources

### Satellite Data
- **Source**: Sentinel-2, Landsat-8
- **Resolution**: 10-30 meters
- **Update Frequency**: Daily
- **Coverage**: Global

### Weather Data
- **Source**: OpenWeatherMap API
- **Parameters**: Temperature, precipitation, wind, pressure
- **Update Frequency**: Every 3 hours
- **Forecast Range**: 5 days

### Economic Data
- **Source**: Yahoo Finance, Alpha Vantage
- **Indicators**: GDP, inflation, exchange rates, commodity prices
- **Update Frequency**: Daily
- **Historical Range**: 10 years

### News Data
- **Source**: NewsAPI, Reuters, Bloomberg
- **Languages**: English, Chinese, Spanish
- **Update Frequency**: Real-time
- **Retention**: 30 days

### Social Media
- **Source**: Twitter API v2
- **Metrics**: Sentiment, volume, engagement
- **Update Frequency**: Real-time
- **Languages**: Multi-language support

## Model Architecture

### Individual Risk Models

Each data source has a dedicated risk assessment model:

1. **Satellite Model**: Convolutional Neural Network for infrastructure monitoring
2. **Weather Model**: Time series model for weather impact prediction
3. **Economic Model**: Ensemble model for economic indicator analysis
4. **News Model**: Transformer-based sentiment and event detection
5. **Social Model**: Multi-modal sentiment and trend analysis

### Fusion Engine

The Multimodal Transformer Fusion Engine combines predictions from all sources:

- **Architecture**: Transformer with attention mechanisms
- **Input**: Concatenated feature vectors from all models
- **Output**: Unified risk score with confidence intervals
- **Training**: Supervised learning on historical supply chain events

## Caching Strategy

### Multi-Level Caching

1. **Request-Level Cache**
   - TTL: 5 minutes
   - Scope: Identical API requests
   - Storage: Redis

2. **Data-Level Cache**
   - TTL: 1 hour
   - Scope: Raw data from external APIs
   - Storage: Redis

3. **Model-Level Cache**
   - TTL: 24 hours
   - Scope: Model predictions and features
   - Storage: Redis + Memory

### Cache Keys

- Predictions: `predictions:{location_hash}:{params_hash}`
- Data: `data:{source}:{location}:{timestamp}`
- Models: `models:{model_name}:{version}`

## Security

### Authentication
- Bearer token authentication
- Role-based access control (demo/admin)
- Token rotation support

### Security Headers
- Content Security Policy
- CORS protection
- HTTPS enforcement (production)
- Request rate limiting

### Data Privacy
- No personal data storage
- Anonymized analytics
- GDPR compliance measures

## Performance

### Benchmarks
- **Prediction Latency**: < 500ms (95th percentile)
- **API Response Time**: < 200ms (average)
- **Cache Hit Rate**: > 80%
- **Uptime**: > 99.9%

### Optimization
- Model warming on startup
- Async processing for data collection
- Connection pooling for databases
- Batch processing for multiple predictions

## Monitoring and Alerting

### Metrics Collection
- Prometheus metrics endpoint: `/metrics`
- Custom business metrics
- System performance metrics
- Error tracking and logging

### Alert Types
1. **Risk Alerts**: High-risk situations detected
2. **Quality Alerts**: Data quality issues
3. **System Alerts**: Performance degradation
4. **Security Alerts**: Authentication failures

### Dashboard
Access the monitoring dashboard at `/dashboard` for:
- Real-time risk visualization
- System health monitoring
- Data quality metrics
- Performance analytics

## Troubleshooting

### Common Issues

1. **High Prediction Latency**
   - Check model cache status
   - Verify data source availability
   - Monitor system resources

2. **Low Cache Hit Rate**
   - Review cache TTL settings
   - Check Redis connectivity
   - Verify request patterns

3. **Data Quality Issues**
   - Monitor external API status
   - Check data validation logs
   - Review data freshness metrics

### Support

For technical support and questions:
- Check the troubleshooting section
- Review system logs
- Monitor dashboard alerts
- Contact system administrators

## API Clients

### Python Client Example

```python
import requests

class RiskIntelligenceClient:
    def __init__(self, base_url, token):
        self.base_url = base_url
        self.headers = {"Authorization": f"Bearer {token}"}
    
    def predict_risk(self, location):
        response = requests.post(
            f"{self.base_url}/predict",
            json={"location": location},
            headers=self.headers
        )
        return response.json()

# Usage
client = RiskIntelligenceClient(
    "http://localhost:8000",
    "demo_user_token_12345"
)

risk = client.predict_risk("Shanghai Port, China")
print(f"Risk Level: {risk['overall_risk']}")
```

### JavaScript Client Example

```javascript
class RiskIntelligenceClient {
    constructor(baseUrl, token) {
        this.baseUrl = baseUrl;
        this.headers = {
            'Authorization': `Bearer ${token}`,
            'Content-Type': 'application/json'
        };
    }
    
    async predictRisk(location) {
        const response = await fetch(`${this.baseUrl}/predict`, {
            method: 'POST',
            headers: this.headers,
            body: JSON.stringify({ location })
        });
        return response.json();
    }
}

// Usage
const client = new RiskIntelligenceClient(
    'http://localhost:8000',
    'demo_user_token_12345'
);

client.predictRisk('Shanghai Port, China')
    .then(risk => console.log(`Risk Level: ${risk.overall_risk}`));
```
