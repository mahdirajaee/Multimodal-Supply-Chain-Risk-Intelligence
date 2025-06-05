# Installation & Quick Start Guide

## Prerequisites
- Python 3.8+
- 8GB+ RAM recommended
- GPU optional but recommended for training

## Installation

```bash
# Clone and navigate to project
cd /path/to/project

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Setup environment
python scripts/setup.py --all
```

## Quick Start

### 1. Train Models (Optional)
```bash
python main.py mode=train
```

### 2. Start API Server
```bash
python main.py mode=api
```

### 3. Launch Dashboard
```bash
streamlit run scripts/dashboard.py
```

### 4. Full System (Recommended)
```bash
python scripts/run.py --mode full
```

Access:
- API: http://localhost:8000
- Dashboard: http://localhost:8501
- API Docs: http://localhost:8000/docs

## Docker Deployment

```bash
# Build and run
docker-compose up --build

# Background mode
docker-compose up -d
```

## API Usage

```python
import requests

# Get risk prediction
response = requests.post('http://localhost:8000/predict', json={
    'timeframe_days': 30,
    'include_recommendations': True
})

risk_data = response.json()
print(f"Overall Risk: {risk_data['risk_scores']['overall_risk']:.2%}")
```

## Testing

```bash
# Run all tests
python scripts/run.py --mode test

# Run specific test
python -m pytest tests/test_models.py -v
```

## Configuration

Edit `config/default.yaml` to customize:
- Model parameters
- Data source settings
- API configuration
- Training hyperparameters

## Environment Variables

Add to `.env` file:
```
TWITTER_API_KEY=your_key_here
REDDIT_CLIENT_ID=your_id_here
WEATHER_API_KEY=your_key_here
NEWS_API_KEY=your_key_here
```
