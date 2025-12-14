# Intrusion Tracker Backend

FastAPI backend for network intrusion detection using machine learning models.

## Overview

This API provides endpoints for classifying network traffic patterns using three different ML models:
- Logistic Regression (binary classification)
- LightGBM (multi-class classification)
- Feed-Forward Neural Network (multi-class classification)

## Setup

1. Clone the repository and navigate to the project directory
```bash
git clone <repository-url>
cd intrusionTrackerBackend
```

2. Create a virtual environment and activate it
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate  # Windows
```

3. Install dependencies
```bash
pip install fastapi uvicorn pandas numpy torch lightgbm scikit-learn python-dotenv joblib
```

4. Create a `.env` file with your configuration
```env
API_HOST=0.0.0.0
API_PORT=8000
BEARER_TOKEN=your-secure-token-here
ALLOWED_ORIGINS=http://localhost:3000,https://yourdomain.com
```

5. Ensure model files are in the `models/` directory:
- `logreg_W.npy` and `logreg_b.npy` (logistic regression)
- `multiclass_lightgbm.joblib` (LightGBM)
- `ffnn_multiclass.pt` (neural network)

## Running the Server

```bash
# Development
uvicorn router:app --reload --host 0.0.0.0 --port 8000

# Production
uvicorn router:app --host 0.0.0.0 --port 8000 --workers 4
```

## API Endpoints

### Public Endpoints

**GET** `/intrusiondetection/api_name`  
Returns API name and version

**GET** `/intrusiondetection/health`  
Health check endpoint

### Protected Endpoints (Require Bearer Token)

All prediction endpoints require:
- Header: `Authorization: Bearer <your-token>`
- Body: `{"features": [array of 34 numeric values]}`

**POST** `/intrusiondetection/predict/logreg`  
Binary classification (0 or 1)

**POST** `/intrusiondetection/predict/lightgbm`  
Multi-class classification

**POST** `/intrusiondetection/predict/ffnn`  
Multi-class classification using neural network

### Example Request

```bash
curl -X POST http://localhost:8000/intrusiondetection/predict/logreg \
  -H "Authorization: Bearer your-secure-token-here" \
  -H "Content-Type: application/json" \
  -d '{
    "features": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
                 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
                 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
                 0.1, 0.2, 0.3, 0.4]
  }'
```

Response: `{"prediction": 0}`

## Project Structure

```
intrusionTrackerBackend/
├── .env                    # Environment configuration
├── cors.py                 # CORS configuration
├── predict.py              # Model loading and prediction logic
├── router.py               # FastAPI routes
├── test.py                 # Test scripts
└── models/                 # ML model files
    ├── ffnn_multiclass.pt
    ├── logreg_b.npy
    ├── logreg_W.npy
    └── multiclass_lightgbm.joblib
```

## Testing

Run the test script to verify everything is working:
```bash
python test.py
```

## Notes

- All models expect exactly 34 numeric features as input
- Features should be normalized according to training data
- Store bearer tokens securely and never commit them
- Configure CORS origins appropriately for production
- Use HTTPS in production environments