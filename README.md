# Intrusion Tracker Backend

FastAPI backend for network intrusion detection using machine learning models.

## Project Structure

```
intrusionTrackerBackend/
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

## API Endpoints

### Public Endpoints

**GET** `/intrusiondetection/api_name`  
Returns API name and version

**GET** `/intrusiondetection/health`  
Health check endpoint

### Protected Endpoints

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

## Models

Three ML models for network traffic classification:
- **Logistic Regression**: Binary classification
- **LightGBM**: Multi-class classification  
- **Feed-Forward Neural Network**: Multi-class classification

All models expect exactly 34 numeric features as input, normalized according to training data.

## Setup

```bash
# Clone and install dependencies
git clone <repository-url>
cd intrusionTrackerBackend
python -m venv .venv
source .venv/bin/activate  # Linux/Mac or .venv\Scripts\activate on Windows
pip install fastapi uvicorn pandas numpy torch lightgbm scikit-learn python-dotenv joblib
```

Environment variables:
- `API_HOST`: Server host (default: 0.0.0.0)
- `API_PORT`: Server port (default: 8000)
- `BEARER_TOKEN`: Authentication token
- `ALLOWED_ORIGINS`: CORS allowed origins

Model files must be placed in the `models/` directory.