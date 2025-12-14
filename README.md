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

### 🔓 Public Endpoints

#### Get API Information
```http
GET /intrusiondetection/api_name
```

**Response:**
```json
{
  "name": "Intrusion Detection API",
  "version": "1.0.0"
}
```

**Example:**
```bash
curl -X GET http://localhost:8000/intrusiondetection/api_name
```

---

#### Health Check
```http
GET /intrusiondetection/health
```

**Response:**
```json
{
  "status": "healthy"
}
```

**Example:**
```bash
curl -X GET http://localhost:8000/intrusiondetection/health
```

### 🔒 Protected Endpoints

> **Authentication Required:** All prediction endpoints require a Bearer token in the Authorization header.

#### Common Request Format

All prediction endpoints expect the same request body structure:

**Request Body:**
```json
{
  "features": [
    // Array of exactly 34 numeric values
  ]
}
```

**Required Headers:**
```http
Authorization: Bearer <your-token>
Content-Type: application/json
```

---

#### Logistic Regression Prediction
```http
POST /intrusiondetection/predict/logreg
```

Binary classification endpoint that returns either 0 (normal) or 1 (intrusion).

**Example Request:**
```bash
curl -X POST http://localhost:8000/intrusiondetection/predict/logreg \
  -H "Authorization: Bearer your-secure-token-here" \
  -H "Content-Type: application/json" \
  -d '{
    "features": [
      0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
      0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
      0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
      0.1, 0.2, 0.3, 0.4
    ]
  }'
```

**Response:**
```json
{
  "prediction": 0
}
```

---

#### LightGBM Prediction
```http
POST /intrusiondetection/predict/lightgbm
```

Multi-class classification using LightGBM model.

**Example Request:**
```bash
curl -X POST http://localhost:8000/intrusiondetection/predict/lightgbm \
  -H "Authorization: Bearer your-secure-token-here" \
  -H "Content-Type: application/json" \
  -d '{
    "features": [
      0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
      0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
      0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
      0.1, 0.2, 0.3, 0.4
    ]
  }'
```

**Response:**
```json
{
  "prediction": 2
}
```

---

#### Feed-Forward Neural Network Prediction
```http
POST /intrusiondetection/predict/ffnn
```

Multi-class classification using a neural network model.

**Example Request:**
```bash
curl -X POST http://localhost:8000/intrusiondetection/predict/ffnn \
  -H "Authorization: Bearer your-secure-token-here" \
  -H "Content-Type: application/json" \
  -d '{
    "features": [
      0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
      0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
      0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
      0.1, 0.2, 0.3, 0.4
    ]
  }'
```

**Response:**
```json
{
  "prediction": 3
}
```

### 📝 Response Codes

| Status Code | Description |
|-------------|-------------|
| `200 OK` | Successful prediction |
| `401 Unauthorized` | Invalid or missing Bearer token |
| `422 Unprocessable Entity` | Invalid request body (e.g., wrong number of features) |
| `500 Internal Server Error` | Server error during prediction |

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