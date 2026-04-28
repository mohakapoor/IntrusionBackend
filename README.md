# Intrusion Tracker Backend

FastAPI backend for network intrusion detection using machine learning models.

## Project Structure

```
intrusionTrackerBackend/
├── cors.py                 # CORS configuration
├── predict.py              # Model loading and prediction logic
├── router.py               # FastAPI routes
├── utils.py                # Data sampling and scaling utilities
├── test.py                 # Test scripts
└── models/                 # ML model files
    ├── ffnn_multiclass.pt
    ├── isolation_forest.joblib
    ├── autoencoder_model.pth
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

> **Authentication Required:** All prediction endpoints require authentication (Bearer token for HTTP, JSON token for WebSocket).

#### Hybrid Detection (Recommended)
```http
POST /intrusiondetection/predict/hybrid
```
Uses the hierarchical pipeline: Anomaly detection (Unsupervised) -> Classification (Supervised).

---

#### Prediction By Index
```http
POST /intrusiondetection/predict/hybrid_by_index
```
**Request Body:**
```json
{
  "index": 17502
}
```

---

#### WebSocket Streaming
```http
WS /intrusiondetection/ws/stream
```
Streams the dataset in batches of 100 rows.

**Authentication & Start Command:**
After connecting, send the following JSON message to start the stream:
```json
{
  "command": "start",
  "token": "your_secure_token_here",
  "start_at": 0
}
```

---

#### Individual Model Endpoints
- `POST /intrusiondetection/predict/logreg`
- `POST /intrusiondetection/predict/lightgbm`
- `POST /intrusiondetection/predict/ffnn`
- `POST /intrusiondetection/predict/autoencoder`
- `POST /intrusiondetection/predict/isolationForest`

## Models & Logic

1.  **Unsupervised Layer**: Autoencoder & Isolation Forest act as "flaggers" for anomalies.
2.  **Supervised Layer**: LightGBM, FFNN, and LogReg provide precise classification.
3.  **LGBM Veto**: In the hybrid pipeline, LightGBM acts as the final judge to reduce false positives from the unsupervised models.

## Setup

```bash
# Install dependencies
pip install fastapi uvicorn pandas numpy torch lightgbm scikit-learn python-dotenv joblib polars websockets
```

**Environment Variables (.env):**
- `TOKEN`: The secure token used for all authenticated requests.
- `PATH_STANDARD`, `PATH_MINMAX`, `PATH_PCA`, `PATH_LIGHTGBM`: Paths to the model files.