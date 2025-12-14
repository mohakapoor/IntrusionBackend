# Intrusion Tracker Backend

A FastAPI-based backend API for network intrusion detection using multiple machine learning models to classify network traffic patterns and identify potential security threats.

## 🚀 Features

- **Multiple ML Models**: Supports three different machine learning models for various classification needs
  - Logistic Regression (binary classification)
  - LightGBM (multi-class classification)
  - Feed-Forward Neural Network (multi-class classification)
- **RESTful API**: Clean and intuitive API endpoints for predictions
- **Authentication**: Secure Bearer token authentication for prediction endpoints
- **High Performance**: Built with FastAPI for fast, asynchronous request handling
- **CORS Support**: Configurable Cross-Origin Resource Sharing
- **Health Monitoring**: Built-in health check endpoint

## 🛠️ Technology Stack

- **Framework**: FastAPI
- **ML Libraries**: 
  - PyTorch (Neural Network)
  - LightGBM (Gradient Boosting)
  - NumPy (Logistic Regression)
  - scikit-learn
- **Server**: Uvicorn ASGI server
- **Authentication**: Bearer token-based
- **Data Processing**: Pandas, NumPy

## 📋 API Endpoints

### Public Endpoints

#### Get API Information
```
GET /intrusiondetection/api_name
```
Returns the API name and version information.

**Response:**
```json
{
  "api_name": "Intrusion Detection API"
}
```

#### Health Check
```
GET /intrusiondetection/health
```
Checks if the API is running and all models are loaded correctly.

**Response:**
```json
{
  "status": "healthy",
  "models_loaded": true
}
```

### Protected Endpoints (Require Authentication)

#### Logistic Regression Prediction
```
POST /intrusiondetection/predict/logreg
```
Binary classification using logistic regression model.

**Headers:**
```
Authorization: Bearer <your-token>
```

**Request Body:**
```json
{
  "features": [0.1, 0.2, 0.3, ..., 0.34]  // Array of 34 numeric values
}
```

**Response:**
```json
{
  "prediction": 0  // 0 or 1
}
```

#### LightGBM Prediction
```
POST /intrusiondetection/predict/lightgbm
```
Multi-class classification using LightGBM model.

**Headers:**
```
Authorization: Bearer <your-token>
```

**Request Body:**
```json
{
  "features": [0.1, 0.2, 0.3, ..., 0.34]  // Array of 34 numeric values
}
```

**Response:**
```json
{
  "prediction": 2  // Integer class label
}
```

#### Feed-Forward Neural Network Prediction
```
POST /intrusiondetection/predict/ffnn
```
Multi-class classification using a feed-forward neural network.

**Headers:**
```
Authorization: Bearer <your-token>
```

**Request Body:**
```json
{
  "features": [0.1, 0.2, 0.3, ..., 0.34]  // Array of 34 numeric values
}
```

**Response:**
```json
{
  "prediction": 3  // Integer class label
}
```

## 🔧 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Virtual environment (recommended)

### Setup Instructions

1. **Clone the repository**
```bash
git clone <repository-url>
cd intrusionTrackerBackend
```

2. **Create and activate virtual environment**
```bash
python -m venv .venv

# On Linux/Mac
source .venv/bin/activate

# On Windows
.venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install fastapi uvicorn pandas numpy torch lightgbm scikit-learn python-dotenv joblib
```

4. **Set up environment variables**
Create a `.env` file in the project root:
```env
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000

# Authentication
BEARER_TOKEN=your-secure-token-here

# CORS Configuration
ALLOWED_ORIGINS=http://localhost:3000,https://yourdomain.com

# Model Paths (optional, defaults provided)
LOGREG_W_PATH=models/logreg_W.npy
LOGREG_B_PATH=models/logreg_b.npy
LIGHTGBM_PATH=models/multiclass_lightgbm.joblib
FFNN_PATH=models/ffnn_multiclass.pt
```

5. **Ensure model files are in place**
The following model files should be in the `models/` directory:
- `logreg_W.npy` - Logistic regression weights
- `logreg_b.npy` - Logistic regression bias
- `multiclass_lightgbm.joblib` - LightGBM model
- `ffnn_multiclass.pt` - PyTorch neural network model

## 🚀 Usage

### Starting the Server

```bash
# Development mode
uvicorn router:app --reload --host 0.0.0.0 --port 8000

# Production mode
uvicorn router:app --host 0.0.0.0 --port 8000 --workers 4
```

### Example API Calls

#### Check API Health
```bash
curl http://localhost:8000/intrusiondetection/health
```

#### Make a Prediction (Logistic Regression)
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

#### Make a Prediction (LightGBM)
```bash
curl -X POST http://localhost:8000/intrusiondetection/predict/lightgbm \
  -H "Authorization: Bearer your-secure-token-here" \
  -H "Content-Type: application/json" \
  -d '{
    "features": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
                 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
                 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
                 0.1, 0.2, 0.3, 0.4]
  }'
```

## 🤖 Model Descriptions

### Logistic Regression Model
- **Type**: Binary classifier
- **Input**: 34 numeric features
- **Output**: Binary prediction (0 or 1)
- **Use Case**: Simple binary intrusion detection (normal vs. anomalous traffic)

### LightGBM Model
- **Type**: Multi-class classifier
- **Input**: 34 numeric features
- **Output**: Integer class label
- **Use Case**: Detailed classification of different types of network intrusions
- **Advantages**: Fast inference, handles categorical features well

### Feed-Forward Neural Network (FFNN)
- **Type**: Multi-class classifier
- **Architecture**: Multi-layer perceptron with ReLU activations
- **Input**: 34 numeric features
- **Output**: Integer class label
- **Use Case**: Complex pattern recognition in network traffic
- **Framework**: PyTorch

## 📊 Input/Output Specifications

### Input Format
All prediction endpoints expect a JSON payload with a single field:

```json
{
  "features": [<34 numeric values>]
}
```

**Feature Requirements:**
- Exactly 34 numeric values
- Values should be normalized/scaled according to training data
- Order must match the training feature order

### Output Format
All prediction endpoints return a JSON response:

```json
{
  "prediction": <integer>
}
```

**Prediction Values:**
- Logistic Regression: 0 (normal) or 1 (intrusion)
- LightGBM: Integer class label (varies by dataset)
- FFNN: Integer class label (varies by dataset)

## 🔒 Security Considerations

### Authentication
- All prediction endpoints require Bearer token authentication
- Token must be included in the `Authorization` header
- Format: `Authorization: Bearer <token>`
- Store tokens securely and never commit them to version control

### Best Practices
1. **Use HTTPS in production** to encrypt data in transit
2. **Rotate tokens regularly** to minimize security risks
3. **Implement rate limiting** to prevent abuse
4. **Log access attempts** for security monitoring
5. **Validate input data** to prevent injection attacks

### CORS Configuration
- Configure allowed origins in the `.env` file
- Only allow trusted domains in production
- Use specific origins instead of wildcards

## 🛠️ Development

### Project Structure
```
intrusionTrackerBackend/
├── .env                    # Environment configuration
├── .venv/                  # Virtual environment
├── cors.py                 # CORS configuration
├── predict.py              # Prediction logic and models
├── router.py               # FastAPI routes and main app
├── test.py                 # Test scripts
├── models/                 # ML model files
│   ├── ffnn_multiclass.pt
│   ├── logreg_b.npy
│   ├── logreg_W.npy
│   └── multiclass_lightgbm.joblib
└── README.md               # This file
```

### Testing
Run the test script to verify model loading and predictions:
```bash
python test.py
```

### Adding New Models
1. Train your model and save it to the `models/` directory
2. Add loading logic in `predict.py`
3. Create a new prediction endpoint in `router.py`
4. Update authentication if needed
5. Document the new endpoint in this README

## 🚀 Deployment

### Production Considerations
1. **Use a production ASGI server** like Gunicorn with Uvicorn workers
2. **Set up a reverse proxy** (Nginx/Apache) for SSL termination
3. **Configure process management** with systemd or supervisor
4. **Monitor performance** with tools like Prometheus
5. **Set up logging** to track errors and usage

### Example Production Command
```bash
gunicorn router:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

### Docker Deployment (Optional)
Create a Dockerfile:
```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "router:app", "--host", "0.0.0.0", "--port", "8000"]
```

## 📝 License

[Add your license information here]

## 🤝 Contributing

[Add contribution guidelines here]

## 📧 Contact

[Add contact information here]