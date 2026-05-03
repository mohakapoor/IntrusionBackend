# Intrusion Tracker Backend

[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com)
[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)

An enterprise-grade **Network Intrusion Detection System (NIDS)** backend powered by a state-of-the-art hierarchical machine learning pipeline. This system doesn't just predict; it verifies anomalies through a multi-layered "Veto" logic to ensure maximum accuracy and minimum false alarms.

---

## Live Demo

You can explore the project and its capabilities live at:
**[https://www.mohakapoor.in/projects/IntrusionDetection](https://www.mohakapoor.in/projects/IntrusionDetection)**

---

## Key Features

- **Hierarchical ML Pipeline**: Combines Unsupervised Anomaly Detection (Autoencoders, Isolation Forests) with Supervised Classification (LightGBM, XGBoost, FFNN).
- **Real-time Inference**: Optimized for low-latency network flow analysis.
- **WebSocket Streaming**: Stream and evaluate massive datasets with live accuracy telemetry.
- **Secure by Design**: Token-based authentication and restricted CORS policies.
- **Containerized**: Ready for deployment via Docker and Docker Compose.
- **Detailed Analytics**: Built-in tools for generating confusion matrices and classification reports.

---

## Quick Start

### 1. Environment Setup
Clone the repository and create your `.env` file:
```bash
cp .env.example .env # Ensure you fill in your secure TOKEN
```

### 2. Run with Docker (Recommended)
```bash
docker-compose up --build
```

### 3. Access the API
The API will be available at `http://localhost:8000`. 
Check health: `GET /intrusiondetection/health`

---

## Deep Dive

For technical implementation details, model architectures, and full API specifications, please refer to our:

### [Detailed Documentation](./documentation.md)

---

## Tech Stack

- **Framework**: FastAPI
- **Data Engine**: Polars & Pandas
- **ML/DL**: PyTorch, Scikit-Learn, LightGBM, XGBoost
- **Deployment**: Docker, Uvicorn
- **Utilities**: Joblib, NumPy, Matplotlib/Seaborn

---
Developed by **Mohak Kapoor**