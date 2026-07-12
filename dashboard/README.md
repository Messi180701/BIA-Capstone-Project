# E-Commerce Customer Scoring API — FastAPI

Real-time prediction endpoint using the trained Random Forest classifier.  
**Model Performance:** Accuracy 98.73% | ROC-AUC 99.94% | F1 98.54%

---

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Ensure model files are in models/ folder
#    models/best_classifier_rf.pkl
#    models/kmeans_model.pkl
#    models/scalar.pkl

# 3. Run the API
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Open **http://localhost:8000/docs** for the interactive Swagger UI.

---

## Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | API & model health check |
| GET | `/model/info` | Model metadata & hyperparameters |
| POST | `/predict` | Single customer prediction |
| POST | `/predict/batch` | Batch prediction (up to 500 customers) |

---

## Example: Single Prediction

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "customer_id": "12345",
    "recency": 10,
    "frequency": 15,
    "monetary": 9500.0
  }'
```

**Response:**
```json
{
  "customer_id": "12345",
  "recency": 10,
  "frequency": 15,
  "monetary": 9500.0,
  "segment": "VIP",
  "is_high_value": true,
  "high_value_probability": 0.9800,
  "confidence": "Very High",
  "marketing_strategy": "Exclusive loyalty rewards, early access, personal account manager.",
  "segment_description": "Highest value customers. Recent, frequent, high spenders.",
  "priority": "HIGH",
  "timestamp": "2026-07-01T10:00:00"
}
```

---

## Example: Batch Prediction

```bash
curl -X POST "http://localhost:8000/predict/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "customers": [
      {"customer_id": "A1", "recency": 5,  "frequency": 20, "monetary": 12000},
      {"customer_id": "B2", "recency": 200,"frequency": 1,  "monetary": 150},
      {"customer_id": "C3", "recency": 15, "frequency": 4,  "monetary": 1800}
    ]
  }'
```

**Response includes:**
- Individual predictions per customer
- `segment_distribution` — count per segment across the batch
- `high_value_percentage` — % of batch that are high-value

---

## Input Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `customer_id` | string | No | Your internal customer ID |
| `recency` | int (0–1000) | Yes | Days since last purchase |
| `frequency` | int (1–1000) | Yes | Number of unique orders |
| `monetary` | float (0–500000) | Yes | Total spend in £ |

---

## Deployment (Production)

```bash
# Using Gunicorn + Uvicorn workers
pip install gunicorn
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000

# Docker
docker build -t customer-scoring-api .
docker run -p 8000:8000 customer-scoring-api
```
