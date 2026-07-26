# E-Commerce Customer Segmentation, Prediction & AI Business Assistant

An end-to-end customer intelligence solution that combines **RFM analytics**, **K-Means segmentation**, **Random Forest prediction**, **Power BI**, **Streamlit**, **FastAPI**, and an **Ollama-powered AI Business Assistant**.

## Project overview

The project helps an e-commerce business understand customer behaviour and turn analytical results into practical actions. It segments customers using Recency, Frequency and Monetary value, predicts whether a customer is high value, and answers business questions using verified Python tools rather than invented LLM statistics.

## Key features

- Data cleaning and exploratory analysis of online retail transactions
- RFM feature engineering and customer profiling
- K-Means customer segmentation
- Random Forest high-value customer prediction
- Interactive Streamlit customer predictor
- FastAPI REST endpoint with Swagger documentation
- Ollama Cloud tool calling for verified business insights
- Segment-level marketing recommendations
- Power BI dashboard for executive reporting

## Architecture

```text
Raw e-commerce data
        │
        ▼
Data cleaning and EDA (Pandas / Jupyter)
        │
        ▼
RFM feature engineering
        │
        ├──────────────► K-Means segmentation
        │
        └──────────────► Random Forest classification
                              │
                 ┌────────────┴────────────┐
                 ▼                         ▼
          Streamlit UI                FastAPI backend
                                             │
                                             ▼
                                   Ollama Cloud LLM
                                             │
                                             ▼
                                Verified Python analysis tools
```

## Repository structure

```text
BIA-Capstone-Project/
├── data/
│   ├── raw/
│   ├── processed/
│   └── outputs/rfm_with_predictions.csv
├── notebooks/
├── models/
│   ├── best_classifier_rf.pkl
│   ├── kmeans_model.pkl
│   └── scalar.pkl
├── dashboard/
│   ├── fastapi_app/
│   │   ├── fastapi_app.py
│   │   ├── ai_assistant.py
│   │   ├── business_analysis.py
│   │   ├── requirements.txt
│   │   └── Dockerfile
│   └── streamlit_app/
│       ├── main.py
│       ├── requirements.txt
│       └── Dockerfile
└── README.md
```

## Local setup

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd BIA-Capstone-Project
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install FastAPI dependencies

```bash
pip install -r dashboard/fastapi_app/requirements.txt
```

Create `dashboard/fastapi_app/.env`:

```env
OLLAMA_API_KEY=your_ollama_cloud_api_key
OLLAMA_MODEL=gpt-oss:20b
```

Start the API:

```powershell
cd dashboard\fastapi_app
python -m uvicorn fastapi_app:app --reload
```

Swagger UI: `http://127.0.0.1:8000/docs`

### 4. Install and run Streamlit

From the repository root:

```bash
pip install -r dashboard/streamlit_app/requirements.txt
streamlit run dashboard/streamlit_app/main.py
```

The local Streamlit app uses `http://127.0.0.1:8000` as its default backend.

To use another backend URL:

Windows PowerShell:

```powershell
$env:FASTAPI_URL="https://your-fastapi-service.onrender.com"
streamlit run dashboard/streamlit_app/main.py
```

## API endpoint

### POST `/ai-business-assistant`

Request:

```json
{
  "question": "Where should I spend my marketing budget?"
}
```

Response:

```json
{
  "question": "Where should I spend my marketing budget?",
  "answer": "AI-generated explanation based on verified analysis tools."
}
```

## Example business questions

- Which segment contributes the most revenue?
- How many customers are present in each segment?
- Give me an overview of all customer groups.
- Where should I spend my marketing budget?
- Which customers should receive retention campaigns?

## Screenshots

Add screenshots after deployment:

```text
assets/
├── streamlit-home.png
├── customer-prediction.png
├── ai-business-assistant.png
├── swagger-api.png
└── power-bi-dashboard.png
```

Then embed them in this section:

```markdown
![Streamlit dashboard](assets/streamlit-home.png)
![AI Business Assistant](assets/ai-business-assistant.png)
```

## Deployment

Recommended deployment architecture:

- **FastAPI backend:** Render Web Service
- **Streamlit frontend:** Streamlit Community Cloud
- **LLM:** Ollama Cloud

### FastAPI on Render

Use these settings:

```text
Root Directory: dashboard/fastapi_app
Build Command: pip install -r requirements.txt
Start Command: uvicorn fastapi_app:app --host 0.0.0.0 --port $PORT
```

Add these environment variables in Render:

```text
OLLAMA_API_KEY=<your secret key>
OLLAMA_MODEL=gpt-oss:20b
```

### Streamlit Community Cloud

Use this entrypoint:

```text
dashboard/streamlit_app/main.py
```

In Advanced settings → Secrets, add:

```toml
FASTAPI_URL = "https://your-fastapi-service.onrender.com"
```

The application also supports a normal `FASTAPI_URL` environment variable.

## Security

- Never commit `.env` or API keys.
- Store the Ollama key in Render environment variables.
- Store the backend URL in Streamlit secrets or environment variables.
- Rotate any key that has been exposed in a ZIP, screenshot or commit history.

## Technology stack

Python, Pandas, NumPy, Scikit-learn, Jupyter, Streamlit, FastAPI, Ollama Cloud, Power BI, Git and GitHub.

## Author

**Viral Shah**  
Data Science and AI Capstone Project
