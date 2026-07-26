FROM python:3.11-slim
WORKDIR /app
COPY dashboard/fastapi_app/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY dashboard/fastapi_app ./dashboard/fastapi_app
COPY data/outputs ./data/outputs
WORKDIR /app/dashboard/fastapi_app
EXPOSE 8000
CMD ["sh", "-c", "uvicorn fastapi_app:app --host 0.0.0.0 --port ${PORT:-8000}"]
