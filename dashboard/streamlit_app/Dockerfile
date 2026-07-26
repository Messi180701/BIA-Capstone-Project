FROM python:3.11-slim
WORKDIR /app
COPY dashboard/streamlit_app/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY dashboard/streamlit_app ./dashboard/streamlit_app
COPY models ./models
EXPOSE 8501
CMD ["streamlit", "run", "dashboard/streamlit_app/main.py", "--server.address=0.0.0.0", "--server.port=8501"]
