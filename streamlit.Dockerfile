FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app
COPY api /app/api

RUN pip install --no-cache-dir -r /app/api/requirements-streamlit.txt

ENV STREAMLIT_API_BASE_URL=https://scoringclients.onrender.com \
    STREAMLIT_CLIENTS_CSV=/app/api/data/clients_sample.csv

WORKDIR /app/api
EXPOSE 10000
CMD ["sh", "-c", "streamlit run streamlit_app.py --server.port=${PORT:-10000} --server.address=0.0.0.0 --server.headless=true"]
