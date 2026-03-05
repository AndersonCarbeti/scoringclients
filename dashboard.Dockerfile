FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app
COPY . /app

RUN pip install --no-cache-dir -r "Projet dashboard/requirements.txt"

EXPOSE 10000
CMD ["sh", "-c", "streamlit run 'Projet dashboard/app_dashboard.py' --server.port=${PORT:-10000} --server.address=0.0.0.0"]
