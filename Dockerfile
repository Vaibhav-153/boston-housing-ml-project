FROM python:3.12-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN python scripts/download_data.py && python -m src.train
EXPOSE 8000
CMD ["sh", "-c", "gunicorn --workers=2 --bind 0.0.0.0:${PORT:-8000} app:app"]
