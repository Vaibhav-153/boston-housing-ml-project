FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt

COPY . .

# Build the model inside the image so a clean Docker build is runnable.
RUN python scripts/download_data.py && python -m src.train

EXPOSE 8000

CMD gunicorn --workers=2 --bind 0.0.0.0:${PORT:-8000} app:app
