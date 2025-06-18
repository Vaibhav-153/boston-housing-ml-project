FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Copy all files into the container
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port Render will assign
EXPOSE 8000

# Start the app using Gunicorn (Render sets PORT env variable)
CMD gunicorn --workers=4 --bind 0.0.0.0:$PORT app:app