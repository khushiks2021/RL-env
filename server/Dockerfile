 # server/Dockerfile
  FROM python:3.11-slim

  # Set working directory
  WORKDIR /app

  # Copy server requirements first (for Docker layer caching)
  COPY server/requirements.txt /app/server/requirements.txt
  RUN pip install --no-cache-dir -r server/requirements.txt

  # Copy all code
  COPY models.py         /app/models.py
  COPY server/           /app/server/

  # Expose port
  EXPOSE 8000

  # Health check — judges use this
  HEALTHCHECK --interval=30s --timeout=10s --start-period=5s \
    CMD curl -f http://localhost:8000/health || exit 1

  # Start server
  CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
