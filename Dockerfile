FROM ghcr.io/bibradar/ml_backend_base:latest

WORKDIR /app

COPY . .

CMD ["poetry", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]