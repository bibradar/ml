# Use the official Python image from the Docker Hub
FROM python:3.12-slim

# Set environment variables
# ENV POETRY_VERSION=1.1.11

# Install Poetry and PostgreSQL client
RUN apt-get update && apt-get install -y postgresql-client && pip install "poetry"

# Set the working directory
WORKDIR /app

# Copy the pyproject.toml and poetry.lock files
COPY pyproject.toml poetry.lock ./

# Install the dependencies
RUN poetry install --no-root --no-dev

# Copy the application code
COPY . .

# Expose the port FastAPI will run on
EXPOSE 8000

# Command to run the application
CMD ["poetry", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]