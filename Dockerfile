# FROM
FROM python:3.11-slim

# workingdir
WORKDIR /app

# Copy requirements.txt first
# This allows Docker to cache the dependency installation layer
COPY requirements.txt .

# Install all Python dependencies required by the project
# --no-cache-dir keeps the Docker image smaller
RUN pip install --no-cache-dir -r requirements.txt

# Copy the complete project into the /app directory
COPY . .

# Prevent Python from creating .pyc files
ENV PYTHONDONTWRITEBYTECODE=1

# Make Python output appear immediately in Docker logs
ENV PYTHONUNBUFFERED=1

# Expose port 5000 because Flask runs on port 5000
EXPOSE 5000

# Start the Flask application when the container starts
CMD ["python", "app.py"]