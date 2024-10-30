# Dockerfile for training
FROM python:3.8-slim

# Install dependencies
RUN pip install mlflow scikit-learn

# Set up working directory
WORKDIR /app

# Copy source code
COPY src /app/src
COPY data /app/data

# Set environment variables (Optional)
ENV MLFLOW_TRACKING_URI=file:/mlops_project/mlruns

# Run training script
CMD ["python", "src/train.py"]
