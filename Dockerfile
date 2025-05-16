# Use a slim Python base image
#FROM python:3.12-slim
FROM public.ecr.aws/lambda/python:3.12

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Set working directory to where Lambda expects the code
WORKDIR ${LAMBDA_TASK_ROOT}

# Install system dependencies
#RUN apt-get update && apt-get install -y --no-install-recommends build-essential gcc libffi-dev libpq-dev curl && apt-get clean && rm -rf /var/lib/apt/lists/*

# Upgrade pip
#RUN pip install --upgrade pip
RUN python -m pip install --upgrade --no-cache-dir pip

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ /var/task

# Set the Lambda function handler
CMD ["main.lambda_handler"]

