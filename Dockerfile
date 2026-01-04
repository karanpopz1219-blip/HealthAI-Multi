FROM python:3.11-slim

# Reduce image size and set a working directory
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install OS deps for common Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy minimal files and install Python deps
COPY requirements.txt ./
RUN pip install --upgrade pip setuptools wheel
RUN pip install -r requirements.txt || true

# Copy project
COPY . /app

# Expose Streamlit port
EXPOSE 8501

# Default env to suppress noisy warnings in demo
ENV HEALTHAI_SUPPRESS_WARNINGS=1

# Start Streamlit app
CMD ["streamlit", "run", "health_care_bot.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
