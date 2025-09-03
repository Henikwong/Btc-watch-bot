FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    gfortran \
    libblas-dev \
    liblapack-dev \
    libatlas-base-dev \
    libfreetype6-dev \
    pkg-config \
    ca-certificates \
    curl \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "main.py"]
