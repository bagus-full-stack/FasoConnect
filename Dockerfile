FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    libsndfile1 ffmpeg gcc g++ curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ARG HF_TOKEN
ENV HF_TOKEN=${HF_TOKEN}

RUN python preload_models.py

EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
