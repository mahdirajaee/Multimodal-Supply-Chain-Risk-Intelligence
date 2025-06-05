FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN python scripts/setup.py --setup-env

EXPOSE 8000 8501

ENV PYTHONPATH=/app

CMD ["python", "scripts/run.py", "--mode", "full"]
