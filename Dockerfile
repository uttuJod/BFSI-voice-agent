FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends build-essential git && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
# CPU torch keeps the image small; the LLM runs in the vllm service.
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt
COPY . .
RUN mkdir -p results
EXPOSE 8000 8010
CMD ["python", "-m", "app.main"]
