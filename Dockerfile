FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/apps/api/src:/app/platform/core/src

WORKDIR /app

COPY pyproject.toml README.md ./
COPY apps ./apps
COPY platform ./platform

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir ".[dev]"

EXPOSE 8000

CMD ["uvicorn", "pra_api.main:app", "--host", "0.0.0.0", "--port", "8000"]

