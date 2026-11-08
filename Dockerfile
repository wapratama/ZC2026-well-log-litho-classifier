# syntax=docker/dockerfile:1
# Well-log lithology API. Build:  docker build -t lithology-api .
# Run:                            docker run --rm -p 8000:8000 lithology-api

ARG PYTHON_VERSION=3.11
ARG UV_VERSION=latest
FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv

# ---------- builder: resolve exactly what uv.lock says, runtime dependencies only ----------
FROM python:${PYTHON_VERSION}-slim AS builder
COPY --from=uv /uv /bin/uv
ENV UV_LINK_MODE=copy UV_COMPILE_BYTECODE=1 UV_PYTHON_DOWNLOADS=never
WORKDIR /app
# Copy only the dependency files first so this layer is cached until they change.
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --no-install-project

# ---------- runtime: slim image, no build tools, non-root ----------
FROM python:${PYTHON_VERSION}-slim AS runtime
# LightGBM needs the OpenMP runtime, which the slim image does not include.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
RUN useradd --create-home --uid 1000 appuser
WORKDIR /app
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MODEL_PATH=/app/models/model.joblib

COPY --from=builder /app/.venv /app/.venv
COPY lithology ./lithology
COPY predict.py ./
COPY models/model.joblib ./models/model.joblib

USER appuser
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status == 200 else 1)"

# One worker: LightGBM already uses every core for a single request.
CMD ["uvicorn", "predict:app", "--host", "0.0.0.0", "--port", "8000"]
