FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

WORKDIR /app

# Use the official CPU wheel index for the local demo container. Install the
# remaining application dependencies from PyPI in the next layer.
COPY requirements-api.txt requirements-transformers.txt ./
RUN python -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu "torch>=2.4" \
    && python -m pip install --no-cache-dir -r requirements-api.txt -r requirements-transformers.txt

COPY src ./src
COPY frontend ./frontend

RUN useradd --create-home --uid 10001 appuser
USER 10001:10001

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
  CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"]

CMD ["python", "-m", "uvicorn", "scamshield.api.app:app", "--app-dir", "src", "--host", "0.0.0.0", "--port", "8000"]
