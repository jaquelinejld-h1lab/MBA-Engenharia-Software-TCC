# syntax=docker/dockerfile:1.7
# Imagem da interface Streamlit. Cliente HTTP puro da API: a imagem nao contem
# scikit-learn, mlflow nem o modelo; o unico contato com o modelo e via HTTP.
#
# Build (na raiz do repositorio):
#   docker build -f docker/Dockerfile.app -t hypertension-app:latest .
# Run (API acessivel em http://api:8000 na rede do compose):
#   docker run --rm -p 8501:8501 -e HTN_APP__API_URL=http://api:8000 hypertension-app:latest

ARG PYTHON_VERSION=3.11

# ------------------------------------------------------------------ builder
FROM python:${PYTHON_VERSION}-slim AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /build
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

COPY requirements-app.lock ./
RUN pip install --upgrade pip && pip install -r requirements-app.lock

# ------------------------------------------------------------------ runtime
FROM python:${PYTHON_VERSION}-slim AS runtime

ENV PATH="/opt/venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0

RUN groupadd --system app && useradd --system --gid app --create-home app

WORKDIR /app
COPY --from=builder /opt/venv /opt/venv
COPY --chown=app:app configs ./configs
COPY --chown=app:app src ./src
RUN chown -R app:app /app

USER app
EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=3).status == 200 else 1)"

CMD ["streamlit", "run", "src/app/main.py"]
