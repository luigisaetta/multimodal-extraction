FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

WORKDIR /app

# Minimal OS packages often needed by PDF/image stacks.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY deployment/docker/requirements-ui.txt /tmp/requirements-ui.txt
RUN python -m pip install --upgrade pip \
    && pip install -r /tmp/requirements-ui.txt

# Copy source code in image for immutable deployments.
# During local development docker-compose mounts ../../:/app over this.
COPY . /app

EXPOSE 8501

CMD ["python", "-m", "streamlit", "run", "multimodal_extraction/ui/streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501"]
