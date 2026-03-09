# Docker deployment (Streamlit UI)

This folder contains a Docker Compose deployment for:

- `multimodal_extraction/ui/streamlit_app.py`

## Files

- `Dockerfile`: Python 3.11 image for the app
- `docker-compose.yml`: single service (`streamlit-ui`)
- `requirements-ui.txt`: runtime dependencies for the UI and pipeline modules it imports

## Prerequisites

- Docker + Docker Compose plugin
- A valid `config_private.py` in project root
- If Oracle wallet is required by your DB setup, ensure wallet path in `config_private.py` exists inside the container

## Run

From project root:

```bash
docker compose -f deployment/docker/docker-compose.yml up --build
```

Then open:

- http://localhost:8501

## Stop

```bash
docker compose -f deployment/docker/docker-compose.yml down
```

## Notes

- Compose mounts `../../` to `/app`, so local code changes are reflected without image rebuild.
- If you need immutable deployment, remove the bind mount (`volumes`) from `docker-compose.yml`.
- If your wallet directory is outside the repo, add an extra volume mapping, for example:
  - `/absolute/host/wallet:/opt/oracle/wallet:ro`
  - and set `VECTOR_WALLET_DIR = "/opt/oracle/wallet"` in `config_private.py`.
