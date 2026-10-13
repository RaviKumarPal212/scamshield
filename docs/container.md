# Local container run

The container recipe packages the FastAPI app and static web interface, but it deliberately does not copy the model weights into the image. The selected model weights are about 541 MB and are mounted read-only at runtime along with the small model-selection lock file. This keeps the image and model lifecycle separate and avoids accidentally publishing the model artifact.

## Requirements

- Docker Desktop with Docker Compose available from PowerShell.
- The local model directory `models/sms_distilmbert/` and lock file `reports/sms_model_selection.json` already present.
- Internet access during the first image build to download Python dependencies. At runtime the app is configured to load model files locally and remain offline.

## Build and run

From the project folder:

```powershell
docker compose up --build
```

In a second PowerShell window, check the container state and startup logs:

```powershell
docker compose ps
docker compose logs scamshield
```

Wait for the health state to become `healthy`, then open `http://127.0.0.1:8001/`. The API health endpoint is `http://127.0.0.1:8001/health`. This Compose configuration binds only to the local machine; it is not a public deployment.

Stop the app with `Ctrl+C`, or from another PowerShell window run:

```powershell
docker compose down
```

The container runs as a non-root user, has a read-only root filesystem, mounts model files read-only, and sets the Hugging Face/Transformers offline flags. The API itself does not persist submitted messages. A public deployment needs a separate privacy, abuse-prevention, capacity, and hosting review before exposing the endpoint to the internet.

The PyTorch CPU wheel source follows the [official PyTorch installation selector](https://pytorch.org/get-started/locally/). Model and lock mounts use Docker Compose read-only bind mounts; see the [Docker Compose volumes reference](https://docs.docker.com/reference/compose-file/services/).
