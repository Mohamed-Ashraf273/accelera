# Deployment Module

The Deployment Module provides command-line utilities for packaging trained model
artifacts, running them behind a FastAPI prediction service, and deploying the
service with Docker, Heroku, or EC2.

## Overview

Deployment code lives in `accelera/src/deployment_module`.

- `vcs.py` stores and restores snapshots of `config.json` and `models/`.
- `accelera_deployment/deployment.py` prepares Docker build files and runs
  local, Heroku, or EC2 deployment commands.
- `accelera_deployment/server.py` serves predictions through FastAPI.

Most examples below assume you are running commands from the deployment module
directory:

```bash
cd accelera/src/deployment_module
```

## Prerequisites

- Docker for local and container deployments.
- Heroku CLI for Heroku commands.
- SSH access and `rsync` for EC2 commands.
- A valid `config.json` with model artifact paths relative to
  `accelera/src/deployment_module`.

Example `config.json`:

```json
{
  "models": {
    "scaler_path": "models/scaler.pkl",
    "model_path": "models/model.pkl"
  },
  "schema": {
    "features": [
      {"name": "sepal length (cm)", "type": "number", "min": 0},
      {"name": "sepal width (cm)", "type": "number", "min": 0},
      {"name": "petal length (cm)", "type": "number", "min": 0},
      {"name": "petal width (cm)", "type": "number", "min": 0}
    ]
  },
  "tracking": {
    "enabled": true,
    "path": "prediction_logs/predictions.jsonl"
  }
}
```

Absolute model paths are rejected during Docker preparation because artifacts
must be copied into the Docker image from inside the deployment module.

## Prediction API

The generated service exposes:

| Endpoint | Method | Input | Output |
| --- | --- | --- | --- |
| `/health` | `GET` | None | Service, model, schema, and tracking status |
| `/gui` | `GET` | Browser | Schema-driven prediction GUI |
| `/tracking/summary` | `GET` | None | Tracking totals from the prediction log |
| `/predict` | `POST` | JSON payload with an `input` field | row count and predictions |
| `/predict/csv` | `POST` | Multipart CSV upload named `file` | filename, row count, predictions |

JSON prediction example:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"input": [[5.1, 3.5, 1.4, 0.2]]}'
```

CSV prediction example:

```bash
curl -X POST http://localhost:8000/predict/csv \
  -F "file=@sample_input.csv"
```

Browser GUI:

```text
http://localhost:8000/gui
```

If `config.json` includes `schema.features`, the GUI builds a single-row input
form from that schema and also keeps CSV upload available. If no schema is
configured, the GUI shows CSV upload only.

## Input Validation

The prediction service can validate request data with Great Expectations before
running the model. Configure expected columns under `schema.features` in
`config.json`. Each feature supports:

| Field | Description |
| --- | --- |
| `name` | Input column name. CSV validation uses this directly. |
| `type` | One of `number`, `integer`, `string`, or `boolean`. Defaults to `number`. |
| `required` | Whether values must be non-null. Defaults to `true`. |
| `min` / `max` | Numeric lower and upper bounds. |
| `allowed_values` | Explicit list of accepted values. |

Validation failures return HTTP `422` with the failed schema checks.

## Prediction Tracking

When `tracking.enabled` is true, each prediction request is appended to a JSONL
file. The log records timestamp, endpoint, status, row count, latency,
predictions for successful requests, and error details for failed requests.

The default log path is:

```text
prediction_logs/predictions.jsonl
```

Use `/tracking/summary` to inspect total requests, total rows, status counts,
and the most recent event.

## Model Snapshot Commands

Use `vcs.py` to save, inspect, and restore deployable model snapshots. Each
snapshot stores the current `config.json` and `models/` directory under
`experiments/<commit-hash>/`.

| Command | Description |
| --- | --- |
| `python vcs.py init` | Create the deployment experiment index. |
| `python vcs.py commit -m "message"` | Snapshot the current config and model files. |
| `python vcs.py log` | Show saved snapshots, newest first. |
| `python vcs.py show <hash>` | Show one snapshot's metadata, config, and model files. |
| `python vcs.py deploy <hash>` | Restore a snapshot into the live `config.json` and `models/`. |
| `python vcs.py status` | Show snapshot count, HEAD, and deployed snapshot. |

Typical snapshot workflow:

```bash
python vcs.py init
python vcs.py commit -m "baseline random forest"
python vcs.py log
python vcs.py deploy <commit-hash>
python vcs.py status
```

`<hash>` can be a short prefix. Use enough characters to avoid matching the
wrong snapshot.

## Docker Commands

Use `accelera_deployment/deployment.py` for container preparation and local
Docker runs.

| Command | Description |
| --- | --- |
| `python accelera_deployment/deployment.py prepare` | Write `accelera_deployment/requirements.txt` and `Dockerfile`. |
| `python accelera_deployment/deployment.py build` | Build the local Docker image named `ml-model`. |
| `python accelera_deployment/deployment.py build --no-cache` | Build `ml-model` without using Docker layer cache. |
| `python accelera_deployment/deployment.py run-local` | Run the existing local `ml-model` image. |
| `python accelera_deployment/deployment.py local` | Run `prepare`, `build`, and `run-local` in sequence. |
| `python accelera_deployment/deployment.py local --no-cache` | Run the full local workflow with a fresh Docker build. |

Local deployment:

```bash
python accelera_deployment/deployment.py local
```

The container defaults to port `8000`. Set `PORT` before running to use another
port:

```bash
PORT=9000 python accelera_deployment/deployment.py local
```

`run-local` stops any existing Docker container publishing the selected port
before starting the new container. Deployment commands print the browser GUI URL
after the service starts or is released.

## Heroku Commands

Heroku commands default to app name `accelera1`. Pass `--app <name>` to use a
different app.

| Command | Description |
| --- | --- |
| `python accelera_deployment/deployment.py heroku-login` | Run `heroku login`. |
| `python accelera_deployment/deployment.py heroku-create --app <name>` | Create a Heroku app with the container stack. |
| `python accelera_deployment/deployment.py heroku-container-login` | Log in to the Heroku container registry. |
| `python accelera_deployment/deployment.py heroku-push --app <name>` | Prepare files and push the `web` container. |
| `python accelera_deployment/deployment.py heroku-release --app <name>` | Release the pushed `web` container. |
| `python accelera_deployment/deployment.py heroku-open --app <name>` | Open the Heroku app. |
| `python accelera_deployment/deployment.py heroku-deploy --app <name>` | Run login, container login, push, release, and open. |
| `python accelera_deployment/deployment.py heroku-deploy --app <name> --create` | Create the app before running the full deployment flow. |

Full Heroku deployment:

```bash
python accelera_deployment/deployment.py heroku-deploy \
  --app accelera-production \
  --create
```

Incremental push and release:

```bash
python accelera_deployment/deployment.py heroku-container-login
python accelera_deployment/deployment.py heroku-push --app accelera-production
python accelera_deployment/deployment.py heroku-release --app accelera-production
```

## EC2 Commands

EC2 commands require `--host`. The default SSH user is `ec2-user`, the default
remote directory is `~/deployment-app`, and the default Docker image and
container names are both `ml-model`.

| Command | Description |
| --- | --- |
| `python accelera_deployment/deployment.py ec2-deploy --host <host>` | Sync deployment files to EC2, build the Docker image, and run the container. |
| `python accelera_deployment/deployment.py ec2-deploy --host <host> --key ~/.ssh/key.pem` | Deploy using an SSH private key. |
| `python accelera_deployment/deployment.py ec2-deploy --host <host> --user ubuntu` | Deploy with a non-default SSH user. |
| `python accelera_deployment/deployment.py ec2-deploy --host <host> --remote-dir ~/apps/accelera` | Sync into a custom remote directory. |
| `python accelera_deployment/deployment.py ec2-deploy --host <host> --port 9000` | Expose the service on a custom public port. |
| `python accelera_deployment/deployment.py ec2-deploy --host <host> --image <image> --container <name>` | Use custom Docker image and container names. |
| `python accelera_deployment/deployment.py ec2-deploy --host <host> --install-docker` | Install and start Docker on the EC2 host if missing. |
| `python accelera_deployment/deployment.py ec2-deploy --host <host> --no-cache` | Build the remote Docker image without cache. |
| `python accelera_deployment/deployment.py ec2-stop --host <host>` | Stop the default EC2 container. |
| `python accelera_deployment/deployment.py ec2-stop --host <host> --container <name>` | Stop a named EC2 container. |
| `python accelera_deployment/deployment.py ec2-logs --host <host>` | Follow logs from the default EC2 container. |
| `python accelera_deployment/deployment.py ec2-logs --host <host> --container <name>` | Follow logs from a named EC2 container. |

Basic EC2 deployment:

```bash
python accelera_deployment/deployment.py ec2-deploy \
  --host 1.2.3.4 \
  --user ec2-user \
  --key ~/.ssh/accelera.pem \
  --install-docker
```

After deployment, open the selected port, `8000` by default:

```text
http://<host>:8000
http://<host>:8000/gui
```

Make sure the EC2 security group allows inbound traffic on the selected port.

## Generated Docker Image

`prepare` writes a Dockerfile that:

- Uses `python:3.11-slim`.
- Installs FastAPI, Uvicorn, Great Expectations, scikit-learn, category
  encoders, NumPy, Pandas, Pydantic, and multipart upload support.
- Copies `server.py`, `modelservice.py`, `schema_validation.py`, `tracking.py`,
  `config.json`, and configured model artifacts into `/app`.
- Starts Uvicorn on `0.0.0.0` with `${PORT:-8000}`.

## Notes

- `python model.py` runs the sample Iris training and prediction script. It is
  not a subcommand-based CLI.
- The EC2 deployment command performs a post-start health check against
  `/health`, which is exposed by the FastAPI service.

## Related Modules

- [Core Pipeline](core-pipeline.md) - Train models for deployment.
- [AutoML Module](automl.md) - Auto-generate deployable pipelines.
- [Benchmark Platform](benchmark.md) - Track deployed model performance.

---

**Last Updated**: June 2026
