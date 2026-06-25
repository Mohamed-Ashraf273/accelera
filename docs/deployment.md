# Deployment Module Reference & Configuration Guide

The Accelera Deployment Module streamlines transitioning machine learning pipelines from local training environments to production-ready API services. It packages trained estimators and preprocessing models, performs schema validation at runtime using Great Expectations, logs request and response telemetry, and manages deployments to local Docker environments, Heroku, or AWS EC2 instances.

---

## Key Architecture

The deployment pipeline is built around a standardized prediction pipeline:
1. **Three-Artifact Pipeline Structure**: Models are composed of exactly three hardcoded pickle (`.pkl`) artifacts:
   - `preprocessing_pipeline.pkl`: Contains all standard scikit-learn transformers (imputers, scaling, one-hot encoders, category-encoders, etc.).
   - `feature_selector.pkl`: Handles selecting the relevant feature set (e.g., `SelectPercentile`).
   - `final_model.pkl`: The final trained model estimator (e.g., `RandomForestClassifier`).
2. **Path Resolution**: The model configuration metadata schema is defined inside the training script and written to `config.json`. The `ModelService` looks up config settings and relative model files dynamically based on the active registry root.
3. **Import Isolation**: Test and server modules resolve dependencies cleanly using isolated namespace mapping.

---

## Detailed `config.json` Explanation

The service runtime and schema validation logic are defined entirely by the `config.json` configuration file. Below is an annotated structure showing every field's purpose:

```json
{
  "models": {
    "preprocessing_pipeline": "models/preprocessing_pipeline.pkl",
    "feature_selector": "models/feature_selector.pkl",
    "final_model": "models/final_model.pkl"
  },
  "schema": {
    "features": [
      {
        "name": "alcohol",
        "type": "number",
        "required": true,
        "min": 1.0,
        "max": 15.0
      },
      {
        "name": "origin",
        "type": "string",
        "required": false,
        "allowed_values": ["domestic", "imported"]
      }
    ]
  },
  "tracking": {
    "enabled": true,
    "path": "prediction_logs/predictions.jsonl"
  }
}
```

### 1. `models` Object
Defines relative file system paths to the serialized pipeline artifacts. Paths must be relative to the `.accelera_deployment/` directory so they can be copied correctly during containerization.
* **`preprocessing_pipeline`** *(string)*: Path to the preprocessor pipeline.
* **`feature_selector`** *(string)*: Path to the feature selection model.
* **`final_model`** *(string)*: Path to the core estimator model.

### 2. `schema` Object
Specifies validation constraints for incoming payloads. If `schema` contains a non-empty `features` list, the validation engine runs Great Expectations suites on every prediction request.
* **`features`** *(array of objects)*: List of expected input fields. Each feature object supports:
  * **`name`** *(string, required)*: The exact key name expected in JSON requests or the column name in uploaded CSVs.
  * **`type`** *(string, default: `"number"`)*: Datatype coercion target. Supported options:
    * `"number"`: Coerces values into float.
    * `"integer"`: Coerces values into integer and raises a validation error if any decimal value is found.
    * `"string"`: Coerces values into string representations.
    * `"boolean"`: Auto-coerces boolean values and representations (e.g. `True`, `False`, `"true"`, `"false"`, `1`, `0`) to python boolean states.
  * **`required`** *(boolean, default: `true`)*: If true, returns a validation error if the field is null or missing.
  * **`min`** / **`max`** *(numeric, optional)*: Declares numeric lower and upper bounds.
  * **`allowed_values`** *(array, optional)*: Explicit list of allowed values (e.g., `["domestic", "imported"]`).

### 3. `tracking` Object
Controls prediction request logging.
* **`enabled`** *(boolean)*: Toggles telemetry logging.
* **`path`** *(string)*: Path to log predictions. Logs are stored as a line-separated JSON file (JSONL).

---

## Model Version Control (VCS) CLI Reference

The `vcs.py` module manages snapshots of models and active configuration files. Snapshots are stored under `experiments/<commit-hash>/` using a 7-character commit SHA key.

All commands below should be prefixed with `PYTHONPATH=.` (or `PYTHONPATH=/home/mazen/Desktop/GP/Accelera` when running from a custom directory).

| Command | Usage Example | Description |
| --- | --- | --- |
| **`init`** | `python -m accelera.src.deployment.vcs init` | Initializes the `.accelera_deployment` directory, establishing the default `experiments/` registry and creating an empty `experiments.json` configuration file. |
| **`commit`** | `python -m accelera.src.deployment.vcs commit -m "message"` | Snapshots the active model binaries and `config.json`, calculates the commit SHA, and saves the commit details under the `experiments/` registry. |
| **`status`** | `python -m accelera.src.deployment.vcs status` | Displays the status of the local registry, showing the total number of commits, the current `HEAD` commit, and which commit is marked as currently deployed. |
| **`log`** | `python -m accelera.src.deployment.vcs log` | Lists all committed model snapshots in reverse chronological order, including hashes, creation dates, messages, and deployment flags. |
| **`show`** | `python -m accelera.src.deployment.vcs show <hash>` | Outputs detailed information for a specific commit hash, including its full config, metadata, and files included. |
| **`deploy`** | `python -m accelera.src.deployment.vcs deploy <hash>` | Roll back or deploy a historical model snapshot. Copies the corresponding files and configuration back into the active runtime workspace. |

---

## Deployment Orchestration CLI Reference

The `deployment.py` module coordinates the building of Docker images and deployment to local and remote hosts.

### 1. Local & Docker Commands

| Command | Usage Example | Description |
| --- | --- | --- |
| **`prepare`** | `python accelera/src/deployment/deployment.py prepare` | Staging step. Copies project dependencies, serving scripts, and serialized models to a temporary staging workspace, and compiles the `Dockerfile` and `requirements.txt`. |
| **`build`** | `python accelera/src/deployment/deployment.py build [--no-cache]` | Compiles a Docker image named `ml-model` from the staging workspace. Use `--no-cache` to force rebuilds from scratch. |
| **`run-local`** | `python accelera/src/deployment/deployment.py run-local` | Exposes the service locally on a target port. Stops any existing Docker container using the port and runs the container. Read port using the `PORT` env variable (default: `8000`). |
| **`local`** | `python accelera/src/deployment/deployment.py local [--no-cache]` | Combines the `prepare`, `build`, and `run-local` steps in sequence for a complete local run. |

### 2. Heroku Deployment Commands

Heroku deployment coordinates pushed Docker images to Heroku's Container Registry. Pass `--app <app_name>` to specify the target application.

| Command | Usage Example | Description |
| --- | --- | --- |
| **`heroku-login`** | `python accelera/src/deployment/deployment.py heroku-login` | Triggers the interactive Heroku CLI authentication workflow. |
| **`heroku-create`** | `python accelera/src/deployment/deployment.py heroku-create --app my-app` | Creates a new Heroku app under the specified name with container deployment stack configuration. |
| **`heroku-container-login`** | `python accelera/src/deployment/deployment.py heroku-container-login` | Logs Docker CLI into the Heroku Container Registry (`registry.heroku.com`). |
| **`heroku-push`** | `python accelera/src/deployment/deployment.py heroku-push --app my-app` | Prepares the staging folder, builds the Docker container, and pushes it to the Heroku registry. |
| **`heroku-release`** | `python accelera/src/deployment/deployment.py heroku-release --app my-app` | Activates and releases the pushed web container to serve traffic on Heroku. |
| **`heroku-open`** | `python accelera/src/deployment/deployment.py heroku-open --app my-app` | Launches the default browser pointing to the active Heroku application. |
| **`heroku-deploy`** | `python accelera/src/deployment/deployment.py heroku-deploy --app my-app [--create]` | Sequentially logs in, builds, pushes, releases, and opens the Heroku app. Use `--create` to create it first. |

### 3. AWS EC2 Deployment Commands

Orchestrates syncing files, building Docker, and running containers on remote EC2 instances over SSH.

| Argument / Subcommand | Usage Example | Description |
| --- | --- | --- |
| **`ec2-deploy`** | `python accelera/src/deployment/deployment.py ec2-deploy --host 54.12.34.56 --key key.pem` | Main EC2 deployment script. Syncs files via `rsync`, installs Docker if missing (if `--install-docker` is passed), builds the container on the target host, and boots it. |
| **`ec2-stop`** | `python accelera/src/deployment/deployment.py ec2-stop --host 54.12.34.56 --key key.pem` | Stops and removes the active model container running on the EC2 host. |
| **`ec2-logs`** | `python accelera/src/deployment/deployment.py ec2-logs --host 54.12.34.56 --key key.pem` | Follows and displays real-time container log output on the remote EC2 instance. |

#### Supported Arguments for EC2 Commands:
* `--host` *(required)*: Target public DNS name or IP address of the EC2 instance.
* `--user` *(default: `ec2-user`)*: SSH login username (e.g. `ubuntu`, `admin`).
* `--key` *(optional)*: Local path to the private SSH key file (`.pem`).
* `--port` *(default: `8000`)*: Public port to publish and expose on the remote instance.
* `--remote-dir` *(default: `~/deployment-app`)*: Server path where the code and staging files are uploaded.
* `--image` / `--container` *(default: `ml-model`)*: Override names for the built image and running container.
* `--install-docker`: If present, attempts to automatically install and start the Docker daemon on the EC2 instance using remote packet managers.

## Complete E2E Example Script

For a complete, automated demonstration of the entire pipeline, see the example script:
```bash
python examples/deployment_demo.py
```
This script acts as a reference example that automates downloading the required SSH credentials, initializing the VCS registry, training a classification pipeline, committing the artifacts, and executing an AWS EC2 container deployment.

---

## Setup & Running from a Custom Directory

To run deployment workflows from a custom subdirectory (e.g. `deployment_test/`), you must set the `PYTHONPATH` environment variable so Python can find the `accelera` library.

### 1. Initialize the VCS Registry
Create the `.accelera_deployment` workspace in your current folder:
```bash
PYTHONPATH=/home/mazen/Desktop/GP/Accelera python -m accelera.src.deployment.vcs init
```

### 2. Generate the Config and Models
Ensure `config.json` and model pickle files are generated by running the model training script:
```bash
PYTHONPATH=/home/mazen/Desktop/GP/Accelera python /home/mazen/Desktop/GP/Accelera/examples/model.py
```
This writes `config.json` and the models to `models/` under the active directory.

### 3. Commit the Model to VCS
```bash
PYTHONPATH=/home/mazen/Desktop/GP/Accelera python -m accelera.src.deployment.vcs commit -m "Your commit message"
```

### 4. Build and Run the Container Service
Automatically stage, build, and run the container locally:
```bash
PYTHONPATH=/home/mazen/Desktop/GP/Accelera python /home/mazen/Desktop/GP/Accelera/accelera/src/deployment/deployment.py local
```

---

## Prediction API Reference

The serving application exposes the following endpoints:

| Endpoint | Method | Purpose |
| --- | --- | --- |
| `/health` | `GET` | Return service status, model path config, and tracking details. |
| `/gui` | `GET` | Render an interactive form for predictions based on the input schema. |
| `/tracking/summary` | `GET` | Retrieve prediction tracking aggregates (total calls, successful, failed). |
| `/predict` | `POST` | Input list of JSON objects to perform batch or single predictions. |
| `/predict/csv` | `POST` | Upload a multipart CSV file to perform batch predictions. |

### JSON Inference Call Example
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"input": [[13.71, 1.86, 2.36, 16.6, 101.0, 2.61, 2.88, 0.27, 1.69, 3.8, 1.11, 4.0, 1035.0]]}'
```

---

## Testing the Module

To run all unit and integration tests for the deployment engine, run:
```bash
PYTHONPATH=accelera/src/deployment:accelera/src pytest accelera/src/deployment/tests
```
