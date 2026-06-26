import os
import pickle
import subprocess
import sys
import urllib.request
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder

from accelera.src.accelera_pipe.core.pipeline import Pipeline
from accelera.src.e2e.tabular.e2e import E2E

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.chdir(PROJECT_ROOT)


def get_random_row(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        raise ValueError("DataFrame is empty")
    df_features = df.drop(columns=["price"], errors="ignore")
    return df_features.sample(n=1)


config = {"target_col": "price"}

accpipe = Pipeline()
accpipe.preprocess(
    "p1",
    OneHotEncoder(),
    columns=[
        "mainroad",
        "guestroom",
        "basement",
        "hotwaterheating",
        "airconditioning",
        "prefarea",
        "furnishingstatus",
    ],
).model("lr", LinearRegression())

e2e = E2E()
predictions, executed_graph = e2e(
    graph=accpipe,
    content="https://drive.google.com/uc?id=1VMtLcWDcigwkimpf-eWVMZ7zJMUf7wxs",
    config=config,
)


with open("pipeline.pkl", "rb") as f:
    pipeline = pickle.load(f)

print(pipeline.predict(get_random_row(e2e.df)))


print("\nDeploying e2e pipeline to EC2")
drive_link = (
    "https://docs.google.com/uc?export=download&id=16-thQ5VnYDHUH5gvoY1tCP0n5Ovaw1fP"
)
deployment_root = PROJECT_ROOT / ".accelera_deployment"
deployment_root.mkdir(exist_ok=True)
ssh_key_path = deployment_root / "ssh.pem"

with urllib.request.urlopen(drive_link) as r, open(ssh_key_path, "wb") as f:
    f.write(r.read())
os.chmod(ssh_key_path, 0o600)

python_bin = sys.executable
deploy_script = (
    Path(__file__).resolve().parents[1] / "accelera/src/deployment/deployment.py"
)

cmd = [
    python_bin,
    str(deploy_script),
    "ec2-deploy",
    "--host",
    "13.60.235.233",
    "--user",
    "ubuntu",
    "--key",
    str(ssh_key_path),
    "--install-docker",
    "--graph-runtime",
]
subprocess.run(cmd, check=True)

print("\nEC2 container logs")
logs_cmd = [
    "ssh",
    "-o",
    "StrictHostKeyChecking=accept-new",
    "-i",
    str(ssh_key_path),
    "ubuntu@13.60.235.233",
    "sudo docker logs --tail 200 ml-model",
]
subprocess.run(logs_cmd, check=True)
