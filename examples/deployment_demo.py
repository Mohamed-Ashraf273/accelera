import os
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path
from types import SimpleNamespace

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from accelera.src.deployment import vcs
from examples.model import train_model

DRIVE_LINK = (
    "https://docs.google.com/uc?export=download&id=16-thQ5VnYDHUH5gvoY1tCP0n5Ovaw1fP"
)

print("=========== Accelera Deployment Demo Start ========")


original_cwd = os.getcwd()

demo_dir = Path(__file__).resolve().parent / "demo_deployment_env"
if demo_dir.exists():
    shutil.rmtree(demo_dir)
demo_dir.mkdir(parents=True, exist_ok=True)
(demo_dir / ".accelera_deployment").mkdir()

os.chdir(demo_dir)

with urllib.request.urlopen(DRIVE_LINK) as r, open("ssh.pem", "wb") as f:
    f.write(r.read())
os.chmod("ssh.pem", 0o600)

print("\n1: Initializing VCS")
vcs.init(SimpleNamespace())

print("\n2: Train ML Pipeline")
results = train_model()
print(f"Model trained {results['model']}")
print("pkl files:")
for name, path in results["deployment_config"]["models"].items():
    print(f"  - {name}: {path}")

print("\n3: Commit the model artifacts (for test)")
vcs.commit(SimpleNamespace(message="demo commit  wine model pipeline"))

print("\n4:  Testing VCS status and log...")
print(" VCS Status:")
vcs.status(SimpleNamespace())
print("\nVCS Log :")
vcs.log(SimpleNamespace())

print("\n5: Deploying models to EC2")
python_bin = sys.executable
deploy_script = (
    Path(__file__).resolve().parents[1] / "accelera/src/deployment/deployment.py"
)

cmd = [
    python_bin,
    str(deploy_script),
    "ec2-deploy",
    "--host",
    "16.171.151.79",
    "--user",
    "ubuntu",
    "--key",
    str(demo_dir / "ssh.pem"),
    "--install-docker",
]
subprocess.run(cmd, check=True)

os.chdir(original_cwd)
if demo_dir.exists():
    shutil.rmtree(demo_dir)
    print(f"\nClean up demo environment{demo_dir}")

print("\n==== Accelera deployment demo End =========")
