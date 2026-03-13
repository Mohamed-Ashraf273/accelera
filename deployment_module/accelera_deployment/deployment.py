import json
import os
import subprocess

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(project_root)

with open("config.json", "r", encoding="utf-8") as f:
    configurations = json.load(f)

with open("accelera_deployment/requirements.txt", "w") as req:
    req.write("fastapi\n")
    req.write("uvicorn[standard]\n")
    req.write("scikit-learn\n")
    req.write("numpy\n")
    req.write("pandas\n")
    req.write("pydantic\n")
    req.write("python-multipart\n")

with open("Dockerfile", "w") as f:
    f.write("FROM python:3.11-slim\n")
    f.write("WORKDIR /app\n")
    f.write("COPY accelera_deployment/requirements.txt requirements.txt \n")
    f.write("COPY accelera_deployment/server.py server.py\n")
    f.write("COPY accelera_deployment/modelservice.py modelservice.py\n")
    f.write("COPY config.json config.json\n")
    for pkl in configurations["models"].values():
        if os.path.isfile(pkl):
            f.write(f"COPY {pkl} /app/{pkl}\n")
    f.write("RUN pip install --no-cache-dir -r requirements.txt\n")
    f.write("EXPOSE 8000\n")
    f.write("CMD uvicorn server:app --host 0.0.0.0 --port ${PORT:-8000}\n")

print("Dockerfile written successfully")

subprocess.run("docker build --no-cache -t ml-model .", shell=True, check=True)

subprocess.run(
    "docker ps -q --filter publish=8000 | xargs -r docker stop", shell=True
)

print("\n--- Starting container  ---\n")
print("API available at http://localhost:${PORT:-8000}\n")
subprocess.run(
    "docker run --rm -p ${PORT:-8000}:${PORT:-8000} ml-model", shell=True, check=True
)


# subprocess.run("heroku login", shell=True, check=True)
# subprocess.run("heroku create  accelera1 --stack container", shell=True, check=True)
# subprocess.run("heroku container:login", shell=True, check=True)
# subprocess.run("heroku container:push web --app accelera1", shell=True, check=True)
# subprocess.run("heroku container:release web --app accelera1", shell=True, check=True)
# subprocess.run("heroku open --app accelera1", shell=True, check=True)
