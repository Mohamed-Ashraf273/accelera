import subprocess
from pathlib import Path

from pyngrok import ngrok

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "benchmark"

processes = []

try:
    subprocess.run(["npm", "install"], cwd=BENCHMARK / "backend", check=True)
    subprocess.run(["npm", "install"], cwd=BENCHMARK / "frontend", check=True)

    backend = subprocess.Popen(
        ["npm", "run", "dev"],
        cwd=BENCHMARK / "backend",
    )
    processes.append(backend)

    frontend = subprocess.Popen(
        ["npm", "run", "dev"],
        cwd=BENCHMARK / "frontend",
    )
    processes.append(frontend)

    public_url = ngrok.connect(5173)
    print(f"\nFrontend is available here: {public_url}\n")

    for p in processes:
        p.wait()

except KeyboardInterrupt:
    print("\nStopping servers...")

finally:
    ngrok.kill()

    for p in processes:
        p.terminate()
