import subprocess
from pathlib import Path

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

    for p in processes:
        p.wait()

except KeyboardInterrupt:
    print("\nStopping servers...")

finally:
    for p in processes:
        p.terminate()
