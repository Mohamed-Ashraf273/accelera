import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from accelera.src.config import config
from accelera.src.utils.parallelizer import parallelizer


def run_command(command, cwd):
    start = time.perf_counter()
    result = subprocess.run(
        command,
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )
    elapsed = time.perf_counter() - start
    return {
        "command": " ".join(map(str, command)),
        "returncode": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
        "time_seconds": elapsed,
    }


def compile_and_run_c(source_path, cwd):
    compiler = shutil.which("g++") or shutil.which("clang++") or shutil.which("c++")
    if compiler is None:
        return {
            "command": "compile",
            "returncode": 1,
            "stdout": "",
            "stderr": "No C++ compiler found. Install g++, clang++, or c++.",
            "time_seconds": 0.0,
        }

    with tempfile.TemporaryDirectory() as tmpdir:
        exe_suffix = ".exe" if os.name == "nt" else ""
        output_path = Path(tmpdir) / f"parallelized_hard_eval_002{exe_suffix}"
        compile_cmd = [
            compiler,
            "-O3",
            "-std=c++17",
            "-fopenmp",
            str(source_path),
            "-o",
            str(output_path),
        ]
        compile_result = run_command(compile_cmd, cwd)
        if compile_result["returncode"] != 0:
            return compile_result

        run_result = run_command([str(output_path)], cwd)
        run_result["compile_time_seconds"] = compile_result["time_seconds"]
        return run_result


def print_result(title, result):
    print(f"\n[{title}]")
    print(f"Command: {result['command']}")
    print(f"Return code: {result['returncode']}")
    print(f"Time: {result['time_seconds']:.6f} s")
    if "compile_time_seconds" in result:
        print(f"Compile time: {result['compile_time_seconds']:.6f} s")
    print(f"Output: {result['stdout']}")
    if result["stderr"]:
        print(f"Errors:\n{result['stderr']}")


print("\n" + "=" * 80)
print("=== Parallelizer Demo ===")
print("=" * 80)

repo_root = Path(config.REPO_ROOT)
python_path = repo_root / "data" / "hard_test_files" / "hard_eval_002.py"
c_path = repo_root / "parallelized_hard_eval_002.c"

print(f"Parallelizing original Python code from {python_path}")
parallelized_code = parallelizer.parallelize(str(python_path))

if parallelized_code is not None:
    print("\nGenerated parallelized code preview:")
    print(parallelized_code)
else:
    print(f"Parallelized C code written to {c_path}")

python_result = run_command([sys.executable, str(python_path)], repo_root)
c_result = compile_and_run_c(c_path, repo_root)

print("\n" + "=" * 80)
print("=== Benchmark Results ===")
print("=" * 80)
print_result("Original hard_eval_002.py", python_result)
print_result("parallelized_hard_eval_002.c", c_result)
