import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from accelera.src.config import config
from accelera.src.parallelizer.parallelizer import parallelizer


def print_timing(label, elapsed):
    print(f"[timing] {label}: {elapsed:.6f} s")


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


def instrument_c_runtime(source_path):
    code = source_path.read_text()
    if "ACCELERA_RUNTIME_TIMER" in code:
        return source_path

    if "int main()" not in code:
        return source_path

    instrumented = code
    if "#include <chrono>" not in instrumented:
        instrumented = instrumented.replace(
            "#include <iostream>",
            "#include <iostream>\n#include <chrono>",
            1,
        )

    instrumented = instrumented.replace(
        "int main() {",
        (
            "int main() {\n"
            "    auto __accelera_timer_start = "
            "std::chrono::high_resolution_clock::now();"
        ),
        1,
    )
    instrumented = re.sub(
        r"(?m)^(\s*)return\s+0\s*;",
        (
            r"\1auto __accelera_timer_end = "
            "std::chrono::high_resolution_clock::now();\n"
            r"\1std::cerr << "
            '"ACCELERA_RUNTIME_TIMER main_body_seconds="'
            "\n"
            r"\1          << std::chrono::duration<double>("
            "__accelera_timer_end - __accelera_timer_start).count()\n"
            r"\1          << std::endl;\n"
            r"\1return 0;"
        ),
        instrumented,
        count=1,
    )

    instrumented_path = source_path.with_name(f"{source_path.stem}_timed.cpp")
    instrumented_path.write_text(instrumented)
    return instrumented_path


def compile_and_run_c(source_path, cwd):
    total_start = time.perf_counter()

    start = time.perf_counter()
    config.ensure_llvm_on_path()
    print_timing("ensure LLVM on PATH", time.perf_counter() - start)

    start = time.perf_counter()
    compiler = shutil.which("g++") or shutil.which("clang++") or shutil.which("c++")
    print_timing("find C++ compiler", time.perf_counter() - start)

    if compiler is None:
        return {
            "command": "compile",
            "returncode": 1,
            "stdout": "",
            "stderr": "No C++ compiler found. Install g++, clang++, or c++.",
            "time_seconds": 0.0,
        }

    print(f"[timing] compiler: {compiler}")

    start = time.perf_counter()
    benchmark_source_path = instrument_c_runtime(source_path)
    print_timing("prepare timed C++ source", time.perf_counter() - start)

    start = time.perf_counter()
    with tempfile.TemporaryDirectory() as tmpdir:
        print_timing("create temporary directory", time.perf_counter() - start)
        exe_suffix = ".exe" if os.name == "nt" else ""
        output_path = Path(tmpdir) / f"parallelized_hard_eval_002{exe_suffix}"
        compile_cmd = [
            compiler,
            "-O3",
            "-std=c++17",
            "-fopenmp",
            str(benchmark_source_path),
            "-o",
            str(output_path),
        ]

        print("[timing] compile command:", " ".join(map(str, compile_cmd)))
        start = time.perf_counter()
        compile_result = run_command(compile_cmd, cwd)
        print_timing("compile subprocess total", time.perf_counter() - start)
        if compile_result["returncode"] != 0:
            return compile_result

        start = time.perf_counter()
        run_result = run_command([str(output_path)], cwd)
        print_timing("executable subprocess total", time.perf_counter() - start)
        run_result["compile_time_seconds"] = compile_result["time_seconds"]
        run_result["total_c_path_seconds"] = time.perf_counter() - total_start
        print_timing("compile_and_run_c total", run_result["total_c_path_seconds"])
        return run_result


def print_result(title, result):
    print(f"\n[{title}]")
    print(f"Command: {result['command']}")
    print(f"Return code: {result['returncode']}")
    print(f"Time: {result['time_seconds']:.6f} s")
    if "compile_time_seconds" in result:
        print(f"Compile time: {result['compile_time_seconds']:.6f} s")
    if "total_c_path_seconds" in result:
        print(f"Total C path time: {result['total_c_path_seconds']:.6f} s")
    print(f"Output: {result['stdout']}")
    if result["stderr"]:
        print(f"Errors:\n{result['stderr']}")


print("\n" + "=" * 80)
print("File running: examples/code_parallelizer_demo.py")
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
