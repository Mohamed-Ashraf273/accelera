import argparse
import difflib
import json
import math
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from accelera.src.parallelizer.parallelizer import extract_loops
from accelera.src.parallelizer.parallelizer import parallelizer
from accelera.src.parallelizer.py2cpp_converter import py2cpp_converter

repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))


def _read_code_for_compile(path: Path) -> str:
    code = path.read_text()
    if path.suffix == ".py":
        return py2cpp_converter(code)
    return code


def _pragma_lines(code: str) -> list[str]:
    return [
        line.strip()
        for line in code.splitlines()
        if line.strip().startswith("#pragma omp")
    ]


def _pragma_stats(generated_code: str, gold_code: str) -> dict:
    generated = _pragma_lines(generated_code)
    gold = _pragma_lines(gold_code)
    exact = sum(1 for g, p in zip(gold, generated) if g == p)
    similarities = [
        difflib.SequenceMatcher(None, g, p).ratio() for g, p in zip(gold, generated)
    ]
    return {
        "generated_pragmas": len(generated),
        "gold_pragmas": len(gold),
        "pragma_exact": exact,
        "pragma_similarity_avg": (
            sum(similarities) / len(similarities) if similarities else 0.0
        ),
    }


def _compile_code(code: str, output_path: Path, label: str) -> tuple[bool, str]:
    print(f"  compiling {label}...", flush=True)
    source_path = output_path.with_suffix(".cpp")
    source_path.write_text(code)
    compiler = shutil.which("g++") or shutil.which("clang++") or "c++"
    cmd = [
        compiler,
        "-O3",
        "-std=c++17",
        "-fopenmp",
        str(source_path),
        "-o",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0, result.stderr


def _run_binary(
    binary_path: Path,
    runs: int,
    label: str,
) -> tuple[bool, str, float]:
    print(f"  running {label} ({runs} run(s))...", flush=True)
    output = ""
    latencies = []
    for _ in range(runs):
        start = time.perf_counter()
        result = subprocess.run(
            [str(binary_path)],
            capture_output=True,
            text=True,
        )
        latencies.append((time.perf_counter() - start) * 1000)
        if result.returncode != 0:
            return False, result.stderr, 0.0
        output = result.stdout.strip()
    return True, output, sum(latencies) / len(latencies)


def _compare_outputs(baseline_output: str, parallel_output: str) -> dict:
    if baseline_output.strip() == parallel_output.strip():
        return {
            "matches": True,
            "max_abs_error": 0.0,
            "max_rel_error": 0.0,
        }

    baseline_tokens = baseline_output.split()
    parallel_tokens = parallel_output.split()
    if len(baseline_tokens) != len(parallel_tokens):
        return {
            "matches": False,
            "max_abs_error": 0.0,
            "max_rel_error": 0.0,
        }

    max_abs_error = 0.0
    max_rel_error = 0.0
    for baseline_token, parallel_token in zip(baseline_tokens, parallel_tokens):
        try:
            baseline_value = float(baseline_token)
            parallel_value = float(parallel_token)
        except ValueError:
            if baseline_token != parallel_token:
                return {
                    "matches": False,
                    "max_abs_error": max_abs_error,
                    "max_rel_error": max_rel_error,
                }
            continue

        abs_error = abs(baseline_value - parallel_value)
        rel_error = abs_error / max(abs(baseline_value), 1e-12)
        max_abs_error = max(max_abs_error, abs_error)
        max_rel_error = max(max_rel_error, rel_error)
        if not math.isclose(
            baseline_value,
            parallel_value,
            rel_tol=1e-6,
            abs_tol=1e-6,
        ):
            return {
                "matches": False,
                "max_abs_error": max_abs_error,
                "max_rel_error": max_rel_error,
            }

    return {
        "matches": True,
        "max_abs_error": max_abs_error,
        "max_rel_error": max_rel_error,
    }


def _generated_output_path(source_path: Path) -> Path:
    return source_path.parent / f"parallelized_{source_path.stem}.c"


def _gold_path(source_path: Path, gold_dir: Path) -> Path:
    suffix = ".c" if source_path.suffix == ".py" else source_path.suffix
    return gold_dir / f"parallelized_{source_path.stem}{suffix}"


def _evaluate_file(
    source_path: Path,
    gold_dir: Path,
    runs: int,
    index: int,
    total: int,
) -> dict:
    print(f"[{index}/{total}] Evaluating {source_path}", flush=True)
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        temp_source = tmp_dir / source_path.name
        shutil.copy2(source_path, temp_source)

        start = time.perf_counter()
        parallelize_ok = True
        parallelize_error = ""
        print("  parallelizing...", flush=True)
        try:
            parallelizer.parallelize(str(temp_source))
        except Exception as exc:
            parallelize_ok = False
            parallelize_error = str(exc)
        parallelize_latency_ms = (time.perf_counter() - start) * 1000

        generated_path = _generated_output_path(temp_source)
        generated_code = (
            generated_path.read_text() if generated_path.exists() else ""
        )
        gold_file = _gold_path(source_path, gold_dir)
        gold_code = gold_file.read_text() if gold_file.exists() else ""

        pragma_stats = _pragma_stats(generated_code, gold_code)

        baseline_code = _read_code_for_compile(source_path)
        baseline_bin = tmp_dir / "baseline_bin"
        parallel_bin = tmp_dir / "parallel_bin"

        baseline_compile_ok, baseline_compile_error = _compile_code(
            baseline_code, baseline_bin, "baseline"
        )
        parallel_compile_ok = False
        parallel_compile_error = ""
        baseline_run_ok = False
        parallel_run_ok = False
        baseline_output = ""
        parallel_output = ""
        baseline_latency_ms = 0.0
        parallel_latency_ms = 0.0

        if generated_code:
            parallel_compile_ok, parallel_compile_error = _compile_code(
                generated_code, parallel_bin, "parallelized"
            )

        if baseline_compile_ok:
            baseline_run_ok, baseline_output, baseline_latency_ms = _run_binary(
                baseline_bin, runs, "baseline"
            )
        if parallel_compile_ok:
            parallel_run_ok, parallel_output, parallel_latency_ms = _run_binary(
                parallel_bin, runs, "parallelized"
            )

        output_comparison = _compare_outputs(baseline_output, parallel_output)
        output_matches = (
            baseline_run_ok and parallel_run_ok and output_comparison["matches"]
        )
        speedup = (
            baseline_latency_ms / parallel_latency_ms
            if output_matches and parallel_latency_ms > 0
            else 0.0
        )

        result = {
            "file": str(source_path),
            "gold_file": str(gold_file),
            "classification": {
                "total": len(extract_loops(baseline_code)),
                "correct": pragma_stats["pragma_exact"],
                "errors": [],
            },
            "parallelize_ok": parallelize_ok,
            "parallelize_error": parallelize_error,
            "parallelize_latency_ms": parallelize_latency_ms,
            **pragma_stats,
            "baseline_compile_ok": baseline_compile_ok,
            "baseline_compile_error": baseline_compile_error,
            "baseline_run_ok": baseline_run_ok,
            "parallel_compile_ok": parallel_compile_ok,
            "parallel_compile_error": parallel_compile_error,
            "parallel_run_ok": parallel_run_ok,
            "output_matches": output_matches,
            "baseline_output": baseline_output,
            "parallel_output": parallel_output,
            "max_abs_error": output_comparison["max_abs_error"],
            "max_rel_error": output_comparison["max_rel_error"],
            "baseline_latency_ms": baseline_latency_ms,
            "parallel_latency_ms": parallel_latency_ms,
            "speedup": speedup,
        }
        print(
            "  done: "
            f"parallelize_ok={parallelize_ok}, "
            f"output_matches={output_matches}, "
            f"speedup={speedup:.3f}",
            flush=True,
        )
        return result


def _summarize(files: list[dict]) -> dict:
    classifier_total = sum(f["classification"]["total"] for f in files)
    classifier_correct = sum(f["classification"]["correct"] for f in files)
    gold_pragmas = sum(f["gold_pragmas"] for f in files)
    generated_pragmas = sum(f["generated_pragmas"] for f in files)
    pragma_exact_matches = sum(f["pragma_exact"] for f in files)
    matching_files = [f for f in files if f["output_matches"]]

    return {
        "files_total": len(files),
        "files_parallelized_without_error": sum(f["parallelize_ok"] for f in files),
        "files_compiled_and_ran": sum(f["parallel_run_ok"] for f in files),
        "files_output_matches": len(matching_files),
        "classifier_total_loops": classifier_total,
        "classifier_correct_loops": classifier_correct,
        "classifier_accuracy": (
            classifier_correct / classifier_total if classifier_total else 0.0
        ),
        "gold_pragmas": gold_pragmas,
        "generated_pragmas": generated_pragmas,
        "pragma_exact_matches": pragma_exact_matches,
        "pragma_exact_accuracy": (
            pragma_exact_matches / gold_pragmas if gold_pragmas else 0.0
        ),
        "pragma_similarity_avg": (
            sum(f["pragma_similarity_avg"] for f in files) / len(files)
            if files
            else 0.0
        ),
        "avg_parallelize_latency_ms": (
            sum(f["parallelize_latency_ms"] for f in files) / len(files)
            if files
            else 0.0
        ),
        "avg_baseline_latency_ms": (
            sum(f["baseline_latency_ms"] for f in matching_files)
            / len(matching_files)
            if matching_files
            else 0.0
        ),
        "avg_parallel_latency_ms": (
            sum(f["parallel_latency_ms"] for f in matching_files)
            / len(matching_files)
            if matching_files
            else 0.0
        ),
        "avg_speedup": (
            sum(f["speedup"] for f in matching_files) / len(matching_files)
            if matching_files
            else 0.0
        ),
        "files_faster_after_parallelization": sum(
            f["output_matches"] and f["speedup"] > 1.0 for f in files
        ),
        "files_slower_after_parallelization": sum(
            f["output_matches"] and 0.0 < f["speedup"] < 1.0 for f in files
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        default="data/hard_test_files",
        type=Path,
    )
    parser.add_argument(
        "--gold-dir",
        default="data/hard_parallelized_files",
        type=Path,
    )
    parser.add_argument(
        "--output",
        default="hard_parallelizer_eval_results_generated.json",
        type=Path,
    )
    parser.add_argument("--runs", default=3, type=int)
    args = parser.parse_args()

    files = sorted(
        path
        for path in args.input_dir.iterdir()
        if path.suffix in {".c", ".cpp", ".py"}
    )
    results = [
        _evaluate_file(path, args.gold_dir, args.runs, i, len(files))
        for i, path in enumerate(files, start=1)
    ]
    report = {"summary": _summarize(results), "files": results}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
