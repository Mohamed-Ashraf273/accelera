import argparse
import json
import shutil
import subprocess
import tempfile
import time
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np

from accelera.src.config import config
from accelera.src.utils.parallelizer import Parallelizer
from accelera.src.utils.parallelizer import extract_features
from accelera.src.utils.parallelizer import extract_loops
from accelera.src.utils.parallelizer import vectorize_features
from accelera.src.utils.py2cpp_converter import py2cpp_converter

CPP_SUFFIXES = {".c", ".cc", ".cpp", ".cxx"}


def log(message: str, enabled: bool = True) -> None:
    if enabled:
        print(message, flush=True)


def install_classifier_cache(parallelizer: Parallelizer, verbose: bool) -> None:
    original_classify = parallelizer._classify
    cache = {}
    stats = {"calls": 0, "hits": 0}

    def cached_classify(embedding):
        key = embedding.tobytes()
        if key in cache:
            stats["hits"] += 1
            log(f"      classifier cache hit ({stats['hits']} hits)", verbose)
            ok, value = cache[key]
            if ok:
                return value
            raise value

        stats["calls"] += 1
        log(f"      classifier request #{stats['calls']}", verbose)
        try:
            prediction = original_classify(embedding)
        except Exception as exc:
            cache[key] = (False, exc)
            log(f"      classifier error: {exc}", verbose)
            raise
        cache[key] = (True, prediction)
        log(f"      classifier response: {prediction}", verbose)
        return prediction

    parallelizer._classify = cached_classify


def normalize_pragma(pragma: str) -> str:
    pragma = pragma.strip()
    if pragma.startswith("#pragma"):
        pragma = pragma.removeprefix("#pragma").strip()
    return " ".join(pragma.split())


def extract_pragmas(code: str) -> list[str]:
    return [
        normalize_pragma(line)
        for line in code.splitlines()
        if line.strip().startswith("#pragma omp")
    ]


def loop_value(loop, name: str):
    if isinstance(loop, dict):
        return loop[name]
    return getattr(loop, name)


def compile_cpp(
    source_path: Path, binary_path: Path, verbose: bool = True
) -> tuple[bool, str]:
    log(f"    compiling {source_path.name}", verbose)
    compiler = shutil.which("g++") or shutil.which("clang++")
    if not compiler:
        return False, "No C++ compiler found"

    cmd = [
        compiler,
        "-std=c++17",
        "-O2",
        "-fopenmp",
        "-x",
        "c++",
        str(source_path),
        "-o",
        str(binary_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return False, result.stderr.strip()
    return True, ""


def run_binary(
    binary_path: Path, repeats: int, run_timeout: int | None, verbose: bool = True
) -> tuple[bool, str, float]:
    durations = []
    output = None
    for repeat in range(repeats):
        log(f"    running {binary_path.name} ({repeat + 1}/{repeats})", verbose)
        start = time.perf_counter()
        try:
            result = subprocess.run(
                [str(binary_path)],
                capture_output=True,
                text=True,
                timeout=run_timeout,
            )
        except subprocess.TimeoutExpired:
            return False, f"Timed out after {run_timeout}s", 0.0
        durations.append(time.perf_counter() - start)
        if result.returncode != 0:
            return False, result.stderr.strip(), 0.0
        if output is None:
            output = result.stdout.strip()
        elif output != result.stdout.strip():
            return False, "Program output changed between runs", 0.0
    return True, output or "", sum(durations) / len(durations)


def source_to_cpp(source_path: Path) -> str:
    source = source_path.read_text(encoding="utf-8")
    if source_path.suffix == ".py":
        return py2cpp_converter(source)
    return source


def expected_gold_path(source_path: Path, gold_dir: Path) -> Path:
    suffix = ".c" if source_path.suffix == ".py" else source_path.suffix
    return gold_dir / f"parallelized_{source_path.stem}{suffix}"


def classify_loops(parallelizer: Parallelizer, cpp_code: str, verbose: bool) -> dict:
    loops = [
        loop for loop in extract_loops(cpp_code) if loop_value(loop, "type") == "for"
    ]
    log(f"    classifying {len(loops)} loops", verbose)
    correct = 0
    errors = []
    for index, loop in enumerate(loops, 1):
        log(f"      loop {index}/{len(loops)}", verbose)
        try:
            features = extract_features(loop_value(loop, "code"))
            embedding = vectorize_features(features)
            prediction = parallelizer._classify(embedding)
            correct += int(prediction != "none")
        except Exception as exc:
            errors.append(str(exc))
    return {"total": len(loops), "correct": correct, "errors": errors}


def evaluate_file(
    source_path: Path,
    gold_path: Path,
    parallelizer: Parallelizer,
    work_dir: Path,
    repeats: int,
    run_timeout: int,
    index: int,
    total: int,
    verbose: bool,
) -> dict:
    log(f"[{index}/{total}] evaluating {source_path.name}", verbose)
    file_work_dir = work_dir / source_path.stem
    file_work_dir.mkdir()
    copied_source = file_work_dir / source_path.name
    shutil.copy2(source_path, copied_source)

    log("  preparing baseline C++", verbose)
    baseline_cpp = source_to_cpp(source_path)
    baseline_cpp_path = file_work_dir / f"{source_path.stem}_baseline.cpp"
    baseline_cpp_path.write_text(baseline_cpp, encoding="utf-8")

    classification = classify_loops(parallelizer, baseline_cpp, verbose)

    baseline_binary = file_work_dir / "baseline.out"
    baseline_compile_ok, baseline_compile_error = compile_cpp(
        baseline_cpp_path, baseline_binary, verbose
    )
    baseline_run_ok = False
    baseline_output = ""
    baseline_latency_s = 0.0
    if baseline_compile_ok:
        baseline_run_ok, baseline_output, baseline_latency_s = run_binary(
            baseline_binary, repeats, run_timeout, verbose
        )

    parallelize_error = ""
    parallelize_latency_s = 0.0
    generated_path = file_work_dir / f"parallelized_{source_path.stem}.c"
    try:
        log("  running parallelizer", verbose)
        start = time.perf_counter()
        parallelizer.parallelize(str(copied_source))
    except Exception as exc:
        parallelize_error = str(exc)
    finally:
        parallelize_latency_s = time.perf_counter() - start

    generated_exists = generated_path.exists()
    generated_code = (
        generated_path.read_text(encoding="utf-8") if generated_exists else ""
    )
    gold_code = gold_path.read_text(encoding="utf-8")
    generated_pragmas = extract_pragmas(generated_code)
    gold_pragmas = extract_pragmas(gold_code)

    pragma_pairs = list(zip(generated_pragmas, gold_pragmas))
    pragma_exact = sum(1 for generated, gold in pragma_pairs if generated == gold)
    pragma_similarity = [
        SequenceMatcher(None, generated, gold).ratio()
        for generated, gold in pragma_pairs
    ]

    parallel_binary = file_work_dir / "parallel.out"
    parallel_compile_ok = False
    parallel_compile_error = ""
    parallel_run_ok = False
    parallel_output = ""
    parallel_latency_s = 0.0
    if generated_exists:
        parallel_compile_ok, parallel_compile_error = compile_cpp(
            generated_path, parallel_binary, verbose
        )
        if parallel_compile_ok:
            parallel_run_ok, parallel_output, parallel_latency_s = run_binary(
                parallel_binary, repeats, run_timeout, verbose
            )

    output_matches = (
        baseline_run_ok and parallel_run_ok and baseline_output == parallel_output
    )
    speedup = (
        baseline_latency_s / parallel_latency_s
        if baseline_latency_s > 0 and parallel_latency_s > 0
        else 0.0
    )

    return {
        "file": str(source_path),
        "gold_file": str(gold_path),
        "classification": classification,
        "parallelize_ok": parallelize_error == "",
        "parallelize_error": parallelize_error,
        "parallelize_latency_ms": parallelize_latency_s * 1000,
        "generated_pragmas": len(generated_pragmas),
        "gold_pragmas": len(gold_pragmas),
        "pragma_exact": pragma_exact,
        "pragma_similarity_avg": float(np.mean(pragma_similarity))
        if pragma_similarity
        else 0.0,
        "baseline_compile_ok": baseline_compile_ok,
        "baseline_compile_error": baseline_compile_error,
        "baseline_run_ok": baseline_run_ok,
        "parallel_compile_ok": parallel_compile_ok,
        "parallel_compile_error": parallel_compile_error,
        "parallel_run_ok": parallel_run_ok,
        "output_matches": output_matches,
        "baseline_latency_ms": baseline_latency_s * 1000,
        "parallel_latency_ms": parallel_latency_s * 1000,
        "speedup": speedup,
    }


def summarize(results: list[dict]) -> dict:
    total_files = len(results)
    classifier_total = sum(r["classification"]["total"] for r in results)
    classifier_correct = sum(r["classification"]["correct"] for r in results)
    gold_pragmas = sum(r["gold_pragmas"] for r in results)
    generated_pragmas = sum(r["generated_pragmas"] for r in results)
    exact_pragmas = sum(r["pragma_exact"] for r in results)

    runnable = [r for r in results if r["baseline_run_ok"] and r["parallel_run_ok"]]
    speedups = [r["speedup"] for r in runnable if r["speedup"] > 0]

    return {
        "files_total": total_files,
        "files_parallelized_without_error": sum(
            r["parallelize_ok"] for r in results
        ),
        "files_compiled_and_ran": sum(
            r["baseline_run_ok"] and r["parallel_run_ok"] for r in results
        ),
        "files_output_matches": sum(r["output_matches"] for r in results),
        "classifier_total_loops": classifier_total,
        "classifier_correct_loops": classifier_correct,
        "classifier_accuracy": classifier_correct / classifier_total
        if classifier_total
        else 0.0,
        "gold_pragmas": gold_pragmas,
        "generated_pragmas": generated_pragmas,
        "pragma_exact_matches": exact_pragmas,
        "pragma_exact_accuracy": exact_pragmas / gold_pragmas
        if gold_pragmas
        else 0.0,
        "pragma_similarity_avg": float(
            np.mean([r["pragma_similarity_avg"] for r in results])
        )
        if results
        else 0.0,
        "avg_parallelize_latency_ms": float(
            np.mean([r["parallelize_latency_ms"] for r in results])
        )
        if results
        else 0.0,
        "avg_baseline_latency_ms": float(
            np.mean([r["baseline_latency_ms"] for r in runnable])
        )
        if runnable
        else 0.0,
        "avg_parallel_latency_ms": float(
            np.mean([r["parallel_latency_ms"] for r in runnable])
        )
        if runnable
        else 0.0,
        "avg_speedup": float(np.mean(speedups)) if speedups else 0.0,
        "files_faster_after_parallelization": sum(
            r["speedup"] > 1.03 for r in runnable
        ),
        "files_slower_after_parallelization": sum(
            r["speedup"] < 0.97 for r in runnable
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-dir", default="data/test_files")
    parser.add_argument("--gold-dir", default="data/parallelized_files")
    parser.add_argument("--output", default="data/parallelizer_eval_results.json")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--request-timeout",
        type=int,
        default=None,
        help="Model request timeout in seconds. Use 0 to disable.",
    )
    parser.add_argument(
        "--run-timeout",
        type=int,
        default=10,
        help="Compiled binary timeout in seconds. Use 0 to disable.",
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    verbose = not args.quiet
    if args.request_timeout is not None:
        request_timeout = args.request_timeout or None
        object.__setattr__(config, "REQUEST_TIMEOUT_S", request_timeout)
        log(f"Using request timeout: {request_timeout or 'disabled'}", verbose)

    run_timeout = args.run_timeout or None
    log(f"Using run timeout: {run_timeout or 'disabled'}", verbose)

    test_dir = Path(args.test_dir)
    gold_dir = Path(args.gold_dir)
    sources = sorted(
        path
        for path in test_dir.iterdir()
        if path.suffix == ".py" or path.suffix in CPP_SUFFIXES
    )
    if args.limit:
        sources = sources[: args.limit]

    parallelizer = Parallelizer()
    install_classifier_cache(parallelizer, verbose)
    results = []
    log(f"Evaluating {len(sources)} files with {args.repeats} repeats each", verbose)
    with tempfile.TemporaryDirectory(prefix="parallelizer_eval_") as tmp:
        work_dir = Path(tmp)
        for index, source_path in enumerate(sources, 1):
            gold_path = expected_gold_path(source_path, gold_dir)
            if not gold_path.exists():
                raise FileNotFoundError(f"Missing gold file for {source_path}")
            results.append(
                evaluate_file(
                    source_path,
                    gold_path,
                    parallelizer,
                    work_dir,
                    args.repeats,
                    run_timeout,
                    index,
                    len(sources),
                    verbose,
                )
            )
            summary = summarize(results)
            log(
                "  progress: "
                f"{summary['files_parallelized_without_error']}/{index} "
                "parallelized, "
                f"{summary['files_compiled_and_ran']}/{index} ran",
                verbose,
            )

    payload = {"summary": summarize(results), "files": results}
    Path(args.output).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()
