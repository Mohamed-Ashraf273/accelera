# Current Code Parallelizer Generator Report

## Purpose

This report documents the current code parallelizer generator while the next model
training run is in progress. It explains the working pipeline, the Python-to-C++
conversion path, the validation steps around generated OpenMP pragmas, and the
offline evaluation setup added under `data/`.

## Current Pipeline

The current generator is orchestrated by `accelera/src/utils/parallelizer.py`.
Its job is to take a C, C++, or Python source file, identify loops, classify
whether each loop should be parallelized, generate an OpenMP pragma for selected
loops, and write a new parallelized C/C++ output file.

The high-level flow is:

1. Read the input file.
2. If the input is Python, convert it to C++ using `py2cpp_converter`.
3. Extract loops from the C/C++ source using the Clang-backed loop extractor.
4. For each `for` loop, calculate hand-written static features.
5. Vectorize those features into the classifier embedding.
6. Send the embedding to the classifier endpoint.
7. If the classifier says the loop is parallelizable, send the loop and class to
   the generator endpoint.
8. Validate the generated pragma.
9. Insert the pragma directly before the loop.
10. Write the generated file as `parallelized_<input_stem>.c`.
11. If `clang-format` is installed, format the generated output file.

## Python Input Path

Python files cannot be analyzed directly by the Clang loop extractor, so the
parallelizer first converts supported Python syntax to C++.

The converter is intentionally small and deterministic. It supports common
constructs such as:

- simple variable assignments
- `for` loops over `range(...)`
- arithmetic expressions
- basic `if` statements
- simple function definitions
- `print(...)`, emitted as `std::cout`

For example:

```python
n = 10
sum = 0

for i in range(n):
    sum += i

print(sum)
```

is converted into C++ shaped like:

```cpp
#include <iostream>

int main() {
    int n = 10;
    int sum = 0;
    for (int i = 0; i < n; i++) {
        sum += i;
    }
    (std::cout << sum << std::endl);
    return 0;
}
```

After conversion, Python and C++ inputs use the same loop extraction,
classification, generation, validation, insertion, formatting, compile, and run
evaluation path.

If Python conversion fails because unsupported syntax appears, the parallelizer
cannot safely continue. In evaluation, such files should be counted as conversion
failures rather than generator failures.

## Current Generator Behavior

The generator endpoint receives:

- `code_snippet`: the extracted loop source
- `cls`: the classifier output class
- `max_len`: configured maximum generation length

It returns a pragma string. The local code then normalizes it so it starts with
`#pragma`, balances simple brackets, and removes unsafe `num_threads(...)`
clauses when the generated thread expression references a name that does not
appear in the surrounding code or loop.

Example bad generation:

```cpp
#pragma omp parallel for num_threads(t) reduction(+ : sum)
for (int i = 0; i < 5; i++) {
    sum += i;
}
```

If `t` does not appear in the source code or loop, validation removes only the
invalid clause:

```cpp
#pragma omp parallel for reduction(+ : sum)
for (int i = 0; i < 5; i++) {
    sum += i;
}
```

This avoids compile failures from hallucinated thread-count variables while
preserving useful clauses such as `reduction`.

## Gold Evaluation Corpus

Two new directories hold the offline evaluation corpus:

- `data/test_files`
- `data/parallelized_files`

`data/test_files` contains 400 source files:

- 200 Python files
- 200 C++ files

`data/parallelized_files` contains the corresponding gold outputs. Every loop in
these gold files has an expected OpenMP pragma. Python source files map to
parallelized C++ gold files with a `.c` suffix, matching the current
parallelizer output convention.

Example mapping:

```text
data/test_files/eval_001.py
data/parallelized_files/parallelized_eval_001.c

data/test_files/eval_002.cpp
data/parallelized_files/parallelized_eval_002.cpp
```

The corpus focuses on loops that are intended to be valid parallelization
targets:

- independent array initialization
- independent array transforms
- scalar sum reductions
- dot-product reductions
- simple nested loops with `collapse(2)`
- static scheduling examples
- fixed numeric `num_threads(...)` examples

## Evaluation Script

The evaluator lives at:

```text
tools/evaluate_parallelizer_generator.py
```

Run it from the repository root:

```bash
python3 tools/evaluate_parallelizer_generator.py
```

Useful options:

```bash
python3 tools/evaluate_parallelizer_generator.py --limit 20
python3 tools/evaluate_parallelizer_generator.py --repeats 10
python3 tools/evaluate_parallelizer_generator.py --request-timeout 2
python3 tools/evaluate_parallelizer_generator.py --output data/eval_results.json
```

The script expects the classifier and generator endpoints configured in
`accelera/src/config.py` to be running. If the model service is offline,
parallelization metrics will report errors for those files.

## Latest Evaluation Results

The current measured report is stored at:

```text
data/parallelizer_eval_results.json
```

This run evaluated the first 20 files from the 400-file corpus. The results
should be treated as a smoke/evaluation slice, not the final full-corpus result.

| Metric | Value | How it is measured |
| --- | ---: | --- |
| Files evaluated | 20 | Count of file entries in `data/parallelizer_eval_results.json`. |
| Baseline files compiled | 20 / 20 | The evaluator converts Python files to C++ when needed, compiles the serial baseline with `g++` or `clang++`, and counts successful compiler exits. |
| Baseline files ran | 20 / 20 | Count of baseline binaries that executed successfully for the configured repeat count. |
| Files parallelized without endpoint/script error | 20 / 20 | Count of files where `parallelizer.parallelize(...)` completed without raising an exception. |
| Files where generated output compiled and ran | 20 / 20 | Count of files where both the baseline and generated parallelized binary compiled and executed successfully. |
| Files with matching serial/parallel output | 20 / 20 | Count of files where baseline output equals generated parallel output after both programs run. |
| Classifier loop accuracy | 40 / 50 = 80.0% | Every loop in this corpus is expected to be parallelizable. A classifier prediction is counted correct when it is not `none`. |
| Gold pragmas | 50 | Number of expected `#pragma omp ...` lines extracted from the gold files. |
| Generated pragmas | 40 | Number of `#pragma omp ...` lines extracted from the generated files. |
| Exact pragma accuracy | 20 / 50 = 40.0% | Generated and gold pragmas are normalized for whitespace and compared in order. Exact semantic alternatives still count as mismatches if the strings differ. |
| Average pragma similarity | 89.02% | For each generated/gold pragma pair, the evaluator computes `difflib.SequenceMatcher(...).ratio()` and averages the per-file scores. |
| Average parallelizer latency | 238.74 ms/file | Wall-clock time spent inside `parallelizer.parallelize(...)`, including local feature extraction and classifier requests. Compile and binary runtime are excluded. |
| Average baseline runtime | 1.25 ms | Runtime of the serial baseline, averaged only over files where both baseline and generated binaries ran successfully. |
| Average parallel runtime | 2.01 ms | Runtime of the generated parallel binary, averaged over the same comparable files. |
| Average speedup | 0.64x | Computed as `baseline_latency_ms / parallel_latency_ms` for files where both binaries run. Values below 1.0 mean the parallel version is slower. |
| Faster after parallelization | 0 | Count of comparable files with speedup greater than `1.03`. |
| Slower after parallelization | 20 | Count of comparable files with speedup less than `0.97`. |

### Latest Result Interpretation

The classifier predicted a parallelizable class for 40 of 50 loops, which is a
reasonable positive-class signal for this smoke slice. After switching pragma
construction to deterministic rules and adding conservative safety checks, all
20 generated files now compile, run, and match the serial baseline.

However, all 20 comparable files are slower after parallelization. Average
runtime changed from 1.25 ms serial to 2.01 ms parallel, giving an average
speedup of only 0.64x. In other words, the generated code is correct on this
slice, but OpenMP is not beneficial for these workloads.

This does not mean OpenMP is ineffective in general. These loops are intentionally
small and not computationally intense. For loops that finish in around 1-2 ms,
OpenMP overhead can dominate the actual work:

- thread scheduling
- reduction setup and merge
- runtime bookkeeping
- synchronization

For this non-hard corpus, using parallelization is usually not a good latency
choice. It is still useful for testing classifier and pragma correctness, but it
is not a fair performance benchmark for speedup.

## Hard Workload Evaluation Results

To measure whether parallelization can improve real runtime, a second small
corpus was added:

```text
data/hard_test_files
data/hard_parallelized_files
```

This corpus contains three heavier files:

- 2 C++ files with large arithmetic/vector loops
- 1 Python file with large reduction loops, converted to C++ before evaluation

The latest hard-workload run produced:

| Metric | Value | How it is measured |
| --- | ---: | --- |
| Files evaluated | 3 | Count of files in the hard-workload run. |
| Files parallelized without endpoint/script error | 3 / 3 | Count of files where `parallelizer.parallelize(...)` completed without raising an exception. |
| Files where generated output compiled and ran | 2 / 3 | Count of files where both the serial baseline and generated parallelized binary compiled and executed successfully. |
| Files with matching serial/parallel output | 2 / 3 | Count of files where generated parallel output matched the serial baseline output. |
| Classifier loop accuracy | 5 / 11 = 45.45% | Every hard-workload loop is expected to be parallelizable. A prediction is counted correct when it is not `none`. |
| Gold pragmas | 8 | Number of expected `#pragma omp ...` lines in the hard gold files. |
| Generated pragmas | 4 | Number of generated `#pragma omp ...` lines extracted from generated outputs. |
| Exact pragma accuracy | 3 / 8 = 37.5% | Generated and gold pragmas are normalized for whitespace and compared in order. |
| Average pragma similarity | 90.91% | Average `difflib.SequenceMatcher(...).ratio()` between generated and gold pragma pairs. |
| Average parallelizer latency | 210.66 ms/file | Wall-clock time inside `parallelizer.parallelize(...)`, including local feature extraction and classifier requests. |
| Average baseline runtime | 55.24 ms | Runtime of the serial baseline, averaged only over files where both baseline and generated binaries ran successfully. |
| Average parallel runtime | 22.54 ms | Runtime of the generated parallel binary, averaged over the same comparable files. |
| Average speedup | 3.33x | Computed as `baseline_latency_ms / parallel_latency_ms` for comparable files. |
| Faster after parallelization | 2 | Count of comparable files with speedup greater than `1.03`. |
| Slower after parallelization | 0 | Count of comparable files with speedup less than `0.97`. |

### Hard Workload Interpretation

The hard workload confirms that useful speedup is possible when loops are large
enough to amortize OpenMP overhead. Two of the three hard files compiled, ran,
matched the serial baseline, and improved runtime:

- `hard_eval_001.cpp`: 65.66 ms baseline, 35.81 ms parallel, 1.83x speedup
- `hard_eval_002.py`: 44.82 ms baseline, 9.27 ms parallel, 4.84x speedup

The remaining file, `hard_eval_000.cpp`, compiled but did not complete the full
run-and-match comparison. The hard run therefore shows clear performance
potential, but also shows that loop selection still needs work:

- classifier recall remains low: only 5 of 11 parallelizable loops were
  classified as parallelizable
- pragma coverage is partial: only 4 of 8 expected pragmas were generated
- exact pragma correctness is 3 of 8 expected pragmas

The important conclusion is:

```text
Small-loop evaluation now shows correctness, but OpenMP overhead dominates.
Hard-loop evaluation shows real speedup when selected loops are suitable.
```

## Metrics Measured

The evaluator writes a JSON report with both a `summary` section and per-file
details.

### Classifier Correctness

All 400 gold files contain parallelizable loops. For this corpus, classifier
correctness is measured as:

```text
classifier_correct_loops / classifier_total_loops
```

A loop is counted as correct when the classifier returns anything other than
`none`.

This is a positive-class stress test. It does not measure false positives on
non-parallelizable loops.

### Pragma Accuracy

The script extracts `#pragma omp ...` lines from the generated file and gold
file, then compares them in order.

It reports:

- total generated pragmas
- total gold pragmas
- exact pragma matches
- exact pragma accuracy
- average string similarity for near-miss analysis

Exact match is strict. For example, these may be semantically close but not exact:

```cpp
#pragma omp parallel for reduction(+:sum)
#pragma omp parallel for reduction(+ : sum)
```

The similarity score helps identify those near misses.

### Functional Correctness

For every file, the evaluator compiles and runs:

- the serial baseline C++ code
- the generated parallelized C++ code

For Python files, the baseline is the Python source converted to C++ first, so
the timing comparison stays within compiled C++ execution.

A file is counted as functionally correct when:

1. baseline compile succeeds
2. baseline run succeeds
3. generated compile succeeds
4. generated run succeeds
5. baseline output equals generated output

The summary field is:

```text
files_output_matches
```

### End-to-End Stability

The script reports:

```text
files_compiled_and_ran
```

This answers: out of 400 files, how many generated outputs compiled and ran
without runtime failure?

### Latency Before and After Parallelization

For each file, the evaluator runs the serial baseline and parallelized binary
multiple times and records average runtime.

Summary fields:

```text
avg_baseline_latency_ms
avg_parallel_latency_ms
avg_speedup
files_faster_after_parallelization
files_slower_after_parallelization
```

This helps distinguish useful parallelization from overhead. For small loops,
OpenMP startup costs can make parallel code slower even when the pragma is
technically correct. This is expected and should be treated as a performance
signal, not necessarily a correctness failure.

### Parallelizer Latency

The script also measures the time spent by the parallelizer itself:

```text
avg_parallelize_latency_ms
```

This includes local feature extraction and remote classifier/generator requests.
It does not include compile time.

## Interpretation Guide

Strong current-generator results should look like:

- high classifier accuracy on this positive-class corpus
- generated pragma count close to gold pragma count
- high exact pragma accuracy or high similarity
- most files compile and run
- most outputs match
- parallel latency not consistently worse than baseline

If pragma accuracy is high but runtime is slower, the generator is likely
semantically correct but over-parallelizing small workloads.

If generated outputs fail to compile, inspect:

- hallucinated variables in clauses
- invalid `private(...)` or `reduction(...)` variables
- malformed parentheses
- pragmas inserted at wrong locations

If outputs compile but differ, inspect:

- missing reductions
- unsafe shared scalar writes
- loop-carried dependencies
- incorrect `collapse(...)`

## Current Limitations

This evaluation corpus is intentionally controlled. It gives a stable target for
checking progress while training finishes, but it is not a complete benchmark.

Known limitations:

- no negative-class false-positive measurement
- small loop bodies may exaggerate OpenMP overhead
- Python coverage is limited to syntax supported by `py2cpp_converter`
- exact pragma matching is stricter than semantic equivalence
- generated `.c` files are compiled as C++ in the evaluator because the current
  parallelizer output uses C++ constructs such as `#include <iostream>`

## Recommended Next Metrics

After the new model finishes training, add:

- a negative corpus with loops that must not be parallelized
- larger workloads for stable performance measurements
- semantic pragma normalization for `reduction(+:sum)` versus
  `reduction(+ : sum)`
- compile diagnostics grouped by failure type
- confusion matrix for classifier output classes
- per-pragma-clause precision and recall
