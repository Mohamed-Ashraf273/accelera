import json
import os
import re
from pathlib import Path

import numpy as np
import requests

from accelera.src.config import config
from accelera.src.utils.py2cpp_converter import py2cpp_converter

try:
    from code_parallelizer_utils import extract_loops as _extract_loops
    from code_parallelizer_utils import write_loops_to_json as _write_loops_to_json
except ImportError:
    _extract_loops = None
    _write_loops_to_json = None


def extract_loops(code: str, clang_args: list | None = None) -> list:
    if clang_args is None:
        clang_args = list(config.DEFAULT_CLANG_ARGS)
    return _extract_loops(code, clang_args)


def write_loops_to_json(loops: list, output_json: str) -> bool:
    return _write_loops_to_json(loops, output_json)


def pragma_to_class(label: str, pragma: str) -> str:
    if label == "False":
        return "none"
    if re.search(r"reduction\s*\(", pragma):
        return "reduction"
    return "parallel_for"


def extract_features(code: str) -> dict:
    features = {
        "has_reduction_plus": False,
        "has_reduction_mul": False,
        "has_reduction_general": False,
        "reduction_var": None,
        "has_loop_carried_dep": False,
        "has_pointer_aliasing": False,
        "control_flow_inside": False,
        "array_writes": [],
        "array_reads": [],
        "loop_bound_constant": False,
        "has_consecutive_nested_loops": False,
        "max_consecutive_loop_depth": 1,
        "has_nested_braces": False,
        "has_raw_dependency": False,
        "has_war_dependency": False,
        "dependency_distance": 0,
        "has_indirect_access": False,
        "stride_pattern": 1,
        "memory_complexity": 0,
        "branch_count": 0,
        "has_early_exit": False,
        "function_call_count": 0,
        "arithmetic_op_count": 0,
        "memory_op_count": 0,
        "is_reduction_max": False,
        "is_reduction_min": False,
        "reduction_var_count": 0,
        "trip_count_computable": False,
        "estimated_iterations": 0,
        "vectorizable": False,
        "read_var_count": 0,
        "write_var_count": 0,
    }

    code_clean = re.sub(
        r"//.*?$|/\*.*?\*/", "", code, flags=re.MULTILINE | re.DOTALL
    )

    if re.search(r"\bfor\b\s*\(.*?\)\s*[{]?\s*\bfor\b", code_clean, re.DOTALL):
        features["has_consecutive_nested_loops"] = True

    for_positions = [m.start() for m in re.finditer(r"\bfor\b", code_clean)]
    max_consecutive = 1
    current = 1
    for i in range(1, len(for_positions)):
        if for_positions[i] - for_positions[i - 1] < 200:
            current += 1
            max_consecutive = max(max_consecutive, current)
        else:
            current = 1
    features["max_consecutive_loop_depth"] = max_consecutive

    brace_depth = 0
    max_brace_depth = 0
    for char in code_clean:
        if char == "{":
            brace_depth += 1
            max_brace_depth = max(max_brace_depth, brace_depth)
        elif char == "}":
            brace_depth -= 1
    features["has_nested_braces"] = max_brace_depth >= 3

    lines = [
        line.strip()
        for line in code_clean.split("\n")
        if "=" in line and line.strip()
    ]
    reduction_vars = set()
    for line in lines:
        if match := re.search(r"(\w+)\s*\+=\s*", line):
            var = match.group(1)
            if not re.search(rf"\b{var}\s*\[", code_clean):
                features["has_reduction_plus"] = True
                features["has_reduction_general"] = True
                features["reduction_var"] = var
                reduction_vars.add(var)
        elif match := re.search(r"(\w+)\s*=\s*\1\s*\+", line):
            var = match.group(1)
            if not re.search(rf"\b{var}\s*\[", code_clean):
                features["has_reduction_plus"] = True
                features["has_reduction_general"] = True
                features["reduction_var"] = var
                reduction_vars.add(var)
        elif match := re.search(r"(\w+)\s*\*=\s*", line):
            var = match.group(1)
            if not re.search(rf"\b{var}\s*\[", code_clean):
                features["has_reduction_mul"] = True
                features["has_reduction_general"] = True
                reduction_vars.add(var)

    features["reduction_var_count"] = len(reduction_vars)

    if re.search(r"\w+\[\s*\w+\s*-\s*1\s*\]", code_clean):
        features["has_loop_carried_dep"] = True
        features["has_raw_dependency"] = True
        features["dependency_distance"] = 1

    writes = re.findall(r"(\w+)\s*\[[^\]]+\]\s*=", code_clean)
    reads = re.findall(r"(\w+)\s*\[[^\]]+\]", code_clean)
    features["array_writes"] = list(set(writes))
    features["array_reads"] = list(set(reads) - set(writes))
    features["write_var_count"] = len(set(writes))
    features["read_var_count"] = len(set(reads))

    if re.search(r"\w+\s*\[\s*\w+\s*\[[^\]]+\]\s*\]", code_clean):
        features["has_indirect_access"] = True
        features["memory_complexity"] = 2

    if matches := re.findall(r"\w+\s*<\s*(\d+)", code_clean):
        if any(int(m) < 100 for m in matches):
            features["loop_bound_constant"] = True
            features["trip_count_computable"] = True
        features["estimated_iterations"] = max([int(m) for m in matches], default=0)

    features["branch_count"] = len(
        re.findall(r"\bif\b|\belse\b|\bswitch\b", code_clean)
    )
    features["control_flow_inside"] = features["branch_count"] > 0

    if re.search(r"\bbreak\b|\bcontinue\b|\breturn\b", code_clean):
        features["has_early_exit"] = True

    features["function_call_count"] = len(re.findall(r"\w+\s*\(", code_clean)) - len(
        re.findall(r"\bfor\b|\bif\b|\bwhile\b", code_clean)
    )

    features["arithmetic_op_count"] = len(
        re.findall(r"[+\-](?!=)", code_clean)
    ) + len(re.findall(r"[*/%](?!=)", code_clean))

    features["memory_op_count"] = len(re.findall(r"\w+\s*\[", code_clean))

    if re.search(r"\bmax\b|\bfmax\b", code_clean):
        features["is_reduction_max"] = True
        features["has_reduction_general"] = True
    if re.search(r"\bmin\b|\bfmin\b", code_clean):
        features["is_reduction_min"] = True
        features["has_reduction_general"] = True

    common_vars = set(reads) & set(writes)
    if common_vars:
        features["has_war_dependency"] = True

    features["vectorizable"] = (
        features["branch_count"] == 0
        and features["function_call_count"] == 0
        and not features["has_loop_carried_dep"]
        and not features["has_indirect_access"]
    )

    return features


def vectorize_features(features: dict) -> np.ndarray:
    vec = np.zeros(40, dtype=np.float32)
    bool_keys = [
        "has_reduction_plus",
        "has_reduction_mul",
        "has_reduction_general",
        "has_loop_carried_dep",
        "has_pointer_aliasing",
        "control_flow_inside",
        "loop_bound_constant",
        "has_consecutive_nested_loops",
        "has_nested_braces",
        "has_raw_dependency",
        "has_war_dependency",
        "has_indirect_access",
        "has_early_exit",
        "is_reduction_max",
        "is_reduction_min",
        "trip_count_computable",
        "vectorizable",
    ]
    for i, key in enumerate(bool_keys):
        vec[i] = 1.0 if features.get(key, False) else 0.0

    vec[17] = min(len(features.get("array_writes", [])), 5) / 5.0
    vec[18] = min(len(features.get("array_reads", [])), 5) / 5.0
    vec[19] = min(features.get("max_consecutive_loop_depth", 1), 4) / 4.0
    vec[20] = min(features.get("max_consecutive_loop_depth", 1), 4) / 4.0

    def hash_array_names(names, dim=5):
        h = np.zeros(dim)
        for name in names[:5]:
            seed = sum(ord(c) for c in name) % 1000
            for j in range(dim):
                h[j] = (h[j] + ((seed * (j + 1)) % 101) / 101.0) % 1.0
        return h / max(1, len(names)) if names else h

    vec[21:26] = hash_array_names(features.get("array_writes", []))
    vec[26:31] = hash_array_names(features.get("array_reads", []))
    vec[31] = 1.0 if features.get("reduction_var") else 0.0
    vec[32] = min(features.get("branch_count", 0), 10) / 10.0
    vec[33] = min(features.get("function_call_count", 0), 10) / 10.0
    vec[34] = min(features.get("arithmetic_op_count", 0), 50) / 50.0
    vec[35] = min(features.get("memory_op_count", 0), 50) / 50.0

    mem_ops = features.get("memory_op_count", 0)
    arith_ops = features.get("arithmetic_op_count", 0)
    if mem_ops > 0:
        ratio = arith_ops / mem_ops
        vec[36] = min(ratio, 5.0) / 5.0
    else:
        vec[36] = 0.0

    vec[37] = min(features.get("read_var_count", 0), 20) / 20.0
    vec[38] = min(features.get("write_var_count", 0), 20) / 20.0
    vec[39] = min(features.get("reduction_var_count", 0), 5) / 5.0

    return vec


def validate_pragma(pragma: str) -> str:
    if not pragma.startswith("#pragma"):
        pragma = f"#pragma {pragma}"

    brackets = {"(": ")", "{": "}", "[": "]"}
    stack = []

    for char in pragma:
        if char in brackets:
            stack.append(char)
        elif char in brackets.values():
            if not stack:
                return pragma.rstrip(char)

            top = stack.pop()
            if brackets[top] != char:
                return pragma.rstrip(char)

    while stack:
        pragma += brackets[stack.pop()]

    return pragma


class Parallelizer:
    def __init__(self):
        self.root_path = config.REPO_ROOT
        self.cache_dir = config.cache_dir
        self.classifier_endpoint = config.CLASSIFIER_ENDPOINT
        self.generator_endpoint = config.GENERATOR_ENDPOINT

    def _classify(self, embedding):
        try:
            response = requests.post(
                self.classifier_endpoint, json={"embedding": embedding.tolist()}
            )
            result = response.json()
            if response.status_code != 200:
                raise RuntimeError(
                    "Error while parallelizing, "
                    "in classifier with error: "
                    f"{result.get('error', 'Unknown error')}"
                )

            return result["result"]
        except Exception as e:
            raise RuntimeError(
                f"Error while parallelizing, in classifier with error: {e}"
            )

    def _generate_omp_pragma_with_loop(self, loop_code: str, loop_class: str) -> str:
        if loop_class == "none":
            return loop_code

        payload = {
            "code_snippet": loop_code,
            "cls": loop_class,
            "max_len": config.GENERATOR_MAX_LEN,
        }

        try:
            response = requests.post(
                self.generator_endpoint,
                json=payload,
                timeout=config.REQUEST_TIMEOUT_S,
            )
            if response.status_code != 200:
                raise RuntimeError(
                    f"Error while parallelizing, "
                    f"in generator with error: {response.text}"
                )
            pragma = response.json().get("pragma", "").strip()
            pragma = validate_pragma(pragma)
            return f"{pragma}\n{loop_code}"
        except Exception as e:
            raise RuntimeError(
                f"Error while parallelizing, in generator with error: {e}"
            )

    def parallelize(self, file_path: str) -> str:
        with open(file_path, "r") as file:
            code = file.read()

        if file_path.endswith(".py"):
            code = py2cpp_converter(code)

        loops = extract_loops(code)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        json_path = self.cache_dir / f"extracted_loops_{Path(file_path).stem}.json"

        if not os.path.exists(json_path):
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            try:
                write_loops_to_json(loops, str(json_path))
            except Exception as e:
                raise RuntimeError("Error writing JSON") from e

        with open(json_path, "r") as f:
            loops_data = json.load(f)

        code_lines = code.split("\n")

        shift = 0
        for loop in loops_data:
            loop_code = loop["code"]
            start_line = loop["start_line"]
            end_line = loop["end_line"]
            loop_type = loop["type"]

            if loop_type != "for":
                continue

            features = extract_features(loop_code)
            embedding = vectorize_features(features)
            pred_class = self._classify(embedding)

            if pred_class != "none":
                pragma_with_loop = self._generate_omp_pragma_with_loop(
                    loop_code, pred_class
                )
                target_start = start_line - 1 + shift
                target_end = end_line + shift
                new_segment = pragma_with_loop.split("\n")
                code_lines[target_start:target_end] = new_segment
                current_segment_len = target_end - target_start
                new_segment_len = len(new_segment)
                shift += new_segment_len - current_segment_len

        current_dir = Path(file_path).parent
        final_output_path = current_dir / f"parallelized_{Path(file_path).stem}.c"

        with open(final_output_path, "w") as file:
            file.write("\n".join(code_lines))


parallelizer = Parallelizer()
