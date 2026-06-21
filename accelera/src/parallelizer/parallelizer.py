import ast
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path

import numpy as np
import requests

from accelera.src.config import config
from accelera.src.parallelizer.py2cpp_converter import py2cpp_converter
from accelera.src.utils.accelera_utils import print_msg

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


def _compiled_extension_suffix() -> str:
    return sysconfig.get_config_var("EXT_SUFFIX") or ".so"


def _pymethod_cache_path(code: str, func_name: str) -> Path:
    from accelera.src.parallelizer.cpp_compiler import compile_opt_flag

    cache_key = "\n".join(
        [
            config.COMPILE_CACHE_VERSION,
            func_name,
            compile_opt_flag(),
            _compiled_extension_suffix(),
            code,
        ]
    )
    source_hash = hashlib.md5(cache_key.encode()).hexdigest()
    return config.cache_dir / "compiled" / f"pymethod_{source_hash}.json"


def _load_cached_pymethod(cache_path: Path, func_name: str):
    if not cache_path.exists():
        return None

    try:
        module_name = json.loads(cache_path.read_text())["module_name"]
        module_path = (
            config.cache_dir
            / "compiled"
            / f"{module_name}{_compiled_extension_suffix()}"
        )
        if not module_path.exists():
            return None

        loaded_module = sys.modules.get(module_name)
        if loaded_module is not None:
            return getattr(loaded_module, func_name)

        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:
            return None

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        sys.modules[module_name] = module
        return getattr(module, func_name)
    except (OSError, KeyError, json.JSONDecodeError, AttributeError, ImportError):
        return None


def _write_pymethod_cache(cache_path: Path, module_name: str) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps({"module_name": module_name}, indent=2))


def _processed_code_cache_path(code: str) -> Path:
    cache_key = "\n".join([config.COMPILE_CACHE_VERSION, code])
    source_hash = hashlib.md5(cache_key.encode()).hexdigest()
    return config.cache_dir / "processed_code" / f"{source_hash}.cpp"


def _loop_to_dict(loop) -> dict:
    if isinstance(loop, dict):
        return loop
    return {
        "code": loop.code,
        "start_line": loop.start_line,
        "end_line": loop.end_line,
        "type": loop.type,
    }


def pragma_to_class(label: str, pragma: str) -> str:
    label = str(label)
    pragma = str(pragma or "")

    if label == "False":
        return "none"

    if re.search(r"\breduction\s*\(", pragma):
        return "reduction"

    return "parallel_for"


def _strip_comments(code: str) -> str:
    return re.sub(r"//.*?$|/\*.*?\*/", "", code, flags=re.MULTILINE | re.DOTALL)


def _extract_loop_structure_features(code_clean: str) -> dict:
    features = {
        "has_consecutive_nested_loops": False,
        "max_consecutive_loop_depth": 1,
        "has_nested_braces": False,
    }

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

    return features


def _extract_reduction_features(code_clean: str) -> dict:
    features = {
        "has_reduction_plus": False,
        "has_reduction_mul": False,
        "has_reduction_general": False,
        "reduction_var": None,
        "is_reduction_max": False,
        "is_reduction_min": False,
        "reduction_var_count": 0,
    }

    reduction_vars = set()
    lines = [
        line.strip()
        for line in code_clean.split("\n")
        if "=" in line and line.strip()
    ]

    for line in lines:
        if match := re.search(r"\b([A-Za-z_]\w*)\s*\+=\s*", line):
            var = match.group(1)
            if not re.search(rf"\b{var}\s*\[", code_clean):
                features["has_reduction_plus"] = True
                features["has_reduction_general"] = True
                features["reduction_var"] = var
                reduction_vars.add(var)
        elif match := re.search(r"\b([A-Za-z_]\w*)\s*=\s*\1\s*\+", line):
            var = match.group(1)
            if not re.search(rf"\b{var}\s*\[", code_clean):
                features["has_reduction_plus"] = True
                features["has_reduction_general"] = True
                features["reduction_var"] = var
                reduction_vars.add(var)
        elif match := re.search(r"\b([A-Za-z_]\w*)\s*\*=\s*", line):
            var = match.group(1)
            if not re.search(rf"\b{var}\s*\[", code_clean):
                features["has_reduction_mul"] = True
                features["has_reduction_general"] = True
                features["reduction_var"] = var
                reduction_vars.add(var)
        elif match := re.search(r"\b([A-Za-z_]\w*)\s*=\s*\1\s*\*", line):
            var = match.group(1)
            if not re.search(rf"\b{var}\s*\[", code_clean):
                features["has_reduction_mul"] = True
                features["has_reduction_general"] = True
                features["reduction_var"] = var
                reduction_vars.add(var)

    if re.search(r"\bmax\b|\bfmax\b", code_clean):
        features["is_reduction_max"] = True
        features["has_reduction_general"] = True
    if re.search(r"\bmin\b|\bfmin\b", code_clean):
        features["is_reduction_min"] = True
        features["has_reduction_general"] = True

    features["reduction_var_count"] = len(reduction_vars)
    return features


def _has_scalar_reduction_update(code: str) -> bool:
    reduction_features = _extract_reduction_features(_strip_comments(code))
    return bool(
        reduction_features["reduction_var"]
        or reduction_features["is_reduction_max"]
        or reduction_features["is_reduction_min"]
    )


def _is_declared_inside_loop(loop_code: str, var_name: str) -> bool:
    return bool(
        re.search(
            rf"\b(?:auto|bool|char|double|float|int|long|short|size_t)\s+"
            rf"{re.escape(var_name)}\b",
            loop_code,
        )
    )


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

    code_clean = _strip_comments(code)
    features.update(_extract_loop_structure_features(code_clean))
    features.update(_extract_reduction_features(code_clean))

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


def _find_matching_paren(text: str, open_index: int) -> int | None:
    depth = 0
    for i in range(open_index, len(text)):
        if text[i] == "(":
            depth += 1
        elif text[i] == ")":
            depth -= 1
            if depth == 0:
                return i
    return None


def _consecutive_for_depth(code: str) -> int:
    code = _strip_comments(code)
    depth = 0
    pos = 0

    while True:
        match = re.search(r"\bfor\s*\(", code[pos:])
        if not match:
            return depth

        start = pos + match.start()
        if code[pos:start].strip(" \t\r\n{"):
            return depth

        open_index = code.find("(", start)
        close_index = _find_matching_paren(code, open_index)
        if close_index is None:
            return depth

        depth += 1
        pos = close_index + 1
        while pos < len(code) and code[pos].isspace():
            pos += 1
        if pos < len(code) and code[pos] == "{":
            pos += 1


def _is_independent_array_write_loop(loop_code: str) -> bool:
    features = extract_features(loop_code)
    return bool(
        features.get("array_writes")
        and not features.get("has_loop_carried_dep")
        and not features.get("has_indirect_access")
        and not features.get("has_early_exit")
    )


def _resolve_loop_class(loop_code: str, pred_class: str) -> str:
    if pred_class != "none":
        return pred_class
    if _has_scalar_reduction_update(loop_code):
        return "reduction"
    if _is_independent_array_write_loop(loop_code):
        return "parallel_for"
    if _consecutive_for_depth(loop_code) > 1:
        return "parallel_for"
    return "none"


class Parallelizer:
    def __init__(self):
        self.root_path = config.REPO_ROOT
        self.cache_dir = config.cache_dir
        self.classifier_endpoint = config.CLASSIFIER_ENDPOINT

    def _classify(self, loop_code: str):
        features = vectorize_features(extract_features(loop_code))
        try:
            response = requests.post(
                self.classifier_endpoint,
                json={"embedding": features.tolist()},
                timeout=config.REQUEST_TIMEOUT_S,
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

    def _generate_omp_pragma_with_loop(
        self, loop_code: str, loop_class: str, code_context: str = ""
    ) -> str:
        if loop_class == "none":
            return loop_code

        features = extract_features(loop_code)
        clauses = []

        depth = _consecutive_for_depth(loop_code)
        if depth > 1:
            clauses.append(f"collapse({depth})")

        reduction_var = features.get("reduction_var")
        if (
            features.get("has_reduction_plus")
            and reduction_var
            and not _is_declared_inside_loop(loop_code, reduction_var)
        ):
            clauses.append(f"reduction(+ : {reduction_var})")
        elif features.get("has_reduction_mul"):
            if match := re.search(r"\b([A-Za-z_]\w*)\s*\*=", loop_code):
                reduction_var = match.group(1)
                if not _is_declared_inside_loop(loop_code, reduction_var):
                    clauses.append(f"reduction(* : {reduction_var})")

        pragma = "#pragma omp parallel for"
        if clauses:
            pragma = f"{pragma} {' '.join(clauses)}"

        return f"{pragma}\n{loop_code}"

    def _select_parallel_loops(self, loops_data: list) -> list:
        selected_loops = []
        for loop in loops_data:
            loop_code = loop["code"]
            if loop["type"] != "for":
                continue

            loop_class = _resolve_loop_class(loop_code, "none")
            if loop_class != "none":
                selected_loops.append((loop, loop_class))
                continue

            try:
                pred_class = self._classify(loop_code)
            except RuntimeError:
                pred_class = "none"
            loop_class = _resolve_loop_class(loop_code, pred_class)

            if loop_class != "none":
                selected_loops.append((loop, loop_class))

        return selected_loops

    def _apply_parallel_loops(self, code: str, selected_loops: list) -> str:
        code_lines = code.split("\n")
        selected_ranges = [
            (loop["start_line"], loop["end_line"]) for loop, _ in selected_loops
        ]
        shift = 0
        for loop, pred_class in selected_loops:
            loop_code = loop["code"]
            start_line = loop["start_line"]
            end_line = loop["end_line"]

            if any(
                outer_start < start_line and end_line <= outer_end
                for outer_start, outer_end in selected_ranges
            ):
                continue

            pragma_with_loop = self._generate_omp_pragma_with_loop(
                loop_code, pred_class, code
            )
            target_start = start_line - 1 + shift
            target_end = end_line + shift
            new_segment = pragma_with_loop.split("\n")
            code_lines[target_start:target_end] = new_segment
            current_segment_len = target_end - target_start
            new_segment_len = len(new_segment)
            shift += new_segment_len - current_segment_len

        return "\n".join(code_lines)

    def _process_file(
        self, file_path: str, output_dir: str | Path | None = None
    ) -> None:
        with open(file_path, "r") as source_file:
            code = source_file.read()

        if file_path.endswith(".py"):
            code = py2cpp_converter(code)

        loops = extract_loops(code)
        file_name = hashlib.md5(code.encode()).hexdigest()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        json_path = self.cache_dir / f"extracted_loops_{file_name}.json"

        if not os.path.exists(json_path):
            try:
                write_loops_to_json(loops, str(json_path))
            except Exception as e:
                raise RuntimeError("Error writing JSON") from e

        with open(json_path, "r") as f:
            loops_data = json.load(f)

        parallelized_code = self._apply_parallel_loops(
            code, self._select_parallel_loops(loops_data)
        )
        source_path = Path(file_path)
        output_path = (
            Path(output_dir) if output_dir is not None else Path(config.REPO_ROOT)
        )
        output_path.mkdir(parents=True, exist_ok=True)
        final_output_path = output_path / f"parallelized_{source_path.stem}.c"
        with open(final_output_path, "w") as output_file:
            output_file.write(parallelized_code)

        if clang_format := shutil.which("clang-format"):
            subprocess.run([clang_format, "-i", str(final_output_path)], check=True)

    def _process_code(self, code: str) -> str:
        cache_path = _processed_code_cache_path(code)
        if cache_path.exists():
            return cache_path.read_text()

        try:
            ast.parse(code)
        except SyntaxError:
            pass
        else:
            code = py2cpp_converter(code)

        loops_data = [_loop_to_dict(loop) for loop in extract_loops(code)]
        parallelized_code = self._apply_parallel_loops(
            code, self._select_parallel_loops(loops_data)
        )
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(parallelized_code)
        return parallelized_code

    def _optimize_pymethod(self, func, source=None):
        from accelera.src.parallelizer.cpp_compiler import compile_parallelized_code
        from accelera.src.parallelizer.cpp_compiler import compiled_module_name
        from accelera.src.utils.source_backed_function import SourceBackedFunction

        try:
            code = source or SourceBackedFunction(func).compilation_source()
            cache_path = _pymethod_cache_path(code, func.__name__)
            cached_func = _load_cached_pymethod(cache_path, func.__name__)
            if cached_func is not None:
                return cached_func

            cpp_code = self.parallelize(code)
            module_name = compiled_module_name(
                cpp_code,
                func.__name__,
                _compiled_extension_suffix(),
            )
            optimized_func = compile_parallelized_code(cpp_code, func.__name__)
            _write_pymethod_cache(cache_path, module_name)
            return optimized_func
        except (
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
            subprocess.CalledProcessError,
        ):
            print_msg(
                f"Failed to optimize function '{func.__name__}', "
                "falling back to original function.",
                level="warning",
            )
            return func

    def _optimize_pyinstance(self, instance):
        if "transform" not in instance.__class__.__dict__:
            return instance

        method = getattr(instance, "transform", None)
        if method is None:
            return instance

        optimized_method = self._optimize_pymethod(method)
        if optimized_method is not method:
            setattr(instance, "transform", optimized_method)

        return instance

    def parallelize(self, content, output_dir: str | Path | None = None) -> str:
        from accelera.src.custom.transformer import CustomTransformer
        from accelera.src.utils.accelera_utils import is_custom_function
        from accelera.src.utils.source_backed_function import SourceBackedFunction

        if isinstance(content, str):
            if os.path.isfile(content):
                return self._process_file(content, output_dir)
            return self._process_code(content)
        elif isinstance(content, CustomTransformer):
            return self._optimize_pyinstance(content)
        elif is_custom_function(content):
            source_func = SourceBackedFunction(content)
            source_func.set_runtime_func(
                self._optimize_pymethod(
                    content,
                    source=source_func.compilation_source(),
                )
            )
            return source_func
        else:
            return content


parallelizer = Parallelizer()
