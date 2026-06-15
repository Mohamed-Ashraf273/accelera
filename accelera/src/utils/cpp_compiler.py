import hashlib
import importlib.util
import os
import re
import subprocess
import sysconfig

from accelera.src.config import config


def _python_include_flags() -> list[str]:
    paths = sysconfig.get_paths()
    includes = [paths["include"], paths.get("platinclude")]
    return [f"-I{path}" for path in includes if path]


def _pybind11_include_flag() -> str:
    try:
        import pybind11

        return f"-I{pybind11.get_include()}"
    except ImportError:
        pass

    include_dir = config.REPO_ROOT / "build" / "_deps" / "pybind11-src" / "include"
    if not include_dir.exists():
        raise RuntimeError(
            "pybind11 headers were not found. Install pybind11 or build "
            "Accelera first so build/_deps/pybind11-src/include exists."
        )
    return f"-I{include_dir}"


def _to_numpy_preprocess_cpp(cpp_code: str, func_name: str, module_name: str) -> str:
    code = cpp_code
    if "int main()" in code:
        code = code[: code.index("int main()")].rstrip()

    code = re.sub(r"template\s*<[^>]+>\s*\n", "", code)
    code = re.sub(
        rf"auto\s+{re.escape(func_name)}\s*\([^)]*\)\s*\{{",
        (
            f"py::array_t<double> {func_name}(py::array_t<double> X) {{\n"
            "    auto x = X.mutable_unchecked<2>();"
        ),
        code,
        count=1,
    )
    code = re.sub(r"\blen\s*\(\s*X\s*\[\s*\w+\s*\]\s*\)", "x.shape(1)", code)
    code = re.sub(r"\blen\s*\(\s*X\s*\)", "x.shape(0)", code)
    code = re.sub(
        r"\bX\s*\[\s*([^\]]+)\s*\]\s*\[\s*([^\]]+)\s*\]",
        r"x(\1, \2)",
        code,
    )
    code = re.sub(r"\bint\s+s\s*=\s*0\s*;", "double s = 0.0;", code)

    return (
        "#include <cmath>\n"
        "#include <pybind11/numpy.h>\n"
        "#include <pybind11/pybind11.h>\n\n"
        "namespace py = pybind11;\n\n"
        f"{code}\n\n"
        f"PYBIND11_MODULE({module_name}, m) {{\n"
        f'    m.def("{func_name}", &{func_name});\n'
        "}\n"
    )


def compile_parallelized_code(cpp_code: str, func_name: str):
    source_hash = hashlib.md5(cpp_code.encode()).hexdigest()
    module_name = f"accelera_parallel_{source_hash}"
    build_dir = config.cache_dir / "compiled"
    build_dir.mkdir(parents=True, exist_ok=True)

    extension_suffix = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
    cpp_path = build_dir / f"{module_name}.cpp"
    module_path = build_dir / f"{module_name}{extension_suffix}"

    if not module_path.exists():
        cpp_path.write_text(
            _to_numpy_preprocess_cpp(cpp_code, func_name, module_name)
        )
        cxx = os.getenv("CXX", "c++")
        cmd = [
            cxx,
            "-O3",
            "-shared",
            "-std=c++17",
            "-fPIC",
            "-fopenmp",
            *_python_include_flags(),
            _pybind11_include_flag(),
            str(cpp_path),
            "-o",
            str(module_path),
        ]
        subprocess.run(cmd, check=True)

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load compiled module: {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, func_name)
