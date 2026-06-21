import hashlib
import importlib.util
import os
import re
import shutil
import subprocess
import sys
import sysconfig
from functools import lru_cache
from pathlib import Path

from accelera.src.config import config


@lru_cache(maxsize=1)
def python_include_flags() -> tuple[str, ...]:
    paths = sysconfig.get_paths()
    includes = [paths["include"], paths.get("platinclude")]
    return tuple(f"-I{path}" for path in includes if path)


@lru_cache(maxsize=1)
def python_library_flags() -> tuple[str, ...]:
    if os.name != "nt":
        return ()

    flags = []
    lib_dirs = [
        sysconfig.get_config_var("LIBDIR"),
        Path(sys.prefix) / "libs",
        Path(sys.base_prefix) / "libs",
        Path(sys.exec_prefix) / "libs",
        Path(sys.base_exec_prefix) / "libs",
    ]

    library = sysconfig.get_config_var("LDLIBRARY")
    if not library:
        version = f"{sys.version_info.major}{sys.version_info.minor}"
        library = f"python{version}.lib"

    for libdir in lib_dirs:
        if libdir and (Path(libdir) / library).exists():
            flags.append(str(Path(libdir) / library))
            break

    return tuple(flags)


@lru_cache(maxsize=1)
def pybind11_include_flag() -> str:
    try:
        import pybind11

        return f"-I{pybind11.get_include()}"
    except ImportError:
        pass

    candidates = [
        config.REPO_ROOT / "build" / "_deps" / "pybind11-src" / "include",
        Path(sys.prefix) / "Lib" / "site-packages" / "pybind11" / "include",
        Path(sys.prefix) / "lib" / "site-packages" / "pybind11" / "include",
    ]

    for include_dir in candidates:
        if (include_dir / "pybind11" / "pybind11.h").exists():
            return f"-I{include_dir}"

    raise RuntimeError(
        "pybind11 headers were not found. Install pybind11 with "
        "`pip install pybind11` or run `pip install -r requirements.txt`."
    )


def default_cxx() -> str:
    if cxx := os.getenv("CXX"):
        return cxx

    if os.name != "nt":
        return "c++"

    for candidate in (
        shutil.which("clang++"),
        Path(os.getenv("ProgramFiles", "")) / "LLVM" / "bin" / "clang++.exe",
        Path(os.getenv("LOCALAPPDATA", ""))
        / "Programs"
        / "LLVM"
        / "bin"
        / "clang++.exe",
    ):
        if candidate and Path(candidate).exists():
            return str(candidate)

    return "clang++"


def compiler_command(cxx: str) -> list[str]:
    if os.getenv("ACCELERA_USE_COMPILER_CACHE", "1") != "1":
        return [cxx]

    for cache_tool in ("sccache", "ccache"):
        if compiler_cache := shutil.which(cache_tool):
            return [compiler_cache, cxx]

    return [cxx]


def to_numpy_code_cpp(cpp_code: str, func_name: str, module_name: str) -> str:
    code = cpp_code
    if "int main()" in code:
        code = code[: code.index("int main()")].rstrip()

    code = re.sub(r"template\s*<[^>]+>\s*", "", code)
    code = re.sub(r"\bT\d+\b", "double", code)
    function_pattern = (
        rf"auto\s+{re.escape(func_name)}\s*\(\s*[^,()\s]+\s+"
        r"([A-Za-z_]\w*)\s*\)\s*\{"
    )
    function_match = re.search(function_pattern, code)
    if function_match is None:
        raise ValueError(
            "Only functions with one named input parameter can be compiled."
        )

    input_name = function_match.group(1)
    escaped_input_name = re.escape(input_name)
    code = re.sub(
        function_pattern,
        (
            f"py::array_t<double> {func_name}(py::array_t<double> {input_name}) {{\n"
            f"    auto x = {input_name}.mutable_unchecked<2>();"
        ),
        code,
        count=1,
    )
    code = re.sub(
        rf"\blen\s*\(\s*{escaped_input_name}\s*\[\s*\w+\s*\]\s*\)",
        "x.shape(1)",
        code,
    )
    code = re.sub(rf"\blen\s*\(\s*{escaped_input_name}\s*\)", "x.shape(0)", code)
    code = re.sub(
        rf"\b{escaped_input_name}\s*\[\s*([^\]]+)\s*\]\s*\[\s*([^\]]+)\s*\]",
        r"x(\1, \2)",
        code,
    )
    code = re.sub(
        r"\b(?:int|long\s+long|auto)\s+s\s*=\s*0\s*;",
        "double s = 0.0;",
        code,
    )

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


def compile_opt_flag() -> str:
    return os.getenv("ACCELERA_CPP_OPT_LEVEL", "-O0")


def needs_openmp(cpp_code: str) -> bool:
    return "#pragma omp" in cpp_code


def compiled_module_name(
    cpp_code: str,
    func_name: str,
    extension_suffix: str,
) -> str:
    cache_key = "\n".join(
        [
            config.COMPILE_CACHE_VERSION,
            func_name,
            compile_opt_flag(),
            extension_suffix,
            cpp_code,
        ]
    )
    source_hash = hashlib.md5(cache_key.encode()).hexdigest()
    return f"accelera_parallel_{source_hash}"


def compile_parallelized_code(cpp_code: str, func_name: str):
    extension_suffix = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
    module_name = compiled_module_name(cpp_code, func_name, extension_suffix)
    loaded_module = sys.modules.get(module_name)
    if loaded_module is not None:
        return getattr(loaded_module, func_name)

    build_dir = config.cache_dir / "compiled"
    build_dir.mkdir(parents=True, exist_ok=True)

    cpp_path = build_dir / f"{module_name}.cpp"
    module_path = build_dir / f"{module_name}{extension_suffix}"

    if not module_path.exists():
        cpp_path.write_text(to_numpy_code_cpp(cpp_code, func_name, module_name))
        cxx = default_cxx()
        compile_flags = [
            compile_opt_flag(),
            "-shared",
            "-std=c++17",
            "-DNDEBUG",
        ]
        if needs_openmp(cpp_code):
            compile_flags.append("-fopenmp")
        if os.name != "nt":
            compile_flags.extend(["-fPIC", "-pipe"])

        cmd = [
            *compiler_command(cxx),
            *compile_flags,
            *python_include_flags(),
            pybind11_include_flag(),
            str(cpp_path),
            *python_library_flags(),
            "-o",
            str(module_path),
        ]
        subprocess.run(cmd, check=True)

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load compiled module: {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules[module_name] = module
    return getattr(module, func_name)
