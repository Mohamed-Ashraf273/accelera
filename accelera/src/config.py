"""Project-wide configuration.

Keep constants and shared defaults here so they can be reused across the
`accelera` package without duplicating values.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path


def _repo_root_from_this_file() -> Path:
    return Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Config:
    REPO_ROOT: Path = _repo_root_from_this_file()
    CACHE_DIR_NAME: str = ".accelera_cache"
    BUILD_DIR_NAME: str = "build"
    BINDINGS_SUBDIR: str = "bindings"
    API_SUBDIR: str = "api"

    DEFAULT_CLANG_ARGS: tuple[str, ...] = ("-std=c++17",)
    CLANG_FORMAT_BIN: str = os.getenv(
        "ACCELERA_CLANG_FORMAT_BIN", "clang-format"
    )
    ENABLE_CPP_FORMATTING: bool = (
        os.getenv("ACCELERA_ENABLE_CPP_FORMATTING", "1") == "1"
    )

    CLASSIFIER_ENDPOINT: str = os.getenv(
        "ACCELERA_CLASSIFIER_ENDPOINT",
        "https://accelera-ai-open-mp-classifier.hf.space/predict",
    )
    GENERATOR_ENDPOINT: str = os.getenv(
        "ACCELERA_GENERATOR_ENDPOINT",
        "https://accelera-ai-open-mp-generator.hf.space/generate",
    )

    GENERATOR_MAX_LEN: int = int(
        os.getenv("ACCELERA_GENERATOR_MAX_LEN", "1500")
    )
    REQUEST_TIMEOUT_S: int = int(os.getenv("ACCELERA_REQUEST_TIMEOUT_S", "10"))

    # If 1, `accelera/__init__.py` will
    # raise if optional `accelera.api` import fails.
    STRICT_API_IMPORT: bool = (
        os.getenv("ACCELERA_STRICT_API_IMPORT", "0") == "1"
    )

    @property
    def cache_dir(self) -> Path:
        return self.REPO_ROOT / self.CACHE_DIR_NAME

    @property
    def bindings_dir(self) -> Path:
        return self.REPO_ROOT / self.BUILD_DIR_NAME / self.BINDINGS_SUBDIR

    @property
    def api_dir(self) -> Path:
        # Where the generated public API package lives: accelera/api
        return self.REPO_ROOT / "accelera" / self.API_SUBDIR

    def ensure_bindings_on_syspath(self) -> None:
        bindings = str(self.bindings_dir)
        if bindings not in sys.path:
            sys.path.insert(0, bindings)


config = Config()
