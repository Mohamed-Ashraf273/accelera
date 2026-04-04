from __future__ import annotations

import os
import re
import shutil
from typing import Any
from typing import Dict
from typing import List
from urllib.parse import urlparse

import pandas as pd
import requests
from platformdirs import user_cache_dir

from accelera.src.config import config
from accelera.src.utils.accelera_utils import print_msg


class DatasetRetriever:
    def __init__(self):
        self.is_connected = False
        self.cache_dir = None

    def _fetch_url_content(self, url: str) -> Dict[str, Any]:
        raw_url = url
        parsed = urlparse(url)

        if "drive.google.com" in parsed.netloc:
            raw_url = url

        try:
            response = requests.get(
                raw_url,
                headers={"User-Agent": "Mozilla/5.0"},
                timeout=config.REQUEST_TIMEOUT_S,
            )
            response.raise_for_status()

            return {
                "content": response.text,
                "source": raw_url,
            }

        except Exception as exc:
            return {"error": f"Failed to fetch page content: {exc}"}

    def available_datasets(
        self, url: str = config.DATASET_DRIVE_FOLDER_ENDPOINT
    ) -> List[str]:
        result = self._fetch_url_content(url)

        if "error" in result:
            return []

        html = result["content"]
        names = re.findall(r"[\w\-. ]+\.csv", html)

        if not names:
            raise ValueError("No datasets found")

        return list(
            set(name.replace("x22", "").split(".csv")[0] for name in names)
        )

    def connect(self):
        self.is_connected = True
        self.cache_dir = user_cache_dir("accelera")
        os.makedirs(self.cache_dir, exist_ok=True)

    def retrieve_dataset(
        self, dataset_name: str, encoding: str = "utf-8", df: bool = False
    ) -> pd.DataFrame:
        def get_df(dataset_path: str):
            return pd.read_csv(
                dataset_path,
                encoding=encoding,
                sep=None,
                engine="python",
                on_bad_lines="warn",
            )

        if not self.is_connected or not self.cache_dir:
            raise RuntimeError(
                "Dataset retriever is not connected. Call `connect()` first."
            )

        if dataset_name not in config.DATASETS:
            raise KeyError(
                f"Unknown dataset '{dataset_name}'. "
                f"Available: {self.available_datasets()}"
            )

        dataset_path = os.path.join(self.cache_dir, f"{dataset_name}.csv")
        if os.path.exists(dataset_path):
            if df:
                return get_df(dataset_path)
            return str(dataset_path)

        try:
            import gdown
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Missing optional dependency 'gdown'. Run pip install gdown."
            ) from exc

        gdown.download(
            id=config.DATASETS[dataset_name]["id"],
            output=dataset_path,
            quiet=False,
        )
        if df:
            return get_df(dataset_path)
        return str(dataset_path)

    def close(self):
        if not self.is_connected:
            raise RuntimeError("Dataset retriever is not connected")

        self.is_connected = False
        if self.cache_dir:
            shutil.rmtree(self.cache_dir)
        else:
            print_msg("Cache directory not found", level="warning")

        self.cache_dir = None


retriever = DatasetRetriever()
