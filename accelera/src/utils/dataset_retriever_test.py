import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from accelera.src.utils.dataset_retriever import DatasetRetriever


class TestDatasetRetriever:
    @pytest.fixture
    def connected_retriever(self, tmp_path):
        retriever = DatasetRetriever()
        retriever.is_connected = True
        retriever.cache_dir = str(tmp_path)
        return retriever

    @pytest.fixture
    def fake_gdown(self, monkeypatch):
        calls = []

        def download(**kwargs):
            calls.append(kwargs)
            Path(kwargs["output"]).write_text("feature,target\n1,0\n")

        monkeypatch.setitem(
            sys.modules,
            "gdown",
            SimpleNamespace(download=download),
        )
        monkeypatch.setattr(
            "accelera.src.utils.dataset_retriever.print_msg",
            lambda *args, **kwargs: None,
        )
        return calls

    def test_available_datasets_extracts_unique_names(self, monkeypatch):
        retriever = DatasetRetriever()
        monkeypatch.setattr(
            retriever,
            "_fetch_url_content",
            lambda url: {
                "content": (
                    "<a>customer_purchase_data.csv</a>"
                    "<a>Housing.csv</a>"
                    "<a>Housing.csv</a>"
                    "<a>Samplex22.csv</a>"
                )
            },
        )

        result = retriever.available_datasets()

        assert set(result) == {
            "customer_purchase_data",
            "Housing",
            "Sample",
        }

    def test_connect_and_close_manage_cache_dir(self, monkeypatch, tmp_path):
        cache_dir = tmp_path / "accelera-cache"
        monkeypatch.setattr(
            "accelera.src.utils.dataset_retriever.user_cache_dir",
            lambda _: str(cache_dir),
        )

        retriever = DatasetRetriever()
        retriever.connect()

        assert retriever.is_connected is True
        assert retriever.cache_dir == str(cache_dir)
        assert cache_dir.exists()

        retriever.close()

        assert retriever.is_connected is False
        assert retriever.cache_dir is None
        assert not cache_dir.exists()

    def test_retrieve_dataset_requires_connect(self):
        retriever = DatasetRetriever()

        with pytest.raises(RuntimeError, match="Call `connect\\(\\)` first"):
            retriever.retrieve_dataset("customer_purchase_data")

    def test_retrieve_dataset_returns_cached_dataframe(
        self, connected_retriever, fake_gdown
    ):
        dataset_path = Path(connected_retriever.cache_dir) / "cached.csv"
        dataset_path.write_text("feature,target\n1,0\n2,1\n")

        result = connected_retriever.retrieve_dataset("cached", df=True)

        expected = pd.DataFrame({"feature": [1, 2], "target": [0, 1]})
        pd.testing.assert_frame_equal(result, expected)
        assert fake_gdown == []

    def test_retrieve_dataset_downloads_known_dataset(
        self, connected_retriever, fake_gdown, monkeypatch
    ):
        monkeypatch.setitem(
            connected_retriever.retrieve_dataset.__globals__["config"].DATASETS,
            "toy_dataset",
            {"id": "file-123", "metadata": {}},
        )

        result = connected_retriever.retrieve_dataset("toy_dataset")

        assert result == str(Path(connected_retriever.cache_dir) / "toy_dataset.csv")
        assert fake_gdown == [
            {
                "id": "file-123",
                "output": result,
                "quiet": False,
            }
        ]
