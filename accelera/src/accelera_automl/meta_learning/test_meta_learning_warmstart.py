from pathlib import Path

import numpy as np

from accelera.src.accelera_automl.meta_learning.warmstart import resolve_path


def test_resolve_path_finds_packaged_classification_metadata():
    meta_learning_root = (
        Path(__file__).resolve().parents[1] / "meta_learning_data" / "json"
    )

    path = resolve_path(
        task="classification",
        y=np.array([0, 1]),
        scoring=None,
        meta_learning_root=meta_learning_root,
    )

    assert path == meta_learning_root / "accuracy_binary.classification_dense.json"
