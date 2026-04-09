import numpy as np
import pytest
from PIL import Image

from accelera.src.automl.core.classification_image_testing_preprocessing import (  # noqa: E501
    ClassificationImageTestingPreprocessing,
)
from accelera.src.utils.preprocessing import save_pickle


class TestClassificationImageTestingPreprocessing:
    def test_constructor(self, tmp_path):
        data_info = {"image_size": (125, 125)}
        save_pickle(tmp_path, data_info, "data_info.pkl")
        class2label_mapping = {"cats": 0, "dogs": 1}
        save_pickle(tmp_path, class2label_mapping, "class2label_mapping.pkl")
        with pytest.raises(
            ValueError, match="Image paths must be list of paths not none"
        ):
            ClassificationImageTestingPreprocessing(
                image_paths=None, folder_path=tmp_path
            )

        with pytest.raises(ValueError, match="Image paths must be list of paths"):
            ClassificationImageTestingPreprocessing(
                image_paths="path", folder_path=tmp_path
            )
        with pytest.raises(ValueError, match="Image paths is empty list"):
            ClassificationImageTestingPreprocessing(
                image_paths=[], folder_path=tmp_path
            )

        with pytest.raises(
            ValueError, match="Class names must be list of class names"
        ):
            ClassificationImageTestingPreprocessing(
                image_paths=["path"], image_class_names=0, folder_path=tmp_path
            )

        with pytest.raises(
            ValueError, match="Image paths length must equal class names length"
        ):
            ClassificationImageTestingPreprocessing(
                image_paths=["path1", "path2"],
                image_class_names=[0],
                folder_path=tmp_path,
            )
        with pytest.raises(ValueError, match="There is no valid image exists"):
            ClassificationImageTestingPreprocessing(
                image_paths=["path1", "path2"],
                image_class_names=[0, 1],
                folder_path=tmp_path,
            )
        paths = []
        for i in range(5):
            img = Image.fromarray(
                np.random.randint(0, 256, (225, 225, 3), dtype=np.uint8)
            )
            img.save(tmp_path / f"img_{i}.png")
            paths.append(str(tmp_path / f"img_{i}.png"))
        preprocesseor = ClassificationImageTestingPreprocessing(
            image_paths=paths,
            image_class_names=None,
            folder_path=tmp_path,
        )
        assert len(preprocesseor.valid_images) == 5

    def test_common_preprocessing(self, tmp_path):
        data_info = {"image_size": (125, 125)}
        save_pickle(tmp_path, data_info, "data_info.pkl")
        class2label_mapping = {"cats": 0, "dogs": 1}
        save_pickle(tmp_path, class2label_mapping, "class2label_mapping.pkl")
        paths = []
        for i in range(5):
            img = Image.fromarray(
                np.random.randint(0, 256, (225, 225, 3), dtype=np.uint8)
            )
            img.save(tmp_path / f"img_{i}.png")
            paths.append(str(tmp_path / f"img_{i}.png"))
        testing_loader, invalid_paths = ClassificationImageTestingPreprocessing(
            image_paths=paths,
            image_class_names=None,
            folder_path=tmp_path,
        ).common_preprocessing()
        assert testing_loader is not None
        assert len(invalid_paths) == 0
        images = next(iter(testing_loader))
        assert len(images) == 4
        labels = ["cats", "dogs", "cats", "birds", "dogs"]
        with pytest.raises(
            ValueError, match="this class name not in the training class"
        ):
            testing_loader, invalid_paths = ClassificationImageTestingPreprocessing(
                image_paths=paths,
                image_class_names=labels,
                folder_path=tmp_path,
            ).common_preprocessing()
        labels = ["cats", "dogs", "cats", "dogs", "dogs"]
        testing_loader, invalid_paths = ClassificationImageTestingPreprocessing(
            image_paths=paths,
            image_class_names=labels,
            folder_path=tmp_path,
        ).common_preprocessing()
        assert testing_loader is not None
        assert len(invalid_paths) == 0
        batch = next(iter(testing_loader))
        assert len(batch) == 2
        assert len(batch[0]) == 4
        assert len(batch[1]) == 4
