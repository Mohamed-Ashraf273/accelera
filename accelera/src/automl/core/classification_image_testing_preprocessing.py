from torch.utils.data import DataLoader

from accelera.src.automl.core.classification_image_dataset import (
    ClassificationImageDataset,
)
from accelera.src.automl.core.preprocessing_base import PreprocessingBase
from accelera.src.automl.utils.preprocessing import check_path_exists
from accelera.src.automl.utils.preprocessing import collect_function
from accelera.src.automl.utils.preprocessing import is_valid_image
from accelera.src.automl.utils.preprocessing import load_pickle


class ClassificationImageTestingPreprocessing(PreprocessingBase):
    def __init__(
        self,
        image_paths,
        image_class_names=None,
        folder_path=None,
        batch_size=4,
    ):
        super().__init__(folder_path)
        self.image_paths = image_paths
        self.image_class_names = image_class_names
        self.batch_size = batch_size
        self.valid_images = []
        self.valid_images_class_names = []
        self.invalid_images = []
        if self.image_paths is None:
            raise ValueError("Image paths must be list of paths not none")
            
        if not isinstance(self.image_paths, list):
            raise ValueError("Image paths must be list of paths")
        if len(self.image_paths)==0:
            raise ValueError("Image paths is empty list")
        if self.image_class_names is not None and not isinstance(
            self.image_class_names, list
        ):
            raise ValueError("Class names must be list of class names")
        if self.image_class_names is not None and len(
            self.image_class_names
        ) != len(self.image_paths):
            raise ValueError("Image paths length must equal class names length")
        for i, path in enumerate(self.image_paths):
            if is_valid_image(path):
                self.valid_images.append(path)
                if self.image_class_names is not None:
                    self.valid_images_class_names.append(
                        self.image_class_names[i]
                    )
            else:
                self.invalid_images.append(path)
        if len(self.valid_images)==0:
            raise ValueError("There is no valid image exists")
        
        check_path_exists(self.folder_path, "data_info.pkl")
        self.image_size = load_pickle(self.folder_path, "data_info.pkl")[
            "image_size"
        ]
        check_path_exists(self.folder_path, "class2label_mapping.pkl")
        self.class2label_mapping = load_pickle(
            self.folder_path, "class2label_mapping.pkl"
        )

    def common_preprocessing(self):
        labels = None
        if self.image_class_names is not None:
            labels = []
            for class_name in self.valid_images_class_names:
                if class_name not in self.class2label_mapping:
                    raise ValueError(
                        "this class name not in the training class"
                    )
                labels.append(self.class2label_mapping[class_name])
    
        dataset = ClassificationImageDataset(
            self.valid_images, labels, self.image_size, augment=False
        )
        testing_loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collect_function,
        )
        return testing_loader, self.invalid_images
