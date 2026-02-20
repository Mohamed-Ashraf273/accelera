from torch.utils.data import DataLoader

from accelera.src.automl.core.segmentation_image_dataset import (
    SegmentationImageDataset,
)
from accelera.src.automl.core.preprocessing_base import PreprocessingBase
from accelera.src.automl.utils.preprocessing import check_path_exists
from accelera.src.automl.utils.preprocessing import collect_function_segmentation
from accelera.src.automl.utils.preprocessing import is_valid_image
from accelera.src.automl.utils.preprocessing import load_pickle


class SegmentationImageTestingPreprocessing(PreprocessingBase):
    def __init__(
        self,
        image_paths,
        image_masks=None,
        folder_path=None,
        batch_size=4,
    ):
        super().__init__(folder_path)
        self.image_paths = image_paths
        self.batch_size = batch_size
        self.image_masks = image_masks
        self.valid_images = []
        self.valid_masks = []
        self.invalid_images = []
        if not isinstance(self.image_paths, list):
            raise ValueError("Image paths must be list of paths")
        if self.image_masks is not None and not isinstance(self.image_masks, list):
            raise ValueError("masks must be list of masks paths")
        if self.image_masks is not None and len(self.image_masks) != len(
            self.image_paths
        ):
            raise ValueError("image paths length must equal masks length")
        for i, path in enumerate(self.image_paths):
            if is_valid_image(path):
                if self.image_masks is not None:
                    if is_valid_image(self.image_masks[i]):
                        self.valid_images.append(path)
                        self.valid_masks.append(self.image_masks[i])
                    else:
                        self.invalid_images.append(path)
                else:
                    self.valid_images.append(path)
            else:
                self.invalid_images.append(path)

        check_path_exists(self.folder_path, "data_info.pkl")
        info = load_pickle(self.folder_path, "data_info.pkl")
        self.image_size = info["image_size"]
        self.mask_type = info["mask_type"]
        self.mask_classes = info["mask_classes"]
        self.binary_mask_threshold=info["binary_mask_threshold"]

    def common_preprocessing(self):
        self.valid_masks = None if len(self.valid_masks) == 0 else self.valid_masks
        dataset = SegmentationImageDataset(
            self.valid_images,
            self.valid_masks,
            self.image_size,
            self.mask_type,
            self.mask_classes,
            self.binary_mask_threshold,
            augment=False,
        )
        testing_loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collect_function_segmentation,
        )
        return testing_loader, self.invalid_images
