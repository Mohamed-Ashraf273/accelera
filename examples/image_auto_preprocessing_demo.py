from accelera.src.automl.core.classification_image_testing_preprocessing import (  # noqa: E501
    ClassificationImageTestingPreprocessing,
)
from accelera.src.automl.core.classification_image_training_preprocessing import (  # noqa: E501
    ClassificationImageTrainingPreprocessing,
)
from accelera.src.automl.wrappers.classification_images_after_loader import (
    ClassificationImagesAfterLoader,
)
from accelera.src.automl.core.segmentation_image_training_preprocessing import (
    SegmentationImageTrainingPreprocessing,
)
from accelera.src.automl.core.segmentation_image_testing_preprocessing import (
    SegmentationImageTestingPreprocessing,
)
from accelera.src.automl.wrappers.segmentation_images_after_loader import (
    SegmentationImagesAfterLoader,
)
from accelera.src.automl.wrappers.segmentation_images_after_loader import (
    SegmentationImagesAfterLoader,
)
import torch

training_preprocessor = ClassificationImageTrainingPreprocessing(
    training_folder_images="./PetImages",
    folder_path="PetImagesReport",
    validation_folder_images=None,
    split_training=True,
    val_size=0.2,
    random_state=23,
    images_size=(224, 224),
    augment=True,
    horizontal_flip=True,
    vertical_flip=True,
    rotation=True,
    brightness=True,
    contrast=True
)
training_loader, validation_loader = training_preprocessor.common_preprocessing()
testing_loader, invalid_path = ClassificationImageTestingPreprocessing(
    ["./PetImages/Cat/3.jpg", "./PetImages/Dog/3.jpg"],
    image_class_names=["Cat", "Dog"],
    folder_path="./PetImagesReport",
).common_preprocessing()
print(invalid_path)
images, labels = next(iter(testing_loader))
graph = ClassificationImagesAfterLoader(
    images=images,
    labels=labels,
    folder_path="./PetImagesReport",
    label2class_mapping=training_preprocessor.label2class_mapping,
    title="Testing",
    file_name="Testing",
)
graph.build_graph()
# --------------------------------------------------------
# Segementation

training_loader,val_loader=training_preprocessor = SegmentationImageTrainingPreprocessing(
    training_folder_images="./tumer_data/images",
    training_folder_masks="./tumer_data/masks",
    folder_path="tumerReport",
    binary_mask_threshold=128,
    validation_folder_images=None,
    augment=True,
    horizontal_flip=True,
    vertical_flip=True,
    rotation=True,
    split_training=True,
    val_size=0.2,
    random_state=23,
    images_size=(224, 224),
).common_preprocessing()
images, masks = next(iter(training_loader))
print(images.shape)
print(masks.shape)
print(masks[0].dtype, torch.unique(masks[0]))


testing_loader, invalid_path = SegmentationImageTestingPreprocessing(
    [
        "./tumer_data/images/1.png",
        "./tumer_data/images/2.png",
    ],
    image_masks=[
        "./tumer_data/masks/1.png",
        "./tumer_data/masks/2.png",
    ],
    folder_path="./tumerReport",
).common_preprocessing()
images, masks = next(iter(testing_loader))
print(images.shape)
print(masks.shape)
graph = SegmentationImagesAfterLoader(
    images=images,
    masks=masks,
    folder_path="./tumerReport",
    title="Testing",
    file_name="Testing",
)
graph.build_graph()