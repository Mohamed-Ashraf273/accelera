from accelera.src.automl.core.classification_image_training_preprocessing import (
    ClassificationImageTrainingPreprocessing,
)

training_preprocessor = ClassificationImageTrainingPreprocessing(
    training_folder_images="./PetImages",
    folder_path="PetImagesReport",
    validation_folder_images=None,
    split_training=True,
    val_size=0.2,
    random_state=23,
    images_size=(32, 32),
)
training_preprocessor.common_preprocessing()