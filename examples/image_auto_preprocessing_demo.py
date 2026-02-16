from accelera.src.automl.core.classification_image_testing_preprocessing import (  # noqa: E501
    ClassificationImageTestingPreprocessing,
)
from accelera.src.automl.core.classification_image_training_preprocessing import (  # noqa: E501
    ClassificationImageTrainingPreprocessing,
)
from accelera.src.automl.wrappers.classification_images_after_loader import (
    ClassificationImagesAfterLoader,
)

training_preprocessor = ClassificationImageTrainingPreprocessing(
    training_folder_images="./PetImages",
    folder_path="PetImagesReport",
    validation_folder_images=None,
    split_training=True,
    val_size=0.2,
    random_state=23,
    images_size=(224, 224),
)
training_loader, validation_loader = (
    training_preprocessor.common_preprocessing()
)
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
