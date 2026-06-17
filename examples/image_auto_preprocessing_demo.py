
from accelera.src.automl.core.classification_image_training_preprocessing import (  # noqa: E501
    ClassificationImageTrainingPreprocessing,
)

from accelera.src.automl.core.segmentation_image_training_preprocessing import (
    SegmentationImageTrainingPreprocessing,
)
import json
def get_data_set_info():
    with open("auto_preproceesing_ds.json", "r") as f:
        ds = json.loads(f.read())
    return ds
def classifcation_problem(training_folder_images,folder_path,augment):
    augment=augment=="True"
    training_preprocessor = ClassificationImageTrainingPreprocessing(
        training_folder_images=training_folder_images,
        folder_path=folder_path,
        validation_folder_images=None,
        split_training=True,
        val_size=0.2,
        random_state=23,
        images_size=(224, 224),
        augment=augment
    )
    return training_preprocessor.common_preprocessing()

    
def segemenation_problem(training_folder_images,training_folder_masks,folder_path,augment):
    augment=augment=="True"
    return (
        SegmentationImageTrainingPreprocessing(
            training_folder_images=training_folder_images,
            training_folder_masks=training_folder_masks,
            folder_path=folder_path,
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
    )
def main():
    ds = get_data_set_info()

    for dataset_type, datasetsObj in ds.items():

        if dataset_type != "image_dataset":
            continue
        for problem_type, datasets in datasetsObj.items():
            for _,ds_info in datasets.items():
                if problem_type=="classification":
                    classifcation_problem(ds_info["train_folder"],ds_info["report_path"],ds_info["augment"])
                else:
                    segemenation_problem(ds_info["train_folder_images"],ds_info["train_folder_masks"],ds_info["report_path"],ds_info["augment"])
                    

if __name__ == "__main__":
    main()
