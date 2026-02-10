from accelera.src.automl.core.tabular_preprocessing_base import (
    TabularPreprocessingBase,
)
from accelera.src.automl.utils.preprocessing import check_path_exists, load_pickle


class TestingTabularPreprocessingBase(TabularPreprocessingBase):
    def __init__(self, df, folder_path=None):
        super().__init__(df, folder_path)
        check_path_exists(self.folder_path, "target_preprocessor.pkl")
        check_path_exists(self.folder_path, "training_preprocessor.pkl")
        self.target_preprocessor = load_pickle(
            self.folder_path, "target_preprocessor.pkl"
        )
        self.training_preprocessor = load_pickle(
            self.folder_path, "training_preprocessor.pkl"
        )
        if self.target_preprocessor is None:
            raise ValueError(
                "target_preprocessor cannot be None please run training preprocessing first to create the target preprocessor"
            )

        if self.training_preprocessor is None:
            raise ValueError(
                "training_preprocessor cannot be None please run training preprocessing first to create the training preprocessor"
            )
        self.features_only = False
