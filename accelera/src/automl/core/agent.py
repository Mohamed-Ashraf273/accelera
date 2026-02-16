from accelera.src.automl.core.classical_training_preprocessing import (
    ClassicalTrainingPreprocessing,
)
from accelera.src.automl.utils.sampler import sample
from accelera.src.utils.accelera_utils import print_msg


class AutoAccelera:
    def __init__(self, algorithm="default", problem_type=None):
        self.algorithm = self._get_algo(algorithm)
        self.problem_type = problem_type

    def _get_algo(self, algorithm_name):
        if algorithm_name == "default":
            from accelera.src.automl.algorithms.algorithm import Algorithm

            return Algorithm()
        else:
            raise ValueError(f"Algorithm '{algorithm_name}' is not supported.")

    def get_pipeline(self, df, target_column: str):
        print_msg("Getting pipeline for the given dataset...", level="info")
        tp = ClassicalTrainingPreprocessing(
            df,
            target_column,
            problem_type=self.problem_type,
            folder_path="./preprocessing_temp",
        )
        X_train, y_train, X_test, y_test = tp.common_preprocessing()

        if X_train.shape[0] > 10000:
            X_train, y_train, _ = sample(X_train, y_train)

        if X_test is not None and y_test is not None:
            if X_test.shape[0] > 2000:
                X_test, y_test, _ = sample(X_test, y_test)

        best_pipeline = self.algorithm.run(
            X_train,
            y_train,
            X_test,
            y_test,
        )

        return best_pipeline
