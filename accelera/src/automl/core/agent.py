from accelera.src.automl.utils.preprocessing import common_preprocessing
from accelera.src.automl.utils.sampler import sample
from accelera.src.utils.accelera_utils import print_msg


class AutoAccelera:
    def __init__(self, algorithm="default"):
        self.algorithm = self._get_algo(algorithm)

    def _get_algo(self, algorithm_name):
        if algorithm_name == "default":
            from accelera.src.automl.algorithms.algorithm import Algorithm

            return Algorithm()

        elif algorithm_name == "gug":
            from accelera.src.automl.algorithms.gug import Gug

            return Gug()
        else:
            raise ValueError(f"Algorithm '{algorithm_name}' is not supported.")

    def get_pipeline(self, df, target_column: str):
        print_msg("Getting pipeline for the given dataset...", level="info")
        X_train, y_train, X_test, y_test = common_preprocessing(
            df, target_column
        )

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
