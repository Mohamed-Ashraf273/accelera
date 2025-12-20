from accelera.src.automl.algorithms.algorithm import Algorithm


# genetic using graph (gug) algorithm implementation
class Gug(Algorithm):
    def __init__(self):
        super().__init__()

    def run(
        self,
        X_train,
        y_train,
        X_test=None,
        y_test=None,
        time_limit=None,
        metric="f1_score",
    ):
        print("Running GUG algorithm...")
        return self.pipeline
