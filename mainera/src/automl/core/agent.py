class AutoMLAgent:
    def __init__(
        self, time_limit=None, metric=None, n_trials=None, use_llm=False
    ):
        self.time_limit = time_limit
        self.metric = metric
        self.n_trials = n_trials
        self.use_llm = use_llm

    def fit(self, X, y, task):
        print("Fitting the AutoML agent...")
        pass
