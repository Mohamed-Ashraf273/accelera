class PreprocessingBase:
    def __init__(self, df, target_col, problem_type="classification"):
        self.problem_type = problem_type
        self.target_col = target_col
        self.df = df
        if self.problem_type not in ["classification", "regression"]:
            raise ValueError(
                "problem_type must be either 'classification' or 'regression'"
            )
        if df is None:
            raise ValueError("Dataframe cannot be None")
        if target_col not in df.columns:
            raise ValueError("target_col must be one of the dataframe columns")

    def lower_data(self):
        for col in self.df.columns:
            if self.df[col].dtype == "object":
                self.df[col] = self.df[col].str.lower()

    def common_preprocessing(self):
        raise NotImplementedError("Must implement common_preprocessing method.")
