from accelera.src.automl.wrappers.graph_base import GraphBase


class TabularGraphBase(GraphBase):
    def __init__(self, df, col_name, target_name, folder_path):
        super().__init__(folder_path)
        self.df = df
        self.col_name = col_name
        self.target_name = target_name
        if self.col_name is not None:
            self.nulls_percent = 100 * self.df[col_name].isna().mean()
            self.graph_df = self.df.dropna(subset=[self.col_name])
