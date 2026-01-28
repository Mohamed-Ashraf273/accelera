class GraphBase:
    def __init__(self, df, col_name, target_name):
        self.df = df
        self.col_name = col_name
        self.target_name = target_name
        if col_name == target_name:
            self.graph_df = df[[col_name]].dropna()
        else:
            self.graph_df = df[[col_name, target_name]].dropna()

    def build_graph(self):
        raise NotImplementedError("Subclasses should implement this method")
