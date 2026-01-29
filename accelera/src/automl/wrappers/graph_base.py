import os
import matplotlib.pyplot as plt


class GraphBase:
    def __init__(self, df, col_name, target_name, folder_path):
        self.df = df
        self.col_name = col_name
        self.target_name = target_name
        self.folder_path = os.path.join(folder_path, "graphs")
        if not os.path.exists(self.folder_path):
            os.makedirs(self.folder_path)
        if col_name == target_name:
            self.graph_df = df[[col_name]].dropna()
        else:
            self.graph_df = df[[col_name, target_name]].dropna()
        plt.style.use("dark_background")

    def build_graph(self):
        raise NotImplementedError("Subclasses should implement this method")
