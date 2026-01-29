import os
from accelera.src.automl.wrappers.graph_base import GraphBase
import seaborn as sns
import matplotlib.pyplot as plt


class CorrelationGraph(GraphBase):
    def __init__(self, df, col_name, target_name, folder_path):
        super().__init__(df, col_name, target_name, folder_path)

    def build_graph(self):
        numerical_df = self.df.select_dtypes(include=["number"])
        corr = numerical_df.dropna().corr()
        _, ax = plt.subplots(1, 1, figsize=(12, 4))
        sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
        ax.set_title("Correlation Matrix")
        plt.tight_layout()
        plt.savefig(os.path.join(self.folder_path, f"correlation_matrix.png"))
        plt.close()
