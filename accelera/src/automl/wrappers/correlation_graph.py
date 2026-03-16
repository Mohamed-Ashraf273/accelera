import os

import matplotlib.pyplot as plt
import seaborn as sns

from accelera.src.automl.wrappers.tabular_graph_base import TabularGraphBase


class CorrelationGraph(TabularGraphBase):
    def __init__(self, df, col_name, target_name, folder_path):
        super().__init__(df, col_name, target_name, folder_path)

    def build_graph(self):
        numerical_df = self.df.select_dtypes(include=["number"])
        if not numerical_df.empty:
            corr = numerical_df.dropna().corr()
            _, ax = plt.subplots(1, 1, figsize=(12, 8))
            sns.heatmap(corr, annot=True,fmt=".2f",linewidths=0.5, cmap="coolwarm", ax=ax)
            ax.set_title("Correlation Matrix")
            plt.tight_layout()
            plt.savefig(
                os.path.join(self.folder_path, "correlation_matrix.png")
            )
            plt.close()
