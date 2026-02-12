import os

import matplotlib.pyplot as plt
import seaborn as sns

from accelera.src.automl.wrappers.tabular_graph_base import TabularGraphBase


class TargetClassification(TabularGraphBase):
    def __init__(self, df, col_name, target_name, folder_path):
        super().__init__(df, col_name, target_name, folder_path)

    def build_graph(self):
        _, ax = plt.subplots(1, 2, figsize=(12, 4))
        # pie plot of nulls percent
        ax[0].pie(
            [float(self.nulls_percent), float(100 - self.nulls_percent)],
            labels=["Nulls", "Not Nulls"],
            autopct="%1.1f%%",
            colors=["#021D25", "#ADD8E6"],
        )
        sns.countplot(data=self.graph_df, x=self.col_name, ax=ax[1])
        ax[1].set_title(f"{self.col_name} Distribution")
        ax[1].set_xlabel(self.col_name)
        ax[1].set_ylabel("Count")
        plt.tight_layout()
        plt.savefig(os.path.join(self.folder_path, f"{self.col_name}.png"))
        plt.close()
