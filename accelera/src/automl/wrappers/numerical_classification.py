import os

import matplotlib.pyplot as plt
import seaborn as sns

from accelera.src.automl.wrappers.tabular_graph_base import TabularGraphBase


class NumericalClassification(TabularGraphBase):
    def __init__(self, df, col_name, target_name, folder_path):
        super().__init__(df, col_name, target_name, folder_path)

    def build_graph(self):
        _, ax = plt.subplots(1, 3, figsize=(12, 4))
        # pie plot of nulls percent
        ax[0].pie(
            [float(self.nulls_percent), float(100 - self.nulls_percent)],
            labels=["Nulls", "Not Nulls"],
            autopct="%1.1f%%",
            colors=["#021D25", "#ADD8E6"],
        )
        ax[0].set_title(f"{self.col_name} Null percentage")
        sns.histplot(data=self.graph_df, x=self.col_name, ax=ax[1], kde=True)
        ax[1].set_title(f"{self.col_name} Distribution")
        ax[1].set_xlabel(self.col_name)
        ax[1].set_ylabel("Count")
        self.graph_df = self.graph_df[[self.col_name, self.target_name]].dropna()
        sns.boxplot(data=self.graph_df, x=self.target_name, y=self.col_name, ax=ax[2])
        ax[2].set_title(f"{self.col_name} Distribution by {self.target_name}")
        ax[2].set_xlabel(self.target_name)
        ax[2].set_ylabel(self.col_name)
        plt.tight_layout()
        plt.savefig(os.path.join(self.folder_path, f"{self.col_name}.png"))
        plt.close()
