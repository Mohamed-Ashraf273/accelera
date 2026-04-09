import os

import matplotlib.pyplot as plt
import seaborn as sns

from accelera.src.automl.wrappers.tabular_graph_base import TabularGraphBase


class OrdinalRegression(TabularGraphBase):
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
        sns.countplot(
            data=self.graph_df,
            x=self.col_name,
            ax=ax[1],
            order=sorted(self.graph_df[self.col_name].unique()),
        )
        ax[1].set_title(f"{self.col_name} Distribution")
        ax[1].set_xlabel(self.col_name)
        ax[1].set_ylabel("Count")
        for label in ax[1].get_xticklabels():
            label.set_rotation(45)
            label.set_horizontalalignment("right")
        self.graph_df = self.graph_df[[self.col_name, self.target_name]].dropna()
        sns.boxplot(
            data=self.graph_df,
            x=self.col_name,
            y=self.target_name,
            ax=ax[2],
            order=sorted(self.graph_df[self.col_name].unique()),
        )
        ax[2].set_title(f"{self.col_name} vs {self.target_name}\n Distribution")
        ax[2].set_xlabel(self.col_name)
        ax[2].set_ylabel(self.target_name)
        for label in ax[2].get_xticklabels():
            label.set_rotation(45)
            label.set_horizontalalignment("right")
        plt.tight_layout()
        plt.savefig(os.path.join(self.folder_path, f"{self.col_name}.png"))
        plt.close()
