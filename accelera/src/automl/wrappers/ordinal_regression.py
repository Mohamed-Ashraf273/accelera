from accelera.src.automl.wrappers.graph_base import GraphBase
import seaborn as sns
import matplotlib.pyplot as plt
import os


class OrdinalRegression(GraphBase):
    def __init__(self, df, col_name, target_name, folder_path):
        super().__init__(df, col_name, target_name, folder_path)

    def build_graph(self):
        _, ax = plt.subplots(1, 2, figsize=(12, 4))
        sns.countplot(
            data=self.graph_df,
            x=self.col_name,
            ax=ax[0],
            order=sorted(self.graph_df[self.col_name].unique()),
        )
        ax[0].set_title(f"{self.col_name} Distribution")
        ax[0].set_xlabel(self.col_name)
        ax[0].set_ylabel("Count")
        sns.boxplot(
            data=self.graph_df,
            x=self.col_name,
            y=self.target_name,
            ax=ax[1],
            order=sorted(self.graph_df[self.col_name].unique()),
        )
        ax[1].set_title(f"{self.col_name} vs {self.target_name} Distribution")
        ax[1].set_xlabel(self.col_name)
        ax[1].set_ylabel(self.target_name)
        plt.tight_layout()
        plt.savefig(os.path.join(self.folder_path, f"{self.col_name}.png"))
        plt.close()
