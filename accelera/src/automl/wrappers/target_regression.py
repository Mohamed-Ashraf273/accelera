from accelera.src.automl.wrappers.graph_base import GraphBase
import seaborn as sns
import matplotlib.pyplot as plt


class TargetRegression(GraphBase):
    def __init__(self, df, col_name, target_name):
        super().__init__(df, col_name, target_name)

    def build_graph(self):
        _, ax = plt.subplots(1, 2, figsize=(12, 4))
        sns.histplot(data=self.graph_df, x=self.target_name, ax=ax[0], kde=True)
        ax[0].set_title(f"{self.target_name} Distribution")
        ax[0].set_xlabel(self.target_name)
        ax[0].set_ylabel("Count")
        sns.boxplot(data=self.graph_df, y=self.target_name, ax=ax[1])
        ax[1].set_title(f"{self.col_name} boxplot")
        ax[1].set_ylabel(self.target_name)
        plt.tight_layout()
        plt.show()
