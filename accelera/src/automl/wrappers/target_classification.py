from accelera.src.automl.wrappers.graph_base import GraphBase
import seaborn as sns
import matplotlib.pyplot as plt
import os


class TargetClassification(GraphBase):
    def __init__(self, df, col_name, target_name, folder_path):
        super().__init__(df, col_name, target_name, folder_path)

    def build_graph(self):
        _, ax = plt.subplots(1, 1, figsize=(12, 4))
        sns.countplot(data=self.graph_df, x=self.target_name, ax=ax)
        ax.set_title(f"{self.target_name} Distribution")
        ax.set_xlabel(self.target_name)
        ax.set_ylabel("Count")
        plt.tight_layout()
        plt.savefig(os.path.join(self.folder_path, f"{self.col_name}.png"))
        plt.close()
