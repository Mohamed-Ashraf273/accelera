from accelera.src.automl.wrappers.graph_base import GraphBase
import seaborn as sns
import matplotlib.pyplot as plt


class TargetClassification(GraphBase):
    def __init__(self, df, col_name, target_name):
        super().__init__(df, col_name, target_name)

    def build_graph(self):
        _, ax = plt.subplots(1, 1, figsize=(12, 4))
        sns.countplot(data=self.graph_df, x=self.target_name, ax=ax)
        ax.set_title(f"{self.target_name} Distribution")
        ax.set_xlabel(self.target_name)
        ax.set_ylabel("Count")
