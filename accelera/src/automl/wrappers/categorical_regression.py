from accelera.src.automl.wrappers.graph_base import GraphBase
import seaborn as sns
import matplotlib.pyplot as plt


class CategoricalRegression(GraphBase):
    def __init__(self, df, col_name, target_name):
        super().__init__(df, col_name, target_name)
        if self.graph_df[col_name].nunique() > 5:
            top_5_categories = self.graph_df[col_name].value_counts().nlargest(5)
            self.graph_df[col_name] = self.graph_df[col_name].where(
                self.graph_df[col_name].isin(top_5_categories.index), other="Other"
            )

    def build_graph(self):
        _, ax = plt.subplots(1, 2, figsize=(12, 4))
        sns.countplot(data=self.graph_df, x=self.col_name, ax=ax[0])
        ax[0].set_title(f"{self.col_name} Distribution")
        ax[0].set_xlabel(self.col_name)
        ax[0].set_ylabel("Count")
        sns.boxplot(data=self.graph_df, x=self.col_name, y=self.target_name, ax=ax[1])
        ax[1].set_title(f"{self.col_name} vs {self.target_name} Distribution")
        ax[1].set_xlabel(self.col_name)
        ax[1].set_ylabel(self.target_name)
        plt.tight_layout()
        plt.show()
