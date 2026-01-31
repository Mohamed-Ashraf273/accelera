from accelera.src.automl.wrappers.graph_base import GraphBase
import seaborn as sns
import matplotlib.pyplot as plt
import os


class CategoricalRegression(GraphBase):
    def __init__(self, df, col_name, target_name, folder_path):
        super().__init__(df, col_name, target_name, folder_path)
        if self.graph_df[col_name].nunique() > 5:
            top_5_categories = self.graph_df[col_name].value_counts().nlargest(5)
            self.graph_df[col_name] = self.graph_df[col_name].where(
                self.graph_df[col_name].isin(top_5_categories.index), other="Other"
            )
            self.is_top5_applied = True
        else:
            self.is_top5_applied = False

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
        sns.countplot(data=self.graph_df, x=self.col_name, ax=ax[1])
        if self.is_top5_applied:
            ax[1].set_title(f"{self.col_name} Distribution (Top 5 Categories + Other)")
            ax[2].set_title(
                f"{self.col_name} (Top 5 Categories + Other) vs {self.target_name} Distribution"
            )
        else:
            ax[1].set_title(f"{self.col_name} Distribution")
            ax[2].set_title(f"{self.col_name} vs {self.target_name} Distribution")
        ax[1].set_xlabel(self.col_name)
        ax[1].set_ylabel("Count")
        self.graph_df = self.graph_df[[self.col_name, self.target_name]].dropna()
        sns.boxplot(data=self.graph_df, x=self.col_name, y=self.target_name, ax=ax[2])

        ax[2].set_xlabel(self.col_name)
        ax[2].set_ylabel(self.target_name)
        plt.tight_layout()
        plt.savefig(os.path.join(self.folder_path, f"{self.col_name}.png"))
        plt.close()
