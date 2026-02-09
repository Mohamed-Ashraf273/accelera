import os
from collections import Counter

import matplotlib.pyplot as plt
import seaborn as sns

from accelera.src.automl.wrappers.graph_base import GraphBase


class TextGraph(GraphBase):
    def __init__(self, df, col_name, target_name, folder_path):
        super().__init__(df, col_name, target_name, folder_path)
        words = self.graph_df[col_name].str.split().explode()
        top_7_words = Counter(words).most_common(7)
        self.word, self.count = zip(*top_7_words)

    def build_graph(self):
        _, ax = plt.subplots(1, 2, figsize=(12, 4))
        ax[0].pie(
            [float(self.nulls_percent), float(100 - self.nulls_percent)],
            labels=["Nulls", "Not Nulls"],
            autopct="%1.1f%%",
            colors=["#021D25", "#ADD8E6"],
        )
        self.graph_df = self.graph_df[
            [self.col_name, self.target_name]
        ].dropna()
        sns.barplot(x=list(self.word), y=list(self.count), ax=ax[1])
        ax[1].set_title(f"Top 7 words in {self.col_name}")
        ax[1].set_xlabel("Words")
        ax[1].set_ylabel("Count")

        plt.tight_layout()
        plt.savefig(os.path.join(self.folder_path, f"{self.col_name}.png"))
        plt.close()
