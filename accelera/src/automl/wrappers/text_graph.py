from accelera.src.automl.wrappers.graph_base import GraphBase
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
import os


class TextGraph(GraphBase):
    def __init__(self, df, col_name, target_name, folder_path):
        super().__init__(df, col_name, target_name, folder_path)
        words = self.graph_df[col_name].str.split().explode()
        top_5_words = Counter(words).most_common(5)
        self.word, self.count = zip(*top_5_words)

    def build_graph(self):
        _, ax = plt.subplots(1, 1, figsize=(12, 4))
        sns.barplot(x=list(self.word), y=list(self.count), ax=ax)
        ax.set_title(f"Top 5 words in {self.col_name}")
        ax.set_xlabel("Words")
        ax.set_ylabel("Count")

        plt.tight_layout()
        plt.savefig(os.path.join(self.folder_path, f"{self.col_name}.png"))
        plt.close()
