import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
from accelera.src.automl.wrappers.graph_base import GraphBase


class DisplaySampleImagesClassification(GraphBase):
    def __init__(
        self,
        paths,
        labels,
        label2class_mapping,
        folder_path,
        n_sample=4,
        title="",
        file_name="",
    ):
        super().__init__(folder_path)
        self.label2class_mapping = label2class_mapping
        self.title = title
        self.file_name = file_name
        self.n_sample = n_sample
        classes = list(self.label2class_mapping.keys())
        self.df = pd.DataFrame({"paths": paths, "labels": labels})
        self.temp_df = {}
        for class_idx in classes:
            class_paths = self.df[self.df["labels"] == class_idx]
            class_df = class_paths.sample(
                n=min(self.n_sample, len(class_paths)), random_state=42
            )
            self.temp_df[label2class_mapping[class_idx]] = class_df["paths"]

    def build_graph(self):
        classes = list(self.temp_df.keys())
        num_rows = len(classes)
        fig, ax = plt.subplots(
            num_rows, self.n_sample, figsize=(4 * self.n_sample, 4 * num_rows)
        )
        fig.suptitle(self.title, fontsize=20)
        ax = np.atleast_2d(ax)
        for row, class_name in enumerate(classes):
            paths = self.temp_df[class_name]
            for col, path in enumerate(paths):
                ax[row][col].axis("off")
                img = plt.imread(path)
                ax[row][col].imshow(img)
                ax[row][col].set_title(class_name)
        plt.tight_layout()
        plt.savefig(os.path.join(self.folder_path, f"{self.file_name}.png"))
        plt.close()
