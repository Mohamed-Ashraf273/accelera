import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from accelera.src.automl.wrappers.graph_base import GraphBase


class ImageLabelClassification(GraphBase):
    def __init__(
        self,
        labels,
        label2class_mapping,
        invalid_labels,
        folder_path,
        title="",
        file_name="",
    ):
        super().__init__(folder_path)
        label2class_mapping = label2class_mapping
        self.valid_df = pd.DataFrame({"labels": labels})
        self.valid_df["labels"] = self.valid_df["labels"].map(
            label2class_mapping
        )
        self.total_df = None
        self.invalid_df = None
        self.title = title
        self.file_name = file_name
        if invalid_labels is not None and len(invalid_labels) > 0:
            self.invalid_df = pd.DataFrame({"labels": invalid_labels})
            self.invalid_df["labels"] = self.invalid_df["labels"].map(
                label2class_mapping
            )
            self.total_df = pd.concat(
                [self.valid_df, self.invalid_df], axis=0, ignore_index=True
            )
            self.length = [0, 0]
            self.length[0] = len(self.valid_df["labels"]) / len(
                self.total_df["labels"]
            )
            self.length[1] = len(self.invalid_df["labels"]) / len(
                self.total_df["labels"]
            )
        else:
            self.total_df = self.valid_df
            self.length = [1, 0]

    def build_graph(self):
        if self.invalid_df is not None:
            fig, ax = plt.subplots(1, 4, figsize=(16, 5))
        else:
            fig, ax = plt.subplots(1, 3, figsize=(12, 5))

        fig.suptitle(self.title, fontsize=20)
        sns.countplot(data=self.total_df, x="labels", ax=ax[0])
        ax[0].set_title("Total images classes distribution")
        ax[0].set_xlabel("Classes")
        ax[0].set_ylabel("Count")
        ax[1].pie(
            [float(self.length[0] * 100), float(100 * self.length[1])],
            labels=["valid images", "invalid images"],
            autopct="%1.1f%%",
            colors=["#021D25", "#ADD8E6"],
        )
        ax[1].set_title("Valid vs Invalid percentage")
        sns.countplot(data=self.valid_df, x="labels", ax=ax[2])
        ax[2].set_title("Valid images classes distribution")
        ax[2].set_xlabel("Classes")
        ax[2].set_ylabel("Count")
        if self.invalid_df is not None:
            sns.countplot(data=self.invalid_df, x="labels", ax=ax[3])
            ax[3].set_title("Invalid images classes distribution")
            ax[3].set_xlabel("Classes")
            ax[3].set_ylabel("Count")
        plt.tight_layout()
        plt.savefig(os.path.join(self.folder_path, f"{self.file_name}.png"))
        plt.close()
