import os

import matplotlib.pyplot as plt
import numpy as np

from accelera.src.automl.wrappers.graph_base import GraphBase


class ClassificationImagesAfterLoader(GraphBase):
    def __init__(
        self,
        images,
        labels,
        label2class_mapping,
        folder_path,
        title="",
        file_name="",
    ):
        super().__init__(folder_path)
        self.images = images
        self.labels = labels
        self.title = title
        self.file_name = file_name
        self.label2class_mapping = label2class_mapping

    def build_graph(self):
        cols = len(self.images)
        fig, ax = plt.subplots(1, cols, figsize=(4 * cols, 4))
        if cols == 1:
            ax = [ax]
        fig.suptitle(self.title, fontsize=20)
        for i, img in enumerate(self.images):
            img = img.permute(1, 2, 0).numpy()
            img = np.clip(img, 0, 1)
            label = self.labels[i].item()
            class_name = self.label2class_mapping[label]
            ax[i].imshow(img)
            ax[i].set_title(class_name)
            ax[i].axis("off")

        plt.tight_layout()
        plt.savefig(os.path.join(self.folder_path, f"{self.file_name}.png"))
        plt.close()
