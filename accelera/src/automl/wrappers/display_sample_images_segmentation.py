import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from accelera.src.automl.wrappers.graph_base import GraphBase


class DisplaySampleImagesSegmentation(GraphBase):
    def __init__(
        self,
        paths,
        masks,
        mask_type,
        folder_path,
        n_sample=4,
        title="",
        file_name="",
    ):
        super().__init__(folder_path)
        self.title = title
        self.file_name = file_name
        self.n_sample = n_sample
        self.mask_type=mask_type
        self.df = pd.DataFrame({"paths": paths, "masks": masks})
        self.sample = self.df.sample(n=min(self.n_sample, len(paths)), random_state=42)

    def build_graph(self):
        cols=len(self.sample)
        fig, ax = plt.subplots(2,cols , figsize=(4 * cols, 4 * 2))
        fig.suptitle(self.title, fontsize=20)
        ax = np.atleast_2d(ax)
        for i, (image_path, mask_path) in enumerate(
            zip(self.sample["paths"], self.sample["masks"])
        ):
            ax[0][i].axis("off")
            img = plt.imread(image_path)
            ax[0][i].imshow(img)
            ax[0][i].set_title(f"image {os.path.split(image_path)[-1]}")
            ax[1][i].axis("off")
            mask = plt.imread(mask_path)
            if self.mask_type in ["binary","grayscale_intensity"]:
                ax[1][i].imshow(mask,cmap="gray")
            else:
                ax[1][i].imshow(mask,cmap="tab20")
                
            ax[1][i].set_title(f"mask {os.path.split(mask_path)[-1]}")
        plt.tight_layout()
        plt.savefig(os.path.join(self.folder_path, f"{self.file_name}.png"))
        plt.close()
