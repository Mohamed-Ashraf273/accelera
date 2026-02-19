import os

import matplotlib.pyplot as plt
import numpy as np

from accelera.src.automl.wrappers.graph_base import GraphBase


class SegmentationImagesAfterLoader(GraphBase):
    def __init__(
        self,
        images,
        masks,
        mask_type,
        folder_path,
        title="",
        file_name="",
    ):
        super().__init__(folder_path)
        self.images = images
        self.masks = masks
        self.title = title
        self.file_name = file_name
        self.mask_type = mask_type

    def build_graph(self):
        cols = len(self.images)
        fig, ax = plt.subplots(2, cols, figsize=(4 * cols, 4*2))
        ax = np.atleast_2d(ax)

        fig.suptitle(self.title, fontsize=20)
        for i, (img,mask) in enumerate(zip(self.images,self.masks)):
            img = img.numpy().transpose(1, 2, 0)
            img = np.clip(img, 0, 1)
            ax[0][i].imshow(img)
            ax[0][i].set_title("image")
            ax[0][i].axis("off")
            mask=mask.numpy()
            if mask.ndim==3:
                mask=mask[0]
            if self.mask_type=="binary":
                ax[1][i].imshow(mask,cmap="gray")
            else:
                ax[1][i].imshow(mask,cmap="tab20")
            ax[1][i].set_title("mask")
            ax[1][i].axis("off")
                
                
            

        plt.tight_layout()
        plt.savefig(os.path.join(self.folder_path, f"{self.file_name}.png"))
        plt.close()
