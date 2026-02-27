import os

import matplotlib.pyplot as plt

from accelera.src.automl.wrappers.graph_base import GraphBase


class Segmentation_data_summary(GraphBase):
    def __init__(
        self,
        valid,
        invalid,
        folder_path,
        title="",
        file_name="",
    ):
        super().__init__(folder_path)
        self.title = title
        self.file_name = file_name

        self.length = [1, 0]
        if valid is not None and len(valid) > 0:
            total_length = len(valid) + len(invalid)
            self.length[0] = len(valid) / total_length
            self.length[1] = len(invalid) / total_length

    def build_graph(self):
        fig, ax = plt.subplots(1, 1, figsize=(12, 5))

        fig.suptitle(self.title, fontsize=20)
        ax.pie(
            [float(self.length[0] * 100), float(100 * self.length[1])],
            labels=[
                "valid compination of images & masks",
                "invalid compination of images & masks",
            ],
            autopct="%1.1f%%",
            colors=["#021D25", "#ADD8E6"],
        )
        ax.set_title("Valid vs Invalid percentage")
        plt.tight_layout()
        plt.savefig(os.path.join(self.folder_path, f"{self.file_name}.png"))
        plt.close()
