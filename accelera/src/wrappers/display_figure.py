import os

import matplotlib.pyplot as plt

from accelera.src.core.metric_display import MetricDisplay
from accelera.src.utils.accelera_utils import create_folder

plt.style.use("dark_background")


class DisplayFigure(MetricDisplay):
    def __init__(self, metric_name, values, folderpath):
        super().__init__(metric_name, values)
        self.folderpath = folderpath

    def execute(self):
        content = (
            "<div>\n"
            f"<h3>Metric name: {self.metric_name}</h3>\n"
            "<div class='metric-container'>\n"
        )
        sub_folder_path = os.path.join(self.folderpath, self.metric_name)
        create_folder(sub_folder_path)
        for value in self.values:
            plot_func = value["plot_func"]
            result = value["result"]
            img_path = os.path.join(
                sub_folder_path, f"{value['metric id']}.png"
            )
            plt = plot_func(result)
            if plt is None:
                raise ValueError("The plot_func must return the plt object")
            plt.title(f"{self.metric_name} - Metric ID: {value['metric id']}")
            plt.savefig(img_path)
            plt.close()
            img_src = os.path.join(
                self.metric_name, f"{value['metric id']}.png"
            )
            img_alt = f"{self.metric_name}_{value['metric id']}"
            new_content = (
                f"<div>\n<img src='{img_src}' alt='{img_alt}' />\n</div>\n"
            )
            content = content + new_content
        content = content + "</div>\n</div>\n"
        return content
