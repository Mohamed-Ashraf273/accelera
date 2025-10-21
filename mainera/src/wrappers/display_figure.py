import os

import matplotlib.pyplot as plt

from mainera.src.core.metric_display import MetricDisplay
from mainera.src.utils.mainera_utils import create_folder

plt.style.use("dark_background")


class DisplayFigure(MetricDisplay):
    def __init__(self, metric_name, values, folderpath):
        super().__init__(metric_name, values)
        self.folderpath = folderpath

    def execute(self):
        content = (
            f"### Metric name: {self.metric_name}\n\n"
            "<div style='display: grid; "
            "grid-template-columns: repeat(2, 1fr); gap: 20px;'>\n"
        )
        sub_folder_path = os.path.join(self.folderpath, self.metric_name)
        create_folder(sub_folder_path)
        for value in self.values:
            plot_func = value["plot_func"]
            result = value["result"]
            img_path = os.path.join(sub_folder_path, f"{value['metric id']}.png")
            plt = plot_func(result)
            if plt is None:
                raise ValueError("The plot_func must return the plt object")
            plt.title(f"{self.metric_name} - Metric ID: {value['metric id']}")
            plt.savefig(img_path)
            plt.close()
            new_content = (
                f'<div  style="overflow-x:auto;max-width:400px;">\n\n'
                f"![{self.metric_name}_{value['metric id']}]"
                f"({os.path.join(self.metric_name,value['metric id'])}.png)\n"
                "</div>\n"
            )
            content = content + new_content
        content = content + "</div>\n"
        return content
