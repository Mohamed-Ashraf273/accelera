import os

import matplotlib.pyplot as plt
import pandas as pd

from .metric_display_wrapper import MetricDisplayWrapper

plt.style.use("dark_background")


class DisplaySignleNumberWrapper(MetricDisplayWrapper):
    def __init__(self, metric_name, values, folderpath):
        super().__init__(metric_name, values)
        self.folderpath = folderpath

    def execute(self):
        ids, results = (
            [value["metric id"] for value in self.values],
            [value["result"] for value in self.values],
        )
        data = {
            "Metric ID": ids,
            "Metric Value": results,
        }
        img_path = os.path.join(self.folderpath, f"{self.metric_name}.png")
        table = pd.DataFrame(data).transpose().round(3).to_html()
        plt.plot(ids, results, marker="o")
        plt.xlabel("Metric ID")
        plt.ylabel("Metric Value")
        plt.title(f"{self.metric_name}")
        plt.savefig(img_path)
        content = (
            f"### Metric name: {self.metric_name}\n\n"
            f"Table \n{table}\n\n"
            f"Graph \n\n![{self.metric_name}]({self.metric_name}.png)\n"
        )
        return content
