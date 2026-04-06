import os

import matplotlib.pyplot as plt
import pandas as pd

from accelera.src.accelera_pipe.core.metric_display import MetricDisplay

plt.style.use("dark_background")


class DisplaySingleNumber(MetricDisplay):
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
        table = (
            pd.DataFrame(data)
            .transpose()
            .round(3)
            .to_html(border=1, justify="center")
        )
        plt.plot(ids, results, marker="o")
        plt.xlabel("Metric ID")
        plt.ylabel("Metric Value")
        plt.title(f"{self.metric_name}")
        plt.savefig(img_path)
        plt.close()
        content = (
            "<div>\n"
            f"<h3>Metric name: {self.metric_name}</h3>\n"
            f"<div><p>Table</p> \n{table}\n</div>\n"
            f"<div><p>Graph</p> \n"
            f"<img src='{self.metric_name}.png' "
            f"alt='{self.metric_name}' /></div>\n"
            "</div>\n"
        )
        return content
