import numpy as np
import pandas as pd
from mainera.src.wrappers.metric_display_wrapper import MetricDisplayWrapper


class DisplayArraySingleWrapper(MetricDisplayWrapper):
    def __init__(self, metric_name, values):
        super().__init__(metric_name, values)

    def execute(self):
        content = f"### Metric name: {self.metric_name}\n\n"
        ids = [value["metric id"] for value in self.values]
        labels_name = self.handel_labels_name()
        data = {"Metric ID": ids}
        for value in self.values:
            for i in range(len(value["result"])):
                if labels_name[i] not in data:
                    data[labels_name[i]] = [value["result"][i]]
                else:
                    data[labels_name[i]].append([value["result"][i]])

        table = pd.DataFrame(data).to_html(index=False)
        content = content + table + "\n"
        return content
