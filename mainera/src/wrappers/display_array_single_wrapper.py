import numpy as np
import pandas as pd

from mainera.src.wrappers.metric_display_wrapper import MetricDisplayWrapper


class DisplayArraySingleWrapper(MetricDisplayWrapper):
    def __init__(self, metric_name, values):
        super().__init__(metric_name, values)

    def execute(self):
        content = f"### Metric name: {self.metric_name}\n\n"
        ids, results = (
            [value["metric id"] for value in self.values],
            [
                np.array2string(
                    np.array(value["result"]),
                    separator=", ",
                    max_line_width=100,
                )
                for value in self.values
            ],
        )
        data = {"Metric ID": ids, "Metric Value": results}
        table = pd.DataFrame(data).to_html(index=False)
        content = content + table + "\n"
        return content
