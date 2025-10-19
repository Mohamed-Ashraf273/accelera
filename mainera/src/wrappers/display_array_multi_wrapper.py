import numpy as np
import pandas as pd
from mainera.src.wrappers.metric_display_wrapper import MetricDisplayWrapper


class DisplayMultiArrayWrapper(MetricDisplayWrapper):
    def __init__(self, metric_name, values):
        super().__init__(metric_name, values)

    def execute(self):
        content = f"### Metric name: {self.metric_name}\n"
        labels_name = self.handel_labels_name()
        for value in self.values:
            data = {"labels": labels_name, "value": []}
            for i in range(value["result"].shape[0]):
                array_str = np.array2string(
                    np.array(value["result"][i]), separator=", ", max_line_width=80
                )
                data["value"].append(array_str)

            table = pd.DataFrame(data).to_html(index=False)
            new_content = (
                "<div>\n"
                '<h3 style="color:yellow;">\n'
                f"Metric id :{value['metric id']}</h3>\n\n"
                f"{table}\n"
                "</div>\n"
            )
            content = content + new_content
        content = content
        return content
