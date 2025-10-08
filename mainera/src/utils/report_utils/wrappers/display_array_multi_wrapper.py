from .metric_display_wrapper import MetricDisplayWrapper
import numpy as np


class DisplayMultiArrayWrapper(MetricDisplayWrapper):
    def __init__(self, metric_name, values):
        super().__init__(metric_name, values)

    def execute(self):
        content = f"### Metric name: {self.metric_name}\n"
        for value in self.values:
            array_str = np.array2string(
                np.array(value["result"]), separator=", ", max_line_width=80
            )
            new_content = (
                f"<div>\n"
                f"<h3>Metric id :{value['metric id']}</h3>\n\n"
                f"<pre>{array_str}</pre>\n"
                "</div>\n"
            )
            content = content + new_content
        content = content
        return content
