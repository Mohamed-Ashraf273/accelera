import numpy as np

from mainera.src.wrappers.metric_display_wrapper import MetricDisplayWrapper


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
                "<div>\n"
                '<h3 style="color:yellow;">\n'
                f"Metric id :{value['metric id']}</h3>\n\n"
                f"<pre>{array_str}</pre>\n"
                "</div>\n"
            )
            content = content + new_content
        content = content
        return content
