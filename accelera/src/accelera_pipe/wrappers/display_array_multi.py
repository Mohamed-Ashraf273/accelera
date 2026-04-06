import numpy as np
import pandas as pd

from accelera.src.accelera_pipe.core.metric_display import MetricDisplay


class DisplayMultiArray(MetricDisplay):
    def __init__(self, metric_name, values):
        super().__init__(metric_name, values)

    def execute(self):
        content = f"<div>\n<h3>Metric name: {self.metric_name}</h3>\n"
        labels_name = self.handle_name(
            "labels_name", self.values[0]["result"].shape[0]
        )
        headers_name = self.handle_name(
            "headers_name", self.values[0]["result"].shape[1]
        )
        for value in self.values:
            data = {"labels": labels_name}
            for row_idx in range(value["result"].shape[0]):
                for col_idx in range(len(value["result"][row_idx])):
                    array_str = np.array2string(
                        np.array(value["result"][row_idx][col_idx]),
                        separator=", ",
                        max_line_width=80,
                    )
                    if headers_name[col_idx] not in data:
                        data[headers_name[col_idx]] = [array_str]
                    else:
                        data[headers_name[col_idx]].append(array_str)

            table = pd.DataFrame(data).to_html(
                index=False, border=1, justify="center"
            )
            new_content = (
                "<div>\n"
                "<h4>\n"
                f"Metric id :{value['metric id']}</h4>\n\n"
                f"{table}\n"
                "</div>\n"
            )
            content = content + new_content + "</div>\n"
        content = content
        return content
