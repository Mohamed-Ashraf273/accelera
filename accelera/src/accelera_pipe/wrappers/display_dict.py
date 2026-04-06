import pandas as pd

from accelera.src.accelera_pipe.core.metric_display import MetricDisplay


class DisplayDict(MetricDisplay):
    def __init__(self, metric_name, values):
        super().__init__(metric_name, values)

    def execute(self):
        content = (
            "<div>\n"
            f"<h3>Metric name: {self.metric_name}</h3>\n"
            "<div class='metric-container'>\n"
        )
        for value in self.values:
            table = (
                pd.DataFrame(value["result"])
                .transpose()
                .round(3)
                .to_html(border=1, justify="center")
            )
            new_content = "<div>\n"
            new_content += "<h4>\n"
            new_content += (
                f"Metric id :{value['metric id']}</h4>\n {table}\n</div>\n"
            )
            content = content + new_content
        content = content + "</div>\n</div>\n"

        return content
