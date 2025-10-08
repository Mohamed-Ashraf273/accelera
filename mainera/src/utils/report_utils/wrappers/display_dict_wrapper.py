from .metric_display_wrapper import MetricDisplayWrapper
import pandas as pd


class DisplayDictWrapper(MetricDisplayWrapper):
    def __init__(self, metric_name, values):
        super().__init__(metric_name, values)

    def execute(self):
        content = (
            f"### Metric name: {self.metric_name}\n"
            "<div style='display: grid; grid-template-columns: repeat(2, 1fr); gap: 20px;'>\n"
        )
        for value in self.values:
            table = pd.DataFrame(value["result"]).transpose().round(3).to_html()
            new_content = f'<div style="overflow-x:auto;max-width:400px;"><h3 style="color:yellow;">Metric id :{value["metric id"]}</h3>\n\n {table}\n</div>\n'
            content = content + new_content
        content = content + "</div>\n"
        return content
