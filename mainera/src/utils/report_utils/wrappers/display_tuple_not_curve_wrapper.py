from .metric_display_wrapper import MetricDisplayWrapper
import pandas as pd


class DisplayTupleNotCurveWrapper(MetricDisplayWrapper):
    def __init__(self, metric_name, values, folderpath):
        super().__init__(metric_name, values)
        self.folderpath = folderpath

    def execute(self):
        content = (
            f"### Metric name: {self.metric_name}\n"
            "<div style='display: grid; grid-template-columns: repeat(2,  1fr); gap: 20px;'>\n"
        )

        for value in self.values:
            data = {}
            for i in range(len(value["tuple_argums"]["labels"])):
                data[value["tuple_argums"]["labels"][i]] = value["result"][i]
            table = pd.DataFrame(data).to_html(index=False)
            new_content = (
                f'<div style="overflow-x:auto;max-width:400px;">\n'
                f'<h3 style="color:yellow;">Metric id :{value['metric id']}</h3>\n\n'
                f"{table}\n"
                "</div>\n"
            )
            content = content + new_content
        content = content + "</div>\n"
        return content
