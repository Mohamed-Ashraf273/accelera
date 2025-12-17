import pandas as pd

from accelera.src.core.metric_display import MetricDisplay


class DisplayArraySingle(MetricDisplay):
    def __init__(self, metric_name, values):
        super().__init__(metric_name, values)

    def execute(self):
        content = f"<div>\n<h3>Metric name: {self.metric_name}</h3>\n"
        ids = [value["metric id"] for value in self.values]
        labels_name = self.handle_name(
            "labels_name", len(self.values[0]["result"])
        )
        data = {"Metric ID": ids}
        for value in self.values:
            for i in range(len(value["result"])):
                if labels_name[i] not in data:
                    data[labels_name[i]] = [value["result"][i]]
                else:
                    data[labels_name[i]].append(value["result"][i])

        table = pd.DataFrame(data).to_html(
            index=False, border=1, justify="center"
        )
        content = content + table + "</div>\n"
        return content
