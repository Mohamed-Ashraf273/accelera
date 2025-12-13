import pandas as pd

from accelera.src.core.metric_display import MetricDisplay


class DisplayTupleNotCurve(MetricDisplay):
    def __init__(self, metric_name, values, folderpath):
        super().__init__(metric_name, values)
        self.folderpath = folderpath

    def execute(self):
        content = (
            f"### Metric name: {self.metric_name}\n"
            "<div style='display: grid; "
            "grid-template-columns: repeat(2,  1fr); gap: 20px;'>\n"
        )

        for value in self.values:
            data = {}
            labels_name = self.handle_name(
                "labels_name", len(value["result"][0])
            )
            headers_name = self.handle_name(
                "headers_name", len(self.values[0]["result"])
            )
            data["labels"] = labels_name
            for i in range(len(headers_name)):
                data[headers_name[i]] = value["result"][i]
            table = pd.DataFrame(data).to_html(index=False)
            new_content = (
                '<div style="overflow-x:auto;max-width:400px;">\n'
                '<h3 style="color:yellow;">\n'
                f"Metric id :{value['metric id']}</h3>\n\n"
                f"{table}\n"
                "</div>\n"
            )
            content = content + new_content
        content = content + "</div>\n"
        return content
