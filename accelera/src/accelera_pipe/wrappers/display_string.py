from accelera.src.accelera_pipe.core.metric_display import MetricDisplay


class DisplayString(MetricDisplay):
    def __init__(self, metric_name, values):
        super().__init__(metric_name, values)

    def execute(self):
        content = f"<div>\n<h3>Metric name: {self.metric_name}</h3>\n"
        for value in self.values:
            new_content = (
                "<div>\n"
                "<h4>\n"
                f"Metric id: {value['metric id']}</h4>\n"
                f"<pre>{value['result'].strip()}</pre>\n"
                "</div>\n"
            )
            content = content + new_content
        content = content + "</div>\n"
        return content
