from mainera.src.core.metric_display import MetricDisplay


class DisplayString(MetricDisplay):
    def __init__(self, metric_name, values):
        super().__init__(metric_name, values)

    def execute(self):
        content = f"### Metric name: {self.metric_name}\n"
        for value in self.values:
            new_content = (
                "<div>\n"
                '<h3 style="color:yellow;">\n'
                f"Metric id: {value['metric id']}</h3>\n\n"
                f"<pre>{value['result'].strip()}</pre>\n"
                "</div>\n"
            )
            content = content + new_content
        content = content
        return content
