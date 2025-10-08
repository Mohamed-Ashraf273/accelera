from .metric_display_wrapper import MetricDisplayWrapper


class DisplayStringWrapper(MetricDisplayWrapper):
    def __init__(self, metric_name, values):
        super().__init__(metric_name, values)

    def execute(self):
        content = f"### Metric name: {self.metric_name}\n"
        for value in self.values:
            new_content = (
                f"<div>\n"
                f'<h3 style="color:yellow;">Metric id :{value['metric id']}</h3>\n\n'
                f"<pre>{value['result'].strip()}</pre>\n"
                "</div>\n"
            )
            content = content + new_content
        content = content
        return content
