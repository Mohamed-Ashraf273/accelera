class MetricDisplay:
    def __init__(self, metric_name, values):
        self.metric_name = metric_name
        self.values = values

    def handle_labels_name(self):
        if self.values[0]["labels_name"]:
            if not isinstance(self.values[0]["labels_name"], list):
                raise TabError("'labels_name must be a list'")
            if len(self.values[0]["labels_name"]) != len(
                self.values[0]["result"]
            ):
                raise ValueError(
                    "length of labels_name dose not match the "
                    "length of the result from the metric "
                    "The length of  result from the "
                    f"metric is {len(self.values[0]['result'])} "
                    "but the length of the 'labels_name' "
                    f"is {len(self.values[0]['labels_name'])}'"
                )
            labels_name = self.values[0]["labels_name"]
        else:
            labels_name = list(range(len(self.values[0]["result"])))
        return labels_name

    def execute(self):
        pass
