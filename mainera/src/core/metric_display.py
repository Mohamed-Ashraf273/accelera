from abc import ABC
from abc import abstractmethod


class MetricDisplay(ABC):
    def __init__(self, metric_name, values):
        self.metric_name = metric_name
        self.values = values

    def handle_name(self, field_name, required_lenght):
        if self.values[0][field_name]:
            if not isinstance(self.values[0][field_name], list):
                raise TabError(f"'{field_name} must be a list'")
            if len(self.values[0][field_name]) != required_lenght:
                raise ValueError(
                    f"length of {field_name} dose not match the "
                    "length of the result from the metric "
                    f"The length of {field_name} of the metric is {required_lenght} "
                    f"but the length of the '{field_name}' "
                    f"is {len(self.values[0][field_name])}'"
                )
            names = self.values[0][field_name]
            return names
        return list(range(required_lenght))

    @abstractmethod
    def execute(self):
        pass
