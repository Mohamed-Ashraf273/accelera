import os
import numpy as np
from abc import ABC
from abc import abstractmethod
from mainera.src.utils.mainera_utils import create_folder
from mainera.src.wrappers.display_array_multi_wrapper import (
    DisplayMultiArrayWrapper,
)
from mainera.src.wrappers.display_array_single_wrapper import (
    DisplayArraySingleWrapper,
)
from mainera.src.wrappers.display_dict_wrapper import DisplayDictWrapper
from mainera.src.wrappers.display_single_number_wrapper import (
    DisplaySignleNumberWrapper,
)
from mainera.src.wrappers.display_string_wrapper import DisplayStringWrapper
from mainera.src.wrappers.display_tuple_curve_wrapper import (
    DisplayTupleCurveWrapper,
)
from mainera.src.wrappers.display_tuple_not_curve_wrapper import (
    DisplayTupleNotCurveWrapper,
)


class ReportWrapper(ABC):
    def __init__(self, folderpath, results):
        self.folderpath = folderpath
        self.results = results
        create_folder(folderpath)
        self.metric_ids = []

    def handle_metric(self):
        final_metric = {}
        for i in range(len(self.metric_ids)):
            metric_name = self.results[i]["metric name"]
            metric_value = self.results[i]["result"]
            metric_tuple_argums = self.results[i]["tuple_argums"]
            metric_obj = {
                "metric id": self.metric_ids[i],
                "result": metric_value,
                "tuple_argums": metric_tuple_argums,
            }
            print(metric_obj)
            if metric_name in final_metric:
                final_metric[metric_name].append(metric_obj)
            else:
                final_metric[metric_name] = [metric_obj]
        return final_metric

    def metric_display(self):
        metric = self.handle_metric()
        metric_content = (
            "## Metrics Summary\n"
            "- Each metric is displayed with its user-defined name,"
            " unique identifier (ID) and the corresponding results.\n"
            "- Depending on the metric type, the results may include scalar values,"
            " arrays, dictionaries,strings, curves, or tuples.\n"
            "- All metrics are presented in a structured and consistent format to facilitate"
            " clear interpretation and comparison.\n"
        )
        for metric_name, values in metric.items():
            if isinstance(values[0]["result"], (int, float)):
                obj = DisplaySignleNumberWrapper(metric_name, values, self.folderpath)
                content = obj.execute()
            elif (
                isinstance(values[0]["result"], (np.ndarray))
                and values[0]["result"].ndim > 1
            ):
                obj = DisplayMultiArrayWrapper(metric_name, values)
                content = obj.execute()
            elif (
                isinstance(values[0]["result"], (np.ndarray))
                and values[0]["result"].ndim == 1
            ):
                obj = DisplayArraySingleWrapper(metric_name, values)
                content = obj.execute()
            elif isinstance(values[0]["result"], dict):
                obj = DisplayDictWrapper(metric_name, values)
                content = obj.execute()
            elif isinstance(values[0]["result"], str):
                obj = DisplayStringWrapper(metric_name, values)
                content = obj.execute()
            elif isinstance(values[0]["result"], (tuple)):
                if not values[0]["tuple_argums"]["is_curve"]:
                    obj = DisplayTupleNotCurveWrapper(
                        metric_name, values, self.folderpath
                    )
                    content = obj.execute()
                else:
                    obj = DisplayTupleCurveWrapper(metric_name, values, self.folderpath)
                    content = obj.execute()
            metric_content = metric_content + "\n" + content
        return metric_content

    def create_readme_file(self, content):
        readme_path = os.path.join(self.folderpath, "README.md")
        with open(readme_path, encoding="utf-8", mode="w") as f:
            f.write(content)

    @abstractmethod
    def execute(self):
        pass
