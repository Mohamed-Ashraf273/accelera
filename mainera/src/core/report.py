import os
from abc import ABC
from abc import abstractmethod

import numpy as np

from mainera.src.utils.mainera_utils import create_folder
from mainera.src.wrappers.display_array_multi import DisplayMultiArray
from mainera.src.wrappers.display_array_single import DisplayArraySingle
from mainera.src.wrappers.display_dict import DisplayDict
from mainera.src.wrappers.display_single_number import DisplaySingleNumber
from mainera.src.wrappers.display_string import DisplayString
from mainera.src.wrappers.display_figure import DisplayFigure
from mainera.src.wrappers.display_tuple_not_curve import DisplayTupleNotCurve


class Report(ABC):
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
            metric_plot_func = self.results[i]["plot_func"]
            metric_lables_name = self.results[i]["labels_name"]
            metric_headers_name = self.results[i]["headers_name"]
            metric_obj = {
                "metric id": self.metric_ids[i],
                "result": metric_value,
                "plot_func": metric_plot_func,
                "labels_name": metric_lables_name,
                "headers_name": metric_headers_name,
            }
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
            "- Depending on the metric type, the results may include "
            "scalar values, arrays, dictionaries, strings, curves, or tuples.\n"
            "- All metrics are presented in a structured and consistent format "
            "to facilitate clear interpretation and comparison.\n"
        )
        for metric_name, values in metric.items():
            if isinstance(values[0]["result"], (int, float)):
                obj = DisplaySingleNumber(metric_name, values, self.folderpath)
                content = obj.execute()
            elif (
                isinstance(values[0]["result"], (np.ndarray))
                and values[0]["result"].ndim > 1
            ):
                if values[0]["plot_func"] is None:
                    obj = DisplayMultiArray(metric_name, values)
                    content = obj.execute()
                else:
                    obj = DisplayFigure(metric_name, values,self.folderpath)
                    content = obj.execute()
            elif (
                isinstance(values[0]["result"], (np.ndarray))
                and values[0]["result"].ndim == 1
            ):
                obj = DisplayArraySingle(metric_name, values)
                content = obj.execute()
            elif isinstance(values[0]["result"], dict):
                obj = DisplayDict(metric_name, values)
                content = obj.execute()
            elif isinstance(values[0]["result"], str):
                obj = DisplayString(metric_name, values)
                content = obj.execute()
            elif isinstance(values[0]["result"], (tuple)):
                if values[0]["plot_func"] is None:
                    obj = DisplayTupleNotCurve(metric_name, values, self.folderpath)
                    content = obj.execute()
                else:
                    obj = DisplayFigure(metric_name, values, self.folderpath)
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
