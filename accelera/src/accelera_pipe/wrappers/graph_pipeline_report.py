import numpy as np

from accelera.src.accelera_pipe.core.report_base import ReportBase
from accelera.src.accelera_pipe.wrappers.display_array_multi import (
    DisplayMultiArray,
)
from accelera.src.accelera_pipe.wrappers.display_array_single import (
    DisplayArraySingle,
)
from accelera.src.accelera_pipe.wrappers.display_dict import DisplayDict
from accelera.src.accelera_pipe.wrappers.display_figure import DisplayFigure
from accelera.src.accelera_pipe.wrappers.display_single_number import (
    DisplaySingleNumber,
)
from accelera.src.accelera_pipe.wrappers.display_string import DisplayString
from accelera.src.accelera_pipe.wrappers.display_tuple_not_curve import (
    DisplayTupleNotCurve,
)
from accelera.src.utils.accelera_utils import create_folder


class GraphPipelineReport(ReportBase):
    def __init__(self, folderpath, results):
        super().__init__(folderpath)
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
            "<h2> Metrics Summary</h2>\n"
            "<ul>\n"
            "<li>Each metric is displayed with its user-defined name,"
            " unique identifier (ID) and the corresponding results.</li>\n"
            "<li>Depending on the metric type, the results may include "
            "scalar values, arrays, dictionaries, "
            "strings, curves, or tuples.</li>\n"
            "<li>All metrics are presented in a structured "
            "and consistent format "
            "to facilitate clear interpretation and comparison.</li>\n"
            "</ul>\n"
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
                    obj = DisplayFigure(metric_name, values, self.folderpath)
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
                    obj = DisplayTupleNotCurve(
                        metric_name, values, self.folderpath
                    )
                    content = obj.execute()
                else:
                    obj = DisplayFigure(metric_name, values, self.folderpath)
                    content = obj.execute()
            metric_content = metric_content + "\n" + content
        return metric_content

    def execute(self):
        pass
