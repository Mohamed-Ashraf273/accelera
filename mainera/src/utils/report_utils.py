import os
import textwrap
import xml.etree.ElementTree as ET

import numpy as np
from graphviz import Digraph

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


class Report:
    def __init__(self, folderpath, xmlpath, results):
        self.folderpath = folderpath
        self.xmlpath = xmlpath
        self.metric_ids = []
        self.results = results
        create_folder(folderpath)

    def create_graph_img(self):
        try:
            tree = ET.parse(self.xmlpath)
            root = tree.getroot()
        except FileNotFoundError:
            raise FileNotFoundError(
                f"XML file: {self.xmlpath} "
                "not found please run serialize function"
            )
        except ET.ParseError as e:
            raise ValueError(f"Invalid XML file {self.xmlpath}: {e}")

        graph = Digraph()
        graph.attr("graph")

        for layer in root.findall("layers/layer"):
            node_id = layer.get("id")
            node_name = layer.get("name")
            node_type = layer.get("type")
            selected_in_path = layer.get("selected_in_path")
            color = "red" if selected_in_path == "true" else "blue"
            if node_type == "METRIC":
                self.metric_ids.append(node_id)
            node_full_name = f"{node_id}\\n{node_type}\\n{node_name}"
            graph.node(node_id, label=node_full_name, color=color)

        for edge in root.findall("edges/edge"):
            from_id, to_id = edge.get("from-layer"), edge.get("to-layer")
            graph.edge(from_id, to_id)

        img_path = os.path.join(self.folderpath, "graph")
        graph.render(img_path, format="png", cleanup=True)

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
            if metric_name in final_metric:
                final_metric[metric_name].append(metric_obj)
            else:
                final_metric[metric_name] = [metric_obj]
        return final_metric

    def metric_display(self):
        metric = self.handle_metric()
        metric_content = """## Metrics Summary\n
            - Each metric is displayed with its
              user-defined name, unique identifier (ID), 
            and the corresponding results.\n
            - Depending on the metric type, the results may 
            include scalar values, arrays, dictionaries, 
            strings, curves, or tuples.\n
            - All metrics are presented in a structured and 
            consistent format to facilitate clear 
            interpretation and comparison.\n"""
        for metric_name, values in metric.items():
            if isinstance(values[0]["result"], (int, float)):
                obj = DisplaySignleNumberWrapper(
                    metric_name, values, self.folderpath
                )
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
                    obj = DisplayTupleCurveWrapper(
                        metric_name, values, self.folderpath
                    )
                    content = obj.execute()
            metric_content = metric_content + "\n" + content
        return metric_content

    def create_readme_file(self):
        self.create_graph_img()
        metric_content = self.metric_display()
        content = textwrap.dedent(
            """\
        # Report
        This is the automated report for the 
        pipeline created using **Mainera**.  
        It includes both:  
        - A graphical representation of the pipeline structure
        - A summary of the resulting metrics
        ## Graphical Representation
        ![Pipeline Graph](graph.png)
        ### Description
        The graph illustrates the **nodes** in the 
        pipeline and their **connections**.  
        - Each node is labeled with its:
            - **ID**
            - **Type** (one of: `Input`, `Preprocess`, 
            `Model`, `Predict`, `Merge`, `Metric`)
            - **Name**
        - **Node colors**:
            -  **Red nodes** → selected in the final execution path  
            -  **Blue nodes** → present in the 
            structure but not part of the final path 
              
        """
        )
        content = content + "\n" + metric_content
        readme_path = os.path.join(self.folderpath, "README.md")
        with open(readme_path, encoding="utf-8", mode="w") as f:
            f.write(content)
