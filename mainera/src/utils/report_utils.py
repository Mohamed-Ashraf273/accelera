import os
import re
import textwrap
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
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

    def get_branches(self, nodes, tree, current_node_id="0", path=None):
        if path is None:
            path = []
        if nodes[current_node_id]["node_type"] == "METRIC":
            return [{"path": path, "metrics": nodes[current_node_id]}]

        path.append(nodes[current_node_id])
        if current_node_id not in tree:
            return [{"path": path, "metrics": {}}]

        all_branches = []
        for child_id in tree[current_node_id]:
            child_branches = self.get_branches(nodes, tree, child_id, list(path))
            all_branches.extend(child_branches)
        return all_branches

    def grouby_branches(self, branches):
        grouped_branches = {}
        for branch in branches:
            key = tuple(node["node_id"] for node in branch["path"])
            if key not in grouped_branches:
                grouped_branches[key] = branch["path"]
            if branch["metrics"]:
                grouped_branches[key].append(branch["metrics"])
        return list(grouped_branches.values())

    def display_branch(self, branch, title):
        ids = [node["node_id"] for node in branch]
        names = [node["node_name"] for node in branch]
        types = [node["node_type"] for node in branch]
        data = {"Node ID": ids, "Node Name": names, "Node Type": types}
        table = pd.DataFrame(data).to_html(index=False)
        content = (
            '<div style="overflow-x:auto;max-width:400px;">\n'
            f'<h3 style="color:yellow;"> {title}</h3>\n'
            f"{table}\n"
            "</div>\n"
        )
        return content

    def display_branches(self, branches, best_branch):
        branches = self.grouby_branches(branches)
        content = (
            "## Pipeline Branches\n"
            "<div style='display: grid; \n"
            "grid-template-columns: repeat(2,  1fr); gap: 10px;'>\n"
        )
        branch_id = 1
        for branch in branches:
            title = f"Branch {branch_id}"
            content+=self.display_branch(branch, title)
            branch_id += 1
        content=content + "</div>\n"
        if best_branch:
            content += self.display_branch(best_branch, "Best Branch")
        return content

    def create_graph_img(self):
        try:
            xml_tree = ET.parse(self.xmlpath)
            root = xml_tree.getroot()
        except FileNotFoundError:
            raise FileNotFoundError(
                f"XML file: {self.xmlpath} " "not found please run serialize function"
            )
        except ET.ParseError as e:
            raise ValueError(f"Invalid XML file {self.xmlpath}: {e}")

        graph = Digraph()
        graph.attr("graph")
        nodes = {}
        tree = {}
        best_branch = []

        for layer in root.findall("layers/layer"):
            node_id = layer.get("id")
            node_name = layer.get("name")
            node_name = re.sub(r"_\d+", "", node_name)
            node_type = layer.get("type")
            selected_in_path = layer.get("selected_in_path")
            color = "red" if selected_in_path == "true" else "blue"
            if node_type == "METRIC":
                self.metric_ids.append(node_id)
            node_full_name = f"{node_id}\\n{node_type}\\n{node_name}"
            graph.node(node_id, label=node_full_name, color=color)
            node_object = {
                "node_id": node_id,
                "node_type": node_type,
                "node_name": node_name,
            }
            nodes[node_id] = node_object
            if selected_in_path == "true":
                best_branch.append(node_object)

        for edge in root.findall("edges/edge"):
            from_id, to_id = edge.get("from-layer"), edge.get("to-layer")
            graph.edge(from_id, to_id)
            if from_id in tree:
                tree[from_id].append(to_id)
            else:
                tree[from_id] = [to_id]
        img_path = os.path.join(self.folderpath, "graph")
        graph.render(img_path, format="png", cleanup=True)
        return nodes, tree, best_branch

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
        metric_content = (
            "## Metrics Summary\n"
            "- Each metric is displayed with its user-defined name, unique identifier (ID) and the corresponding results.\n"
            "- Depending on the metric type, the results may include scalar values, arrays, dictionaries,strings, curves, or tuples.\n"
            "- All metrics are presented in a structured and consistent format to facilitate clear interpretation and comparison.\n"
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

    def create_readme_file(self):
        nodes, tree, best_branch = self.create_graph_img()
        branches = self.get_branches(nodes, tree)
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
        branch_content = self.display_branches(branches, best_branch)
        content = content + "\n" + branch_content + "\n" + metric_content
        readme_path = os.path.join(self.folderpath, "README.md")
        with open(readme_path, encoding="utf-8", mode="w") as f:
            f.write(content)
