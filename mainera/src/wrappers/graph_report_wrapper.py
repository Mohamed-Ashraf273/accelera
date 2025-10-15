import os
import re
import textwrap
import xml.etree.ElementTree as ET

import pandas as pd
from graphviz import Digraph

from mainera.src.wrappers.report_wrapper import ReportWrapper


class GraphReport(ReportWrapper):
    def __init__(self, folderpath, xmlpath, results):
        super().__init__(folderpath, results)
        self.xmlpath = xmlpath
        self.metric_ids = []

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
            child_branches = self.get_branches(
                nodes, tree, child_id, list(path)
            )
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
            content += self.display_branch(branch, title)
            branch_id += 1
        content = content + "</div>\n"
        if best_branch:
            content += self.display_branch(best_branch, "Best Branch")
        return content

    def create_graph_img(self):
        try:
            xml_tree = ET.parse(self.xmlpath)
            root = xml_tree.getroot()
        except FileNotFoundError:
            raise FileNotFoundError(
                f"XML file: {self.xmlpath} "
                "not found please run serialize function"
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

    def execute(self):
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
        self.create_readme_file(content)
