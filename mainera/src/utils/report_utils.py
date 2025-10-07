from mainera.src.utils.mainera_utils import create_folder
import xml.etree.ElementTree as ET
from graphviz import Digraph
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import re
import textwrap

plt.style.use("dark_background")


class MetricDisplay:
    def __init__(self, metric_name, values):
        self.metric_name = metric_name
        self.values = values

    def execute(self):
        pass


class DisplaySignleNumber(MetricDisplay):
    def __init__(self, metric_name, values, folderpath):
        super().__init__(metric_name, values)
        self.folderpath = folderpath

    def execute(self):
        ids, results = [value["metric id"] for value in self.values], [
            value["result"] for value in self.values
        ]
        data = {
            "Metric ID": ids,
            "Metric Value": results,
        }
        img_path = os.path.join(self.folderpath, f"{self.metric_name}.png")
        table = pd.DataFrame(data).transpose().round(3).to_html()
        plt.plot(ids, results, marker="o")
        plt.xlabel("Metric ID")
        plt.ylabel("Metric Value")
        plt.title(f"{self.metric_name}")
        plt.savefig(img_path)
        content = (
            f"### Metric name: {self.metric_name}\n\n"
            f"Table \n{table}\n\n"
            f"Graph \n\n![{self.metric_name}]({self.metric_name}.png)\n"
        )
        return content


class DisplayString(MetricDisplay):
    def __init__(self, metric_name, values):
        super().__init__(metric_name, values)

    def execute(self):
        content = f"### Metric name: {self.metric_name}\n"
        for value in self.values:
            new_content = (
                f"<div>\n"
                f"<h3>Metric id :{value['metric id']}</h3>\n\n"
                f"<pre>{value['result'].strip()}</pre>\n"
                "</div>\n"
            )
            content = content + new_content
        content = content
        return content


class DisplayDict(MetricDisplay):
    def __init__(self, metric_name, values):
        super().__init__(metric_name, values)

    def execute(self):
        content = (
            f"### Metric name: {self.metric_name}\n"
            "<div style='display: grid; grid-template-columns: repeat(auto-fill, minmax(400px, 1fr)); gap: 10px;'>\n"
        )
        for value in self.values:
            table = pd.DataFrame(value["result"]).transpose().round(3).to_html()
            new_content = (
                f"<div><h3>Metric id :{value["metric id"]}</h3>\n\n {table}\n</div>\n"
            )
            content = content + new_content
        content = content + "</div>\n"
        return content


class DisplayArraySingle(MetricDisplay):
    def __init__(self, metric_name, values):
        super().__init__(metric_name, values)

    def execute(self):
        content = f"### Metric name: {self.metric_name}\n\n"
        ids, results = [value["metric id"] for value in self.values], [
            np.array2string(
                np.array(value["result"]), separator=", ", max_line_width=100
            )
            for value in self.values
        ]
        data = {"Metric ID": ids, "Metric Value": results}
        table = pd.DataFrame(data).to_html(index=False)
        content = content + table + "\n"
        return content


class DisplayMultiArray(MetricDisplay):
    def __init__(self, metric_name, values):
        super().__init__(metric_name, values)

    def execute(self):
        content = f"### Metric name: {self.metric_name}\n"
        for value in self.values:
            array_str = np.array2string(
                np.array(value["result"]), separator=", ", max_line_width=80
            )
            new_content = (
                f"<div>\n"
                f"<h3>Metric id :{value['metric id']}</h3>\n\n"
                f"<pre>{array_str}</pre>\n"
                "</div>\n"
            )
            content = content + new_content
        content = content
        return content


class DisplayTupleNotCurve(MetricDisplay):
    def __init__(self, metric_name, values, folderpath):
        super().__init__(metric_name, values)
        self.folderpath = folderpath

    def execute(self):
        content = (
            f"### Metric name: {self.metric_name}\n"
            "<div style='display: grid; grid-template-columns: repeat(auto-fill, minmax(400px, 1fr)); gap: 10px;'>\n"
        )

        for value in self.values:
            data = {}
            for i in range(len(value["tuple_argums"]["labels"])):
                data[value["tuple_argums"]["labels"][i]] = value["result"][i]
            table = pd.DataFrame(data).to_html(index=False)
            new_content = (
                f"<div>\n"
                f"<h3>Metric id :{value['metric id']}</h3>\n\n"
                f"{table}\n"
                "</div>\n"
            )
            content = content + new_content
        content = content + "</div>\n"
        return content


class DisplayTupleCurve(MetricDisplay):
    def __init__(self, metric_name, values, folderpath):
        super().__init__(metric_name, values)
        self.folderpath = folderpath

    def execute(self):
        content = (
            f"### Metric name: {self.metric_name}\n\n"
            "<div style='display: grid; grid-template-columns: repeat(auto-fill, minmax(400px, 1fr)); gap: 10px;'>\n"
        )
        for value in self.values:
            plot_func = value["tuple_argums"]["plot_func"]
            result = value["result"]
            img_path = os.path.join(
                self.folderpath, f"{self.metric_name}_{value['metric id']}.png"
            )
            plt = plot_func(result)
            if plt is None:
                raise ValueError("The plot_func must return the plt object")
            plt.title(f"{self.metric_name} - Metric ID: {value['metric id']}")
            plt.savefig(img_path)
            plt.close()
            new_content = (
                f"<div>\n\n"
                f"![{self.metric_name}_{value['metric id']}]({self.metric_name}_{value['metric id']}.png)\n"
                "</div>\n"
            )
            content = content + new_content
        content = content+ "</div>\n"
        return content


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
                f"XML file: {self.xmlpath} not found please run serialze function"
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
            node_name = re.sub(r"_copy_\d+$", " ", node_name)
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
        metric_content = "## Metrcis Summary\n"
        for metric_name, values in metric.items():
            if isinstance(values[0]["result"], (int, float)):
                obj = DisplaySignleNumber(metric_name, values, self.folderpath)
                content = obj.execute()
            elif (
                isinstance(values[0]["result"], (np.ndarray))
                and values[0]["result"].ndim > 1
            ):
                obj = DisplayMultiArray(metric_name, values)
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
                if values[0]["tuple_argums"]["is_curve"] == False:
                    obj = DisplayTupleNotCurve(metric_name, values, self.folderpath)
                    content = obj.execute()
                else:
                    obj = DisplayTupleCurve(metric_name, values, self.folderpath)
                    content = obj.execute()
            metric_content = metric_content + "\n" + content
        return metric_content

    def create_readme_file(self):
        self.create_graph_img()
        metric_content = self.metric_display()
        content = textwrap.dedent(
            """\
        # Report
        This is the automated report for the pipeline created using **Mainera**.  
        It includes both:  
        - A graphical representation of the pipeline structure
        - A summary of the resulting metrics
        ## Graphical Representation
        ![Pipeline Graph](graph.png)
        ### Description
        The graph illustrates the **nodes** in the pipeline and their **connections**.  
        - Each node is labeled with its:
            - **ID**
            - **Type** (one of: `Input`, `Preprocess`, `Model`, `Predict`, `Merge`, `Metric`)
            - **Name**
        - **Node colors**:
            -  **Red nodes** → selected in the final execution path  
            -  **Blue nodes** → present in the structure but not part of the final path 
              
        """
        )
        content = content + "\n" + metric_content
        readme_path = os.path.join(self.folderpath, "README.md")
        with open(readme_path, encoding="utf-8", mode="w") as f:
            f.write(content)
