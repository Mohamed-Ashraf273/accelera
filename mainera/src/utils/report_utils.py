from mainera.src.utils.mainera_utils import create_folder
import xml.etree.ElementTree as ET
from graphviz import Digraph
import os
import re
import textwrap

class Report:
    def __init__(self,folderpath,xmlpath,results):
        self.folderpath = folderpath
        self.xmlpath=xmlpath
        self.metric_ids=[]
        self.results=results
        create_folder(folderpath)
    def create_graph_img(self):
        try:
            tree=ET.parse(self.xmlpath)
            root=tree.getroot()
        except FileNotFoundError:
             raise FileNotFoundError(f"XML file: {self.xmlpath} not found please run serialze function" )
        except ET.ParseError as e:
            raise ValueError(f"Invalid XML file {self.xmlpath}: {e}")    
        
        graph=Digraph()
        for layer in root.findall("layers/layer"):
            node_id=layer.get("id")
            node_name=layer.get("name")
            node_type=layer.get("type")
            selected_in_path=layer.get("selected_in_path")
            color="red" if selected_in_path=="true" else "blue"
            if node_type=="METRIC":
                self.metric_ids.append(node_id)
            node_name = re.sub(r'_copy_\d+$', ' ', node_name)
            node_full_name=f"{node_id}\\n{node_type}\\n{node_name}"
            graph.node(node_id,label=node_full_name,color=color)
            
        for edge in root.findall("edges/edge"):
            from_id,to_id=edge.get("from-layer"),edge.get("to-layer")
            graph.edge(from_id,to_id)    
            
        img_path=os.path.join(self.folderpath,"graph")
        graph.render(img_path,format="png",cleanup=True)
        
    def handle_metric(self):
        final_metric={}
        for i in range(len(self.metric_ids)):
           metric_name= self.results[i]['metric name']
           metric_value=self.results[i]['result']
           metric_obj={"id":self.metric_ids[i],"result":metric_value}
           if metric_name in final_metric:
            final_metric[metric_name].append(metric_obj)
           else:
            final_metric[metric_name]=[metric_obj] 
        return final_metric    
    def create_readme_file(self):
        self.create_graph_img()
        self.handle_metric()
        content=textwrap.dedent("""\
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
            - 🔴 **Red nodes** → selected in the final execution path  
            - 🔵 **Blue nodes** → present in the structure but not part of the final path  
                """)
        readme_path=os.path.join(self.folderpath,"README.md")
        with open(readme_path,encoding="utf-8",mode="w") as f:
            f.write(content)