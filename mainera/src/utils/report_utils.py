from mainera.src.utils.mainera_utils import create_folder
import xml.etree.ElementTree as ET
from graphviz import Digraph
import os
import re

class Report:
    def __init__(self,folderpath,xmlpath):
        self.folderpath = folderpath
        self.xmlpath=xmlpath
        create_folder(folderpath)
    def create_graph_img(self):
        try:
            tree=ET.parse(self.xmlpath)
            root=tree.getroot()
        except ET.ParseError as e:
            raise ValueError(f"Invalid XML file {self.xmlpath}: {e}")    
        
        graph=Digraph()
        for layer in root.findall("layers/layer"):
            node_id,node_name,node_type=layer.get("id"),layer.get("name"),layer.get("type")
            node_name = re.sub(r'_copy_\d+$', ' ', node_name)
            node_full_name=f"{node_id}:{node_type}\\n{node_name}"
            graph.node(node_id,label=node_full_name)
        for edge in root.findall("edges/edge"):
            from_id,to_id=edge.get("from-layer"),edge.get("to-layer")
            graph.edge(from_id,to_id)    
            
        img_path=os.path.join(self.folderpath,"graph")
        graph.render(img_path,format="png",cleanup=True)
        return f"{img_path}.png"

