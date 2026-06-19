import os

import matplotlib.pyplot as plt


class GraphBase:
    def __init__(self, folder_path):
        self.folder_path = os.path.join(folder_path, "graphs")
        if not os.path.exists(self.folder_path):
            os.makedirs(self.folder_path)
        plt.style.use("dark_background")

    def build_graph(self):
        raise NotImplementedError("Subclasses should implement this method")
