import os
import pandas as pd

from accelera.src.core.report_base import ReportBase


class ImagePreprocessingReport(ReportBase):
    def __init__(self, folderpath, report_data):
        super().__init__(folderpath)
        self.content = ""
        self.report_data = report_data
        self.data_overview_training = self.report_data["data_overview"].get(
            "training_folder"
        )
        self.data_overview_validation = self.report_data["data_overview"].get(
            "validation_folder"
        )
        self.split_data = self.report_data.get("split_data")
        self.graphs = self.report_data.get("graphs")

    def show_unoreder_list(self, classes, title=None):
        self.content += "<div>\n"
        if title is not None:
            self.content += f"{title}\n"
        self.content += "<ul>\n"
        for class_name in classes:
            self.content += f"<li>{class_name}</li>\n"
        self.content += "</ul>\n"
        self.content += "</div>\n"

    def show_paragraph_section(self, text, title=None):
        self.content += "<div>\n"
        if title is not None:
            self.content += f"{title}\n"
        self.content += f"<p>{text}</p>\n"
        self.content += "</div>\n"

    def show_table_section_from_df(self, dictionary, title=None):
        self.content += "<div>\n"
        if title is not None:
            self.content += f"{title}\n"
        self.content += dictionary.to_html(index=False)
        self.content += "</div>\n"

    def show_folder_overview(self, over_view, folder_type):
        self.content += "<div>\n"
        self.content += f"<h3>{folder_type} Folder Overview</h2>\n"
        self.show_unoreder_list(over_view["classes"], f"<h4>Classes</h4>")
        if over_view.get("mapping") is not None:
            df = pd.DataFrame(over_view["mapping"].items(), columns=["Class", "Label"])
            self.show_table_section_from_df(df, f"<h4>Classes 2 Labels Mapping </h4>")
        self.show_paragraph_section(
            f"Count : {over_view["images_len"]}", f"<h4>Total Images </h4>"
        )
        self.content += "<div>\n"
        self.content += "<h4>Invalid Images</h4>\n"
        self.show_paragraph_section(f"Count : {over_view["invalid_len"]}")
        self.show_unoreder_list(over_view["invalid_images"])
        self.content += "</div>\n"
        self.show_table_section_from_df(
            over_view["random_sample"], "<h4>Random Sample</h4>\n"
        )
        self.content += "</div>\n"

    def show_overview(self):
        self.content += "<div>\n"
        self.content += "<h2>Data Overview</h2>\n"
        self.show_folder_overview(self.data_overview_training, "Training")
        if self.data_overview_validation is not None:
            self.show_folder_overview(self.data_overview_training, "Validation")
        self.content += "</div>\n"

    def show_data_split(self):
        if self.split_data is None:
            return
        self.content += "<div>\n"
        self.content += "<h2>Splitting</h2>\n"
        self.show_paragraph_section(
            f"{self.split_data["validation_size"]}", "<h3>Validation Size</h3>\n"
        )
        self.content += "<div>\n"
        self.content += "<h3>Training After Splitting</h3>\n"
        self.show_paragraph_section(
            f"{self.split_data["training_data_size"]}", f"<h4>Total Images </h4>"
        )
        self.show_table_section_from_df(
            self.split_data["random_training_sample"], "<h4>Random Sample</h4>\n"
        )

        self.content += "</div>\n"
        self.content += "<div>\n"
        self.content += "<h3>Validation After Splitting</h3>\n"
        self.show_paragraph_section(
            f"{self.split_data["validation_data_size"]}", f"<h4>Total Images </h4>"
        )
        self.show_table_section_from_df(
            self.split_data["random_validation_sample"], "<h4>Random Sample</h4>\n"
        )
        self.content += "</div>\n"
        self.content += "</div>\n"

    def execute(self):
        self.show_overview()
        self.show_data_split()
        self.show_graphs(self.graphs)
        full_content = self.start_content + self.content + self.end_content
        self.create_html_file(full_content)
