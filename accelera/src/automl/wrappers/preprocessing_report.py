import os
import pandas as pd

from accelera.src.core.report_base import ReportBase


class PreprocessingReport(ReportBase):
    def __init__(self, folderpath, report_data):
        super().__init__(folderpath)
        self.content = ""
        self.report_data = report_data
        self.data_overview = self.report_data["data_overview"]
        self.drop_duplicates = self.report_data["drop_duplicates"]
        self.split = self.report_data["split"]
        self.drop_columns = self.report_data["drop_columns"]
        self.graphs = self.report_data["graphs"]
        self.preprocessing = self.report_data["preprocessing"]
        self.after_preprocessing = self.report_data["after_preprocessing"]

    def show_data_heads(self, obj, field_name, name="dataset"):
        self.content += f"""<h3>First 5 rows of the {name}:</h3>\n
        {obj[field_name].to_html(index=False)}"""

    def show_info(self, obj):
        self.content += "<h3>Data Information:</h3>\n"
        self.content += f"<pre>{obj['info']}</pre>\n"

    def show_numeric_statis(self, obj):
        if obj["numerical_describe"] is not None:
            self.content += "<h3>Numerical Statistics:</h3>\n"
            self.content += f"{obj['numerical_describe'].to_html()}\n"

    def show_categoric_statis(self, obj):
        if obj["categorical_describe"] is not None:
            self.content += "<h3>Categorical Statistics:</h3>\n"
            self.content += f"{obj['categorical_describe'].to_html()}\n"

    def show_nulls(self, obj):
        self.content += "<h3>Missing Values:</h3>\n"
        missing_df = obj["missing_values"].to_frame(name="Missing Values")
        self.content += f"{missing_df.to_html()}\n"

    def show_dupplicats(self, obj):
        self.content += "<h3>Duplicates:</h3>\n"
        self.content += f"<p> number of duplicates rows: {obj['duplicates_sum']}</p>\n"
        self.content += "<p> Percentage of duplicates "
        self.content += f"rows: {obj['duplicates_percentage']} %</p>\n"

    def show_shape(self, obj):
        self.content += "<h3>Data Shape:</h3>\n"
        self.content += f"<p> {obj['shape']}</p> \n"

    def show_data_overview(self):
        self.content += "<div>\n"
        self.content += "<h2>Data Overview</h2>\n"
        self.show_data_heads(self.data_overview, "data_head")
        self.content += "</div>\n"
        self.content += "<div>\n"
        self.content += "<h2>Data Overview After lowering dataset</h2>\n"
        self.show_data_heads(self.data_overview, "lower_data_head")
        self.show_info(self.data_overview)
        self.show_shape(self.data_overview)
        self.show_numeric_statis(self.data_overview)
        self.show_categoric_statis(self.data_overview)
        self.show_nulls(self.data_overview)
        self.show_dupplicats(self.data_overview)
        self.content += "</div>\n"

    def show_drop_duplicates(self):
        self.content += "<div>\n"
        self.content += "<h2> After Drop Duplicates</h2>\n"
        self.show_shape(self.drop_duplicates)
        self.show_dupplicats(self.drop_duplicates)
        self.content += "</div>\n"

    def show_split(self):
        self.content += "<div>\n"
        self.content += "<h2> Train / Validation Split</h2>\n"
        self.content += "<h3> Test Size</h3>\n"
        self.content += f"<p>{self.split['test_size']}</p>\n"
        self.content += "<h3>Training set</h3>\n"
        self.content += f"X_train shape : {self.split['X_train_shape']}</p>\n"
        self.content += f"y_train shape : {self.split['y_train_shape']}</p>\n"
        self.content += "<h3>validation set</h3>\n"
        self.content += f"X_val shape : {self.split['X_val_shape']}</p>\n"
        self.content += f"y_val shape : {self.split['y_val_shape']}</p>\n"
        self.content += "</div>\n"

    def show_drop_col(self):
        self.content += "<div>\n"
        self.content += "<h2>Drop Columns</h2>\n"

        if self.drop_columns["col_drop"]:
            self.content += "<h3>Columns Removed & Justifications</h3>\n"
            self.content += "<table >\n"
            self.content += "<tr><th>Column</th><th>Reason</th></tr>\n"
            for col, reason in self.drop_columns["col_drop"].items():
                self.content += f"<tr><td>{col}</td><td>{reason}</td></tr>\n"
            self.content += "</table>\n"
            self.show_data_heads(self.drop_columns, "X_trian_head", "X train")
            self.show_data_heads(self.drop_columns, "X_val_head", "X validation")

        else:
            self.content += "<p>No columns were dropped</p>\n"
        self.content += "</div>\n"

    def show_graphs(self):
        self.content += "<div>\n"
        self.content += "<h2>Graphs</h2>\n"
        folder_path = os.path.join(self.graphs["folder_path"], "graphs")
        if not os.path.exists(folder_path):
            self.content += (
                f"<h3>There is no any image in the folder {folder_path}</h3>\n"
            )
        else:
            for image_name in self.graphs["images_name"]:
                image_file = os.path.join(folder_path, f"{image_name}.png")
                if os.path.exists(image_file):
                    image_file = os.path.join(".", "graphs", f"{image_name}.png")
                    self.content += f"<img src='{image_file}' "
                    self.content += "style='max-width:100%; margin:10px 0;'/>\n"
        self.content += "</div>\n"

    def show_preprocessing(self):
        self.content += "<div>\n"
        self.content += "<h2>Preprocessing</h2>\n"
        self.content += "<table>\n"

        self.content += "<tr><th>Column</th><th>Predicted Type</th><th>Preprocessing Steps</th></tr>\n"

        for item in self.preprocessing:
            col_name, col_type, col_preprocessing = (
                item["col_name"],
                item["col_type"].title(),
                item["col_preprocessing"],
            )
            self.content += f"<tr><td>{col_name}</td>"
            self.content += f"<td>{col_type}</td>"
            self.content += "<td>"
            for i, step in enumerate(col_preprocessing):
                self.content += f"{step.title()} "
                if i != len(col_preprocessing) - 1:
                    self.content += "-> "
            self.content += "</td>"
            self.content += "</tr>\n"

        self.content += "</table>\n"
        self.content += "</div>\n"

    def show_after_preprocessing(self):
        self.content += "<div>\n"
        self.content += "<h2>After Preprocessing</h2>\n"
        training_df = pd.concat(
            [
                self.after_preprocessing["X_train_processed"],
                self.after_preprocessing["y_train_processed"],
            ],
            axis=1,
        )
        val_df = pd.concat(
            [
                self.after_preprocessing["X_val_processed"],
                self.after_preprocessing["y_val_processed"],
            ],
            axis=1,
        )
        self.after_preprocessing["training_df"] = training_df
        self.after_preprocessing["val_df"] = val_df
        self.show_data_heads(
            self.after_preprocessing, "training_df", "Training After Preprocessing Head"
        )
        self.show_data_heads(
            self.after_preprocessing, "val_df", "Validation After Preprocessing Head"
        )
        self.content += "</div>\n"

    def execute(self):
        self.show_data_overview()
        self.show_drop_duplicates()
        self.show_split()
        self.show_drop_col()
        self.show_graphs()
        self.show_preprocessing()
        self.show_after_preprocessing()
        full_content = self.start_content + self.content + self.end_content
        self.create_html_file(full_content)
