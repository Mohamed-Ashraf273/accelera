from accelera.src.core.report_base import ReportBase
import io


class PreprocessingReport(ReportBase):
    def __init__(self, folderpath, df):
        super().__init__(folderpath)
        self.content = ""
        self.df = df
        self.numeric_df=self.df.select_dtypes(include="number")
        self.categorical_df=self.df.select_dtypes(include="object")
    def show_data_overview(self):
        self.content += "<h2>Data Overview</h2>\n"
        self.content += f"""<h3>First 5 rows of the dataset:</h3>\n
        {self.df.head().to_html(index=False, border=1, justify='center')}"""
        self.content += "<h3>Data Information:</h3>\n"
        io_buffer = io.StringIO()
        self.df.info(buf=io_buffer)
        self.content += f"<pre>{io_buffer.getvalue()}</pre>\n"
        if not self.numeric_df.empty:
            self.content += "<h3>Numerical Statistics:</h3>\n"
            self.content += f"{self.numeric_df.describe().to_html()}"
        if not self.categorical_df.empty:
            self.content += "<h3>Categorical Statistics:</h3>\n"
            self.content += f"{self.categorical_df.describe().to_html( border=1, justify='center')}"
        self.content += "<h3>Missing Values:</h3>\n"
        missing_values = self.df.isnull().sum()
        self.content += f"{missing_values[missing_values > 0].to_frame(name='Missing Values').to_html( border=1, justify='center')}"
        self.content+= "<h3>Duplicates:</h3>\n"
        self.content += f"<p> number of duplicates rows: {self.df.duplicated().sum()}</p>\n"
    
    def execute(self):
        self.show_data_overview()
        full_content = self.start_content + self.content + self.end_content
        self.create_html_file(full_content)
