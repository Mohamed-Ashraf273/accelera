import os


class ReportBase:
    def __init__(self, folderpath):
        self.folderpath = folderpath
        self.start_content = """
        <!DOCTYPE html>
        <html>
        <head>
          <meta charset="UTF-8">
          <meta name="viewport" content="width=device-width, initial-scale=1.0">
          <title>Accelera Report</title>
          <style>
          body {
            font-family: Arial, sans-serif;
            line-height: 1.6;
            background-color: #000000;
            color: #ffffff;
            margin-left:30px
            }
            h1 {
            color: #ffcc00;
            text-align: center;
            }
            h2{
            font-style:italic;
            text-align: center;
            }
            h3{
            color: #ffee00;
            }
            h4 {
            color: #ffcc00; 
            }
            
            table{
            border-collapse: collapse;
            width: 80%;
            }
             th, td {
            border: 1px solid #444;
            padding: 6px;
            text-align: center;
            }
            .metric-container {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 20px;
            }
            div{
                margin:40px 10px ;
            }
        </style>
        </head>
        <body>
        <h1>Accelera Report</h1>
        """
        self.end_content = """
        </body>
        </html>
        """

    def create_html_file(self, content):
        readme_path = os.path.join(self.folderpath, "report.html")
        with open(readme_path, encoding="utf-8", mode="w") as f:
            f.write(content)

    def execute(self):
        raise NotImplementedError("Subclasses must implement this method")
