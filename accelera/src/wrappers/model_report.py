import os
import textwrap

import matplotlib.pyplot as plt

from accelera.src.wrappers.graph_pipeline_report import GraphPipelineReport


class ModelReport(GraphPipelineReport):
    def __init__(self, folderpath, results, history=None):
        super().__init__(folderpath, results)
        if history and not isinstance(history, dict):
            raise TypeError(
                f"Expected 'history' to be a dict has "
                "the evaluation metrics of "
                f"the model to plot them, got {type(history).__name__}"
            )
        self.history = history
        self.metric_ids = list(map(str, range(len(results))))

    def display_history(self):
        try:
            content = "<h2> Model History</h2>\n"
            training_evaluation = []
            for key in self.history.keys():
                if not key.startswith("val_"):
                    training_evaluation.append(key)
            plt.figure(figsize=(8, 8))
            for i in range(len(training_evaluation)):
                plt.subplot(len(training_evaluation), 1, i + 1)
                plt.plot(
                    self.history[training_evaluation[i]],
                    label=training_evaluation[i],
                )
                plt.suptitle(f"{training_evaluation[i]}")
                val_name = f"val_{training_evaluation[i]}"
                validation = self.history[val_name]
                if validation:
                    plt.plot(validation, label=val_name)
                plt.xlabel("epochs")
                plt.legend()
            plt.tight_layout()
            path = os.path.join(self.folderpath, "history")
            plt.savefig(path)
            content += "<img src='history.png' alt='Model History' />\n"
            return content
        except Exception as e:
            print(f"Error while generating model history: {e}")
            return "<h2> Model History</h2>\nError: Could not generate plot.\n"

    def execute(self):
        metric_content = self.metric_display()
        content = textwrap.dedent(
            """\
        <p>This is the automated report for the 
        model  created using <i>Accelera</i>.</p>
        <p>
        It provides a comprehensive overview of the model’s 
        performance, including:</p>
        <ul>
        <li>
        Performance Summary — Highlighting key evaluation 
        metrics and model results</li>
        <li>
        - Training Graphs — Visualizing deep learning history 
        (loss, accuracy, and other tracked metrics) if exists
        </li>
        </ul>
        """
        )
        if self.history:
            content += self.display_history()
        content = (
            self.start_content
            + "\n"
            + content
            + "\n"
            + metric_content
            + self.end_content
        )
        self.create_html_file(content)
