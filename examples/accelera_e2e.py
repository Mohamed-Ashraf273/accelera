from accelera.src.accelera_pipe.core.pipeline import Pipeline
from accelera.src.e2e.tabular.e2e import E2E

graoh = Pipeline()
e2e = E2E()

config = {
    "target_col": "price",
    "problem_type": "regression",
    "folder_path": "e2e_housing_output",
}

predictions, executed_graph = e2e(
    content="https://drive.google.com/file/d/1VMtLcWDcigwkimpf-eWVMZ7zJMUf7wxs/view?usp=drive_link",
    config=config,
)

print(predictions)
