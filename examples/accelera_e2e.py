from sklearn.metrics import r2_score

from accelera.src.accelera_pipe.core.pipeline import Pipeline
from accelera.src.e2e.tabular.e2e import E2E

graph = Pipeline()


e2e = E2E()

config = {
    "target_col": "price",
    "problem_type": "regression",
    "folder_path": "e2e_housing_output",
}

predictions, executed_graph = e2e(
    content="https://drive.google.com/uc?id=1VMtLcWDcigwkimpf-eWVMZ7zJMUf7wxs",
    config=config,
)
y_test = predictions[1]

print(r2_score(y_test, predictions[0]))
