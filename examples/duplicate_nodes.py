from accelera.src.accelera_pipe.core.pipeline import Pipeline as accpipe
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from accelera.src.accelera_pipe.wrappers.graph_report import GraphReport
from accelera.src.utils.accelera_utils import serialize
import numpy as np

pipeline = accpipe()
pipeline.branch(
    "b1",
    pipeline.preprocess("p1", StandardScaler(), branch=True),
    pipeline.preprocess("p2", MinMaxScaler(), branch=True),
    pipeline.preprocess("p3", StandardScaler(), branch=True),
).branch(
    "b2",
    pipeline.model("m1", LogisticRegression(max_iter=1000), branch=True),
    pipeline.model("m2", SVC(C=10), branch=True),
    pipeline.model("m3", LogisticRegression(max_iter=1000), branch=True),
)

res, final = pipeline(np.random.randn(10, 10), np.array([0, 1] * 5))
serialize(pipeline, "duplicate_nodes_pipeline.xml")
report = GraphReport("test duplicate nodes", "duplicate_nodes_pipeline.xml", res)
img_path = report.execute()