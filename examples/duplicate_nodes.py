import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from accelera.src.accelera_pipe.core.pipeline import Pipeline as accpipe
from accelera.src.accelera_pipe.wrappers.graph_report import GraphReport
from accelera.src.utils.accelera_utils import serialize

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
