import os

import pandas as pd

from accelera.src.automl.core.agent import AutoAccelera

agent = AutoAccelera()
current_dir = os.path.dirname(__file__)
df = pd.read_csv(os.path.join(current_dir, "Titanic-Dataset.csv"))
best_pipeline = agent.get_pipeline(df, "Survived")
