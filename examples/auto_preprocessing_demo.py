import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVC

from accelera.src.automl.core.testing_preprocessing import TestingPreprocessing
from accelera.src.automl.core.training_preprocessing import (
    TrainingPreprocessing,
)

tatanic_df = pd.read_csv("Titanic-Dataset.csv")
tatanic_df_copy = tatanic_df[:5].copy()
print(tatanic_df.head())
training_preprocessor = TrainingPreprocessing(
    tatanic_df, "Survived", "classification", "./titanic_preprocessing"
)
X_train, y_train, X_val, y_val = training_preprocessor.common_preprocessing()
print(X_train[:5])
print("Titanic Dataset")
print("Random Forest Classifier")
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
print(model.score(X_val, y_val))
print("testing")
testing_preprocessor = TestingPreprocessing(
    tatanic_df_copy, "./titanic_preprocessing"
)
X_test, y_test = testing_preprocessor.common_preprocessing()
print(X_test)
print("Predictions:")
print(model.predict(X_test), y_test)
print("Logistic Regression")
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)
print(model.score(X_val, y_val))
print("SVC")
model = SVC()
model.fit(X_train, y_train)
print(model.score(X_val, y_val))

##############################################
print("House Price")

price_df = pd.read_csv("Housing.csv")
price_df_copy = price_df[:5].copy()
training_preprocessor = TrainingPreprocessing(
    price_df, "price", "regression", "./house_price_preprocessing"
)
X_train, y_train, X_val, y_val = training_preprocessor.common_preprocessing()

print(X_train[:5])
print("Random Forest Regressor")
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)
print(model.score(X_val, y_val))
print(mean_squared_error(y_val, model.predict(X_val)))
testing_preprocessor = TestingPreprocessing(
    price_df_copy, "./house_price_preprocessing"
)
X_test, y_test = testing_preprocessor.common_preprocessing()
print("Predictions:")
print(model.predict(X_test), y_test)
model = LinearRegression()
print("Linear Regression")
model.fit(X_train, y_train)
print(model.score(X_val, y_val))
print(mean_squared_error(y_val, model.predict(X_val)))
#####################################################3
print("Heart Disease df")
heart_df = pd.read_csv("./heart.csv")

training_preprocessor = TrainingPreprocessing(
    heart_df, "target", "classification", "./heart"
)
X_train, y_train, X_val, y_val = training_preprocessor.common_preprocessing()
print("Random Forest Classifier")
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
print(model.score(X_val, y_val))
#######################################
print("Iris dataset")

iris = load_iris()
iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
iris_df["species"] = iris.target
training_preprocessor = TrainingPreprocessing(
    iris_df, "species", "classification", "./iris"
)
X_train, y_train, X_val, y_val = training_preprocessor.common_preprocessing()
print("Random Forest Classifier")
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
print(model.score(X_val, y_val))
