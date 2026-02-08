import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report
from accelera.src.automl.core.testing_preprocessing import TestingPreprocessing
from accelera.src.automl.core.training_preprocessing import (
    TrainingPreprocessing,
)

tatanic_df = pd.read_csv("Titanic-Dataset.csv")
tatanic_df_copy = tatanic_df[:5].copy()
training_preprocessor = TrainingPreprocessing(
    tatanic_df, "Survived", "classification", "./titanic_preprocessing"
)
X_train, y_train, X_val, y_val = training_preprocessor.common_preprocessing()
print("Titanic Dataset")
print("Random Forest Classifier")
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
print(model.score(X_val, y_val))
print("testing")
testing_preprocessor = TestingPreprocessing(tatanic_df_copy, "./titanic_preprocessing")
X_test, y_test = testing_preprocessor.common_preprocessing()

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

print("Review dataset")
review_df = pd.read_csv("./TestReviews.csv")
training_preprocessor = TrainingPreprocessing(
    review_df, "class", "classification", "./reviews", text_colums_name=["review"]
)
X_train, y_train, X_val, y_val = training_preprocessor.common_preprocessing()
print("Random Forest Classifier")
model = RandomForestClassifier(random_state=42, class_weight="balanced")
model.fit(X_train, y_train)
print(model.score(X_val, y_val))
print("Logistic Regression")
model = LogisticRegression(random_state=42, class_weight="balanced")
model.fit(X_train, y_train)
print(model.score(X_val, y_val))
print(confusion_matrix(y_val, model.predict(X_val)))
test_data = pd.DataFrame(
    {"review": ["This product is great!", "this product is terrible!"], "class": [1, 0]}
)
testing_preprocessor = TestingPreprocessing(test_data, "./reviews")
X_test, y_test = testing_preprocessor.common_preprocessing()
print("Predictions:")
print(model.predict(X_test))
print("correct prediction:", y_test)
print(classification_report(y_val, model.predict(X_val)))
