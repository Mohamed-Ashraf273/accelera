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

print("----------------------------Titanic Dataset-----------------------")
titanic_df = pd.read_csv("Titanic-Dataset.csv")
titanic_df_copy = titanic_df[:5].copy()
training_preprocessor = TrainingPreprocessing(
    titanic_df, "Survived", "classification", "./titanic_preprocessing"
)
X_train, y_train, X_val, y_val = training_preprocessor.common_preprocessing()
print("Random Forest Classifier")
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
print(model.score(X_val, y_val))
print("Logistic Regression")
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)
print("Score:", model.score(X_val, y_val))
print("SVC")
model = SVC()
model.fit(X_train, y_train)
print("Score : ", model.score(X_val, y_val))
print("Testing")
testing_preprocessor = TestingPreprocessing(titanic_df_copy, "./titanic_preprocessing")
X_test, y_test = testing_preprocessor.common_preprocessing()
print("Predictions SVC:")
print(model.predict(X_test))
print("Actual", y_test)
##############################################
print("----------------------------House Prices Dataset-----------------------")
price_df = pd.read_csv("Housing.csv")
price_df_copy = price_df[:5].copy()
training_preprocessor = TrainingPreprocessing(
    price_df, "price", "regression", "./house_price_preprocessing"
)
X_train, y_train, X_val, y_val = training_preprocessor.common_preprocessing()

print("Random Forest Regressor")
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)
print("Score: ", model.score(X_val, y_val))
print("MSE: ", mean_squared_error(y_val, model.predict(X_val)))
print("Linear Regression")
model = LinearRegression()
model.fit(X_train, y_train)
print("Score: ", model.score(X_val, y_val))
print("MSE: ", mean_squared_error(y_val, model.predict(X_val)))
print("Testing")
testing_preprocessor = TestingPreprocessing(
    price_df_copy, "./house_price_preprocessing"
)
X_test, y_test = testing_preprocessor.common_preprocessing()
print("Predictions:")
print(model.predict(X_test))
print("Actual")
print(y_test)
#####################################################3
print("----------------------------Heart Disease Dataset-----------------------")

heart_df = pd.read_csv("./heart.csv")

training_preprocessor = TrainingPreprocessing(
    heart_df, "target", "classification", "./heart"
)
X_train, y_train, X_val, y_val = training_preprocessor.common_preprocessing()
print("Random Forest Classifier")
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
print("Score: ", model.score(X_val, y_val))
print("Logistic Regression")
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)
print("Score:", model.score(X_val, y_val))
#######################################
print("----------------------------Iris Dataset-----------------------")

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
print("Score: ", model.score(X_val, y_val))
print("Logistic Regression")
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)
print("Score:", model.score(X_val, y_val))
print("----------------------------Review dataset-----------------------")
review_df = pd.read_csv("./TestReviews.csv")
training_preprocessor = TrainingPreprocessing(
    review_df, "class", "classification", "./reviews", text_colums_name=["review"]
)
X_train, y_train, X_val, y_val = training_preprocessor.common_preprocessing()
print("Random Forest Classifier")
model = RandomForestClassifier(random_state=42, class_weight="balanced")
model.fit(X_train, y_train)
print("Score: ", model.score(X_val, y_val))
print("Logistic Regression")
model = LogisticRegression(random_state=42, class_weight="balanced")
model.fit(X_train, y_train)
print("Score: ", model.score(X_val, y_val))
print("Confusion Matrix")
print(confusion_matrix(y_val, model.predict(X_val)))
print("Classification Report")

print(classification_report(y_val, model.predict(X_val)))
test_data = pd.DataFrame(
    {"review": ["This product is great!", "this product is terrible!"], "class": [1, 0]}
)
print("Testing")
testing_preprocessor = TestingPreprocessing(test_data, "./reviews")
X_test, y_test = testing_preprocessor.common_preprocessing()
print("Predictions:")
print(model.predict(X_test))
print("correct prediction:", y_test)
print("----------------------------Sentiment analysis dataset-----------------------")

sentiment_df = pd.read_csv("./DailyDialog.csv")
training_preprocessor = TrainingPreprocessing(
    sentiment_df,
    "sentiment",
    "classification",
    "./sentiment",
    text_colums_name=["text"],
)

X_train, y_train, X_val, y_val = training_preprocessor.common_preprocessing()
print("Random Forest Classifier")
model = RandomForestClassifier(random_state=42, class_weight="balanced")
model.fit(X_train, y_train)
print("Score:", model.score(X_val, y_val))
print("Logistic Regression")
model = LogisticRegression(random_state=42, class_weight="balanced")
model.fit(X_train, y_train)
print("Score: ", model.score(X_val, y_val))
print("Confusion Matrix")
print(confusion_matrix(y_val, model.predict(X_val)))
print("Classification Report")
print(classification_report(y_val, model.predict(X_val)))
