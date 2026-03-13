import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVC

from accelera.src.automl.core.classical_testing_preprocessing import (
    ClassicalTestingPreprocessing,
)
from accelera.src.automl.core.classical_training_preprocessing import (
    ClassicalTrainingPreprocessing,
)
from accelera.src.automl.core.text_testing_preprocesing import (
    TextTestingPreprocessing,
)
from accelera.src.automl.core.text_training_preprocessing import (
    TextTrainingPreprocessing,
)

print("----------------------------Titanic Dataset-----------------------")
titanic_df = pd.read_csv("Titanic-Dataset.csv")
titanic_df_copy = titanic_df[:5].copy()
training_preprocessor = ClassicalTrainingPreprocessing(
    titanic_df, "Survived", "classification", "./titanic_preprocessing"
)
X_train, y_train, X_val, y_val = training_preprocessor.common_preprocessing()
print("Random Forest Classifier")
model = RandomForestClassifier(random_state=42, class_weight="balanced")
model.fit(X_train, y_train)
print(model.score(X_val, y_val))
print("Logistic Regression")
model = LogisticRegression(random_state=42, class_weight="balanced")
model.fit(X_train, y_train)
print("Score:", model.score(X_val, y_val))
print("SVC")
model = SVC()
model.fit(X_train, y_train)
print("Score : ", model.score(X_val, y_val))
print("Testing")
testing_preprocessor = ClassicalTestingPreprocessing(
    titanic_df_copy, "./titanic_preprocessing"
)
X_test, y_test = testing_preprocessor.common_preprocessing()
print("Predictions SVC:")
print(model.predict(X_test))
print("Actual\n", y_test)
print("Confusion Matrix")
print(confusion_matrix(y_val, model.predict(X_val)))
print("Classification Report")
print(classification_report(y_val, model.predict(X_val)))

##############################################
print("----------------------------House Prices Dataset-----------------------")
price_df = pd.read_csv("Housing.csv")
price_df_copy = price_df[:5].copy()
training_preprocessor = ClassicalTrainingPreprocessing(
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
testing_preprocessor = ClassicalTestingPreprocessing(
    price_df_copy, "./house_price_preprocessing"
)
X_test, y_test = testing_preprocessor.common_preprocessing()
print("Predictions:")
print(model.predict(X_test))
print("Actual")
print(y_test)

#####################################################3
print(
    "----------------------------Heart Disease Dataset-----------------------"
)

heart_df = pd.read_csv("./heart.csv")

training_preprocessor = ClassicalTrainingPreprocessing(
    heart_df, "target", "classification", "./heart"
)
X_train, y_train, X_val, y_val = training_preprocessor.common_preprocessing()
print("Random Forest Classifier")
model = RandomForestClassifier(random_state=42, class_weight="balanced")
model.fit(X_train, y_train)
print("Score: ", model.score(X_val, y_val))
print("Logistic Regression")
model = LogisticRegression(random_state=42, class_weight="balanced")
model.fit(X_train, y_train)
print("Score:", model.score(X_val, y_val))
print("Confusion Matrix")
print(confusion_matrix(y_val, model.predict(X_val)))
print("Classification Report")
print(classification_report(y_val, model.predict(X_val)))

#######################################
print("----------------------------Purchase Dataset-----------------------")

purchase_df = pd.read_csv("./customer_purchase_data.csv")

training_preprocessor = ClassicalTrainingPreprocessing(
    purchase_df, "PurchaseStatus", "classification", "./PurchaseStatus"
)
X_train, y_train, X_val, y_val = training_preprocessor.common_preprocessing()
print("Random Forest Classifier")
model = RandomForestClassifier(random_state=42, class_weight="balanced")
model.fit(X_train, y_train)
print("Score: ", model.score(X_val, y_val))
print("Logistic Regression")
model = LogisticRegression(random_state=42, class_weight="balanced")
model.fit(X_train, y_train)
print("Score:", model.score(X_val, y_val))
print("Confusion Matrix")
print(confusion_matrix(y_val, model.predict(X_val)))
print("Classification Report")
print(classification_report(y_val, model.predict(X_val)))

####################################
print("----------------------------Iris Dataset-----------------------")

iris = load_iris()
iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
iris_df["species"] = iris.target
training_preprocessor = ClassicalTrainingPreprocessing(
    iris_df, "species", "classification", "./iris"
)
X_train, y_train, X_val, y_val = training_preprocessor.common_preprocessing()
print("Random Forest Classifier")
model = RandomForestClassifier(random_state=42, class_weight="balanced")
model.fit(X_train, y_train)
print("Score: ", model.score(X_val, y_val))
print("Logistic Regression")
model = LogisticRegression(random_state=42, class_weight="balanced")
model.fit(X_train, y_train)
print("Score:", model.score(X_val, y_val))
print("Confusion Matrix")
print(confusion_matrix(y_val, model.predict(X_val)))
print("Classification Report")
print(classification_report(y_val, model.predict(X_val)))


print("----------------------------Review dataset-----------------------")
review_df = pd.read_csv("./TestReviews.csv")
training_preprocessor = TextTrainingPreprocessing(
    review_df, "class", text_col="review", folder_path="./reviews"
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
    {
        "review": [
            "This product is great!",
            "this product is terrible!",
            "I don't like it",
            "I like it",
        ],
        "class": [1, 0, 0, 1],
    }
)
print("Testing")
testing_preprocessor = TextTestingPreprocessing(test_data, "./reviews")
X_test, y_test = testing_preprocessor.common_preprocessing()
print("Predictions:")
print(model.predict(X_test))
print("correct prediction:", y_test)
print(
    "------------------------Sentiment analysis dataset-----------------------"
)

sentiment_df = pd.read_csv("./DailyDialog.csv")
training_preprocessor = TextTrainingPreprocessing(
    sentiment_df,
    "sentiment",
    text_col="text",
    folder_path="./sentiment",
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
