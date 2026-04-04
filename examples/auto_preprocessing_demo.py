import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from accelera.src.automl.core.classical_training_preprocessing import (
    ClassicalTrainingPreprocessing,
)
from accelera.src.automl.core.text_testing_preprocesing import (
    TextTestingPreprocessing,
)
from accelera.src.automl.core.text_training_preprocessing import (
    TextTrainingPreprocessing,
)

print(
    "----------------------------student_exam_performance_dataset-----------------------"
)
student_exam = pd.read_csv("student_exam_performance_dataset.csv")
training_preprocessor = ClassicalTrainingPreprocessing(
    student_exam, "pass_fail", "Classification", "./student_exam"
)
X_train, y_train, X_val, y_val = training_preprocessor.common_preprocessing()
print("Random Forest classifier")
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
print("Score: ", model.score(X_val, y_val))
print("Logistic Regression")
model = LogisticRegression(random_state=42, class_weight="balanced")
model.fit(X_train, y_train)
print("Score: ", model.score(X_val, y_val))
print(classification_report(y_val, model.predict(X_val)))
print("----------------------------job salary Dataset-----------------------")
job = pd.read_csv("job_salary_prediction_dataset.csv")
training_preprocessor = ClassicalTrainingPreprocessing(
    job, "salary", "Regression", "./job_salary"
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

print("----------------------------spine Dataset-----------------------")
spine = pd.read_csv("Dataset_spine.csv")
training_preprocessor = ClassicalTrainingPreprocessing(
    spine, "Class_att", "classification", "./Dataset_spine"
)
X_train, y_train, X_val, y_val = training_preprocessor.common_preprocessing()
print("Logistic Regression")
model = LogisticRegression(random_state=42, class_weight="balanced")
model.fit(X_train, y_train)
print("Score:", model.score(X_val, y_val))
print("SVC")
model = SVC()
model.fit(X_train, y_train)
print("Score : ", model.score(X_val, y_val))
print("Confusion Matrix")
print(confusion_matrix(y_val, model.predict(X_val)))
print("Classification Report")
print(classification_report(y_val, model.predict(X_val)))
print("----------------------------diabetes Dataset-----------------------")
diab = pd.read_csv("diabetes_dataset.csv")
training_preprocessor = ClassicalTrainingPreprocessing(
    diab, "diagnosed_diabetes", "classification", "./diabetes_dataset"
)
X_train, y_train, X_val, y_val = training_preprocessor.common_preprocessing()
print("Logistic Regression")
model = LogisticRegression(random_state=42, class_weight="balanced")
model.fit(X_train, y_train)
print("Score:", model.score(X_val, y_val))
print("SVC")
model = SVC()
model.fit(X_train, y_train)
print("Score : ", model.score(X_val, y_val))
print("Confusion Matrix")
print(confusion_matrix(y_val, model.predict(X_val)))
print("Classification Report")
print(classification_report(y_val, model.predict(X_val)))

print("----------------------------student Dataset-----------------------")
student = pd.read_csv("student_placement_synthetic.csv")
training_preprocessor = ClassicalTrainingPreprocessing(
    student, "placement_status", "classification", "./student_report"
)
X_train, y_train, X_val, y_val = training_preprocessor.common_preprocessing()

print("Logistic Regression")
model = LogisticRegression(random_state=42, class_weight="balanced")
model.fit(X_train, y_train)
print("Score:", model.score(X_val, y_val))
print("Confusion Matrix")
print(confusion_matrix(y_val, model.predict(X_val)))
print("Classification Report")
print(classification_report(y_val, model.predict(X_val)))

print("----------------------------Titanic Dataset-----------------------")
titanic_df = pd.read_csv("Titanic-Dataset.csv")
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
print("Predictions SVC:")
print("Confusion Matrix")
print(confusion_matrix(y_val, model.predict(X_val)))
print("Classification Report")
print(classification_report(y_val, model.predict(X_val)))

##############################################
print("----------------------------House Prices Dataset-----------------------")
price_df = pd.read_csv("Housing.csv")
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
model = LogisticRegression(
    random_state=42, class_weight="balanced", max_iter=1000
)
model.fit(X_train, y_train)
print("Score:", model.score(X_val, y_val))
print("Confusion Matrix")
print(confusion_matrix(y_val, model.predict(X_val)))
print("Classification Report")
print(classification_report(y_val, model.predict(X_val)))
print("--------------------------without auto preporcessing")
heart_df = pd.read_csv("./heart.csv")
heart_df.drop_duplicates(inplace=True)
X = heart_df.drop(columns=["target"])
y = heart_df["target"]
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print("Random Forest Classifier")
model = RandomForestClassifier(random_state=42, class_weight="balanced")
model.fit(X_train, y_train)
print("Score: ", model.score(X_val, y_val))
print("Logistic Regression")
model = LogisticRegression(
    random_state=42, class_weight="balanced", max_iter=1000
)
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
model = LogisticRegression(
    random_state=42, class_weight="balanced", max_iter=10000
)
model.fit(X_train, y_train)
print("Score:", model.score(X_val, y_val))
print("Confusion Matrix")
print(confusion_matrix(y_val, model.predict(X_val)))
print("Classification Report")
print(classification_report(y_val, model.predict(X_val)))
print("--------------------------without auto preporcessing")
purchase_df = pd.read_csv("./customer_purchase_data.csv")
purchase_df.drop_duplicates(inplace=True)
X = purchase_df.drop(columns=["PurchaseStatus"])
y = purchase_df["PurchaseStatus"]
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print("Random Forest Classifier")
model = RandomForestClassifier(random_state=42, class_weight="balanced")
model.fit(X_train, y_train)
print("Score: ", model.score(X_val, y_val))
print("Logistic Regression")
model = LogisticRegression(
    random_state=42, class_weight="balanced", max_iter=10000
)
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
X_val, y_val = testing_preprocessor.common_preprocessing()
print("Predictions:")
print(model.predict(X_val))
print("correct prediction:", y_val)
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
