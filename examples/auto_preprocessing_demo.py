from accelera.src.automl.utils.preprocessing import *
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn.svm import SVC

tatanic_df = pd.read_csv("Titanic-Dataset.csv")
print(tatanic_df.head())
X_train, y_train, X_test, y_test = common_preprocessing(tatanic_df, "Survived")
print(X_train[:5])
print("Titanic Dataset")
print("Random Forest Classifier")
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
print(model.score(X_test, y_test))
print("Logistic Regression")
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)
print(model.score(X_test, y_test))
print("SVC")
model = SVC()
model.fit(X_train, y_train)
print(model.score(X_test, y_test))

##############################################
print("House Price")

price_df = pd.read_csv("Housing.csv")
X_train, y_train, X_test, y_test = common_preprocessing(price_df, "price","regression")

print(X_train[:5])
print("Random Forest Regressor")
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)
print(model.score(X_test, y_test))
print(mean_squared_error(y_test, model.predict(X_test)))
model = LinearRegression()
print("Linear Regression")
model.fit(X_train, y_train)
print(model.score(X_test, y_test))
print(mean_squared_error(y_test, model.predict(X_test)))
