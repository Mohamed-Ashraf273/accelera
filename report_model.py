from mainera.src.wrappers.model_report_wrapper import ModelReport
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import  train_test_split
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
data=pd.read_csv("Titanic-Dataset.csv")
data.drop(axis=1,columns=['PassengerId','Name','Ticket','Cabin'],inplace=True)
print(data.head())
print(data.isnull().sum())
data['Age'].fillna(data['Age'].mean(),inplace=True)
data.dropna(inplace=True)
encoder=LabelEncoder()
data['Sex']=encoder.fit_transform(data['Sex'])
data['Embarked']=encoder.fit_transform(data['Embarked'])
print(data.isnull().sum())
X,y=data.drop(columns=['Survived'],axis=1),data['Survived']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)
print(y_test)
scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)
model=SVC()
model.fit(X_train,y_train)
predict=model.predict(X_test)
accuracy=accuracy_score(y_test,predict)
results=[{"metric name":"accuracy","result":accuracy,"tuple_argums":None},
         {"metric name":"accuracy","result":accuracy,"tuple_argums":None}]
report=ModelReport("model_report",results=results)
report.execute()
####################3Tensorflow
# 5️⃣ Build model
model = keras.Sequential([
    layers.Dense(64, activation="relu", input_shape=(X_train.shape[1],)),
    layers.Dense(32, activation="relu"),
    layers.Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy","recall","precision"]
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=30,
    batch_size=16,
    verbose=1
)
predictions=model.predict(X_test)
predictions=(predictions > 0.5).astype(int)
accuracy=accuracy_score(y_test,predictions)
classification=classification_report(y_test,predictions)
confusion=confusion_matrix(y_test,predictions)
results=[{"metric name":"accuracy","result":accuracy,"tuple_argums":None},
         {"metric name":"accuracy","result":accuracy,"tuple_argums":None},
         {"metric name":"classification","result":classification,"tuple_argums":None},
         {"metric name":"confusion","result":confusion,"tuple_argums":None}]
print("\nTraining history keys:", history.history.keys())
report=ModelReport("model_report",results=results,history=history.history)
report.execute()