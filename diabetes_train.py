import pandas as pd

# Load dataset
data = pd.read_csv("datasets/diabetes.csv")

# First 5 rows
print("\nFIRST 5 ROWS")
print(data.head())

# Shape of dataset
print("\nDATASET SHAPE")
print(data.shape)

# Information about dataset
print("\nDATASET INFO")
print(data.info())

# Statistical summary
print("\nSTATISTICAL SUMMARY")
print(data.describe())

# Check missing values
print("\nMISSING VALUES")
print(data.isnull().sum())

# Features (inputs)
X = data.drop("Outcome", axis=1)

# Target (output)
y = data["Outcome"]

print("\nFEATURES")
print(X.head())

print("\nTARGET")
print(y.head())

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTRAINING DATA SHAPE")
print(X_train.shape)

print("\nTESTING DATA SHAPE")
print(X_test.shape)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)

print("\nDATA SCALED SUCCESSFULLY")

from sklearn.linear_model import LogisticRegression

# Create model
model = LogisticRegression()

# Train model
model.fit(X_train, y_train)

print("\nMODEL TRAINED SUCCESSFULLY")
# Make predictions
y_pred = model.predict(X_test)

print("\nPREDICTIONS")
print(y_pred[:10])
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)

print("\nMODEL ACCURACY")
print(accuracy)
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

print("\nCONFUSION MATRIX")
print(cm)
from sklearn.metrics import classification_report

print("\nCLASSIFICATION REPORT")
print(classification_report(y_test, y_pred))
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42
)

rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)

rf_accuracy = accuracy_score(y_test, rf_pred)

print("\nRANDOM FOREST ACCURACY")
print(rf_accuracy)

from sklearn.svm import SVC

svm_model = SVC(
    kernel='rbf',
    C=1.0,
    gamma='scale'
)

svm_model.fit(X_train, y_train)

svm_pred = svm_model.predict(X_test)

svm_accuracy = accuracy_score(y_test, svm_pred)

print("\nSVM ACCURACY")
print(svm_accuracy)

from xgboost import XGBClassifier

xgb_model = XGBClassifier(
    n_estimators=100,
    learning_rate=0.05,
    max_depth=4,
    random_state=42
)

xgb_model.fit(X_train, y_train)

xgb_pred = xgb_model.predict(X_test)

xgb_accuracy = accuracy_score(y_test, xgb_pred)

print("\nXGBOOST ACCURACY")
print(xgb_accuracy)

print("\nMODEL COMPARISON")

print(f"Logistic Regression Accuracy: {accuracy}")

print(f"Random Forest Accuracy: {rf_accuracy}")

print(f"SVM Accuracy: {svm_accuracy}")

print(f"XGBoost Accuracy: {xgb_accuracy}")

import pickle

pickle.dump(model, open("diabetes_model.pkl", "wb"))
pickle.dump(scaler, open("diabetes_scaler.pkl", "wb"))

print("\nMODEL SAVED SUCCESSFULLY")

sample_data = pd.DataFrame(
    [[5,166,72,19,175,25.8,0.587,51]],
    columns=X.columns
)

sample_data = scaler.transform(sample_data)

prediction = model.predict(sample_data)

print("\nPREDICTION RESULT")

if prediction[0] == 1:
    print("Diabetes Detected")
else:
    print("No Diabetes")