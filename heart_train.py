import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from xgboost import XGBClassifier


# =========================================
# LOAD DATASET
# =========================================

data = pd.read_csv("datasets/heart.csv")

print("\nFIRST 5 ROWS")
print(data.head())


# =========================================
# DATASET INFORMATION
# =========================================

print("\nDATASET SHAPE")
print(data.shape)

print("\nDATASET INFO")
print(data.info())

print("\nSTATISTICAL SUMMARY")
print(data.describe())

print("\nMISSING VALUES")
print(data.isnull().sum())


# =========================================
# FEATURES AND TARGET
# =========================================

X = data.drop("target", axis=1)

y = data["target"]

print("\nFEATURES")
print(X.head())

print("\nTARGET")
print(y.head())


# =========================================
# TRAIN TEST SPLIT
# =========================================

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


# =========================================
# FEATURE SCALING
# =========================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)

print("\nDATA SCALED SUCCESSFULLY")


# =========================================
# LOGISTIC REGRESSION
# =========================================

lr_model = LogisticRegression()

lr_model.fit(X_train, y_train)

lr_pred = lr_model.predict(X_test)

lr_accuracy = accuracy_score(y_test, lr_pred)

print("\nLOGISTIC REGRESSION ACCURACY")
print(lr_accuracy)


# =========================================
# RANDOM FOREST
# =========================================

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


# =========================================
# SVM
# =========================================

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


# =========================================
# XGBOOST
# =========================================

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


# =========================================
# MODEL COMPARISON
# =========================================

print("\nMODEL COMPARISON")

print(f"Logistic Regression Accuracy: {lr_accuracy}")

print(f"Random Forest Accuracy: {rf_accuracy}")

print(f"SVM Accuracy: {svm_accuracy}")

print(f"XGBoost Accuracy: {xgb_accuracy}")


# =========================================
# CONFUSION MATRIX
# =========================================

print("\nCONFUSION MATRIX")

print(confusion_matrix(y_test, lr_pred))


# =========================================
# CLASSIFICATION REPORT
# =========================================

print("\nCLASSIFICATION REPORT")

print(classification_report(y_test, lr_pred))


# =========================================
# SAVE BEST MODEL
# =========================================

pickle.dump(rf_model, open("heart_model.pkl", "wb"))
pickle.dump(scaler, open("heart_scaler.pkl", "wb"))

print("\nMODEL SAVED SUCCESSFULLY")


# =========================================
# SAMPLE PREDICTION
# =========================================

sample_data = pd.DataFrame(
    [[63,1,3,145,233,1,0,150,0,2.3,0,0,1]],
    columns=X.columns
)

sample_data = scaler.transform(sample_data)

prediction = rf_model.predict(sample_data)

print("\nPREDICTION RESULT")

if prediction[0] == 1:
    print("Heart Disease Detected")
else:
    print("No Heart Disease")