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

data = pd.read_csv("datasets/breast.csv")

print("\nFIRST 5 ROWS")
print(data.head())


# =========================================
# REMOVE UNNECESSARY COLUMNS
# =========================================

# Remove empty column if present
if 'Unnamed: 32' in data.columns:
    data.drop('Unnamed: 32', axis=1, inplace=True)

# Remove ID column
if 'id' in data.columns:
    data.drop('id', axis=1, inplace=True)


# =========================================
# CONVERT TARGET COLUMN
# M = Malignant = 1
# B = Benign = 0
# =========================================

data['diagnosis'] = data['diagnosis'].map({
    'M': 1,
    'B': 0
})


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

X = data.drop("diagnosis", axis=1)

y = data["diagnosis"]

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

print(confusion_matrix(y_test, rf_pred))


# =========================================
# CLASSIFICATION REPORT
# =========================================

print("\nCLASSIFICATION REPORT")

print(classification_report(y_test, rf_pred))


# =========================================
# SAVE BEST MODEL
# =========================================

# Change this model if another gives highest accuracy
pickle.dump(svm_model, open("cancer_model.pkl", "wb"))
pickle.dump(scaler, open("cancer_scaler.pkl", "wb"))

print("\nMODEL SAVED SUCCESSFULLY")


# =========================================
# SAMPLE PREDICTION
# =========================================

sample_data = pd.DataFrame(
    [[17.99,10.38,122.8,1001.0,0.1184,0.2776,0.3001,
      0.1471,0.2419,0.07871,1.095,0.9053,8.589,
      153.4,0.006399,0.04904,0.05373,0.01587,
      0.03003,0.006193,25.38,17.33,184.6,2019.0,
      0.1622,0.6656,0.7119,0.2654,0.4601,0.1189]],
    columns=X.columns
)

sample_data = scaler.transform(sample_data)

prediction = svm_model.predict(sample_data)

print("\nPREDICTION RESULT")

if prediction[0] == 1:
    print("Breast Cancer Detected")
else:
    print("No Breast Cancer")