import streamlit as st
import pickle
import pandas as pd
st.title("Version2 test")
st.sidebar.header("Model Performance")

# =========================================
# LOAD MODELS
# =========================================

diabetes_model = pickle.load(open("diabetes_model.pkl", "rb"))

heart_model = pickle.load(open("heart_model.pkl", "rb"))

cancer_model = pickle.load(open("cancer_model.pkl", "rb"))

diabetes_scaler = pickle.load(open("diabetes_scaler.pkl", "rb"))

heart_scaler = pickle.load(open("heart_scaler.pkl", "rb"))

cancer_scaler = pickle.load(open("cancer_scaler.pkl", "rb"))


# =========================================
# PAGE TITLE
# =========================================

st.title("AI Multi Disease Prediction System")


# =========================================
# SIDEBAR
# =========================================

disease = st.sidebar.selectbox(
    "Select Disease",
    ["Diabetes", "Heart Disease", "Breast Cancer"]
)


# =========================================
# DIABETES PREDICTION
# =========================================

if disease == "Diabetes":

    st.header("Diabetes Prediction")

    pregnancies = st.number_input("Pregnancies")

    glucose = st.number_input("Glucose")

    blood_pressure = st.number_input("Blood Pressure")

    skin_thickness = st.number_input("Skin Thickness")

    insulin = st.number_input("Insulin")

    bmi = st.number_input("BMI")

    diabetes_pedigree = st.number_input("Diabetes Pedigree Function")

    age = st.number_input("Age")


    if st.button("Predict Diabetes"):

        input_data = pd.DataFrame(
            [[pregnancies, glucose, blood_pressure,
              skin_thickness, insulin, bmi,
              diabetes_pedigree, age]]
        )

        input_data = diabetes_scaler.transform(input_data)
        prediction = diabetes_model.predict(input_data)
        probability=diabetes_model.predict_proba(input_data)
        risk=probability[0][1]*100
        if risk<30:
            st.success("Low risk")
        elif risk<70:
            st.warning("Moderate risk")
        else:
            st.error("High risk")

        if prediction[0] == 1:
            st.error(f"Diabetes risk: {risk:.2f}%")
            st.progress(float(risk/100))
        else:
            st.success("No Diabetes Detected")
st.info(
"""
Diabetes is a chronic condition
where blood glucose levels become
too high.
"""
)

# =========================================
# HEART DISEASE PREDICTION
# =========================================

if disease == "Heart Disease":

    st.header("Heart Disease Prediction")

    age = st.number_input("Age")

    sex = st.number_input("Sex (1 = Male, 0 = Female)")

    cp = st.number_input("Chest Pain Type")

    trestbps = st.number_input("Resting Blood Pressure")

    chol = st.number_input("Cholesterol")

    fbs = st.number_input("Fasting Blood Sugar")

    restecg = st.number_input("Rest ECG")

    thalach = st.number_input("Max Heart Rate")

    exang = st.number_input("Exercise Induced Angina")

    oldpeak = st.number_input("Oldpeak")

    slope = st.number_input("Slope")

    ca = st.number_input("CA")

    thal = st.number_input("Thal")


    if st.button("Predict Heart Disease"):

        input_data = pd.DataFrame(
            [[age, sex, cp, trestbps, chol,
              fbs, restecg, thalach,
              exang, oldpeak, slope,
              ca, thal]]
        )
        input_data = heart_scaler.transform(input_data)
        probability = heart_model.predict_proba(input_data)
        risk = probability[0][1] * 100

        prediction = heart_model.predict(input_data)

        if prediction[0] == 1:
            st.error(f"Heart disease Risk: {risk:.2f}%")
            st.progress(float(risk/100))
        else:
            st.success("No Heart Disease Detected")
st.info(
    """Heart disease is a general term for conditions affecting the heart or blood vessels"""
)


# =========================================
# BREAST CANCER PREDICTION
# =========================================

if disease == "Breast Cancer":

    st.header("Breast Cancer Prediction")

    radius_mean = st.number_input("Radius Mean")

    texture_mean = st.number_input("Texture Mean")

    perimeter_mean = st.number_input("Perimeter Mean")

    area_mean = st.number_input("Area Mean")

    smoothness_mean = st.number_input("Smoothness Mean")

    compactness_mean = st.number_input("Compactness Mean")

    concavity_mean = st.number_input("Concavity Mean")

    concave_points_mean = st.number_input("Concave Points Mean")

    symmetry_mean = st.number_input("Symmetry Mean")

    fractal_dimension_mean = st.number_input("Fractal Dimension Mean")

    radius_se = st.number_input("Radius SE")

    texture_se = st.number_input("Texture SE")

    perimeter_se = st.number_input("Perimeter SE")

    area_se = st.number_input("Area SE")

    smoothness_se = st.number_input("Smoothness SE")

    compactness_se = st.number_input("Compactness SE")

    concavity_se = st.number_input("Concavity SE")

    concave_points_se = st.number_input("Concave Points SE")

    symmetry_se = st.number_input("Symmetry SE")

    fractal_dimension_se = st.number_input("Fractal Dimension SE")

    radius_worst = st.number_input("Radius Worst")

    texture_worst = st.number_input("Texture Worst")

    perimeter_worst = st.number_input("Perimeter Worst")

    area_worst = st.number_input("Area Worst")

    smoothness_worst = st.number_input("Smoothness Worst")

    compactness_worst = st.number_input("Compactness Worst")

    concavity_worst = st.number_input("Concavity Worst")

    concave_points_worst = st.number_input("Concave Points Worst")

    symmetry_worst = st.number_input("Symmetry Worst")

    fractal_dimension_worst = st.number_input("Fractal Dimension Worst")


    if st.button("Predict Breast Cancer"):

        input_data = pd.DataFrame(
            [[radius_mean, texture_mean, perimeter_mean,
              area_mean, smoothness_mean,
              compactness_mean, concavity_mean,
              concave_points_mean, symmetry_mean,
              fractal_dimension_mean, radius_se,
              texture_se, perimeter_se, area_se,
              smoothness_se, compactness_se,
              concavity_se, concave_points_se,
              symmetry_se, fractal_dimension_se,
              radius_worst, texture_worst,
              perimeter_worst, area_worst,
              smoothness_worst, compactness_worst,
              concavity_worst, concave_points_worst,
              symmetry_worst, fractal_dimension_worst]]
        )
        input_data = cancer_scaler.transform(input_data)
        prediction = cancer_model.predict(input_data)
        probability=cancer_model.predict_proba(input_data)
        risk = probability[0][1] * 100

        if prediction[0] == 1:
            st.error(f"Breast Cancer Risk: {risk:.2f}%")
            st.progress(float(risk/100))
        else:
            st.success("No Breast Cancer Detected")
st.info(
    """Breast cancer is a disease where abnormal cells in the breast tissues multiply uncontrolablly,usually forming tumors."""
)