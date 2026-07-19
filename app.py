import streamlit as st
import pickle
import pandas as pd
st.set_page_config(
    page_title="Medscope",
    page_icon="logo.png",
    layout="wide"
)
st.title("🏥 AI Multi Disease Prediction System")
st.caption("Machine Learning Based Clinical Decision Support Tool")
st.warning(
    "⚠️ This application is intended for educational purposes only and should not replace professional medical advice."
)
# =========================================
# LOAD MODELS ONLY ONCE
# =========================================

@st.cache_resource
def load_models():

    diabetes_model = pickle.load(open("diabetes_model.pkl", "rb"))
    heart_model = pickle.load(open("heart_model.pkl", "rb"))
    cancer_model = pickle.load(open("cancer_model.pkl", "rb"))

    diabetes_scaler = pickle.load(open("diabetes_scaler.pkl", "rb"))
    heart_scaler = pickle.load(open("heart_scaler.pkl", "rb"))
    cancer_scaler = pickle.load(open("cancer_scaler.pkl", "rb"))

    return (
        diabetes_model,
        heart_model,
        cancer_model,
        diabetes_scaler,
        heart_scaler,
        cancer_scaler
    )
# =========================================
# LOAD MODELS
# =========================================

(
    diabetes_model,
    heart_model,
    cancer_model,
    diabetes_scaler,
    heart_scaler,
    cancer_scaler
) = load_models()

# =========================================
# SIDEBAR
# =========================================

disease = st.sidebar.selectbox(
    "Select Disease",
    ["Diabetes", "Heart Disease", "Breast Cancer"]
)

st.sidebar.markdown("---")
st.sidebar.header("📊 Model Information")

if disease == "Diabetes":
    st.sidebar.metric("Accuracy", "75.32%")
    st.sidebar.write("**Algorithm:** Logistic Regression")
    st.sidebar.write("**Dataset:** PIMA Indians")
    st.sidebar.write("**Features:** 8")

elif disease == "Heart Disease":
    st.sidebar.metric("Accuracy", "98.36%")
    st.sidebar.write("**Algorithm:** Random Forest")
    st.sidebar.write("**Dataset:** UCI Heart Disease")
    st.sidebar.write("**Features:** 13")

elif disease == "Breast Cancer":
    st.sidebar.metric("Accuracy", "98.24%")
    st.sidebar.write("**Algorithm:** Random Forest")
    st.sidebar.write("**Dataset:** Breast Cancer Wisconsin")
    st.sidebar.write("**Features:** 30")
# =========================================
# DIABETES PREDICTION
# =========================================

if disease == "Diabetes":
    st.info(
    """
    Diabetes is a chronic condition
    where blood glucose levels become
    too high.
    """
    )

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
        # ----------------------------
# Risk Level
# ----------------------------

        if risk < 30:
            st.success("🟢 Low Risk")
        elif risk < 70:
            st.warning("🟡 Moderate Risk")
        else:
            st.error("🔴 High Risk")

        # ----------------------------
        # Prediction
        # ----------------------------

        if prediction[0] == 1:

            st.error(f"Diabetes Detected")

            st.metric(
                label="Prediction Probability",
                value=f"{risk:.2f}%"
            )

            st.progress(min(risk / 100, 1.0))

        else:

            st.success("No Diabetes Detected")

            st.metric(
                label="Prediction Probability",
                value=f"{100-risk:.2f}%"
            )

            st.progress(min((100-risk)/100,1.0))

# ----------------------------
# Patient Summary
# ----------------------------

        st.subheader("Patient Summary")

        summary = pd.DataFrame({
            "Parameter":[
                "Pregnancies",
                "Glucose",
                "Blood Pressure",
                "Skin Thickness",
                "Insulin",
                "BMI",
                "Diabetes Pedigree",
                "Age"
            ],
            "Value":[
                pregnancies,
                glucose,
                blood_pressure,
                skin_thickness,
                insulin,
                bmi,
                diabetes_pedigree,
                age
            ]
        })

        st.dataframe(summary, use_container_width=True)
        
        

# =========================================
# HEART DISEASE PREDICTION
# =========================================

if disease == "Heart Disease":

    st.header("Heart Disease Prediction")
    st.info(
        """Heart disease is a general term for conditions affecting the heart or blood vessels"""
    )

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
            ca, thal]],

            columns=[
                "age","sex","cp","trestbps","chol",
                "fbs","restecg","thalach","exang",
                "oldpeak","slope","ca","thal"
            ]
        )

        input_data = heart_scaler.transform(input_data)

        prediction = heart_model.predict(input_data)

        probability = heart_model.predict_proba(input_data)

        risk = probability[0][1] * 100


        # ------------------------
        # Risk Level
        # ------------------------

        if risk < 30:
            st.success("🟢 Low Risk")

        elif risk < 70:
            st.warning("🟡 Moderate Risk")

        else:
            st.error("🔴 High Risk")


        # ------------------------
        # Prediction
        # ------------------------

        if prediction[0] == 1:

            st.error("Heart Disease Detected")

            st.metric(
                "Prediction Probability",
                f"{risk:.2f}%"
            )

            st.progress(min(risk/100,1.0))

        else:

            st.success("No Heart Disease Detected")

            st.metric(
                "Prediction Probability",
                f"{100-risk:.2f}%"
            )

            st.progress(min((100-risk)/100,1.0))


        # ------------------------
        # Patient Summary
        # ------------------------

        st.subheader("Patient Summary")

        summary = pd.DataFrame({
            "Parameter":[
                "Age",
                "Sex",
                "Chest Pain",
                "Resting BP",
                "Cholesterol",
                "Fasting Blood Sugar",
                "Rest ECG",
                "Max Heart Rate",
                "Exercise Angina",
                "Old Peak",
                "Slope",
                "CA",
                "Thal"
            ],
            "Value":[
                age,
                sex,
                cp,
                trestbps,
                chol,
                fbs,
                restecg,
                thalach,
                exang,
                oldpeak,
                slope,
                ca,
                thal
            ]
        })

        st.dataframe(summary, use_container_width=True)

# =========================================
# BREAST CANCER PREDICTION
# =========================================

if disease == "Breast Cancer":

    st.header("Breast Cancer Prediction")
    st.info(
        """Breast cancer is a disease where abnormal cells in the breast tissues multiply uncontrolablly,usually forming tumors."""
    )

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
            [[
                radius_mean, texture_mean, perimeter_mean,
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
                symmetry_worst, fractal_dimension_worst
            ]]
        )

        input_data = cancer_scaler.transform(input_data)

        prediction = cancer_model.predict(input_data)

        probability = cancer_model.predict_proba(input_data)

        risk = probability[0][1] * 100


        # ------------------------
        # Risk Level
        # ------------------------

        if risk < 30:
            st.success("🟢 Low Risk")

        elif risk < 70:
            st.warning("🟡 Moderate Risk")

        else:
            st.error("🔴 High Risk")


        # ------------------------
        # Prediction
        # ------------------------

        if prediction[0] == 1:

            st.error("Breast Cancer Detected")

            st.metric(
                "Prediction Probability",
                f"{risk:.2f}%"
            )

            st.progress(min(risk/100,1.0))

        else:

            st.success("No Breast Cancer Detected")

            st.metric(
                "Prediction Probability",
                f"{100-risk:.2f}%"
            )

            st.progress(min((100-risk)/100,1.0))


        # ------------------------
        # Patient Summary
        # ------------------------

        st.subheader("Patient Summary")

        summary = pd.DataFrame({
            "Parameter":[
                "Radius Mean",
                "Texture Mean",
                "Perimeter Mean",
                "Area Mean",
                "Smoothness Mean",
                "Compactness Mean",
                "Concavity Mean",
                "Concave Points Mean",
                "Symmetry Mean",
                "Fractal Dimension Mean",
                "Radius SE",
                "Texture SE",
                "Perimeter SE",
                "Area SE",
                "Smoothness SE",
                "Compactness SE",
                "Concavity SE",
                "Concave Points SE",
                "Symmetry SE",
                "Fractal Dimension SE",
                "Radius Worst",
                "Texture Worst",
                "Perimeter Worst",
                "Area Worst",
                "Smoothness Worst",
                "Compactness Worst",
                "Concavity Worst",
                "Concave Points Worst",
                "Symmetry Worst",
                "Fractal Dimension Worst"
            ],

            "Value":[
                radius_mean,
                texture_mean,
                perimeter_mean,
                area_mean,
                smoothness_mean,
                compactness_mean,
                concavity_mean,
                concave_points_mean,
                symmetry_mean,
                fractal_dimension_mean,
                radius_se,
                texture_se,
                perimeter_se,
                area_se,
                smoothness_se,
                compactness_se,
                concavity_se,
                concave_points_se,
                symmetry_se,
                fractal_dimension_se,
                radius_worst,
                texture_worst,
                perimeter_worst,
                area_worst,
                smoothness_worst,
                compactness_worst,
                concavity_worst,
                concave_points_worst,
                symmetry_worst,
                fractal_dimension_worst
            ]
        })

        st.dataframe(summary, use_container_width=True)