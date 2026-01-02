import streamlit as st
import pandas as pd
import joblib
import numpy as np

# ==============================
# Load trained objects
# ==============================
model = joblib.load("titanic_model.pkl")
scaler = joblib.load("scaler.pkl")
model_columns = joblib.load("columns.pkl")

st.title("🚢 Titanic Survival Predictor")
st.write("Enter passenger details to predict survival probability.")

# ==============================
# Sidebar inputs
# ==============================
with st.sidebar:
    st.header("Passenger Information")

    pclass = st.selectbox("Ticket Class", [1, 2, 3])
    age = st.slider("Age", 0, 100, 30)

    sibsp = st.number_input("Siblings / Spouses Aboard", 0, 10, 0)
    parch = st.number_input("Parents / Children Aboard", 0, 10, 0)

    fare = st.number_input("Fare Paid", 0.0, 500.0, 32.0)

    sex = st.selectbox("Sex", ["male", "female"])
    embarked = st.selectbox("Embarked Port", ["S", "C", "Q"])

    title = st.selectbox(
        "Title",
        ["Mr", "Mrs", "Miss", "Master", "Misc"]
    )

# ==============================
# Preprocessing (MATCH TRAINING)
# ==============================
def preprocess_input():
    # Create empty row with correct columns
    input_data = pd.DataFrame(0, index=[0], columns=model_columns)

    # Basic features
    input_data["Pclass"] = pclass
    input_data["Age"] = age
    input_data["Log_Fare"] = np.log1p(fare)

    # Family features
    family_size = sibsp + parch + 1
    input_data["FamilySize"] = family_size

    if family_size == 1:
        input_data["FamilyGroup_Alone"] = 1
    elif 2 <= family_size <= 4:
        input_data["FamilyGroup_Small"] = 1
    else:
        input_data["FamilyGroup_Large"] = 1

    # Sex encoding
    input_data[f"Sex_{sex}"] = 1

    # Embarked encoding
    input_data[f"Embarked_{embarked}"] = 1

    # Title encoding
    input_data[f"Title_{title}"] = 1

    # Cabin (default: no cabin info)
    input_data["HasCabin"] = 0

    # Fare bins (must match training logic)
    if fare < 10:
        input_data["FareBin_Low"] = 1
    elif fare < 30:
        input_data["FareBin_Mid"] = 1
    elif fare < 100:
        input_data["FareBin_High"] = 1
    else:
        input_data["FareBin_Very High"] = 1

    # Scale
    return scaler.transform(input_data)

# ==============================
# Prediction
# ==============================
if st.button("Predict Survival"):
    features = preprocess_input()
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1]

    if prediction == 1:
        st.success(f"✅ Passenger likely SURVIVED\n\nProbability: **{probability:.2%}**")
    else:
        st.error(f"❌ Passenger likely DID NOT survive\n\nProbability: **{probability:.2%}**")
