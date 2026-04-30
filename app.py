import streamlit as st
import pandas as pd
from tensorflow.keras.models import load_model
import pickle

st.title("Passenger Survival Chance in the Titanic Journey")

# ---------------- INPUTS ----------------
pclass = st.slider("Enter Passenger Class", 1, 3)
sex = st.selectbox("Passenger Gender", ['male', 'female'])
sibsp = st.slider("No. of Siblings/Spouse", 0, 8)
parch = st.slider("No. of Parents/Children", 0, 8)
fare = st.number_input("Passenger Fare", 0.0)

embarked = st.selectbox(
    "Boarding Station",
    ['Southhampton', 'Chebourg', 'Queenstown']
)

# ---------------- DATAFRAME ----------------
data = pd.DataFrame([{
    'Pclass': pclass,
    'Sex': sex,
    'SibSp': sibsp,
    'Parch': parch,
    'Fare': fare,
    'Embarked': embarked
}])

# ---------------- LOAD MODEL ----------------
model = load_model('model.h5')

with open('label_encoder.pkl', 'rb') as file:
    label = pickle.load(file)

with open('onehot_encoder.pkl', 'rb') as file:
    onehot = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# ---------------- PREPROCESSING ----------------

# Encode Sex
data['Sex'] = label.transform(data['Sex'])

# OneHot Encode Embarked
embarked_encoded = onehot.transform(data[['Embarked']])

embarked_encoded = pd.DataFrame(
    embarked_encoded,
    columns=onehot.get_feature_names_out(['Embarked'])
)

# Merge dataframe correctly
data = pd.concat(
    [data.drop(columns=['Embarked']), embarked_encoded],
    axis=1
)

# Scale numerical columns
data[['Pclass','SibSp','Parch','Fare']] = scaler.transform(
    data[['Pclass','SibSp','Parch','Fare']]
)

# ---------------- PREDICTION ----------------
if st.button('Predict Survival'):

    y = model.predict(data)
    y = y[0][0]

    if y > 0.5:
        result = "✅ Passenger WILL survive"
    else:
        result = "❌ Passenger will NOT survive"

    st.write("Survival Probability:", float(y))
    st.success(result)