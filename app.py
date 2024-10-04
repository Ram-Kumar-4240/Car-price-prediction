import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sqlalchemy import create_engine

X_train_1 = pd.read_csv('Dataset\\X_train_1.csv') 

OneHotEncoding_columns = ['ft', 'bt', 'oem', 'Insurance Validity', 'Transmission', 'Gear Box', 'city']
label_encoding_columns = ['Engine', 'No of Cylinder', 'Seating Capacity', 'modelYear']
numerical_columns = ['Mileage','Torque','Length','Width','Height','Wheel Base','Kerb Weight','Max Power']

def load_objects():
    models = {}
    with open('models\\model_xgb.pkl', 'rb') as f:
        models = pickle.load(f)
    with open('models\\scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('models\\onehot_encoder.pkl', 'rb') as f:
        onehot_encoder = pickle.load(f)
    # Load label encoders for each specific column
    label_encoders = {}
    for col in label_encoding_columns:
        with open(f'models\\label_encoder_{col}.pkl', 'rb') as f:
            label_encoders[col] = pickle.load(f)
    return models, scaler, onehot_encoder, label_encoders

models, scaler, onehot_encoder, label_encoders = load_objects()

st.set_page_config(
    page_title="Car Price Prediction",
    page_icon="🚘",
    layout="wide",
)

st.markdown("<h1 style='text-align: center;'>🚘 Car Price Prediction</h1>", unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)


sorted_seating_capacity = sorted(X_train_1['Seating Capacity'].unique())


OneHotEncoding_columns = ['ft', 'bt', 'oem', 'Insurance Validity', 'Transmission', 'Gear Box', 'city']
label_encoding_columns = ['Engine', 'No of Cylinder', 'Seating Capacity', 'modelYear']
numerical_columns = ['Mileage','Torque','Length','Width','Height','Wheel Base','Kerb Weight','Max Power']

with col1:
    fuel_type = st.selectbox("Fuel Type", sorted(X_train_1['ft'].unique()))
    torque = st.number_input("Torque (Nm)", min_value=50, max_value=1000, value=250)
    length = st.number_input("Length (mm)", min_value=1000, max_value=6000, value=4000)
    body_type = st.selectbox("Body Type", X_train_1['bt'].unique())


with col2:
    oem = st.selectbox("Brand", sorted(X_train_1['oem'].unique())) 
    model_year = st.selectbox("Model Year", sorted(X_train_1['modelYear'].unique())) 
    insurance_validity = st.selectbox("Insurance Validity", sorted(X_train_1['Insurance Validity'].unique()))
    no_of_cylinders = st.selectbox("Number of Cylinders", sorted(X_train_1['No of Cylinder'].unique()))
    width = st.number_input("Width (mm)", min_value=1000, max_value=3000, value=1800)

with col3:
    transmission = st.selectbox("Transmission", sorted(X_train_1['Transmission'].unique()))
    mileage = st.number_input("Mileage (km/l)", min_value=6, max_value=36, value=15)
    engine = st.selectbox("Engine Capacity (cc)", sorted(X_train_1['Engine'].unique()))
    height = st.number_input("Height (mm)", min_value=1000, max_value=2500, value=1500)
    seating_capacity = st.selectbox("Seating Capacity", sorted_seating_capacity)
with col4:
    city = st.selectbox("City", sorted(X_train_1['city'].unique()))
    max_power = st.number_input("Max Power (bhp)", min_value=0, max_value=1000, value=100)
    wheel_base = st.number_input("Wheel Base (mm)", min_value=1000, max_value=4000, value=2700)
    kerb_weight = st.number_input("Kerb Weight (kg)", min_value=500, max_value=4000, value=1500)
    gear_box = st.selectbox("Gear Box", sorted(X_train_1['Gear Box'].unique()))


input_data = {
    'ft': fuel_type,
    'bt': body_type,
    'oem': oem,
    'modelYear': model_year,
    'Insurance Validity': insurance_validity,
    'Transmission': transmission,
    'Mileage': mileage,
    'Engine': engine,
    'Torque': torque,
    'No of Cylinder': no_of_cylinders,
    'Length': length,
    'Width': width,
    'Height': height,
    'Wheel Base': wheel_base,
    'Kerb Weight': kerb_weight,
    'Gear Box': gear_box,
    'Seating Capacity': seating_capacity,
    'city': city,
    'Max Power': max_power
}

input_df = pd.DataFrame([input_data])

input_encoded = pd.DataFrame(onehot_encoder.transform(input_df[OneHotEncoding_columns]), 
                             columns=onehot_encoder.get_feature_names_out(OneHotEncoding_columns))

for col in label_encoding_columns:
    try:
        if not hasattr(label_encoders[col], 'classes_'):
            raise ValueError(f"The encoder for column '{col}' is not a valid LabelEncoder.")
        input_values = input_df[col].unique()
        
        known_classes = label_encoders[col].classes_
        
        for val in input_values:
            if val not in known_classes:
                # st.warning(f"Value '{val}' in column '{col}' is unseen. Mapping to default.")
                input_df[col].replace(val, known_classes[0], inplace=True)
        
        input_df[col] = label_encoders[col].transform(input_df[col])
    
    except Exception as e:
        st.error(f"Error in encoding column '{col}': {e}")


input_df = input_df.drop(columns=OneHotEncoding_columns)

input_df = pd.concat([input_df, input_encoded], axis=1)

input_scaled = scaler.transform(input_df)


if st.button('🚀 Predict'):
    try:
        model = models
        prediction = model.predict(input_scaled)
        predicted_price = prediction[0]
        st.markdown("### **Predicted Price:**")
        st.markdown(f"<h1 style='text-align: center; color: green;'>₹{predicted_price:,.2f}</h1>", unsafe_allow_html=True)
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
