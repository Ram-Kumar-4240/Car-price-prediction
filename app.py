import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from streamlit_option_menu import option_menu

# Load your data and models here
X_train_1 = pd.read_csv('Dataset\\X_train_1.csv') 

OneHotEncoding_columns = ['ft', 'bt', 'oem', 'Insurance Validity', 'Transmission', 'Gear Box', 'city']
label_encoding_columns = ['Engine', 'No of Cylinder', 'Seating Capacity', 'modelYear', 'ownerNo']
numerical_columns = ['Mileage', 'Torque', 'Length', 'Width', 'Height', 'Wheel Base', 'Kerb Weight', 'Max Power']

# Load objects function
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

# Set page config
st.set_page_config(page_title="Car Price Predictor", layout="wide")

# Custom CSS
st.markdown("""
<style>
    .stApp { font-family: 'Poppins', sans-serif; }
    .stApp {
        background: linear-gradient(to right, #f6f9fc, #e9f1f7);
    }
    .card {
        background-color: #ffffff;
        border-radius: 15px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        padding: 20px;
        margin-bottom: 20px;
    }
    .big-font {
        font-size:30px !important;
        font-weight: bold;
    }
    .medium-font {
        font-size:20px !important;
    }
    .predict-btn {
        background-color: #4CAF50;
        color: white;
        font-size: 18px;
        padding: 10px 20px;
        border-radius: 10px;
        transition: background-color 0.3s ease;
    }
    .predict-btn:hover {
        background-color: #45a049;
    }
</style>
""", unsafe_allow_html=True)


# Sidebar navigation
with st.sidebar:
    selected = option_menu("Main Menu", ["Home", "Predict", "Insights"], 
        icons=['house', 'graph-up', 'lightbulb'], menu_icon="cast", default_index=0)

# Home Page
if selected == "Home":
    st.markdown("<h1 style='text-align: center;'>Discover the Value of Your Dream Car</h1>", unsafe_allow_html=True)
    
    st.image("images\\ford-mustang-429.jpg", use_column_width=True)
    
    st.markdown("<p class='medium-font'>Our advanced machine learning model helps you estimate car prices based on various features. Whether you're buying or selling, get accurate predictions in seconds!</p>", unsafe_allow_html=True)
    
    st.markdown("### How it works:")
    st.write("1. Navigate to the 'Predict' page")
    st.write("2. Input your car's details")
    st.write("3. Click 'Predict' to get an estimated price")
    st.write("4. Explore 'Insights' to understand market trends")

# Predict Page
elif selected == "Predict":
    st.markdown("<h1 style='text-align: center;'>Car Price Prediction</h1>", unsafe_allow_html=True)
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        fuel_type = st.selectbox("Fuel Type", sorted(X_train_1['ft'].unique()))
        torque = st.number_input("Torque (Nm)", min_value=50, max_value=1000, value=250)
        body_type = st.selectbox("Body Type", X_train_1['bt'].unique())
        transmission = st.selectbox("Transmission", sorted(X_train_1['Transmission'].unique()))

    with col2:
        oem = st.selectbox("Brand", sorted(X_train_1['oem'].unique())) 
        city = st.selectbox("City", sorted(X_train_1['city'].unique()))
        model_year = st.selectbox("Model Year", sorted(X_train_1['modelYear'].unique())) 
        insurance_validity = st.selectbox("Insurance Validity", sorted(X_train_1['Insurance Validity'].unique()))

    with col3:
        mileage = st.number_input("Mileage (km/l)", min_value=6, max_value=36, value=15)
        engine = st.selectbox("Engine Capacity (cc)", sorted(X_train_1['Engine'].unique()))
        no_of_cylinders = st.selectbox("Number of Cylinders", sorted(X_train_1['No of Cylinder'].unique()))
        seating_capacity = st.selectbox("Seating Capacity", sorted(X_train_1['Seating Capacity'].unique()))

    with col4:
        max_power = st.number_input("Max Power (bhp)", min_value=0, max_value=1000, value=100)
        wheel_base = st.number_input("Wheel Base (mm)", min_value=1000, max_value=4000, value=2700)
        gear_box = st.selectbox("Gear Box", sorted(X_train_1['Gear Box'].unique()))
        ownerNo = st.selectbox("ownerNo", sorted(X_train_1['ownerNo'].unique()))

    with col5:
        length = st.number_input("Length (mm)", min_value=1000, max_value=6000, value=4000)
        width = st.number_input("Width (mm)", min_value=1000, max_value=3000, value=1800)
        height = st.number_input("Height (mm)", min_value=1000, max_value=2500, value=1500)
        kerb_weight = st.number_input("Kerb Weight (kg)", min_value=500, max_value=4000, value=1500)
    
    # Create input dataframe
    input_data = {
        'ft': fuel_type,
        'bt': body_type,
        'oem': oem,
        'ownerNo': ownerNo,
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

    # One-hot encoding
    input_encoded = pd.DataFrame(onehot_encoder.transform(input_df[OneHotEncoding_columns]), 
                                columns=onehot_encoder.get_feature_names_out(OneHotEncoding_columns))

    # Label encoding
    for col in label_encoding_columns:
        try:
            input_values = input_df[col].unique()
            known_classes = label_encoders[col].classes_
            for val in input_values:
                if val not in known_classes:
                    input_df[col].replace(val, known_classes[0], inplace=True)
            input_df[col] = label_encoders[col].transform(input_df[col])
        except Exception as e:
            st.error(f"Error in encoding column '{col}': {e}")

    # Drop OneHotEncoded columns from input_df and concatenate with encoded data
    input_df = input_df.drop(columns=OneHotEncoding_columns)
    input_df = pd.concat([input_df, input_encoded], axis=1)

    # Scale input data
    input_scaled = scaler.transform(input_df)
    
    # Prediction button and output
    if st.button('Predict', key='predict_button', help="Click to get the predicted price"):
        try:
            model = models
            prediction = model.predict(input_scaled)
            predicted_price = prediction[0]
            st.markdown("### **Predicted Price:**")
            st.markdown(f"<h1 style='text-align: center; color: #4CAF50;'>₹{predicted_price:,.2f}</h1>", unsafe_allow_html=True)
            st.info("💡 Did you know? The color of a car can affect its resale value!")
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")

# Insights Page
elif selected == "Insights":
    st.markdown("<h1 style='text-align: center;'>Market Insights</h1>", unsafe_allow_html=True)
    
    st.subheader("Mileage Distribution")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.histplot(X_train_1['Mileage'], kde=True, ax=ax, color='purple')
    st.pyplot(fig)
    
    st.subheader("Correlation Heatmap")
    correlation_matrix = X_train_1[numerical_columns].corr()
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='Blues', ax=ax)
    st.pyplot(fig)
