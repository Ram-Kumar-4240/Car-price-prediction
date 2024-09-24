import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sqlalchemy import create_engine

# Create SQLAlchemy engine
engine = create_engine('mysql+mysqlconnector://root:@localhost/car_price')

# Fetch unique values from the 'Gear Box', 'city', and 'oem' columns
def get_unique_values(column_name):
    query = f"SELECT DISTINCT `{column_name}` FROM car_details"
    result = pd.read_sql(query, con=engine)  
    return result[column_name].dropna().tolist()  

# Fetch unique values for Gear Box, city, and oem
unique_gear_box = get_unique_values('Gear Box')
unique_city = get_unique_values('city')
unique_oem = get_unique_values('oem')

# Set up page configuration
st.set_page_config(
    page_title="Car Price Prediction",
    page_icon="🚘",
    layout="wide",
)

# Load the pre-trained models
def load_models():
    models = {}
    with open('xgb_model.pkl', 'rb') as f:
        models['XGBRegressor'] = pickle.load(f)
    with open('hgb_model.pkl', 'rb') as f:
        models['HistGradientBoosting'] = pickle.load(f)
    with open('decision_tree_model.pkl', 'rb') as f:
        models['DecisionTreeRegressor'] = pickle.load(f)
    with open('rf_model.pkl', 'rb') as f:
        models['RandomForestRegressor'] = pickle.load(f)
    with open('linear_regression_model.pkl', 'rb') as f:
        models['LinearRegression'] = pickle.load(f)
    with open('svm_model.pkl', 'rb') as f:
        models['SVR'] = pickle.load(f)
    return models

# Load pre-processing objects
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('label_encoder.pkl', 'rb') as f:
    le = pickle.load(f)
with open('imputer.pkl', 'rb') as f:
    imputer = pickle.load(f)

# Dictionary for storing loaded models
models = load_models()

# Function to navigate between pages
def navigate_page():
    if "selected_model" in st.session_state:
        prediction_page()
    else:
        home_page()

# Home page function
# Home page function
def home_page():
    st.markdown("<h1 style='text-align: center;'>🚘 Car Price Prediction</h1>", unsafe_allow_html=True)
    

    st.markdown("## Select a model to proceed:")
    
    # Display model accuracies
    st.markdown("### Model Accuracies:")
    st.markdown("""
    - **XGBRegressor:** R²: **93.60%**
    - **HistGradientBoostingRegressor:** R²: **91.14%**
    - **DecisionTreeRegressor:** R²: **71.67%**
    - **RandomForestRegressor:** R²: **93.41%**
    - **LinearRegression:** R²: **64.99%**
    - **Support Vector Regressor:** R²: **-7.74%**
    """)
    
    # Model selection
    selected_model = st.selectbox(
        "Choose a Model",
        ["XGBRegressor", "HistGradientBoosting", "DecisionTreeRegressor", "RandomForestRegressor", "LinearRegression", "SVR"]
    )
    
    if st.button("Proceed"):
        # Store the selected model in session_state
        st.session_state["selected_model"] = selected_model


# Fetch all car details and store in a DataFrame
def fetch_all_car_details():
    query = "SELECT oem, km, city, modelYear, price, car_links FROM car_details"  # Include car_links
    car_details_df = pd.read_sql(query, con=engine)
    return car_details_df

# Prediction page function
def prediction_page():
    st.markdown(f"<h1 style='text-align: center;'>🚘 Car Price Prediction with {st.session_state['selected_model']}</h1>", unsafe_allow_html=True)

    # Create input sections with columns
    st.markdown("### Enter Car Details Below:")
    
    col1, col2, col3, col4, col5, col6 = st.columns(6)

    with col1:
        oem = st.selectbox("Brand", unique_oem)    

    with col2:
        fuel_type = st.selectbox("Fuel Type", ['Petrol', 'Diesel', 'CNG', 'Electric', 'Hybrid'])

    with col3:
        km = st.number_input("Kilometers Driven", min_value=1000, max_value=100000, value=12000)

    with col4:
        owner_no = st.number_input("Number of Owners", min_value=0, max_value=5, value=1)

    with col5:
        city = st.selectbox("City", unique_city)

    with col6:
        transmission = st.selectbox("Transmission", ['Manual', 'Automatic'])

    # More input columns
    col7, col8, col9, col10, col11, col12 = st.columns(6)
    
    with col7:
        model_year = st.number_input("Model Year", min_value=2000, max_value=2023, value=2018)
    
    with col8:
        engine = st.number_input("Engine Capacity (cc)", min_value=500, max_value=5000, value=1500)
    
    with col9:
        mileage = st.number_input("Mileage (km/l)", min_value=6, max_value=36, value=15)

    with col10:
        gear_box = st.selectbox("Gear Box", unique_gear_box)    

    with col11:
        max_power = st.number_input("Max Power (bhp)", min_value=50, max_value=500, value=100)
    
    with col12:
        torque = st.number_input("Torque (Nm)", min_value=50, max_value=1000, value=200)
    
    # Dimensions input
    col13, col14, col15 = st.columns(3)
    
    with col13:
        length = st.number_input("Length (mm)", min_value=3000, max_value=6000, value=4000)
    
    with col14:
        width = st.number_input("Width (mm)", min_value=1200, max_value=2500, value=1800)

    with col15:
        height = st.number_input("Height (mm)", min_value=1200, max_value=2500, value=1500)

    # Collect input data into a dictionary
    input_data = {
        'fuel': fuel_type,
        'oem': oem,
        'city': city,
        'transmission': transmission,
        'Gear Box': gear_box,
        'km': km,
        'ownerNo': owner_no,
        'modelYear': model_year,
        'Engine': engine,
        'Mileage': mileage,
        'Max Power': max_power,
        'Torque': torque,
        'Length': length,
        'Width': width,
        'Height': height
    }

    # Convert input data to DataFrame
    input_df = pd.DataFrame([input_data])

    # Encode categorical features
    for column in ['fuel', 'oem', 'city', 'transmission', 'Gear Box']:
        if input_df[column].iloc[0] in le.classes_:
            input_df[column] = le.transform(input_df[column])
        else:
            input_df[column] = 0

    # Scale the input features
    try:
        input_scaled = scaler.transform(input_df)
    except ValueError as e:
        st.error(f"Error in scaling features: {e}")
        st.stop()

    # Prediction button
    st.markdown("##")
    if st.button('🚀 Predict'):
        selected_model = st.session_state["selected_model"]
        model = models[selected_model]
        
        try:
            prediction = model.predict(input_scaled)
            predicted_price = prediction[0]
            st.markdown("### **Predicted Price:**")
            st.markdown(f"<h1 style='text-align: center; color: green;'>₹{predicted_price:,.2f}</h1>", unsafe_allow_html=True)
            
            # Calculate price range
            lower_bound = max(0, predicted_price - 100000)  # Ensure it doesn't go below 0
            upper_bound = predicted_price + 100000
            
            # Fetch all car details
            car_details_df = fetch_all_car_details()
            
            # Filter car details based on the predicted price range
            similar_cars = car_details_df[(car_details_df['price'] >= lower_bound) & (car_details_df['price'] <= upper_bound)]

            # Display fetched car details if any exist
            if not similar_cars.empty:
                st.markdown("### **Similar Car Details:**")
                
                # Customize the displayed columns
                similar_cars = similar_cars[['oem', 'km', 'city', 'modelYear', 'price', 'car_links']]  # Include car_links
                
                # Add headers for the columns
                header_col1, header_col2, header_col3, header_col4, header_col5, header_col6 = st.columns(6)
                header_col1.markdown("**Brand (OEM)**")
                header_col2.markdown("**Kilometers Driven**")
                header_col3.markdown("**City**")
                header_col4.markdown("**Model Year**")
                header_col5.markdown("**Price**")
                header_col6.markdown("**Car Link**")  # Header for the car link

                # Display data in the same order as headers
                for i in range(len(similar_cars)):
                    row = similar_cars.iloc[i]
                    col1, col2, col3, col4, col5, col6 = st.columns(6)
                    
                    # Display row data
                    col1.text(row['oem'])
                    col2.text(row['km'])
                    col3.text(row['city'])
                    col4.text(row['modelYear'])
                    col5.text(f"₹{row['price']:,.2f}")
                    col6.markdown(f"[View Car]({row['car_links']})", unsafe_allow_html=True)  # Create clickable link

            else:
                st.markdown("### No similar cars found in this price range.")

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")

# Call the navigation function
navigate_page()
