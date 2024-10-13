import streamlit as st
import pandas as pd
import pickle

# Load the trained SVR model
with open('Ragi_svr.pkl', 'rb') as file:
    model = pickle.load(file)

# Function to make predictions
def predict_yield(input_data):
    prediction = model.predict(input_data)
    return prediction

# Streamlit App UI
st.title('Ragi Yield Prediction')
st.write('This app predicts the yield of Ragi based on input parameters.')

# Select box for year (data is till 2020)
year = st.selectbox('Select Year', [year for year in range(2000, 2024)])  

# Select box for district (list your available districts)
district = st.selectbox('Select District', ['Almora', 'Bageshwar', 'Chamoli', 'Champawat', 'Dehradun','Nainital','Pauri Garhwal','Pithoragarh','Rudraprayag','Tehri Garhwal','Uttarkashi'])

# Input fields for user to enter data
area = st.number_input('Area (in hectares)', min_value=0.0)
production = st.number_input('Production (in tonnes)', min_value=0.0)
ndvi = st.number_input('NDVI', min_value=0.0, max_value=1.0)
evi = st.number_input('EVI', min_value=0.0, max_value=1.0)
fpar = st.number_input('FPAR', min_value=0.0, max_value=60.0)
gpp = st.number_input('GPP', min_value=0.0, max_value=10.0)
lai = st.number_input('LAI', min_value=0.0, max_value=30.0)
precipitation = st.number_input('Precipitation', min_value=0.0, max_value=60.0)
sm = st.number_input('Soil Moisture', min_value=0.0, max_value=40.0)
lst_day = st.number_input('LST (Day)', min_value=0.0, max_value=40.0)
lst_night = st.number_input('LST (Night)', min_value=0.0, max_value=30.0)



# Prepare input data as a DataFrame
input_data = pd.DataFrame({
    'Area': [area],
    'Production': [production],
    'NDVI': [ndvi],
    'EVI': [evi],
    'FPAR': [fpar],
    'GPP': [gpp],
    'LAI': [lai],
    'Precipitation': [precipitation],
    'Soil Moisture': [sm],
    'LST_Day': [lst_day],
    'LST_Night': [lst_night],
    'Year': [year],
    'District': [district]
})

# Predict yield when the button is clicked
if st.button('Predict Yield'):
    input_data = pd.get_dummies(input_data, columns=['District'], drop_first=True)
    input_data = input_data.reindex(columns=model.feature_names_in_, fill_value=0)
    prediction = predict_yield(input_data)
    st.write(f'Predicted Yield: {prediction[0]:.2f} tonnes/ha')
