import joblib
import streamlit as st
import pickle
import pandas as pd

# Load the trained model
model = pickle.load(open('decision_itree_model.pkl', 'rb'))

st.title('Heating and Cooling Load Prediction')
st.sidebar.header('Building Data')

# FUNCTION to get user input
def user_input():
    relative_compactness = st.sidebar.slider('Relative Compactness', 0.62, 0.98, 0.75, 0.01)
    surface_area = st.sidebar.slider('Surface Area', 514.5, 808.5, 671.5, 1.0)
    wall_area = st.sidebar.slider('Wall Area', 245.0, 416.5, 330.0, 1.0)
    roof_area = st.sidebar.slider('Roof Area', 110.25, 220.5, 192.5, 1.0)
    overall_height = st.sidebar.slider('Overall Height', 3.5, 7.0, 5.25, 0.25)
    orientation = st.sidebar.slider('Orientation', 2, 5, 3, 1)
    glazing_area = st.sidebar.slider('Glazing Area', 0.0, 0.4, 0.2, 0.1)
    glazing_area_distribution = st.sidebar.slider('Glazing Area Distribution', 1, 5, 3, 1)

    data = {
        'Relative Compactness': relative_compactness,
        'Surface Area': surface_area,
        'Wall Area': wall_area,
        'Roof Area': roof_area,
        'Overall Height': overall_height,
        'Orientation': orientation,
        'Glazing Area': glazing_area,
        'Glazing Area Distribution': glazing_area_distribution
    }

    features = pd.DataFrame(data, index=[0])
    return features

user_data = user_input()

# Display user input
st.header('Building Data')
st.write(user_data)

# Predict heating and cooling load
heating_load = model.predict(user_data)
cooling_load = model.predict(user_data)

# Display predictions
st.subheader('Heating Load Prediction')
st.write('Predicted Heating Load:', heating_load[0])

st.subheader('Cooling Load Prediction')
st.write('Predicted Cooling Load:', cooling_load[0])
