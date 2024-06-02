import streamlit as st
import pandas as pd
import numpy as np
import joblib             

from sklearn.ensemble import ExtraTreesClassifier
from model import ordinal_encoder,get_prediction



st.set_page_config(page_title ="Accident Severity Prediction App", page_icon="🚧", layout="wide")

st.markdown(
    """
    <style>
    [data-testid= "stAppViewContainer"] {
        
        background-image: url('https://png.pngtree.com/png-clipart/20230803/original/pngtree-a-car-accident-near-the-yellow-signage-transport-auto-direction-vector-picture-image_9469682.png');
        background-size: cover; 
        
    }
    </style>
    """,
    unsafe_allow_html=True
)


options_day = ['Sunday', "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]

options_age = ['18-30', '31-50', 'Over 51', 'Unknown', 'Under 18']
# options_acc_area = ['Other', 'Office areas', 'Residential areas', ' Church areas',
#       ' Industrial areas', 'School areas', '  Recreational areas',
#       ' Outside rural areas', ' Hospital areas', '  Market areas',
#       'Rural village areas', 'Unknown', 'Rural village areasOffice areas',
#       'Recreational areas']

# options_cause = ['No distancing', 'Changing lane to the right',
#       'Changing lane to the left', 'Driving carelessly',
#       'No priority to vehicle', 'Moving Backward',
#       'No priority to pedestrian', 'Other', 'Overtaking',
#       'Driving under the influence of drugs', 'Driving to the left',
#       'Getting off the vehicle improperly', 'Driving at high speed',
#       'Overturning', 'Turnover', 'Overspeed', 'Overloading', 'Drunk driving',
#       'Unknown', 'Improper parking']
# options_vehicle_type = ['Automobile', 'Lorry (41-100Q)', 'Other', 'Pick up upto 10Q',
#       'Public (12 seats)', 'Stationwagen', 'Lorry (11-40Q)',
#       'Public (13-45 seats)', 'Public (> 45 seats)', 'Long lorry', 'Taxi',
#       'Motorcycle', 'Special vehicle', 'Ridden horse', 'Turbo', 'Bajaj', 'Bicycle']
options_driver_exp = ['5-10yr', '2-5yr', 'Above 10yr', '1-2yr', 'Below 1yr', 'No Licence', 'unknown']
# options_lanes = ['Two-way (divided with broken lines road marking)', 'Undivided Two way',
#       'other', 'Double carriageway (median)', 'One way',
#       'Two-way (divided with solid lines road marking)', 'Unknown']
options_road_surface_conditions = ['Dry','Wet or damp','Snow','Flood over 3cm. deep']
options_junction_type = ['Y Shape', 'No junction', 'Crossing', 'Other', 'Unknown', 'O Shape', 'T Shape', 'X Shape']
options_light_condition = ['Daylight', 'Darkness lights - lit', 'Darkness - no lighting', 'Darkness - lights unlit']

features = ['hour','day_of_week','casualties','vehicles_involved','driver_age','driving_experience','road_surface_conditions','junction_type','light_condition','minute']

st.markdown("<h1 style='text-align: center;'>Accident Severity Prediction App 🚧</h1>", unsafe_allow_html=True)


def main():
    with st.form('prediction form'):
        st.subheader('Enter the input for the following features:')
        hour = st.slider("Pickup hour: ",0,23,value=0, format="%d")
        day_of_week = st.selectbox("Select Day of week:", options =options_day)
        casualties = st.slider("Casualties: ", 1, 8, value=0, format="%d")
#       accident_cause = st.selectbox("Select Accident Cause: ", options=options_cause)
        vehicles_involved = st.slider("Vehicles involved: ", 1, 7, value=0, format="%d")
#       vehicle_type = st.selectbox("Select Vehicle Type: ", options=options_vehicle_type)
        driver_age = st.selectbox("Select Driver Age: ", options=options_age)
#       accident_area = st.selectbox("Select Accident Area: ", options=options_acc_area)
        driving_experience = st.selectbox("Select Driving Experience: ", options=options_driver_exp)
#       lanes = st.selectbox("Select Lanes: ", options=options_lanes)
        minute = st.slider("Pickup minute:",0,59,value=0,format='%d')
        road_surface_conditions = st.selectbox("Selct Road Surface Conditions: ", options= options_road_surface_conditions)
        junction_type = st.selectbox("Selet Junction Type: ", options = options_junction_type)
        light_condition = st.selectbox("Select Light Condition: ", options= options_light_condition)

        submit = st.form_submit_button('Predict')
        
    if submit:
        day_of_week = ordinal_encoder(day_of_week,options_day)
#       accident_cause= ordinal_encoder(accident_cause,options_cause)
#       vehicle_type = ordinal_encoder(vehicle_type,options_vehicle_type)
        driver_age = ordinal_encoder(driver_age,options_age)
#       accident_area = ordinal_encoder(accident_area,options_acc_area)
        driving_experience = ordinal_encoder(driving_experience,options_driver_exp)
#       lanes = ordinal_encoder(lanes,options_lanes)
        road_surface_conditions = ordinal_encoder(road_surface_conditions, options_road_surface_conditions)
        junction_type = ordinal_encoder(junction_type, options_junction_type)
        light_condition = ordinal_encoder(light_condition, options_light_condition)
        
        data  = np.array([hour,day_of_week,casualties,vehicles_involved, 
                            driver_age,driving_experience,minute,road_surface_conditions,junction_type,light_condition]).reshape(1,-1)
        
        
        pred  = get_prediction(data =data)
        
#       st.write(f"The predicted severity is :{pred[0]}")
        if pred == 0:
            st.write(f"The severity prediction is Fatal Injury⚠")
        elif pred == 1:
            st.write(f"The severity prediction is serious injury")
        else:
            st.write(f"The severity prediction is slight injury")
        st.balloons()
        
if __name__ == '__main__':
    main()
        
        