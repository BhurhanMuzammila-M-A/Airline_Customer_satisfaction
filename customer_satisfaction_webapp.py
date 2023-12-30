# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 19:43:17 2023

@author: admin
"""

import numpy as np
import pickle
import streamlit as st

#loading the saved model
loaded_model = pickle.load(open('D:/MUZA Imr/ML/trained_model.sav','rb'))

#Creating_a_function_for_prediction
def customer_satisfaction(input_data):

    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if (prediction[0] == 0):
      return'The Customer is not satisfied with the service'
    else:
      return'The Customer is satisfied with the service'
      
      
      


def main():
    
    
    # giving a title
    st.title('Airlines Customer Satisfaction Web App')
    st.info("Gender - Female:0, Male:1--\nClass - Business:0, Eco:1, Eco-plus:2---\nType of travel = Business travel:0, personal travel:1,")
    
    Gender = st.text_input('Gender')
    Customer_Type = st.text_input('Customer_Type')
    Age = st.text_input('Age') 
    Class = st.text_input('Class')
    Flight_Distance = st.text_input('Flight_Distance')
    Seat_comfort = st.text_input('Seat_comfort')
    Food_and_drink = st.text_input('Food_and_drink')
    Gate_location = st.text_input('Gate_location')
    Inflight_wifi_service = st.text_input('Inflight_wifi_service')
    Inflight_entertainment = st.text_input('Inflight_Entertainment')
    Online_support = st.text_input('Online_support')
    Ease_of_Online_booking = st.text_input('Ease_of_Online_booking') 
    On_board_service = st.text_input('On-board_service')
    Leg_room_service = st.text_input('Leg_room_service')
    Baggage_handling = st.text_input('Baggage_handling')
    Checkin_service = st.text_input('Checkin_service')
    Cleanliness = st.text_input('Cleanliness')
    Online_boarding = st.text_input('Online_boarding')
    
           
    # code for Prediction
    satisfaction = ''
    
    # creating a button for Prediction
    
    if st.button('Customer Satisfaction Result'):
        satisfaction = customer_satisfaction([Gender,Customer_Type,Age,Class,Flight_Distance,Seat_comfort,Food_and_drink,Gate_location,Inflight_wifi_service,Inflight_entertainment,Online_support,Ease_of_Online_booking,On_board_service,Leg_room_service,Baggage_handling,Checkin_service,Cleanliness,Online_boarding])
        
        
    st.success(satisfaction)
    
    
    
    
    
if __name__ == '__main__':
    main()
    



