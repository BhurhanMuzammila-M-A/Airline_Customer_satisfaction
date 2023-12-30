# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pickle

#load the saved model

loaded_model = pickle.load(open('D:/MUZA Imr/ML/trained_model.sav'))
input_data = (0,0,65,1,265,0,0,0,2,3,3,0,3,5,3,2,1)

# changing the input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = loaded_model.predict(input_data_reshaped)
print(prediction)

if (prediction[0] == 0):
  print('The Customer is not satisfied with the service')
else:
  print('The Customer is satisfied with the service')