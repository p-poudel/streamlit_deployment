import numpy as np
import pickle

loaded_model = pickle.load(open('trained_model.sav', 'rb'))
loaded_scaler = pickle.load(open('scaler.sav', 'rb'))

input_data = (4,110,92,0,0,37.6,0.191,30)

np_input_data = np.asarray(input_data)

reshaped_data = np_input_data.reshape(1,-1)

std_data = loaded_scaler.transform(reshaped_data)
print(std_data)

prediction = loaded_model.predict(std_data)
print(prediction)