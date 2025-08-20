import numpy as np
import pickle
import streamlit as st

# loading the saved model
loaded_model = pickle.load(open('trained_model.sav', 'rb'))
loaded_scaler = pickle.load(open('scaler.sav', 'rb'))

# creating a function for prediction
def diabetes_prediction(input_data):
    np_input_data = np.asarray(input_data)

    reshaped_data = np_input_data.reshape(1,-1)

    std_data = loaded_scaler.transform(reshaped_data)

    prediction = loaded_model.predict(std_data)

    if(prediction[0] == 0):
        return 'The person is not diabetic.'
    else:
        return 'The person is diabetic.'

def main():

    # giving a title
    st.title('Diabetes Prediction Web App')

    # getting the input data from the user
    Pregnancies = st.text_input('Number of Pregnancies')
    Glucose = st.text_input('Glucose Level')
    BloodPressure = st.text_input('Blood Pressure')
    SkinThickness = st.text_input('Thickness of Skin')
    Insulin = st.text_input('Insulin Level')
    BMI = st.text_input('BMI Value')
    DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function Value')
    Age = st.text_input('Age of the Person')

    # prediction on user input
    diagnosis = ''

    # creating a 'Predict' button
    if st.button('Predict Diabetes'):
        diagnosis = diabetes_prediction([Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age])
    st.success(diagnosis)

if __name__ == '__main__':
    main()