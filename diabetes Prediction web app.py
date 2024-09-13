import numpy as np
import pickle
import streamlit as st

# Load the trained model
loaded_model = pickle.load(open('C:\\Users\\Gunjan\\OneDrive\\Desktop\\diabetes prediction miniproject\\mini project sem4\\trained_model.sav', 'rb'))

# Creating function for prediction
def diabetes_prediction(input_data):
    # Changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data, dtype=float)
    # Reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    prediction = loaded_model.predict(input_data_reshaped)
    if prediction[0] == 0:
        return 'The person is not diabetic'
    else:
        return 'The person is diabetic'

def main():
    # Giving title
    st.title('Diabetes Prediction')
    
    # Getting the input data from the user
    Pregnancies = st.text_input('Number of Pregnancies', '')
    Glucose = st.text_input('Glucose Level', '')
    BloodPressure = st.text_input('Blood Pressure value', '')
    SkinThickness = st.text_input('Skin Thickness Value', '')
    Insulin = st.text_input('Insulin value', '')
    BMI = st.text_input('BMI value', '')
    DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value', '')
    Age = st.text_input('Age of the person', '')
    
    # Code for prediction
    diagnosis = ''
    
    # Creating button for prediction
    if st.button('Diabetes Test Result'):
        try:
            input_data = [float(Pregnancies), float(Glucose), float(BloodPressure), float(SkinThickness), 
                          float(Insulin), float(BMI), float(DiabetesPedigreeFunction), float(Age)]
            diagnosis = diabetes_prediction(input_data)
        except ValueError:
            diagnosis = 'Please enter valid numerical values for all fields'
    
    st.success(diagnosis)

if __name__ == '__main__':
    main()
