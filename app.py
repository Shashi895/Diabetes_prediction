import numpy as np
import pickle
import streamlit as st
# st.title("DiabetesPrediction")
st.set_page_config(page_title="DiabetesPrediction", page_icon=":shark:", layout="centered", initial_sidebar_state="auto", menu_items=None)
st.set_option('deprecation.showfileUploaderEncoding', False)
hide_st_style="""
    <style>
        #MainMenu {visibility:hidden;}
        footer {visibility:hidden;}
    </style>
"""
st.markdown(hide_st_style,unsafe_allow_html=True)
loaded_model=pickle.load(open('logreg.pkl','rb'))


def diabetes_prediction(input_data):
    input_data_array=np.asarray(input_data)

    #Reshape the array as we are predicting for one instance
    input_data_reshaped=input_data_array.reshape(1,-1)

    prediction=loaded_model.predict(input_data_reshaped)
    print(prediction)

    if prediction[0]==0:
        return "The person is not diabetic"
    else:
        return "The person is diabetic"
    

def main():
    st.title("Diabetes Prediction Web App")

    Pregnancies=st.text_input("Number of Pregnancies")
    Glucose=st.text_input("Glucose level")
    BloodPressure=st.text_input("Blood pressure value")
    SkinThickness=st.text_input("Skin thickness value")
    Insulin=st.text_input("Insulin level")
    BMI=st.text_input("BMI value")
    DiabetesPedigreeFunction=st.text_input("Diabetes pedigree function value")
    Age=st.text_input("Age of the person")

    diagnosis=''

    if st.button('Diabetes Test Result'):
        # Check if any field is empty
        if not all([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]):
            st.error("Please fill up all the fields.")
        else:
            diagnosis = diabetes_prediction([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI,
                                             DiabetesPedigreeFunction, Age])
            st.success(diagnosis)


if __name__=='__main__':
    main()
logreg=pickle.load(open('logreg.pkl','rb'))


