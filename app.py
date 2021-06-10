# loading in the model to predict on the data

import pickle
import streamlit as st

model = pickle.load(open('model.pkl', 'rb'))

def welcome():
    return 'welcome all'


# defining the function which will make the prediction using
# the data which the user inputs
def prediction(Contract, MonthlyCharges, PaperlessBilling,SeniorCitizen):
    predictions = model.predict(
        [[Contract, MonthlyCharges, PaperlessBilling,SeniorCitizen]])
    print(predictions)
    return predictions


# this is the main function in which we define our webpage
def main():
    # giving the webpage a title
    st.title("Customer Churn Prediction")

    # here we define some of the front end elements of the web page like
    # the font and background color, the padding and the text to be displayed
    html_temp = """
    <div style ="background-color:blue;padding:13px">
    <h1 style ="color:black;text-align:center;">Streamlit Churn Classifier ML App </h1>
    </div>
    """
    # this line allows us to display the front end aspects we have
    # defined in the above code
    st.markdown(html_temp, unsafe_allow_html=True)

    # the following lines create text boxes in which the user can enter
    # the data required to make the prediction

    Contract= st.text_input("Contract", " ")
    MonthlyCharges= st.text_input("MonthlyCharges", " ")
    PaperlessBilling = st.text_input("PaperlessBilling", " ")
    SeniorCitizen  = st.text_input("SeniorCitizen", " ")

    result = ""

    # the below line ensures that when the button called 'Predict' is clicked,
    # the prediction function defined above is called to make the prediction
    # and store it in the variable result
    if st.button("Predict"):
        result = predictions(Contract, MonthlyCharges, PaperlessBilling,SeniorCitizen)
    st.success('The output is {}'.format(result))