import streamlit as st
import pandas as pd
import os
import joblib
import numpy as np
base= os.getcwd()

Models_DF = pd.DataFrame({"Models":["Linear Regression","Huber Linear Model",
"Decision Tree", "Random Forest", "Support Vector Regressor",
"Xtreme Gradient Boosting"],
"Keys":["LM", "HLM", "DTR", "RF", "SVR", "XGB"]})

st.markdown('''<meta charset="UTF-8"><center><h1>&#127951; Cricket Predictor &#127951;</h1></center>''', unsafe_allow_html=True)

st.markdown("<HR><h3>Select the thing you want to Predict :-</h3>", unsafe_allow_html=True)

Runs_radio = st.radio("", ("Runs", "Wickets"))

st.markdown("<HR><h4>Select your Choice of model to use for prediction :-</h4>", unsafe_allow_html=True)

model_selection = st.selectbox("", options=Models_DF["Models"].values)
#st.dataframe(Models_DF)
for i in  Models_DF["Models"].index:
    if Models_DF["Models"].iloc[i] == model_selection:
        model_suffix = Models_DF["Keys"].iloc[i]

if Runs_radio =="Wickets":

    model = joblib.load(os.path.join(base, "Models", "Wickets", "Wickets_"+model_suffix+".pkl"))
    st.markdown("<HR><h3>Inputs for predicting the total wickets he/she will take :-</h4>", unsafe_allow_html=True)
    
    
    st.markdown("<h4><br>Enter the number of overs :- </h4>", unsafe_allow_html=True)
    W_O = st.number_input("", min_value=0.0, value=0.0)

    st.markdown("<h4><br>Enter Runs lost by Bowler :- </h4>", unsafe_allow_html=True)
    W_R = st.number_input("", min_value=0, step=1, value=0, key = "W.R")

    st.markdown("<h4><br>Enter the number of Matches played by bowler :- </h4>", unsafe_allow_html=True)
    W_M = st.number_input("", min_value=0, step=1, value=0, key="W_M")

    st.markdown("<h4><br>Enter the number of 4W taken by Bowlers :- </h4>", unsafe_allow_html=True)
    W_4W = st.number_input("", min_value=0, step=1, value=0, key="W_4W")

    st.markdown("<h4><br>Enter the Year :- </h4>", unsafe_allow_html=True)
    W_Y = st.number_input("", min_value=1970, step=1, key="R_Y")
    dat_for_Pred = [[W_O, W_R, W_M, W_4W]]

if Runs_radio == "Runs":

    model = joblib.load(os.path.join(base, "Models", "Runs", "Runs_"+model_suffix+".pkl"))
    st.markdown("<HR><h3>Inputs for predicting the total Runs the Player will score :-</h4>", unsafe_allow_html=True)
    
    st.markdown("<h4><br>Enter the previous year :- </h4>", unsafe_allow_html=True)
    R_Y = st.number_input("", min_value=1970, step=1)

    st.markdown("<h4><br>Enter Innings played by  the Player :- </h4>", unsafe_allow_html=True)
    R_I = st.number_input("", min_value=0, step=1, value=0, key = "R.I")

    st.markdown("<h4><br>Enter the number of Matches played by Player :- </h4>", unsafe_allow_html=True)
    R_M = st.number_input("", min_value=0, step=1, value=0, key="R_M")

    
    st.markdown("<h4><br>Enter the number of 50's scored by Player :- </h4>", unsafe_allow_html=True)
    R_50S = st.number_input("", min_value=0, step=1, value=0, key="R_50S")

    st.markdown("<h4><br>Enter the Highest runs scored by Player :- </h4>", unsafe_allow_html=True)
    R_HS = st.number_input("", min_value=0, step=1, value=0, key="R_HS")

    st.markdown("<h4><br>Enter the number of 100's scored by Player :- </h4>", unsafe_allow_html=True)
    R_100S = st.number_input("", min_value=0, step=1, value=0, key="R_100S")

    st.markdown("<h4><br>Enter the average runs scored by Player :- </h4>", unsafe_allow_html=True)
    R_AS = st.number_input("", min_value=0, step=1, value=0, key="R_AS")

    st.markdown("<h4><br>Enter the number of times the Player was catch out:- </h4>", unsafe_allow_html=True)
    R_CO = st.number_input("", min_value=0, step=1, value=0, key="R_CO")

    R_SO_YN = st.radio("Does the player was on field while He/She scored his/her High Score", ("Yes", "No"))

    if R_SO_YN =="Yes":
        R_SO_Y = 1
        R_SO_N = 0

    else:
        R_SO_Y = 0
        R_SO_N = 1

    if model_suffix == "XGB":
        # Fix for XGB model
        dat_for_pred = [[R_50S, R_M, R_I, R_Y, R_SO_Y, R_HS, R_100S, R_AS, R_CO, R_SO_N]]
    else:
        dat_for_pred = [[R_Y, R_I, R_M, R_50S, R_HS, R_100S, R_AS, R_CO, R_SO_Y, R_SO_N]]


Predict_Btn = st.button("Click this Button for predicting the value")
    
if Predict_Btn:
    st.balloons()
    pred = int(np.round(model.predict(dat_for_Pred))[0])
    if Runs_radio=="Runs":
        st.markdown(f"<h4><br>The Player Will Score around:- {pred} Runs</h4>", unsafe_allow_html=True)
    else:
        st.markdown(f"<h4><br>The Player Will Take around:- {pred} Wickets</h4>", unsafe_allow_html=True)
    
    st.success('Hurray you have predicted the results')


st.markdown("<br><hr style=\"height:10px\">", unsafe_allow_html=True)
st.header("Thank you for using my ML model I hope you liked it")
st.markdown(":smile: :smile: :smile: :smile: :smile:")
st.markdown("Made with :heart: By Harshit Pratap Singh \n")
st.markdown(" :email: 4harshitsingh@gmail.com :iphone: - 8840192004")