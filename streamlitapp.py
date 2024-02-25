import streamlit as st
import pandas as pd
import pickle
import numpy as np
from datetime import datetime

#Using cache_data method to load the model and csv file only once.
@st.cache_data
def load_data():
    data = pd.read_csv("data.csv")
    return data

@st.cache_data
def load_model():
    with open("model.pkl","rb") as file:
        model = pickle.load(file)
    return model
#Running load functions

model = load_model()
data = load_data()

#Setting up title
st.title("Flight Price Predictor")

#Getting user input, note that dropdownlist values are coming from our csv file.
origin = st.selectbox(options = data["from"].unique(),label = "FROM")
destination = st.selectbox(options = data["to"].unique(),label = "TO")

#Now we filter carriers for origin and destination input, 
#which is necessary otherwise we would be able to select some carriers that does not operate on the desired parkour and predicted price would be no better than a random value.
filtered_carriers = data[(data["from"] == origin) & (data["to"] == destination)]["carrier"].unique()

#After filtering, we select the carrier and also the date.
carrier = st.selectbox(options = filtered_carriers,label = "CARRIER")
#We specify our date range with the daterange that the model was fit to. Not necessary but will increase the accuracy of the model for the user eye.
start_date = datetime(2024, 3, 15)
end_date = datetime(2024, 12, 25)
date = st.date_input("Travel Date", min_value = start_date, max_value = end_date, value = datetime(2024, 7, 7))

#We extract necessary parameters from user date input.
day_of_week = date.weekday()
month_number = date.month
day_of_month = int(date.strftime("%d"))
day_of_year = int(date.strftime('%j'))

#Now we try and filter the possible total stop selection, to somewhat validate user input to increase accuracy of the calculations. 
#Otherwise the model would have predicted something that does not exist in a realworl scenario.
filtered_stops = data[(data["from"] == origin) & (data["to"] == destination) & (data["carrier"] == carrier)]["total_stops"].unique()
#We take totalstop input.
total_stops = st.selectbox(options = filtered_stops,label = "How Many Stops for Roundtrip? (Total)")

#Once again we get the dropdownlist options from our csv file, these could have been manually hardcoded, but this method will help if we decide to add more routes to the model.
tod_ob = data["time_of_day_outbound"].unique()
#Taking time of day outbound input, remember we have dropped time of day inbound parameter from the model as it had nearly 0 corelation with price.
time_of_day_ob = st.selectbox(options=tod_ob,label = "Outbound Flight Time Range")

#Now this is where the trick happens, users will likely have no idea how long will the flight take because of that having it as a user input is only going to make the web-page more crowded and return potentially weird predictions. 
#Therefore we use our data's mean value for the specific parkour and carrier pair in combination with total_stops input from user.
ob_duration_minutes  = round(data[(data["from"] == origin) & (data["to"] == destination) & 
                            (data["carrier"] == carrier) & (data["total_stops"] == total_stops)]["outbound_duration_minutes"].mean())
ib_duration_minutes  = round(data[(data["from"] == origin) & (data["to"] == destination) & 
                            (data["carrier"] == carrier) & (data["total_stops"] == total_stops)]["inbound_duration_minutes"].mean())

#Now that we have taken all the input from user, it is time to use them in our model. For that we will transform our inputs to their encoded counterparts thanks to our pickle file.
crr= model["carriers_encoded"].transform([carrier])[0]
to = model["to_encoded"].transform([destination])[0]
org = model["from_encoded"].transform([origin])[0]
todob= model["tod_ob_encoded"].transform([time_of_day_ob])[0]

#Creating a button to make the prediction.
click = st.button("Click to Predict")
#Putting all the information gathered in a numpy array to feed the model.
if click:
    prediction = model["model"].predict(np.array([[total_stops,day_of_week,month_number,day_of_month,ob_duration_minutes,ib_duration_minutes,day_of_year,to,org,crr,todob]]))
    st.header(f"Predicted price is €{round(prediction[0])}")