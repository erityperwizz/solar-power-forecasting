import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import streamlit as st
from sklearn.preprocessing import LabelEncoder

@st.cache_data
def load_data():
    file_path = os.path.join(os.path.dirname(__file__), "mumbai-2023.csv")
    df = pd.read_csv(file_path)
    
    # Dropping
    df = df.dropna().copy()
    df.drop(['Time', 'Precip.'], axis=1, inplace=True)
    
    # Mapping 'Condition' values
    df['Condition'] = df['Condition'].replace({
        'Haz': 'Haze',
        'Light Rain': 'Rain',
        'Heavy Rain': 'Rain',
        'Heavy Rain Shower': 'Rain',
        'Light Rain with Thunder': 'Rain',
        'Drizzle': 'Rain',
        'Showers in the Vicinity': 'Rain',
        'Light Drizzle': 'Rain',
        'Thunder': 'Rain',
        'T-Storm': 'Rain',
        'Light Rain Shower': 'Rain',
        'Heavy T-Storm': 'Rain',
        'Rain Shower': 'Rain',
        'Heavy Rain / Windy': 'Rain',
        'Light Rain / Windy': 'Rain',
        'Partly Cloudy': 'Fog',
        'Mostly Cloudy': 'Fog',
        'Fog / Windy': 'Fog',
        'Smoke / Windy': 'Smoke',
        'Haze / Windy': 'Haze'
    })
    
    # Removing units and converting to float
    df['Temperature'] = df['Temperature'].str.replace('°C', '').astype('float64')
    df['Dew Point'] = df['Dew Point'].str.replace('°C', '').astype('float64')
    df['Humidity'] = df['Humidity'].str.replace('%', '').astype('float64')
    df['Wind Speed'] = df['Wind Speed'].str.replace('km/h', '').astype('float64')
    df['Wind Gust'] = df['Wind Gust'].str.replace('km/h', '').astype('float64')
    df['Pressure'] = df['Pressure'].str.replace(',', '').str.replace('hPa', '').astype('float64')
    
    # Label encoding for 'Wind'
    le = LabelEncoder()
    df['Wind'] = le.fit_transform(df['Wind'])

    return df


dts = load_data() 

X = dts.iloc[:, :-1].to_numpy(dtype=float)
y = dts.iloc[:, -1].astype(str).to_numpy(dtype=object)

X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)  # Train-test split for model training

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

def prediction(inputs):
    inputs = np.array(inputs).reshape(1, -1)  # Reshape inputs to be a 2D array
    panel = model.predict(inputs)
    return panel[0]

st.title("Solar Panel Prediction App")  

Temperature = st.number_input('Temperature')
Dew_Point = st.number_input('Dew Point')

st.write("Enter humidity 0-100")
Humidity = st.number_input('Humidity')

Wind = st.number_input('Wind')

st.write("Enter wind speed in kmph")
Wind_Speed = st.number_input('Wind Speed')

st.write("Enter pressure in mm")
Pressure = st.number_input('Pressure')


if st.button('Predict'):
    user_inputs = [Temperature, Dew_Point, Humidity, Wind, Wind_Speed, 0, Pressure]
    weather = prediction(user_inputs)

    def map_weather_to_panel(weather):
        weather_to_panel_mapping = {
            'Snow': 'Bifacial Panels',
            'Smoke': 'Monocrystalline or Polycrystalline Panels',
            'Haze': 'Monocrystalline or Polycrystalline Panels',
            'Winter': 'PERC (Passivated Emitter Rear Contact) or Thin-Film Panels',
            'Rain': 'PERC (Passivated Emitter Rear Contact) or Thin-Film Panels',
            'Mist': 'PERC (Passivated Emitter Rear Contact) or Thin-Film Panels',
            'Fog' : 'Monocrystalline'
            # Add more conditions as needed
        }

        return weather_to_panel_mapping.get(weather, 'Unknown')

    panel_type = map_weather_to_panel(weather)
    st.write(f'Type of solar panel recommended for {weather}: {panel_type}')

    
