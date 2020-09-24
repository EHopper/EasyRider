import streamlit as st
import pandas as pd
import numpy as np
import requests
import json

import pydeck as pdk
import seaborn as sns

from util import config

@st.cache
def load_data():
    df = pd.read_csv(config.PROCESSED_DATA_PATH + 'ridewgps_labelled.csv')

    return df

@st.cache
def load_ride(rte_id):
    filename = '{}{}'.format(config.RAW_TRIPS_PATH, rte_id)
    with open(filename, mode='r') as localfile:
        r = localfile.read()
    return json.loads(r)

data = load_data()

# Set up ride style choice:
st.title('It\'s the Catskills!')
st.subheader('Go to the sidebar to generate a ride suggestion!')
st.sidebar.subheader('Riders! What are you in the mood for?')
opt_descriptions = [
    'Chilling out in the saddle', 'Pretty relaxed, with some climbing',
    'Half-day of touring', 'Training for VO2-max', 'Training for strength',
    'Training for a century']
label_dict = dict.fromkeys(opt_descriptions)
label_order = [5, 2, 1, 0, 3, 4] # From visual inspection
for i, l in enumerate(label_order):
    label_dict[opt_descriptions[i]] = l
option = st.sidebar.selectbox('Pick one:', opt_descriptions)


if option:
    N_RIDES = 2
    if N_RIDES > 1:
        grammar = 'These rides'
    else:
        grammar = 'This ride'
    st.markdown('{} suitable for {}'.format(grammar, option.lower()))
    colours = sns.color_palette('husl', N_RIDES)

    top_rides = []
    for i in range(N_RIDES):
        random_ride = data[data.labels == label_dict[option]].id.sample().values[0]
        ride = load_ride(random_ride)
        # st.markdown('{}'.format(random_ride))
        ride_points = pd.DataFrame(ride['track_points'])[['y', 'x']]
        top_rides += [{'name': 'Ride {}: {}'.format(i, random_ride),
                       'colour': colours[i],
                       'path': [[row.x, row.y] for i, row in ride_points.iterrows()]}]
    ride_layers = pdk.Layer(
        type="PathLayer",
        data=pd.DataFrame(top_rides),
        pickable=True,
        get_color="colour",
        width_scale=20,
        width_min_pixels=2,
        get_path="path",
        get_width=5,
    )
    st.pydeck_chart(pdk.Deck(
        map_style="mapbox://styles/mapbox/light-v9",
        initial_view_state={"latitude": 41.979, "longitude": -74.218, "zoom":8},
        layers=ride_layers,
    ))
else:
    st.pydeck_chart(pdk.Deck(
        map_style="mapbox://styles/mapbox/light-v9",
        initial_view_state={"latitude": 41.979, "longitude": -74.218, "zoom":9},
        #layers=ride_layers,
    ))
