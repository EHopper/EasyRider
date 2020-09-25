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
    df = pd.read_csv(config.PROCESSED_DATA_PATH + 'ridewgps_labelled_culled.csv')

    return df

@st.cache
def load_ride(rte_id):
    filename = '{}{}.csv'.format(config.RAW_TRIPS_PATH, rte_id)
    df = pd.read_csv(filename).dropna()
    return df

data = load_data()

# Set up ride style choice:
st.title('It\'s the Catskills!')
st.subheader('Go to the sidebar to personalise your ride suggestion!')
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


N_RIDES = 5
if N_RIDES > 1:
    grammar = 'These rides'
else:
    grammar = 'This ride'
st.markdown('{} suitable for {}'.format(grammar, option.lower()))
colours = sns.color_palette('husl', N_RIDES)

top_rides = []
min_lat = 90
min_lon = 180
max_lat = -90
max_lon = -180
for i in range(N_RIDES):
    random_ride = data[data.labels == label_dict[option]].id.sample().values[0]
    # st.markdown('{}'.format(random_ride))
    ride_points = load_ride(random_ride)
    top_rides += [{'name': 'Ride {}: {}'.format(i, random_ride),
                   'color': colours[i],
                   'path': [[row.x, row.y] for i, row in ride_points.iterrows()]}]
    max_lat = max((ride_points.y.max(), max_lat))
    max_lon = max((ride_points.x.max(), max_lon))
    min_lat = min((ride_points.y.max(), min_lat))
    min_lon = min((ride_points.x.max(), min_lon))
    # st.markdown('{},{} - {},{}'.format(min_lat, min_lon, max_lat, max_lon))
ride_df = pd.DataFrame(top_rides)
ride_df.color = ride_df.color.apply(
    lambda x: tuple(int(v * 255) for v in x)
)
#ride_df[['name', 'color']]
ride_layers = pdk.Layer(
    type="PathLayer",
    data=ride_df,
    pickable=True,
    get_color="color",
    width_scale=20,
    width_min_pixels=2,
    get_path="path",
    get_width=5,
)

st.pydeck_chart(pdk.Deck(
    map_style="mapbox://styles/mapbox/light-v9",
    initial_view_state=pdk.data_utils.compute_view([
        [max_lon + 0.25, max_lat + 0.25],
        [max_lon + 0.25, min_lat - 0.25],
        [min_lon - 0.25, max_lat + 0.25],
        [min_lon - 0.25, min_lat - 0.25],
    ]),
    layers=ride_layers,
))
# else:
#     st.pydeck_chart(pdk.Deck(
#         map_style="mapbox://styles/mapbox/light-v9",
#         initial_view_state={"latitude": 41.979, "longitude": -74.218, "zoom":9},
#         #layers=ride_layers,
#     ))
