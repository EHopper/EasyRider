import streamlit as st
import pandas as pd
import numpy as np
import requests
import json

import sklearn.neighbors

import pydeck as pdk
import seaborn as sns

from util import config
from util import process_data

@st.cache
def load_data():
    df = pd.read_csv(config.PROCESSED_DATA_PATH + 'trips.csv', index_col=0)
    return df

@st.cache
def load_presets():
    presets, presets_labels = process_data.set_presets()
    presets = process_data.apply_scaling(presets)
    return (presets, presets_labels)

@st.cache(allow_output_mutation=True)
def fit_tree(df, feature_importance):
    LEAF_SIZE = 20
    return sklearn.neighbors.KDTree(df * feature_importance, leaf_size=LEAF_SIZE)

@st.cache
def load_ride(rte_id):
    filename = '{}{}.csv'.format(config.RAW_TRIPS_PATH, rte_id)
    df = pd.read_csv(filename).dropna()
    return df

data = load_data()
feature_importance = [1] * data.shape[1]
tree = fit_tree(data, feature_importance)

# Set up ride style choice:
st.title('It\'s the Catskills!')
st.subheader('Go to the sidebar to personalise your ride suggestion!')
st.sidebar.subheader('Riders! What are you in the mood for?')
presets, presets_labels = load_presets()
self_enter_lab = 'Enter my own'
option = st.sidebar.selectbox('Pick one:', [self_enter_lab] + presets_labels)

if option == self_enter_lab:
    st.markdown('Please enter your preferred ride parameters in the sidebar, or select one of our preset options.')
    dist_in = st.sidebar.slider('Distance:', min_value=5., max_value=100., value=20.)
    slope_in = st.sidebar.slider('Average slope:', min_value=1., max_value=10., value=5.)
    speed_in = st.sidebar.slider('Average speed:', min_value=5., max_value=20., value=10.)
    breaks_in = st.sidebar.select_slider('Opportunity for breaks:', ['low', 'medium', 'high'], value='medium')

    breaks_dict = {'low': 1.0, 'medium': 0.85, 'high': 0.7}
    breaks_in = breaks_dict[breaks_in]

    chosen = pd.DataFrame([[dist_in, slope_in, speed_in, breaks_in]],
                        columns=presets.columns)
    chosen
    if chosen.iloc[0].isna().sum():
        st.markdown('Please fill in all fields!')
    chosen = process_data.apply_scaling(chosen)
else:
    ind = presets_labels.index(option)
    chosen = presets.loc[[ind]]

if not chosen.iloc[0].isna().sum():
    N_RIDES = 5

    dist, ind = tree.query(chosen, k=N_RIDES)

    colours = sns.color_palette('husl', N_RIDES)

    top_rides = []
    min_lat = 90
    min_lon = 180
    max_lat = -90
    max_lon = -180
    for i in range(N_RIDES):
        ride_ind = data.index[ind[0, i]]
        ride_points = load_ride(ride_ind)
        top_rides += [{'name': 'Ride {}: {}'.format(i, ride_ind),
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
else:
    st.pydeck_chart(pdk.Deck(
        map_style="mapbox://styles/mapbox/light-v9",
        initial_view_state={"latitude": 41.979, "longitude": -74.218, "zoom":9},
        #layers=ride_layers,
    ))
