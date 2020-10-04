import streamlit as st
import pandas as pd
import numpy as np

import sklearn.neighbors

import pydeck as pdk
import seaborn as sns

from util import config
from util import mapping
from util import clean_data
from util import process_data

@st.cache(suppress_st_warning=True)
def load_data():
    st.write('Loading data!')
    trips = pd.read_feather(config.PROCESSED_DATA_PATH + 'trips_culled_scaled.feather')
    trips.set_index('rte_id', inplace=True)

    gridpts_at_rte_1000 = pd.read_feather(config.PROCESSED_DATA_PATH + 'gridpts_at_rte_culled_1000.feather')
    gridpts_at_rte_1000.set_index('rte_id', inplace=True)

    grid_pts_1000 = pd.read_feather(config.MODEL_PATH + 'grid_points_culled_1000.feather')
    grid_pts_1000.set_index('grid_id', inplace=True)

    return trips, grid_pts_1000, gridpts_at_rte_1000

@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def load_coarse_grid():
    st.write('Loading coarse grid!')
    grid_pts_75 = pd.read_feather(config.MODEL_PATH + 'grid_points_culled_75.feather')
    grid_pts_75.set_index('grid_id', inplace=True)

    rtes_at_grid_75 = pd.read_feather(config.MODEL_PATH + 'rtes_at_grid_culled_75.feather')
    rtes_at_grid_75.set_index('grid_id', inplace=True)

    loc_tree = sklearn.neighbors.KDTree(grid_pts_75[['lat', 'lon']])

    return grid_pts_75, rtes_at_grid_75, loc_tree

def load_presets():
    presets, presets_labels = process_data.set_presets()
    presets = process_data.apply_scaling(presets)

    feature_importance_dict = {'dist': 1.,
                      'avg_slope_climbing': 0.2,
                      'avg_slope_descending': 0.01,
                      'max_slope': 0.2,
                      'dist_climbing': 0.3,
                      'dist_downhill': 0.01,
                      'dist_6percent': 0.2,
                      'dist_9percent': 0.2,
                      'dist_12percent': 0.1,
                      'avg_speed': 0.}

    return (presets, presets_labels, feature_importance_dict)

def fit_tree(df, feature_importance):
    LEAF_SIZE = 20
    return sklearn.neighbors.KDTree(df * feature_importance, leaf_size=LEAF_SIZE)

st.write('Got to load some data')
# Load trip data (fine)
trips, grid_pts_fine, gridpts_at_rte_fine = load_data()

# Load coarser grid data for calculating distances
grid_pts_coarse, rtes_at_grid_coarse, loc_tree = load_coarse_grid()


# Set up ride style choice:
st.title('It\'s the Catskills!')
st.subheader('Go to the sidebar to personalise your ride suggestion!')
st.sidebar.subheader('Riders! What are you in the mood for?')
start_location_yn = st.sidebar.checkbox('Specify start location?')
if start_location_yn:
    MAX_DIST_FROM_START = 10 # miles
    start_lat = st.sidebar.text_input('Latitude (N):', 42)
    start_lon = st.sidebar.text_input('Longitude (W):', 74.25)
    start_lon = float(start_lon) * -1 # convert to degrees east
    start_lat = float(start_lat)

    trips_use, unscaled_dists = process_data.add_distance_to_start_feature(
        start_lat, start_lon, trips, grid_pts_coarse, rtes_at_grid_coarse, loc_tree, MAX_DIST_FROM_START
    )
else:
    trips_use = trips.copy()
    start_lat, start_lon = ('', '')

presets, presets_labels, feature_importance_dict = load_presets()
self_enter_lab = 'Enter my own'
option = st.sidebar.selectbox('Pick one:', [self_enter_lab] + presets_labels)

if option == self_enter_lab:
    st.markdown('Please enter your preferred ride parameters in the sidebar, or select one of our preset options.')

    if st.sidebar.checkbox('Distance?'):
        v0 = st.sidebar.slider('', min_value=5., max_value=100., value=20., step=2.5)
    else:
        v0 = 0.
        feature_importance_dict['dist'] = 0.

    if st.sidebar.checkbox('Average climbing slope?'):
        avg_slope_climbing = st.sidebar.slider('', min_value=3., max_value=9., step=0.5)
    else:
        v1 = 0.
        feature_importance_dict['avg_slope_climbing'] = 0.

    # if_max_slope =
    #
    # slope_in = st.sidebar.slider('Average slope:', min_value=1., max_value=10., value=2., step=0.5)
    # speed_in = st.sidebar.slider('Average speed:', min_value=6., max_value=20., value=10., step=2.)
    # speed_in = st.sidebar.slider('Average speed:', min_value=6., max_value=20., value=10., step=2.)


    chosen = pd.DataFrame([[v0, v1, 0., 0., 0., 0., 0., 0., 0., 0.]],
                        columns=presets.columns)
    if chosen.iloc[0].isna().sum():
        st.markdown('Please fill in all fields!')
    chosen = process_data.apply_scaling(chosen)
else:
    ind = presets_labels.index(option)
    chosen = presets.loc[[ind]]

chosen['lab'] = 'Your input'
chosen.set_index('lab', inplace=True)

if not chosen.iloc[0].isna().sum():
    N_RIDES = 5

    if start_location_yn:
        chosen['dist_to_start'] = 0.
        feature_importance_dict['dist_to_start'] = 1.
    feature_sc = [v for v in feature_importance_dict.values()]
    tree = fit_tree(trips_use, feature_sc)
    dists, df_inds = tree.query(chosen * feature_sc, k=5)
    dists, df_inds = dists.flatten(), df_inds.flatten()
    neighbour_rte_ids = trips_use.index[df_inds].tolist()

    # Find original values of the returned routes
    trips_unscaled = process_data.remove_scaling(trips_use.loc[neighbour_rte_ids])
    if start_location_yn:
        trips_unscaled['dist_to_start'] = unscaled_dists.loc[neighbour_rte_ids]

    if start_location_yn:
        chosen_unscaled = process_data.remove_scaling(chosen.drop('dist_to_start', axis=1))
        chosen_unscaled['dist_to_start'] = 0.
    else:
        chosen_unscaled = process_data.remove_scaling(chosen)
    st.dataframe(trips_unscaled.append(chosen_unscaled))

    r = process_data.plot_NN(
        neighbour_rte_ids, grid_pts_fine, gridpts_at_rte_fine,
        (start_lat, start_lon, start_location_yn),
    )

    st.pydeck_chart(r)
else:
    st.pydeck_chart(pdk.Deck(
        map_style="mapbox://styles/mapbox/light-v9",
        initial_view_state={"latitude": 41.979, "longitude": -74.218, "zoom":9},
        #layers=ride_layers,
    ))
