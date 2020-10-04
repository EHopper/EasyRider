import pandas as pd
import numpy as np
import os

import seaborn as sns
import pydeck as pdk

import sklearn.preprocessing
import sklearn.neighbors

from util import config
from util import mapping

def set_presets():
    df = pd.read_csv(config.PROCESSED_DATA_PATH + 'trips.csv')

    presets_descriptions = [
        'Chilling out in the saddle', 'Pretty relaxed, with some climbing',
        'Half-day of touring', 'Training for VO2-max', 'Training for strength',
        'Training for a century']

    presets = pd.DataFrame({
        'dist': [10., 15., 45., 20., 10., 85.],
        'avg_slope_climbing': [3., 6., 5., 5., 8., 4.],
        'avg_slope_descending': [-3., -6., -5., -5., -8., -4.],
        'max_slope': [6., 10., 10., 6., 15., 10.],
        'dist_climbing': [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
        'dist_downhill': [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
        'dist_6percent': [0.05, 0.05, 0.05, 0.05, 0.05, 0.05],
        'dist_9percent': [0.01, 0.01, 0.01, 0.01, 0.01, 0.01],
        'dist_12percent': [0.005, 0.005, 0.005, 0.005, 0.005, 0.005],
        'avg_speed': [10., 8., 15., 18., 8., 15.],
    })

    return presets, presets_descriptions

def scale_dataset(df, column_importance):
    # Scaling is ((X - mean) / std ) * column_importance
    scaler = sklearn.preprocessing.StandardScaler()
    cols = df.columns
    df[cols] = scaler.fit_transform(df[cols])
    df[cols] *= column_importance

    scaler_df = pd.DataFrame(np.vstack((scaler.mean_, scaler.scale_, column_importance)),
                     columns=cols, index=['mean', 'std', 'column_importance'])
    scaler_df.reset_index().to_feather(config.MODEL_PATH + 'feature_scaling.feather')

    return df

def remove_scaling(df):
    # Scaling is ((X - mean) / std ) * column_importance
    scaler = pd.read_feather(config.MODEL_PATH + 'feature_scaling.feather')
    scaler = scaler.set_index('index')
    df_unscale = df.copy()
    for col in scaler.columns:
        df_unscale[col] = (df_unscale[col] / scaler.loc['column_importance', col]
                    * scaler.loc['std', col] + scaler.loc['mean', col])

    return df_unscale

def apply_scaling(df):
    # Scaling is ((X - mean) / std ) * column_importance
    scaler = pd.read_feather(config.MODEL_PATH + 'feature_scaling.feather')
    scaler = scaler.set_index('index')
    df_scale = df.copy()
    for col in scaler.columns:
        df_scale[col] = ((df_scale[col] - scaler.loc['mean', col])
                            / scaler.loc['std', col]
                            * scaler.loc['column_importance', col])

    return df_scale


def add_distance_to_start_feature(lat, lon, trips_df, grid_pts, rtes_at_grid,
                                  loc_tree, max_dist_from_start=10):

    dist_to_point = calc_dist_from_point_to_rtes(lat, lon, grid_pts, rtes_at_grid, loc_tree)
    trips_loc = pd.merge(trips_df, dist_to_point, on='rte_id', how='inner').set_index('rte_id')
    # Filter out routes that are too close
    trips_loc = trips_loc[trips_loc['dist_to_start'] < max_dist_from_start] # miles
    unscaled_dists = trips_loc[['dist_to_start']].copy()
    trips_loc['dist_to_start'] = trips_loc['dist_to_start'].apply(bin_ride_distance, args=[max_dist_from_start])

    return trips_loc, unscaled_dists

def calc_dist_from_point_to_rtes(start_lat, start_lon, grid_pts, rtes_at_grid, loc_tree):
    dists, inds = loc_tree.query(np.array([[start_lat, start_lon]]), k=2000)
    dists *= mapping.degrees_to_miles_ish(1)
    rtes_done = set()
    ds = []
    for dist_to_point, i in zip(dists.ravel(), inds.ravel()):
        grid_id = grid_pts.index[i]
        rtes_at_point = set(rtes_at_grid.loc[grid_id].rte_ids)
        rtes_at_point -= rtes_done
        ds += [{'dist_to_start': dist_to_point, 'rte_id': rte} for rte in rtes_at_point]
        rtes_done = rtes_done.union(rtes_at_point)

    return pd.DataFrame(ds)


def bin_ride_distance(x, max_dist_from_start):
    if x < 1:
        return 0
    if x < max_dist_from_start / 5:
        return 0.5
    if x < max_dist_from_start / 2:
        return 3
    else:
        return 10

def plot_NN(rte_ids, grid_pts, gridpts_at_rte, start_locs):
    n_rides = len(rte_ids)
    start_lat, start_lon, start_location_yn = start_locs


    route_layers = []
    colours = sns.color_palette('husl', n_rides)
    map_lims = np.array([[90, -90, 180, -180]])
    for i, rte_id in enumerate(rte_ids):
        rte_layer, map_lims = load_rte_into_map_layer(rte_id, gridpts_at_rte, grid_pts, map_lims, colours[i])
        route_layers += rte_layer

    # Find actual limits of the map
    if start_location_yn:
        map_lims = np.vstack((map_lims,
                              np.array([[start_lat, start_lat, start_lon, start_lon]])
                            ))
        route_layers += [lat_lon_layer(start_lon, start_lat, 'start')]

    max_lat, max_lon = tuple(map_lims[:, 1:4:2].max(axis=0))
    min_lat, min_lon = tuple(map_lims[:, 0:3:2].min(axis=0))

    return pdk.Deck(
        map_style="mapbox://styles/mapbox/light-v10",
        initial_view_state=pdk.data_utils.compute_view([
            [max_lon + 0.25, max_lat + 0.25],
            [max_lon + 0.25, min_lat - 0.25],
            [min_lon - 0.25, max_lat + 0.25],
            [min_lon - 0.25, min_lat - 0.25],
        ]),
        layers=route_layers,
        tooltip={"html": "<b>Ride {name}</b>", "style": {"color": "white"}},
        mapbox_key=os.getenv('MAPBOX_API_KEY')

    )

def lat_lon_layer(lat, lon, label):
    return pdk.Layer(
            type="ScatterplotLayer",
            data=pd.DataFrame({'name': [label], 'size': [2],
                               'coordinates': [[lat, lon]]}),
            opacity= 1,
            filled=True,
            stroked=False,
            radius_min_pixels=3,
            get_position="coordinates",
            get_radius="size",
            get_fill_color=[255, 0, 0],
    )

def load_rte_into_map_layer(rte_id, gridpts_at_rte, grid_pts, map_lims, colour):
    grid_at_rte_i = gridpts_at_rte.loc[rte_id][0].tolist() # Find grid points at that route
    rte_lat_lon = grid_pts.loc[grid_at_rte_i][['lat', 'lon']] # Find lat/lon at those grid points

    res = [pdk.Layer(
            type="ScatterplotLayer",
            data=pd.DataFrame({
                'name': ['{}'.format(rte_id)] * rte_lat_lon.shape[0],
                'coordinates': [[row.lon, row.lat] for i, row in rte_lat_lon.iterrows()],
                'size': [1] * rte_lat_lon.shape[0],
            }),
            opacity= 1,
            pickable=True,
            filled=True,
            stroked=False,
            radius_min_pixels=1,
            get_position="coordinates",
            get_radius="size",
            get_fill_color=[int(x * 255) for x in colour],
        )
    ]
    map_lims = np.vstack((map_lims,
                        np.array([[rte_lat_lon.lat.min(),
                                   rte_lat_lon.lat.max(),
                                   rte_lat_lon.lon.min(),
                                   rte_lat_lon.lon.max()]])))
    return res, map_lims
