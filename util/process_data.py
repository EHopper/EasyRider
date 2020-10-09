import pandas as pd
import numpy as np
import os

import seaborn as sns
import pydeck as pdk

import sklearn.preprocessing
import sklearn.neighbors

from util import config
from util import mapping


def get_features_from_rte_files(rte_ids, ifsave=False):

    trip_data = []
    for i, rte_id in enumerate(rte_ids):
        if not i % 1000: print(i)

        rte_df = pd.read_feather(config.CLEAN_TRIPS_PATH + '{}.feather'.format(rte_id))
        if rte_df.dist.sum() < 1:
            continue

        # Remove any trips that circle around too often
        if 5 * mapping.dist_lat_lon(rte_df.lat.min(), rte_df.lon.min(), rte_df.lat.max(), rte_df.lon.max()) < rte_df.dist.sum():
            continue

        features = calc_features(rte_df, rte_id)

        if (features['dist'] < 1
            or 10 < features['avg_slope_climbing']
            or features['avg_slope_descending'] < -10
            or 25 < features['max_slope']
            ):
            continue


        trip_data += [features]

    res = pd.DataFrame(trip_data)
    res.fillna(0, inplace=True)
    if ifsave:
        res.to_feather(config.PROCESSED_DATA_PATH + 'trips_unscaled.feather')
    return res

def calc_features(rte_df, rte_id):

    dist = rte_df.dist.sum()
    max_slope = rte_df.slope.max()
    avg_slope_climbing = rte_df[rte_df.slope > 1].slope.mean()
    avg_slope_descending = rte_df[rte_df.slope < 1].slope.mean()
    dist_climbing = rte_df[rte_df.slope > 3].dist.sum() / dist
    dist_downhill = rte_df[rte_df.slope < -3].dist.sum() / dist
    dist_6percent = rte_df[rte_df.slope > 6].dist.sum() / dist
    dist_9percent = rte_df[rte_df.slope > 9].dist.sum() / dist
    dist_12percent = rte_df[rte_df.slope > 12].dist.sum() / dist

    if rte_df.time.max() == 0:
        avg_speed = 0.
    else:
        avg_speed = rte_df[rte_df.speed > 2].speed.mean()

    return {'rte_id': rte_id, 'dist': dist,
            'avg_slope_climbing': avg_slope_climbing, 'avg_slope_descending': avg_slope_descending,
            'max_slope': max_slope,
            'dist_climbing': dist_climbing,
            'dist_downhill': dist_downhill,
            'dist_6percent': dist_6percent,
            'dist_9percent': dist_9percent,
            'dist_12percent': dist_12percent,
            }


def set_presets():

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
        'detour_score': [10, 10, 10, 10, 10, 10],
        'popularity': [10, 10, 10, 10, 10, 10],
    })

    return presets, presets_descriptions

def engineer_features(df):

    df_eng = df.copy()
    df_eng['dist'] = np.log(df.dist +1e-2)
    df_eng['dist_6percent'] = np.log(df.dist_6percent + 1e-2)
    df_eng['dist_9percent'] = np.log(df.dist_9percent + 1e-2)
    df_eng['dist_12percent'] = np.log(df.dist_12percent + 1e-2)

    # df_eng.to_feather(config.PROCESSED_DATA_PATH + 'trips_eng.feather')
    return df_eng

def reverse_engineer_features(df):

    df_reverse = df.copy()
    df_reverse['dist'] = np.exp(df.dist) - 1e-2
    df_reverse['dist_6percent'] = np.exp(df.dist_6percent) - 1e-2
    df_reverse['dist_9percent'] = np.exp(df.dist_9percent) - 1e-2
    df_reverse['dist_12percent'] = np.exp(df.dist_12percent) - 1e-2

    return df_reverse


def scale_dataset(df):
    # Scaling is ((X - mean) / std ) * column_importance
    scaler = sklearn.preprocessing.StandardScaler()
    cols = df.columns.tolist()
    if 'rte_id' in cols:
        cols.remove('rte_id')

    df[cols] = scaler.fit_transform(df[cols])

    scaler_df = pd.DataFrame(np.vstack((scaler.mean_, scaler.scale_)),
                     columns=cols, index=['mean', 'std'])
    scaler_df.reset_index().to_feather(config.MODEL_PATH + 'feature_scaling.feather')

    # df.to_feather(config.PROCESSED_DATA_PATH + 'trips_scaled.feather')

    return df

def remove_scaling(df):
    # Scaling is ((X - mean) / std ) * column_importance
    scaler = pd.read_feather(config.MODEL_PATH + 'feature_scaling.feather')
    scaler = scaler.set_index('index')
    df_unscale = df.copy()
    for col in scaler.columns:
        df_unscale[col] = (df_unscale[col]
                    * scaler.loc['std', col] + scaler.loc['mean', col])

    return df_unscale

def apply_scaling(df):
    # Scaling is ((X - mean) / std ) * column_importance
    scaler = pd.read_feather(config.MODEL_PATH + 'feature_scaling.feather')
    scaler = scaler.set_index('index')
    df_scale = df.copy()
    for col in scaler.columns:
        df_scale[col] = ((df_scale[col] - scaler.loc['mean', col])
                            / scaler.loc['std', col])

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
    colours = sns.color_palette(["#7A2008", "#d4350b", "#ff5224",  "#b68679", "#df8770"])
    # colours = sns.color_palette('husl', n_rides)
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
