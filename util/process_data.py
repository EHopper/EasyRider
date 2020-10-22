import pandas as pd
import numpy as np
import os

import seaborn as sns
import pydeck as pdk

import sklearn.preprocessing
import sklearn.neighbors

from util import config
from util import mapping


def get_features_from_rte_files(rte_ids):
    """ Extract potentially relevant features from each ride.

    The features are calculated by passing the ride dataframe to calc_features().  As this more detailed data is sometimes very noisy, we also do an additional filtering here, replicating the removal of outliers done in clean_data.filter_trips().

    Arguments:
        rte_ids
            - list of integers
            - The IDs of rides that we want to process, where rides are saved at
                     config.CLEAN_TRIPS_PATH [rte_id].feather

    Returns:
        A dataframe of features
            - pd.DataFrame
            - Columns are 'rte_id' and a variety of summary features calculated for each ride in calc_features()
    """
    trip_data = []
    for i, rte_id in enumerate(rte_ids):
        if not i % 1000: print(i)
        rte = pd.read_feather(
            os.path.join(config.CLEAN_TRIPS_PATH, '{}.feather'.format(rte_id))
        )

        features = calc_features(rte, rte_id)

        if (features['dist'] < 1        # Do some filtering, as clean_data.filter_trips()
            or 12 < features['avg_slope_climbing']
            or features['avg_slope_descending'] < -12
            or 25 < features['max_slope']
            ):
            continue


        trip_data += [features]

    return pd.DataFrame(trip_data).fillna(0)

def calc_features(rte_df, rte_id):
    """ Calculate some potentially important summary features of a ride.

    Here, we calculate
        dist:                   total length of a ride (miles)
        max_slope:              maximum slope on a ride
        avg_slope_climbing:     average slope when climbing, here defined as > 1% slope
        avg_slope_descending:   average slope when descending,  < -1% slope
        dist_climbing:          length of ride above a 3% slope (mi)
        dist_downhill:          length of ride below a -3% slope (mi)
        dist_6percent:          proportion of ride above a 6% climb
        dist_9percent:          proportion of ride above a 9% climb
        dist_12percent:         proportion of ride above a 12% climb

    Arguments:
        rte_df
            - pd.DataFrame
            - Units:    Imperial (miles, feet, miles per hour, hours)
                        Degrees North and East
                        Slope is given as a percentage
    """

    dist = rte_df.dist.sum()
    max_slope = rte_df.slope.max()
    avg_slope_climbing = rte_df[rte_df.slope > 1].slope.mean()
    avg_slope_descending = rte_df[rte_df.slope < -1].slope.mean()
    dist_climbing = rte_df[rte_df.slope > 3].dist.sum() / dist
    dist_downhill = rte_df[rte_df.slope < -3].dist.sum() / dist
    dist_6percent = rte_df[rte_df.slope > 6].dist.sum() / dist
    dist_9percent = rte_df[rte_df.slope > 9].dist.sum() / dist
    dist_12percent = rte_df[rte_df.slope > 12].dist.sum() / dist


    return {'rte_id': rte_id, 'dist': dist,
            'avg_slope_climbing': avg_slope_climbing, 'avg_slope_descending': avg_slope_descending,
            'max_slope': max_slope,
            'dist_climbing': dist_climbing,
            'dist_downhill': dist_downhill,
            'dist_6percent': dist_6percent,
            'dist_9percent': dist_9percent,
            'dist_12percent': dist_12percent,
            }


def engineer_normal_features(df):
    """ Engineer features that have a closer to normal distribution

    The features below have strong positive skew - try to remove by taking the log.
    Note that have to add a small number because there are zeros in the dataset.

    Arguments:
        df
            - pd.DataFrame
            - Assumed to have the following columns:
                'dist', 'dist_6percent', 'dist_9percent', 'dist_12percent'
              and that these are the only columns with a positive skewness

    Returns:
        df_eng
            - pd.DataFrame
            - As input, but columns listed above have been replaced with their log
    """

    df_eng = df.copy()
    skewed_cols = ['dist', 'dist_6percent', 'dist_9percent', 'dist_12percent']

    for col in skewed_cols:
        df_eng[col] = np.log(df[col] + 1e-2)

    return df_eng

def reverse_engineer_features(df):
    """ Undo the changes made in engineer_normal_features()

    This is useful if you want to get the actual values back again, i.e. ride length not log(ride length) - for human interpretability.

    Arguments:
        df
            - pd.DataFrame
            - Assumed to have the following columns:
                'dist', 'dist_6percent', 'dist_9percent', 'dist_12percent'
              and that these are the only columns that were adjusted in engineer_normal_features()

    Returns:
        df_eng
            - pd.DataFrame
            - As input, but columns listed above have been replaced with their exponent
    """

    df_reverse = df.copy()
    skewed_cols = ['dist', 'dist_6percent', 'dist_9percent', 'dist_12percent']
    for col in skewed_cols:
        df_reverse[col] = np.exp(df[col]) - 1e-2

    return df_reverse


def scale_dataset(df):
    """ Calculate appropriate scaling for the dataset using sklearn's StandardScaler

    For the nearest-neighbours algorithm, want all features to be scaled equally as a baseline (we can then weight them differently from uniform).  We also need to be able to apply the same scaling to user inputs, so this data needs to be saved.

    StandardScaler in sklearn demeans and is normalised by the standard deviation
            ((X - mean) / std)

    Arguments:
        df
            - pd.DataFrame
            - Features are all assumed to be numeric and approximately normal

    Returns:
        df
            - pd.DataFrame
            - Features are demeaned and scaled to unit standard deviation
        scaler_df
            - pd.DataFrame
            - Columns are the same as the input df (except for 'rte_id', which is skipped if present)
            - Rows are indexed as 'mean' and 'std'
                - the mean and standard deviation calculated by StandardScaler

    """
    # Scaling is ((X - mean) / std )
    scaler = sklearn.preprocessing.StandardScaler()
    cols = df.columns.tolist()
    if 'rte_id' in cols:
        cols.remove('rte_id')

    df[cols] = scaler.fit_transform(df[cols])

    scaler_df = pd.DataFrame(np.vstack((scaler.mean_, scaler.scale_)),
                     columns=cols, index=['mean', 'std'])

    return df, scaler_df

def remove_scaling(df):
    """ Undo the scaling done in scale_dataset() or apply_scaling()

    This is used to increase human readability on scaled datasets.

    NOTE: it is assumed that the appropriate scaling dataframe will be saved at
                config.MODEL_PATH 'feature_scaling.feather'
    This dataframe should have n_scaled_features + 1 columns and two rows:
        'index':  values are strings, 'mean' and 'std'
        All other column names should be a subset of the columns in the input df, where the first row is the mean and the second row is the standard deviation.

    Arguments:
        df
            - pd.DataFrame
            - Unitless, as these have all been normalised
            - Columns should be a superset of the columns in 'feature_scaling.feather', other than the 'index' column

    Returns:
        df_unscale
            - pd.DataFrame
            - Units:    back to what they should be!
            - All columns in 'feature_scaling.feather' are unscaled
                    i.e. X * std + mean

    """
    scaler = pd.read_feather(
        os.path.join(config.MODEL_PATH, 'feature_scaling.feather')
    )
    scaler = scaler.set_index('index')
    df_unscale = df.copy()
    for col in scaler.columns:
        df_unscale[col] = (df_unscale[col]
                    * scaler.loc['std', col] + scaler.loc['mean', col])

    return df_unscale

def apply_scaling(df):
    """ Apply the scaling calculated in scale_dataset()

    This is used to ensure user inputs or other new datasets are scaled in the same way.

    NOTE: it is assumed that the appropriate scaling dataframe will be saved at
                config.MODEL_PATH 'feature_scaling.feather'
    This dataframe should have n_scaled_features + 1 columns and two rows:
        'index':  values are strings, 'mean' and 'std'
        All other column names should be a subset of the columns in the input df, where the first row is the mean and the second row is the standard deviation.

    Arguments:
        df
            - pd.DataFrame
            - Columns should be a superset of the columns in 'feature_scaling.feather', other than the 'index' column

    Returns:
        df_scale
            - pd.DataFrame
            - Units:    unitless!
            - All columns in 'feature_scaling.feather' are unscaled
                    i.e. (X - mean) / std
    """

    scaler = pd.read_feather(
        os.path.join(config.MODEL_PATH, 'feature_scaling.feather')
    )
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
