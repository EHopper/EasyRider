""" Clean some data
"""
import pandas as pd
import numpy as np
import datetime
import re
import pathlib
import joblib
import os
import collections
import json

import sklearn.neighbors

from util import config
from util import mapping


def load_clean_ridewgps_trips():
    """
    """
    trips = pd.read_csv(config.RAW_DATA_PATH + 'ridewgps_trips.csv')
    clean_ridewgps_rides(trips)
    clean_trips(trips)

    return filter_trips(trips)

def load_ridewgps_allrides():
    """ Load both trips and routes

    PROBLEM: routes are often copied, so have duplicated data.
    Trips ONLY gives more of a heat map, rides people have actually done.
    (Though also messier...)
    """

    trips = pd.read_csv(config.RAW_DATA_PATH + 'ridewgps_trips.csv')
    routes = pd.read_csv(config.RAW_DATA_PATH + 'ridewgps_routes.csv')


    trips['is_trip'] = True
    routes['is_trip'] = False

    # Useful columns from trips
    trips = clean_trips(trips)
    routes['avg_speed'] = trips.avg_speed.mean()

    # Excess columns in 'trips' are things like HR, cadence, time of day, etc
    useless_cols =['Unnamed: 0', 'visibility', 'deleted_at',
                   'postal_code', 'locality', 'administrative_area',
                   'country_code', 'short_location']
    common_cols = [col for col in trips.columns
                    if col in routes.columns and col not in useless_cols]
    return trips[common_cols].append(routes[common_cols]).reset_index(drop=True)

def filter_trips(df):
    MIN_TIME = 0.5
    MAX_TIME = 15
    MIN_AVERAGE_SPEED = 5
    MAX_AVERAGE_SPEED = 25
    MAX_SPEED = 50
    MAX_DISTANCE = 125
    MAX_AVG_SLOPE = 12

    return df[((df['is_stationary'] == False)
               & (MIN_TIME < df['moving_time'])
               & (df['moving_time'] < MAX_TIME)
               & (MIN_TIME < df['duration'])
               & (df['duration'] < MAX_TIME)
               & (df['distance'] < MAX_DISTANCE)
               & (MIN_AVERAGE_SPEED < df['avg_speed'])
               & (df['avg_speed'] < MAX_AVERAGE_SPEED)
               & (df['max_speed'] < MAX_SPEED)
               & (df['avg_slope'] < MAX_AVG_SLOPE))].reset_index(drop=True)

def filter_cleaned_trips(df):
    feather_directory = pathlib.Path(config.CLEAN_TRIPS_PATH)
    clean_trip = [rte_id.stem for rte_id in feather_directory.glob('*.feather')]
    return df[(df['rte_id'].isin(clean_trip))]


def clean_trips(df):
    """
    """

    # Make units imperial
    df['avg_speed'] = mapping.km_to_mi(df['avg_speed'])
    df['max_speed'] = mapping.km_to_mi(df['max_speed'])
    df['duration'] /= (60 * 60) # seconds to hours
    df['moving_time'] /= (60 * 60)

    df['if_weekend'] = df['departed_at'].apply(lambda x: 5 < rwgps_strtime(x).weekday())
    df['prop_moving'] = df['moving_time'] / df['duration']

    useless_cols = ([col for col in df.columns if re.search("hr|watts|cad|power_estimated", col)]
                 + ['calories', 'gear_id', 'is_gps', 'processed', 'route_id',
                    'source_type', 'time_zone', 'utc_offset']
    )

    df.drop(useless_cols, axis=1, inplace=True)



def clean_ridewgps_rides(df):
    """
    """

    # Do unit conversion
    df['distance'] = mapping.metres_to_miles(df['distance'])
    df['elevation_gain'] = mapping.metres_to_feet(df['elevation_gain'])
    df['elevation_loss'] = mapping.metres_to_feet(df['elevation_loss'])

    # Fill NA
    df['description'].fillna('', inplace=True)
    for col in ['first_lat', 'first_lng', 'last_lat', 'last_lng']:
        df[col].fillna(df[col].mean(), inplace=True)


    # Make more pared down features
    df['update_days'] = df.apply(lambda x: rwgps_strtime(x.updated_at) - rwgps_strtime(x.created_at), axis=1)
    df['update_days'] = df['update_days'].apply(lambda x: x.days)
    df['if_updated'] = df['update_days'] > 0

    df['elevation_net'] = df['elevation_gain'] - df['elevation_loss']
    df['elevation_total'] = df['elevation_gain'] + df['elevation_loss']
    df['avg_slope'] = mapping.feet_to_miles(df['elevation_total']) / df['distance'] * 100

    df['photos'] = df['highlighted_photo_id'] > 0

    user_counts = df['user_id'].value_counts()
    USER_RIDES_CUTOFF = 100
    df['big_user'] = df['user_id'].apply(
        lambda x: x in user_counts[USER_RIDES_CUTOFF <= user_counts].index.tolist()
    )

    df['crow_distance'] = mapping.dist_lat_lon(df['first_lat'], df['first_lng'],
                                               df['last_lat'], df['last_lng'])


    drop_cols = ['updated_at', 'created_at', 'highlighted_photo_id',
                 'group_membership_id', 'elevation_loss', 'track_id',
                 'sw_lng', 'sw_lat', 'ne_lng', 'ne_lat',
                 'first_lat', 'first_lng', 'last_lat', 'last_lng',
                 ]
    useless_cols =['Unnamed: 0', 'visibility', 'deleted_at',
                   'postal_code', 'locality', 'administrative_area',
                   'country_code', 'short_location']

    df.drop(drop_cols + useless_cols, axis=1, inplace=True)


def rwgps_strtime(x:str):
    return datetime.datetime.strptime(x, '%Y-%m-%dT%H:%M:%S-%f:00')


def check_missing(df):
    for col in df.columns:
        print('{}: {} missing of {}'.format(col, df[col].isna().sum(), df.shape[0]))

def moving_window_average(t_series, N=5):
  # Calculate average over moving window
  # N: number of points to average over
  return np.convolve(t_series, np.ones((N, )) / N, mode='same')


def clean_single_trip(df):

    WINDOW_LENGTH_SPEED = 50
    WINDOW_LENGTH_SLOPE = 50

    important_cols = ['y','x','e','t']
    if set(important_cols) - set(df.columns):
        return pd.DataFrame()
    df = df[important_cols].dropna().reset_index(drop=True)
    df.rename(columns={'y':'lat', 'x':'lon', 'e':'elevation', 't':'time'}, inplace=True)

    # Calculate distance
    offset = df.iloc[1:][['lat', 'lon']].copy()
    offset = offset.append(offset.iloc[-1]).reset_index(drop=True)
    df['dist'] = mapping.dist_lat_lon(df['lat'], df['lon'],
                                      offset['lat'], offset['lon'])
    df['cum_dist'] = np.cumsum(df['dist']) # these are in miles

    # Calculate time and speed
    df['time'] = df['time'] - df['time'].min()
    dt = df['time'].diff() / 60 / 60
    dt.at[0] = dt[1]
    df['speed'] = moving_window_average(df['dist'] / dt, WINDOW_LENGTH_SPEED) # mph

    # Calculate slope
    df['slope'] = mapping.metres_to_miles(df['elevation'].diff()) / df['dist'] * 100
    df['slope'].fillna(0, inplace=True)
    df.at[df['dist'] == 0, 'slope'] = 0
    df['slope'] = moving_window_average(df['slope'], WINDOW_LENGTH_SLOPE)


    return df#.dropna()

def make_road_backbone(grid_fname, rtes_at_grid_fname, pts_per_degree):
    # PTS_PER_DEGREE = 75
    locs, tree = initialise_road_backbone_grid(pts_per_degree, 'kdTree_locs_{}'.format(pts_per_degree))

    df = pd.read_feather(config.PROCESSED_DATA_PATH + 'trips_culled.feather')

    grid_pts = pd.DataFrame(columns=['grid_id', 'lat', 'lon', 'breadcrumb_count'])
    grid_dict = collections.defaultdict(set)
    print('Looping through route IDs')
    for i, rte_id in enumerate(df['rte_id'].tolist()):
        if not i % 100: print(i)
        rte_pts = assign_trip_to_backbone_grid(rte_id, locs, tree)
        grid_pts = add_rte_info_to_grid(grid_pts, rte_pts)
        add_rte_info_to_grid_dict(grid_dict, rte_pts, rte_id)

    grid_pts['n_routes'] = [len(grid_dict[key]) for key in grid_dict]
    save_gridpts(grid_pts, grid_dict, grid_fname, rtes_at_grid_fname)
    print('Total points: {}'.format(grid_pts.shape[0]))
    # tidy_road_backbone(grid_pts, grid_dict, pts_per_degree)
    # print('Merged points: {}'.format(grid_pts.shape[0]))
    # save_gridpts(grid_pts, grid_dict, grid_fname + 'merged', rtes_at_grid_fname + 'merged')


    return grid_pts, grid_dict

def assign_trip_to_backbone_grid(rte_id, locs, tree):

    rte = pd.read_feather(config.CLEAN_TRIPS_PATH + '{}.feather'.format(rte_id))
    # Find nearest neighbours among grid points
    d, i = tree.query(rte[['lat', 'lon']])
    rte['grid_id'] = i

    grid_pts = rte.groupby('grid_id')[['lat', 'lon']].agg(['mean', 'count'])
    grid_pts.columns = ['lat', 'breadcrumb_count', 'lon', 'duplicate']

    return grid_pts[['lat', 'lon', 'breadcrumb_count']].reset_index()

def tidy_road_backbone(grid_pts, grid_dict, pts_per_degree):
    """ Merge grid points that are closer together than some threshold
    """
    MIN_DISTANCE = 1/(2 * pts_per_degree) # in degrees

    tree = sklearn.neighbors.KDTree(grid_pts[['lat', 'lon']])

    grid_pts.reset_index(inplace=True)
    print('Looping through grid points')
    for ii, row in grid_pts.iterrows():

        if not ii % 5000: print(ii)
        if row['breadcrumb_count'] == 0: # If row has already been merged, skip it
            continue

            neighbours, _ = tree.query_radius(
                row[['lat', 'lon']].values.reshape(1, -1), r=MIN_DISTANCE
            )
            neighbours = list(neighbours)
            neighbours.remove(ii) # Take the point itself out of the list

            for pt in neighbours:
                id0 = int(grid_pts.at[ii, 'grid_id'])
                id1 = int(grid_pts.at[ii, 'grid_id'])
                ct0 = int(grid_pts.at[ii, 'breadcrumb_count'])
                ct1 = int(grid_pts.at[ii, 'breadcrumb_count'])

                if ct1 == 0: # If this point has already been merged to another
                    continue

                grid_pts.at[ii, 'lat'] = (
                    (grid_pts.at[ii, 'lat'] * ct0 + grid_pts.at[pt, 'lat'] * ct1)
                    / (ct0 + ct1)
                )
                grid_pts.at[ii, 'lon'] = (
                    (grid_pts.at[ii, 'lon'] * ct0 + grid_pts.at[pt, 'lon'] * ct1)
                    / (ct0 + ct1)
                )
                grid_pts.at[ii, 'breadcrumb_count'] = ct0 + ct1
                grid_pts.at[pt, 'breadcrumb_count'] = 0 # Zero out other entry

                # Merge the dictionary entry
                grid_dict[id0] = grid_dict[id0].union(grid_dict[id1])
                grid_dict[id1] = set() # Zero out the other entry

                grid_pts.at[ii, 'n_routes'] = len(dict_m[id0])

        zeroed = [k for k, v in grid_dict.items() if not v]
        for k in zeroed:
            del grid_dict[k]

def find_gridpts_at_rte(grid_dict, gridpts_at_rte_fname, pts_per_degree):
    rte_dict = collections.defaultdict(list)
    for grid_id, rte_set in grid_dict.items():
        for rte_id in rte_set:
            rte_dict[rte_id] += [grid_id]

    gridpts_at_rte = []
    for k, v in rte_dict.items():
        gridpts_at_rte += [{'rte_id': k, 'grid_ids': list(v)}]

    gridpts_at_rte = pd.DataFrame(gridpts_at_rte)
    gridpts_at_rte.to_feather(config.PROCESSED_DATA_PATH
                            + '{}_{}.feather'.format(gridpts_at_rte_fname, pts_per_degree))
    return gridpts_at_rte

def load_rte_pts(dict_filename):
    with open(config.MODEL_PATH + dict_filename + '.json', 'r') as fn:
        rd = json.load(fn)
    route_dict = dict()
    for key in rd:
        route_dict[int(key)] = rd[key]

    return route_dict


def save_gridpts(df, setdict, df_filename, dict_filename):
    # Save files
    df.to_feather(config.MODEL_PATH + df_filename + '.feather')
    rts_at_grid = []
    for k, v in setdict.items():
        rts_at_grid += [{'grid_id': k, 'rte_ids': list(v)}]
    rts_at_grid = pd.DataFrame(rts_at_grid)
    rts_at_grid.to_feather(config.MODEL_PATH + dict_filename + '.feather')
    # df.to_csv(config.MODEL_PATH + df_filename + '.csv', index=False)
    # gd = dict.fromkeys(setdict.keys())
    # for key in gd:
    #     gd[key] = list(setdict[key])
    # with open(config.MODEL_PATH + dict_filename + '.json', 'w') as fp:
    #     json.dump(gd, fp)

def load_gridpts(grid_filename, rtes_at_grid_filename):
    grid_pts = pd.read_feather(config.MODEL_PATH + df_filename + '.feather')
    rtes_at_grid = pd.read_feather(config.MODEL_PATH + rtes_at_grid_filename + '.feather')
    # with open(config.MODEL_PATH + dict_filename + '.json', 'r') as fn:
    #     gd = json.load(fn)
    # grid_dict = collections.defaultdict(set)
    # for key in gd:
    #     grid_dict[int(key)] = set(gd[key])

    return grid_pts, rtes_at_grid


def add_rte_info_to_grid_dict(grid_dict, rte_pts, rte_id):
    for grid_pt in rte_pts['grid_id'].tolist():
        grid_dict[grid_pt].add(rte_id)


def add_rte_info_to_grid(grid_pts, rte_pts):
    """ Joined dataframes and take the weighted average of the lat/lon

    Note that it assumes any nans are from the outer join missing values ONLY
    """

    joined = pd.merge(grid_pts, rte_pts, on='grid_id', how='outer').fillna(0)
    joined['breadcrumb_count'] = joined.breadcrumb_count_x + joined.breadcrumb_count_y
    joined['lat'] = ((joined.lat_x * joined.breadcrumb_count_x
                      + joined.lat_y * joined.breadcrumb_count_y)
                    / joined['breadcrumb_count'])
    joined['lon'] = ((joined.lon_x * joined.breadcrumb_count_x
                      + joined.lon_y * joined.breadcrumb_count_y)
                    / joined['breadcrumb_count'])


    return joined[['grid_id', 'lat', 'lon', 'breadcrumb_count']]



def initialise_road_backbone_grid(pts_per_degree:int, tree_fname:str):

    LAT_RANGE = (40.5, 43.5)
    LON_RANGE = (-75.5, -72.5)

    lats = np.linspace(LAT_RANGE[0], LAT_RANGE[1],
                        pts_per_degree * int((np.ptp(LAT_RANGE))))
    lons = np.linspace(LON_RANGE[0], LON_RANGE[1],
                        pts_per_degree * int((np.ptp(LON_RANGE))))
    all_lats, all_lons = np.meshgrid(lats, lons)
    all_lats = all_lats.flatten().tolist()
    all_lons = all_lons.flatten().tolist()
    locs = pd.DataFrame({'lat': all_lats, 'lon': all_lons})

    tree = sklearn.neighbors.KDTree(locs)
    joblib.dump(tree, config.MODEL_PATH + tree_fname + '.joblib')

    return locs, tree

def convert_csv_trip_to_feather(fn):
    a = pd.read_csv(fn)

    a['rte_id'] = int(fn[:-4].split('/')[-1])
    a[['rte_id', 'time', 'lat', 'lon', 'elevation', 'dist', 'speed', 'slope']].to_feather(fn[:-4] + '.feather')
