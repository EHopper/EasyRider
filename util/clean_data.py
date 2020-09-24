""" Clean some data
"""
import pandas as pd
import numpy as np
import datetime
import re


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

    return df[((df['is_stationary'] == False)
               & (MIN_TIME < df['moving_time'])
               & (df['moving_time'] < MAX_TIME)
               & (MIN_TIME < df['duration'])
               & (df['duration'] < MAX_TIME)
               & (df['distance'] < MAX_DISTANCE)
               & (MIN_AVERAGE_SPEED < df['avg_speed'])
               & (df['avg_speed'] < MAX_AVERAGE_SPEED)
               & (df['max_speed'] < MAX_SPEED))].reset_index(drop=True)

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
