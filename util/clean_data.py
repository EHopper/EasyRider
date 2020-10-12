""" Clean some data
"""
import pandas as pd
import numpy as np
import pathlib
import os

from util import config
from util import mapping


def convert_ride_data_from_SI(ride_df):

    # Do unit conversion
    ride_df['distance'] = mapping.metres_to_miles(ride_df['distance'])
    ride_df['elevation_gain'] = mapping.metres_to_feet(ride_df['elevation_gain'])
    ride_df['elevation_loss'] = mapping.metres_to_feet(ride_df['elevation_loss'])

    ride_df['avg_slope'] = (mapping.feet_to_miles(ride_df['elevation_gain']
                                                  + ride_df['elevation_loss'])
                            / ride_df['distance'] * 100)

    if 'avg_speed' in ride_df.columns: # For trips, but not routes
        ride_df['avg_speed'] = mapping.km_to_mi(ride_df['avg_speed'])
        ride_df['max_speed'] = mapping.km_to_mi(ride_df['max_speed'])
        ride_df['duration'] /= (60 * 60) # seconds to hours
        ride_df['moving_time'] /= (60 * 60)


def filter_trips(ride_df):
    MIN_TIME = 0.5
    MAX_TIME = 15
    MIN_AVERAGE_SPEED = 5
    MAX_AVERAGE_SPEED = 25
    MAX_SPEED = 50
    MAX_DISTANCE = 125
    MAX_AVG_SLOPE = 12
    MAX_RATIO_DIAGONAL_TO_TOTAL_DISTANCE = 4

    # Want to remove any trips that loop around too much - these are often mountain bike routes
    ride_df['diag_distance'] = mapping.dist_lat_lon(
        ride_df.sw_lat, ride_df.sw_lng, ride_df.ne_lat, ride_df.ne_lng
    )

    return ride_df[((ride_df['is_stationary'] == False)
                    & (MIN_TIME <= ride_df['moving_time'])
                    & (ride_df['moving_time'] < MAX_TIME)
                    & (MIN_TIME <= ride_df['duration'])
                    & (ride_df['duration'] < MAX_TIME)
                    & (ride_df['distance'] < MAX_DISTANCE)
                    & (MIN_AVERAGE_SPEED <= ride_df['avg_speed'])
                    & (ride_df['avg_speed'] <= MAX_AVERAGE_SPEED)
                    & (ride_df['max_speed'] <= MAX_SPEED)
                    & (ride_df['avg_slope'] <= MAX_AVG_SLOPE)
                    & (ride_df['distance'] <= MAX_RATIO_DIAGONAL_TO_TOTAL_DISTANCE * ride_df['diag_distance'])
                    )].reset_index(drop=True)

def clean_single_trip(df_raw):
    """ Clean a single trip from the raw dataframe

    Cleaning steps:
        - Rename the columns to have clearer names, e.g. y -> lon
        - Drop nulls
        - Calculate distances between lat/lon breadcrumbs
        - Set time to be in hours, with 0 at the start of the ride
        - Calculate slope and speed at every point, and smooth these values

    Note that for this to work with 'route' data, a dummy column with fake time needs to be added to the dataframe.

    Arguments:
        df_raw
            - pd.DataFrame
            - Units:    SI (metres, seconds), degrees North and East
            - Raw trip track_points from Ride With GPS
    Returns
        df
            - pd.DataFrame
            - Units:    Imperial (miles, feet, miles per hour, hours)
                        Degrees North and East
                        Slope is given as a percentage
            - Data has had cleaning steps described above applied

    """

    WINDOW_LENGTH_SPEED = 50
    WINDOW_LENGTH_SLOPE = 50

    important_cols = ['y','x','e','t'] # Some rides have columns like heart rate, cadence
    if set(important_cols) - set(df_raw.columns):
        return pd.DataFrame()
    df = df_raw[important_cols].dropna().reset_index(drop=True)
    df.rename(columns={'y':'lat', 'x':'lon', 'e':'elevation', 't':'time'}, inplace=True)

    # Calculate distance
    offset = df.iloc[1:][['lat', 'lon']].copy()
    offset = offset.append(offset.iloc[-1]).reset_index(drop=True)
    df['dist'] = mapping.dist_lat_lon(df['lat'], df['lon'],
                                      offset['lat'], offset['lon'])

    # Calculate time and speed
    df['time'] = df['time'] - df['time'].min()
    dt = df['time'].diff() / 60 / 60
    dt.at[0] = dt[1]
    df['speed'] = moving_window_average(df['dist'] / dt, WINDOW_LENGTH_SPEED) # mph

    # Calculate slope
    df['elevation'] = mapping.metres_to_feet(df['elevation'])
    df['slope'] = mapping.feet_to_miles(df['elevation'].diff()) / df['dist'] * 100
    df['slope'].fillna(0, inplace=True)
    df.at[df['dist'] == 0, 'slope'] = 0
    df['slope'] = moving_window_average(df['slope'], WINDOW_LENGTH_SLOPE)

    return df

def moving_window_average(t_series:np.array, N:int=5):
    """ Calculate average over moving window
    Arguments:
        t_series
            - (n_points, ) np.array
            - e.g. time series data
        N
            - int
            - Default:  5
            - Number of points to average over in the moving window
    Returns:
        smoothed array
            - (n_points, ) np.array
            - Using the mode 'same', so the size will be the same as the input array - there may be odd boundary effects
    """
    return np.convolve(t_series, np.ones((N, )) / N, mode='same')
