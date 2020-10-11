""" Let's get us some data
"""

import pathlib
import requests
import pandas as pd
import time
import os

from util import config
from util import mapping


def search_ridewithgps(start_location:str = 'Shokan, NY', distance_from_start:float = 25.) -> list:
    """ Call the Ride with GPS API in a loop to get all results fitting search parameters

    Arguments:
        start_location
            - str
            - String with place name on which to centre search in API call
            - Default value: 'Shokan, NY'
        distance_from_start
            - float
            - Units:    miles
            - Radius around 'start_location' in which to search for rides
            - Default value: 25.

    Returns:
        ridewgps_data
            - list of dictionaries
            - Units:    SI, e.g. distance in metres
            - Each item in list is a dictionary e.g.
                {'type': 'trip', 'trip': {all of the trip info as a dictionary}}
                - Note that the key of the second entry is the value of the first entry
                - Can either be 'route' or 'trip', where a 'trip' is a ride recorded by GPS and uploaded; a 'route' is a ride planned on their website

    """

    # Can only return 300 rides at a time from the API, though 'total_results' gives the total number of results that fits the search criteria
    ridewgps_data, total_results = ridewithgps_api_search(
        start_location=start_location, distance_from_start=distance_from_start,
        length_min=1., # shorter rides are just people testing their GPS, probably
        max_results=300
    )

    results_remaining = total_results - len(ridewgps_data)
    while True:
        print('{} of {} results collected'.format(total_results - results_remaining, total_results))
        d_more, _ = ridewithgps_api_search(
            start_location=start_location, distance_from_start=distance_from_start,
            length_min=mapping.metres_to_miles(ridewgps_data[-1][ridewgps_data[-1]['type']]['distance']), # Results returned in order of ascending length
            max_results=min(results_remaining, 300)
        )
        ridewgps_data += d_more
        results_remaining -= len(d_more)

        if results_remaining <= 0 or not d_more:
            break

    return ridewgps_data

def ridewithgps_api_search(keywords:str='', start_location:str='Portland, OR',
                           distance_from_start:float=15., elevation_max:float=200000.,
                           elevation_min:float=0., length_max:float=1200.,
                           length_min:float=0., offset:int=0, max_results:int=20,
                           sort_by:str='length asc',
                           ) -> (list, int):
    """ Retrieve Ride with GPS listings

    Note that default values of the parameters are defaults for API - i.e. if enter a blank string to these fields, these are the values that are used.

    Arguments:
        keywords
            - str
        start_location
            - str
        distance_from_start
            - float
            - Units: miles
        elevation_max
            - float
            - Units: feet
        elevation_min
            - float
            - Units: feet
        length_max
            - float
            - Units: miles
        length_min
            - float
            - Units: miles
        offset
            - int?
            - Not sure what this does, tbh
        max_results
            - int
            - Maximum results returned by API, capped at 300
        sort_by
            - str
            - How to sort the returned results
            - Options: 'length asc', ??? (presumably 'length desc'!)
    Returns
        - data['results']
            - list
            - Units:    distance in metres
            - Each item in list is a dictionary e.g.
                {'type': 'route', 'route': {all of the route info as a dictionary}}
                - Note that the key of the second entry is the value of the first entry
        - data['results_count']
            - int
            - Total possible listings that fit the search query
            - Note that number actually returned is controlled by 'max_results'

    """
    if max_results > 300:
        print('Max results is capped at 300!  Adjusting your search parameters')
        max_results = 300

    URL = "http://ridewithgps.com/find/search.json"
    PARAMS = {'search[keywords]': keywords,
              'search[start_location]': start_location,
              'search[start_distance]': distance_from_start,
              'search[elevation_max]': elevation_max,
              'search[elevation_min]': elevation_min,
              'search[length_max]': length_max,
              'search[length_min]': length_min,
              'search[offset]': offset,
              'search[limit]': max_results,
              'search[sort_by]': sort_by,
              }

    r = requests.get(url = URL, params = PARAMS)

    data = r.json()

    return data['results'], data['results_count']


def parse_ridewithgps_search_results(ridewgps_data:list, ifsave:bool=True) -> (pd.DataFrame, pd.DataFrame):
    """ Turn RideWithGPS API results into pandas DataFrames

    Arguments:
        ridewgps_data
            - list of dictionaries
            - Units:    SI
            - Each item in list is a dictionary e.g.
                {'type': 'trip', 'trip': {all of the ride info as a dictionary}}
                - Note that the key of the second entry is the value of the first entry
                - This can either be 'trip' or 'route'
            - Ride info keys (trips and routes) include
                'id', 'created_at', 'name', 'description', 'user_id', 'highlighted_photo_id', 'distance', 'elevation_gain', 'first_lat', 'first_lng', 'last_lat', 'last_lng', 'sw_lat', 'sw_lng', 'ne_lat', 'ne_lng'
            - Trip info keys include
                'avg_cad', 'avg_hr', 'avg_power_estimated', 'avg_speed', 'avg_watts', 'calories', 'departed_at', 'duration', 'gear_id', 'is_gps', 'is_stationary', 'max_cad', 'max_hr', 'max_speed', 'max_watts', 'min_cad', 'min_hr', 'min_watts', 'moving_time', 'source_type', 'time_zone'
            - Route info keys include
                'best_for_id', 'has_course_points', 'pavement_type_id'
        ifsave
            - bool
            - Default:  True
            - If True, save the outputs to disk:
                config.RAW_DATA_PATH + 'ridewgps_[routes|trips].feather'

    Returns:
        routes
            - pd.DataFrame
            - Units:    SI
            - Columns are made up of ride and route info keys, as detailed above
        trips
            - pd.DataFrame
            - Units:    SI
            - Columns are made up of ride and trip info keys, as detailed above

        If ifsave, these dataframes are saved to disk at
            config.RAW_DATA_PATH ridewgps_[trips|routes].feather

    """

    routes = pd.DataFrame([dd['route'] for dd in ridewgps_data if 'route' in dd])
    trips = pd.DataFrame([dd['trip'] for dd in ridewgps_data if 'trip' in dd])

    if ifsave:
        pathlib.Path(config.RAW_DATA_PATH).mkdir(parents=True, exist_ok=True)
        trips.to_feather(os.path.join(config.RAW_DATA_PATH, 'ridewgps_trips.feather'))
        routes.to_feather(os.path.join(config.RAW_DATA_PATH, 'ridewgps_routes.feather'))

    return routes, trips


def ridewithgps_api_ride(rte_id:int, ride_type:str, ifsave:bool=True) -> pd.DataFrame:
    """ Get individual ride from Ride With GPS API and save to disk

    Arguments:
        rte_id
            - int
            - Route ID, as in ridewithgps.com/[ride_type]/[rte_id]
        ride_type:
            - str
            - This should be either 'trips' or 'routes'
        ifsave
            - bool
            - Default:  True
            - If True, save the ride to
                config.RAW_ROUTES_PATH + [rte_id].feather
              and the whole binary response to
                config.RAW_RIDEWGPS_PATH + route_[rte_id]

    Returns:
        rte_id
            - pd.DataFrame
            - Units:    SI, degrees North and East
            - The 'track_points' of the ride are extracted and returned
            - A lot of metadata is discarded, but this commonly has a lot of Nulls
            - Columns include
                'e'     elevation in metres
                'x'     longitude in degrees E
                'y'     latitude in degrees N
                'd'     distance in metres
                't'     time stamp for each breadcrumb
                            Note: t is only in trips, not routes

        If ifsave, the ride is saved to disk

    """
    if not isinstance(rte_id, int):
        print('Ride ID must be an integer!')
        return

    if ride_type == 'routes':
        save_dir = config.RAW_ROUTES_PATH
    elif ride_type == 'trips':
        save_dir = config.RAW_TRIPS_PATH
    else:
        print('You must specify either \'routes\' or \'trips\' for ride_type')
        return

    if os.path.exists(os.path.join(save_dir, '{}.feather'.format(rte_id))):
        # If have already downloaded ride, just load from disk
        return pd.read_feather(os.path.join(save_dir, '{}.feather'.format(rte_id)))

    URL = "http://ridewithgps.com/{}/{}.json".format(ride_type, rte_id)
    r = requests.get(url = URL)
    d = r.json()
    if 'track_points' not in d:
        return

    rte_df = pd.DataFrame(d['track_points'])

    if ifsave:
        write_ridewithgps_request(r, '{}_{}'.format(ride_type, rte_id))

        pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)
        rte_df.to_feather(os.path.join(save_dir, '{}.feather'.format(rte_id)))

    return rte_df

def write_ridewithgps_request(data, fname:str):
    """ Save whole Ride With GPS API request file to disk

    Arguments:
        data
            - Response object from requests.get()
        fname
            - str
            - Filename at which to save file
                config.RAW_RIDEWGPS_PATH + fname

    Returns:
        Data is saved to disk as a binary file

    """

    pathlib.Path(config.RAW_RIDEWGPS_PATH).mkdir(parents=True, exist_ok=True)
    fname = os.path.join(config.RAW_RIDEWGPS_PATH, fname)
    with open(fname, mode='wb') as localfile:
        localfile.write(data.content)
