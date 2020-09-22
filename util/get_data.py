""" Let's get us some data
"""

import requests
import pandas as pd
import datetime
import glob
import os


def search_ridewithgps(start_location:str = 'Shokan, NY', distance_from_start:float = 25.):
    """ Call the Ride with GPS API in a loop to get all results fitting search parameters
    """

    d, total_results = ridewithgps_api_search(
        start_location=start_location, distance_from_start=distance_from_start,
        length_min=1., # shorter rides are just people testing their GPS, probably
        max_results=300
    )

    results_remaining = total_results - len(d)
    while True:
        print('{} of {} results collected'.format(total_results - results_remaining, total_results))
        d_more, n = ridewithgps_api_search(
            start_location=start_location, distance_from_start=distance_from_start,
            length_min=metres_to_miles(d[-1][d[-1]['type']]['distance']),
            max_results=min(results_remaining, 300)
        )
        d += d_more
        results_remaining -= len(d_more)
        if results_remaining <= 0 or not d_more:
            break


    return d

def parse_ridewithgps_search_results(ridewgps_search):

    routes_list = [dd['route'] for dd in ridewgps_search if 'route' in dd]
    trips_list = [dd['trip'] for dd in ridewgps_search if 'trip' in dd]

    return pd.DataFrame(trips_list), pd.DataFrame(routes_list)


def metres_to_miles(dist:float):
    """ Convert metres to miles (1609.3 metres to the mile) """
    return dist / 1609.34

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

    raw_data_path = '/home/emily/Documents/ViewFinder/data/raw/'
    if not os.path.isdir(raw_data_path + 'ridewgps/'):
        os.mkdir(raw_data_path + 'ridewgps/')
    filename = '{}ridewgps/rwg_{}_'.format(raw_data_path,
                                           datetime.date.today().strftime("%Y-%m-%d"))
    fns = glob.glob(filename + '*')
    filename += str(len(fns))
    with open(filename, mode='wb') as localfile:
        localfile.write(r.content)
    data = r.json()

    return data['results'], data['results_count']
