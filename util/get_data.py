""" Let's get us some data
"""

import requests


def get_ridewithgps():
    URL = "http://ridewithgps.com/find/search.json"
    PARAMS = {'search[start_location]': 'Shokan, NY',
              'search[start_distance]': 25,
              }
    r = requests.get(url = URL, params = PARAMS)
    data = r.json()
    print('{} results returned'.format(data['results_count']))

    return data['results'], data['results_count']
