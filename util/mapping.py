""" Useful functions for dealing with maps and locations
"""

import numpy as np
import math
import osmium

EARTH_RADIUS = 6371 # km
# Unit conversions
def metres_to_miles(dist:float):
    """ Convert metres to miles (1609.3 metres to the mile) """
    return dist / 1609.34

def metres_to_feet(dist:float):
    return dist * 3.281

def km_to_mi(dist:float):
    return metres_to_miles(dist) * 1e3

def approx_dist_lat_lon(lat1, lon1, lat2, lon2):
    """ Give approximate distance between two lat/lon pairs


    http://www.movable-type.co.uk/scripts/latlong.html
    const x = (λ2-λ1) * Math.cos((φ1+φ2)/2);
    const y = (φ2-φ1);
    const d = Math.sqrt(x*x + y*y) * R;
    """
    lat1 = deg_to_rad(lat1)
    lon1 = deg_to_rad(lon1)
    lat2 = deg_to_rad(lat2)
    lon2 = deg_to_rad(lon2)

    x = (lon2 - lon1) * np.cos((lat1 + lat2) / 2)
    y = lat2 - lat1

    return km_to_mi(np.sqrt(x ** 2 + y ** 2) * EARTH_RADIUS)

def deg_to_rad(phi):
    return phi * math.pi / 180

def dist_lat_lon(lat1, lon1, lat2, lon2):
    """ Better distance calculator

    http://www.movable-type.co.uk/scripts/latlong.html
    Javascript code:
    const R = 6371e3; // metres
    const φ1 = lat1 * Math.PI/180; // φ, λ in radians
    const φ2 = lat2 * Math.PI/180;
    const Δφ = (lat2-lat1) * Math.PI/180;
    const Δλ = (lon2-lon1) * Math.PI/180;

    const a = Math.sin(Δφ/2) * Math.sin(Δφ/2) +
              Math.cos(φ1) * Math.cos(φ2) *
              Math.sin(Δλ/2) * Math.sin(Δλ/2);
    const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1-a));

    const d = R * c; // in metres
    """

    radius = km_to_mi(EARTH_RADIUS) # output in miles
    d_lat = deg_to_rad(lat1 - lat2)
    d_lon = deg_to_rad(lon1 - lon2)
    lat1 = deg_to_rad(lat1)
    lat2 = deg_to_rad(lat2)

    a = (np.sin(d_lat / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(d_lon / 2) ** 2)
    return radius * 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a)) ** 2


class LatLonHandler(osmium.SimpleHandler):
    def __init__(self):
        super(LatLonHandler).__init__()
        self.latlons = []

    def node(self, n):
        self.latlons.append([o.tags['place']])
        self.num_nodes += 1
