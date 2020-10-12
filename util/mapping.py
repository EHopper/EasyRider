""" Useful functions for dealing with geographic unit conversions.
"""

import numpy as np

EARTH_RADIUS = 6371 # km

# Unit conversions
def metres_to_miles(dist:float) -> float:
    """ Convert metres to miles (1609.3 metres to the mile) """
    return dist / 1609.34

def metres_to_feet(dist:float) -> float:
    """ Convert metres to feet (3.281 feet to the metre) """
    return dist * 3.281

def km_to_mi(dist:float) -> float:
    """ Convert kilometres to miles (1.6093 km to the mile)"""
    return metres_to_miles(dist) * 1e3

def feet_to_miles(dist:float) -> float:
    """ Convert feet to miles (5280 feet to the mile)"""
    return dist / 5280

def mps_to_mph(speed:float) -> float:
    """ Convert metres per second to miles per hour """
    return metres_to_miles(speed) * 60 * 60

def deg_to_rad(phi):
    return phi * np.pi / 180

def dist_lat_lon(lat0, lon0, lat1, lon1):
    """ Distance calculator between lat/lon points

    Note that this is approximate, e.g. assumes a spherical Earth.

    Equations taken from
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

    Arguments:
        lat0
            - float
            - Units:    degrees N
            - Latitude of first point
        lon0
            - float
            - Units:    degrees E
            - Longitude of first point
        lat1
            - float
            - Units:    degrees N
            - Latitude of second point
        lon1
            - float
            - Units:    degrees E
            - Longitude of second point

    Returns
        distance between the two points (best here)
            - float
            - Units:    miles
    """

    radius = km_to_mi(EARTH_RADIUS) # output in miles
    d_lat = deg_to_rad(lat0 - lat1)
    d_lon = deg_to_rad(lon0 - lon1)
    lat0 = deg_to_rad(lat0)
    lat1 = deg_to_rad(lat1)

    a = (np.sin(d_lat / 2) ** 2
        + np.cos(lat0) * np.cos(lat1) * np.sin(d_lon / 2) ** 2)
    return radius * 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))

def approx_dist_lat_lon(lat0:float, lon0:float, lat1:float, lon1:float) -> float:
    """ Give more approximate distance between two lat/lon pairs

    Equations taken from
        http://www.movable-type.co.uk/scripts/latlong.html
    const x = (λ2-λ1) * Math.cos((φ1+φ2)/2);
    const y = (φ2-φ1);
    const d = Math.sqrt(x*x + y*y) * R;

    Arguments:
        lat0
            - float
            - Units:    degrees N
            - Latitude of first point
        lon0
            - float
            - Units:    degrees E
            - Longitude of first point
        lat1
            - float
            - Units:    degrees N
            - Latitude of second point
        lon1
            - float
            - Units:    degrees E
            - Longitude of second point

    Returns
        distance between the two points (approximate)
            - float
            - Units:    miles
    """

    lat0 = deg_to_rad(lat0)
    lon0 = deg_to_rad(lon0)
    lat1 = deg_to_rad(lat1)
    lon1 = deg_to_rad(lon1)

    x = (lon1 - lon0) * np.cos((lat0 + lat1) / 2)
    y = lat1 - lat0

    return km_to_mi(np.sqrt(x ** 2 + y ** 2) * EARTH_RADIUS)

def degrees_to_miles_ish(distance_degrees:float, typical_lat:float=42.) -> float:
    """ Convert degrees lat/lon to miles, very approximately

    Anywhere not on the Equator, 1 degree latitude is longer than 1 degree longitude.  If we just want a really rough approximation, we can convert degrees to miles by taking the average miles per degree, both in latitude and longitude.

    Arguments:
        distance_degrees
            - float
            - Units:    degrees (some combination of latitude and longitude)
            - Distance between two lat/lon points as if in Cartesian space
        typical_lat
            - float
            - Units:    degrees North
            - Default:  42.
            - Average latitude for your area of interest - which is about 42.N for the Catskills

    Returns:
        very approximate (but fast!) conversion to miles
            - float
            - Units:    miles (ish)

    """

    deg_lat = dist_lat_lon(typical_lat - 0.5, 0, typical_lat + 0.5, 0)
    deg_lon = dist_lat_lon(typical_lat, 0, typical_lat, 1)

    return distance_degrees * np.mean((deg_lat, deg_lon))
