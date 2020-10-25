""" Useful functions for dealing with geographic unit conversions.

Also for plotting using pydeck.
"""
import os
import numpy as np
import pandas as pd
import pydeck as pdk
import seaborn as sns

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

def plot_rides(rte_ids:list, grid_pts:pd.DataFrame, gridpts_at_rte:pd.DataFrame,
               start_locs:tuple=(0, 0, 0)):
    """ Return a pydeck deck with the given rides plotted on it

    This deck can be plotted using the deck.to_html('save_name.html') command.

    Arguments:
        rte_ids
            - list of str
            - Unique ride IDs, used to identify ride in gridpts_at_rte
        grid_pts
            - pd.DataFrame
            - Units: degrees N and degrees E
            - Index: grid IDs
            - Column names include
                lat       - latitude (degrees N) of grid point
                lon       - longitude (degrees E) of grid point
        gridpts_at_rte
            - pd.DataFrame
            - All values are integers (ID keys)
            - Index: ride IDs
            - Column name:
                grid_ids  - for each ride, a list of the grid IDs it passes through
        start_locs
            - tuple of start_lat, start_lon, start_location_yn
            - Units:  degrees N and degrees E
            - If there is a start location to plot, this gives its coordinates
            - start_location_yn is a Boolean, as there may be no start location to plot
            - Default is (0, 0, 0) - i.e. will not plot a start location as start_location_yn = 0


    Returns:
        A pydeck deck with the rides (and start location) within it as ScatterplotLayers
            - Initial view state will be centred on the input ride locations
    """
    n_rides = len(rte_ids)
    start_lat, start_lon, start_location_yn = start_locs

    # Record map limits for the purpose of sizing the output map
    # min latitude, max latitude, min longitude, max longitude
    map_lims = np.zeros((len(rte_ids), 4))

    ride_layers = []
    colours = sns.color_palette('husl', n_rides)
    for i, rte_id in enumerate(rte_ids):
        # Load each ride into a pydeck ScatterplotLayer
        lats, lons = find_rte_coordinates(rte_id, gridpts_at_rte, grid_pts)
        ride_layers += [lat_lon_pts_to_map_layer(
            lats, lons, [str(rte_id)], [int(x * 255) for x in colours[i % n_rides]], 1
        )]
        map_lims[i, :] = np.array([min(lats), max(lats), min(lons), max(lons)])

    # If there is a start location, plot this as a red dot
    if start_location_yn:
        map_lims = np.vstack(
            (map_lims, np.array([[start_lat, start_lat, start_lon, start_lon]]))
        )
        ride_layers += [lat_lon_pts_to_map_layer(
            [start_lon], [start_lat], ['start'], [255, 0, 0], 3
        )]

    # Find actual limits of the map
    max_lat, max_lon = tuple(map_lims[:, 1:4:2].max(axis=0))
    min_lat, min_lon = tuple(map_lims[:, 0:3:2].min(axis=0))

    return map_layers_to_map(
        ride_layers, (min_lat, max_lat, min_lon, max_lon), "<b>Ride {name}</b>"
    )

def map_layers_to_map(map_layers:list, loc_lims:tuple, tip_label:str):
    """ Make a pydeck Deck object containing the given map layers

    Arguments:
        map_layers
            - list of pydeck layers
            - All of these layers will be added to the pydeck Deck
        loc_lims
            - tuple of (min_lat, max_lat, min_lon, max_lon)
            - Used to define the initial view state
        tip_label
            - str
            - The label that appears on mouse hover
            - Can either be uniform or involve some column that is in all of the data DataFrames in every layer passed to the function, e.g. "{name}"

    Returns:
        A pydeck deck with the passed map_layers included
            - Initial view state will be centred on the loc_lims

    """
    min_lat, max_lat, min_lon, max_lon = loc_lims

    return pdk.Deck(
        map_style="mapbox://styles/mapbox/light-v10",
        initial_view_state=pdk.data_utils.compute_view([
            [max_lon + 0.25, max_lat + 0.25],
            [max_lon + 0.25, min_lat - 0.25],
            [min_lon - 0.25, max_lat + 0.25],
            [min_lon - 0.25, min_lat - 0.25],
        ]),
        layers=map_layers,
        tooltip={"html": tip_label, "style": {"color": "white"}},
        mapbox_key=os.getenv('MAPBOX_API_KEY')

    )


def find_rte_coordinates(rte_id, gridpts_at_rte, grid_pts):
    """ Retrive latitude and longitude points of a ride

    Note that these are not ordered!  These are the lat/lons of the grid points passed through by the ride in an arbitary order and without repetition.  That is, they only make sense as dots, not as a line.

    Arguments:
        rte_id
            - str
            - Unique ride ID, used to identify ride in gridpts_at_rte
        gridpts_at_rte
            - pd.DataFrame
            - All values are integers (ID keys)
            - Index: ride IDs
            - Column name:
                grid_ids
                  - For each ride, a list of the grid IDs it passes through
        grid_pts
            - pd.DataFrame
            - Units: degrees N and degrees E
            - Index: grid IDs
            - Column names include
                lat
                    - latitude (degrees N) of grid point
                lon
                    - longitude (degrees E) of grid point

    Returns:
        A tuple of lists, (lats, lons)
            lats
                - list of floats
                - Units: degrees N
                - Relevant grid point latitudes
            lons
                - list of floats
                - Units: degrees E
                - Relevant grid point longitudes


    """
    # Find grid points for that ride
    grid_at_rte_id = gridpts_at_rte.loc[rte_id, 'grid_ids']

    return (grid_pts.loc[grid_at_rte_id].lat.tolist(),
            grid_pts.loc[grid_at_rte_id].lon.tolist())


def lat_lon_pts_to_map_layer(lats, lons, labels, colour:tuple=(255, 0, 0), size:int=1):
    """ Make a pydeck ScatterplotLayer for lat/lon points.

    Arguments:
        lats
            - list of floats
            - Units: degrees N
            - Latitudes of points to add to pydeck layer
        lons
            - list of floats
            - Units: degrees E
            - Longitudes of points to add to pydeck layer
        labels
            - list of strings
            - Label for each of the passed points, for the mouse-hover
            - If list is length 1, will give every point the same label
        colour
            - tuple of length 3 - (R, G, B)
            - Colour to plot all of these points in RGB space
            - Default value: (255, 0, 0)   i.e. red
        size
            - int
            - Marker size of each point on the map

    Returns:
        pydeck ScatterplotLayer with filled circles at each passed location

    """
    if len(labels) == 1:  # Ensure there is a label for every point passed
        labels = labels * len(lats)

    return pdk.Layer(
            type="ScatterplotLayer",
            data=pd.DataFrame({'name': labels,
                               'coordinates': [
                                    [lon, lat] for lon, lat in zip(lons, lats)]
                             }),
            opacity= 1,
            pickable=True,
            filled=True,
            stroked=False,
            radius_min_pixels=size,
            get_position="coordinates",
            get_fill_color=colour,
            get_size=size,
    )
