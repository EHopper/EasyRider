""" Make road backbone using CLIQUE clustering
"""

import os
import pathlib
import collections
import pandas as pd
import numpy as np

import sklearn.neighbors

from util import config
from util import mapping



def make_road_backbone(rte_ids, grid_fname, rtes_at_grid_fname, pts_per_degree):

    LAT_RANGE = (40.5, 43.5)
    LON_RANGE = (-75.5, -72.5)

    tree = initialise_road_backbone_grid(pts_per_degree, LAT_RANGE, LON_RANGE)

    grid_pts = pd.DataFrame(columns=['grid_id', 'lat', 'lon', 'breadcrumb_count'])
    grid_dict = collections.defaultdict(set)
    print('Looping through route IDs')
    for i, rte_id in enumerate(rte_ids):
        if not i % 100: print(i)
        rte_pts = assign_trip_to_backbone_grid(rte_id, tree)
        grid_pts = add_rte_info_to_grid(grid_pts, rte_pts)
        add_rte_info_to_grid_dict(grid_dict, rte_pts, rte_id)

    # Count the number of rides going through each grid point (location popularity)
    grid_pts['n_routes'] = [len(grid_dict[key]) for key in grid_dict]

    print('Total unique grid points: {}'.format(grid_pts.shape[0]))
    # tidy_road_backbone(grid_pts, grid_dict, pts_per_degree)
    # print('Merged points: {}'.format(grid_pts.shape[0]))
    # save_gridpts(grid_pts, grid_dict, grid_fname + 'merged', rtes_at_grid_fname + 'merged')


    return grid_pts, grid_dict

def initialise_road_backbone_grid(pts_per_degree:int, lat_range:tuple, lon_range:tuple
                                 ) -> (sklearn.neighbors._kd_tree.KDTree, pd.DataFrame):
    """ Generate an initial backbone grid, and fit a KD-Tree

    Start the CLIQUE clustering algorithm with some grid points that are evenly spaced in latitude and longitude.  Fit a KD-Tree to speed up finding the grid points that are nearest neighbours of the ride lat/lons.

    These grid points will later be adjusted to be at the centre of all of their assigned lat/lon breadcrumbs.

    Arguments:
        pts_per_degree
            - int
            - Units:    1 / degree (North and East treated as if equal)
            - Approximate grid spacing will be 1/pts_per_degree degrees
        lat_range
            - tuple
            - Units:    degrees N
            - (minimum latitude of interest, maximum latitude of interest)
        lon_range
            - tuple
            - Units:    degrees E
            - (minimum longitude of interest, maximum longitude of interest)

    Returns
        KD-Tree for the grid
            - sklearn.neighbors._kd_tree.KDTree
            - Units:    degrees (ish)
            - Note that this tree assumes that degrees North and degrees East are the same size - so will tend to bias towards points that are along a N-S axis
                - That is, 0.1 degrees difference N-S will be treated as if equal to 0.1 degrees difference E-W, even though it is about 35% further at the latitude of the Catskills
                - This will distort our final grid, but not enough to be problematic for the use case

    """

    min_lat, max_lat = lat_range
    min_lon, max_lon = lon_range
    lats = np.linspace(min_lat, max_lat, int(pts_per_degree * np.ptp(lat_range)))
    lons = np.linspace(min_lon, max_lon, int(pts_per_degree * np.ptp(lon_range)))

    all_lats, all_lons = np.meshgrid(lats, lons)
    all_lats = all_lats.flatten().tolist()
    all_lons = all_lons.flatten().tolist()
    # Note: the index of this DataFrame corresponds to 'grid_id', used later
    locs = pd.DataFrame({'lat': all_lats, 'lon': all_lons})

    return sklearn.neighbors.KDTree(locs)

def assign_trip_to_backbone_grid(rte_id:str, tree:sklearn.neighbors._kd_tree.KDTree) -> pd.DataFrame:
    """ Load in a single ride and find the grid locations it clusters to.

    Given a pre-calculated KD-Tree, find the indices of the tree's input grid that are closest to each of the ride's lat/lon breadcrumb points.

    Note that the returned ride grid points are unique - that is, if a ride doubles back through a point, it will still only appear once.  This is deliberate from the perspective that (especially for a coarse grid) the number of associated grid points should be minimised for latency and storage reasons.  However, this means that the path information (i.e. order of points travelled through) is not preserved.

    Arguments:
        rte_id
            - str
            - Ride ID, used to identify file to load -
                config.CLEAN_TRIPS_PATH [rte_id].feather
        tree
            - sklearn.neighbors._kd_tree.KDTree
            - Units:    degrees (ish)
            - Note: distances are approximate, as degrees North and degrees East are treated as equal

    Returns
        grid_pts
            - pd.DataFrame
            - Units:    degrees N, degrees E, integer count
            - The grid points travelled through by the route, where
                - 'grid_id': primary key of each grid point included in this ride
                - 'lat': the average latitude of each gridpoint from this ride
                - 'lon': the average longitude of each gridpoint from this ride
                - 'breadcrumb_count': the number of lat/lon breadcrumbs at each grid point from this ride
                    - Note: these are not necessarily contiguous

    """
    rte = pd.read_feather(
        os.path.join(config.CLEAN_TRIPS_PATH, '{}.feather'.format(rte_id))
    )

    # Find nearest neighbours among grid points
    d, grid_ind = tree.query(rte[['lat', 'lon']])
    rte['grid_id'] = grid_ind

    grid_pts = rte.groupby('grid_id')[['lat', 'lon']].agg(['mean', 'count'])
    grid_pts.columns = ['lat', 'breadcrumb_count', 'lon', 'duplicate_count']

    return grid_pts[['lat', 'lon', 'breadcrumb_count']].reset_index()


def add_rte_info_to_grid(grid_pts:pd.DataFrame, rte_pts:pd.DataFrame) -> pd.DataFrame:
    """ Update the grid_pts by incorporating data from a new ride.

    Given a DataFrame with the current location and breadcrumb_count of all grid points, add in the new ride information by incorporating into a weighted average value for the latitude and longitude.

    Note that this code assumes any nans are from the outer join unmatched rows ONLY, so they can safely be filled with zeros (as breadcrumb_count = 0 means they will not contribute to the weighted means).

    Arguments:
        grid_pts
            - pd.DataFrame
            - Units:    degrees N, degrees E, integer count
            - Current state of knowledge about the grid backbone
                - That is, average location in lat/lon of the cluster centre weighted by the number of lat/lon breadcrumbs from each contributing ride
            - Columns: 'grid_id', 'lat', 'lon', 'breadcrumb_count'
        rte_pts:
            - pd.DataFrame
            - Units:    degrees N, degrees E, integer count
            - Grid backbone information from a single ride
            - Columns: 'grid_id', 'lat', 'lon', 'breadcrumb_count'

    Returns:
        joined
            - pd.DataFrame
            - Units:    degrees N, degrees E, integer count
            - Updated state of knowledge about the grid backbone
            - Taken as an outer join of the two input DataFrames, where
                - 'grid_id': column on which to join
                - 'lat', 'lon': average of the two input dataframes, weighted by 'breadcrumb_count'
                - 'breadcrumb_count': sum of the two input dataframes
    """

    joined = pd.merge(grid_pts, rte_pts, on='grid_id', how='outer').fillna(0)
    joined['breadcrumb_count'] = joined.breadcrumb_count_x + joined.breadcrumb_count_y
    # Take weighted average for latitude and longitude of grid point
    joined['lat'] = ((joined.lat_x * joined.breadcrumb_count_x
                      + joined.lat_y * joined.breadcrumb_count_y)
                    / joined['breadcrumb_count'])
    joined['lon'] = ((joined.lon_x * joined.breadcrumb_count_x
                      + joined.lon_y * joined.breadcrumb_count_y)
                    / joined['breadcrumb_count'])


    return joined[['grid_id', 'lat', 'lon', 'breadcrumb_count']]

def add_rte_info_to_grid_dict(grid_dict:collections.defaultdict,
                              rte_pts:pd.DataFrame, rte_id:int):
    """ Add the ride ID to the list of ride IDs at each relevant grid point

    Arguments:
        grid_dict
            - collections.defaultdict (default value: empty set)
            - Keys in this dictionary are grid_ids (integers)
            - Values are a set of ride IDs (integers) for each grid point
        rte_pts
            - pd.DataFrame
            - Columns of interest here is only 'grid_id', where each row corresponds to a unique grid_id that this ride passes through
        rte_id
            - int
            - The unique ride identifier

    Returns:
        grid_dict is updated in place
    """
    for grid_id in rte_pts['grid_id'].tolist():
        grid_dict[grid_id].add(rte_id)

def find_gridpts_at_rte(grid_dict, gridpts_at_rte_fname, pts_per_degree):

    
    rte_dict = collections.defaultdict(list)
    for grid_id, rte_set in grid_dict.items():
        for rte_id in rte_set:
            rte_dict[rte_id] += [grid_id]

    gridpts_at_rte = []
    for k, v in rte_dict.items():
        gridpts_at_rte += [{'rte_id': k, 'grid_ids': list(v)}]

    gridpts_at_rte = pd.DataFrame(gridpts_at_rte)

    return gridpts_at_rte



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
