import pandas as pd
import numpy as np

import sklearn.preprocessing
import sklearn.neighbors

from util import config
from util import mapping

def set_presets():
    df = pd.read_csv(config.PROCESSED_DATA_PATH + 'trips.csv')

    presets_descriptions = [
        'Chilling out in the saddle', 'Pretty relaxed, with some climbing',
        'Half-day of touring', 'Training for VO2-max', 'Training for strength',
        'Training for a century']

    presets = pd.DataFrame({
        'distance': [10., 15., 45., 20., 10., 85.],
        'avg_slope': [2., 5., 3., 2., 8., 2.],
        'avg_speed': [10., 8., 15., 18., 8., 15.],
        'prop_moving': [0.7, 0.7, 0.8, 1., 0.7, 0.8]
    })

    return presets, presets_descriptions

def scale_dataset(df, column_importance):
    # Scaling is ((X - mean) / std ) * column_importance
    scaler = sklearn.preprocessing.StandardScaler()
    cols = df.columns
    df[cols] = scaler.fit_transform(df[cols])
    df[cols] *= column_importance

    scaler_df = pd.DataFrame(np.vstack((scaler.mean_, scaler.scale_, column_importance)),
                     columns=cols, index=['mean', 'std', 'column_importance'])
    scaler_df.to_csv(config.MODEL_PATH + 'feature_scaling.csv')

    return df

def remove_scaling(df):
    # Scaling is ((X - mean) / std ) * column_importance
    scaler = pd.read_csv(config.MODEL_PATH + 'feature_scaling.csv', index_col=0)
    df_unscale = df.copy()
    for col in scaler.columns:
        df_unscale[col] = (df_unscale[col] / scaler.loc['column_importance', col]
                    * scaler.loc['std', col] + scaler.loc['mean', col])

    return df_unscale

def apply_scaling(df):
    # Scaling is ((X - mean) / std ) * column_importance
    scaler = pd.read_csv(config.MODEL_PATH + 'feature_scaling.csv', index_col=0)
    df_scale = df.copy()
    for col in scaler.columns:
        df_scale[col] = ((df_scale[col] - scaler.loc['mean', col])
                            / scaler.loc['std', col]
                            * scaler.loc['column_importance', col])

    return df_scale

def calc_distance_from_start(lat, lon):

    grid_pts, grid_dict = clean_data.load_gridpts(
        'road_backbone_merged', 'grid_rte_ids_merged'
    )

    tree = sklearn.neighbors.KDTree(grid_pts[['lat', 'lon']])
