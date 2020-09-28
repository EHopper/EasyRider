import pandas as pd
import numpy as np

import sklearn.preprocessing

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

def apply_scaling(df):
    # Scaling is ((X - mean) / std ) * column_importance
    scaler = pd.read_csv(config.MODEL_PATH + 'feature_scaling.csv', index_col=0)

    for col in scaler.columns:
        df[col] = ((df[col] - scaler.loc['mean', col]) / scaler.loc['std', col]
                   * scaler.loc['column_importance', col])

    return df
