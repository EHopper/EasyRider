# config.py
import os
import pathlib
import pandas as pd


PROJECT_PATH = '/home/emily/Documents/ViewFinder'


DATA_PATH = os.path.join(PROJECT_PATH, 'data')
RAW_DATA_PATH = os.path.join(DATA_PATH, 'raw')
RAW_RIDEWGPS_PATH = os.path.join(DATA_PATH, 'ridewgps')
RAW_ROUTES_PATH = os.path.join(RAW_DATA_PATH, 'routes')
RAW_TRIPS_PATH = os.path.join(RAW_DATA_PATH, 'trips')

CLEAN_DATA_PATH = os.path.join(DATA_PATH, 'cleaned')
CLEAN_TRIPS_PATH = os.path.join(CLEAN_DATA_PATH, 'trips')
PROCESSED_DATA_PATH = os.path.join(DATA_PATH, 'processed')

MODEL_PATH = os.path.join(PROJECT_PATH, 'models')


def save_df(df:pd.DataFrame, save_path:str, file_name:str):
    """ Save dataframes as .feather files

    Arguments:
        df
            - pd.DataFrame to save
            - Note that index must be default values!  i.e. integers in range(df.shape[0])
        save_path
            - str
            - Path to the directory to save in
                File will be saved at       save_path file_name
        file_name
            - str
            - File name to save to


    Returns:
        [save_path file_name].feather is saved to disk
    """
    pathlib.Path(save_path).mkdir(parents=True, exist_ok=True)
    df.to_feather(
        os.path.join(save_path, file_name)
    )
