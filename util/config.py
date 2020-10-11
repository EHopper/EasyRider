# config.py
import os

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
