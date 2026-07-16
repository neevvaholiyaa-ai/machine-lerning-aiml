import pandas as pd
import numpy as np
import joblib
import os

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")

def load_artifact(name):
    return joblib.load(os.path.join(MODELS_DIR, name))

def preprocess_input(key, input_df):
    if key == "df1":
        return input_df.values
    elif key == "df2":
        return input_df.values
    elif key == "df3":
        return input_df.values
    elif key == "df4":
        return input_df.values
    elif key == "df5":
        return input_df.values
    elif key == "df6":
        return input_df.values
    elif key == "df7":
        return input_df.values
    elif key == "df8":
        return input_df.values
    return input_df.values
