"""

    Helper functions for the pretrained model to be used within our API.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Plase follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.

    Importantly, you will need to modify this file by adding
    your own data preprocessing steps within the `_preprocess_data()`
    function.
    ----------------------------------------------------------------------

    Description: This file contains several functions used to abstract aspects
    of model interaction within the API. This includes loading a model from
    file, data preprocessing, and model prediction.

"""

# Helper Dependencies
import numpy as np
import pandas as pd
import pickle
import json

def _preprocess_data(data):
    """Private helper function to preprocess data for model prediction.

    NB: If you have utilised feature engineering/selection in order to create
    your final model you will need to define the code here.


    Parameters
    ----------
    data : str
        The data payload received within POST requests sent to our API.

    Returns
    -------
    Pandas DataFrame : <class 'pandas.core.frame.DataFrame'>
        The preprocessed data, ready to be used our model for prediction.

    """
    # Convert the json string to a python dictionary object
    feature_vector_dict = json.loads(data)
    # Load the dictionary as a Pandas DataFrame.
    feature_vector_df = pd.DataFrame.from_dict([feature_vector_dict])

    # ---------------------------------------------------------------
    # drop 14 unnecessary variables
    to_drop = ['Order No',
                'User Id',
                'Vehicle Type',
                'Platform Type',
                'Placement - Day of Month',
                'Placement - Weekday (Mo = 1)',
                'Placement - Time',
                'Confirmation - Day of Month',
                'Confirmation - Weekday (Mo = 1)',
                'Confirmation - Time',
                'Arrival at Pickup - Day of Month',
                'Arrival at Pickup - Weekday (Mo = 1)',
                'Arrival at Pickup - Time',
                'Rider Id']

    df = feature_vector_df.copy()
    df.drop(to_drop, axis = 1, inplace = True)

    #encode categorical variables
    df.loc[df['Personal or Business'] == 'Personal', 'Personal or Business'] = 1
    df.loc[df['Personal or Business'] == 'Business', 'Personal or Business'] = 0

    # converting object data types for Pickup Times to date_time
    df['Pickup - Time'] = pd.to_datetime(df['Pickup - Time'])
    df['Pickup - Time'] = df['Pickup - Time'].apply(lambda time: time.hour)

    # fill precipitation null values with 0
    df['Precipitation in millimeters'].fillna(0,inplace=True)

    # impute temperature values by mean
    df.Temperature.fillna(23.25,inplace=True)

    predict_vector = df
    # ------------------------------------------------------------------------

    return predict_vector

def load_model(path_to_model:str):
    """Adapter function to load our pretrained model into memory.

    Parameters
    ----------
    path_to_model : str
        The relative path to the model weights/schema to load.
        Note that unless another file format is used, this needs to be a
        .pkl file.

    Returns
    -------
    <class: sklearn.estimator>
        The pretrained model loaded into memory.

    """
    return pickle.load(open(path_to_model, 'rb'))

def make_prediction(data, model):
    """Prepare request data for model prediciton.

    Parameters
    ----------
    data : str
        The data payload received within POST requests sent to our API.
    model : <class: sklearn.estimator>
        An sklearn model object.

    Returns
    -------
    list
        A 1-D python list containing the model prediction.

    """
    # Data preprocessing.
    prep_data = _preprocess_data(data)
    # Perform prediction with model and preprocessed data.
    prediction = model.predict(prep_data)
    # Format as list for output standerdisation.
    return prediction[0].tolist()
