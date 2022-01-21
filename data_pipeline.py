import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from data_fetcher import Fetcher, Do_nothing
import numpy as np
import pandas as pd


def transformation_pipeline(data, building_id=122, meter=0, primary_use=99):
    '''
    returns pipeline and data_cleaned
    '''
    fetcher = Fetcher(building_id, meter, primary_use)  # to clean the data
    data_cleaned = fetcher.transform(data)

    numerical_pipeline = Pipeline([('imputer', SimpleImputer()),  # to fill missing values with mean
                                   ('scaler', MinMaxScaler())
                                   ])

    num_attribs = ['meter_reading','air_temperature','dew_temperature','sea_level_pressure','wind_direction','wind_speed'] # columns to transform
    date_attribs=['day','month'	,'hour','area','floor','hour', 'season', 'weekend', 'day_of_the_week']

    full_pipeline = ColumnTransformer([("num", numerical_pipeline, num_attribs),
                                       ("date", Do_nothing(),  date_attribs)])

    return full_pipeline, data_cleaned
