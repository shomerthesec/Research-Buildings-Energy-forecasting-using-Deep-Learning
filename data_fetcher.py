import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class Fetcher(BaseEstimator, TransformerMixin):

    def __init__(self, building_id=122, meter=0, primary_use=99):  # 99 referes to none

        self.building_id = building_id
        self.meter = meter
        self.primary_use = primary_use

    def fit(self, x, y=None):
        return self  # nothing else to do

    def transform(self, x):
        df = x.copy()
        df = df.drop(['Unnamed: 0', 'precip_depth_1_hr',
                     'cloud_coverage', 'site_id', 'square_feet'], axis=1)

        if self.primary_use == 99:

            df = df.query(
                f'building_id=={self.building_id} & meter=={self.meter}')
            df.drop(['building_id', 'meter', 'primary_use'],
                    axis=1, inplace=True)

        else:

            df = df.query(
                f'building_id=={self.building_id} & meter=={self.meter} & primary_use =={self.primary_use}')
            df.drop(['building_id', 'meter', 'primary_use'],
                    axis=1, inplace=True)

        df.loc[:, "timestamp"] = pd.to_datetime(df.loc[:, "timestamp"])
        df['weekend'] = df.timestamp.dt.dayofweek > 4
        df['day_of_the_week'] = df.timestamp.dt.dayofweek
        df.set_index('timestamp', inplace=True)

        return df


class Do_nothing(BaseEstimator, TransformerMixin):

    def __init__(self):  # 99 referes to none
        return

    def fit(self, x, y=None):
        return self  # nothing else to do

    def transform(self, x):
        return x.astype(int)
