import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

class Supplementary():

    def __init__(self):
        pass

    def remove_badColumns(self, df_train, df_val, df_test):
        bad_columns = [column for column in df_train.columns if (df_train[column].values == np.zeros(df_train.shape[0])).all()]
        df_train = df_train.drop(bad_columns, axis=1)
        df_val = df_val.drop(bad_columns, axis=1)
        df_test = df_test.drop(bad_columns, axis=1)
        return df_train, df_val, df_test


    def add_keyword(self, df, keyword, exclude_columns = []):
        keyword = keyword + '-'
        for col in df.columns:
            if not col in exclude_columns:
                new_col_name = f'{keyword} {col}'
                df.rename(columns={col: new_col_name}, inplace=True)
        return df


    def replace_inf_nan_with_column_mean(self, df, exclude_columns=[]):
        for column in df.columns:

            if column in exclude_columns:
                continue

            inf_indices = np.isinf(df[column])
            nan_indices = np.isnan(df[column])
            if inf_indices.any() or nan_indices.any():
                column_mean = np.nanmean(df[column].values[~inf_indices])
                # Replace the inf and nan values with the column mean
                df[column][inf_indices] = column_mean
                df[column][nan_indices] = column_mean

        return df


    def scale_columns_with_minmax(self, df, scaler = None,  exclude_columns=[]):

        # Select only the columns that need to be scaled
        columns_to_scale = [col for col in df.columns if col not in exclude_columns]

        # Create a copy of the DataFrame to avoid modifying the original
        df_scaled = df.copy()

        # Apply MinMaxScaler to the selected columns
        if scaler == None:
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaler.fit(df[columns_to_scale])
            df_scaled[columns_to_scale] = scaler.transform(df[columns_to_scale])
            return df_scaled, scaler

        else:
            df_scaled[columns_to_scale] = scaler.transform(df[columns_to_scale])
            return df_scaled