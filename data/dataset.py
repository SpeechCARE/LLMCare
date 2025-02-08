from utils import *
import pandas as pd
import numpy as np 

def load_dataset():
    sp = Supplementary()

    # Load train, validation, and test textual data
    df_train_text = pd.read_csv(data_path + train_text_path)
    df_val_text = pd.read_csv(data_path + val_text_path)
    df_test_text = pd.read_excel(data_path + test_text_path)

    # Load train, validation, and test linguistic features
    df_train_ling = pd.read_excel(train_ling_path)
    df_valid_ling = pd.read_excel(valid_ling_path)
    df_test_ling = pd.read_excel(test_ling_path)

    # Add a unique keyword to each column's name
    df_train_ling = sp.add_keyword(df_train_ling, 'LING', exclude_columns=['id', 'label'])
    df_valid_ling = sp.add_keyword(df_valid_ling, 'LING', exclude_columns=['id', 'label'])
    df_test_ling = sp.add_keyword(df_test_ling, 'LING', exclude_columns=['id', 'label'])

    # Replace the inf or nan values with the column mean
    df_train_ling = sp.replace_inf_nan_with_column_mean(df_train_ling, exclude_columns=['id', 'label'])
    df_valid_ling = sp.replace_inf_nan_with_column_mean(df_valid_ling, exclude_columns=['id', 'label'])
    df_test_ling = sp.replace_inf_nan_with_column_mean(df_test_ling, exclude_columns=['id', 'label'])

    # Normalize the features between 0 and 1
    df_train_ling, scaler = sp.scale_columns_with_minmax(df_train_ling, exclude_columns=['id', 'label'])
    df_valid_ling = sp.scale_columns_with_minmax(df_valid_ling, scaler=scaler, exclude_columns=['id', 'label'])
    df_test_ling = sp.scale_columns_with_minmax(df_test_ling, scaler=scaler, exclude_columns=['id', 'label'])

    # Load train, validation, and test LIWC features
    df_LIWC_train = pd.read_excel(train_LIWC_path)
    df_LIWC_valid = pd.read_excel(valid_LIWC_path)
    df_LIWC_test = pd.read_excel(test_LIWC_path)

    # Remove all-zero columns (based on the training set)
    df_LIWC_train, df_LIWC_valid, df_LIWC_test = sp.remove_badColumns(df_train=df_LIWC_train, df_val=df_LIWC_valid, df_test=df_LIWC_test)


    # Add a unique keyword to each column name
    df_LIWC_train = sp.add_keyword(df_LIWC_train, 'LIWC', exclude_columns=['id', 'label'])
    df_LIWC_valid = sp.add_keyword(df_LIWC_valid, 'LIWC', exclude_columns=['id', 'label'])
    df_LIWC_test = sp.add_keyword(df_LIWC_test, 'LIWC', exclude_columns=['id', 'label'])

    # Replace the inf or nan values with the column mean
    df_LIWC_train = sp.replace_inf_nan_with_column_mean(df_LIWC_train, exclude_columns=['id', 'label'])
    df_LIWC_valid = sp.replace_inf_nan_with_column_mean(df_LIWC_valid, exclude_columns=['id', 'label'])
    df_LIWC_test = sp.replace_inf_nan_with_column_mean(df_LIWC_test, exclude_columns=['id', 'label'])

    # Normalize the features between 0 and 1
    df_LIWC_train, scaler = sp.scale_columns_with_minmax(df_LIWC_train, exclude_columns=['id', 'label'])
    df_LIWC_valid = sp.scale_columns_with_minmax(df_LIWC_valid, scaler=scaler, exclude_columns=['id', 'label'])
    df_LIWC_test = sp.scale_columns_with_minmax(df_LIWC_test, scaler=scaler, exclude_columns=['id', 'label'])

    # Load train, validation, and test JMIM_LIWC features
    df_jmim_LIWC_train = pd.read_excel(train_jmim_LIWC_path)
    df_jmim_LIWC_valid = pd.read_excel(valid_jmim_LIWC_path)
    df_jmim_LIWC_test = pd.read_excel(test_jmim_LIWC_path)

    # Remove all-zero columns (based on the training set)
    df_jmim_LIWC_train, df_jmim_LIWC_valid, df_jmim_LIWC_test = sp.remove_badColumns(df_train=df_jmim_LIWC_train, df_val=df_jmim_LIWC_valid, df_test=df_jmim_LIWC_test)

    # Add a unique keyword to each column name
    df_jmim_LIWC_train = sp.add_keyword(df_jmim_LIWC_train, 'JMIM', exclude_columns=['id', 'label'])
    df_jmim_LIWC_valid = sp.add_keyword(df_jmim_LIWC_valid, 'JMIM', exclude_columns=['id', 'label'])
    df_jmim_LIWC_test = sp.add_keyword(df_jmim_LIWC_test, 'JMIM', exclude_columns=['id', 'label'])

    # Replace the inf or nan values with the column mean
    df_jmim_LIWC_train = sp.replace_inf_nan_with_column_mean(df_jmim_LIWC_train, exclude_columns=['id', 'label'])
    df_jmim_LIWC_valid = sp.replace_inf_nan_with_column_mean(df_jmim_LIWC_valid, exclude_columns=['id', 'label'])
    df_jmim_LIWC_test = sp.replace_inf_nan_with_column_mean(df_jmim_LIWC_test, exclude_columns=['id', 'label'])

    # Normalize the features between 0 and 1
    df_jmim_LIWC_train, scaler = sp.scale_columns_with_minmax(df_jmim_LIWC_train, exclude_columns=['id', 'label'])
    df_jmim_LIWC_valid = sp.scale_columns_with_minmax(df_jmim_LIWC_valid, scaler=scaler, exclude_columns=['id', 'label'])
    df_jmim_LIWC_test = sp.scale_columns_with_minmax(df_jmim_LIWC_test, scaler=scaler, exclude_columns=['id', 'label'])

    # Merge the textual data, linguistic, and LIWC features based on the 'id' and 'label' columns
    # Train data
    temp = pd.merge(df_train_text, df_train_ling, on=['id', 'label'])
    temp = pd.merge(temp, df_LIWC_train, on=['id', 'label'])
    train_data = pd.merge(temp, df_jmim_LIWC_train, on=['id', 'label'])

    # Validation data
    temp = pd.merge(df_val_text, df_valid_ling, on=['id', 'label'])
    temp = pd.merge(temp, df_LIWC_valid, on=['id', 'label'])
    valid_data = pd.merge(temp, df_jmim_LIWC_valid, on=['id', 'label'])

    # Test data
    temp = pd.merge(df_test_text, df_test_ling, on=['id', 'label'])
    temp = pd.merge(temp, df_LIWC_test, on=['id', 'label'])
    test_data = pd.merge(temp, df_jmim_LIWC_test, on=['id', 'label'])

    return train_data, valid_data, test_data
