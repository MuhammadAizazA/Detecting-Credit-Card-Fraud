from sklearn.preprocessing import OrdinalEncoder
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN


def prepare_data(dataframe):
    """
    The function `prepare_data` takes a dataframe as input, performs various data preprocessing steps
    such as dropping unnecessary columns, converting date and time columns to datetime objects,
    calculating age based on date of birth, filtering out transactions with amount greater than or equal
    to 10000, creating new features related to time, and extracting day, week, and month information,
    and returns the modified dataframe.

    Args:
      dataframe: The parameter "dataframe" is a pandas DataFrame that contains transaction data.

    Returns:
      the modified dataframe after performing various data preparation steps.
    """
    # Drop unnecessary columns
    dataframe = dataframe.drop(
        columns=['Unnamed: 0', 'trans_num', 'first', 'last', 'zip'])

    # Convert date and time columns to datetime objects
    dataframe['trans_date_trans_time'] = pd.to_datetime(
        dataframe['trans_date_trans_time'])
    dataframe['trans_time'] = dataframe['trans_date_trans_time'].dt.time
    dataframe['trans_date'] = dataframe['trans_date_trans_time'].dt.date
    dataframe['trans_month'] = dataframe['trans_date_trans_time'].dt.month

    # Calculate age based on date of birth and transaction date
    dataframe['dob'] = pd.to_datetime(dataframe['dob'])
    dataframe['trans_date'] = pd.to_datetime(dataframe['trans_date'])
    dataframe['age'] = (
        (dataframe['trans_date'] - dataframe['dob']).dt.days // 365.25).astype(int)

    # # Filter out transactions with amount greater than or equal to 10000
    # dataframe = dataframe[dataframe['amt'] < 15000]

    # Create new features related to time
    dataframe['during_officehours'] = ((8 <= dataframe['trans_date_trans_time'].dt.hour) & (
        dataframe['trans_date_trans_time'].dt.hour <= 19)).astype(int)
    dataframe['day_type'] = np.where(
        dataframe['trans_date_trans_time'].dt.dayofweek < 5, 'weekday', 'weekend')

    # Extract day, week, and month information
    dataframe['day_of_month'] = dataframe['trans_date_trans_time'].dt.day
    dataframe['day_of_week'] = dataframe['trans_date_trans_time'].dt.weekday

    # # Create categorical features for different parts of the month
    # dataframe['first_10_days'] = (dataframe['day_of_month'] <= 10).astype(int)
    # dataframe['middle_10_days'] = ((10 < dataframe['day_of_month']) & (
    #     dataframe['day_of_month'] <= 20)).astype(int)
    # dataframe['last_10_days'] = (dataframe['day_of_month'] > 20).astype(int)

    return dataframe


def label_encode_data(dataframe, columns):
    """
    The function label_encode_data takes a dataframe and a list of columns as input, and applies label
    encoding to those columns in the dataframe.

    Args:
      dataframe: The dataframe parameter is the pandas DataFrame that you want to perform label encoding
    on. It should contain the columns that you want to encode.
      columns: The "columns" parameter is a list of column names in the dataframe that you want to label
    encode.

    Returns:
      the modified dataframe with the specified columns label encoded.
    """
    label_encoder = LabelEncoder()

    for col in columns:
        dataframe[col] = label_encoder.fit_transform(dataframe[col])

    return dataframe


def onehot_encode_data(dataframe, columns):
    """
    The function onehot_encode_data takes a dataframe and a list of columns as input, and returns the
    dataframe with the specified columns one-hot encoded.

    Args:
      dataframe: The dataframe parameter is the pandas DataFrame that you want to perform one-hot
    encoding on. It should contain the categorical columns that you want to encode.
      columns: The "columns" parameter is a list of column names in the dataframe that you want to
    one-hot encode.

    Returns:
      the modified dataframe after performing one-hot encoding on the specified columns.
    """
    onehot_encoder = OneHotEncoder(sparse_output=False)
    new_columns = []

    for col in columns:
        encoded_data = onehot_encoder.fit_transform(dataframe[[col]])
        columns_names = [f"{col}_{
            category}" for category in onehot_encoder.get_feature_names_out([col])]
        new_columns.extend(columns_names)

        # Create a new dataframe with the one-hot encoded columns
        encoded_df = pd.DataFrame(
            encoded_data, columns=columns_names, index=dataframe.index)

        # Drop the original column from the original dataframe
        dataframe = dataframe.drop(columns=[col])

    # Concatenate the original dataframe with the encoded dataframe
    dataframe = pd.concat([dataframe, encoded_df], axis=1)

    return dataframe


def frequency_encode_data(dataframe, columns):
    """
    The function frequency_encode_data takes a dataframe and a list of columns as input, and returns the
    dataframe with the values in the specified columns replaced by their frequency of occurrence.

    Args:
      dataframe: The dataframe parameter is the pandas DataFrame that you want to encode the frequencies
    for. It should contain the columns that you want to encode.
      columns: The "columns" parameter is a list of column names in the dataframe that you want to
    encode using frequency encoding.

    Returns:
      the modified dataframe with the encoded frequencies for the specified columns.
    """
    for col in columns:
        freq_map = dataframe[col].value_counts(normalize=True).to_dict()
        dataframe[col] = dataframe[col].map(freq_map)

    return dataframe


def drop_columns(dataframe, df_columns):
    """
    The function `drop_columns` takes a dataframe and a list of column names as input, and returns the
    dataframe with the specified columns dropped.

    Args:
      dataframe: The dataframe parameter is the pandas DataFrame object that you want to modify by
    dropping columns.
      df_columns: The parameter `df_columns` is a list of column names that you want to drop from the
    dataframe.

    Returns:
      the modified dataframe after dropping the specified columns.
    """
    dataframe = dataframe.drop(columns=df_columns)
    return dataframe


def print_group_count(dataframe):
    """
    The function `print_group_count` takes a dataframe as input and prints the group sizes for each
    column in the dataframe.

    Args:
      dataframe: The parameter `dataframe` is expected to be a pandas DataFrame object.
    """
    for column in dataframe.columns:
        print(f"Group sizes for {column}:\n")
        group_sizes = dataframe.groupby(column).size()
        print(group_sizes)
        print("\n" + "="*50 + "\n")


def scale_data(dataframe, columns_to_scale, scaler_type='standard'):
    """
    The function `scale_data` scales specified columns in a dataframe using different types of scalers
    (standard, minmax, or robust) and returns the scaled dataframe.

    Args:
      dataframe: The dataframe parameter is the input dataframe that contains the data you want to
    scale.
      columns_to_scale: The parameter "columns_to_scale" is a list of column names in the dataframe that
    you want to scale. These columns will be selected from the dataframe and scaled using the specified
    scaler type.
      scaler_type: The `scaler_type` parameter specifies the type of scaler to use for scaling the data.
    It has a default value of 'standard', but it can also be set to 'minmax' or 'robust'. Defaults to
    standard

    Returns:
      a scaled dataframe that includes the non-scaled columns, the scaled columns, and the target
    variable 'is_fraud'.
    """
    # Choose the scaler based on the specified type
    try:
        # Choose the scaler based on the specified type
        if scaler_type == 'standard':
            scaler = StandardScaler()
        elif scaler_type == 'minmax':
            scaler = MinMaxScaler()
        elif scaler_type == 'robust':
            scaler = RobustScaler()
        else:
            raise ValueError(
                "Invalid scaler type. Choose 'standard', 'minmax', or 'robust'.")
    except ValueError as e:
        raise ValueError(f"Error: {
                         e}. Please choose a valid scaler type: 'standard', 'minmax', or 'robust'.")

    # Extract the specified columns from the dataframe
    scaled_dataframe = dataframe[columns_to_scale]

    # Fit and transform the selected columns
    X_scaled = pd.DataFrame(scaler.fit_transform(
        scaled_dataframe), columns=columns_to_scale)

    # Keep the non-scaled columns
    remaining_columns = dataframe.drop(columns=columns_to_scale)

    # Concatenate the scaled and non-scaled columns
    scaled_dataframe = pd.concat([remaining_columns, X_scaled], axis=1)

    return scaled_dataframe


def balance_data(X, technique='smote'):
    """
    The `balance_data` function takes in a dataset `X` and applies a specified imbalance correction
    technique to balance the data, returning a new dataframe with the resampled data.

    Args:
      X: X is a pandas DataFrame that contains the input features for the data. It should not include
    the target variable 'is_fraud'.
      technique: The `technique` parameter is used to specify the imbalance correction technique to be
    applied to the data. The available options are:. Defaults to smote

    Returns:
      The function `balance_data` returns a resampled dataframe with balanced data.
    """
    try:
        # Choose the imbalance correction technique based on the specified type
        if technique == 'smote':
            imbalance_technique = SMOTE()
        elif technique == 'random_undersampling':
            imbalance_technique = RandomUnderSampler()
        elif technique == 'smote_enn':
            imbalance_technique = SMOTEENN()
        else:
            raise ValueError(
                "Invalid imbalance correction technique. Choose 'smote', 'random_undersampling', or 'smote_enn'.")
    except ValueError as e:
        raise ValueError(f"Error: {
                         e}. Please choose a valid imbalance correction technique: 'smote', 'random_undersampling', or 'smote_enn'.")

    y = X['is_fraud']
    X = X.drop(columns='is_fraud')  # Specify the column to drop

    # Apply the selected imbalance correction technique to the data
    X_resampled, y_resampled = imbalance_technique.fit_resample(X, y)

    print(pd.Series(y_resampled).value_counts())

    # Create a new dataframe with the resampled data
    resampled_df = pd.DataFrame(X_resampled, columns=X.columns)
    resampled_df['is_fraud'] = y_resampled

    return resampled_df


if '__main__' == __name__:
    print("I'm called From data_utils module")
