import tsfresh
from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_extraction.settings import ComprehensiveFCParameters, MinimalFCParameters
import streamlit as st
import pandas as pd
import polars as pl
 
def feature_extraction(df):
    """Extract features from a time series using TSFresh and display them in Streamlit."""
    st.title("Feature Extraction using TSFresh")
 
    # Select time and value columns
    time_column, value_column = select_columns(df)
 
    # List of available features for the user to select from
    available_features = [
        "sum_values",
        "median",
        "mean",
        "length",
        "standard_deviation",
        "variance",
        "root_mean_square",
        "maximum",
        "absolute_maximum",
        "minimum"
    ]
 
    # Multi-select dropdown to specify features
    selected_features = st.multiselect("Select Features to Extract", available_features)
 
    if st.button("Extract Features"):
        if time_column and value_column:
            # Subset data for better performance (optional, adjust as necessary)
            df_subset = df.head(1000)
 
            # Clean data (remove missing values)
            df_cleaned = clean_data(df_subset, time_column, value_column)
 
            # Extract features
            extracted_features = extract_time_series_features(df_cleaned, time_column, value_column, selected_features)
 
            # Display the extracted features
            display_extracted_features(extracted_features)
        else:
            st.error("Please select both time and value columns.")
 
def select_columns(df):
    """Provide Streamlit UI for selecting time and value columns."""
    time_column = st.selectbox("Select Time Column", df.columns)
    value_column = st.selectbox("Select Value Column", df.columns)
    return time_column, value_column
 
def clean_data(df, time_column, value_column):
    """Clean data by removing rows with missing values."""
    # Ensure that we only keep the relevant columns
    df_cleaned = df[[time_column, value_column]].drop_nulls()  # Drop rows with missing values in Polars
    return df_cleaned
 
def extract_time_series_features(df, time_column, value_column, selected_features):
    """Extract time series features using TSFresh."""
    # Convert Polars DataFrame to Pandas DataFrame for TSFresh compatibility
    df_tsfresh = df.to_pandas()
 
    # Ensure the time column is in datetime format for TSFresh
    df_tsfresh[time_column] = pd.to_datetime(df_tsfresh[time_column])
 
    # Sort the data by time to ensure the time series is ordered
    df_tsfresh = df_tsfresh.sort_values(by=time_column)
 
    # Add an ID column required by TSFresh (if you have one time series, use a constant value like 0)
    df_tsfresh['id'] = 0
 
    # Define the feature extraction settings (only include features selected by the user)
    fc_parameters = MinimalFCParameters()
 
    # Extract features using TSFresh
    with st.spinner('Extracting features...'):
        try:
            extracted_features = extract_features(
                df_tsfresh,
                column_id='id',
                column_sort=time_column,
                default_fc_parameters=fc_parameters,
                n_jobs=1  # Set to 1 instead of -1 for single process execution (avoid parallelization issues)
            )
        except Exception as e:
            st.error(f"Error during feature extraction: {e}")
            return pd.DataFrame()  # Return an empty DataFrame in case of error
 
    # Filter extracted features based on the user's selection
    if selected_features:
        # Dynamically use the name of the selected value column (e.g., 'temperature', 'pressure')
        selected_feature_columns = [f"{value_column}__{feature}" for feature in selected_features]
        extracted_features = extracted_features[selected_feature_columns]
 
    # Impute missing values in the extracted features (TSFresh requires this)
    extracted_features = impute(extracted_features)  # Impute missing values
 
    return extracted_features
 
def display_extracted_features(extracted_features):
    """Display the extracted features in the Streamlit app."""
    if extracted_features.empty:
        st.error("No features extracted.")
    else:
        # Display the extracted features vertically (transposed)
        st.write("Extracted Features (Transposed):")
        st.dataframe(extracted_features.T)  # Transpose the DataFrame to display vertically
    st.success('Feature extraction complete!')