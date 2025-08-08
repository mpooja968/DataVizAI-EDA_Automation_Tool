# schema_manipulation.py

import streamlit as st
import polars as pl
import pandas as pd
import plotly.express as px
import dabl
import math



def select_timestamp_column(df):
    """
    Provides the option to select a timestamp column either automatically or manually.

    Parameters:
    - df (pl.DataFrame): The input dataframe.

    Returns:
    - datetime_column (str or None): The selected timestamp column or None if no valid column is found.
    """
    st.dataframe(st.session_state['df'].head())
    # Timestamp selection option
    timestamp_selection = st.radio("Select Timestamp Option:", ["Automatic", "Manual"], index=0)

    if timestamp_selection == "Automatic":
        # Handle datetime column parsing automatically
        datetime_column = None
        for col in df.columns:
            if df[col].dtype == pl.Datetime:
                datetime_column = col
                break

        if datetime_column is None:
            st.error("No valid datetime column found in the DataFrame.")
        else:
            st.write(f"Automatic timestamp column detected: {datetime_column}")
        return datetime_column

    elif timestamp_selection == "Manual":
        # Allow user to select any column from the dataset (not limited to datetime)
        datetime_column = st.selectbox("Select a Timestamp Column from all available columns:", df.columns)
        st.write(f"Selected Timestamp Column: {datetime_column}")
        return datetime_column
        

# ----------------- Data Manipulation -----------------

# Function to handle data manipulation
def data_manipulation(df, numerical_columns, categorical_columns):
    # Store the dataframe in Streamlit's session state for persistence
    if 'df' not in st.session_state:
        st.session_state.df = df

    # User selects columns to convert
    selected_numerical = st.multiselect("Select numerical columns to convert to categorical:", numerical_columns)

    # Convert numerical to categorical
    if st.button("Preview Conversion: Convert Numerical to Categorical"):
        preview_df = st.session_state.df.clone()  # Use .clone() instead of .copy()
        for col in selected_numerical:
            if preview_df[col].n_unique() <= 10:
                # Preview conversion: cast to string first, then to categorical
                preview_df = preview_df.with_columns(preview_df[col].cast(pl.Utf8).alias(col))  # First cast to string
                preview_df = preview_df.with_columns(preview_df[col].cast(pl.Categorical).alias(col))  # Then cast to categorical

        st.write("**Preview of Converted Numerical Columns to Categorical:**")
        st.dataframe(preview_df[selected_numerical].head())

    selected_categorical = st.multiselect("Select categorical columns to convert to numerical:", categorical_columns)

    # Convert categorical to numerical
    if st.button("Preview Conversion: Convert Categorical to Numerical"):
        preview_df = st.session_state.df.clone()  # Use .clone() instead of .copy()
        for col in selected_categorical:
            # Preview conversion: replace nulls with 0, then cast to numerical
            preview_df = preview_df.with_columns(
                pl.col(col)
                .fill_null(0)
                .cast(pl.Float32)
                .alias(col)
            )
            
        st.write("**Preview of Converted Categorical Columns to Numerical:**")
        st.dataframe(preview_df[selected_categorical].head())

    # Apply Changes button
    if st.button("Apply Changes"):
        for col in selected_numerical:
            if st.session_state.df[col].n_unique() <= 10:
                st.session_state.df = st.session_state.df.with_columns(st.session_state.df[col].cast(pl.Utf8).alias(col))
                st.session_state.df = st.session_state.df.with_columns(st.session_state.df[col].cast(pl.Categorical).alias(col))
                numerical_columns.remove(col)
                categorical_columns.append(col)

        for col in selected_categorical:
            st.session_state.df = st.session_state.df.with_columns(
                pl.col(col)
                .fill_null(0)
                .cast(pl.Float32)
                .alias(col)
            )
            categorical_columns.remove(col)
            numerical_columns.append(col)

        # Update session state for numerical and categorical columns
        st.session_state.numerical_columns = numerical_columns
        st.session_state.categorical_columns = categorical_columns

        st.success("Changes applied successfully!")
        st.write("**Preview of Updated DataFrame:**")
        st.dataframe(st.session_state.df.head())

    # View entire DataFrame
    if st.button("View Manipulated DataFrame"):
        st.write("**Manipulated DataFrame:**")
        st.dataframe(st.session_state.df.head())

    # Return the updated DataFrame for further use
    return st.session_state.df



# ----------------- Data Aggregation -----------------

# Define the data aggregation function
def data_aggregation(df, numerical_columns, categorical_columns):
    # Display the snippet of the last session's DataFrame
    if 'df' in st.session_state:
        st.write("**Existing DataFrame (Snippet of First 5 Rows):**")
        st.dataframe(st.session_state.df.to_pandas().head())  # Display the snippet of the DataFrame from session state

    # User selects the type of aggregation
    aggregation_type = st.radio("Select the type of aggregation:", 
                                ("Using Timestamp and Variable"))

    if aggregation_type == "Using Timestamp and Variable":
        result_df = aggregate_using_timestamp_and_variable(df, numerical_columns, categorical_columns)

    
    # Add Apply Changes button
    if st.button("Apply Changes"):
        # Apply the changes and update the DataFrame in session state
        if 'df' in st.session_state:
            st.session_state.df = df  # Update the session state's df with the current df
            st.success("Changes applied successfully.")
        else:
            st.error("No changes were made.")

    # Return the result after aggregation
    return result_df

# Function for aggregation using timestamp only

# Function for aggregation using timestamp and specific variable selection
def aggregate_using_timestamp_and_variable(df, numerical_columns, categorical_columns):
    time_intervals = ['1min', '5min', '10min', '30min', '1H', '1D', '1W']
    selected_interval = st.selectbox("Select time interval for aggregation:", time_intervals)

    aggregation_option = st.radio(
        "Select aggregation option:",
        ["Aggregate All Columns", "Aggregate Column Wise"]
    )

    # Handle datetime column parsing if necessary
    datetime_column = None
    for col in df.columns:
        if df[col].dtype == pl.Datetime:
            datetime_column = col
            break

    if datetime_column is None:
        st.error("No valid datetime column found in the DataFrame.")
        return df

    # Convert the Polars DataFrame to pandas for aggregation
    pandas_df = df.to_pandas()

    # Convert the datetime column to pandas datetime
    pandas_df[datetime_column] = pd.to_datetime(pandas_df[datetime_column])

    # Aggregation logic
    if aggregation_option == "Aggregate All Columns":
        aggregation_methods = ['Mean', 'Sum', 'Max', 'Min', 'Count', 'Mode', 'First', 'Last', 'Standard Deviation']
        selected_method = st.selectbox("Select aggregation method for all columns:", aggregation_methods)

        if st.button("Aggregate All Columns"):
            try:
                pandas_df.set_index(datetime_column, inplace=True)

                # Identify numeric columns
                numeric_columns = pandas_df.select_dtypes(include=['number']).columns.tolist()

                if not numeric_columns:
                    st.error("No numeric columns available for aggregation.")
                    return df

                # Prepare aggregation dictionary only for numeric columns
                aggregation_dict = {}
                if selected_method == "Mean":
                    aggregation_dict = {col: 'mean' for col in numeric_columns}
                elif selected_method == "Sum":
                    aggregation_dict = {col: 'sum' for col in numeric_columns}
                elif selected_method == "Max":
                    aggregation_dict = {col: 'max' for col in numeric_columns}
                elif selected_method == "Min":
                    aggregation_dict = {col: 'min' for col in numeric_columns}
                elif selected_method == "Count":
                    aggregation_dict = {col: 'count' for col in numeric_columns}
                elif selected_method == "Mode":
                    aggregation_dict = {
                        col: lambda x: x.mode().iloc[0] if not x.mode().empty else None for col in numeric_columns
                    }
                elif selected_method == "First":
                    aggregation_dict = {col: 'first' for col in numeric_columns}
                elif selected_method == "Last":
                    aggregation_dict = {col: 'last' for col in numeric_columns}
                elif selected_method == "Standard Deviation":
                    aggregation_dict = {col: 'std' for col in numeric_columns}

                # Apply aggregation
                aggregated_df = pandas_df.resample(selected_interval).agg(aggregation_dict).reset_index()

                # Convert back to Polars
                aggregated_df_polars = pl.from_pandas(aggregated_df)

                st.success(f"Aggregation using `{selected_method}` completed.")
                st.write("**Aggregated DataFrame Snippet (First 5 Rows):**")
                st.dataframe(aggregated_df_polars.to_pandas().head())

                # Update session state
                st.session_state.df = aggregated_df_polars

                return aggregated_df_polars

            except Exception as e:
                st.error(f"An error occurred during aggregation: {str(e)}")


    elif aggregation_option == "Aggregate Column Wise":
        selected_data_type = st.selectbox("Select data type for aggregation:", ["Numerical", "Categorical"])

        if selected_data_type == "Numerical":
            aggregation_methods = ['Mean', 'Sum', 'Max', 'Min', 'First', 'Last', 'Standard Deviation']
        elif selected_data_type == "Categorical":
            aggregation_methods = ['Mode', 'Count']

        # Create a dictionary to store user-defined aggregation methods for each column
        column_aggregation_map = {}
        columns_to_select_from = numerical_columns if selected_data_type == "Numerical" else categorical_columns
        selected_columns = st.multiselect("Select columns for aggregation:", columns_to_select_from)

        if selected_columns:
            # Define the maximum number of columns per row
            max_columns_per_row = 3
            # Calculate how many rows will be needed
            num_rows = math.ceil(len(selected_columns) / max_columns_per_row)

            # Display the "Aggregation Types" heading
            st.write("**Aggregation Types**")

            # For each row
            for row in range(num_rows):
                # Get the columns for this row (up to 3 per row)
                start_index = row * max_columns_per_row
                end_index = start_index + max_columns_per_row
                columns_in_row = selected_columns[start_index:end_index]

                # Create columns for each column in the row
                cols = st.columns(len(columns_in_row))  # Create as many columns as there are selected columns in this row
                
                # Display the dropdowns horizontally for each column in this row
                aggregation_methods_for_row = []
                for i, col in enumerate(columns_in_row):
                    with cols[i]:  # For each column in the row
                        selected_aggregation_method = st.selectbox(
                            f"{col}:",
                            aggregation_methods,
                            key=f"agg_{col}"
                        )
                        aggregation_methods_for_row.append((col, selected_aggregation_method))

                # Update the aggregation map
                for col, selected_method in aggregation_methods_for_row:
                    column_aggregation_map[col] = selected_method

        if st.button("Aggregate Column Wise"):
            try:
                pandas_df.set_index(datetime_column, inplace=True)

                # Prepare aggregation dictionary
                aggregation_dict = {}
                for col, method in column_aggregation_map.items():
                    if method == "Mean":
                        aggregation_dict[col] = 'mean'
                    elif method == "Sum":
                        aggregation_dict[col] = 'sum'
                    elif method == "Max":
                        aggregation_dict[col] = 'max'
                    elif method == "Min":
                        aggregation_dict[col] = 'min'
                    elif method == "Mode":
                        aggregation_dict[col] = lambda x: x.mode().iloc[0] if not x.mode().empty else None
                    elif method == "Count":
                        aggregation_dict[col] = 'count'
                    elif method == "First":
                        aggregation_dict[col] = 'first'
                    elif method == "Last":
                        aggregation_dict[col] = 'last'
                    elif method == "Standard Deviation":
                        aggregation_dict[col] = 'std'

                # Apply aggregation
                aggregated_df = pandas_df.resample(selected_interval).agg(aggregation_dict).reset_index()

                # Convert back to Polars
                aggregated_df_polars = pl.from_pandas(aggregated_df)

                st.success(f"Column-wise aggregation completed.")
                st.write("**Aggregated DataFrame Snippet (First 5 Rows):**")
                st.dataframe(aggregated_df_polars.to_pandas().head())

                # Update session state
                st.session_state.df = aggregated_df_polars

                return aggregated_df_polars

            except Exception as e:
                st.error(f"An error occurred during column-wise aggregation: {str(e)}")

        return df  # Return unmodified df if no aggregation happened


