import streamlit as st
import polars as pl
import pandas as pd
import plotly.express as px
import numpy as np
from pyod.models.iforest import IForest  # Import Isolation Forest

def detect_outliers(df):
    """
    Detects outliers in numerical columns based on the selected method.
    
    Args:
        df (pl.DataFrame): The input Polars DataFrame with data.

    Returns:
        None: Displays the outlier detection results as a table and plots.
    """
    # Get numerical columns
    numerical_columns = [col for col, dtype in df.schema.items() if dtype in (pl.Float32, pl.Float64)]
    outlier_results = []

    # Method selection via radio button
    method = st.radio("Select Outlier Detection Method", ["IQR", "Z-Score", "MAD(Median Absolute Deviation)", "PyOD Method"])

    # Option for IQR threshold adjustment
    iqr_threshold = 1.5  # Default IQR multiplier
    if method == "IQR":
        iqr_threshold = st.slider("Select IQR Threshold Multiplier", min_value=1.0, max_value=5.0, value=1.5, step=0.1)

    for var in numerical_columns:
        # Remove null values for the analysis
        cleaned_data = df.select(var).drop_nulls()

        # Initialize the outlier data
        outliers = None
        outlier_data = None

        if method == "IQR":
            # Compute Q1 and Q3
            q1 = cleaned_data.select(pl.col(var).quantile(0.25)).to_numpy()[0][0]
            q3 = cleaned_data.select(pl.col(var).quantile(0.75)).to_numpy()[0][0]

            # Compute the IQR
            iqr = q3 - q1

            # Define outlier thresholds using the user-selected multiplier
            lower_bound = q1 - iqr_threshold * iqr
            upper_bound = q3 + iqr_threshold * iqr

            # Identify outliers (values outside the IQR range)
            outliers = cleaned_data.filter(
                (pl.col(var) < lower_bound) | (pl.col(var) > upper_bound)
            )

        elif method == "Z-Score":
            # Compute mean and standard deviation
            mean = cleaned_data.select(pl.col(var).mean()).to_numpy()[0][0]
            std_dev = cleaned_data.select(pl.col(var).std()).to_numpy()[0][0]

            # Z-score calculation and outlier detection
            z_scores = cleaned_data.with_columns(
                ((pl.col(var) - mean) / std_dev).alias("z_score")
            )
            outliers = z_scores.filter(pl.col("z_score").abs() > 3)  # Z-score threshold is 3

        elif method == "MAD(Median Absolute Deviation)":
            # Compute the median and MAD
            median = cleaned_data.select(pl.col(var).median()).to_numpy()[0][0]
            mad = cleaned_data.with_columns(
                (pl.col(var) - median).abs().alias("mad")
            ).select(pl.col("mad")).median().to_numpy()[0][0]

            # Define outlier threshold
            threshold = 3  # MAD threshold
            # We need to create a boolean mask for MAD filtering
            mad_data = cleaned_data.with_columns(
                (pl.col(var) - median).abs().alias("mad")
            )
            outliers = mad_data.filter(pl.col("mad") > threshold)

        elif method == "PyOD Method":
            # Downsampling to 10% of data for faster testing
            sample_size = int(cleaned_data.height * 0.1)
            sampled_data = cleaned_data.sample(n=sample_size, seed=42).to_numpy().flatten().reshape(-1, 1)

            # Initialize Isolation Forest model with parallel processing
            model = IForest(n_jobs=-1, random_state=42)  # Parallel processing with all available cores
            model.fit(sampled_data)

            # Predict outliers using the Isolation Forest model
            outlier_labels = model.predict(sampled_data)  # 1 for outliers, 0 for inliers

            # Filter the outliers based on the predicted labels
            outliers = cleaned_data.filter(pl.col(var).is_in(sampled_data[outlier_labels == 1].flatten()))

        # Get the number and percentage of outliers
        total_count = df.height
        count_outliers = outliers.height
        percentage_outliers = (count_outliers / total_count) * 100 if total_count > 0 else 0

        # Prepare the outliers data for displaying
        outlier_data = outliers

        outlier_results.append({
            'Variable': var,
            'Total Data Points': total_count,
            'Number of Outliers': count_outliers,
            'Percentage of Outliers': f"{percentage_outliers:.2f}%",
            'Outliers Data': outlier_data  # Store the outliers DataFrame
        })

    # Convert results to DataFrame for display
    outlier_df = pd.DataFrame(outlier_results)

    # Display the results as a table
    st.write("Outlier Detection Results:")
    st.dataframe(outlier_df[['Variable', 'Total Data Points', 'Number of Outliers', 'Percentage of Outliers']])

    # Selection for columns and chart type
    selected_columns = st.multiselect("Select Columns for Visualization", options=outlier_df['Variable'].tolist())
    chart_type = st.selectbox("Select Chart Type", ["Histogram", "Boxplot", "All"])

    if selected_columns:
        for column in selected_columns:
            with st.expander(f"Visualization for {column}", expanded=True):
                outlier_data = outlier_df[outlier_df['Variable'] == column]['Outliers Data'].values[0].to_pandas()

                # Plot histogram
                if chart_type in ["Histogram", "All"]:
                    fig_hist = px.histogram(outlier_data, x=column, title=f'Histogram of Outliers in {column}')
                    st.plotly_chart(fig_hist)

                # Plot boxplot
                if chart_type in ["Boxplot", "All"]:
                    fig_box = px.box(outlier_data, y=column, title=f'Boxplot of Outliers in {column}')
                    st.plotly_chart(fig_box)

# ----------------- Missing Values -----------------

import streamlit as st
import polars as pl
import pandas as pd
import plotly.express as px
import dabl
import math

def get_and_handle_missing_values(df):
    """
    This function calculates the count of missing values for each column,
    displays a table and a plot for visualization, and allows the user to 
    select an imputation method to handle the missing values using the `dabl` package.
    After imputation, it will also show a table of missing values count.
    The changes to the dataframe will only be applied when the 'Apply Changes' button is clicked.
    """
    # Convert Polars DataFrame to Pandas DataFrame (dabl works with Pandas)
    df_pandas = df.to_pandas()

    # Step 1: Get the count of missing values per column before imputation
    def get_missing_count(df_pandas):
        missing_data = df_pandas.isnull().sum().reset_index()
        missing_data.columns = ['Variable', 'Missing Values Count']
        return missing_data

    # Display the missing count before imputation
    missing_df_before = get_missing_count(df_pandas)
    st.write("Missing Values Count Before Imputation:")
    st.dataframe(missing_df_before)  # Table before imputation

    # Plot the missing values count before imputation
    fig_before = px.bar(missing_df_before, 
                        x="Variable", 
                        y="Missing Values Count", 
                        title="Missing Values Count per Column (Before Imputation)", 
                        labels={"Variable": "Column", "Missing Values Count": "Count of Missing Values"},
                        color="Missing Values Count",
                        color_continuous_scale="Viridis")
    st.plotly_chart(fig_before)

    # Step 2: Add an option to select an imputation technique or drop rows/columns
    imputation_method = st.selectbox(
        "Select Imputation Method",
        options=["None", "Auto", "Forward Fill", "Backward Fill", "Drop Rows with Missing Values", "Interpolation"]
    )

    # Step 3: Handle missing values according to the selected method, but don't apply yet
    df_imputed = df_pandas.copy()  # Start with a copy of the dataframe
    if imputation_method != "None":
        if imputation_method == "Auto":
            # Use dabl's imputation functionality which automatically decides the imputation method
            st.write("Imputation handled automatically by `dabl`.")
            df_imputed = dabl.clean(df_pandas)  # Auto-imputation by dabl
            
            # Ensure that there are no remaining missing values
            df_imputed = df_imputed.ffill()  # Forward fill to handle any remaining missing values

        elif imputation_method == "Forward Fill":
            df_imputed = df_pandas.ffill()

        elif imputation_method == "Backward Fill":
            df_imputed = df_pandas.bfill()

        elif imputation_method == "Drop Rows with Missing Values":
            # Drop all rows with any missing values
            df_imputed = df_pandas.dropna()

        elif imputation_method == "Interpolation":
            df_imputed = df_pandas.interpolate(method='linear')  # Linear interpolation

        # Display the missing values count after imputation
        missing_df_after = get_missing_count(df_imputed)
        st.write("Missing Values Count After Imputation:")
        st.dataframe(missing_df_after)  # Table after imputation

        # Plot the missing values count after imputation
        fig_after = px.bar(missing_df_after, 
                           x="Variable", 
                           y="Missing Values Count", 
                           title="Missing Values Count per Column (After Imputation)", 
                           labels={"Variable": "Column", "Missing Values Count": "Count of Missing Values"},
                           color="Missing Values Count",
                           color_continuous_scale="Viridis")
        st.plotly_chart(fig_after)

        # Display the DataFrame preview after imputation for verification
        st.write("Imputed DataFrame Preview:")
        st.dataframe(df_imputed.head())  # Show the imputed DataFrame in Streamlit

        # Step 4: Apply changes button
        apply_changes = st.button("Apply Changes")
        if apply_changes:
            # If button clicked, convert back to Polars and return the imputed DataFrame
            df = pl.from_pandas(df_imputed)
            st.write("Changes applied to the dataframe.")
        else:
            st.write("Changes are not yet applied. Click 'Apply Changes' to finalize.")

    else:
        st.write("No imputation method selected. Data is not modified.")

    return df  # Return the dataframe after imputation, only if changes were applied.
