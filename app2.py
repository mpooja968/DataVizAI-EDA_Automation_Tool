"""
title: ASCVIT V1 [AUTOMATIC STATISTICAL CALCULATION, VISUALIZATION AND INTERPRETATION TOOL]
author: stefanpietrusky
author_url: https://downchurch.studio/
version: 0.1
"""


import re
import subprocess

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px  
import plotly.graph_objects as go  
import seaborn as sns
from matplotlib.patches import Patch
from scipy import stats
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, LogisticRegression
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import streamlit as st
import polars as pl
from plotly.subplots import make_subplots
import scipy.cluster.hierarchy as sch
import tsfresh
from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import impute

from data_overview import display_data_info
from schema_manipulation import select_timestamp_column
from schema_manipulation import data_manipulation
from schema_manipulation import data_aggregation
from descriptive_stats import descriptive_statistics
from outlier_detection import detect_outliers
from outlier_detection import get_and_handle_missing_values
from feature_extraction import feature_extraction


# Function to communicate with the LLM
#def query_llm_via_cli(input_text):
#    """Sends the question and context to the LLM and receives a response"""
#    try:
#        process = subprocess.Popen(
#            ["ollama", "run", "llama3.1"],
#            stdin=subprocess.PIPE,
#            stdout=subprocess.PIPE,
#            stderr=subprocess.PIPE,
#            text=True,
#            encoding='utf-8',
#            errors='ignore',
#            bufsize=1
#        )
#        stdout, stderr = process.communicate(input=f"{input_text}\n", timeout=40)

#        if process.returncode != 0:
#            return f"Error in the model request: {stderr.strip()}"

#        response = re.sub(r'\x1b\[.*?m', '', stdout)
#        return extract_relevant_answer(response)

#    except subprocess.TimeoutExpired:
#        process.kill()
#        return "Timeout for the model request"
#    except Exception as e:
#        return f"An unexpected error has occurred: {str(e)}"

#def extract_relevant_answer(full_response):
#    response_lines = full_response.splitlines()
#    if response_lines:
#        return "\n".join(response_lines).strip()
#    return "No answer received"

from fpdf import FPDF # type: ignore
import os


def generate_pdf(data, file_name="data_overview_report.pdf"):
    """
    Generates a structured PDF report for data overview with formatted tables.

    Parameters:
        data (polars.DataFrame): The dataset to be included in the report.
        file_name (str): The name of the output PDF file.
    """
    if not isinstance(data, pl.DataFrame):
        raise TypeError("Expected a Polars DataFrame, but got a different type.")

    pdf = FPDF(orientation="L", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Title
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Data Overview Report", ln=True, align="C")
    pdf.ln(10)

    # Summary Section
    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 8, f"Number of Rows: {data.height}", ln=True)
    pdf.cell(0, 8, f"Number of Columns: {data.width}", ln=True)
    pdf.ln(5)

    # Data Types Table
    pdf.set_font("Arial", "B", 10)
    pdf.cell(80, 10, "Column Name", border=1, align="C")
    pdf.cell(50, 10, "Data Type", border=1, align="C")
    pdf.ln()

    pdf.set_font("Arial", "", 10)
    for col, dtype in data.schema.items():
        pdf.cell(80, 8, col, border=1, align="C")
        pdf.cell(50, 8, str(dtype), border=1, align="C")
        pdf.ln()

    pdf.ln(10)

    # Data Preview Table
    pdf.set_font("Arial", "B", 10)
    preview_cols = data.columns[:8]  # Limit to first 8 columns for better readability
    col_width = 270 / len(preview_cols)  # Dynamic column width

    for col in preview_cols:
        pdf.cell(col_width, 10, col, border=1, align="C")
    pdf.ln()

    pdf.set_font("Arial", "", 10)
    for i in range(min(5, data.height)):  # Display first 5 rows
        row = data.row(i)
        for value in row[:8]:  
            pdf.cell(col_width, 8, str(value)[:20], border=1, align="C")  # Trim long values
        pdf.ln()

    # Save the file
    pdf.output(file_name)
    return file_name


# Main function to start the app
def main():
    st.title("EDA AUTOMATION TOOL")

    # Sidebar for file upload
    st.sidebar.title("Settings")
    uploaded_file = st.sidebar.file_uploader("**Upload your data file**", type=["csv", "xlsx"])

    # Initialize session state for controlling button states and data storage
    if 'data_loaded' not in st.session_state:
        st.session_state['data_loaded'] = False
    if 'df' not in st.session_state:
        st.session_state['df'] = None
    if 'numerical_columns' not in st.session_state:
        st.session_state['numerical_columns'] = []
    if 'categorical_columns' not in st.session_state:
        st.session_state['categorical_columns'] = []
    if 'show_data' not in st.session_state:
        st.session_state['show_data'] = False
    if 'show_analysis' not in st.session_state:
        st.session_state['show_analysis'] = None

    # Check if a file has been uploaded
    if uploaded_file:
        # Read file (either CSV or Excel)
        if uploaded_file.name.endswith('.csv'):
            temp_df = pl.read_csv(uploaded_file, n_rows=3, try_parse_dates=True)
        else:
            temp_df = pl.read_excel(uploaded_file)

        # Add "All" option for selecting columns
        select_cols = st.sidebar.multiselect("Select variables:", options=["All"] + list(temp_df.columns))

        # Automatically select all columns if "All" is chosen
        if "All" in select_cols:
            select_cols = list(temp_df.columns)

        # Load Data button
        if st.sidebar.button("Load Data") and select_cols:
            # Load the selected columns from the uploaded file
            if uploaded_file.name.endswith('.csv'):
                st.session_state['df'] = pl.read_csv(uploaded_file, columns=select_cols, try_parse_dates=True)
            else:
                st.session_state['df'] = pl.read_excel(uploaded_file)

            # Reset numeric and categorical columns based on the new record
            st.session_state['numerical_columns'] = [col for col, dtype in st.session_state['df'].schema.items() if dtype in (pl.Float32, pl.Float64)]
            st.session_state['categorical_columns'] = [col for col, dtype in st.session_state['df'].schema.items() if dtype == pl.Utf8]

            # Update session state
            st.session_state['last_uploaded_file'] = uploaded_file.name  # Store the filename
            st.session_state['show_data'] = True
            st.session_state['data_loaded'] = True

            # Display a horizontal line
            st.sidebar.markdown("---")  # Add horizontal line

    # Check if data is loaded before proceeding with any analysis
    if st.session_state['data_loaded']:
        # Add Drop Columns functionality
        with st.sidebar:
            st.subheader("Drop Columns")
            with st.expander("Click to drop columns"):
                # Multiselect for choosing columns to drop
                selected_columns = st.multiselect("Select columns to drop:", options=st.session_state['df'].columns)

                # Drop button
                if st.button("Drop Selected Columns"):
                    if selected_columns:
                        # Drop the selected columns
                        st.session_state['df'] = st.session_state['df'].drop(selected_columns)
                        st.success(f"Columns {', '.join(selected_columns)} dropped successfully!")
                    else:
                        st.warning("No columns selected to drop.")

    # Check if data is loaded before proceeding with any analysis
    if st.session_state['data_loaded']:
        # Sidebar buttons for navigating to different sections
        if st.sidebar.button("Data Overview"):
            st.session_state['show_analysis'] = None
            st.session_state['show_data'] = True

        if st.sidebar.button("Data Manipulation"):
            st.session_state['show_data'] = False
            st.session_state['show_analysis'] = 'Data Manipulation'

        if st.sidebar.button("Visualizations"):
            st.session_state['show_data'] = False
            st.session_state['show_analysis'] = 'Visualizations'

        if st.sidebar.button("Outlier Detection"):
            st.session_state['show_data'] = False
            st.session_state['show_analysis'] = 'Outlier Detection'

        if st.sidebar.button("Feature Extraction"):
            st.session_state['show_data'] = False
            st.session_state['show_analysis'] = 'Feature Extraction'

        # Show Schema Manipulation Section with Tabs
        if st.session_state.get('show_analysis') == 'Data Manipulation':
            st.header("Data Manipulation")
            # Tab for selecting between Datatype Conversion, Data Aggregation, and Data Profiling
            tab_selection = st.radio("Select an operation:", ("Timestamp Selection","Datatype Conversion", "Data Aggregation"))

            if tab_selection == "Timestamp Selection":
                st.subheader("Timestamp Selection")
                if st.session_state['df'] is not None:
                    # Call the timestamp selection function
                    datetime_column = select_timestamp_column(st.session_state['df'])
                    if datetime_column:
                        st.write(f"Using '{datetime_column}' as the timestamp column.")
                else:
                    st.error("Data is not loaded. Please load data first.")

            elif tab_selection == "Datatype Conversion":
                st.subheader("Datatype Conversion")
                if st.session_state['df'] is not None:
                    # Update the dataframe in session state after schema manipulation
                    data_manipulation(st.session_state['df'], st.session_state['numerical_columns'], st.session_state['categorical_columns'])
                else:
                    st.error("Data is not loaded. Please load data first.")

            elif tab_selection == "Data Aggregation":
                st.subheader("Data Aggregation")
                if st.session_state['df'] is not None:
                    # Update the dataframe in session state after aggregation
                    data_aggregation(st.session_state['df'], st.session_state['numerical_columns'], st.session_state['categorical_columns'])
                else:
                    st.error("Data is not loaded. Please load data first.")

        # Show data overview
        if st.session_state.get('show_data'):
            if st.session_state['df'] is not None:
                st.header("Data Overview")

                # User input to adjust the number of preview rows
                num_rows = st.number_input(
                    "Select number of rows to display:", 
                    min_value=1, 
                    max_value=len(st.session_state['df']), 
                    value=5, 
                    step=1
                )

                # Display Data Preview with selected rows
                st.write("### Data Preview")
                st.dataframe(st.session_state['df'].head(num_rows))  # Dynamic row selection

                # Call function to display detailed data info
                display_data_info(st.session_state['df'])

                # PDF Download Button
                pdf_path = generate_pdf(st.session_state['df'])
                with open(pdf_path, "rb") as file:
                    st.download_button(label="Download PDF Report", data=file, file_name="Data_Overview_Report.pdf", mime="application/pdf")


            else:
                st.error("No data available. Please load data.")

        # Show analysis (Descriptive Statistics, Outlier Detection, etc.)
        if st.session_state.get('show_analysis'):
            if st.session_state['df'] is not None:
                if st.session_state['show_analysis'] == 'Visualizations':
                    st.header("Visualizations")
                    descriptive_statistics(st.session_state['df'], st.session_state['numerical_columns'])
                elif st.session_state.get('show_analysis') == 'Outlier Detection':
                    #st.header("Data Manipulation")
                    # Tab for selecting between Datatype Conversion, Data Aggregation, and Data Profiling
                    tab_selection = st.radio("Select an operation:", ("Outlier Detection", "Missing Values"))
                    if tab_selection == "Outlier Detection":
                        st.subheader("Outlier Detection")
                        if st.session_state['df'] is not None:
                            # Update the dataframe after outlier removal
                            detect_outliers(st.session_state['df'])
                        else:
                            st.error("Data is not loaded. Please load data first.")
                    elif tab_selection == "Missing Values":
                        st.subheader("Missing Values Handling")
                        if st.session_state['df'] is not None:
                            # Update the dataframe after missing value handling
                            get_and_handle_missing_values(st.session_state['df'])
                        else:
                            st.error("Data is not loaded. Please load data first.")
                    
                elif st.session_state.get('show_analysis') == 'Feature Extraction':
                    st.header("Feature Extraction")
                    if st.session_state['df'] is not None:
                        # Update the dataframe after feature extraction
                        feature_extraction(st.session_state['df'])
                    else:
                        st.error("Data is not loaded. Please load data first.")

            else:
                st.error("No data available. Please load data.")

    else:
        st.sidebar.info("Please select variables and click 'Load Data' to enable further functionalities.")

if __name__ == "__main__":
    main()
