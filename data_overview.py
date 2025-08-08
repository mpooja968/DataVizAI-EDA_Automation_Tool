import streamlit as st
import polars as pl
import pandas as pd
from groq import Groq
import json

# Load API Key from Streamlit Secrets
def get_groq_client():
    api_key = st.secrets["groq"]["api_key"]
    return Groq(api_key=api_key)

def ask_llm(question, context):
    """Query the LLM with dataset-related questions."""
    client = get_groq_client()
    response = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[
            {"role": "system", "content": "You are an expert data scientist. Answer based on the dataset provided."},
            {"role": "user", "content": f"Dataset Info:\n{context}\n\nQuestion: {question}"}
        ]
    )
    return response.choices[0].message.content.strip()

def display_data_info(df):
    st.write("### Data Types")
    dtype_df = pl.DataFrame({
        "Column Name": df.columns,
        "Data Type": [str(df[col].dtype) for col in df.columns]
    })
    st.dataframe(dtype_df.to_pandas())

    # Data Description
    st.write("### Data Description")
    desc = df.describe()
    st.dataframe(desc.to_pandas())

    # Variable Identification
    numerical_columns = [col for col in df.columns if df[col].dtype in [pl.Int64, pl.Float64]]
    categorical_columns = [col for col in df.columns if df[col].dtype == pl.Utf8]
    timeseries_columns = [col for col in df.columns if isinstance(df[col].dtype, pl.Datetime)]

    # Display as DataFrames
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Numerical Variables")
        if numerical_columns:
            st.dataframe(pd.DataFrame({"Numerical Columns": numerical_columns}))
        else:
            st.write("No numerical variables identified.")

        st.subheader("Timeseries Variables")
        if timeseries_columns:
            st.dataframe(pd.DataFrame({"Timeseries Columns": timeseries_columns}))
        else:
            st.write("No timeseries variables identified.")

    with col2:
        st.subheader("Categorical Variables")
        if categorical_columns:
            st.dataframe(pd.DataFrame({"Categorical Columns": categorical_columns}))
        else:
            st.write("No categorical variables identified.")

    # LLM Q&A Section
    st.write("### Ask the LLM about the dataset")
    user_query = st.text_input("Enter your question:")
    if st.button("Get Answer"):
        # A summarized version of dataset info for context
        dataset_summary = f"Columns: {df.columns}\nNumerical: {numerical_columns}\nCategorical: {categorical_columns}\nTimeseries: {timeseries_columns}"
        answer = ask_llm(user_query, dataset_summary)
        st.write("### LLM Response:")
        st.success(answer)

    return df, numerical_columns, categorical_columns, timeseries_columns
