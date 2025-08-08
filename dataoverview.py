import streamlit as st
import polars as pl
import pandas as pd
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, ServiceContext
from llama_index.core.query_engine import PandasQueryEngine
from llama_index.llms import LlamaCPP

# Load Llama Model
def load_llama_model():
    llm = LlamaCPP(
        model_url="https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_K_M.gguf",
        temperature=0.7,
        max_tokens=256
    )
    return llm

# Function to load dataset
def load_dataset(uploaded_file):
    try:
        df = pl.read_csv(uploaded_file)
        return df
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return None

# Convert Polars DataFrame to Pandas (for LlamaIndex)
def convert_to_pandas(df):
    return df.to_pandas()

# Initialize LlamaIndex with Pandas DataFrame
def initialize_llama_index(df_pandas):
    query_engine = PandasQueryEngine(df=df_pandas)
    return query_engine

# Streamlit UI
st.title("📊 Chat with Your Dataset using Llama")

uploaded_file = st.file_uploader("Upload your CSV dataset", type=["csv"])

if uploaded_file:
    df = load_dataset(uploaded_file)
    
    if df is not None:
        st.success("Dataset loaded successfully!")
        st.dataframe(df.head())  # Show sample data
        
        df_pandas = convert_to_pandas(df)
        query_engine = initialize_llama_index(df_pandas)
        
        st.subheader("Ask questions about your dataset:")
        user_query = st.text_input("Enter your question:")

        if st.button("Ask"):
            if user_query:
                response = query_engine.query(user_query)
                st.write("🤖 **Llama Says:**", response)
            else:
                st.warning("Please enter a question.")

