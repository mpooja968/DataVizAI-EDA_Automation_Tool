# DataVizAI-EDA_Automation_Tool
DataVizAI (Data Visualization &amp; AI-powered Insights) is an AI-augmented Exploratory Data Analysis tool built with Python, Streamlit, Polars, Plotly, and LangChain. It automates data preprocessing, statistical analysis, visualization, reporting, and AI-powered insights – reducing EDA time from hours to minutes.

**🛠 Tech Stack**

- Python
- Streamlit
- Polars
- Pandas
- NumPy
- Plotly
- Seaborn
- SciPy
- Scikit-learn
- PyOD
- LangChain
- ReportLab

**Getting Started**

1. Clone the repository:

```bash
git clone https://github.com/yourusername/DataVizAI.git
cd DataVizAI
```

2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the root directory and add your GROQ API key:

```env
GROQ_API_KEY=your_groq_api_key
```

4. Running the application

```bash
streamlit run app.py
```


# **✨ Features**

📂 Data Upload & Preprocessing – CSV/XLSX support, schema detection, type conversion, timestamp selection

📊 Interactive Visualizations – Histograms, boxplots, scatter plots, pair plots, correlation matrices (Plotly & Seaborn)

🛠 Data Cleaning – Missing value imputation (DABL, interpolation, mean, median, mode), outlier detection (IQR, Z-Score, MAD, Isolation Forest)

⚙ Feature Engineering & Scaling – Derived features, Min-Max, Standard, Robust scaling

🤖 LLM Integration – Natural language querying, visualization interpretation, hypothesis testing (LangChain + Llama 3)

📄 Automated PDF Reports – Full dataset overview, stats, visualizations, scrollable data previews


# **🔮 Future Enhancements**

Domain-specific LLM fine-tuning

Automated ML pipeline integration

Full dashboard UI for non-technical users
