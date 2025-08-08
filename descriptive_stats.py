import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from sklearn.linear_model import LinearRegression
from plotly.subplots import make_subplots
import numpy as np
import polars as pl

def descriptive_statistics(df, numerical_columns):
    # Step 1: Selection of plot types
    plot_category = st.selectbox("Select the plot category:", ["Univariate", "Bivariate", "Multivariate"])

    # Step 2: Based on category, display available chart types
    if plot_category == "Univariate":
        chart_types = st.multiselect("Select the diagram:", ["Histogram", "Boxplot"])
    elif plot_category == "Bivariate":
        chart_types = st.multiselect("Select the diagram:", ["Scatter Plot"])
    elif plot_category == "Multivariate":
        chart_types = st.multiselect("Select the diagram:", ["Pairplot", "Correlation matrix"])

    display_chart_description(chart_types)

    # Step 3: Selection of variables for each chart type
    selected_vars = {}
    for chart_type in chart_types:
        selected_vars[chart_type] = select_variables(chart_type, numerical_columns, plot_category)

    # Store the selected variables in session state
    st.session_state.selected_vars = selected_vars

    # Step 4: Logarithmic scale option
    apply_log_scale = get_log_scale_option(chart_types)

    # Step 5: Trigger chart creation
    if st.button("Create diagrams"):
        for chart_type in chart_types:
            create_diagram(chart_type, df, selected_vars[chart_type], apply_log_scale)


def display_chart_description(chart_types):
    """Display description of selected chart types."""
    descriptions = {
        "Histogram": "A histogram shows the distribution of a numerical variable. It helps to recognize how frequently certain values occur in the data and whether there are patterns, such as a normal distribution.",
        "Boxplot": "A boxplot shows the distribution of a numerical variable through its quartiles. It helps to identify outliers and visualize the dispersion of the data.",
        "Pairplot": "A pairplot shows the relationships between different numerical variables through scatterplots. It helps to identify possible relationships between variables.",
        "Correlation matrix": "The correlation matrix shows the linear relationships between numerical variables. A positive correlation indicates that high values in one variable also correlate with high values in another.",
        "Scatter Plot": "A scatter plot displays values for typically two variables for a set of data. It helps to visualize relationships and correlations between variables."
    }
    for chart_type in chart_types:
        st.markdown(f"### {chart_type}")
        st.markdown(descriptions.get(chart_type, ""))


def select_variables(chart_type, numerical_columns, plot_category):
    """Allow the user to select variables based on the plot category and chart type."""
    if plot_category == "Multivariate" and chart_type in ["Pairplot", "Correlation matrix"]:
        return st.multiselect(f"Select variables for {chart_type}:", ["All"] + numerical_columns, default="All")
    elif plot_category == "Bivariate" and chart_type == "Scatter Plot":
        # Ensure exactly 2 variables can be selected for the scatter plot
        return st.multiselect(f"Select exactly two variables for {chart_type}:", numerical_columns, default=numerical_columns[:2], max_selections=2)
    elif plot_category == "Univariate" and chart_type in ["Histogram", "Boxplot"]:
        return st.multiselect(f"Select variables for {chart_type}:", ["All"] + numerical_columns, default="All")
    else:
        return [st.selectbox(f"Select a variable for {chart_type}:", numerical_columns)]


def get_log_scale_option(chart_types):
    """Get the user's preference for applying log scale."""
    if not any(chart in chart_types for chart in ["Correlation matrix"]):
        return st.checkbox("Apply logarithmic scaling?", value=False)
    else:
        return False


def create_diagram(chart_type, df, selected_vars, apply_log_scale):
    """Create the appropriate chart based on user selection."""
    if chart_type == "Histogram":
        plot_histogram(df, apply_log_scale)
    elif chart_type == "Boxplot":
        plot_boxplot(df, apply_log_scale)
    elif chart_type == "Pairplot":
        plot_pairplot(df, selected_vars)
    elif chart_type == "Correlation matrix":
        plot_correlation_matrix(df, selected_vars)
    elif chart_type == "Scatter Plot":
        if len(selected_vars) == 2:
            plot_scatter_plot(df, selected_vars[0], selected_vars[1], apply_log_scale)
        else:
            st.warning("Please select exactly two variables for the scatter plot.")

# ----------------- Plotting Functions -----------------

import plotly.graph_objects as go
import numpy as np
import streamlit as st


def plot_histogram(df, apply_log_scale):
    """Create and display a histogram with optional logarithmic scaling."""
    # Fetch selected variables from session state
    selected_vars = st.session_state.get('selected_vars', {}).get('Histogram', [])
    if not selected_vars:
        st.warning("No variables selected for the histogram.")
        return

    for variable in selected_vars:
        cleaned_data = df.select(pl.col(variable).drop_nulls())[variable].to_list()
        mean_value = np.mean(cleaned_data)
        median_value = np.median(cleaned_data)
        std_value = np.std(cleaned_data)
        std_upper_1 = mean_value + std_value
        std_lower_1 = mean_value - std_value
        std_upper_2 = mean_value + 2 * std_value
        std_lower_2 = mean_value - 2 * std_value
        std_upper_3 = mean_value + 3 * std_value
        std_lower_3 = mean_value - 3 * std_value

        # Create and customize histogram
        fig = go.Figure()

        # Plot the histogram
        fig.add_trace(go.Histogram(x=cleaned_data, opacity=0.3, marker=dict(color='blue'),name=variable))

        # Plot mean, median, and standard deviation lines on the primary axis
        fig.add_trace(go.Scatter(x=[mean_value, mean_value], y=[0, 0.1 * max(cleaned_data)], 
                                 mode='lines', line=dict(color='red', dash='dash'),
                                 name=f'Mean: {mean_value:.2f}', showlegend=True))

        fig.add_trace(go.Scatter(x=[median_value, median_value], y=[0, 0.1 * max(cleaned_data)], 
                                 mode='lines', line=dict(color='green', dash='solid'),
                                 name=f'Median: {median_value:.2f}', showlegend=True))

        fig.add_trace(go.Scatter(x=[std_upper_1, std_upper_1], y=[0, 0.1 * max(cleaned_data)], 
                                 mode='lines', line=dict(color='orange', dash='dot'),
                                 name=f'+1 Std: {std_upper_1:.2f}', showlegend=True))

        fig.add_trace(go.Scatter(x=[std_lower_1, std_lower_1], y=[0, 0.1 * max(cleaned_data)], 
                                 mode='lines', line=dict(color='orange', dash='dot'),
                                 name=f'-1 Std: {std_lower_1:.2f}', showlegend=True))

        # Add ±2 and ±3 standard deviation lines
        fig.add_trace(go.Scatter(x=[std_upper_2, std_upper_2], y=[0, 0.1 * max(cleaned_data)], 
                                 mode='lines', line=dict(color='purple', dash='dot'),
                                 name=f'+2 Std: {std_upper_2:.2f}', showlegend=True))

        fig.add_trace(go.Scatter(x=[std_lower_2, std_lower_2], y=[0, 0.1 * max(cleaned_data)], 
                                 mode='lines', line=dict(color='purple', dash='dot'),
                                 name=f'-2 Std: {std_lower_2:.2f}', showlegend=True))

        fig.add_trace(go.Scatter(x=[std_upper_3, std_upper_3], y=[0, 0.1 * max(cleaned_data)], 
                                 mode='lines', line=dict(color='brown', dash='dot'),
                                 name=f'+3 Std: {std_upper_3:.2f}', showlegend=True))

        fig.add_trace(go.Scatter(x=[std_lower_3, std_lower_3], y=[0, 0.1 * max(cleaned_data)], 
                                 mode='lines', line=dict(color='brown', dash='dot'),
                                 name=f'-3 Std: {std_lower_3:.2f}', showlegend=True))

        # Define secondary y-axis for descriptive statistics
        fig.update_layout(
            title=f"Histogram of {variable}",
            height=400,
            barmode='overlay',
            showlegend=True,
            yaxis2=dict(
                overlaying='y',
                side='right',
                showgrid=False,  # Optional: remove gridlines on secondary axis
            ),
            xaxis=dict(
                title='Value'
            ),
            yaxis=dict(
                title='Count'
            )
        )

        # Update traces to use the secondary y-axis for descriptive statistics
        fig.data[1].update(yaxis='y2')  # Mean line on secondary y-axis
        fig.data[2].update(yaxis='y2')  # Median line on secondary y-axis
        fig.data[3].update(yaxis='y2')  # +1 Std line on secondary y-axis
        fig.data[4].update(yaxis='y2')  # -1 Std line on secondary y-axis
        fig.data[5].update(yaxis='y2')  # +2 Std line on secondary y-axis
        fig.data[6].update(yaxis='y2')  # -2 Std line on secondary y-axis
        fig.data[7].update(yaxis='y2')  # +3 Std line on secondary y-axis
        fig.data[8].update(yaxis='y2')  # -3 Std line on secondary y-axis

        # Optionally apply logarithmic scale
        if apply_log_scale:
            fig.update_yaxes(type="log", row=1, col=1)

        st.plotly_chart(fig)

import numpy as np
import plotly.graph_objects as go
import polars as pl
import streamlit as st

def plot_boxplot(df, apply_log_scale):
    """Create and display a boxplot with optional logarithmic scaling."""
    # Fetch selected variables from session state
    selected_vars = st.session_state.get('selected_vars', {}).get('Boxplot', [])
    if not selected_vars:
        st.warning("No variables selected for the boxplot.")
        return

    for variable in selected_vars:
        cleaned_data = df.select(pl.col(variable).drop_nulls())[variable].to_list()
        
        if not cleaned_data:
            st.warning(f"No valid data for {variable}. Skipping.")
            continue
        
        # Compute statistics
        mean_value = np.mean(cleaned_data)
        median_value = np.median(cleaned_data)
        std_value = np.std(cleaned_data)
        q1 = np.percentile(cleaned_data, 25)
        q3 = np.percentile(cleaned_data, 75)
        iqr = q3 - q1
        lower_whisker = max(min(cleaned_data), q1 - 1.5 * iqr)
        upper_whisker = min(max(cleaned_data), q3 + 1.5 * iqr)

        # Corrected ±1, ±2, ±3 standard deviation lines
        std_upper_1 = mean_value + std_value
        std_lower_1 = mean_value - std_value
        std_upper_2 = mean_value + 2 * std_value
        std_lower_2 = mean_value - 2 * std_value
        std_upper_3 = mean_value + 3 * std_value
        std_lower_3 = mean_value - 3 * std_value

        # Create figure
        fig = go.Figure()

        # Boxplot
        fig.add_trace(go.Box(
            y=cleaned_data, boxpoints='outliers', name=variable,
            marker=dict(color='blue', line=dict(color='black', width=1)),
            opacity=0.7
        ))

        # Add statistical reference lines
        fig.add_trace(go.Scatter(x=[-0.2, 0.2], y=[mean_value, mean_value], mode='lines',
                                 line=dict(color='red', dash='dash'), name=f'Mean: {mean_value:.2f}'))
        fig.add_trace(go.Scatter(x=[-0.2, 0.2], y=[median_value, median_value], mode='lines',
                                 line=dict(color='green', dash='solid'), name=f'Median: {median_value:.2f}'))

        # Add ±1, ±2, ±3 standard deviation lines
        fig.add_trace(go.Scatter(x=[-0.2, 0.2], y=[std_upper_1, std_upper_1], mode='lines',
                                 line=dict(color='orange', dash='dot'), name=f'+1 Std: {std_upper_1:.2f}'))
        fig.add_trace(go.Scatter(x=[-0.2, 0.2], y=[std_lower_1, std_lower_1], mode='lines',
                                 line=dict(color='orange', dash='dot'), name=f'-1 Std: {std_lower_1:.2f}'))
        
        fig.add_trace(go.Scatter(x=[-0.2, 0.2], y=[std_upper_2, std_upper_2], mode='lines',
                                 line=dict(color='purple', dash='dot'), name=f'+2 Std: {std_upper_2:.2f}'))
        fig.add_trace(go.Scatter(x=[-0.2, 0.2], y=[std_lower_2, std_lower_2], mode='lines',
                                 line=dict(color='purple', dash='dot'), name=f'-2 Std: {std_lower_2:.2f}'))

        fig.add_trace(go.Scatter(x=[-0.2, 0.2], y=[std_upper_3, std_upper_3], mode='lines',
                                 line=dict(color='brown', dash='dot'), name=f'+3 Std: {std_upper_3:.2f}'))
        fig.add_trace(go.Scatter(x=[-0.2, 0.2], y=[std_lower_3, std_lower_3], mode='lines',
                                 line=dict(color='brown', dash='dot'), name=f'-3 Std: {std_lower_3:.2f}'))

        fig.update_layout(title=f"Boxplot of {variable}", height=400, showlegend=True)

        if apply_log_scale:
            fig.update_yaxes(type="log")

        st.plotly_chart(fig)




def plot_pairplot(df, selected_vars):
    """Create and display a pairplot with regression lines."""
    num_vars = len(selected_vars)
    fig = make_subplots(rows=num_vars, cols=num_vars, subplot_titles=[f"{var1} vs {var2}" for var1 in selected_vars for var2 in selected_vars])
    
    for i, var1 in enumerate(selected_vars):
        for j, var2 in enumerate(selected_vars):
            if var1 != var2:
                non_nan_data = df.select([pl.col(var1), pl.col(var2)]).drop_nulls()
                fig.add_trace(go.Scatter(x=non_nan_data[var1].to_numpy(), y=non_nan_data[var2].to_numpy(), mode='markers', name='Data Points', marker=dict(color='blue', opacity=0.6)), row=i+1, col=j+1)
                
                X = non_nan_data[var1].to_numpy().reshape(-1, 1)
                y = non_nan_data[var2].to_numpy()
                model = LinearRegression()
                model.fit(X, y)
                y_fit = model.predict(X)
                
                fig.add_trace(go.Scatter(x=X.flatten(), y=y_fit, mode='lines', name='Regression Line', line=dict(color='red', dash='dash')), row=i+1, col=j+1)

    fig.update_layout(title="Pairplot with Regression Lines", height=600, showlegend=False)
    st.plotly_chart(fig)

def plot_correlation_matrix(df, selected_vars):
    """Create and display a correlation matrix heatmap with scale from -1 to +1."""
    if len(selected_vars) > 1:
        corr_matrix = df.select(selected_vars).to_pandas().corr()
        fig = px.imshow(corr_matrix, 
                        color_continuous_scale='RdBu',  # Red-Blue color scale
                        zmin=-1, zmax=1)  # Set color scale range from -1 to +1
        fig.update_layout(title="Correlation Matrix", height=500)
        st.plotly_chart(fig)
    else:
        st.warning("Please select more than one variable for the correlation matrix.")

def plot_scatter_plot(df, var_x, var_y, apply_log_scale):
    """Create and display a scatter plot."""
    non_nan_data = df.select([pl.col(var_x), pl.col(var_y)]).drop_nulls()
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=non_nan_data[var_x].to_numpy(), y=non_nan_data[var_y].to_numpy(), mode='markers', name='Data Points', marker=dict(color='blue', opacity=0.6)))

    fig.update_layout(title=f"Scatter Plot: {var_x} vs {var_y}", height=400)

    if apply_log_scale:
        fig.update_xaxes(type="log")
        fig.update_yaxes(type="log")

    st.plotly_chart(fig)
