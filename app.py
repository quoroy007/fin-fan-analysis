import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# Set page config
st.set_page_config(
    page_title="Fin Fan Project Analysis",
    page_icon="💧",
    layout="wide"
)

# Page title
st.title("Fin Fan Project Analysis Dashboard")
st.write("Interactive analysis of water and energy impacts with adjustable safety factors")

# Function to load and prepare data
@st.cache_data
def load_data():
    try:
        # Try to load from the data directory
        df = pd.read_csv('data/tons.csv', header=0)
    except:
        # Fallback to URL if file not found
        st.warning("Local file not found. Please ensure 'tons.csv' is in the data directory.")
        return None
    
    # Clean column names
    # This depends on the exact structure of your CSV
    # We'll rename based on known structure
    columns = {
        '': 'Year',
        '_1': 'Context',
        '_2': 'DataType',
        '_3': 'Production',
        '_4': 'ProducedUnits',
        'No Fin Fan': 'WaterNoFinFan',
        '_5': 'WaterEffNoFinFan',
        '_6': 'WaterPerfImprovement',
        'With Fin Fan': 'WaterWithFinFan',
        '_7': 'WaterEffWithFinFan',
        'No Fin Fan_1': 'EnergyNoFinFan',
        '_8': 'EnergyEffNoFinFan',
        '_9': 'EnergyPerfImprovement',
        'With Fin Fan_1': 'EnergyWithFinFan',
        '_10': 'EnergyEffWithFinFan'
    }
    
    # Rename columns
    df = df.rename(columns=columns)
    
    # Convert Year to numeric
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
    
    # Drop header row that got included as data
    df = df[df['Year'].notna() & (df['Year'] != 'Year')]
    
    # Convert numeric columns
    numeric_columns = ['Year', 'Production', 'WaterNoFinFan', 'WaterEffNoFinFan', 'WaterPerfImprovement',
                      'WaterWithFinFan', 'WaterEffWithFinFan', 'EnergyNoFinFan', 'EnergyEffNoFinFan',
                      'EnergyPerfImprovement', 'EnergyWithFinFan', 'EnergyEffWithFinFan']
    
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df

# Load data
data = load_data()

# Display raw data in an expander
with st.expander("View Raw Data"):
    if data is not None:
        st.dataframe(data)
    else:
        st.error("Failed to load data. Please check the file path.")

# If data is loaded successfully, continue with the app
if data is not None:
    # Create safety factor sliders
    st.sidebar.header("Safety Factor Adjustments")
    st.sidebar.info("Adjust safety factors to see how changes affect water and energy metrics")
    
    water_safety_factor = st.sidebar.slider(
        "Water Volume Safety Factor (%)",
        min_value=1,
        max_value=10,
        value=5,
        help="Increase water volume by this percentage to account for uncertainties"
    )
    
    energy_safety_factor = st.sidebar.slider(
        "Energy Volume Safety Factor (%)",
        min_value=1,
        max_value=10,
        value=5,
        help="Increase energy volume by this percentage to account for uncertainties"
    )
    
    # Apply safety factors to data
    def apply_safety_factors(df, water_factor, energy_factor):
        # Create a copy of the dataframe
        adjusted_df = df.copy()
        
        # Apply water safety factor
        water_factor_multiplier = 1 + (water_factor / 100)
        adjusted_df['WaterNoFinFan_Adjusted'] = adjusted_df['WaterNoFinFan'] * water_factor_multiplier
        adjusted_df['WaterWithFinFan_Adjusted'] = adjusted_df['WaterWithFinFan'] * water_factor_multiplier
        
        # Recalculate water efficiency
        adjusted_df['WaterEffNoFinFan_Adjusted'] = adjusted_df['WaterNoFinFan_Adjusted'] / adjusted_df['Production']
        adjusted_df['WaterEffWithFinFan_Adjusted'] = adjusted_df['WaterWithFinFan_Adjusted'] / adjusted_df['Production']
        
        # Apply energy safety factor
        energy_factor_multiplier = 1 + (energy_factor / 100)
        adjusted_df['EnergyNoFinFan_Adjusted'] = adjusted_df['EnergyNoFinFan'] * energy_factor_multiplier
        adjusted_df['EnergyWithFinFan_Adjusted'] = adjusted_df['EnergyWithFinFan'] * energy_factor_multiplier
        
        # Recalculate energy efficiency
        adjusted_df['EnergyEffNoFinFan_Adjusted'] = adjusted_df['EnergyNoFinFan_Adjusted'] / adjusted_df['Production']
        adjusted_df['EnergyEffWithFinFan_Adjusted'] = adjusted_df['EnergyWithFinFan_Adjusted'] / adjusted_df['Production']
        
        return adjusted_df
    
    # Apply safety factors
    adjusted_data = apply_safety_factors(data, water_safety_factor, energy_safety_factor)
    
    # Create visualizations
    st.header("Visualizations")
    
    # Water and Energy tab selector
    tab1, tab2 = st.tabs(["Water Analysis", "Energy Analysis"])
    
    with tab1:
        st.subheader("Water Withdrawals and Efficiency")
        
        # Create water volume subplot
        fig1 = make_subplots(
            rows=2, 
            cols=1,
            subplot_titles=("Water Withdrawals (m³)", "Water Efficiency (m³/ton)"),
            vertical_spacing=0.15,
            shared_xaxes=True
        )
        
        # Add water withdrawals (volumes)
        fig1.add_trace(
            go.Bar(
                x=adjusted_data['Year'],
                y=adjusted_data['WaterNoFinFan_Adjusted'],
                name="No Fin Fan",
                marker_color='royalblue',
                hovertemplate='Year: %{x}<br>Water: %{y:.0f} m³<br>Type: %{text}<extra></extra>',
                text=adjusted_data['DataType']
            ),
            row=1, col=1
        )
        
        fig1.add_trace(
            go.Bar(
                x=adjusted_data['Year'],
                y=adjusted_data['WaterWithFinFan_Adjusted'],
                name="With Fin Fan",
                marker_color='lightgreen',
                hovertemplate='Year: %{x}<br>Water: %{y:.0f} m³<br>Type: %{text}<extra></extra>',
                text=adjusted_data['DataType']
            ),
            row=1, col=1
        )
        
        # Add water efficiency
        fig1.add_trace(
            go.Scatter(
                x=adjusted_data['Year'],
                y=adjusted_data['WaterEffNoFinFan_Adjusted'],
                name="No Fin Fan (Efficiency)",
                mode='lines+markers',
                line=dict(color='darkblue', width=3),
                marker=dict(size=8),
                hovertemplate='Year: %{x}<br>Efficiency: %{y:.3f} m³/ton<br>Type: %{text}<extra></extra>',
                text=adjusted_data['DataType']
            ),
            row=2, col=1
        )
        
        fig1.add_trace(
            go.Scatter(
                x=adjusted_data['Year'],
                y=adjusted_data['WaterEffWithFinFan_Adjusted'],
                name="With Fin Fan (Efficiency)",
                mode='lines+markers',
                line=dict(color='darkgreen', width=3),
                marker=dict(size=8),
                hovertemplate='Year: %{x}<br>Efficiency: %{y:.3f} m³/ton<br>Type: %{text}<extra></extra>',
                text=adjusted_data['DataType']
            ),
            row=2, col=1
        )
        
        # Add vertical line at 2024 (start of forecast)
        fig1.add_vline(x=2024.5, line_width=2, line_dash="dash", line_color="red")
        fig1.add_annotation(x=2024.5, y=adjusted_data['WaterNoFinFan_Adjusted'].max()*0.95, 
                            text="Forecast Start", showarrow=False, font=dict(color="red"))
        
        # Update layout
        fig1.update_layout(
            height=700,
            barmode='group',
            hovermode="closest",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5
            )
        )
        
        st.plotly_chart(fig1, use_container_width=True)
        
        # Water savings metrics
        forecast_data = adjusted_data[adjusted_data['DataType'] == 'Forecast']
        if not forecast_data.empty:
            water_savings = forecast_data['WaterNoFinFan_Adjusted'] - forecast_data['WaterWithFinFan_Adjusted']
            water_savings_percent = (water_savings / forecast_data['WaterNoFinFan_Adjusted']) * 100
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Average Water Savings", f"{water_savings.mean():.0f} m³")
            with col2:
                st.metric("Average % Reduction", f"{water_savings_percent.mean():.1f}%")
            with col3:
                st.metric("2030 Water Savings", f"{water_savings.iloc[-1]:.0f} m³")
        
    with tab2:
        st.subheader("Energy Consumption and Efficiency")
        
        # Create energy subplot
        fig2 = make_subplots(
            rows=2, 
            cols=1,
            subplot_titles=("Energy Consumption (MWh)", "Energy Efficiency (MWh/ton)"),
            vertical_spacing=0.15,
            shared_xaxes=True
        )
        
        # Add energy consumption (volumes)
        fig2.add_trace(
            go.Bar(
                x=adjusted_data['Year'],
                y=adjusted_data['EnergyNoFinFan_Adjusted'],
                name="No Fin Fan",
                marker_color='purple',
                hovertemplate='Year: %{x}<br>Energy: %{y:.0f} MWh<br>Type: %{text}<extra></extra>',
                text=adjusted_data['DataType']
            ),
            row=1, col=1
        )
        
        fig2.add_trace(
            go.Bar(
                x=adjusted_data['Year'],
                y=adjusted_data['EnergyWithFinFan_Adjusted'],
                name="With Fin Fan",
                marker_color='orange',
                hovertemplate='Year: %{x}<br>Energy: %{y:.0f} MWh<br>Type: %{text}<extra></extra>',
                text=adjusted_data['DataType']
            ),
            row=1, col=1
        )
        
        # Add energy efficiency
        fig2.add_trace(
            go.Scatter(
                x=adjusted_data['Year'],
                y=adjusted_data['EnergyEffNoFinFan_Adjusted'],
                name="No Fin Fan (Efficiency)",
                mode='lines+markers',
                line=dict(color='darkviolet', width=3),
                marker=dict(size=8),
                hovertemplate='Year: %{x}<br>Efficiency: %{y:.3f} MWh/ton<br>Type: %{text}<extra></extra>',
                text=adjusted_data['DataType']
            ),
            row=2, col=1
        )
        
        fig2.add_trace(
            go.Scatter(
                x=adjusted_data['Year'],
                y=adjusted_data['EnergyEffWithFinFan_Adjusted'],
                name="With Fin Fan (Efficiency)",
                mode='lines+markers',
                line=dict(color='darkorange', width=3),
                marker=dict(size=8),
                hovertemplate='Year: %{x}<br>Efficiency: %{y:.3f} MWh/ton<br>Type: %{text}<extra></extra>',
                text=adjusted_data['DataType']
            ),
            row=2, col=1
        )
        
        # Add vertical line at 2024 (start of forecast)
        fig2.add_vline(x=2024.5, line_width=2, line_dash="dash", line_color="red")
        fig2.add_annotation(x=2024.5, y=adjusted_data['EnergyNoFinFan_Adjusted'].max()*0.95, 
                           text="Forecast Start", showarrow=False, font=dict(color="red"))
        
        # Update layout
        fig2.update_layout(
            height=700,
            barmode='group',
            hovermode="closest",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5
            )
        )
        
        st.plotly_chart(fig2, use_container_width=True)
        
        # Energy metrics
        forecast_data = adjusted_data[adjusted_data['DataType'] == 'Forecast']
        if not forecast_data.empty:
            energy_diff = forecast_data['EnergyWithFinFan_Adjusted'] - forecast_data['EnergyNoFinFan_Adjusted']
            energy_diff_percent = (energy_diff / forecast_data['EnergyNoFinFan_Adjusted']) * 100
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Average Energy Change", f"{energy_diff.mean():.0f} MWh")
            with col2:
                st.metric("Average % Change", f"{energy_diff_percent.mean():.2f}%")
            with col3:
                st.metric("2030 Energy Difference", f"{energy_diff.iloc[-1]:.0f} MWh")
    
    # Key insights
    st.header("Key Insights")
    
    # Get average savings from forecast data
    forecast_data = adjusted_data[adjusted_data['DataType'] == 'Forecast']
    if not forecast_data.empty:
        avg_water_savings_pct = ((forecast_data['WaterNoFinFan_Adjusted'] - forecast_data['WaterWithFinFan_Adjusted']) / 
                                 forecast_data['WaterNoFinFan_Adjusted']).mean() * 100
        
        avg_energy_change_pct = ((forecast_data['EnergyWithFinFan_Adjusted'] - forecast_data['EnergyNoFinFan_Adjusted']) / 
                                forecast_data['EnergyNoFinFan_Adjusted']).mean() * 100
        
        # Create three columns for metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.info(f"**Water Savings**: Average **{avg_water_savings_pct:.1f}%** reduction in water withdrawals with Fin Fan project")
        
        with col2:
            st.info(f"**Energy Impact**: Average **{avg_energy_change_pct:.2f}%** increase in energy consumption with Fin Fan project")
        
        with col3:
            st.info("**Trade-off**: Minimal energy penalty for substantial water conservation benefits")
else:
    st.error("Please upload the data file to continue.")