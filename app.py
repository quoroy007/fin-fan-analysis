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

# ************************************Function to load and prepare data
@st.cache_data
def load_data():
    try:
        # Load CSV with proper headers
        df = pd.read_csv('data/tons.csv')
        
        # Rename columns with unique names
        new_columns = [
            'Year', 'Context', 'DataType', 'Production', 'ProducedUnits',
            'WaterNoFinFan', 'WaterEffNoFinFan', 'WaterPerfImprovement',
            'WaterWithFinFan', 'WaterEffWithFinFan',
            'EnergyNoFinFan', 'EnergyEffNoFinFan', 'EnergyPerfImprovement',
            'EnergyWithFinFan', 'EnergyEffWithFinFan'
        ]
        
        df.columns = new_columns
        
        # Convert columns to numeric
        for col in df.columns:
            if col != 'Context' and col != 'DataType':
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.write("Please check your CSV file structure")
        return None

# **************************************Load data
data = load_data()

# If data is loaded successfully, continue with the app
if data is not None:
    # Create safety factor sliders
    st.sidebar.header("Safety Factor Adjustments")
    st.sidebar.info("Adjust safety factors to see how changes affect water and energy metrics")
    
    water_safety_factor = st.sidebar.slider(
        "Water Savings Confidence Factor (%)",
        min_value=1,
        max_value=10,
        value=5,
        help="Adjust water savings from baseline (30% reduction) to best case (45% reduction)"
    )
    
    energy_safety_factor = st.sidebar.slider(
        "Energy Volume Safety Factor (%)",
        min_value=1,
        max_value=10,
        value=5,
        help="Adjust energy increase from baseline (0.5%) to worst case (2%)"
    )
    
    # Apply safety factors to data
    def apply_safety_factors(df, water_factor, energy_factor):
        # Create a copy of the dataframe
        adjusted_df = df.copy()
        
        # Initialize adjusted columns with original values
        adjusted_df['WaterNoFinFan_Adjusted'] = adjusted_df['WaterNoFinFan']
        adjusted_df['WaterWithFinFan_Adjusted'] = adjusted_df['WaterWithFinFan'] 
        adjusted_df['EnergyNoFinFan_Adjusted'] = adjusted_df['EnergyNoFinFan']
        adjusted_df['EnergyWithFinFan_Adjusted'] = adjusted_df['EnergyWithFinFan']
        
        # Only apply safety factors to forecast data (Year >= 2025)
        forecast_mask = (adjusted_df['Year'] >= 2025) | (adjusted_df['DataType'] == 'Forecast')
        
        # For water in forecast years, keep "No Fin Fan" the same and adjust "With Fin Fan" 
        # to represent a reduction from 30% (baseline) to 45% (best case)
        
        # Calculate water reduction percentage based on safety factor (1% -> 30%, 10% -> 45%)
        base_reduction = 0.30  # 30% baseline reduction
        max_reduction = 0.45   # 45% maximum reduction
        
        # Linear interpolation between 30% and 45% based on safety factor
        reduction_pct = base_reduction + (max_reduction - base_reduction) * ((water_factor - 1) / 9)
        
        # Apply the calculated reduction to get the adjusted "With Fin Fan" value (only for forecast data)
        adjusted_df.loc[forecast_mask, 'WaterWithFinFan_Adjusted'] = adjusted_df.loc[forecast_mask, 'WaterNoFinFan'] * (1 - reduction_pct)
        
        # For energy in forecast years, keep "No Fin Fan" the same and adjust "With Fin Fan"
        # to represent an increase from 0.5% (baseline) to 2% (worst case)
        
        # Calculate energy increase percentage based on safety factor (1% -> 0.5%, 10% -> 2%)
        base_increase = 0.005  # 0.5% baseline increase
        max_increase = 0.02    # 2% maximum increase
        
        # Linear interpolation between 0.5% and 2% based on safety factor
        increase_pct = base_increase + (max_increase - base_increase) * ((energy_factor - 1) / 9)
        
        # Apply the calculated increase to get the adjusted "With Fin Fan" value (only for forecast data)
        adjusted_df.loc[forecast_mask, 'EnergyWithFinFan_Adjusted'] = adjusted_df.loc[forecast_mask, 'EnergyNoFinFan'] * (1 + increase_pct)
        
        # Recalculate water efficiency for all rows
        adjusted_df['WaterEffNoFinFan_Adjusted'] = adjusted_df['WaterNoFinFan_Adjusted'] / adjusted_df['Production']
        adjusted_df['WaterEffWithFinFan_Adjusted'] = adjusted_df['WaterWithFinFan_Adjusted'] / adjusted_df['Production']
        
        # Recalculate energy efficiency for all rows
        adjusted_df['EnergyEffNoFinFan_Adjusted'] = adjusted_df['EnergyNoFinFan_Adjusted'] / adjusted_df['Production']
        adjusted_df['EnergyEffWithFinFan_Adjusted'] = adjusted_df['EnergyWithFinFan_Adjusted'] / adjusted_df['Production']
        
        return adjusted_df
    
    # Apply safety factors
    adjusted_data = apply_safety_factors(data, water_safety_factor, energy_safety_factor)
    
    # Create visualizations
    st.header("Visualizations")
    
    # Water and Energy tab selector
    tab1, tab2, tab3 = st.tabs(["Water Analysis", "Energy Analysis", "Combined Analysis"])
    
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
        
        # Add labels to efficiency lines with staggered positions to avoid overlap
        for i, year in enumerate(adjusted_data['Year']):
            # Label for No Fin Fan efficiency (positioned slightly left)
            fig1.add_annotation(
                x=year - 0.3,
                y=adjusted_data['WaterEffNoFinFan_Adjusted'].iloc[i],
                text=f"{adjusted_data['WaterEffNoFinFan_Adjusted'].iloc[i]:.3f}",
                showarrow=False,
                font=dict(color="darkblue", size=10),
                bgcolor="rgba(255, 255, 255, 0.7)",  # Semi-transparent white background
                bordercolor="darkblue",
                borderwidth=1,
                borderpad=3
            )
            
            # Label for With Fin Fan efficiency (positioned slightly right)
            fig1.add_annotation(
                x=year + 0.3,
                y=adjusted_data['WaterEffWithFinFan_Adjusted'].iloc[i],
                text=f"{adjusted_data['WaterEffWithFinFan_Adjusted'].iloc[i]:.3f}",
                showarrow=False,
                font=dict(color="darkgreen", size=10),
                bgcolor="rgba(255, 255, 255, 0.7)",  # Semi-transparent white background
                bordercolor="darkgreen",
                borderwidth=1,
                borderpad=3
            )
        
        # Add vertical line at 2024 (start of forecast)
        fig1.add_vline(x=2024.5, line_width=2, line_dash="dash", line_color="red")
        fig1.add_annotation(x=2024.5, y=adjusted_data['WaterNoFinFan_Adjusted'].max()*0.95, 
                            text="Forecast Start", showarrow=False, font=dict(color="red"))
        
        # Add value labels to each bar
        for i, year in enumerate(adjusted_data['Year']):
            # Label for No Fin Fan bars
            fig1.add_annotation(
                x=year - 0.2,  # Moved more to the left
                y=adjusted_data['WaterNoFinFan_Adjusted'].iloc[i],
                text=f"{adjusted_data['WaterNoFinFan_Adjusted'].iloc[i]:.0f}",
                showarrow=False,
                yshift=10,
                font=dict(color="white", size=9),  # Smaller font
                xanchor="center"  # Center alignment
            )
            
            # Label for With Fin Fan bars
            fig1.add_annotation(
                x=year + 0.2,  # Moved more to the right
                y=adjusted_data['WaterWithFinFan_Adjusted'].iloc[i],
                text=f"{adjusted_data['WaterWithFinFan_Adjusted'].iloc[i]:.0f}",
                showarrow=False,
                yshift=10,
                font=dict(color="white", size=9),  # Smaller font
                xanchor="center"  # Center alignment
            )
        
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
        
        # Add labels to efficiency lines with staggered positions to avoid overlap
        for i, year in enumerate(adjusted_data['Year']):
            # Label for No Fin Fan efficiency (positioned slightly left)
            fig2.add_annotation(
                x=year - 0.3,
                y=adjusted_data['EnergyEffNoFinFan_Adjusted'].iloc[i],
                text=f"{adjusted_data['EnergyEffNoFinFan_Adjusted'].iloc[i]:.3f}",
                showarrow=False,
                font=dict(color="darkviolet", size=10),
                bgcolor="rgba(255, 255, 255, 0.7)",  # Semi-transparent white background
                bordercolor="darkviolet",
                borderwidth=1,
                borderpad=3
            )
            
            # Label for With Fin Fan efficiency (positioned slightly right)
            fig2.add_annotation(
                x=year + 0.3,
                y=adjusted_data['EnergyEffWithFinFan_Adjusted'].iloc[i],
                text=f"{adjusted_data['EnergyEffWithFinFan_Adjusted'].iloc[i]:.3f}",
                showarrow=False,
                font=dict(color="darkorange", size=10),
                bgcolor="rgba(255, 255, 255, 0.7)",  # Semi-transparent white background
                bordercolor="darkorange",
                borderwidth=1,
                borderpad=3
            )
        
        # Add vertical line at 2024 (start of forecast)
        fig2.add_vline(x=2024.5, line_width=2, line_dash="dash", line_color="red")
        fig2.add_annotation(x=2024.5, y=adjusted_data['EnergyNoFinFan_Adjusted'].max()*0.95, 
                           text="Forecast Start", showarrow=False, font=dict(color="red"))
        
        # Add value labels to each bar
        for i, year in enumerate(adjusted_data['Year']):
            # Label for No Fin Fan bars
            fig2.add_annotation(
                x=year - 0.2,  # Moved more to the left
                y=adjusted_data['EnergyNoFinFan_Adjusted'].iloc[i],
                text=f"{adjusted_data['EnergyNoFinFan_Adjusted'].iloc[i]:.0f}",
                showarrow=False,
                yshift=10,
                font=dict(color="white", size=9),  # Smaller font
                xanchor="center"  # Center alignment
            )
            
            # Label for With Fin Fan bars
            fig2.add_annotation(
                x=year + 0.2,  # Moved more to the right
                y=adjusted_data['EnergyWithFinFan_Adjusted'].iloc[i],
                text=f"{adjusted_data['EnergyWithFinFan_Adjusted'].iloc[i]:.0f}",
                showarrow=False,
                yshift=10,
                font=dict(color="white", size=9),  # Smaller font
                xanchor="center"  # Center alignment
            )
        
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

  with tab3:
        st.subheader("Combined Water and Energy Analysis")
        
        # Create combined visualization
        fig3 = make_subplots(
            rows=2, 
            cols=2,
            subplot_titles=(
                "Water Withdrawals (m³)", 
                "Energy Consumption (MWh)",
                "Water Efficiency (m³/ton)", 
                "Energy Efficiency (MWh/ton)"
            ),
            vertical_spacing=0.12,
            horizontal_spacing=0.08
        )
        
        # Add water withdrawals (top left)
        fig3.add_trace(
            go.Bar(
                x=adjusted_data['Year'],
                y=adjusted_data['WaterNoFinFan_Adjusted'],
                name="Water - No Fin Fan",
                marker_color='royalblue',
                hovertemplate='Year: %{x}<br>Water: %{y:.0f} m³<br>Type: %{text}<extra></extra>',
                text=adjusted_data['DataType'],
                showlegend=True,
                width=0.3
            ),
            row=1, col=1
        )
        
        fig3.add_trace(
            go.Bar(
                x=adjusted_data['Year'],
                y=adjusted_data['WaterWithFinFan_Adjusted'],
                name="Water - With Fin Fan",
                marker_color='lightgreen',
                hovertemplate='Year: %{x}<br>Water: %{y:.0f} m³<br>Type: %{text}<extra></extra>',
                text=adjusted_data['DataType'],
                showlegend=True,
                width=0.3
            ),
            row=1, col=1
        )
        
        # Add energy consumption (top right)
        fig3.add_trace(
            go.Bar(
                x=adjusted_data['Year'],
                y=adjusted_data['EnergyNoFinFan_Adjusted'],
                name="Energy - No Fin Fan",
                marker_color='purple',
                hovertemplate='Year: %{x}<br>Energy: %{y:.0f} MWh<br>Type: %{text}<extra></extra>',
                text=adjusted_data['DataType'],
                showlegend=True,
                width=0.3
            ),
            row=1, col=2
        )
        
        fig3.add_trace(
            go.Bar(
                x=adjusted_data['Year'],
                y=adjusted_data['EnergyWithFinFan_Adjusted'],
                name="Energy - With Fin Fan",
                marker_color='orange',
                hovertemplate='Year: %{x}<br>Energy: %{y:.0f} MWh<br>Type: %{text}<extra></extra>',
                text=adjusted_data['DataType'],
                showlegend=True,
                width=0.3
            ),
            row=1, col=2
        )
        
        # Add water efficiency (bottom left)
        fig3.add_trace(
            go.Scatter(
                x=adjusted_data['Year'],
                y=adjusted_data['WaterEffNoFinFan_Adjusted'],
                name="Water Eff - No Fin Fan",
                mode='lines+markers',
                line=dict(color='darkblue', width=2),
                marker=dict(size=6),
                hovertemplate='Year: %{x}<br>Efficiency: %{y:.3f} m³/ton<br>Type: %{text}<extra></extra>',
                text=adjusted_data['DataType'],
                showlegend=True
            ),
            row=2, col=1
        )
        
        fig3.add_trace(
            go.Scatter(
                x=adjusted_data['Year'],
                y=adjusted_data['WaterEffWithFinFan_Adjusted'],
                name="Water Eff - With Fin Fan",
                mode='lines+markers',
                line=dict(color='darkgreen', width=2),
                marker=dict(size=6),
                hovertemplate='Year: %{x}<br>Efficiency: %{y:.3f} m³/ton<br>Type: %{text}<extra></extra>',
                text=adjusted_data['DataType'],
                showlegend=True
            ),
            row=2, col=1
        )
        
        # Add energy efficiency (bottom right)
        fig3.add_trace(
            go.Scatter(
                x=adjusted_data['Year'],
                y=adjusted_data['EnergyEffNoFinFan_Adjusted'],
                name="Energy Eff - No Fin Fan",
                mode='lines+markers',
                line=dict(color='darkviolet', width=2),
                marker=dict(size=6),
                hovertemplate='Year: %{x}<br>Efficiency: %{y:.3f} MWh/ton<br>Type: %{text}<extra></extra>',
                text=adjusted_data['DataType'],
                showlegend=True
            ),
            row=2, col=2
        )
        
        fig3.add_trace(
            go.Scatter(
                x=adjusted_data['Year'],
                y=adjusted_data['EnergyEffWithFinFan_Adjusted'],
                name="Energy Eff - With Fin Fan",
                mode='lines+markers',
                line=dict(color='darkorange', width=2),
                marker=dict(size=6),
                hovertemplate='Year: %{x}<br>Efficiency: %{y:.3f} MWh/ton<br>Type: %{text}<extra></extra>',
                text=adjusted_data['DataType'],
                showlegend=True
            ),
            row=2, col=2
        )
        
        # Add vertical line at 2024 (start of forecast) to all subplots
        fig3.add_vline(x=2024.5, line_width=2, line_dash="dash", line_color="red", row=1, col=1)
        fig3.add_vline(x=2024.5, line_width=2, line_dash="dash", line_color="red", row=1, col=2)
        fig3.add_vline(x=2024.5, line_width=2, line_dash="dash", line_color="red", row=2, col=1)
        fig3.add_vline(x=2024.5, line_width=2, line_dash="dash", line_color="red", row=2, col=2)
        
        # Update layout
        fig3.update_layout(
            height=800,
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
        
        # Update y-axes titles
        fig3.update_yaxes(title_text="Water (m³)", row=1, col=1)
        fig3.update_yaxes(title_text="Energy (MWh)", row=1, col=2)
        fig3.update_yaxes(title_text="Water Efficiency (m³/ton)", row=2, col=1)
        fig3.update_yaxes(title_text="Energy Efficiency (MWh/ton)", row=2, col=2)
        
        st.plotly_chart(fig3, use_container_width=True)
        
        # Summary metrics for combined view
        forecast_data = adjusted_data[adjusted_data['DataType'] == 'Forecast']
        if not forecast_data.empty:
            water_savings_pct = ((forecast_data['WaterNoFinFan_Adjusted'] - forecast_data['WaterWithFinFan_Adjusted']) / 
                                forecast_data['WaterNoFinFan_Adjusted']).mean() * 100
            
            energy_increase_pct = ((forecast_data['EnergyWithFinFan_Adjusted'] - forecast_data['EnergyNoFinFan_Adjusted']) / 
                                  forecast_data['EnergyNoFinFan_Adjusted']).mean() * 100
            
            trade_off_ratio = water_savings_pct / energy_increase_pct if energy_increase_pct > 0 else 0
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Water Savings", f"{water_savings_pct:.1f}%")
            with col2:
                st.metric("Energy Increase", f"{energy_increase_pct:.2f}%")
            with col3:
                st.metric("Water-Energy Trade-off Ratio", f"{trade_off_ratio:.1f}:1")
                st.caption("Water savings % : Energy increase %")
        
        # Add text explanation
        st.write("""
        ### Water-Energy Trade-off Analysis
