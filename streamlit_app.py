"""
Streamlit app for CO2 emission prediction with real-time updates
"""
import streamlit as st
import numpy as np
import pandas as pd
import joblib
import json
import os
import plotly.express as px
import plotly.graph_objects as go
from pycountry import countries

# Page configuration
st.set_page_config(
    page_title="CO2 Emission Predictor",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load model and feature selector
model_path = 'models/co2_model.pkl'
selector_path = 'models/feature_selector.pkl'
info_path = 'models/model_info.json'

# Check if models exist
if not os.path.exists(model_path):
    st.error("Model not found. Please run 'python train_model.py' first to train and save the model.")
    st.stop()

# Load models (cached for performance)
@st.cache_resource
def load_models():
    """Load the trained model and feature selector"""
    model = joblib.load(model_path)
    selector = joblib.load(selector_path)
    return model, selector

@st.cache_data
def load_model_info():
    """Load model information"""
    with open(info_path, 'r') as f:
        return json.load(f)

@st.cache_data
def load_co2_data():
    """Load and aggregate CO2 emissions data by country"""
    try:
        df = pd.read_csv('Dataset/data_cleaned.csv')
        # Get the most recent year's data for each country, or average if multiple years
        co2_data = df.groupby('country').agg({
            'co2_ttl': 'mean',  # Average total CO2 emissions
            'co2_per_cap': 'mean',  # Average CO2 per capita
            'year': 'max'  # Most recent year
        }).reset_index()
        
        # Map country codes to ISO 3-letter codes for plotly
        country_code_mapping = {}
        country_names = {}
        
        for code in co2_data['country'].unique():
            iso_code = code
            country_name = code
            
            try:
                # Try to find the country by ISO alpha-3 code
                country = countries.get(alpha_3=code)
                if country:
                    iso_code = country.alpha_3
                    country_name = country.name
                else:
                    # Try to find by alpha-2 code
                    country = countries.get(alpha_2=code)
                    if country:
                        iso_code = country.alpha_3
                        country_name = country.name
                    else:
                        # Try searching by name variations
                        for c in countries:
                            if hasattr(c, 'alpha_3') and c.alpha_3 == code:
                                iso_code = c.alpha_3
                                country_name = c.name
                                break
            except Exception:
                pass
            
            country_code_mapping[code] = iso_code
            country_names[code] = country_name
        
        co2_data['iso_code'] = co2_data['country'].map(country_code_mapping)
        co2_data['country_name'] = co2_data['country'].map(country_names)
        
        # Add all countries from pycountry (for complete world map)
        all_countries_df = pd.DataFrame([
            {'iso_code': c.alpha_3, 'country_name': c.name}
            for c in countries
            if hasattr(c, 'alpha_3')
        ])
        
        # Merge with CO2 data, keeping all countries
        merged_data = all_countries_df.merge(
            co2_data[['iso_code', 'co2_ttl', 'co2_per_cap', 'year']],
            on='iso_code',
            how='left'
        )
        
        # Fill missing values with 0 for countries without data
        merged_data['co2_ttl'] = merged_data['co2_ttl'].fillna(0)
        merged_data['co2_per_cap'] = merged_data['co2_per_cap'].fillna(0)
        merged_data['year'] = merged_data['year'].fillna(0)
        
        return merged_data
    except Exception as e:
        st.error(f"Error loading CO2 data: {str(e)}")
        return pd.DataFrame()

def create_co2_map(co2_data, selected_country_code=None):
    """Create a choropleth map showing CO2 emissions by country"""
    if co2_data.empty:
        return None
    
    # Filter out countries with no data (0, NaN, or missing values)
    display_data = co2_data.copy()
    display_data = display_data[
        (display_data['co2_per_cap'].notna()) & 
        (display_data['co2_per_cap'] > 0) &
        (display_data['iso_code'].notna())
    ]
    
    if display_data.empty:
        return None
    
    # Use co2_per_cap for intensity scale
    fig = px.choropleth(
        display_data,
        locations='iso_code',
        color='co2_per_cap',
        hover_name='country_name',
        hover_data={
            'co2_per_cap': ':.2f',
            'co2_ttl': ':,.0f',
            'iso_code': False
        },
        color_continuous_scale='Reds',
        labels={'co2_per_cap': 'CO2 per Capita (metric tons)'},
        title='Global CO2 Emissions per Capita by Country',
        projection='natural earth',
        scope='world'
    )
    
    # Note: Country highlighting is handled via the stats display below the map
    
    fig.update_layout(
        height=600,
        geo=dict(
            showframe=False,
            showcoastlines=True,
            projection_type='natural earth',
            showcountries=True,
            countrycolor='lightgray',
            coastlinecolor='lightgray'
        ),
        margin=dict(l=0, r=0, t=50, b=0),
        coloraxis_colorbar=dict(
            title=dict(
                text="CO2 per Capita<br>(metric tons)",
                font=dict(size=12)
            )
        )
    )
    
    return fig

def get_country_code_from_name(country_name, co2_data):
    """Get ISO country code from country name"""
    if country_name == "Custom" or country_name not in co2_data['country_name'].values:
        return None
    country_row = co2_data[co2_data['country_name'] == country_name].iloc[0]
    return country_row['iso_code']

try:
    model, selector = load_models()
    model_info = load_model_info()
    co2_data = load_co2_data()
except Exception as e:
    st.error(f"Error loading models: {str(e)}")
    st.stop()

selected_features = model_info['selected_features']
all_features = model_info['all_features']
feature_ranges = model_info.get('feature_ranges', {})

# Country presets (approximate real-world values) - all values as floats
COUNTRY_PRESETS = {
    "Custom": None,
    # Major Economies
    "USA": {
        "cereal_yield": 7500.0, "fdi_perc_gdp": 1.8, "gni_per_cap": 70000.0,
        "gdp_billions": 25000.0, "en_per_cap": 7000.0, "prot_area_perc": 15.0,
        "pop_urb_aggl_perc": 45.0, "pop_growth_perc": 0.6, "urb_pop_growth_perc": 1.2
    },
    "China": {
        "cereal_yield": 6000.0, "fdi_perc_gdp": 1.2, "gni_per_cap": 12000.0,
        "gdp_billions": 18000.0, "en_per_cap": 2500.0, "prot_area_perc": 18.0,
        "pop_urb_aggl_perc": 35.0, "pop_growth_perc": 0.4, "urb_pop_growth_perc": 2.8
    },
    "Japan": {
        "cereal_yield": 6000.0, "fdi_perc_gdp": 0.5, "gni_per_cap": 42000.0,
        "gdp_billions": 4900.0, "en_per_cap": 3500.0, "prot_area_perc": 20.0,
        "pop_urb_aggl_perc": 55.0, "pop_growth_perc": -0.2, "urb_pop_growth_perc": 0.3
    },
    "Germany": {
        "cereal_yield": 7500.0, "fdi_perc_gdp": 2.5, "gni_per_cap": 55000.0,
        "gdp_billions": 4200.0, "en_per_cap": 3800.0, "prot_area_perc": 38.0,
        "pop_urb_aggl_perc": 50.0, "pop_growth_perc": 0.2, "urb_pop_growth_perc": 0.5
    },
    "India": {
        "cereal_yield": 3000.0, "fdi_perc_gdp": 1.5, "gni_per_cap": 2200.0,
        "gdp_billions": 3300.0, "en_per_cap": 700.0, "prot_area_perc": 5.0,
        "pop_urb_aggl_perc": 15.0, "pop_growth_perc": 1.0, "urb_pop_growth_perc": 2.5
    },
    "UK": {
        "cereal_yield": 8000.0, "fdi_perc_gdp": 2.2, "gni_per_cap": 48000.0,
        "gdp_billions": 3100.0, "en_per_cap": 2800.0, "prot_area_perc": 28.0,
        "pop_urb_aggl_perc": 48.0, "pop_growth_perc": 0.5, "urb_pop_growth_perc": 0.8
    },
    "France": {
        "cereal_yield": 7500.0, "fdi_perc_gdp": 1.5, "gni_per_cap": 45000.0,
        "gdp_billions": 2900.0, "en_per_cap": 3600.0, "prot_area_perc": 33.0,
        "pop_urb_aggl_perc": 47.0, "pop_growth_perc": 0.3, "urb_pop_growth_perc": 0.6
    },
    "Italy": {
        "cereal_yield": 6000.0, "fdi_perc_gdp": 1.0, "gni_per_cap": 36000.0,
        "gdp_billions": 2100.0, "en_per_cap": 2400.0, "prot_area_perc": 22.0,
        "pop_urb_aggl_perc": 42.0, "pop_growth_perc": -0.1, "urb_pop_growth_perc": 0.4
    },
    "Canada": {
        "cereal_yield": 4500.0, "fdi_perc_gdp": 2.8, "gni_per_cap": 52000.0,
        "gdp_billions": 2100.0, "en_per_cap": 7800.0, "prot_area_perc": 12.0,
        "pop_urb_aggl_perc": 40.0, "pop_growth_perc": 0.9, "urb_pop_growth_perc": 1.3
    },
    "South Korea": {
        "cereal_yield": 6500.0, "fdi_perc_gdp": 0.8, "gni_per_cap": 35000.0,
        "gdp_billions": 1800.0, "en_per_cap": 5500.0, "prot_area_perc": 16.0,
        "pop_urb_aggl_perc": 52.0, "pop_growth_perc": 0.1, "urb_pop_growth_perc": 0.7
    },
    "Australia": {
        "cereal_yield": 2000.0, "fdi_perc_gdp": 2.1, "gni_per_cap": 58000.0,
        "gdp_billions": 1700.0, "en_per_cap": 5200.0, "prot_area_perc": 20.0,
        "pop_urb_aggl_perc": 38.0, "pop_growth_perc": 1.2, "urb_pop_growth_perc": 1.5
    },
    "Spain": {
        "cereal_yield": 3500.0, "fdi_perc_gdp": 1.8, "gni_per_cap": 31000.0,
        "gdp_billions": 1400.0, "en_per_cap": 2400.0, "prot_area_perc": 27.0,
        "pop_urb_aggl_perc": 44.0, "pop_growth_perc": 0.1, "urb_pop_growth_perc": 0.5
    },
    "Mexico": {
        "cereal_yield": 3500.0, "fdi_perc_gdp": 2.3, "gni_per_cap": 10000.0,
        "gdp_billions": 1300.0, "en_per_cap": 1500.0, "prot_area_perc": 14.0,
        "pop_urb_aggl_perc": 28.0, "pop_growth_perc": 1.1, "urb_pop_growth_perc": 1.8
    },
    "Indonesia": {
        "cereal_yield": 5000.0, "fdi_perc_gdp": 2.0, "gni_per_cap": 4200.0,
        "gdp_billions": 1200.0, "en_per_cap": 900.0, "prot_area_perc": 11.0,
        "pop_urb_aggl_perc": 22.0, "pop_growth_perc": 1.1, "urb_pop_growth_perc": 2.2
    },
    "Netherlands": {
        "cereal_yield": 9000.0, "fdi_perc_gdp": 3.5, "gni_per_cap": 56000.0,
        "gdp_billions": 1000.0, "en_per_cap": 4600.0, "prot_area_perc": 15.0,
        "pop_urb_aggl_perc": 51.0, "pop_growth_perc": 0.4, "urb_pop_growth_perc": 0.9
    },
    "Saudi Arabia": {
        "cereal_yield": 5000.0, "fdi_perc_gdp": 1.0, "gni_per_cap": 24000.0,
        "gdp_billions": 840.0, "en_per_cap": 6500.0, "prot_area_perc": 5.0,
        "pop_urb_aggl_perc": 35.0, "pop_growth_perc": 1.5, "urb_pop_growth_perc": 2.1
    },
    "Turkey": {
        "cereal_yield": 3000.0, "fdi_perc_gdp": 1.3, "gni_per_cap": 11000.0,
        "gdp_billions": 820.0, "en_per_cap": 1500.0, "prot_area_perc": 9.0,
        "pop_urb_aggl_perc": 32.0, "pop_growth_perc": 1.3, "urb_pop_growth_perc": 2.0
    },
    "Switzerland": {
        "cereal_yield": 6500.0, "fdi_perc_gdp": 4.2, "gni_per_cap": 95000.0,
        "gdp_billions": 810.0, "en_per_cap": 3200.0, "prot_area_perc": 13.0,
        "pop_urb_aggl_perc": 46.0, "pop_growth_perc": 0.7, "urb_pop_growth_perc": 0.8
    },
    "Poland": {
        "cereal_yield": 4500.0, "fdi_perc_gdp": 2.0, "gni_per_cap": 18000.0,
        "gdp_billions": 680.0, "en_per_cap": 2300.0, "prot_area_perc": 23.0,
        "pop_urb_aggl_perc": 38.0, "pop_growth_perc": -0.2, "urb_pop_growth_perc": 0.2
    },
    "Belgium": {
        "cereal_yield": 9000.0, "fdi_perc_gdp": 3.8, "gni_per_cap": 52000.0,
        "gdp_billions": 580.0, "en_per_cap": 4200.0, "prot_area_perc": 13.0,
        "pop_urb_aggl_perc": 49.0, "pop_growth_perc": 0.5, "urb_pop_growth_perc": 0.6
    },
    "Argentina": {
        "cereal_yield": 4000.0, "fdi_perc_gdp": 1.5, "gni_per_cap": 9500.0,
        "gdp_billions": 490.0, "en_per_cap": 2000.0, "prot_area_perc": 8.0,
        "pop_urb_aggl_perc": 30.0, "pop_growth_perc": 0.9, "urb_pop_growth_perc": 1.4
    },
    "Sweden": {
        "cereal_yield": 5500.0, "fdi_perc_gdp": 2.3, "gni_per_cap": 62000.0,
        "gdp_billions": 590.0, "en_per_cap": 4800.0, "prot_area_perc": 15.0,
        "pop_urb_aggl_perc": 43.0, "pop_growth_perc": 0.6, "urb_pop_growth_perc": 0.9
    },
    "Thailand": {
        "cereal_yield": 3000.0, "fdi_perc_gdp": 2.5, "gni_per_cap": 7200.0,
        "gdp_billions": 510.0, "en_per_cap": 1900.0, "prot_area_perc": 20.0,
        "pop_urb_aggl_perc": 25.0, "pop_growth_perc": 0.2, "urb_pop_growth_perc": 1.2
    },
    "Israel": {
        "cereal_yield": 3500.0, "fdi_perc_gdp": 1.2, "gni_per_cap": 44000.0,
        "gdp_billions": 480.0, "en_per_cap": 2800.0, "prot_area_perc": 6.0,
        "pop_urb_aggl_perc": 48.0, "pop_growth_perc": 1.6, "urb_pop_growth_perc": 1.8
    },
    "Norway": {
        "cereal_yield": 4500.0, "fdi_perc_gdp": 1.5, "gni_per_cap": 95000.0,
        "gdp_billions": 480.0, "en_per_cap": 5500.0, "prot_area_perc": 17.0,
        "pop_urb_aggl_perc": 41.0, "pop_growth_perc": 0.7, "urb_pop_growth_perc": 1.1
    },
    "Ireland": {
        "cereal_yield": 8500.0, "fdi_perc_gdp": 5.2, "gni_per_cap": 85000.0,
        "gdp_billions": 530.0, "en_per_cap": 3200.0, "prot_area_perc": 14.0,
        "pop_urb_aggl_perc": 40.0, "pop_growth_perc": 0.8, "urb_pop_growth_perc": 1.0
    },
    "Singapore": {
        "cereal_yield": 0.0, "fdi_perc_gdp": 3.8, "gni_per_cap": 65000.0,
        "gdp_billions": 400.0, "en_per_cap": 5200.0, "prot_area_perc": 3.0,
        "pop_urb_aggl_perc": 100.0, "pop_growth_perc": 0.9, "urb_pop_growth_perc": 1.2
    },
    "Malaysia": {
        "cereal_yield": 3500.0, "fdi_perc_gdp": 2.8, "gni_per_cap": 11200.0,
        "gdp_billions": 430.0, "en_per_cap": 2400.0, "prot_area_perc": 13.0,
        "pop_urb_aggl_perc": 28.0, "pop_growth_perc": 1.1, "urb_pop_growth_perc": 2.1
    },
    "South Africa": {
        "cereal_yield": 2500.0, "fdi_perc_gdp": 1.2, "gni_per_cap": 6000.0,
        "gdp_billions": 420.0, "en_per_cap": 2600.0, "prot_area_perc": 8.0,
        "pop_urb_aggl_perc": 26.0, "pop_growth_perc": 1.2, "urb_pop_growth_perc": 1.8
    },
    "Philippines": {
        "cereal_yield": 3800.0, "fdi_perc_gdp": 2.1, "gni_per_cap": 3800.0,
        "gdp_billions": 400.0, "en_per_cap": 500.0, "prot_area_perc": 15.0,
        "pop_urb_aggl_perc": 20.0, "pop_growth_perc": 1.4, "urb_pop_growth_perc": 2.3
    },
    "Chile": {
        "cereal_yield": 5500.0, "fdi_perc_gdp": 3.5, "gni_per_cap": 15000.0,
        "gdp_billions": 310.0, "en_per_cap": 2100.0, "prot_area_perc": 21.0,
        "pop_urb_aggl_perc": 32.0, "pop_growth_perc": 0.8, "urb_pop_growth_perc": 1.3
    },
    "Finland": {
        "cereal_yield": 4000.0, "fdi_perc_gdp": 1.8, "gni_per_cap": 54000.0,
        "gdp_billions": 280.0, "en_per_cap": 6000.0, "prot_area_perc": 14.0,
        "pop_urb_aggl_perc": 39.0, "pop_growth_perc": 0.2, "urb_pop_growth_perc": 0.5
    },
    "Bangladesh": {
        "cereal_yield": 4500.0, "fdi_perc_gdp": 1.2, "gni_per_cap": 2600.0,
        "gdp_billions": 460.0, "en_per_cap": 300.0, "prot_area_perc": 4.0,
        "pop_urb_aggl_perc": 18.0, "pop_growth_perc": 1.1, "urb_pop_growth_perc": 2.8
    },
    "Pakistan": {
        "cereal_yield": 3000.0, "fdi_perc_gdp": 0.8, "gni_per_cap": 1500.0,
        "gdp_billions": 350.0, "en_per_cap": 450.0, "prot_area_perc": 12.0,
        "pop_urb_aggl_perc": 16.0, "pop_growth_perc": 2.0, "urb_pop_growth_perc": 2.9
    },
    "Vietnam": {
        "cereal_yield": 5500.0, "fdi_perc_gdp": 2.3, "gni_per_cap": 3600.0,
        "gdp_billions": 360.0, "en_per_cap": 800.0, "prot_area_perc": 7.0,
        "pop_urb_aggl_perc": 19.0, "pop_growth_perc": 0.9, "urb_pop_growth_perc": 2.5
    },
    "Romania": {
        "cereal_yield": 4000.0, "fdi_perc_gdp": 2.1, "gni_per_cap": 14000.0,
        "gdp_billions": 280.0, "en_per_cap": 1800.0, "prot_area_perc": 24.0,
        "pop_urb_aggl_perc": 33.0, "pop_growth_perc": -0.5, "urb_pop_growth_perc": 0.1
    },
    "Czech Republic": {
        "cereal_yield": 6000.0, "fdi_perc_gdp": 2.8, "gni_per_cap": 24000.0,
        "gdp_billions": 290.0, "en_per_cap": 3500.0, "prot_area_perc": 26.0,
        "pop_urb_aggl_perc": 41.0, "pop_growth_perc": 0.1, "urb_pop_growth_perc": 0.3
    },
    "New Zealand": {
        "cereal_yield": 7000.0, "fdi_perc_gdp": 1.9, "gni_per_cap": 48000.0,
        "gdp_billions": 250.0, "en_per_cap": 4100.0, "prot_area_perc": 32.0,
        "pop_urb_aggl_perc": 36.0, "pop_growth_perc": 1.0, "urb_pop_growth_perc": 1.4
    },
    "Peru": {
        "cereal_yield": 3500.0, "fdi_perc_gdp": 2.5, "gni_per_cap": 6800.0,
        "gdp_billions": 240.0, "en_per_cap": 800.0, "prot_area_perc": 17.0,
        "pop_urb_aggl_perc": 24.0, "pop_growth_perc": 1.0, "urb_pop_growth_perc": 1.6
    },
    "Iraq": {
        "cereal_yield": 2000.0, "fdi_perc_gdp": 0.5, "gni_per_cap": 5000.0,
        "gdp_billions": 230.0, "en_per_cap": 1500.0, "prot_area_perc": 1.0,
        "pop_urb_aggl_perc": 29.0, "pop_growth_perc": 2.3, "urb_pop_growth_perc": 2.8
    },
    "Qatar": {
        "cereal_yield": 0.0, "fdi_perc_gdp": 0.3, "gni_per_cap": 61000.0,
        "gdp_billions": 180.0, "en_per_cap": 12000.0, "prot_area_perc": 2.0,
        "pop_urb_aggl_perc": 48.0, "pop_growth_perc": 1.2, "urb_pop_growth_perc": 1.8
    },
    "Algeria": {
        "cereal_yield": 1500.0, "fdi_perc_gdp": 0.8, "gni_per_cap": 3800.0,
        "gdp_billions": 190.0, "en_per_cap": 1200.0, "prot_area_perc": 8.0,
        "pop_urb_aggl_perc": 27.0, "pop_growth_perc": 1.8, "urb_pop_growth_perc": 2.4
    },
    "Kazakhstan": {
        "cereal_yield": 1200.0, "fdi_perc_gdp": 2.2, "gni_per_cap": 9000.0,
        "gdp_billions": 200.0, "en_per_cap": 4500.0, "prot_area_perc": 5.0,
        "pop_urb_aggl_perc": 20.0, "pop_growth_perc": 1.1, "urb_pop_growth_perc": 1.5
    },
    "Hungary": {
        "cereal_yield": 6000.0, "fdi_perc_gdp": 3.2, "gni_per_cap": 18000.0,
        "gdp_billions": 180.0, "en_per_cap": 2400.0, "prot_area_perc": 22.0,
        "pop_urb_aggl_perc": 37.0, "pop_growth_perc": -0.3, "urb_pop_growth_perc": 0.2
    },
    "Ukraine": {
        "cereal_yield": 4500.0, "fdi_perc_gdp": 1.5, "gni_per_cap": 4200.0,
        "gdp_billions": 160.0, "en_per_cap": 2200.0, "prot_area_perc": 4.0,
        "pop_urb_aggl_perc": 31.0, "pop_growth_perc": -0.5, "urb_pop_growth_perc": -0.1
    },
    "Morocco": {
        "cereal_yield": 2000.0, "fdi_perc_gdp": 1.8, "gni_per_cap": 3200.0,
        "gdp_billions": 140.0, "en_per_cap": 600.0, "prot_area_perc": 9.0,
        "pop_urb_aggl_perc": 23.0, "pop_growth_perc": 1.2, "urb_pop_growth_perc": 2.1
    },
    "Slovakia": {
        "cereal_yield": 5500.0, "fdi_perc_gdp": 3.5, "gni_per_cap": 21000.0,
        "gdp_billions": 120.0, "en_per_cap": 3000.0, "prot_area_perc": 24.0,
        "pop_urb_aggl_perc": 35.0, "pop_growth_perc": 0.1, "urb_pop_growth_perc": 0.4
    },
    "Ecuador": {
        "cereal_yield": 3000.0, "fdi_perc_gdp": 1.2, "gni_per_cap": 6000.0,
        "gdp_billions": 110.0, "en_per_cap": 900.0, "prot_area_perc": 19.0,
        "pop_urb_aggl_perc": 22.0, "pop_growth_perc": 1.3, "urb_pop_growth_perc": 1.9
    },
    "Oman": {
        "cereal_yield": 1000.0, "fdi_perc_gdp": 1.5, "gni_per_cap": 19000.0,
        "gdp_billions": 80.0, "en_per_cap": 5800.0, "prot_area_perc": 3.0,
        "pop_urb_aggl_perc": 33.0, "pop_growth_perc": 2.0, "urb_pop_growth_perc": 2.5
    },
    "Azerbaijan": {
        "cereal_yield": 2500.0, "fdi_perc_gdp": 2.8, "gni_per_cap": 4800.0,
        "gdp_billions": 55.0, "en_per_cap": 1500.0, "prot_area_perc": 10.0,
        "pop_urb_aggl_perc": 28.0, "pop_growth_perc": 1.1, "urb_pop_growth_perc": 1.6
    },
    "Sri Lanka": {
        "cereal_yield": 4000.0, "fdi_perc_gdp": 1.0, "gni_per_cap": 3900.0,
        "gdp_billions": 85.0, "en_per_cap": 500.0, "prot_area_perc": 26.0,
        "pop_urb_aggl_perc": 14.0, "pop_growth_perc": 0.3, "urb_pop_growth_perc": 1.0
    },
    "Myanmar": {
        "cereal_yield": 3800.0, "fdi_perc_gdp": 1.8, "gni_per_cap": 1300.0,
        "gdp_billions": 75.0, "en_per_cap": 350.0, "prot_area_perc": 6.0,
        "pop_urb_aggl_perc": 12.0, "pop_growth_perc": 0.8, "urb_pop_growth_perc": 1.8
    },
    "Tanzania": {
        "cereal_yield": 1200.0, "fdi_perc_gdp": 2.5, "gni_per_cap": 1100.0,
        "gdp_billions": 70.0, "en_per_cap": 400.0, "prot_area_perc": 32.0,
        "pop_urb_aggl_perc": 8.0, "pop_growth_perc": 3.0, "urb_pop_growth_perc": 4.2
    },
    "Kenya": {
        "cereal_yield": 1800.0, "fdi_perc_gdp": 1.2, "gni_per_cap": 1800.0,
        "gdp_billions": 95.0, "en_per_cap": 500.0, "prot_area_perc": 12.0,
        "pop_urb_aggl_perc": 11.0, "pop_growth_perc": 2.3, "urb_pop_growth_perc": 3.8
    },
    "Ghana": {
        "cereal_yield": 2000.0, "fdi_perc_gdp": 2.8, "gni_per_cap": 2300.0,
        "gdp_billions": 75.0, "en_per_cap": 400.0, "prot_area_perc": 16.0,
        "pop_urb_aggl_perc": 13.0, "pop_growth_perc": 2.1, "urb_pop_growth_perc": 3.2
    },
    "Ethiopia": {
        "cereal_yield": 2500.0, "fdi_perc_gdp": 1.5, "gni_per_cap": 950.0,
        "gdp_billions": 110.0, "en_per_cap": 300.0, "prot_area_perc": 17.0,
        "pop_urb_aggl_perc": 7.0, "pop_growth_perc": 2.5, "urb_pop_growth_perc": 4.0
    },
    "Uganda": {
        "cereal_yield": 2000.0, "fdi_perc_gdp": 2.2, "gni_per_cap": 840.0,
        "gdp_billions": 42.0, "en_per_cap": 250.0, "prot_area_perc": 16.0,
        "pop_urb_aggl_perc": 6.0, "pop_growth_perc": 3.2, "urb_pop_growth_perc": 4.5
    },
    "Nepal": {
        "cereal_yield": 3000.0, "fdi_perc_gdp": 0.5, "gni_per_cap": 1200.0,
        "gdp_billions": 36.0, "en_per_cap": 400.0, "prot_area_perc": 24.0,
        "pop_urb_aggl_perc": 9.0, "pop_growth_perc": 1.0, "urb_pop_growth_perc": 2.1
    },
    "Cambodia": {
        "cereal_yield": 3500.0, "fdi_perc_gdp": 3.2, "gni_per_cap": 1600.0,
        "gdp_billions": 27.0, "en_per_cap": 350.0, "prot_area_perc": 23.0,
        "pop_urb_aggl_perc": 10.0, "pop_growth_perc": 1.4, "urb_pop_growth_perc": 2.6
    },
    "Rwanda": {
        "cereal_yield": 2500.0, "fdi_perc_gdp": 1.8, "gni_per_cap": 820.0,
        "gdp_billions": 11.0, "en_per_cap": 200.0, "prot_area_perc": 9.0,
        "pop_urb_aggl_perc": 5.0, "pop_growth_perc": 2.4, "urb_pop_growth_perc": 3.5
    },
    "Mozambique": {
        "cereal_yield": 1200.0, "fdi_perc_gdp": 3.5, "gni_per_cap": 500.0,
        "gdp_billions": 16.0, "en_per_cap": 350.0, "prot_area_perc": 26.0,
        "pop_urb_aggl_perc": 4.0, "pop_growth_perc": 2.8, "urb_pop_growth_perc": 3.8
    },
    "Nigeria": {
        "cereal_yield": 1500.0, "fdi_perc_gdp": 1.0, "gni_per_cap": 2200.0,
        "gdp_billions": 440.0, "en_per_cap": 200.0, "prot_area_perc": 8.0,
        "pop_urb_aggl_perc": 12.0, "pop_growth_perc": 2.5, "urb_pop_growth_perc": 3.5
    },
    "Brazil": {
        "cereal_yield": 4500.0, "fdi_perc_gdp": 2.0, "gni_per_cap": 8500.0,
        "gdp_billions": 1900.0, "en_per_cap": 1500.0, "prot_area_perc": 30.0,
        "pop_urb_aggl_perc": 25.0, "pop_growth_perc": 0.7, "urb_pop_growth_perc": 1.5
    }
}

def predict_co2_emissions(
    cereal_yield, fdi_perc_gdp, gni_per_cap, en_per_cap,
    pop_urb_aggl_perc, prot_area_perc, gdp,
    pop_growth_perc, urb_pop_growth_perc
):
    """Predict CO2 emissions per capita based on input features"""
    try:
        features = np.array([[
            cereal_yield, fdi_perc_gdp, gni_per_cap, en_per_cap,
            pop_urb_aggl_perc, prot_area_perc, gdp,
            pop_growth_perc, urb_pop_growth_perc
        ]])
        
        features_reduced = selector.transform(features)
        prediction = model.predict(features_reduced)[0]
        return prediction
        
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        return None

# Feature name mapping for display
FEATURE_NAMES = {
    "cereal_yield": "Cereal Yield",
    "gni_per_cap": "GNI per Capita",
    "en_per_cap": "Energy Use per Capita",
    "pop_urb_aggl_perc": "Population in Urban Agglomerations >1M (%)",
    "prot_area_perc": "Protected Areas (% of total land area)",
    "gdp": "GDP (Billions $)",
    "pop_growth_perc": "Population Growth (annual %)",
    "urb_pop_growth_perc": "Urban Population Growth (annual %)"
}

# Main app
st.title(" CO2 Emission Predictor")
st.markdown("**Predict CO2 emissions per capita using economic and demographic indicators.**")
st.markdown("---")

# Sidebar for inputs
st.sidebar.header("⚙️ Input Parameters")
st.sidebar.caption("Adjust these values to see how they affect CO2 emissions predictions")

default_cereal = float(feature_ranges.get('cereal_yield', {}).get('mean', 3000))
default_fdi = float(feature_ranges.get('fdi_perc_gdp', {}).get('mean', 3))
default_gni = float(feature_ranges.get('gni_per_cap', {}).get('mean', 5000))
default_gdp_b = float(feature_ranges.get('gdp', {}).get('mean', 1e11) / 1e9)
default_en = float(feature_ranges.get('en_per_cap', {}).get('mean', 1500))
default_prot = float(feature_ranges.get('prot_area_perc', {}).get('mean', 10))
default_urb = float(feature_ranges.get('pop_urb_aggl_perc', {}).get('mean', 30))
default_pop_g = float(feature_ranges.get('pop_growth_perc', {}).get('mean', 1.5))
default_urb_g = float(feature_ranges.get('urb_pop_growth_perc', {}).get('mean', 2.5))

# Section: Economic Indicators
st.sidebar.markdown("---")
st.sidebar.markdown("###  Economic Indicators")
st.sidebar.caption("Financial and economic factors")

# Format slider values with units
cereal_yield = st.sidebar.slider(
    "Cereal Yield",
    min_value=23.0,
    max_value=29147.0,
    value=default_cereal,
    step=10.0,
    format="%.0f",
    help="kg per hectare"
)
st.sidebar.caption(f"**{cereal_yield:.0f} kg/ha**")

fdi_perc_gdp = st.sidebar.slider(
    "Foreign Direct Investment",
    min_value=-44.2,
    max_value=113.2,
    value=default_fdi,
    step=0.1,
    format="%.1f",
    help="% of GDP"
)
st.sidebar.caption(f"**{fdi_perc_gdp:.1f}%**")

gni_per_cap = st.sidebar.slider(
    "GNI per Capita",
    min_value=210.0,
    max_value=140280.0,
    value=default_gni,
    step=100.0,
    format="%.0f",
    help="Atlas $"
)
st.sidebar.caption(f"**${gni_per_cap:.0f}**")

# GDP normalized to billions
gdp_min_billions = 0.06
gdp_max_billions = 30510.0
gdp_billions = st.sidebar.slider(
    "GDP",
    min_value=gdp_min_billions,
    max_value=gdp_max_billions,
    value=default_gdp_b,
    step=max(0.1, (gdp_max_billions - gdp_min_billions) / 1000),
    format="%.0f",
    help="Billions $ (Range: $60M to $30.5T)"
)
st.sidebar.caption(f"**{gdp_billions:.0f} Billion $**")
gdp = gdp_billions * 1e9  # Convert back to actual value

# Section: Energy & Environment
st.sidebar.markdown("---")
st.sidebar.markdown("###  Energy & Environment")
st.sidebar.caption("Energy consumption and environmental protection")

en_per_cap = st.sidebar.slider(
    "Energy Use per Capita",
    min_value=10.0,
    max_value=16684.0,
    value=default_en,
    step=10.0,
    format="%.0f",
    help="kg oil equivalent"
)
st.sidebar.caption(f"**{en_per_cap:.0f} kg oil eq.**")

prot_area_perc = st.sidebar.slider(
    "Protected Areas",
    min_value=0.0,
    max_value=61.3,
    value=default_prot,
    step=0.1,
    format="%.1f",
    help="% of total land area"
)
st.sidebar.caption(f"**{prot_area_perc:.1f}%**")

# Section: Population Indicators
st.sidebar.markdown("---")
st.sidebar.markdown("###  Population Indicators")
st.sidebar.caption("Demographic and urbanization trends")

pop_urb_aggl_perc = st.sidebar.slider(
    "Urban Agglomerations >1M",
    min_value=0.0,
    max_value=100.0,
    value=default_urb,
    step=0.1,
    format="%.1f",
    help="% of population"
)
st.sidebar.caption(f"**{pop_urb_aggl_perc:.1f}%**")

pop_growth_perc = st.sidebar.slider(
    "Population Growth",
    min_value=-2.67,
    max_value=4.86,
    value=default_pop_g,
    step=0.1,
    format="%.2f",
    help="annual %"
)
st.sidebar.caption(f"**{pop_growth_perc:.2f}%**")

urb_pop_growth_perc = st.sidebar.slider(
    "Urban Population Growth",
    min_value=-1.23,
    max_value=4.04,
    value=default_urb_g,
    step=0.1,
    format="%.2f",
    help="annual %"
)
st.sidebar.caption(f"**{urb_pop_growth_perc:.2f}%**")

# Real-time prediction
prediction = predict_co2_emissions(
    cereal_yield, fdi_perc_gdp, gni_per_cap, en_per_cap,
    pop_urb_aggl_perc, prot_area_perc, gdp,
    pop_growth_perc, urb_pop_growth_perc
)

# Initialize map country selector in session state if not exists
if 'map_country_selector' not in st.session_state:
    st.session_state.map_country_selector = 'All Countries'

# Main content area
col1, col2 = st.columns([2,2.5])

with col1:
    # Section 1: Prediction Results
    st.markdown("####  Prediction Results")
    if prediction is not None:
        st.markdown(f"# **{prediction:.3f} metric tons**")
        st.caption("Predicted CO2 emissions per capita based on your inputs")
    st.markdown("---")
    
    # Section 2: Input Summary
    st.markdown("#### 📝 Your Input Summary")
    st.markdown("**Key Parameters Used:**")
    col_sum1, col_sum2 = st.columns(2)
    with col_sum1:
        st.metric("Energy per Capita", f"{en_per_cap:.0f} kg", help="kg oil equivalent")
        st.metric("Urban Population", f"{pop_urb_aggl_perc:.1f}%", help="% in urban agglomerations >1M")
    with col_sum2:
        st.metric("Population Growth", f"{pop_growth_perc:.2f}%", help="Annual growth rate")
    st.markdown("---")

with col2:
    # Section 4: Interactive Map
    st.markdown("####  Global CO2 Emissions Map")
    st.caption("Explore CO2 emissions worldwide. **Darker red = Higher emissions per capita**")
    
    # Country search/selection for map
    if not co2_data.empty:
        # Get only countries with data for selection
        countries_with_data = co2_data[
            (co2_data['co2_per_cap'].notna()) & 
            (co2_data['co2_per_cap'] > 0)
        ]['country_name'].dropna().unique()
        all_countries = ['All Countries'] + sorted(countries_with_data.tolist())
        
        map_country_selection = st.selectbox(
            "🔍 Search & Select Country:",
            options=all_countries,
            index=0,
            key="map_country_selector",
            help="Select a country to view its CO2 emissions statistics. The map shows CO2 emissions per capita in intensity scale (red = higher emissions)."
        )
        
        # Country Statistics section (placed below selection bar)
        st.markdown("#### 🌍 Country Data")
        if map_country_selection != "All Countries":
            country_stats = co2_data[co2_data['country_name'] == map_country_selection]
            if not country_stats.empty:
                stats = country_stats.iloc[0]
                st.markdown(f"**{map_country_selection}**")
                if stats['co2_per_cap'] > 0:
                    col_stat1, col_stat2 = st.columns(2)
                    with col_stat1:
                        st.metric("CO2 per Capita", f"{stats['co2_per_cap']:.2f} t", help="metric tons per person")
                    with col_stat2:
                        st.metric("Total CO2", f"{stats['co2_ttl']/1000:.1f} Mt", help="Million metric tons")
                else:
                    st.info("No CO2 emissions data available for this country.")
        else:
            st.info("👆 Select a country above to view its statistics")
        st.markdown("---")
        
        # Determine selected country code for highlighting
        selected_country_code = None
        if map_country_selection != "All Countries":
            selected_country_code = get_country_code_from_name(map_country_selection, co2_data)
        
        # Create and display the map
        co2_map = create_co2_map(co2_data, selected_country_code)
        if co2_map:
            st.plotly_chart(co2_map, use_container_width=True)
            st.caption("💡 **Tip:** Hover over countries to see detailed emissions data")
        else:
            st.error("Unable to create map. Please check the data.")
    else:
        st.error("CO2 data not available. Please check the dataset.")
