import streamlit as st

from config.settings import RANDOM_SEED, PAGE_CONFIG
from utils.model_evaluation import set_random_seeds
from models import AVAILABLE_MODELS
from state.app_state import AppState
from ui import (
    render_sidebar, 
    render_sidebar_footer, 
    render_historical_tab,
    render_model_tab,
    render_comparison_tab
)

# Page configuration
set_random_seeds(RANDOM_SEED)
st.set_page_config(**PAGE_CONFIG)
AppState.initialize()

# Page start 
st.title("Forecast Tool")

# Sidebar 
render_sidebar()
render_sidebar_footer()

# Main display area 
if not AppState.has_data():
    st.info("👈 Upload a CSV file from the sidebar to get started")
    st.stop()

historical_data = AppState.get_data()

# Create tabs
tab_names = ["Historical Data"] + list(AVAILABLE_MODELS.keys()) + ["Model Comparison"]
tabs = st.tabs(tab_names)

# Historical Data Preview tab
with tabs[0]:
    render_historical_tab(historical_data)

# Model tabs dynamic
for i, (model_name, model_class) in enumerate(AVAILABLE_MODELS.items()):
    with tabs[i + 1]:
        render_model_tab(model_name, model_class, historical_data)
        
# Model Comparison tab   
with tabs[-1]:  
    render_comparison_tab(historical_data)