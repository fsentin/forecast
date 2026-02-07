import streamlit as st
import pandas as pd
from state import AppState
from services import DataService
from utils.input_validation import validate_columns

def render_sidebar():
    """Render the sidebar for data upload and configuration.
    
    Handles the complete workflow for sidebar: file upload, column selection 
    and validation, preprocessing, loading.
    """
    st.sidebar.header("Configure Data")
    st.sidebar.subheader("1. Upload File")
    
    uploaded_file = st.sidebar.file_uploader(
        label="Upload your CSV data file",
        type=['csv'],
        key=f"uploader_{AppState.get_upload_key()}"
    )
    
    if uploaded_file is None:
        st.sidebar.info("ℹ️ Please upload a CSV file to proceed.")
        AppState.clear_preprocessing_state()
        return
    
    # Load data
    data = pd.read_csv(uploaded_file)
    columns = data.columns.tolist()
    
    if len(columns) < 2:
        st.sidebar.error("⚠️ The dataset must have at least two columns.")
        return
    
    # Column selection
    st.sidebar.subheader("2. Select Columns")
    
    date_column = st.sidebar.selectbox("Date Column", options=columns)
    value_column = st.sidebar.selectbox(
        "Value Column", 
        options=columns, 
        index=1 if len(data.columns) > 1 else 0
    )
    
    # Validate column content
    is_valid, errors = validate_columns(data, date_column, value_column)
    
    if not is_valid:
        for error in errors:
            st.sidebar.error(f"⚠️ {error}")
        return  # Don't clear data, just show errors and let user fix column selection 
    
    preprocessing_pending, pending_df, pending_missing_count = AppState.get_preprocessing_state()
    
    if preprocessing_pending:
        _render_preprocessing_options(pending_df, pending_missing_count)
        return
    
    # Outlier removal option
    st.sidebar.subheader("3. Preprocessing Options")
    remove_outliers = st.sidebar.checkbox(
        "Remove Outliers",
        value=False,
        help="Detect and remove outliers using IQR method (Q1-1.5×IQR, Q3+1.5×IQR). "
             "Outliers are replaced with interpolated values. Useful for cleaning noisy data."
    )
    
    # Show load button
    if st.sidebar.button("Load Data", type="primary", key="load_data_btn"):
        _handle_data_load(data, date_column, value_column, remove_outliers)

def _render_preprocessing_options(df: pd.DataFrame, missing_count: int) -> None:
    """Render preprocessing options for handling missing dates.
    
    Presents user with preprocessing choices and applies the selected
    method using DataService.
    
    Args:
        df: DataFrame with missing dates detected.
        missing_count: Number of missing dates found.
    """
    st.sidebar.warning(f"⚠️ Found {missing_count} gaps in data.")
    st.sidebar.subheader("Handle Time Series Gaps")
    
    fill_method = st.sidebar.radio(
        "Fill gaps with:",
        options=["Linear interpolation", "Zeros"],
        key="fill_method"
    )
    
    if st.sidebar.button("Apply and Load", key="apply_fill", type="primary"):
        # Apply preprocessing using service
        method = "interpolate" if fill_method == "Interpolate (linear)" else "zero"
        processed_df = DataService.apply_preprocessing(df, method)
        
        # Load processed data
        AppState.set_data(processed_df)
        AppState.clear_preprocessing_state()
        
        st.sidebar.success(f"✅ Data loaded with {fill_method.split()[0].lower()}")
        st.rerun()


def _handle_data_load(data: pd.DataFrame, date_column: str, value_column: str, remove_outliers: bool = False) -> None:
    """Process and load the selected data into application state.
    
    Delegates data preparation and validation to DataService,
    handling the preprocessing workflow when missing dates are detected.
    Optionally removes outliers before loading.
    
    Args:
        data: Raw uploaded dataframe.
        date_column: Name of the date column.
        value_column: Name of the value column.
        remove_outliers: Whether to detect and remove outliers using IQR method.
    """
    try:
        # Prepare dataframe using service
        df = DataService.prepare_dataframe(data, date_column, value_column)
        
        # Optionally remove outliers
        if remove_outliers:
            df, outlier_count = DataService.detect_and_remove_outliers(df)
            if outlier_count > 0:
                st.sidebar.info(f"ℹ️ Removed {outlier_count} outlier(s) and replaced with interpolated values")
        
        # Check for missing dates using service
        has_missing, missing_count = DataService.check_gaps(df)
        
        if has_missing:
            # Defer to preprocessing workflow
            AppState.set_preprocessing_state(True, df, missing_count)
            st.rerun()
        else:
            # Load directly
            AppState.set_data(df)
            st.sidebar.success("✅ Data loaded")
            st.rerun()
        
    except Exception as e:
        st.sidebar.error(f"❌ Error loading data: {e}")
        # Don't clear data - let user try different columns

def render_sidebar_footer():
    """Render the sidebar footer with reset button."""
    st.sidebar.divider()
    if st.sidebar.button("🔄 Reset All", help="Clear all data and trained models"):
        AppState.reset_all()
        st.rerun()
