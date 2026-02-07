import streamlit as st
from utils.plotting import plot_forecast
from utils.timeseries import format_duration
from config.settings import CHART_COLORS

def render_historical_tab(historical_data):
    hist_data_preview, hist_data_visual = st.columns([2, 4],  gap='medium')

    with hist_data_preview:
        st.caption("PREVIEW")
        st.dataframe(historical_data.head(8), width='stretch')
        
    with hist_data_visual:
        st.caption("VISUALIZATION")
        with st.spinner("Creating visualization...", show_time=True):
            fig = plot_forecast(
                historical_data=historical_data,
                historical_color=CHART_COLORS['historical'],
                height=400
            )
            st.plotly_chart(fig, width='stretch')

    st.caption("TIME PERIOD")
    with st.container(border=True):
        info = st.columns(4, gap="small")
        info[0].metric("Total Date Points", f"{len(historical_data):,}")
        info[1].metric("↔ Duration", f"{format_duration(historical_data.index.min(), historical_data.index.max())}")
        info[2].metric("▶️ Start Date", f"{historical_data.index.min().date()}")
        info[3].metric("⏹️ End Date", f"{historical_data.index.max().date()}")

    st.caption("VALUE STATS") 
    with st.container(border=True):
        basicstats = st.columns(4, gap="small")
        basicstats[0].metric("μ Mean", f"{historical_data['value'].mean():,.2f}")
        basicstats[1].metric("σ Std Dev", f"{historical_data['value'].std():,.2f}")
        basicstats[2].metric("🔻Min Value", f"{historical_data['value'].min():,.2f}")
        basicstats[3].metric("🔺Max Value", f"{historical_data['value'].max():,.2f}")