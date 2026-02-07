import pandas as pd
import plotly.graph_objects as go
import plotly.colors
from plotly.subplots import make_subplots
from scipy import stats

def _add_traces(fig: go.Figure, 
                series_dict: dict, 
                color_dict: dict, 
                width_dict: dict | None = None) -> go.Figure:
    """Helper to add multiple traces to a figure"""
    for name, series in series_dict.items():
        if series is None:
            continue
        color = color_dict.get(name, "#000000")
        width = width_dict.get(name, 2) if width_dict else 2
        fig.add_trace(go.Scatter(
            x=series.index,
            y=series['value'].values,
            mode="lines",
            name=name,
            line=dict(color=color, width=width)
        ))
    return fig


def plot_forecast(
    historical_data: pd.DataFrame,
    forecast_data: pd.DataFrame | None = None,
    historical_color: str = "#1f77b4",
    forecast_color: str = "#FF6B6B",
    show_split_line: bool = True,
    split_line_color: str = "gray",
    height: int = 500,
    xaxis_title: str = "Date",
    yaxis_title: str = "Value"
) -> go.Figure:
    fig = go.Figure()

    # Add historical and forecast traces
    series_dict = {"Historical": historical_data, "Forecast": forecast_data}
    color_dict = {"Historical": historical_color, "Forecast": forecast_color}
    width_dict = {"Historical": 2, "Forecast": 2.5}
    _add_traces(fig, series_dict, color_dict, width_dict)

    if forecast_data is not None and show_split_line:
        fig.add_vline(
            x=historical_data.index[-1],
            line_dash="dot",
            line_color=split_line_color,
            line_width=1.5
        )

    fig.update_layout(
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        hovermode="x unified",
        showlegend=False,
        height=height
    )
    return fig


def plot_train_test_forecast(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    test_predictions: pd.DataFrame,
    forecast_data: pd.DataFrame | None = None,
    train_color: str = "#1f77b4",
    test_color: str = "#2ca02c",
    prediction_color: str = "#ff7f0e",
    forecast_color: str = "#FF6B6B",
    height: int = 500
) -> go.Figure:
    fig = go.Figure()
    series_dict = {
        "Training": train_data,
        "Actual (Test)": test_data,
        "Predicted (Test)": test_predictions,
        "Future Forecast": forecast_data
    }
    color_dict = {
        "Training": train_color,
        "Actual (Test)": test_color,
        "Predicted (Test)": prediction_color,
        "Future Forecast": forecast_color
    }
    width_dict = {"Future Forecast": 2.5}
    _add_traces(fig, series_dict, color_dict, width_dict)

    # Add vertical line at end of test
    fig.add_vline(x=test_data.index[-1], line_dash="dot", line_color="gray")

    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Value",
        hovermode="x unified",
        height=height
    )
    return fig


def plot_model_comparison(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    model_predictions: dict,  # {model_name: {'test': df, 'forecast': df}}
    selected_models: list,
    historical_color: str = "#1f77b4",
    height: int = 400
) -> go.Figure:
    fig = go.Figure()

    # Colors for models
    palette = plotly.colors.qualitative.Plotly[1:]
    model_colors = {model: palette[i % len(palette)] for i, model in enumerate(model_predictions.keys())}

    # Historical data
    series_dict = {
        "Training Data": train_data,
        "Test Data Actual": test_data
    }
    color_dict = {"Training Data": historical_color, "Test Data Actual": historical_color}
    width_dict = {"Training Data": 2, "Test Data Actual": 2}
    _add_traces(fig, series_dict, color_dict, width_dict)

    # Vertical lines at train/test boundaries
    fig.add_vline(x=train_data.index[-1], line_dash="dot", line_color="gray")
    fig.add_vline(x=test_data.index[-1], line_dash="dot", line_color="gray")

    # Add predictions and forecasts for selected models
    for model_name in selected_models:
        if model_name not in model_predictions:
            continue
        preds = model_predictions[model_name]
        color = model_colors[model_name]
        _add_traces(fig,
                    {"{0} Test".format(model_name): preds.get('test'),
                     "{0} Forecast".format(model_name): preds.get('forecast')},
                    {f"{model_name} Test": color, f"{model_name} Forecast": color})

    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Value",
        hovermode='x unified',
        height=height,
    )
    return fig