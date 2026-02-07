import streamlit as st
from splitters.holdoutpct import HoldoutPctSplitter
from state.app_state import AppState
from services import ModelService
from utils.plotting import (
                            plot_forecast, 
                            plot_train_test_forecast
                        )
from utils.model_evaluation import AVAILABLE_METRICS
from config.settings import (
    CHART_COLORS,
    TRAIN_PCT_DEFAULT,
    TRAIN_PCT_MIN,
    TRAIN_PCT_MAX,
    HORIZON_DEFAULT,
    HORIZON_MIN,
    HORIZON_MAX
)

def render_model_tab(
    model_name,
    model_class,
    historical_data,
):
    
        # Two-column layout: config on left, visualization on right
        config_col, vis_col = st.columns([2, 4], gap="medium")
        
        with config_col:
            st.write("#### Configure Model")
            with st.container(border=True):
                st.caption("MODEL HYPERPARAMETERS")
                # Check if model supports recommendations
                if model_class.has_recommendations():
                    if st.button("Get Recommendations", key=f"{model_name.lower()}_recommend"):
                        with st.spinner("Analyzing data..."):
                            recommendations = model_class.get_recommendations(historical_data)
                        
                        # Apply recommendations to session state
                        for param_name, (value, explanation) in recommendations.items():
                            AppState.set_hyperparameter(model_name, param_name, value)
                        
                        # Show summary
                        rec_text = ", ".join([f"{p}={v[0]}" for p, v in recommendations.items()])
                        st.success(f"Recommended: {rec_text}")
                        st.caption("These are suggestions based on the qualities of time-series data. Experiment with different hyperparameters for best results.")
                        
                        with st.expander("📊 See explanations"):
                            for param_name, (value, explanation) in recommendations.items():
                                st.write(f"**{param_name}:** {explanation}")
                
                hyperparams = render_hyperparameters(model_name, model_class, layout='vertical')
            
            with st.container(border=True):
                st.caption("FORECAST & EVALUATION")
                # Forecast settings
                horizon = st.number_input(
                    "Forecast Horizon (steps)",
                    min_value=HORIZON_MIN,
                    max_value=HORIZON_MAX,
                    value=HORIZON_DEFAULT,
                    key=f"{model_name.lower()}_horizon",
                    help="Number of **time steps to predict** into the future"
                )
                train_pct = st.slider(
                    "Evaluation Training Split (%)",
                    min_value=TRAIN_PCT_MIN,
                    max_value=TRAIN_PCT_MAX,
                    value=TRAIN_PCT_DEFAULT,
                    key=f"{model_name.lower()}_train_pct",
                    help="Percentage of **historical data used for training** during evaluation, remaining data tests model performance. " \
                    "Final forecast uses all historic data."
                )

                # Calculate split points based on current slider value
                splitter = HoldoutPctSplitter(train_pct=train_pct)
                train_data, test_data = splitter.split(historical_data)
                st.caption(f"Training: {len(train_data)} points | Testing: {len(test_data)} points")

            # Train button
            if st.button(f"Train & Forecast", type="primary", key=f"train_{model_name.lower()}"):
                try:
                    with st.spinner(f"Training {model_name}...", show_time=True):
                        metrics, predictions, test_preds = ModelService.train_and_evaluate(
                            model_class=model_class,
                            data=historical_data,
                            train_pct=train_pct,
                            horizon=horizon,
                            upload_key=AppState.get_upload_key(),
                            **hyperparams
                        )
                        AppState.set_model_results(model_name, metrics, predictions, test_preds, train_pct, horizon)
                        st.success(f"✅ {model_name} training complete!")
                        st.rerun()
                        
                except Exception as e:
                    st.error(f"❌ Training failed: {str(e)}")
                    st.caption("Try adjusting hyperparameters or using 'Get Recommendations'")
                        
        with vis_col:
            results = AppState.get_model_results(model_name)

            if not results['trained']:
                st.empty()
            else:
                st.write("#### Forecast Results")
                with st.container(border=True):
                    forecast = results['predictions']

                    fig = plot_forecast(
                        historical_data=historical_data,
                        forecast_data=forecast,
                        height=500
                    )
                    st.plotly_chart(fig, width='stretch')

                    # Show metrics
                    metrics = results['metrics']
                    metric_cols = st.columns(len(metrics))
                    for col, (metric_key, value) in zip(metric_cols, metrics.items()):
                        if value is not None:
                            metric_config = AVAILABLE_METRICS.get(metric_key, {})
                            display_name = metric_config.get('name', metric_key.upper())
                            suffix = metric_config.get('suffix', '')
                            col.metric(display_name, f"{value:,.2f}{suffix}", help=metric_config.get('description', ''))

                    with st.expander("📊 View Evaluation Details"):
                        # Get stored test predictions
                        test_predictions = results['test_predictions']
                        test_size = len(test_predictions)
                        train_data = historical_data.iloc[:-test_size]
                        test_data = historical_data.iloc[-test_size:]
                        
                        fig_detailed = plot_train_test_forecast(
                            train_data=train_data,
                            test_data=test_data,
                            test_predictions=test_predictions,
                            forecast_data=forecast,
                            train_color=CHART_COLORS['historical'],
                            test_color="#2ca02c",
                            prediction_color="#ff7f0e",
                            forecast_color=CHART_COLORS['forecast'],
                            height=400
                        )
                        st.plotly_chart(fig_detailed, width='stretch')

                    st.download_button(
                        label="📥 Export Forecast",
                        data=forecast.to_csv(index=True),
                        file_name=f"{model_name.lower()}_forecast.csv",
                        mime="text/csv"
                    )

def render_hyperparameters(model_name, model_class, layout='horizontal') -> dict:
    """Dynamically render hyperparameter input widgets for a model.
    
    Creates number inputs, sliders, or select boxes based on the model's
    hyperparameter configuration. Supports both horizontal (multi-column)
    and vertical (single-column) layouts.
    
    Args:
        model_name: Name of the model (e.g., 'ARIMA', 'Prophet').
        model_class: Model class with get_hyperparameters() static method.
        layout: Layout style - 'horizontal' (default) or 'vertical'.
    
    Returns:
        Dictionary mapping hyperparameter names to their current values.
    """
    hyperparam_config = model_class.get_hyperparameters()
    hyperparams = {}
    
    if layout == 'horizontal':
        cols = st.columns(min(len(hyperparam_config), 5))
        for i, (param_name, config) in enumerate(hyperparam_config.items()):
            with cols[i % 5]:
                hyperparams[param_name] = render_single_param(model_name, param_name, config)
    
    else:  # vertical
        for param_name, config in hyperparam_config.items():
            hyperparams[param_name] = render_single_param(model_name, param_name, config)
    
    return hyperparams

def render_single_param(model_name: str, param_name: str, config: dict) -> int | float | str | bool:
    """Render a single hyperparameter input widget.
    
    Creates the appropriate Streamlit widget (number_input, selectbox, or checkbox)
    based on the parameter type specified in the config.
    
    Args:
        model_name: Name of the model (used for unique widget keys).
        param_name: Name of the parameter (e.g., 'p', 'd', 'q', 'normalize').
        config: Parameter configuration dictionary with keys:
            - type: 'int', 'float', 'select', or 'checkbox'
            - label: Display label for the widget
            - min/max: Range for numeric inputs
            - default: Default value
            - help: Tooltip text
            - options: List of choices (for 'select' type only)
    
    Returns:
        The current value from the widget (int, float, str, or bool).
    """
    if config['type'] == 'int':
        return st.number_input(
            f"**{config['label']}**",
            min_value=config['min'],
            max_value=config['max'],
            value=config['default'],
            key=f"{model_name.lower()}_{param_name}",
            help=config['help']
        )
    elif config['type'] == 'float':
        return st.number_input(
            f"**{config['label']}**",
            min_value=config['min'],
            max_value=config['max'],
            value=config['default'],
            step=0.001,
            format="%.3f",
            key=f"{model_name.lower()}_{param_name}",
            help=config['help']
        )
    elif config['type'] == 'select':
        return st.selectbox(
            f"**{config['label']}**",
            options=config['options'],
            index=config['options'].index(config['default']),
            key=f"{model_name.lower()}_{param_name}",
            help=config['help']
        )
    elif config['type'] == 'checkbox':  
        return st.checkbox(
            f"{config['label']}",
            value=config['default'],
            key=f"{model_name.lower()}_{param_name}",
            help=config['help']
        )
    

def render_metric_leaderboard(metric_key: str, metric_config: dict, model_results: dict, selected_models: list) -> None:
    """Renders a metric comparison leaderboard as a row of cards.
    
    Displays model performance for a single metric, highlighting the best-
    performing model with a trophy emoji and showing deltas for others.
    
    Args:
        metric_key: Metric identifier (e.g., 'mae', 'rmse').
        metric_config: Configuration dictionary with keys:
            - name: Display name for the metric
            - description: Explanation text
            - lower_is_better: True if lower values are better
        model_results: Dictionary mapping model names to their result dicts.
        selected_models: List of model names to display in the leaderboard.
        
    Note:
        Silently returns if the metric doesn't exist in the results.
    """
    metric_name = metric_config['name']
    
    # Check if this metric exists
    if metric_key not in list(model_results.values())[0]:
        return
    
    st.write(f"#### {metric_name}")
    st.caption(f"{metric_config['description']}")
    
    # Get values and find best
    metric_values = {model: results[metric_key] for model, results in model_results.items()}
    
    if metric_config['lower_is_better']:
        best_value = min(metric_values.values())
        best_model = min(metric_values, key=metric_values.get)
    else:
        best_value = max(metric_values.values())
        best_model = max(metric_values, key=metric_values.get)
    
    # Display as cards
    cols = st.columns(len(selected_models))
    for idx, model_name in enumerate(selected_models):
        with cols[idx]:
            value = metric_values[model_name]
            is_best = model_name == best_model
            
            # Calculate delta
            delta = "🏆" if is_best else f"{value - best_value:+,.2f}"
            
            # Render card
            if is_best:
                with st.container(border=True):
                    st.metric(model_name, f"{value:,.2f}", delta=delta, delta_color="off")
            else:
                delta_color = "inverse" if metric_config['lower_is_better'] else "normal"
                st.metric(model_name, f"{value:,.2f}", delta=delta, delta_color=delta_color)