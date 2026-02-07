import streamlit as st

from config.settings import CHART_COLORS
from models import AVAILABLE_MODELS
from splitters.holdoutpct import HoldoutPctSplitter
from state.app_state import AppState
from utils.model_evaluation import AVAILABLE_METRICS
from utils.plotting import plot_model_comparison


def render_comparison_tab(historical_data):
    trained_models = AppState.get_all_trained_models()
    num_trained = sum(trained_models.values())
    
    if num_trained == 0:
        st.info("ℹ️ Train at least one model to see comparisons.")
    elif num_trained == 1:
        st.info("ℹ️ Train at least 2 models to see meaningful comparisons.")
    else:
        train_pcts = {}
        for model_name in AVAILABLE_MODELS.keys():
            if trained_models[model_name]:
                results = AppState.get_model_results(model_name)
                train_pcts[model_name] = results['train_pct']  # saved value, not UI value
        
        # Check if all have same split
        train_pct_values = list(train_pcts.values())
        all_same_split = all(pct == train_pct_values[0] for pct in train_pct_values)
        
        if not all_same_split:
            st.warning(
                f"⚠️ **Cannot compare models trained on different data splits!**\n\n"
                f"Current train/test splits:\n"
                + "\n".join([f"- **{model}**: {pct}% train / {100-pct}% test" 
                            for model, pct in train_pcts.items()])
                + "\n\nPlease retrain all models with the same train/test split percentage."
            )
            st.stop()

        # All models trained on same split
        train_pct = train_pct_values[0]
        st.success(f"✅ All models trained on {train_pct}% / {100-train_pct}% split")

        # Recalculate split using the verified common train_pct
        splitter = HoldoutPctSplitter(train_pct=train_pct)
        train_data, test_data = splitter.split(historical_data)
        
        # Model selection multiselect
        trained_model_names = [name for name, is_trained in trained_models.items() if is_trained]
        
        selected_models = st.multiselect(
            "Select models to compare:",
            options=trained_model_names,
            default=trained_model_names,  
            help="Choose which models to display in the comparison"
        )
        
        if len(selected_models) == 0:
            st.info("ℹ️ Please select at least one model to display.")
            st.stop()
    
        model_predictions = {}
        model_results = {}
        for model_name in selected_models:
            results = AppState.get_model_results(model_name)
            model_predictions[model_name] = {
                "test": results["test_predictions"],
                "forecast": results["predictions"],
            }
            model_results[model_name] = results["metrics"]

        fig = plot_model_comparison(
            train_data=train_data,
            test_data=test_data,
            model_predictions=model_predictions,
            selected_models=selected_models,
            historical_color=CHART_COLORS['historical'],
            height=400
        )
        st.plotly_chart(fig, width='stretch')
                            
        st.caption("📈 LEADERBOARD")
        for metric_key, metric_config in AVAILABLE_METRICS.items():
            render_metric_leaderboard(metric_key, metric_config, model_results, selected_models)

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